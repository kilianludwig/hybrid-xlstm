# Copyright 2026 Kilian Ludwig
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from cli.utils import add_group_arg

DEFAULT_MODELS = ["ylstm_hb_c0", "ylstm_hb_c1", "ylstm_hb_c2", "ylstm_hb_c4"]
DEFAULT_SEQ_LENS = [16, 64, 256, 1024, 2048, 4096]
DEFAULT_BATCH_SIZE = 128
DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "benchmark",
        help="Time forward-only inference and full train steps per architecture",
    )
    add_group_arg(p)
    p.add_argument(
        "--models", "-m", nargs="+", default=None, metavar="NAME",
        help=f"Model names to benchmark. Default: {' '.join(DEFAULT_MODELS)}",
    )
    p.add_argument(
        "--seq-lens", nargs="+", type=int, default=None, metavar="T",
        help=f"Sequence lengths to scan. Default: {DEFAULT_SEQ_LENS}",
    )
    p.add_argument(
        "--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--hidden-dim", type=int, default=None,
        help="Backbone width. Default: the LSTM-matched Chomsky width per model.",
    )
    p.add_argument(
        "--num-layers", type=int, default=2,
        help="Backbone depth (default: 2, the matched-protocol depth)",
    )
    p.add_argument(
        "--input-dim", type=int, default=None,
        help="Synthetic input width. Default: the param-matching reference dim.",
    )
    p.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Untimed warmup iterations (default: {DEFAULT_WARMUP})",
    )
    p.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help=f"Timed iterations per cell (default: {DEFAULT_REPEATS})",
    )
    p.add_argument(
        "--device", "-d", default="auto", choices=["cpu", "cuda", "mps", "auto"],
        help="Device to benchmark on (default: auto)",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path. A sibling .json is written alongside. "
             "Default: experiments/<group>/benchmarks/<timestamp>/benchmark.csv",
    )
    p.set_defaults(func=benchmark)


def _resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _matched_widths() -> Dict[str, int]:
    from configs.param_matching import DEFAULT_TOLERANCE, matched_model_overrides

    overrides = matched_model_overrides("chomsky", DEFAULT_TOLERANCE)
    widths: Dict[str, int] = {}
    for name, ov in overrides.items():
        width = ov.get("hidden_dim") or ov.get("d_model") or ov.get("embedding_dim")
        if width:
            widths[name] = int(width)
    return widths


def _base_model_name(name: str) -> str:
    head, sep, tail = name.rpartition("_c")
    if sep and tail.isdigit():
        return head
    return name


def _resolve_hidden_dim(model_name: str, widths: Dict[str, int]) -> int:
    from configs.defaults import MODEL_DEFAULTS

    for key in (model_name, _base_model_name(model_name)):
        if key in widths:
            return widths[key]
    return int(MODEL_DEFAULTS.get(model_name, {}).get("hidden_dim", 64))


def _build(model_name: str, input_dim: int, hidden_dim: int, num_layers: int,
           device: torch.device) -> torch.nn.Module:
    from ml.training.pipeline import _instantiate_model

    model = _instantiate_model(
        model_name, input_dim, hidden_dim, {"num_layers": num_layers},
    )
    return model.to(device)


def _forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _time_inference(model, x, device, warmup: int, repeats: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _forward(model, x)
        _sync(device)
        start = time.perf_counter()
        for _ in range(repeats):
            _forward(model, x)
        _sync(device)
    return (time.perf_counter() - start) / repeats


def _time_train_step(model, x, device, warmup: int, repeats: int) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        _forward(model, x).square().mean().backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        step()
    _sync(device)
    return (time.perf_counter() - start) / repeats


def _measure(model_name: str, seq_len: int, args, device, widths) -> Dict[str, Any]:
    hidden_dim = args.hidden_dim or _resolve_hidden_dim(model_name, widths)
    row: Dict[str, Any] = {
        "model": model_name,
        "chunk_size": None,
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "hidden_dim": hidden_dim,
        "num_layers": args.num_layers,
        "input_dim": args.input_dim,
        "parameters": None,
        "inference_ms_per_batch": None,
        "inference_tokens_per_second": None,
        "train_step_ms": None,
        "train_tokens_per_second": None,
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "status": "ok",
    }

    tokens = args.batch_size * seq_len
    try:
        model = _build(
            model_name, args.input_dim, hidden_dim, args.num_layers, device,
        )
        row["chunk_size"] = getattr(model, "chunk_size", None)
        row["parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        x = torch.randn(args.batch_size, seq_len, args.input_dim, device=device)

        infer_s = _time_inference(model, x, device, args.warmup, args.repeats)
        row["inference_ms_per_batch"] = round(infer_s * 1e3, 4)
        row["inference_tokens_per_second"] = round(tokens / infer_s, 1)

        train_s = _time_train_step(model, x, device, args.warmup, args.repeats)
        row["train_step_ms"] = round(train_s * 1e3, 4)
        row["train_tokens_per_second"] = round(tokens / train_s, 1)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        row["status"] = "oom"
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return row


def _default_output(group: Optional[str]) -> Path:
    from configs.paths import group_dir

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return group_dir(group) / "benchmarks" / stamp / "benchmark.csv"


def benchmark(args: argparse.Namespace) -> Optional[Path]:
    from configs.param_matching import REFERENCE_INPUT_DIM
    from ml.models import MODEL_REGISTRY

    models = args.models or list(DEFAULT_MODELS)
    seq_lens = args.seq_lens or list(DEFAULT_SEQ_LENS)
    if args.input_dim is None:
        args.input_dim = REFERENCE_INPUT_DIM

    missing = [m for m in models if m not in MODEL_REGISTRY]
    if missing:
        print(
            f"Error: not in the model registry: {', '.join(missing)}.\n"
            "Hint: pass --group (or set HYBRID_GROUP) so the group's model "
            "adapters are loaded.",
            file=sys.stderr,
        )
        return None

    device = _resolve_device(args.device)
    widths = _matched_widths()

    output_path = Path(args.output) if args.output else _default_output(
        getattr(args, "group", None)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Benchmarking {len(models)} model(s) x {len(seq_lens)} sequence "
        f"length(s) on {device}  "
        f"[batch {args.batch_size}, warmup {args.warmup}, repeats {args.repeats}]"
    )

    rows: List[Dict[str, Any]] = []
    for model_name in models:
        for seq_len in seq_lens:
            row = _measure(model_name, seq_len, args, device, widths)
            rows.append(row)
            if row["status"] == "ok":
                print(
                    f"  {model_name:14s} T={seq_len:>5}  "
                    f"infer {row['inference_tokens_per_second']:>12,.0f} tok/s  "
                    f"({row['inference_ms_per_batch']:>8.3f} ms)  "
                    f"train {row['train_tokens_per_second']:>12,.0f} tok/s  "
                    f"({row['train_step_ms']:>8.3f} ms)"
                )
            else:
                print(f"  {model_name:14s} T={seq_len:>5}  {row['status']}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "device": str(device),
                "torch_version": torch.__version__,
                "platform": platform.platform(),
                "basis": "bare backbone on synthetic input",
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "rows": rows,
            },
            f,
            indent=2,
        )

    print(f"\nbenchmark.csv written to {output_path}")
    return output_path
