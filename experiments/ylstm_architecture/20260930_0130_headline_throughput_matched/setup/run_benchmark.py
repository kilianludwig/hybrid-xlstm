#!/usr/bin/env python3
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
import os
import platform
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HYBRID_GROUP", "ylstm_architecture")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_BLOCK_YLSTM = {
    "hidden_dim": 44,
    "num_layers": 2,
    "num_heads": 1,
    "ffn": "swiglu",
    "fgate_bias_init": 3.0,
    "norm": "feedback",
    "feedback_init_std": 0.0,
}

CONFIGS: Dict[str, Dict[str, Any]] = {
    "transformer": {
        "hidden_dim": 48,
        "d_model": 48,
        "num_layers": 2,
        "nhead": 4,
        "dropout": 0.0,
        "dim_feedforward": 168,
    },
    "mamba": {
        "hidden_dim": 56,
        "d_model": 56,
        "num_layers": 2,
        "dropout": 0.0,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    },
    "slstm": {
        "hidden_dim": 52,
        "embedding_dim": 52,
        "num_blocks": 2,
        "dropout": 0.0,
        "num_heads": 4,
        "add_post_blocks_norm": True,
        "bias_init": "powerlaw_blockdependent",
    },
    "mlstm": {
        "hidden_dim": 52,
        "embedding_dim": 52,
        "num_blocks": 2,
        "dropout": 0.0,
        "num_heads": 4,
        "add_post_blocks_norm": True,
        "fgate_bias_range": None,
    },
    "xlstm": {
        "hidden_dim": 52,
        "embedding_dim": 52,
        "num_blocks": 2,
        "dropout": 0.0,
        "num_heads": 4,
        "add_post_blocks_norm": True,
        "bias_init": "powerlaw_blockdependent",
        "fgate_bias_range": None,
    },
    "hlstm": {
        "hidden_dim": 40,
        "num_layers": 2,
        "split_ratio": 0.5,
        "num_heads": 1,
        "fusion": "gate",
        "ffn": "swiglu",
        "conv_kernel": 4,
        "fgate_bias_init": 3.0,
    },
    "xlstm_hb": {
        "hidden_dim": 46,
        "num_layers": 2,
        "num_heads": 1,
        "ffn": "swiglu",
        "conv_kernel": 4,
        "fgate_bias_init": 3.0,
    },
    "yhlstm": {
        "hidden_dim": 38,
        "num_layers": 2,
        "split_ratio": 0.5,
        "num_heads": 1,
        "fusion": "gate",
        "ffn": "swiglu",
        "conv_kernel": 4,
        "chunk_size": 1,
        "fgate_bias_init": 3.0,
        "norm": "feedback",
        "feedback_init_std": 0.0,
    },
}
for _name in ("ylstm_hb_c0", "ylstm_hb_c1", "ylstm_hb_c2", "ylstm_hb_c4", "ylstm_hb_c8"):
    CONFIGS[_name] = dict(_BLOCK_YLSTM)

SEQ_LENS = [16, 64, 256, 1024, 2048, 4096]
BATCH_SIZE = 128
WARMUP = 3
REPEATS = 10
INPUT_DIM = 8

_CONTEXT_MODELS = {"slstm", "mlstm", "xlstm"}


def _build(model_name: str, seq_len: int, device):
    import inspect

    import torch

    from ml.models import MODEL_REGISTRY
    from ml.training.pipeline import _instantiate_model

    cfg = dict(CONFIGS[model_name])
    if model_name in _CONTEXT_MODELS:
        cfg["context_length"] = seq_len
    hidden = int(cfg.get("hidden_dim") or cfg.get("d_model") or cfg.get("embedding_dim"))
    params = inspect.signature(MODEL_REGISTRY[model_name].__init__).parameters
    accepted = {
        key: value for key, value in cfg.items() if key in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    }
    model = _instantiate_model(model_name, INPUT_DIM, hidden, accepted)
    return model.to(device)


_SIZE_UNIT = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kib": 1024,
    "kb": 1024,
    "mib": 1024 ** 2,
    "mb": 1024 ** 2,
    "gib": 1024 ** 3,
    "gb": 1024 ** 3,
    "tib": 1024 ** 4,
    "tb": 1024 ** 4,
}


def _parse_size(text: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, text, re.IGNORECASE)
    if match is None:
        return None
    scale = _SIZE_UNIT.get(match.group(2).lower())
    if scale is None:
        return None
    return int(float(match.group(1)) * scale)


def _oom_requirement_bytes(device, exc: BaseException) -> Optional[int]:
    import torch

    text = str(exc)
    tried = _parse_size(text, r"Tried to allocate ([0-9.]+)\s*([A-Za-z]+)")
    allocated = _parse_size(
        text, r"([0-9.]+)\s*([A-Za-z]+) is allocated by PyTorch"
    )
    if tried is not None and allocated is not None:
        return allocated + tried
    if device.type != "cuda":
        return tried
    peak = int(torch.cuda.max_memory_allocated(device))
    current = int(torch.cuda.memory_allocated(device))
    if tried is not None:
        return max(peak, current + tried)
    return peak or None


def _mib(nbytes: Optional[int]) -> Optional[float]:
    if nbytes is None:
        return None
    return round(nbytes / (1024 ** 2), 2)


def _reset_peak(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_bytes(device) -> Optional[int]:
    import torch

    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _measure_memory(model_name: str, seq_len: int, device) -> Dict[str, Any]:
    import torch

    from cli.commands.benchmark import _forward, _is_oom, _sync

    cfg = CONFIGS[model_name]
    hidden = int(cfg.get("hidden_dim") or cfg.get("d_model") or cfg.get("embedding_dim"))
    total = None
    if device.type == "cuda":
        total = _mib(torch.cuda.get_device_properties(device).total_memory)
    row: Dict[str, Any] = {
        "model": model_name,
        "chunk_size": None,
        "seq_len": seq_len,
        "batch_size": BATCH_SIZE,
        "hidden_dim": hidden,
        "num_layers": cfg.get("num_layers", cfg.get("num_blocks")),
        "input_dim": INPUT_DIM,
        "parameters": None,
        "inference_peak_mib": None,
        "inference_status": None,
        "train_peak_mib": None,
        "train_status": None,
        "device_total_mib": total,
        "device": str(device),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "status": "ok",
        "backend": None,
        "error": None,
    }

    model = None
    try:
        model = _build(model_name, seq_len, device)
        row["chunk_size"] = getattr(model, "chunk_size", None)
        row["backend"] = getattr(model, "backend", None)
        row["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x = torch.randn(BATCH_SIZE, seq_len, INPUT_DIM, device=device)

        try:
            model.eval()
            _reset_peak(device)
            with torch.no_grad():
                _forward(model, x)
            _sync(device)
            row["inference_peak_mib"] = _mib(_peak_bytes(device))
            row["inference_status"] = "ok"
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            row["inference_peak_mib"] = _mib(_oom_requirement_bytes(device, exc))
            row["inference_status"] = "oom"
            row["status"] = "oom"
            row["error"] = str(exc).splitlines()[0][:300]
            return row

        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            _reset_peak(device)
            optimizer.zero_grad(set_to_none=True)
            _forward(model, x).square().mean().backward()
            optimizer.step()
            _sync(device)
            row["train_peak_mib"] = _mib(_peak_bytes(device))
            row["train_status"] = "ok"
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            row["train_peak_mib"] = _mib(_oom_requirement_bytes(device, exc))
            row["train_status"] = "oom"
            row["status"] = "oom"
            row["error"] = str(exc).splitlines()[0][:300]
    except RuntimeError as exc:
        if not _is_oom(exc):
            row["status"] = "error"
            row["error"] = str(exc).splitlines()[-1][:300]
            return row
        requirement = _mib(_oom_requirement_bytes(device, exc))
        row["inference_peak_mib"] = requirement
        row["inference_status"] = "oom"
        row["status"] = "oom"
        row["error"] = str(exc).splitlines()[0][:300]
    except Exception as exc:  # noqa: BLE001 — record and continue the scan
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}".splitlines()[-1][:300]
        traceback.print_exc()
    finally:
        del model
        if device.type == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    return row


def _measure_cell(model_name: str, seq_len: int, device) -> Dict[str, Any]:
    import torch

    from cli.commands.benchmark import (
        _forward,
        _is_oom,
        _sync,
        _time_inference,
        _time_train_step,
    )

    cfg = CONFIGS[model_name]
    hidden = int(cfg.get("hidden_dim") or cfg.get("d_model") or cfg.get("embedding_dim"))
    row: Dict[str, Any] = {
        "model": model_name,
        "chunk_size": None,
        "seq_len": seq_len,
        "batch_size": BATCH_SIZE,
        "hidden_dim": hidden,
        "num_layers": cfg.get("num_layers", cfg.get("num_blocks")),
        "input_dim": INPUT_DIM,
        "parameters": None,
        "inference_ms_per_batch": None,
        "inference_tokens_per_second": None,
        "train_step_ms": None,
        "train_tokens_per_second": None,
        "device": str(device),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "warmup": WARMUP,
        "repeats": REPEATS,
        "status": "ok",
        "backend": None,
        "error": None,
    }
    tokens = BATCH_SIZE * seq_len
    try:
        model = _build(model_name, seq_len, device)
        row["chunk_size"] = getattr(model, "chunk_size", None)
        row["backend"] = getattr(model, "backend", None)
        row["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x = torch.randn(BATCH_SIZE, seq_len, INPUT_DIM, device=device)
        infer_s = _time_inference(model, x, device, WARMUP, REPEATS)
        row["inference_ms_per_batch"] = round(infer_s * 1e3, 4)
        row["inference_tokens_per_second"] = round(tokens / infer_s, 1)
        train_s = _time_train_step(model, x, device, WARMUP, REPEATS)
        row["train_step_ms"] = round(train_s * 1e3, 4)
        row["train_tokens_per_second"] = round(tokens / train_s, 1)
    except RuntimeError as exc:
        if not _is_oom(exc):
            row["status"] = "error"
            row["error"] = str(exc).splitlines()[-1][:300]
        else:
            row["status"] = "oom"
    except Exception as exc:  # noqa: BLE001 — record and continue the scan
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}".splitlines()[-1][:300]
        traceback.print_exc()
    finally:
        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
    return row


def _write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Throughput of the headline architectures at matched widths.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Instantiate on CPU at sequence length 16 and print parameter counts.",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Record one-iteration peak GPU memory instead of the ten-iteration timing.",
    )
    args = parser.parse_args()

    unknown = [name for name in args.models if name not in CONFIGS]
    if unknown:
        raise SystemExit(f"No matched config for: {', '.join(unknown)}")
    if not args.count_only and not args.output:
        raise SystemExit("--output is required unless --count-only is set")

    import torch

    if args.count_only:
        for name in args.models:
            model = _build(name, 16, torch.device("cpu"))
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name:16s} {params:8d}")
            del model
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available; timings will be on CPU.", file=sys.stderr)

    output = Path(args.output)
    rows: List[Dict[str, Any]] = []
    kind = "memory" if args.memory else "throughput"
    print(
        f"Measuring {kind} for {len(args.models)} model(s) x {len(args.seq_lens)} "
        f"lengths on {device} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )
    measure = _measure_memory if args.memory else _measure_cell
    for name in args.models:
        for seq_len in args.seq_lens:
            row = measure(name, seq_len, device)
            rows.append(row)
            _write(output, rows)
            if args.memory:
                print(
                    f"  {name:14s} T={seq_len:>5}  "
                    f"infer {row['inference_peak_mib']!s:>10} MiB ({row['inference_status']})  "
                    f"train {row['train_peak_mib']!s:>10} MiB ({row['train_status']})",
                    flush=True,
                )
            elif row["status"] == "ok":
                print(
                    f"  {name:14s} T={seq_len:>5}  "
                    f"infer {row['inference_tokens_per_second']:>12,.0f} tok/s  "
                    f"train {row['train_tokens_per_second']:>12,.0f} tok/s  "
                    f"params {row['parameters']}",
                    flush=True,
                )
            else:
                print(
                    f"  {name:14s} T={seq_len:>5}  {row['status']}  {row['error'] or ''}",
                    flush=True,
                )

    meta = {
        "created_at": datetime.now().isoformat(),
        "device": str(device),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "basis": (
            "peak allocated memory, one forward and one train step"
            if args.memory
            else "bare backbone on synthetic input, formal-language matched setup"
        ),
        "batch_size": BATCH_SIZE,
        "warmup": 0 if args.memory else WARMUP,
        "repeats": 1 if args.memory else REPEATS,
        "input_dim": INPUT_DIM,
        "configs": {name: CONFIGS[name] for name in args.models},
        "rows": rows,
    }
    with open(output.with_suffix(".json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
