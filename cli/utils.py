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

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required for config files. Install with: pip install pyyaml")
        sys.exit(1)

    with open(path) as f:
        return yaml.safe_load(f)


def apply_config_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "epochs", None):
        config.setdefault("training", {})["epochs"] = args.epochs
    if getattr(args, "lr", None):
        config.setdefault("training", {})["learning_rate"] = args.lr
    if getattr(args, "batch_size", None):
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if getattr(args, "device", None):
        config.setdefault("training", {})["device"] = args.device
    if getattr(args, "wandb", False):
        config.setdefault("wandb", {})["enabled"] = True
    if getattr(args, "output_dir", None):
        config.setdefault("output", {})["dir"] = args.output_dir
    if getattr(args, "checkpoint_interval", None) is not None:
        config.setdefault("output", {})["checkpoint_interval"] = args.checkpoint_interval
    if getattr(args, "run_name", None):
        config.setdefault("output", {})["run_name"] = args.run_name
    if getattr(args, "no_plots", False):
        config.setdefault("output", {}).setdefault("plots", {})["enabled"] = False
    if getattr(args, "no_early_stop", False):
        config.setdefault("training", {})["early_stop_at_accuracy"] = None
    if getattr(args, "scheduler", None) is not None:
        val = None if args.scheduler == "none" else args.scheduler
        config.setdefault("training", {})["scheduler"] = val
    if getattr(args, "warmup_pct", None) is not None:
        config.setdefault("training", {})["warmup_pct"] = args.warmup_pct
    if getattr(args, "cosine_eta_min_ratio", None) is not None:
        config.setdefault("training", {})["cosine_eta_min_ratio"] = args.cosine_eta_min_ratio
    if getattr(args, "grad_clip", None) is not None:
        config.setdefault("training", {})["grad_clip_max_norm"] = args.grad_clip


def load_experiment_meta(configs_dir: Path) -> Dict[str, Any]:
    meta_path = Path(configs_dir) / "meta.yaml"
    if not meta_path.is_file():
        return {}
    try:
        import yaml
        with open(meta_path) as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def collect_all_run_dirs(outputs_dir: Path) -> List[Path]:
    from configs.paths import is_experiment_dir, is_group_dir

    outputs_dir = Path(outputs_dir)
    dirs: List[Path] = []
    if not outputs_dir.is_dir():
        return dirs

    if outputs_dir.name == "experiments":
        for entry in outputs_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith(".") or entry.name == "_legacy":
                continue
            if is_group_dir(entry):
                dirs.extend(collect_all_run_dirs(entry))
            else:
                run_root = entry / "results" / "outputs"
                if run_root.is_dir():
                    dirs.extend(collect_all_run_dirs(run_root))
        return sorted(set(dirs))

    if is_group_dir(outputs_dir):
        for entry in outputs_dir.iterdir():
            if not is_experiment_dir(entry):
                continue
            run_root = entry / "results" / "outputs"
            if run_root.is_dir():
                dirs.extend(collect_all_run_dirs(run_root))
        return sorted(set(dirs))

    for p in outputs_dir.iterdir():
        if not p.is_dir():
            continue
        if (p / "metrics.json").exists():
            dirs.append(p)
        else:
            for s in p.iterdir():
                if s.is_dir() and (s / "metrics.json").exists():
                    dirs.append(s)
    return sorted(dirs)


def add_training_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", "-c",
        help="Path to one YAML config. Omit when using --suite or --setup.",
    )
    parser.add_argument("--epochs", "-e", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", "-b", type=int, help="Override batch size")
    parser.add_argument("--device", "-d", choices=["cpu", "cuda", "mps", "auto"], help="Device to use")
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory for new runs. Suite and setup training defaults to runs/.",
    )
    parser.add_argument("--run-name", help="Custom run name")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping at 95%% accuracy")
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "step_decay", "reduce_on_plateau", "none"],
        help="LR scheduler (use 'none' to disable)",
    )
    parser.add_argument(
        "--warmup-pct",
        type=float,
        metavar="PCT",
        help="Warmup as fraction of total epochs (e.g. 0.05 = 5%%). "
             "Converted to an even epoch count at runtime. "
             "Overridden by explicit warmup_epochs in YAML.",
    )
    parser.add_argument(
        "--cosine-eta-min-ratio",
        type=float,
        metavar="RATIO",
        help="Minimum LR as a fraction of peak LR for cosine scheduling "
             "(eta_min = lr * ratio). Default: 0.1 (10%% of peak LR). "
             "Set to 0 for classic decay-to-zero behaviour.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        metavar="MAX_NORM",
        help="Gradient clipping max norm (e.g. 1.0). Clips via torch.nn.utils.clip_grad_norm_. "
             "Default: per-model (1.0 for LSTM, disabled for others).",
    )


def add_group_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--group",
        default=None,
        metavar="NAME",
        help=(
            "Experiment group (directory under experiments/). Defaults to "
            "the $HYBRID_GROUP environment variable, then to "
            "configs.paths.DEFAULT_GROUP. Use to scope the command to a "
            "single experiment group."
        ),
    )

