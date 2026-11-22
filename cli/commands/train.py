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
import time
from typing import Any, Dict

from cli.roster import SUITES, select_configs
from cli.utils import add_training_overrides, apply_config_overrides, load_config


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "train",
        help="Train one config, a suite, or every YAML in a setup directory",
    )
    add_training_overrides(p)
    p.add_argument(
        "--suite",
        choices=SUITES,
        help=(
            "Train the confirmation roster for this study: "
            "headline, cells, cross, chunk, or all."
        ),
    )
    p.add_argument(
        "--setup",
        metavar="DIR",
        help=(
            "Train every run YAML under this directory. "
            "With --suite, keep only that suite's models."
        ),
    )
    p.add_argument(
        "--model",
        help="Comma-separated model names to keep (for example hlstm,ylstm).",
    )
    p.add_argument(
        "--task",
        help="Comma-separated task names to keep (for example parity_check).",
    )
    p.add_argument(
        "--checkpoint-interval", "-I",
        type=int,
        metavar="N",
        help="Save a checkpoint every N epochs (e.g. 10)",
    )
    p.set_defaults(func=train)


def train(args: argparse.Namespace) -> None:
    batch = bool(args.suite or args.setup)
    if args.config and batch:
        print("Pass either -c or --suite/--setup, not both.", file=sys.stderr)
        sys.exit(2)
    if not args.config and not batch:
        print("Pass -c, --suite, or --setup.", file=sys.stderr)
        sys.exit(2)
    if not batch:
        config = load_config(args.config)
        apply_config_overrides(config, args)
        run_one(config)
        return

    paths = select_configs(
        suite=args.suite,
        setup=args.setup,
        model=args.model,
        task=args.task,
    )
    if not paths:
        print("No configs matched.", file=sys.stderr)
        sys.exit(1)
    if args.run_name:
        print("Omit --run-name with --suite or --setup. Each config keeps its own run name.", file=sys.stderr)
        sys.exit(2)
    if not args.output_dir:
        args.output_dir = "runs"
    print(f"Training {len(paths)} runs into {args.output_dir}")
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path}")
        config = load_config(str(path))
        apply_config_overrides(config, args)
        run_one(config)


def run_one(config: Dict[str, Any]) -> None:
    _run_with_pipeline(config)


def _run_with_pipeline(config: Dict[str, Any]) -> None:
    import torch
    from torch.utils.data import DataLoader, random_split

    from utils import set_determinism
    from utils.parameters import count_parameters
    from ml.training import Trainer, TrainerConfig, WandbConfig
    from ml.training.optimizer import create_optimizer, create_scheduler, get_resolved_optim_info
    from ml.training.output import RunOutput, OutputConfig, CheckpointListener
    from ml.training.pipeline import build_experiment_components
    from seqarch.console import console
    from seqarch.views import show_training_header

    task_cfg = config.get("task", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})
    output_cfg = config.get("output", {})

    from configs.defaults import TRAINING_DEFAULTS
    seed = train_cfg.get("seed") or task_cfg.get("seed") or TRAINING_DEFAULTS["seed"]
    set_determinism(seed)

    task_name = task_cfg.get("name", "parity_check")
    model_name = model_cfg.get("name", "lstm")

    payload = {
        "model": model_name,
        "dataset_size": train_cfg.get("dataset_size", 10000),
        "task_kwargs": {k: v for k, v in task_cfg.items() if k != "name"},
        "model_kwargs": {k: v for k, v in model_cfg.items() if k != "name"},
        "dropout": model_cfg.get("dropout", 0.1),
    }

    full_dataset, model_builder, loss_fn, metadata, _ = build_experiment_components(
        task_name, payload
    )

    output_config = OutputConfig.from_dict(output_cfg)
    run_output = RunOutput.create(
        base_dir=output_config.dir,
        task_name=task_name,
        model_name=model_name,
        config=config,
        output_config=output_config,
        run_name=output_cfg.get("run_name"),
    )

    val_split = train_cfg.get("val_split", 0.1)
    n = len(full_dataset)
    split_mode = train_cfg.get("split_mode", "legacy")
    test_size = int(train_cfg.get("test_size", 0) or 0)
    test_seed = int(train_cfg.get("test_seed", 43))
    test_ds = None

    if split_mode == "disjoint":
        from ml.tasks.splits import three_way_disjoint_split
        val_size = int(val_split * n)
        if test_size <= 0:
            test_size = val_size
        train_ds, val_ds, test_ds = three_way_disjoint_split(
            full_dataset, val_size, test_size, seed,
        )
        train_size = len(train_ds)
        val_size = len(val_ds)
    else:
        train_size = int((1 - val_split) * n)
        val_size = n - train_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        if test_size > 0:
            from ml.tasks.splits import draw_disjoint_test
            test_ds = draw_disjoint_test(full_dataset, test_size, test_seed)

    model = model_builder()
    param_count = count_parameters(model).total

    device = train_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

    _MIN_GPU_MB = 256
    fallback_reason = ""
    if device == "cuda":
        try:
            free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            if free_mb < _MIN_GPU_MB:
                fallback_reason = (
                    f"only {free_mb:.0f} MB free (need >= {_MIN_GPU_MB})"
                )
                device = "cpu"
        except Exception:
            pass

    gpu_summary = ""
    if device == "cuda":
        try:
            from ml.training.diagnostics import gpu_signature
            gpu = gpu_signature()
            if gpu.get("available"):
                name = gpu.get("name", "?")
                cap = gpu.get("capability", "?")
                free = gpu.get("memory_free_mb")
                total = gpu.get("memory_total_mb")
                mem = (
                    f", {free:.0f}/{total:.0f} MB free"
                    if free is not None and total is not None else ""
                )
                gpu_summary = f"{name} {cap}{mem}"
        except Exception:
            pass

    backend_label = ""
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        backend_label = getattr(backbone, "backend", "") or ""

    show_training_header(
        task_name, model_name, str(run_output.run_dir),
        backend=backend_label,
        device=device,
        gpu_summary=gpu_summary,
    )

    if fallback_reason:
        console.print(
            f"  [bold yellow]Warning:[/bold yellow] CUDA fallback to CPU -- {fallback_reason}"
        )
    test_count = len(test_ds) if test_ds is not None else 0
    test_note = ""
    if test_size > 0:
        test_note = f", [value]{test_count:,}[/value] test"
        if test_count < test_size:
            test_note += f" (requested {test_size:,}; input space exhausted)"
    console.print(
        f"  [label]Dataset:[/label] [value]{train_size:,}[/value] train, "
        f"[value]{val_size:,}[/value] val samples{test_note}"
    )
    console.print(f"  [label]Model:[/label]   [model]{model_name}[/model] with [value]{param_count:,}[/value] parameters")

    model_defaults = model.backbone.default_training_config()
    optimizer = create_optimizer(model, train_cfg, model_defaults)
    epochs = train_cfg.get("epochs", 100)
    scheduler = create_scheduler(optimizer, train_cfg, model_defaults, epochs=epochs)
    optim_info = get_resolved_optim_info(train_cfg, model_defaults, epochs=epochs)
    grad_clip = optim_info.get("grad_clip_max_norm")
    grad_clip_str = f"  grad_clip={grad_clip}" if grad_clip is not None else ""
    console.print(
        f"  [label]Optimizer:[/label] [value]{optim_info['optimizer']}[/value]  "
        f"lr=[value]{optim_info['learning_rate']}[/value]  "
        f"wd=[value]{optim_info['weight_decay']}[/value]  "
        f"scheduler=[value]{optim_info['scheduler'] or 'none'}[/value]{grad_clip_str}"
    )
    console.print()

    trainer_config = TrainerConfig(
        device=device,
        epochs=epochs,
        log_every=train_cfg.get("log_every", 10),
        early_stop_at_accuracy=train_cfg.get("early_stop_at_accuracy", 0.95),
        grad_clip_max_norm=grad_clip,
        wandb_config=WandbConfig(
            enabled=wandb_cfg.get("enabled", False),
            project=wandb_cfg.get("project", "hybrid-sequence-models"),
            run_name=wandb_cfg.get("run_name") or run_output.run_name,
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags", []),
        ),
    )
    experiment_metadata = {
        "task": task_name,
        "model": model_name,
        "batch_size": train_cfg.get("batch_size", 64),
        "dataset_size": payload["dataset_size"],
        "parameter_count": param_count,
        **optim_info,
    }
    listeners = []
    if output_config.save_checkpoint:
        listeners.append(
            CheckpointListener(
                run_output, output_config, model, optimizer, experiment_metadata
            )
        )
    sweep_info = config.get("_sweep")
    trainer = Trainer(
        model, optimizer, loss_fn, trainer_config,
        scheduler=scheduler,
        experiment_metadata=experiment_metadata,
        listeners=listeners,
        sweep_info=sweep_info,
    )
    batch_size = train_cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    start_time = time.time()
    history = trainer.fit(train_loader, val_loader)
    training_time = time.time() - start_time

    final_metrics = None
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        trainer.restore_best_validation()
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        scored = trainer.evaluate(test_loader)
        final_metrics = {
            "test_accuracy": scored["accuracy"],
            "test_token_accuracy": scored["token_accuracy"],
            "test_loss": scored["val_loss"],
            "test_count": test_count,
            "test_size": test_size,
            "test_seed": test_seed,
            "best_val_epoch": trainer.best_val_epoch,
        }
    elif test_size > 0:
        final_metrics = {
            "test_exhausted": True,
            "test_count": 0,
            "test_size": test_size,
            "test_seed": test_seed,
        }

    from rich.panel import Panel

    console.print()
    saved_files = run_output.finalize(
        model=model,
        optimizer=optimizer,
        history=history,
        device=device,
        training_time=training_time,
        extra_metadata=experiment_metadata,
        final_metrics=final_metrics,
    )

    result_lines = [f"[label]Output:[/label] [value]{run_output.run_dir}[/value]"]
    for name, path in saved_files.items():
        for p in (path if isinstance(path, list) else [path]):
            result_lines.append(f"  [dim]{getattr(p, 'name', p)}[/dim]")
    if history:
        final = history[-1]
        result_lines.append(f"\n[label]Final loss:[/label]     [value]{final.get('loss', 'N/A'):.6f}[/value]")
    if final_metrics is not None:
        result_lines.append(
            f"[label]Test accuracy:[/label] [value]{final_metrics['test_accuracy']:.4f}[/value]"
            f" on [value]{final_metrics['test_count']:,}[/value] inputs"
            f" (best val epoch {final_metrics['best_val_epoch']})"
        )
    result_lines.append(f"[label]Training time:[/label]  [value]{training_time:.2f}s[/value]")

    console.print(Panel(
        "\n".join(result_lines),
        title="[bold]Results[/bold]",
        border_style="green",
        padding=(0, 2),
    ))
    console.print()

    del trainer, model, optimizer, scheduler, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
