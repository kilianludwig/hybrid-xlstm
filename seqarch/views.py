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

from typing import Any, Dict, Optional

from rich.panel import Panel
from rich.tree import Tree

from seqarch.console import (
    console,
    create_table,
)


def show_models_table() -> None:
    from configs.registry import MODELS
    from ml.models import MODEL_REGISTRY, XLSTM_LOAD_ERROR
    from utils.env import format_env

    table = create_table("Available Models", "Name", "Status", "Description")
    table.columns[0].style = "model"
    table.columns[1].min_width = 14

    for name in MODELS:
        if name in MODEL_REGISTRY:
            cls = MODEL_REGISTRY[name]
            desc = (cls.__doc__ or "").split("\n")[0][:50]
            table.add_row(name, "[green]available[/green]", desc)
        else:
            reason = "not installed"
            if XLSTM_LOAD_ERROR and name in ("slstm", "mlstm", "xlstm", "xlstm_7_1"):
                reason = str(XLSTM_LOAD_ERROR).split("\n")[0][:45]
            table.add_row(name, f"[dim]{reason}[/dim]", "")

    available = sum(1 for name in MODELS if name in MODEL_REGISTRY)
    console.print()
    console.print(f"  [label]Environment:[/label] [value]{format_env()}[/value]")
    console.print(table)
    console.print(f"  [muted]{available} of {len(MODELS)} models available[/muted]")
    console.print()


def show_tasks_table() -> None:
    from ml.tasks import TASK_REGISTRY, get_task_class

    levels = {
        "regular": ("Type-3: Regular", []),
        "dcf": ("Type-2: DCF", []),
        "cs": ("Type-1: CS", []),
    }

    for name, info in TASK_REGISTRY.items():
        module = info.get("module", "unknown")
        if module in levels:
            try:
                task_cls = get_task_class(name)
                task_instance = task_cls()
                output_len = task_instance.output_length(10)
                task_type = "single_label" if output_len == 1 else "sequence"
            except (ImportError, TypeError, ValueError, AttributeError):
                task_type = "?"
            levels[module][1].append((name, task_type))

    table = create_table("Available Tasks", "Task", "Family", "Type")
    table.columns[0].style = "task"
    table.columns[1].min_width = 18

    for module, (family_label, tasks) in levels.items():
        for name, task_type in sorted(tasks):
            type_style = "green" if task_type == "single_label" else "yellow"
            table.add_row(name, family_label, f"[{type_style}]{task_type}[/{type_style}]")

    table.add_row("mqar", "MQAR", "[yellow]sequence[/yellow]")

    total = sum(len(tasks) for _, tasks in levels.values()) + 1
    console.print()
    console.print(table)
    console.print(f"  [muted]{total} tasks available[/muted]")
    console.print()


def show_task_info(task_name: str) -> None:
    import dataclasses
    from ml.tasks import TASK_REGISTRY, get_task_info, get_task_class

    if task_name not in TASK_REGISTRY:
        console.print(f"  [error]Unknown task '{task_name}'[/error]")
        console.print(f"  Available: {', '.join(sorted(TASK_REGISTRY.keys()))}")
        return

    info = get_task_info(task_name)
    config_class = info["config"]

    try:
        task_cls = get_task_class(task_name)
        task_instance = task_cls()
        output_len = task_instance.output_length(10)
        detected_type = "single_label" if output_len == 1 else "sequence_output"
    except (ImportError, TypeError, ValueError, AttributeError):
        detected_type = "unknown"

    panel_lines = [
        f"[label]Type:[/label]           [value]{detected_type}[/value]",
        f"[label]Chomsky Level:[/label]   [value]{info.get('module', 'unknown').upper()}[/value]",
        f"[label]Config Class:[/label]    [value]{config_class.__name__}[/value]",
    ]

    if dataclasses.is_dataclass(config_class):
        panel_lines.append("")
        panel_lines.append("[section]Configuration Fields[/section]")
        for field in dataclasses.fields(config_class):
            default = field.default if field.default is not dataclasses.MISSING else field.default_factory()
            panel_lines.append(f"  [key]{field.name:<25}[/key] = [value]{default}[/value]")

    console.print()
    console.print(Panel(
        "\n".join(panel_lines),
        title=f"[bold]Task: {task_name}[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def show_config_preview(config: Dict[str, Any]) -> None:
    task_cfg = config.get("task", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})
    wandb_cfg = config.get("wandb", {})

    tree = Tree("[bold]Experiment Configuration[/bold]")

    task_branch = tree.add("[section]Task[/section]")
    for k, v in task_cfg.items():
        task_branch.add(f"[key]{k}[/key] = [value]{v}[/value]")

    model_branch = tree.add("[section]Model[/section]")
    for k, v in model_cfg.items():
        model_branch.add(f"[key]{k}[/key] = [value]{v}[/value]")

    train_branch = tree.add("[section]Training[/section]")
    display_keys = [
        "epochs", "batch_size", "optimizer", "learning_rate", "weight_decay",
        "grad_clip_max_norm", "scheduler", "warmup_pct", "cosine_eta_min_ratio",
        "dataset_size", "val_split", "early_stop_at_accuracy", "seed", "device",
    ]
    for k in display_keys:
        v = train_cfg.get(k)
        if v is not None:
            train_branch.add(f"[key]{k}[/key] = [value]{v}[/value]")

    output_branch = tree.add("[section]Output[/section]")
    output_branch.add(f"[key]dir[/key] = [value]{output_cfg.get('dir', './experiments')}[/value]")
    output_branch.add(f"[key]checkpoint[/key] = [value]{output_cfg.get('save_checkpoint', True)}[/value]")

    wandb_branch = tree.add("[section]W&B[/section]")
    wandb_branch.add(f"[key]enabled[/key] = [value]{wandb_cfg.get('enabled', False)}[/value]")
    if wandb_cfg.get("enabled"):
        wandb_branch.add(f"[key]project[/key] = [value]{wandb_cfg.get('project', '')}[/value]")

    console.print()
    console.print(Panel(tree, border_style="cyan", padding=(1, 2)))
    console.print()


def show_training_header(
    task_name: str,
    model_name: str,
    run_dir: str,
    *,
    backend: str = "",
    device: str = "",
    gpu_summary: str = "",
) -> None:
    from utils.env import format_env

    model_str = f"[model]{model_name}[/model]"
    if backend:
        model_str += f" [muted]({backend})[/muted]"

    device_part = ""
    if device:
        device_part = f"[label]Device:[/label]  [value]{device}[/value]"
        if gpu_summary:
            device_part += f"   [muted]{gpu_summary}[/muted]"

    lines = [
        f"[label]Task:[/label]    [task]{task_name}[/task]",
        f"[label]Model:[/label]   {model_str}",
        f"[label]Env:[/label]     [value]{format_env()}[/value]",
    ]
    if device_part:
        lines.append(device_part)
    lines.append(f"[label]Output:[/label]  [value]{run_dir}[/value]")

    console.print()
    console.print(Panel(
        "\n".join(lines),
        title="Training",
        border_style="bright_cyan",
        padding=(0, 2),
    ))
    console.print()


def show_training_params_summary(
    scale_name: str,
    scale: float,
    label: str,
    training_overrides: Optional[Dict[str, Any]],
    chomsky_seq_len: Optional[int],
    mqar_seq_len: Optional[int],
) -> None:
    from configs.defaults import TRAINING_DEFAULTS

    effective = dict(TRAINING_DEFAULTS)
    if training_overrides:
        effective.update(training_overrides)

    console.print()
    console.print(f"  [label]Scale:[/label]            [value]{scale_name} ({scale}x)[/value]")
    console.print(f"  [label]Label:[/label]            [value]{label or '(auto)'}[/value]")
    console.print(f"  [label]Epochs:[/label]           [value]{effective['epochs']}[/value]")
    console.print(f"  [label]Batch size:[/label]       [value]{effective['batch_size']}[/value]")
    console.print(f"  [label]Dataset size:[/label]     [value]{effective['dataset_size']:,}[/value]")
    console.print(f"  [label]Val split:[/label]        [value]{effective.get('val_split')}[/value]")
    early = effective.get("early_stop_at_accuracy")
    console.print(f"  [label]Early stop:[/label]       [value]{early if early is not None else 'None'}[/value]")
    console.print(f"  [label]Seed:[/label]             [value]{effective.get('seed', 42)}[/value]")
    console.print(f"  [label]Chomsky seq len:[/label]  [value]{chomsky_seq_len}[/value]")
    console.print(f"  [label]MQAR seq len:[/label]     [value]{mqar_seq_len}[/value]")

