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

from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from evaluation.scaling import (
    WeightPolicy,
    compute_task_weights,
    compute_weighted_accuracy,
)
from evaluation.task_metadata import get_scaling_meta


@dataclass
class TaskResult:

    task_name: str
    raw_accuracy: float
    srand: float
    scaled_accuracy: float
    chomsky_level: str
    weight: float = 1.0

    @property
    def weighted_contribution(self) -> float:
        return self.weight * self.scaled_accuracy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task_name,
            "raw_accuracy": round(self.raw_accuracy, 6),
            "srand": round(self.srand, 6),
            "scaled_accuracy": round(self.scaled_accuracy, 6),
            "chomsky_level": self.chomsky_level,
            "weight": round(self.weight, 6),
            "weighted_contribution": round(self.weighted_contribution, 6),
        }


def build_task_results(
    raw_accuracies: Dict[str, float],
    srands: Dict[str, float],
    scaled_accuracies: Dict[str, float],
    weights: Dict[str, float],
) -> List[TaskResult]:
    results: List[TaskResult] = []
    for task in sorted(raw_accuracies):
        meta = get_scaling_meta(task)
        results.append(TaskResult(
            task_name=task,
            raw_accuracy=raw_accuracies[task],
            srand=srands[task],
            scaled_accuracy=scaled_accuracies[task],
            chomsky_level=meta.chomsky_level,
            weight=weights.get(task, 1.0),
        ))
    return results


_HEADER = (
    "| {task:<30s} | {raw:>12s} | {srand:>8s} | {scaled:>14s} "
    "| {weight:>8s} | {contrib:>10s} |"
)
_SEP = "|" + "-" * 32 + "|" + "-" * 14 + "|" + "-" * 10 + "|" + "-" * 16 + "|" + "-" * 10 + "|" + "-" * 12 + "|"
_ROW = (
    "| {task:<30s} | {raw:>12.4f} | {srand:>8.4f} | {scaled:>14.4f} "
    "| {weight:>8.4f} | {contrib:>10.4f} |"
)


def format_task_table(
    results: List[TaskResult],
) -> Tuple[str, List[Dict[str, Any]]]:
    header = _HEADER.format(
        task="Task", raw="Raw Accuracy", srand="srand",
        scaled="Scaled Accuracy", weight="Weight", contrib="Wtd Contr.",
    )
    lines = [header, _SEP]
    dicts: List[Dict[str, Any]] = []

    for r in results:
        lines.append(_ROW.format(
            task=r.task_name,
            raw=r.raw_accuracy,
            srand=r.srand,
            scaled=r.scaled_accuracy,
            weight=r.weight,
            contrib=r.weighted_contribution,
        ))
        dicts.append(r.to_dict())

    return "\n".join(lines), dicts


_LEVEL_LABELS = {
    "regular": "Regular (Type-3)",
    "dcf": "DCF (Type-2)",
    "cs": "CS (Type-1)",
    "mqar": "MQAR",
}


def format_level_summary(
    results: List[TaskResult],
    policy: WeightPolicy = WeightPolicy.UNIFORM_LEVEL,
) -> Tuple[str, Dict[str, Any]]:
    level_accs: Dict[str, List[float]] = {}
    for r in results:
        level_accs.setdefault(r.chomsky_level, []).append(r.scaled_accuracy)

    level_order = ["regular", "dcf", "cs", "mqar"]
    lines: List[str] = []
    summary: Dict[str, Any] = {}

    for lvl in level_order:
        if lvl not in level_accs:
            continue
        accs = level_accs[lvl]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        label = _LEVEL_LABELS.get(lvl, lvl)
        lines.append(
            f"  {label:<20s}  "
            f"mean_scaled={mean_acc:.4f}  "
            f"tasks={len(accs)}"
        )
        summary[lvl] = {
            "label": label,
            "mean_scaled_accuracy": round(mean_acc, 6),
            "num_tasks": len(accs),
            "tasks": sorted(
                r.task_name for r in results if r.chomsky_level == lvl
            ),
        }

    return "\n".join(lines), summary


def format_run_summary(
    results: List[TaskResult],
    weighted_accuracy: float,
) -> Tuple[str, Dict[str, Any]]:
    scaled_vals = [r.scaled_accuracy for r in results]
    if not scaled_vals:
        text = "No task results available."
        return text, {}

    s_min = min(scaled_vals)
    s_max = max(scaled_vals)
    s_med = median(scaled_vals)

    lines = [
        f"  Weighted Accuracy : {weighted_accuracy:.4f}",
        f"  Scaled Acc  min   : {s_min:.4f}",
        f"  Scaled Acc  median: {s_med:.4f}",
        f"  Scaled Acc  max   : {s_max:.4f}",
        f"  Tasks evaluated   : {len(scaled_vals)}",
    ]
    summary = {
        "weighted_accuracy": round(weighted_accuracy, 6),
        "min_scaled_accuracy": round(s_min, 6),
        "median_scaled_accuracy": round(s_med, 6),
        "max_scaled_accuracy": round(s_max, 6),
        "num_tasks": len(scaled_vals),
    }
    return "\n".join(lines), summary


def generate_scaled_report(
    raw_accuracies: Dict[str, float],
    task_configs: Optional[Dict[str, Any]] = None,
    policy: WeightPolicy = WeightPolicy.UNIFORM_LEVEL,
    custom_weights: Optional[Dict[str, float]] = None,
    *,
    clamp: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    from evaluation.scaling import compute_scaled_accuracy, compute_srand

    tasks = sorted(raw_accuracies)
    srands: Dict[str, float] = {}
    scaled: Dict[str, float] = {}
    configs = task_configs or {}

    for t in tasks:
        cfg = configs.get(t)
        seq_len = 32
        if cfg is not None:
            if isinstance(cfg, dict):
                seq_len = cfg.get("sequence_length", 32)
            elif hasattr(cfg, "sequence_length"):
                seq_len = cfg.sequence_length
        meta = get_scaling_meta(t)
        out_len = meta.get_output_length(seq_len, cfg)
        s = compute_srand(t, cfg, output_length=out_len)
        srands[t] = s
        scaled[t] = compute_scaled_accuracy(raw_accuracies[t], s, clamp=clamp)

    weights = compute_task_weights(tasks, policy, custom_weights=custom_weights)
    w_acc = compute_weighted_accuracy(scaled, weights)
    results = build_task_results(raw_accuracies, srands, scaled, weights)

    table_str, table_data = format_task_table(results)
    level_str, level_data = format_level_summary(results, policy)
    summary_str, summary_data = format_run_summary(results, w_acc)

    full_report = "\n".join([
        "=" * 80,
        "  Chomsky Benchmark — Scaled Accuracy Report",
        "=" * 80,
        "",
        "Per-task results:",
        table_str,
        "",
        "Per-level summary:",
        level_str,
        "",
        "Run summary:",
        summary_str,
        "=" * 80,
    ])

    structured = {
        "tasks": table_data,
        "levels": level_data,
        "summary": summary_data,
        "weight_policy": policy.value,
    }
    return full_report, structured
