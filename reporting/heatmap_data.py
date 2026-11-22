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

import math
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from configs.registry import CHOMSKY_BY_HIERARCHY, MQAR_TASKS
from evaluation.reporting import _LEVEL_LABELS
from evaluation.task_metadata import TASK_SCALING_REGISTRY
from visualization.utils import is_valid_number as _is_valid


def _output_length_for_run(run: Dict[str, Any]) -> int:
    from evaluation.task_metadata import get_scaling_meta

    scaling = run.get("scaling") or {}
    if "output_length" in scaling:
        return int(scaling["output_length"])

    run_info = run.get("run_info", {})
    if "output_length" in run_info:
        return int(run_info["output_length"])

    task = run.get("task", "?")
    if task in TASK_SCALING_REGISTRY:
        task_cfg = run.get("config", {}).get("task") or {}
        seq_len = 32
        if isinstance(task_cfg, dict):
            seq_len = task_cfg.get("sequence_length", 32)
        elif hasattr(task_cfg, "sequence_length"):
            seq_len = task_cfg.sequence_length
        meta = get_scaling_meta(task)
        return meta.get_output_length(seq_len, task_cfg)

    return 1


def _collect_by_model_task(
    runs: List[Dict[str, Any]],
    task_ok,
    extract_value,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    cell_values: Dict[Tuple[str, str], List[float]] = {}
    for r in runs:
        task = r.get("task", "?")
        if not task_ok(task):
            continue
        val = extract_value(r)
        if val is None:
            continue
        model = r.get("model", "?")
        cell_values.setdefault((model, task), []).append(float(val))

    results: Dict[str, Dict[str, float]] = {}
    std_results: Dict[str, Dict[str, float]] = {}
    for (model, task), vals in cell_values.items():
        arr = np.array(vals)
        results.setdefault(model, {})[task] = float(np.mean(arr))
        std_results.setdefault(model, {})[task] = float(np.std(arr)) if len(vals) > 1 else 0.0
    return results, std_results


def _extract_best_accuracy(r: Dict[str, Any]):
    best = r.get("best") or {}
    if best.get("test_exhausted"):
        return None
    if best.get("test_accuracy") is not None:
        return best["test_accuracy"]
    val = best.get("val_accuracy")
    if val is None:
        val = best.get("accuracy")
    return val


def solved_test_floor(early_stop: float, test_count: int, band_sd: float = 3.0) -> float:
    sigma = math.sqrt(2.0 * early_stop * (1.0 - early_stop) / test_count)
    return early_stop - band_sd * sigma


def is_solved(run: Dict[str, Any], early_stop: float = 0.95) -> bool:
    best = run.get("best") or {}
    val = best.get("val_accuracy")
    if val is None:
        val = best.get("accuracy")
    if val is None or float(val) < early_stop:
        return False
    test = best.get("test_accuracy")
    count = best.get("test_count")
    if test is None or not count:
        return True
    return float(test) >= solved_test_floor(early_stop, int(count))


def build_heatmap_solved_matrix(
    runs: List[Dict[str, Any]],
    early_stop: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]:
    flags: Dict[Tuple[str, str], List[bool]] = {}
    for r in runs:
        key = (r.get("model", "?"), r.get("task", "?"))
        flags.setdefault(key, []).append(is_solved(r, early_stop))

    fraction: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, str]] = {}
    for (model, task), vals in flags.items():
        fraction.setdefault(model, {})[task] = sum(vals) / len(vals)
        counts.setdefault(model, {})[task] = f"{sum(vals)}/{len(vals)}"
    return fraction, counts


def build_heatmap_scaled_matrix(
    runs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    from evaluation.scaling import compute_scaled_accuracy, compute_srand

    def extract(r):
        raw = _extract_best_accuracy(r)
        if raw is None:
            return None
        if (r.get("best") or {}).get("test_accuracy") is None:
            scaling = r.get("scaling") or {}
            if scaling and "scaled_accuracy" in scaling:
                return scaling["scaled_accuracy"]
        task_config = r.get("config", {}).get("task")
        out_len = _output_length_for_run(r)
        srand = compute_srand(r.get("task", "?"), task_config, output_length=out_len)
        return compute_scaled_accuracy(raw, srand)

    return _collect_by_model_task(
        runs, lambda t: t in TASK_SCALING_REGISTRY, extract
    )


def build_heatmap_raw_matrix(
    runs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    return _collect_by_model_task(
        runs,
        lambda t: t in TASK_SCALING_REGISTRY or t in MQAR_TASKS,
        _extract_best_accuracy,
    )


def build_heatmap_token_accuracy_matrix(
    runs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    def extract(r):
        best = r.get("best") or {}
        if best.get("test_exhausted"):
            return None
        if best.get("test_token_accuracy") is not None:
            return best["test_token_accuracy"]
        val = best.get("val_token_accuracy")
        if val is None:
            val = best.get("token_accuracy")
        return val
    return _collect_by_model_task(
        runs,
        lambda t: t in TASK_SCALING_REGISTRY or t in MQAR_TASKS,
        extract,
    )


def build_heatmap_epoch_matrix(
    runs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    def extract(r):
        epochs = r.get("epochs")
        if epochs is None:
            history = r.get("history")
            if history:
                epochs = len(history)
        return epochs
    return _collect_by_model_task(
        runs,
        lambda t: t in TASK_SCALING_REGISTRY or t in MQAR_TASKS,
        extract,
    )


def heatmap_task_groups() -> Dict[str, List[str]]:
    return {
        _LEVEL_LABELS["regular"]: CHOMSKY_BY_HIERARCHY["regular"],
        _LEVEL_LABELS["dcf"]: CHOMSKY_BY_HIERARCHY["dcf"],
        _LEVEL_LABELS["cs"]: CHOMSKY_BY_HIERARCHY["cs"],
        "MQAR": MQAR_TASKS,
    }


def classify_uninformative_columns(
    raw_results: Dict[str, Dict[str, float]],
    scaled_results: Dict[str, Dict[str, float]],
    all_tasks: List[str],
    early_stop_accuracy: float = 0.95,
    near_zero_threshold: float = 0.05,
) -> Set[int]:
    models_raw = list(raw_results.keys())
    models_scaled = list(scaled_results.keys())
    greyed: Set[int] = set()

    for j, task in enumerate(all_tasks):
        raw_vals = [v for v in (raw_results[m].get(task) for m in models_raw) if _is_valid(v)]
        scaled_vals = [v for v in (scaled_results[m].get(task) for m in models_scaled) if _is_valid(v)]

        if not raw_vals and not scaled_vals:
            greyed.add(j)
            continue

        solved_by_all = raw_vals and all(v >= early_stop_accuracy for v in raw_vals)
        solved_by_none = scaled_vals and all(v <= near_zero_threshold for v in scaled_vals)

        if solved_by_all or solved_by_none:
            greyed.add(j)

    return greyed


def extract_early_stop_accuracy(runs: List[Dict[str, Any]]) -> float:
    values = set()
    for r in runs:
        v = r.get("config", {}).get("training", {}).get("early_stop_at_accuracy")
        if v is not None:
            values.add(float(v))
    if len(values) == 1:
        return values.pop()
    return 0.95


def mean_scaled_accuracy_per_model(
    heatmap_results: Dict[str, Dict[str, float]],
    task_list: List[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for model, task_to_acc in heatmap_results.items():
        vals = [task_to_acc[t] for t in task_list if t in task_to_acc]
        if vals:
            out[model] = float(np.mean(vals))
    return out


def average_performance_per_model(
    heatmap_results: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for model, task_to_acc in heatmap_results.items():
        vals = list(task_to_acc.values())
        if vals:
            out[model] = float(np.mean(vals))
    return out
