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
from collections import defaultdict
from typing import Dict, List

import numpy as np
import yaml

from ml.tasks.splits import input_key
from ml.training.pipeline import build_experiment_components
from reporting.heatmap_data import extract_early_stop_accuracy, solved_test_floor
from reporting.thesis_manifest import (
    entry,
    experiment_dir,
    load_manifest,
    runs_for,
    seed_of,
)
from utils import set_determinism

EXPERIMENTS = [
    "E1_headline",
    "E1_yhlstm_appendix",
    "E1_cell_stacks",
    "E2_block_cross",
    "E3_chunk",
]


def _all_runs() -> List[dict]:
    seen = set()
    runs = []
    for eid in EXPERIMENTS:
        for r in runs_for(eid):
            if r["run_dir"] not in seen:
                seen.add(r["run_dir"])
                runs.append(r)
    return runs


def _task_configs(runs: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in runs:
        cfg = r["config"]
        task = cfg["task"]["name"]
        if task not in out:
            out[task] = {
                "task_kwargs": {k: v for k, v in cfg["task"].items() if k != "name"},
                "dataset_size": cfg["training"]["dataset_size"],
                "val_split": cfg["training"].get("val_split", 0.1),
                "test_size": cfg["training"].get("test_size", 0),
            }
    return out


def print_distinct_inputs(runs: List[dict]) -> None:
    seeds = load_manifest()["seeds"]
    configs = _task_configs(runs)
    print("=== Distinct inputs per task pool (over the five seeds) ===")
    print(f"  {'task':28s} {'pool':>6s} {'distinct min':>13s} {'max':>6s} {'share':>7s}   val/test")
    for task in sorted(configs):
        c = configs[task]
        counts = []
        for seed in seeds:
            set_determinism(seed)
            payload = {
                "model": "lstm",
                "dataset_size": c["dataset_size"],
                "task_kwargs": c["task_kwargs"],
                "model_kwargs": {},
                "dropout": 0.0,
            }
            dataset, _, _, _, _ = build_experiment_components(task, payload)
            keys = {input_key(dataset.inputs[i]) for i in range(len(dataset))}
            counts.append(len(keys))
        n = c["dataset_size"]
        val_size = int(c["val_split"] * n)
        test_size = c["test_size"] or val_size
        n_unique = min(counts)
        if n_unique < val_size + test_size:
            holdout = val_size + test_size
            val_actual = max(1, round(val_size / holdout * n_unique * 0.25))
            test_actual = max(1, round(test_size / holdout * n_unique * 0.25))
        else:
            val_actual, test_actual = val_size, test_size
        print(
            f"  {task:28s} {n:6d} {min(counts):13d} {max(counts):6d} "
            f"{np.mean(counts) / n:7.1%}   {val_actual}/{test_actual}"
        )


def print_gap_statistics(runs: List[dict]) -> None:
    early_stop = extract_early_stop_accuracy(runs)
    stopped = [
        r for r in runs
        if (r["best"].get("val_accuracy") or 0.0) >= early_stop
        and r["best"].get("test_accuracy") is not None
    ]
    by_count: Dict[int, List[dict]] = defaultdict(list)
    for r in stopped:
        by_count[int(r["best"]["test_count"])].append(r)
    budget = sum(int(r["config"]["training"]["epochs"]) for r in runs)
    used = sum(int(r["epochs"]) for r in runs)
    print(f"\n=== Runs: {len(runs)} total, {len(stopped)} stopped at validation >= {early_stop} ===")
    print(f"  epochs used {used} of {budget} budgeted ({used / budget:.1%}); "
          f"median epochs of a stopped run {np.median([r['epochs'] for r in stopped]):.0f}")
    for count in sorted(by_count, reverse=True):
        group = by_count[count]
        gaps = np.array([r["best"]["val_accuracy"] - r["best"]["test_accuracy"] for r in group])
        sigma = math.sqrt(2 * early_stop * (1 - early_stop) / count)
        floor = solved_test_floor(early_stop, count)
        below = [r for r in group if r["best"]["test_accuracy"] < floor]
        print(f"\n--- test sets of {count} inputs: {len(group)} stopped runs ---")
        print(f"  validation minus test: mean {gaps.mean() * 100:+.2f} pp, "
              f"sample sd {gaps.std(ddof=1) * 100:.2f} pp, "
              f"range [{gaps.min() * 100:+.2f}, {gaps.max() * 100:+.2f}] pp")
        near = np.array([
            g for g, r in zip(gaps, group) if r["best"]["val_accuracy"] < early_stop + 0.005
        ])
        if len(near) > 1:
            print(f"  runs stopped below {early_stop + 0.005:.3f}: {len(near)}, "
                  f"sample sd {near.std(ddof=1) * 100:.2f} pp")
        print(f"  binomial sd of the difference at {early_stop}: {sigma * 100:.2f} pp; "
              f"floor {early_stop} - 3 sd = {floor:.4f}")
        print(f"  lowest test accuracy {min(r['best']['test_accuracy'] for r in group):.4f}; "
              f"runs below the floor: {len(below)}")
        for r in below:
            print(f"    {r['model']:12s} {r['task']:28s} seed {seed_of(r)} "
                  f"val {r['best']['val_accuracy']:.4f} test {r['best']['test_accuracy']:.4f}")


def main() -> None:
    runs = _all_runs()
    sources = sorted({s["experiment"] for eid in EXPERIMENTS for s in entry(eid)["sources"]})
    print("Sources: " + ", ".join(sources))
    for src in sources:
        with open(experiment_dir(src) / "setup" / "meta.yaml") as f:
            meta = yaml.safe_load(f)
        print(f"  {src}: split_mode {meta.get('split_mode')}, test_size {meta.get('test_size')}, "
              f"test_seed {meta.get('test_seed')}")
    print_gap_statistics(runs)
    print()
    print_distinct_inputs(runs)


if __name__ == "__main__":
    main()
