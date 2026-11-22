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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from reporting.constants import _MAMBA_VARIANTS


def group_mamba_variants(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for r in runs:
        if r.get("model") in _MAMBA_VARIANTS:
            r["model"] = "mamba"
    return runs


def _normalize_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for h in history:
        row = dict(h)
        if "val/accuracy" in row and "val_accuracy" not in row:
            row["val_accuracy"] = row["val/accuracy"]
        if "val/token_accuracy" in row and "val_token_accuracy" not in row:
            row["val_token_accuracy"] = row["val/token_accuracy"]
        if "train/loss" in row and "loss" not in row:
            row["loss"] = row["train/loss"]
        out.append(row)
    return out


def _enrich_best_from_history(best: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = dict(best)
    if best.get("val_accuracy") is None:
        vals = [
            h.get("val/accuracy", h.get("val_accuracy"))
            for h in history
            if "val/accuracy" in h or "val_accuracy" in h
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            best["val_accuracy"] = max(vals)
    if best.get("val_token_accuracy") is None:
        tok_vals = [
            h.get("val/token_accuracy", h.get("val_token_accuracy"))
            for h in history
            if "val/token_accuracy" in h or "val_token_accuracy" in h
        ]
        tok_vals = [v for v in tok_vals if v is not None]
        if tok_vals:
            best["val_token_accuracy"] = max(tok_vals)
    if best.get("token_accuracy") is None:
        tok_vals = [
            h.get("token_accuracy") for h in history if "token_accuracy" in h
        ]
        tok_vals = [v for v in tok_vals if v is not None]
        if tok_vals:
            best["token_accuracy"] = max(tok_vals)
    return best


def load_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    run_info_path = run_dir / "run_info.json"
    config_path = run_dir / "config.yaml"

    if not metrics_path.exists():
        return None

    data = {"run_dir": run_dir, "run_name": run_dir.name}

    if config_path.exists():
        with open(config_path) as f:
            data["config"] = yaml.safe_load(f)
    else:
        data["config"] = {}

    with open(metrics_path) as f:
        metrics = json.load(f)
    data["history"] = _normalize_history(metrics.get("history", []))
    data["epochs"] = metrics.get("epochs", len(data["history"]))
    data["best"] = _enrich_best_from_history(
        metrics.get("best", {}), metrics.get("history", [])
    )
    data["scaling"] = metrics.get("scaling")
    data["metrics_timestamp"] = metrics.get("timestamp")

    if run_info_path.exists():
        with open(run_info_path) as f:
            data["run_info"] = json.load(f)
    else:
        data["run_info"] = {}
        data["run_info"]["model"] = data["config"].get("model", {}).get("name", "?")
        data["run_info"]["task"] = data["config"].get("task", {}).get("name", "?")

    data["model"] = data["run_info"].get("model", "?")
    data["task"] = data["run_info"].get("task", "?")

    config = data.get("config") or {}
    hpo_block = config.get("_hpo") or {}
    training_cfg = config.get("training") or {}
    hpo_axis = hpo_block.get("axis")
    hpo_variant = hpo_block.get("variant")
    hpo_value = hpo_block.get("value")
    if hpo_value is None and hpo_axis == "learning_rate":
        hpo_value = training_cfg.get("learning_rate")
    data["hpo_variant"] = hpo_variant
    data["hpo_axis"] = hpo_axis
    data["hpo_value"] = hpo_value
    data["hpo_lr"] = training_cfg.get("learning_rate")
    return data


def _collect_run_dirs_from_path(p: Path) -> List[Path]:
    p = Path(p)
    if (p / "metrics.json").exists():
        return [p]
    if not p.is_dir():
        return []

    found: List[Path] = []

    direct_runs = [s for s in p.iterdir() if s.is_dir() and (s / "metrics.json").exists()]
    if direct_runs:
        found.extend(direct_runs)

    nested_outputs = p / "results" / "outputs"
    if nested_outputs.is_dir():
        for s in nested_outputs.iterdir():
            if s.is_dir() and (s / "metrics.json").exists():
                found.append(s)
            elif s.is_dir():
                for ss in s.iterdir():
                    if ss.is_dir() and (ss / "metrics.json").exists():
                        found.append(ss)

    if not found:
        for child in p.iterdir():
            if not child.is_dir() or child.name.startswith(".") or child.name == "_legacy":
                continue
            child_outputs = child / "results" / "outputs"
            if child_outputs.is_dir():
                for s in child_outputs.iterdir():
                    if s.is_dir() and (s / "metrics.json").exists():
                        found.append(s)
                    elif s.is_dir():
                        for ss in s.iterdir():
                            if ss.is_dir() and (ss / "metrics.json").exists():
                                found.append(ss)

    return sorted(set(found))


def experiment_id_for_outputs_dir(outputs_root: Path) -> Optional[str]:
    parts = Path(outputs_root).resolve().parts
    if "results" in parts:
        idx = parts.index("results")
        if idx >= 1:
            return parts[idx - 1]
    if "experiments" in parts:
        idx = parts.index("experiments")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def list_sweep_groups(outputs_dir: Path) -> List[Path]:
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        return []

    def _collect_for_experiment(entry: Path, acc: List[Path]) -> None:
        run_root = entry / "results" / "outputs"
        if not run_root.is_dir():
            return
        run_dirs = [
            s for s in run_root.iterdir()
            if s.is_dir() and (s / "metrics.json").exists()
        ]
        if run_dirs:
            acc.append(run_root)

    if outputs_dir.name == "experiments":
        groups: List[Path] = []
        for entry in outputs_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith(".") or entry.name == "_legacy":
                continue
            if (entry / "results" / "outputs").is_dir():
                _collect_for_experiment(entry, groups)
            else:
                for sub in entry.iterdir():
                    if sub.is_dir() and not sub.name.startswith("."):
                        _collect_for_experiment(sub, groups)
        return sorted(groups)

    if any(
        (e.is_dir() and (e / "results" / "outputs").is_dir())
        for e in outputs_dir.iterdir()
    ):
        groups: List[Path] = []
        for entry in outputs_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith(".") or entry.name == "_legacy":
                continue
            _collect_for_experiment(entry, groups)
        if groups:
            return sorted(groups)

    groups = []
    for p in outputs_dir.iterdir():
        if not p.is_dir():
            continue
        if (p / "metrics.json").exists():
            continue
        run_dirs = [s for s in p.iterdir() if s.is_dir() and (s / "metrics.json").exists()]
        if run_dirs:
            groups.append(p)
    return sorted(groups)


def list_sweep_groups_with_counts(outputs_dir: Path) -> List[Tuple[Path, int]]:
    groups = list_sweep_groups(outputs_dir)
    result = []
    for g in groups:
        count = sum(1 for s in g.iterdir() if s.is_dir() and (s / "metrics.json").exists())
        result.append((g, count))
    return result


def load_runs(
    run_dirs: Optional[List[Path]] = None,
    outputs_dir: Optional[Path] = None,
    pattern: str = "*",
) -> List[Dict[str, Any]]:
    if run_dirs is not None:
        dirs = []
        for d in run_dirs:
            dirs.extend(_collect_run_dirs_from_path(Path(d)))
        dirs = sorted(set(dirs))
    elif outputs_dir is not None:
        outputs_dir = Path(outputs_dir)
        dirs: List[Path] = []
        run_roots: List[Path] = []
        if outputs_dir.is_dir() and outputs_dir.name == "experiments":
            for entry in outputs_dir.iterdir():
                if not entry.is_dir() or entry.name.startswith(".") or entry.name == "_legacy":
                    continue
                run_root = entry / "results" / "outputs"
                if run_root.is_dir():
                    run_roots.append(run_root)
                else:
                    for sub in entry.iterdir():
                        if not sub.is_dir() or sub.name.startswith("."):
                            continue
                        sub_root = sub / "results" / "outputs"
                        if sub_root.is_dir():
                            run_roots.append(sub_root)
            iter_roots = run_roots
        elif (
            outputs_dir.is_dir()
            and any(
                (e.is_dir() and (e / "results" / "outputs").is_dir())
                for e in outputs_dir.iterdir()
            )
        ):
            for entry in outputs_dir.iterdir():
                if not entry.is_dir() or entry.name.startswith(".") or entry.name == "_legacy":
                    continue
                run_root = entry / "results" / "outputs"
                if run_root.is_dir():
                    run_roots.append(run_root)
            iter_roots = run_roots
        else:
            iter_roots = [outputs_dir] if outputs_dir.is_dir() else []
        for root in iter_roots:
            for p in root.iterdir():
                if not p.is_dir():
                    continue
                if (p / "metrics.json").exists():
                    dirs.append(p)
                else:
                    for s in p.iterdir():
                        if s.is_dir() and (s / "metrics.json").exists():
                            dirs.append(s)
        if pattern != "*":
            dirs = [p for p in dirs if Path(p).match(pattern)]
        dirs = sorted(set(dirs))
    else:
        return []

    runs = []
    for d in dirs:
        r = load_run(d)
        if r is not None:
            runs.append(r)
    return runs
