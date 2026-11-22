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
from pathlib import Path
from typing import Any, Dict, List

import yaml

from reporting.loading import load_runs

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
MANIFEST_PATH = EXPERIMENTS_DIR / "thesis_manifest.yaml"
INDEX_PATH = EXPERIMENTS_DIR / "INDEX.md"

_manifest_cache: Dict[str, Any] = {}


def load_manifest() -> Dict[str, Any]:
    if not _manifest_cache:
        with open(MANIFEST_PATH) as f:
            _manifest_cache.update(yaml.safe_load(f))
    return _manifest_cache


def experiment_dir(rel: str) -> Path:
    return EXPERIMENTS_DIR / rel


def entry(experiment_id: str) -> Dict[str, Any]:
    return load_manifest()["experiments"][experiment_id]


def labels() -> Dict[str, str]:
    return dict(load_manifest()["labels"])


def model_of(run: Dict[str, Any]) -> str:
    return (
        run.get("run_info", {}).get("model")
        or run.get("config", {}).get("model", {}).get("name", "")
    )


def seed_of(run: Dict[str, Any]) -> int:
    seed = run.get("run_info", {}).get("seed")
    if seed is None:
        config = run.get("config", {})
        seed = (
            config.get("training", {}).get("seed")
            or config.get("task", {}).get("seed")
            or 42
        )
    return int(seed)


def runs_for(experiment_id: str) -> List[Dict[str, Any]]:
    seeds = set(load_manifest()["seeds"])
    runs: List[Dict[str, Any]] = []
    for source in entry(experiment_id)["sources"]:
        outputs = experiment_dir(source["experiment"]) / "results" / "outputs"
        wanted = set(source["models"])
        tasks = set(source.get("tasks") or [])
        loaded = [
            r
            for r in load_runs(run_dirs=[outputs])
            if model_of(r) in wanted
            and seed_of(r) in seeds
            and (not tasks or r.get("task") in tasks)
        ]
        if tasks:
            replaced = {(model_of(r), r.get("task"), seed_of(r)) for r in loaded}
            runs = [
                r for r in runs
                if (model_of(r), r.get("task"), seed_of(r)) not in replaced
            ]
        runs.extend(loaded)
    return runs


def _config_sources(experiment_id: str) -> List[Dict[str, Any]]:
    e = entry(experiment_id)
    return e.get("config_sources", e["sources"])


def seeds_of(runs: List[Dict[str, Any]]) -> List[int]:
    return sorted({seed_of(r) for r in runs})


def models_for(experiment_id: str) -> List[str]:
    out: List[str] = []
    for group in entry(experiment_id)["models"].values():
        out.extend(m for m in group if m not in out)
    return out


def benchmark_csv(experiment_id: str) -> Path:
    return EXPERIMENTS_DIR / entry(experiment_id)["benchmark_csv"]


def source_meta(experiment_id: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for source in _config_sources(experiment_id):
        with open(experiment_dir(source["experiment"]) / "setup" / "meta.yaml") as f:
            meta = yaml.safe_load(f)
        for m in source["models"]:
            out[m] = meta
    return out


def source_summary_csv(experiment_id: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for source in _config_sources(experiment_id):
        path = experiment_dir(source["experiment"]) / "results" / "report" / "data" / "summary.csv"
        for m in source["models"]:
            out[m] = path
    return out


def lr_hpo_csvs(experiment_id: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for source in entry(experiment_id)["lr_sources"]:
        path = experiment_dir(source["experiment"]) / "results" / "report" / "data" / "lr_hpo_best.csv"
        for m in source["models"]:
            out[m] = path
    return out


_STATUS_TEXT = {
    "thesis": "thesis",
    "predecessor": "predecessor (linked)",
    "config": "config source",
    "superseded": "superseded",
    "invalid": "invalid (pre-fix)",
    "out_of_scope": "out of scope",
}


def write_index() -> Path:
    m = load_manifest()
    lines = [
        "# Experiment index",
        "",
        "Generated from `experiments/thesis_manifest.yaml` by "
        "`python -m reporting.thesis_manifest --write-index`. Do not edit by hand.",
        "",
        "## Thesis experiments",
        "",
    ]
    for eid, e in m["experiments"].items():
        lines.append(f"### {eid}: {e['title']}")
        lines.append("")
        lines.append("| Source | Models |")
        lines.append("|--------|--------|")
        for s in e["sources"]:
            models = ", ".join(s["models"])
            if s.get("tasks"):
                models += f" (replaces {', '.join(s['tasks'])})"
            lines.append(f"| `{s['experiment']}` | {models} |")
        lines.append("")
        if e.get("config_sources"):
            lines.append("Configs (widths, rates, parameter counts): " + "; ".join(
                f"`{s['experiment']}` ({', '.join(s['models'])})" for s in e["config_sources"]
            ))
            lines.append("")
        lines.append("Learning rates: " + "; ".join(
            f"`{s['experiment']}` ({', '.join(s['models'])})" for s in e["lr_sources"]
        ))
        if e.get("pending"):
            lines.append("")
            lines.append("Pending arms: " + "; ".join(e["pending"]))
        if "benchmark_csv" in e:
            lines.append("")
            lines.append(f"Throughput: `{e['benchmark_csv']}`")
        if e.get("supersedes"):
            lines.append("")
            lines.append("Supersedes: " + ", ".join(f"`{x}`" for x in e["supersedes"]))
        lines.append("")

    lineage = m.get("lineage") or []
    if lineage:
        lines += [
            "## Lineage",
            "",
        ]
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for item in lineage:
        by_group.setdefault(item["id"].split("/")[0], []).append(item)
    for group, items in by_group.items():
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| Experiment | Role | Status | Used by / superseded by | Note |")
        lines.append("|------------|------|--------|-------------------------|------|")
        for it in items:
            name = it["id"].split("/", 1)[1]
            link = ", ".join(it.get("used_by", [])) or (
                f"`{it['superseded_by'].split('/', 1)[1]}`" if it.get("superseded_by") else ""
            )
            lines.append(
                f"| `{name}` | {it['role']} | {_STATUS_TEXT[it['status']]} | {link} | {it.get('note', '') or ''} |"
            )
        lines.append("")
    INDEX_PATH.write_text("\n".join(lines))
    return INDEX_PATH


def main() -> None:
    p = argparse.ArgumentParser(description="Resolve thesis experiment sources.")
    p.add_argument("--write-index", action="store_true", help="regenerate experiments/INDEX.md")
    args = p.parse_args()
    if args.write_index:
        print(f"Wrote {write_index()}")
    else:
        m = load_manifest()
        for eid, e in m["experiments"].items():
            print(f"{eid}: {e['title']}")
            for s in e["sources"]:
                print(f"  {s['experiment']}  [{', '.join(s['models'])}]")


if __name__ == "__main__":
    main()
