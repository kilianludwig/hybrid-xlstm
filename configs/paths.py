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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is required for meta inspection
    yaml = None  # type: ignore[assignment]


EXPERIMENTS_DIR = "experiments"
SETUP_SUBDIR = "setup"
RESULTS_SUBDIR = "results"
DISCUSSION_SUBDIR = "discussion"
OUTPUTS_SUBDIR = "outputs"
REPORT_SUBDIR = "report"
GROUP_REPORT_SUBDIR = "group_report"
LEGACY_NAME = "_legacy"
META_FILENAME = "meta.yaml"

GROUP_META_FILENAME = "group.yaml"
GROUP_MODELS_SUBDIR = "ml/models"
DEFAULT_GROUP = "ylstm_architecture"
GROUP_ENV_VAR = "HYBRID_GROUP"


def default_group() -> str:
    env_val = os.environ.get(GROUP_ENV_VAR)
    if env_val:
        return env_val.strip()
    return DEFAULT_GROUP


def _resolve_group(group: Optional[str]) -> str:
    return group if group else default_group()


def is_group_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    if name.startswith(".") or name == LEGACY_NAME:
        return False
    if (path / GROUP_META_FILENAME).is_file():
        return True
    for child in path.iterdir():
        if is_experiment_dir(child):
            return True
    return False


def list_groups(root: Optional[Path] = None) -> List[str]:
    parent = experiments_root() if root is None else Path(root)
    if not parent.is_dir():
        return []
    return sorted(
        entry.name
        for entry in parent.iterdir()
        if is_group_dir(entry)
    )


def resolve_group_for_id(
    config_id: str, root: Optional[Path] = None,
) -> Optional[str]:
    parent = experiments_root() if root is None else Path(root)
    if not parent.is_dir():
        return None
    for group_name in list_groups(parent):
        candidate = parent / group_name / config_id
        if is_experiment_dir(candidate):
            return group_name
    return None


def experiments_root(cwd: Optional[Path] = None) -> Path:
    base = Path(cwd) if cwd is not None else Path.cwd()
    return base / EXPERIMENTS_DIR


def legacy_root(cwd: Optional[Path] = None) -> Path:
    return experiments_root(cwd) / LEGACY_NAME


def group_dir(group: Optional[str] = None, cwd: Optional[Path] = None) -> Path:
    return experiments_root(cwd) / _resolve_group(group)


def group_meta_path(
    group: Optional[str] = None, cwd: Optional[Path] = None,
) -> Path:
    return group_dir(group, cwd) / GROUP_META_FILENAME


def group_report_dir(
    group: Optional[str] = None, cwd: Optional[Path] = None,
) -> Path:
    return group_dir(group, cwd) / GROUP_REPORT_SUBDIR


def group_models_dir(
    group: Optional[str] = None, cwd: Optional[Path] = None,
) -> Path:
    return group_dir(group, cwd) / GROUP_MODELS_SUBDIR


def group_main_path(
    group: Optional[str] = None, cwd: Optional[Path] = None,
) -> Path:
    return group_dir(group, cwd) / "main.py"


def experiment_dir(
    config_id: str,
    group: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> Path:
    if group is None:
        existing = resolve_group_for_id(config_id, root=experiments_root(cwd))
        if existing is not None:
            return experiments_root(cwd) / existing / config_id
    return group_dir(group, cwd) / config_id


def setup_dir(
    config_id: str,
    group: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> Path:
    return experiment_dir(config_id, group, cwd) / SETUP_SUBDIR


def results_dir(
    config_id: str,
    group: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> Path:
    return experiment_dir(config_id, group, cwd) / RESULTS_SUBDIR


def run_outputs_dir(
    config_id: str,
    group: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> Path:
    return results_dir(config_id, group, cwd) / OUTPUTS_SUBDIR


def report_dir(
    config_id: str,
    group: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> Path:
    return results_dir(config_id, group, cwd) / REPORT_SUBDIR


def discussion_dir(
    config_id: str,
    group: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> Path:
    return experiment_dir(config_id, group, cwd) / DISCUSSION_SUBDIR


def is_experiment_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    if name.startswith(".") or name == LEGACY_NAME:
        return False
    if (path / GROUP_META_FILENAME).is_file():
        return False
    return (path / SETUP_SUBDIR).is_dir() or (path / RESULTS_SUBDIR).is_dir()


def _experiment_entries_in(parent: Path) -> List[Path]:
    if not parent.is_dir():
        return []
    return [p for p in parent.iterdir() if is_experiment_dir(p)]


def list_experiments(
    root: Optional[Path] = None,
    group: Optional[str] = None,
) -> List[Tuple[Path, Dict[str, Any]]]:
    if yaml is None:
        return []

    parent = experiments_root() if root is None else Path(root)
    if not parent.is_dir():
        return []

    candidate_dirs: List[Path] = []
    if group is not None:
        candidate_dirs = [parent / group]
    else:
        if any(is_experiment_dir(c) for c in parent.iterdir()):
            candidate_dirs = [parent]
        else:
            candidate_dirs = [parent / g for g in list_groups(parent)]

    result: List[Tuple[Path, Dict[str, Any]]] = []
    for cand in candidate_dirs:
        for entry in _experiment_entries_in(cand):
            meta_path = entry / SETUP_SUBDIR / META_FILENAME
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = yaml.safe_load(f) or {}
                meta.setdefault("id", entry.name)
                meta.setdefault("group", cand.name)
                result.append((entry, meta))
            except (OSError, yaml.YAMLError):
                continue

    result.sort(key=lambda item: item[1].get("id", item[0].name), reverse=True)
    return result


def write_experiment_readmes(experiment_dir: Path) -> None:
    readmes: List[Tuple[str, str]] = [
        (
            SETUP_SUBDIR,
            """\
# setup

Generated YAML configuration files for this experiment.

## Layout

```
setup/
    meta.yaml               -- experiment metadata (id, scale, epochs, ...)
    search_space.yaml       -- HPO axis values (present in _hpo_* experiments only)
    chomsky/                -- Chomsky hierarchy tasks
        cs/                 -- context-sensitive tasks
        dcf/                -- deterministic context-free tasks
        regular/            -- regular language tasks
    mqar/                   -- multi-query associative recall tasks
```

Each YAML file defines one training run (model + task + hyperparameters).
HPO experiments additionally contain variant YAMLs named
``<model>_<task>_<axis><n>.yaml`` for HPO variants.
""",
        ),
        (
            RESULTS_SUBDIR,
            """\
# results

Training outputs and aggregated report for this experiment.

## Layout

```
results/
    outputs/                    -- one directory per completed training run
        <run_name>_<timestamp>/
            config.yaml         -- effective config used for this run
            metrics.json        -- per-epoch train/val accuracy and loss
            run_info.json       -- timing, device, and completion metadata
            checkpoints/        -- saved model weights
            plots/              -- per-run training curves
    report/                     -- aggregated analysis (auto-generated)
        setup.md                -- experiment configuration summary
        report.md               -- model comparison narrative
        report.tex              -- LaTeX version of the report
        summary.csv             -- accuracy table (rows: tasks, cols: models)
        summary.json            -- same data as JSON
        plots/                  -- comparison charts and heatmaps
```

Regenerate the report from the run outputs.
""",
        ),
    ]

    for subdir, content in readmes:
        readme_path = experiment_dir / subdir / "README.md"
        if not readme_path.exists():
            readme_path.write_text(content, encoding="utf-8")


def discover_run_dirs_under_experiments(
    root: Optional[Path] = None,
    group: Optional[str] = None,
) -> List[Path]:
    parent = experiments_root() if root is None else Path(root)
    if not parent.is_dir():
        return []

    if group is not None:
        candidates = [parent / group]
    else:
        if any(is_experiment_dir(c) for c in parent.iterdir()):
            candidates = [parent]
        else:
            candidates = [parent / g for g in list_groups(parent)]

    dirs: List[Path] = []
    for cand in candidates:
        for entry in _experiment_entries_in(cand):
            outputs = entry / RESULTS_SUBDIR / OUTPUTS_SUBDIR
            if outputs.is_dir():
                dirs.append(outputs)
    return sorted(dirs)
