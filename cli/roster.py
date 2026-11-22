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

"""Select training YAMLs for a thesis suite or a setup directory."""

from pathlib import Path
from typing import List, Optional, Set

from cli.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIRMATION_SETUPS = [
    REPO_ROOT / "experiments/ylstm_architecture/20261005_testeval_disjoint_a/setup",
    REPO_ROOT / "experiments/ylstm_architecture/20261005_testeval_disjoint_b/setup",
    REPO_ROOT
    / "experiments/ylstm_architecture/20261117_1606_transformer_mqar_causal_5seed/setup",
]

SUITE_EXPERIMENTS = {
    "headline": "E1_headline",
    "cells": "E1_cell_stacks",
    "cross": "E2_block_cross",
    "chunk": "E3_chunk",
}

SUITES = (*SUITE_EXPERIMENTS, "all")

_SKIP_YAML = {"meta.yaml", "search_space.yaml"}


def _csv(value: Optional[str]) -> Optional[Set[str]]:
    if not value:
        return None
    names = {part.strip() for part in value.split(",") if part.strip()}
    return names or None


def suite_models(suite: str) -> Optional[Set[str]]:
    """Model names for a suite. ``all`` returns None, meaning no model filter."""
    if suite == "all":
        return None
    if suite not in SUITE_EXPERIMENTS:
        raise SystemExit(f"Unknown suite {suite!r}. Choose from: {', '.join(SUITES)}.")
    from reporting.thesis_manifest import models_for

    return set(models_for(SUITE_EXPERIMENTS[suite]))


def run_yamls(setup: Path) -> List[Path]:
    if not setup.is_dir():
        raise SystemExit(f"Setup directory not found: {setup}")
    return sorted(
        path for path in setup.rglob("*.yaml") if path.name not in _SKIP_YAML
    )


def select_configs(
    suite: Optional[str] = None,
    setup: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
) -> List[Path]:
    """Return run YAMLs for a suite, a setup directory, or both.

    A suite alone searches the confirmation roster. ``--setup`` alone searches
    that directory, including learning-rate grids. Both together search the
    given directory and keep only the suite's models.
    """
    if setup:
        candidates = run_yamls(Path(setup))
    else:
        candidates = []
        for directory in CONFIRMATION_SETUPS:
            candidates.extend(run_yamls(directory))

    allowed = suite_models(suite) if suite else None
    model_filter = _csv(model)
    task_filter = _csv(task)
    if model_filter is not None:
        allowed = model_filter if allowed is None else allowed & model_filter

    matched: List[Path] = []
    for path in candidates:
        config = load_config(str(path))
        if not isinstance(config, dict):
            continue
        model_name = (config.get("model") or {}).get("name")
        task_name = (config.get("task") or {}).get("name")
        if allowed is not None and model_name not in allowed:
            continue
        if task_filter is not None and task_name not in task_filter:
            continue
        matched.append(path)
    return matched
