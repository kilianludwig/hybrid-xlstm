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

import importlib.util as _importlib_util
import sys as _sys
from typing import List, Optional, Sequence, Union

# ``mamba`` prefers mamba-ssm when the kernel probe succeeds, else mambapy.
_BASE_MODELS: List[str] = [
    "gru", "lstm", "mamba", "slstm", "mlstm", "xlstm", "transformer",
]


def _read_group_extra_models(group: str) -> List[str]:
    try:
        from configs.paths import group_models_dir
    except Exception:
        return []
    try:
        models_dir = group_models_dir(group)
    except Exception:
        return []
    init_py = models_dir / "__init__.py"
    if not init_py.is_file():
        return []
    mod_name = f"_group_extras.{group}"
    spec = _importlib_util.spec_from_file_location(
        mod_name, init_py,
        submodule_search_locations=[str(models_dir)],
    )
    if spec is None or spec.loader is None:
        return []
    try:
        mod = _importlib_util.module_from_spec(spec)
        _sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    except Exception:
        return []
    declared = getattr(mod, "EXTRA_MODELS", None)
    if not declared:
        return []
    out: List[str] = []
    for name in declared:
        if isinstance(name, str) and name and name not in out:
            out.append(name)
    return out


def _aggregate_extra_models() -> List[str]:
    try:
        from configs.paths import list_groups
    except Exception:
        return []
    try:
        groups = list_groups()
    except Exception:
        return []
    base = set(_BASE_MODELS)
    extras: List[str] = []
    for group in groups:
        for name in _read_group_extra_models(group):
            if name in base or name in extras:
                continue
            extras.append(name)
    return extras


MODELS: List[str] = list(_BASE_MODELS) + _aggregate_extra_models()

CHOMSKY_BY_HIERARCHY = {
    "regular": [
        "parity_check",
        "even_pairs",
        "cycle_navigation",
        "modular_arithmetic",
    ],
    "dcf": [
        "modular_arithmetic_brackets",
        "reverse_string",
        "solve_equation",
        "stack_manipulation",
    ],
    "cs": [
        "binary_addition",
        "binary_multiplication",
        "bucket_sort",
        "compute_sqrt",
        "duplicate_string",
        "missing_duplicate_string",
        "odds_first",
    ],
}

CHOMSKY_TASKS = [
    task for tasks in CHOMSKY_BY_HIERARCHY.values() for task in tasks
]

MQAR_TASKS = ["mqar"]

ALL_TASKS = CHOMSKY_TASKS + MQAR_TASKS

VALID_CHOMSKY_HIERARCHIES = frozenset(CHOMSKY_BY_HIERARCHY.keys())


def get_chomsky_tasks(
    hierarchies: Optional[Union[str, Sequence[str]]] = None,
) -> List[str]:
    if hierarchies is None or hierarchies == "all":
        return list(CHOMSKY_TASKS)
    if isinstance(hierarchies, str):
        hierarchies = [hierarchies]
    tasks = []
    for h in hierarchies:
        if h in CHOMSKY_BY_HIERARCHY:
            tasks.extend(CHOMSKY_BY_HIERARCHY[h])
    return sorted(set(tasks))
