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

from enum import Enum
from typing import Any, Dict, List, Optional

from evaluation.task_metadata import (
    TASK_SCALING_REGISTRY,
    get_scaling_meta,
)


def compute_srand(
    task_name: str,
    task_config: Optional[Any] = None,
    *,
    output_length: int = 1,
) -> float:
    meta = get_scaling_meta(task_name)
    return meta.compute_srand(task_config, output_length=output_length)


def compute_scaled_accuracy(
    raw_accuracy: float,
    srand: float,
    *,
    clamp: bool = True,
) -> float:
    if srand >= 1.0:
        return 0.0
    scaled = (raw_accuracy - srand) / (1.0 - srand)
    if clamp:
        scaled = max(0.0, min(1.0, scaled))
    return scaled


class WeightPolicy(Enum):

    UNIFORM_TASK = "uniform_task"
    UNIFORM_LEVEL = "uniform_level"
    CUSTOM = "custom"


def compute_task_weights(
    tasks: List[str],
    policy: WeightPolicy = WeightPolicy.UNIFORM_LEVEL,
    *,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if policy == WeightPolicy.UNIFORM_TASK:
        return {t: 1.0 for t in tasks}

    if policy == WeightPolicy.CUSTOM:
        if custom_weights is None:
            raise ValueError("custom_weights required for CUSTOM policy")
        missing = set(tasks) - set(custom_weights)
        if missing:
            raise ValueError(f"Missing custom weights for: {sorted(missing)}")
        return {t: custom_weights[t] for t in tasks}

    if policy == WeightPolicy.UNIFORM_LEVEL:
        level_tasks: Dict[str, List[str]] = {}
        for t in tasks:
            meta = get_scaling_meta(t)
            level_tasks.setdefault(meta.chomsky_level, []).append(t)

        num_levels = len(level_tasks)
        if num_levels == 0:
            return {}

        per_level_weight = 1.0 / num_levels
        weights: Dict[str, float] = {}
        for _level, ltasks in level_tasks.items():
            per_task = per_level_weight / len(ltasks)
            for t in ltasks:
                weights[t] = per_task
        return weights

    raise ValueError(f"Unknown weight policy: {policy}")


def compute_weighted_accuracy(
    scaled_accuracies: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    common = set(scaled_accuracies) & set(weights)
    if not common:
        return 0.0
    total_w = sum(weights[t] for t in common)
    if total_w == 0.0:
        return 0.0
    return sum(weights[t] * scaled_accuracies[t] for t in common) / total_w
