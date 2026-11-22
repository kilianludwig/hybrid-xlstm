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

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict

from torch import nn


@dataclass
class ParameterStats:

    total: int
    trainable: int
    frozen: int
    memory_mb: float
    breakdown: "OrderedDict[str, int]" = field(default_factory=OrderedDict)


def count_parameters(model: nn.Module) -> ParameterStats:
    total = 0
    trainable = 0
    groups: Dict[str, int] = {}

    for name, param in model.named_parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel

        parts = name.split(".")
        key = f"{parts[0]}.{parts[1]}" if len(parts) > 1 else parts[0]
        groups[key] = groups.get(key, 0) + numel

    breakdown = OrderedDict(sorted(groups.items(), key=lambda kv: -kv[1]))

    return ParameterStats(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        memory_mb=total * 4 / (1024 ** 2),
        breakdown=breakdown,
    )


def format_parameter_summary(stats: ParameterStats) -> str:
    lines = [
        f"    {'Total Parameters:':<25} {stats.total:,}",
        f"    {'Trainable Parameters:':<25} {stats.trainable:,}",
    ]
    if stats.frozen > 0:
        lines.append(f"    {'Frozen Parameters:':<25} {stats.frozen:,}")
    lines.append(f"    {'Est. Memory (f32):':<25} {stats.memory_mb:.2f} MB")

    if stats.breakdown:
        lines.append("")
        lines.append("    Breakdown by Component:")
        max_name_len = max(len(name) for name in stats.breakdown)
        for name, count in stats.breakdown.items():
            pct = 100.0 * count / stats.total if stats.total > 0 else 0.0
            lines.append(
                f"      {name:<{max_name_len}}  {count:>10,}  ({pct:5.1f}%)"
            )

    return "\n".join(lines)
