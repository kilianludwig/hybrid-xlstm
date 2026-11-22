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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from configs.registry import CHOMSKY_BY_HIERARCHY


@dataclass(frozen=True)
class TaskScalingMeta:

    chomsky_level: str
    output_vocab_size: int
    special_output_tokens: int
    srand_formula: Callable[[Optional[Any]], float]
    has_accuracy_mask: bool = False
    output_length_fn: Callable[[int], int] = field(default=lambda: (lambda seq_len: 1))
    output_length_from_config: Optional[Callable[[Any], Optional[int]]] = None

    @property
    def effective_answer_classes(self) -> int:
        return self.output_vocab_size - self.special_output_tokens

    def compute_srand(
        self,
        task_config: Optional[Any] = None,
        *,
        output_length: int = 1,
    ) -> float:
        per_token = self.srand_formula(task_config)
        if output_length <= 1:
            return per_token
        return per_token ** output_length

    def get_output_length(
        self,
        sequence_length: int = 32,
        task_config: Optional[Any] = None,
    ) -> int:
        if self.output_length_from_config is not None and task_config is not None:
            from_config = self.output_length_from_config(task_config)
            if from_config is not None:
                return from_config
        return self.output_length_fn(sequence_length)


def _level_for_task(task_name: str) -> str:
    for level, tasks in CHOMSKY_BY_HIERARCHY.items():
        if task_name in tasks:
            return level
    raise ValueError(f"Task '{task_name}' not found in CHOMSKY_BY_HIERARCHY")


def _fixed_srand(value: float) -> Callable[[Optional[Any]], float]:
    def _formula(_config: Optional[Any] = None) -> float:
        return value
    return _formula


def _modulus_srand(default_modulus: int = 5) -> Callable[[Optional[Any]], float]:
    def _formula(config: Optional[Any] = None) -> float:
        modulus = default_modulus
        if config is not None and hasattr(config, "modulus"):
            modulus = config.modulus
        return 1.0 / modulus
    return _formula


def _config_int(name: str) -> Callable[[Any], Optional[int]]:
    def _read(config: Any) -> Optional[int]:
        if isinstance(config, dict):
            value = config.get(name)
        else:
            value = getattr(config, name, None)
        return int(value) if value is not None else None
    return _read


def _vocab_srand(default_vocab: int = 2) -> Callable[[Optional[Any]], float]:
    def _formula(config: Optional[Any] = None) -> float:
        vocab = default_vocab
        if config is not None:
            if hasattr(config, "vocab_size"):
                vocab = config.vocab_size
            elif isinstance(config, dict) and "vocab_size" in config:
                vocab = config["vocab_size"]
        return 1.0 / vocab
    return _formula


TASK_SCALING_REGISTRY: Dict[str, TaskScalingMeta] = {
    "parity_check": TaskScalingMeta(
        chomsky_level="regular",
        output_vocab_size=2,
        special_output_tokens=0,
        srand_formula=_fixed_srand(0.5),
        output_length_fn=lambda _: 1,
    ),
    "even_pairs": TaskScalingMeta(
        chomsky_level="regular",
        output_vocab_size=2,
        special_output_tokens=0,
        srand_formula=_fixed_srand(0.5),
        output_length_fn=lambda _: 1,
    ),
    "cycle_navigation": TaskScalingMeta(
        chomsky_level="regular",
        output_vocab_size=5,
        special_output_tokens=0,
        srand_formula=_fixed_srand(0.2),
        output_length_fn=lambda _: 1,
    ),
    "modular_arithmetic": TaskScalingMeta(
        chomsky_level="regular",
        output_vocab_size=5,
        special_output_tokens=0,
        srand_formula=_modulus_srand(5),
        output_length_fn=lambda _: 1,
    ),
    "modular_arithmetic_brackets": TaskScalingMeta(
        chomsky_level="dcf",
        output_vocab_size=5,
        special_output_tokens=0,
        srand_formula=_modulus_srand(5),
        output_length_fn=lambda _: 1,
    ),
    "reverse_string": TaskScalingMeta(
        chomsky_level="dcf",
        output_vocab_size=2,
        special_output_tokens=0,
        srand_formula=_vocab_srand(2),
        output_length_fn=lambda n: n,
    ),
    "solve_equation": TaskScalingMeta(
        chomsky_level="dcf",
        output_vocab_size=5,
        special_output_tokens=0,
        srand_formula=_modulus_srand(5),
        output_length_fn=lambda _: 1,
    ),
    "stack_manipulation": TaskScalingMeta(
        chomsky_level="dcf",
        output_vocab_size=3,
        special_output_tokens=1,
        srand_formula=_fixed_srand(0.5),
        has_accuracy_mask=True,
        output_length_fn=lambda n: n + 1,
    ),
    "binary_addition": TaskScalingMeta(
        chomsky_level="cs",
        output_vocab_size=3,
        special_output_tokens=1,
        srand_formula=_fixed_srand(0.5),
        has_accuracy_mask=True,
        output_length_fn=lambda n: n + 1,
    ),
    "binary_multiplication": TaskScalingMeta(
        chomsky_level="cs",
        output_vocab_size=3,
        special_output_tokens=1,
        srand_formula=_fixed_srand(0.5),
        has_accuracy_mask=True,
        output_length_fn=lambda n: n,
    ),
    "bucket_sort": TaskScalingMeta(
        chomsky_level="cs",
        output_vocab_size=5,
        special_output_tokens=0,
        srand_formula=_vocab_srand(5),
        output_length_fn=lambda n: n,
    ),
    "compute_sqrt": TaskScalingMeta(
        chomsky_level="cs",
        output_vocab_size=2,
        special_output_tokens=0,
        srand_formula=_fixed_srand(0.5),
        output_length_fn=lambda n: math.ceil(n / 2),
    ),
    "duplicate_string": TaskScalingMeta(
        chomsky_level="cs",
        output_vocab_size=2,
        special_output_tokens=0,
        srand_formula=_vocab_srand(2),
        output_length_fn=lambda n: 2 * n,
    ),
    "missing_duplicate_string": TaskScalingMeta(
        chomsky_level="cs",
        output_vocab_size=2,
        special_output_tokens=0,
        srand_formula=_fixed_srand(0.5),
        output_length_fn=lambda _: 1,
    ),
    "odds_first": TaskScalingMeta(
        chomsky_level="cs",
        output_vocab_size=2,
        special_output_tokens=0,
        srand_formula=_vocab_srand(2),
        output_length_fn=lambda n: n,
    ),
    "mqar": TaskScalingMeta(
        chomsky_level="mqar",
        output_vocab_size=256,
        special_output_tokens=0,
        srand_formula=_vocab_srand(256),
        output_length_fn=lambda _: 8,
        output_length_from_config=_config_int("num_kv_pairs"),
    ),
}


def get_scaling_meta(task_name: str) -> TaskScalingMeta:
    if task_name not in TASK_SCALING_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Available: {sorted(TASK_SCALING_REGISTRY)}"
        )
    return TASK_SCALING_REGISTRY[task_name]


def get_chomsky_level(task_name: str) -> str:
    return get_scaling_meta(task_name).chomsky_level
