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

"""
PyTorch adapter for JAX-based Chomsky hierarchy tasks.

Wraps JAX GeneralizationTask classes (Delétang et al., ICLR 2023) into
PyTorch datasets.  JAX is only invoked once at dataset creation; all
training uses pure PyTorch tensors.

Task type (SINGLE_LABEL vs SEQUENCE_OUTPUT) is auto-detected from
output_length(), following the original DeepMind approach.

Three Chomsky levels are supported:
  - Regular (Type-3): parity_check, even_pairs, cycle_navigation, modular_arithmetic
  - DCF (Type-2): modular_arithmetic_brackets, reverse_string, solve_equation, stack_manipulation
  - CS (Type-1): binary_addition, binary_multiplication, bucket_sort, compute_sqrt,
                  duplicate_string, missing_duplicate_string, odds_first

Reference: https://arxiv.org/abs/2207.02098
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Type

import jax
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.tasks.base import TaskConfig, TaskType
from ml.tasks.chomsky_hierarchy.task import GeneralizationTask


# Regular Language Tasks (Type-3)

@dataclass
class ParityCheckConfig(TaskConfig):
    pass


@dataclass
class EvenPairsConfig(TaskConfig):
    pass


@dataclass
class CycleNavigationConfig(TaskConfig):
    cycle_length: int = 5


@dataclass
class ModularArithmeticConfig(TaskConfig):
    modulus: int = 5
    operators: Optional[Tuple[str, ...]] = None


# DCF Language Tasks (Type-2)

@dataclass
class ModularArithmeticBracketsConfig(TaskConfig):
    modulus: int = 5
    mult: bool = False


@dataclass
class ReverseStringConfig(TaskConfig):
    vocab_size: int = 2


@dataclass
class SolveEquationConfig(TaskConfig):
    modulus: int = 5


@dataclass
class StackManipulationConfig(TaskConfig):
    pass


# Context-Sensitive Language Tasks (Type-1)

@dataclass
class BinaryAdditionConfig(TaskConfig):
    pass


@dataclass
class BinaryMultiplicationConfig(TaskConfig):
    pass


@dataclass
class BucketSortConfig(TaskConfig):
    vocab_size: int = 5


@dataclass
class ComputeSqrtConfig(TaskConfig):
    pass


@dataclass
class DuplicateStringConfig(TaskConfig):
    vocab_size: int = 2


@dataclass
class MissingDuplicateStringConfig(TaskConfig):
    pass


@dataclass
class OddsFirstConfig(TaskConfig):
    vocab_size: int = 2

def get_task_type(task: GeneralizationTask, length: int = 10) -> str:
    output_len = task.output_length(length)
    return TaskType.from_output_length(output_len)


def is_single_output(task: GeneralizationTask, length: int = 10) -> bool:
    return task.output_length(length) == 1


class ChomskyTaskDataset(Dataset):
    """PyTorch Dataset wrapping a JAX GeneralizationTask.

    Task type (single-label vs sequence-output) is auto-detected from
    ``task.output_length()``, following the original DeepMind approach.
    """

    def __init__(
        self,
        task: GeneralizationTask,
        count: int,
        length: int,
        seed: int = 42,
    ):
        self.task = task
        self.count = count
        self.length = length
        self.seed = seed

        self.output_len = task.output_length(length)
        self.is_single_output = self.output_len == 1
        self.task_type = TaskType.from_output_length(self.output_len)

        self._generate_samples()

    def _generate_samples(self):
        rng = jax.random.PRNGKey(self.seed)
        batch = self.task.sample_batch(rng, self.count, self.length)

        inputs_np = np.array(batch['input'])
        outputs_np = np.array(batch['output'])

        self.inputs = torch.from_numpy(inputs_np).float()

        # One-hot encoded outputs need argmax conversion to class indices
        if self.is_single_output:
            if outputs_np.ndim == 2:
                self.targets = torch.from_numpy(np.argmax(outputs_np, axis=-1)).long()
            else:
                self.targets = torch.from_numpy(outputs_np).long()
        else:
            if outputs_np.ndim == 3:
                self.targets = torch.from_numpy(np.argmax(outputs_np, axis=-1)).long()
            else:
                self.targets = torch.from_numpy(outputs_np).long()

        # Mask post-termination padding positions (variable-length tasks like
        # binary_addition override accuracy_mask() for this; others return all-ones).
        jax_output = batch['output']
        mask_jax = self.task.accuracy_mask(jax_output)
        mask_np = np.array(mask_jax).astype(bool)
        self.targets[~torch.from_numpy(mask_np)] = -100

        self.input_size = self.task.input_size
        self.output_size = self.task.output_size
    
    def __len__(self) -> int:
        return self.count
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[idx],
            "targets": self.targets[idx],
        }


def _create_task_instance(
    task_class: Type[GeneralizationTask],
    config: TaskConfig,
) -> GeneralizationTask:
    """Create a task instance, passing only parameters its ``__init__`` accepts."""
    import inspect

    sig = inspect.signature(task_class.__init__)
    config_dict = config.to_dict()
    init_kwargs = {
        name: config_dict[name]
        for name, _ in sig.parameters.items()
        if name != 'self' and name in config_dict
    }
    return task_class(**init_kwargs)


def _iter_classification_samples(task, config, count) -> Iterator[Tuple[Any, Any]]:
    """Shared generator for classification (single-label) tasks."""
    rng = jax.random.PRNGKey(config.seed)
    batch = task.sample_batch(rng, count, config.sequence_length)
    inputs = np.array(batch['input'])
    outputs = np.argmax(np.array(batch['output']), axis=-1)
    for i in range(count):
        yield inputs[i].tolist(), int(outputs[i])


def _iter_seq2seq_samples(task, config, count) -> Iterator[Tuple[Any, Any]]:
    """Shared generator for sequence-output tasks."""
    rng = jax.random.PRNGKey(config.seed)
    batch = task.sample_batch(rng, count, config.sequence_length)
    inputs = np.argmax(np.array(batch['input']), axis=-1)
    outputs = np.argmax(np.array(batch['output']), axis=-1)
    for i in range(count):
        yield inputs[i].tolist(), outputs[i].tolist()


def generate_parity_check_dataset(
    config: ParityCheckConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .regular import ParityCheck
    return _iter_classification_samples(ParityCheck(), config, count)


def generate_even_pairs_dataset(
    config: EvenPairsConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .regular import EvenPairs
    return _iter_classification_samples(EvenPairs(), config, count)


def generate_cycle_navigation_dataset(
    config: CycleNavigationConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .regular import CycleNavigation
    return _iter_classification_samples(CycleNavigation(), config, count)


def generate_modular_arithmetic_dataset(
    config: ModularArithmeticConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .regular import ModularArithmetic
    return _iter_classification_samples(
        ModularArithmetic(modulus=config.modulus, operators=config.operators),
        config,
        count,
    )


def generate_modular_arithmetic_brackets_dataset(
    config: ModularArithmeticBracketsConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .dcf import ModularArithmeticBrackets
    return _iter_classification_samples(
        ModularArithmeticBrackets(modulus=config.modulus, mult=config.mult),
        config,
        count,
    )


def generate_reverse_string_dataset(
    config: ReverseStringConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .dcf import ReverseString
    return _iter_seq2seq_samples(
        ReverseString(vocab_size=config.vocab_size), config, count,
    )


def generate_solve_equation_dataset(
    config: SolveEquationConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .dcf import SolveEquation
    return _iter_classification_samples(
        SolveEquation(modulus=config.modulus), config, count,
    )


def generate_stack_manipulation_dataset(
    config: StackManipulationConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .dcf import StackManipulation
    return _iter_seq2seq_samples(StackManipulation(), config, count)


def generate_binary_addition_dataset(
    config: BinaryAdditionConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .cs import BinaryAddition
    return _iter_seq2seq_samples(BinaryAddition(), config, count)


def generate_binary_multiplication_dataset(
    config: BinaryMultiplicationConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .cs import BinaryMultiplication
    return _iter_seq2seq_samples(BinaryMultiplication(), config, count)


def generate_bucket_sort_dataset(
    config: BucketSortConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .cs import BucketSort
    return _iter_seq2seq_samples(
        BucketSort(vocab_size=config.vocab_size), config, count,
    )


def generate_compute_sqrt_dataset(
    config: ComputeSqrtConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .cs import ComputeSqrt
    return _iter_seq2seq_samples(ComputeSqrt(), config, count)


def generate_duplicate_string_dataset(
    config: DuplicateStringConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .cs import DuplicateString
    return _iter_seq2seq_samples(
        DuplicateString(vocab_size=config.vocab_size), config, count,
    )


def generate_missing_duplicate_string_dataset(
    config: MissingDuplicateStringConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .cs import MissingDuplicateString
    return _iter_classification_samples(MissingDuplicateString(), config, count)


def generate_odds_first_dataset(
    config: OddsFirstConfig,
    count: int,
) -> Iterator[Tuple[Any, Any]]:
    from .cs import OddsFirst
    return _iter_seq2seq_samples(
        OddsFirst(vocab_size=config.vocab_size), config, count,
    )


def _create_chomsky_dataset(task, count, length, seed=42, *, expect_single=None):
    import warnings
    dataset = ChomskyTaskDataset(task=task, count=count, length=length, seed=seed)
    if expect_single is True and not dataset.is_single_output:
        warnings.warn(f"Task has output_length={dataset.output_len} > 1, "
                       "but create_classification_dataset was called. "
                       "Task type is automatically detected as SEQUENCE_OUTPUT.")
    elif expect_single is False and dataset.is_single_output:
        warnings.warn(f"Task has output_length={dataset.output_len} == 1, "
                       "but create_seq2seq_dataset was called. "
                       "Task type is automatically detected as SINGLE_LABEL.")
    return dataset


def create_classification_dataset(task, count, length, seed=42):
    return _create_chomsky_dataset(task, count, length, seed, expect_single=True)


def create_seq2seq_dataset(task, count, length, seed=42):
    return _create_chomsky_dataset(task, count, length, seed, expect_single=False)


# Task type (SINGLE_LABEL / SEQUENCE_OUTPUT) is derived dynamically from
# task.output_length() -- not stored here.
TASK_REGISTRY = {
    # Regular (Type-3)
    "parity_check":       {"module": "regular", "class": "ParityCheck",       "config": ParityCheckConfig,       "output_classes": 2},
    "even_pairs":         {"module": "regular", "class": "EvenPairs",         "config": EvenPairsConfig,         "output_classes": 2},
    "cycle_navigation":   {"module": "regular", "class": "CycleNavigation",   "config": CycleNavigationConfig,   "output_classes": 5},
    "modular_arithmetic": {"module": "regular", "class": "ModularArithmetic", "config": ModularArithmeticConfig, "output_classes": lambda cfg: cfg.modulus},
    # DCF (Type-2)
    "modular_arithmetic_brackets": {"module": "dcf", "class": "ModularArithmeticBrackets", "config": ModularArithmeticBracketsConfig, "output_classes": lambda cfg: cfg.modulus},
    "reverse_string":              {"module": "dcf", "class": "ReverseString",              "config": ReverseStringConfig,              "output_classes": lambda cfg: cfg.vocab_size},
    "solve_equation":              {"module": "dcf", "class": "SolveEquation",              "config": SolveEquationConfig,              "output_classes": lambda cfg: cfg.modulus},
    "stack_manipulation":          {"module": "dcf", "class": "StackManipulation",          "config": StackManipulationConfig,          "output_classes": 3},
    # Context-Sensitive (Type-1)
    "binary_addition":          {"module": "cs", "class": "BinaryAddition",          "config": BinaryAdditionConfig,          "output_classes": 3},
    "binary_multiplication":    {"module": "cs", "class": "BinaryMultiplication",    "config": BinaryMultiplicationConfig,    "output_classes": 3},
    "bucket_sort":              {"module": "cs", "class": "BucketSort",              "config": BucketSortConfig,              "output_classes": lambda cfg: cfg.vocab_size},
    "compute_sqrt":             {"module": "cs", "class": "ComputeSqrt",             "config": ComputeSqrtConfig,             "output_classes": 2},
    "duplicate_string":         {"module": "cs", "class": "DuplicateString",         "config": DuplicateStringConfig,         "output_classes": lambda cfg: cfg.vocab_size},
    "missing_duplicate_string": {"module": "cs", "class": "MissingDuplicateString", "config": MissingDuplicateStringConfig, "output_classes": 2},
    "odds_first":               {"module": "cs", "class": "OddsFirst",               "config": OddsFirstConfig,               "output_classes": lambda cfg: cfg.vocab_size},
}


def get_task_info(task_name: str) -> Dict[str, Any]:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_name]


def get_task_class(task_name: str) -> Type[GeneralizationTask]:
    info = get_task_info(task_name)
    import importlib
    mod = importlib.import_module(f".{info['module']}", package=__package__)
    return getattr(mod, info["class"])


def create_task_dataset(
    task_name: str,
    count: int,
    config: Optional[TaskConfig] = None,
    **config_overrides,
) -> ChomskyTaskDataset:
    info = get_task_info(task_name)

    if config is None:
        config = info["config"](**config_overrides)
    elif config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    task_class = get_task_class(task_name)
    task = _create_task_instance(task_class, config)

    return ChomskyTaskDataset(
        task=task,
        count=count,
        length=config.sequence_length,
        seed=config.seed,
    )
