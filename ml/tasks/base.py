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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


Batch = Dict[str, np.ndarray]
Sample = Tuple[List[int], Union[int, List[int]]]


class TaskType:
    SINGLE_LABEL = "single_label"
    SEQUENCE_OUTPUT = "sequence_output"

    @classmethod
    def from_output_length(cls, output_length: int) -> str:
        if output_length == 1:
            return cls.SINGLE_LABEL
        return cls.SEQUENCE_OUTPUT


@dataclass
class TaskConfig:
    sequence_length: int = 32
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class SequenceTask(ABC):

    @property
    @abstractmethod
    def input_size(self) -> int: ...

    @property
    @abstractmethod
    def output_size(self) -> int: ...

    def output_length(self, input_length: int) -> int:
        return 1

    @abstractmethod
    def generate_batch(
        self,
        batch_size: int,
        length: int,
        seed: Optional[int] = None,
    ) -> Batch:
        ...

    def is_single_output(self, length: int = 10) -> bool:
        return self.output_length(length) == 1

    def get_task_type(self, length: int = 10) -> str:
        return TaskType.from_output_length(self.output_length(length))

    @property
    def task_type(self) -> str:
        return self.get_task_type(10)

    def pointwise_loss(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        exp_output = np.exp(output - np.max(output, axis=-1, keepdims=True))
        softmax = exp_output / np.sum(exp_output, axis=-1, keepdims=True)
        return -target * np.log(softmax + 1e-10)

    def accuracy(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        pred = np.argmax(output, axis=-1)
        if target.ndim > 1 and target.shape[-1] > 1:
            target = np.argmax(target, axis=-1)
        return (pred == target).astype(np.float32)

    def accuracy_mask(self, target: np.ndarray) -> np.ndarray:
        if target.ndim == 3:
            return np.ones(target.shape[:-1])
        return np.ones(target.shape)

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class UnifiedTaskDataset(Dataset if TORCH_AVAILABLE else object):

    def __init__(
        self,
        task: SequenceTask,
        count: int,
        length: int,
        seed: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for UnifiedTaskDataset")

        self.task = task
        self.count = count
        self.length = length
        self.seed = seed

        self.output_len = task.output_length(length)
        self.is_single_output = self.output_len == 1
        self.task_type = task.get_task_type(length)

        self._generate_samples()

    def _generate_samples(self):
        batch = self.task.generate_batch(self.count, self.length, self.seed)

        inputs = batch['input']
        outputs = batch['output']

        if isinstance(inputs, np.ndarray):
            self.inputs = torch.from_numpy(inputs).float()
        else:
            self.inputs = torch.tensor(inputs, dtype=torch.float32)

        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        else:
            outputs = torch.tensor(outputs)

        if outputs.dtype in (torch.float32, torch.float64):
            if outputs.ndim == 3:
                self.targets = outputs.argmax(dim=-1).long()
            elif outputs.ndim == 2 and self.is_single_output:
                self.targets = outputs.argmax(dim=-1).long()
            else:
                self.targets = outputs.long()
        else:
            self.targets = outputs.long()

        self.input_size = self.task.input_size
        self.output_size = self.task.output_size
    
    def __len__(self) -> int:
        return self.count
    
    def __getitem__(self, idx: int) -> Dict[str, 'torch.Tensor']:
        return {
            "inputs": self.inputs[idx],
            "targets": self.targets[idx],
        }
