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

from typing import Optional

import numpy as np
import torch

from ml.tasks.base import SequenceTask, Batch, UnifiedTaskDataset

from .data.multiquery_ar import MQARConfig, multiquery_ar


class _DiscreteOneHotMQARTask(SequenceTask):

    def __init__(self, config):
        self.config = config
        self._vocab_size = config.vocab_size
        self._num_kv_pairs = config.num_kv_pairs
        self._input_seq_len = config.input_seq_len

    @property
    def input_size(self) -> int:
        return self._vocab_size

    @property
    def output_size(self) -> int:
        return self._vocab_size

    def output_length(self, input_length: int) -> int:
        return input_length

    def _call_generator(self, batch_size: int, length: int, seed: int):
        raise NotImplementedError

    def generate_batch(self, batch_size: int, length: int, seed: Optional[int] = None) -> Batch:
        effective_seed = seed if seed is not None else self.config.vocab_size
        data_segment = self._call_generator(batch_size, length, effective_seed)
        token_ids = data_segment.inputs.numpy().astype(np.int64)
        inputs_onehot = np.eye(self._vocab_size, dtype=np.float32)[token_ids]
        outputs = data_segment.labels.numpy().astype(np.int64)
        return {"input": inputs_onehot, "output": outputs}


class MQARTask(_DiscreteOneHotMQARTask):

    def _call_generator(self, batch_size, length, seed):
        return multiquery_ar(
            vocab_size=self._vocab_size,
            num_examples=batch_size,
            input_seq_len=length,
            seed=seed,
            power_a=self.config.power_a,
            num_kv_pairs=self._num_kv_pairs,
            random_non_queries=self.config.random_non_queries,
        )


def _create_discrete_mqar_dataset(
    count,
    config,
    config_cls,
    task_cls,
    seed,
    **overrides,
):
    if config is None:
        config = config_cls(**overrides)
    task = task_cls(config)
    return UnifiedTaskDataset(
        task=task,
        count=count,
        length=config.input_seq_len,
        seed=seed,
    )


def create_mqar_dataset(
    count: int,
    config: Optional[MQARConfig] = None,
    seed: int = 42,
    **config_overrides,
) -> UnifiedTaskDataset:
    return _create_discrete_mqar_dataset(
        count, config, MQARConfig, MQARTask, seed, **config_overrides,
    )


__all__ = [
    "MQARTask",
    "create_mqar_dataset",
]
