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

import logging
import random
from typing import Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def input_key(row: torch.Tensor) -> bytes:
    row = row.detach()
    if row.ndim >= 1 and row.shape[-1] > 1 and row.is_floating_point():
        ids = row.argmax(dim=-1)
    else:
        ids = row
    return ids.to(dtype=torch.int32).cpu().contiguous().numpy().tobytes()


def pool_keys(dataset) -> Set[bytes]:
    inputs = dataset.inputs
    return {input_key(inputs[i]) for i in range(inputs.shape[0])}


class RowDataset(Dataset):

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int):
        return {"inputs": self.inputs[idx], "targets": self.targets[idx]}


def _resample(dataset, count: int, seed: int):
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return type(dataset)(dataset.task, count, dataset.length, seed)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def draw_disjoint_test(
    dataset,
    test_size: int,
    test_seed: int,
    *,
    batch_size: int = 8192,
    max_rounds: int = 200,
) -> RowDataset:
    if test_size <= 0:
        return RowDataset(dataset.inputs[:0], dataset.targets[:0])

    seen = pool_keys(dataset)
    chosen_inputs = []
    chosen_targets = []
    for round_index in range(1, max_rounds + 1):
        seed = test_seed * 1_000_003 + round_index
        extra = _resample(dataset, batch_size, seed)
        added = 0
        for i in range(len(extra)):
            key = input_key(extra.inputs[i])
            if key in seen:
                continue
            seen.add(key)
            chosen_inputs.append(extra.inputs[i])
            chosen_targets.append(extra.targets[i])
            added += 1
            if len(chosen_inputs) >= test_size:
                break
        if added == 0 or len(chosen_inputs) >= test_size:
            break

    if not chosen_inputs:
        return RowDataset(dataset.inputs[:0], dataset.targets[:0])
    return RowDataset(torch.stack(chosen_inputs), torch.stack(chosen_targets))


def three_way_disjoint_split(
    dataset,
    val_size: int,
    test_size: int,
    seed: int,
    *,
    train_target: int = 0,
) -> Tuple[RowDataset, RowDataset, RowDataset]:
    inputs = dataset.inputs
    targets = dataset.targets
    n = inputs.shape[0]

    key_to_idx: dict[bytes, int] = {}
    for i in range(n):
        k = input_key(inputs[i])
        if k not in key_to_idx:
            key_to_idx[k] = i
    unique_indices = list(key_to_idx.values())
    n_unique = len(unique_indices)

    val_actual = val_size
    test_actual = test_size
    if n_unique < val_size + test_size:
        holdout = val_size + test_size
        val_actual = max(1, round(val_size / holdout * n_unique * 0.25))
        test_actual = max(1, round(test_size / holdout * n_unique * 0.25))
        if val_actual + test_actual >= n_unique:
            val_actual = max(1, n_unique // 4)
            test_actual = max(1, n_unique // 4)
        log.warning(
            "Input support (%d) < requested val+test (%d+%d). "
            "Reduced to %d val, %d test, %d train keys.",
            n_unique, val_size, test_size,
            val_actual, test_actual, n_unique - val_actual - test_actual,
        )
    train_keys_count = n_unique - val_actual - test_actual

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_unique, generator=rng).tolist()

    train_pool = [unique_indices[perm[i]] for i in range(train_keys_count)]
    val_indices = [unique_indices[perm[train_keys_count + i]] for i in range(val_actual)]
    test_indices = [unique_indices[perm[train_keys_count + val_actual + i]] for i in range(test_actual)]

    val_ds = RowDataset(inputs[val_indices], targets[val_indices])
    test_ds = RowDataset(inputs[test_indices], targets[test_indices])

    if train_target <= 0:
        train_target = n - val_actual - test_actual
    if train_target <= 0:
        train_target = len(train_pool)
    if len(train_pool) == 0:
        empty = torch.empty(0, *inputs.shape[1:], dtype=inputs.dtype)
        empty_t = torch.empty(0, *targets.shape[1:], dtype=targets.dtype)
        return RowDataset(empty, empty_t), val_ds, test_ds

    full_cycles = train_target // len(train_pool)
    remainder = train_target % len(train_pool)
    train_indices = train_pool * full_cycles + train_pool[:remainder]
    train_ds = RowDataset(inputs[train_indices], targets[train_indices])

    return train_ds, val_ds, test_ds
