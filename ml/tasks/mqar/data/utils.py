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

import os 
import hashlib
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import numpy as np

import torch 
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig, DataSegmentConfig

@dataclass
class DataSegment:
    inputs: torch.Tensor
    labels: torch.Tensor
    slices: Dict[str, any] = None

    def __len__(self):
        assert len(self.inputs) == len(self.labels) 
        return len(self.inputs)

    @classmethod
    def from_config(
        cls, 
        config: DataSegmentConfig,
        cache_dir: str = None,
        force_cache: bool = False,
        seed: int = 123
    ):
        def _get_cache_path(config: DataSegmentConfig):
            if cache_dir is None:
                return None
            config_hash = hashlib.md5(
                json.dumps({**config.model_dump(), "_seed": seed}, sort_keys=True).encode()
            ).hexdigest()

            return os.path.join(
                cache_dir,
                f"data_{config_hash}.pt",
            )
        
        if cache_dir is not None:
            try:
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
            except:
                print(f"Could not create cache directory {cache_dir}")
                cache_dir = None
        cache_path = _get_cache_path(config)
        if cache_dir is not None and os.path.exists(cache_path) and not force_cache:
            print(f"Loading data from on-disk cache at {cache_path}...") 
            MAX_RETRIES = 10
            for _ in range(MAX_RETRIES):
                try:
                    data = cls(**torch.load(cache_path))
                    break
                except RuntimeError as e:
                    print(e)
        else:
            print(f"Generating dataset...") 
            data: DataSegment = config.build(seed=seed)

            if cache_dir is not None:
                print(f"Saving dataset to on-disk cache at {cache_path}...") 
                torch.save(asdict(data), cache_path)
        return data


def prepare_data(config: DataConfig) -> Tuple[DataLoader, DataLoader]:  
    if isinstance(config.batch_size, int):
        train_batch_size, test_batch_size = (config.batch_size, config.batch_size)
    else:
        train_batch_size, test_batch_size = config.batch_size
    
    MAX_SEED = 2 ** 32
    np.random.seed(config.seed)
    train_seeds = np.random.randint(0, MAX_SEED // 2, size=len(config.train_configs))
    test_seeds = np.random.randint(MAX_SEED // 2, MAX_SEED, size=len(config.test_configs))
    factory_kwargs = {"cache_dir": config.cache_dir, "force_cache": config.force_cache}
    train_segments = _SyntheticDataset([
        DataSegment.from_config(segment_config, seed=int(seed), **factory_kwargs)
        for segment_config, seed in zip(config.train_configs, train_seeds)
    ], batch_size=train_batch_size)
    test_segments = _SyntheticDataset([
        DataSegment.from_config(segment_config, seed=int(seed), **factory_kwargs)
        for segment_config, seed in zip(config.test_configs, test_seeds)
    ], batch_size=test_batch_size)

    return (
        DataLoader(ds, batch_size=None, num_workers=0,  shuffle=False)
        for ds in [train_segments, test_segments]
    )


class _SyntheticDataset(Dataset):
    def __init__(self, segments: List[DataSegment], batch_size: int):
        self.segments = segments
        self.batch_size = batch_size        
        self.batches = [
            (segment_idx, batch_start)
            for segment_idx, segment in enumerate(self.segments)
            for batch_start in range(0, len(segment), self.batch_size)
        ]

    def __getitem__(self, batch_idx: int):
        segment_idx, batch_start = self.batches[batch_idx]
        segment = self.segments[segment_idx]
        slc = slice(batch_start, batch_start + self.batch_size)
        
        slices = [segment.slices if segment.slices is not None else {}] * self.batch_size
        return segment.inputs[slc], segment.labels[slc], slices      

    def __len__(self):
        return len(self.batches)

