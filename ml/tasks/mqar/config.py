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

from typing import List, Tuple, Union
from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base configuration class using Pydantic."""
    
    def print(self):
        try:
            import rich
            rich.print(self)
        except ImportError:
            print(self)


class DataSegmentConfig(BaseConfig):
    vocab_size: int = 8_192
    num_examples: int = 1_000
    input_seq_len: int = 64

    def build(self, **kwargs):
        raise NotImplementedError()


class DataConfig(BaseConfig):
    train_configs: List[DataSegmentConfig]
    test_configs: List[DataSegmentConfig]

    batch_size: Union[int, Tuple[int, int]] = 32
    seed: int = 123
    cache_dir: str = None
    force_cache: bool = False 

    num_passes: int = 1
