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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    
    task: str = "mqar"
    model: str = "lstm"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(**data)


@dataclass
class WandbConfig:
    
    enabled: bool = False
    project: str = "hybrid-sequence-models"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    log_gradients: bool = False
    log_freq: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WandbConfig":
        valid_keys = {
            "enabled", "project", "entity", "run_name", 
            "tags", "notes", "log_gradients", "log_freq"
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

