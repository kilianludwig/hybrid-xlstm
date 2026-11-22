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
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

Tensor = torch.Tensor


class BaseSequenceModel(nn.Module, ABC):

    model_name: str = "base"

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, inputs: Tensor, *, state: Optional[Tensor] = None, **kwargs: Any
    ) -> Tuple[Tensor, Optional[Tensor]]:
        pass

    def default_training_config(self) -> Dict[str, Any]:
        return {
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "scheduler": "cosine",
            "warmup_pct": 0.05,
        }

