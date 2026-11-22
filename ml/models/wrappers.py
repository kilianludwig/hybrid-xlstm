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

from typing import Any, Optional, Tuple

from torch import nn

from .base import BaseSequenceModel, Tensor


class TokenClassifier(BaseSequenceModel):

    def __init__(
        self,
        backbone: BaseSequenceModel,
        num_classes: int,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_cls_token = use_cls_token

        self.output_layer = nn.Linear(backbone.output_dim, num_classes)
        self.output_dim = num_classes

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        kwargs.setdefault("causal", True)
        features, next_state = self.backbone(inputs, state=state, **kwargs)

        if self.use_cls_token:
            features = features[:, 1:, :]

        logits = self.output_layer(features)
        return logits, next_state

