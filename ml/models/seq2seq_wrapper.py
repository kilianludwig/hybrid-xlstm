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

import torch
from torch import nn

from .base import BaseSequenceModel, Tensor


class AutoregressiveWrapper(BaseSequenceModel):

    model_name = "autoregressive"

    def __init__(
        self,
        backbone: BaseSequenceModel,
        num_classes: int,
        max_output_len: int,
        input_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.max_output_len = max_output_len
        self.input_dim = input_dim

        self.output_head = nn.Linear(backbone.output_dim, num_classes)
        self.output_embedding = nn.Embedding(num_classes, input_dim)
        self.start_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        self.output_dim = num_classes


    def _train_forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        **kwargs: Any,
    ) -> Tuple[Tensor, None]:
        batch_size = inputs.size(0)

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        output_len = targets.size(1)

        safe_targets = targets.clamp(min=0)
        target_embedded = self.output_embedding(safe_targets)

        start = self.start_token.expand(batch_size, 1, -1)
        if output_len > 1:
            shifted = torch.cat([start, target_embedded[:, :-1, :]], dim=1)
        else:
            shifted = start

        full_input = torch.cat([inputs, shifted], dim=1)

        full_output, _ = self.backbone(full_input, state=None, causal=True, **kwargs)

        output_hidden = full_output[:, inputs.size(1):, :]
        logits = self.output_head(output_hidden)

        return logits, None


    @torch.no_grad()
    def _generate(
        self,
        inputs: Tensor,
        output_len: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, None]:
        batch_size = inputs.size(0)
        output_len = output_len or self.max_output_len

        start = self.start_token.expand(batch_size, 1, -1)
        full_seq = torch.cat([inputs, start], dim=1)

        all_logits = []
        for _ in range(output_len):
            features, _ = self.backbone(full_seq, state=None, causal=True, **kwargs)
            logits_t = self.output_head(features[:, -1:, :])
            all_logits.append(logits_t)

            predicted = logits_t.argmax(dim=-1).squeeze(-1)
            next_emb = self.output_embedding(predicted).unsqueeze(1)
            full_seq = torch.cat([full_seq, next_emb], dim=1)

        logits = torch.cat(all_logits, dim=1)
        return logits, None


    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.training and targets is not None:
            return self._train_forward(inputs, targets, **kwargs)
        return self._generate(inputs, **kwargs)

