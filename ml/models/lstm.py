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

from typing import Optional, Tuple

from torch import nn

from .base import BaseSequenceModel, Tensor


class LSTMSequenceModel(BaseSequenceModel):

    model_name = "lstm"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
        forget_gate_bias_init: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

        if forget_gate_bias_init is not None:
            for name, param in self.lstm.named_parameters():
                if "bias_ih" in name:
                    n = param.size(0) // 4
                    param.data[n : 2 * n].fill_(forget_gate_bias_init)

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Tuple[Tensor, Tensor]] = None,
        **_: object,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        outputs, next_state = self.lstm(inputs, state)
        return outputs, next_state
