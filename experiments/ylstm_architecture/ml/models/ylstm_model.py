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

from ml.models.base import BaseSequenceModel, Tensor

from ylstm.ylstm import yLSTM as _yLSTM


class YLSTMSequenceModel(BaseSequenceModel):

    model_name = "ylstm"

    DEFAULT_CHUNK_SIZE = 1

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        chunk_size: Optional[int] = None,
        hid_features: Optional[int] = None,
        fgate_bias_init: Optional[float] = None,
        norm: str = "feedback",
        stack_norm: bool = True,
        feedback_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if chunk_size is None:
            chunk_size = self.DEFAULT_CHUNK_SIZE
        self.chunk_size = int(chunk_size)
        # ``0`` means "single full-sequence chunk"; the actual length is
        # only known at forward time, so build the cells with a valid
        # placeholder and rebind ``chunk_size`` per forward call.
        layer_chunk = self.chunk_size if self.chunk_size > 0 else 1
        self.layers = nn.ModuleList([
            _yLSTM(
                in_features=input_dim if i == 0 else hidden_dim,
                out_features=hidden_dim,
                hid_features=hid_features,
                chunk_size=layer_chunk,
                norm=norm,
                feedback_init_std=feedback_init_std,
            )
            for i in range(num_layers)
        ])
        self.stack_norms = (
            nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            if stack_norm else None
        )
        if fgate_bias_init is not None:
            for layer in self.layers:
                layer.reset_parameters(fgate_bias_init=float(fgate_bias_init))
        self.output_dim = hidden_dim

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Any] = None,
        **_: Any,
    ) -> Tuple[Tensor, None]:
        del state
        cs = inputs.shape[1] if self.chunk_size == 0 else self.chunk_size
        x = inputs
        for i, layer in enumerate(self.layers):
            if layer.chunk_size != cs:
                layer.chunk_size = cs
            x = layer(x)
            if self.stack_norms is not None:
                x = self.stack_norms[i](x)
        return x, None
