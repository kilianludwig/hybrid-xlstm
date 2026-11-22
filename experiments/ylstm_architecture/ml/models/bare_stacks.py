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

from xlstm_torch.mlstm import Layer as _MLSTMLayer
from xlstm_torch.slstm import Layer as _SLSTMLayer


class _BareCellStack(BaseSequenceModel):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fgate_bias_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.layers = nn.ModuleList([
            self._make_layer(
                in_features=input_dim if i == 0 else hidden_dim,
                out_features=hidden_dim,
            )
            for i in range(num_layers)
        ])
        self.stack_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        if fgate_bias_init is not None:
            for layer in self.layers:
                layer.cell.reset_parameters(fgate_bias_init=float(fgate_bias_init))
        self.output_dim = hidden_dim

    def _make_layer(self, in_features: int, out_features: int) -> nn.Module:
        raise NotImplementedError

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Any] = None,
        **_: Any,
    ) -> Tuple[Tensor, None]:
        del state
        x = inputs
        for layer, norm in zip(self.layers, self.stack_norms):
            x = layer(x)
            x = norm(x)
        return x, None


class SLSTMBareSequenceModel(_BareCellStack):

    model_name = "slstm_bare"

    def _make_layer(self, in_features: int, out_features: int) -> nn.Module:
        return _SLSTMLayer(
            in_features=in_features,
            out_features=out_features,
            return_state=False,
        )


class MLSTMBareSequenceModel(_BareCellStack):

    model_name = "mlstm_bare"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        hid_features: Optional[int] = None,
        fgate_bias_init: Optional[float] = None,
    ) -> None:
        self._hid_features = hid_features
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            fgate_bias_init=fgate_bias_init,
        )

    def _make_layer(self, in_features: int, out_features: int) -> nn.Module:
        return _MLSTMLayer(
            in_features=in_features,
            out_features=out_features,
            hid_features=self._hid_features,
            return_state=False,
        )
