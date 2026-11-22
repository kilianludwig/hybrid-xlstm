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

from __future__ import annotations

import torch
from torch import nn

from xlstm_torch.ops import slstm_cell


class Layer(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 return_state: bool = False, chunk_size: int = 1):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, 4 * out_features)
        self.recurrent = nn.Parameter(
            torch.empty(4 * out_features, out_features)
        )
        self.return_state = return_state
        self.chunk_size = chunk_size
        self.reset_parameters()

    @property
    def in_features(self):
        return self.gate_proj.in_features

    @property
    def out_features(self):
        return self.recurrent.shape[-1]

    def reset_parameters(self, fgate_bias_init: float = 3.):
        nn.init.zeros_(self.recurrent)
        nn.init.kaiming_uniform_(self.gate_proj.weight, nonlinearity="linear")
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
            fgate_bias = self.gate_proj.bias[self.out_features:2*self.out_features]
            nn.init.constant_(fgate_bias, fgate_bias_init)

    def initial_state(self, batch_shape: tuple):
        state = self.recurrent.new_zeros(
            (4, *batch_shape, self.out_features)
        )
        state[-1] = -torch.inf
        return state

    def forward(self, x, state=None):
        if state is None:
            state = self.initial_state((x.shape[0], self.chunk_size))

        s = self.gate_proj(x)
        ys = []
        for chunk in torch.split(s, self.chunk_size, dim=1):
            if chunk.shape[1] < state.shape[2]:
                state = state[:, :, :chunk.shape[1]]

            y, state = slstm_cell(chunk, self.recurrent, state)
            ys.append(y)

        out = torch.cat(ys, dim=1)
        return (out, state) if self.return_state else out
