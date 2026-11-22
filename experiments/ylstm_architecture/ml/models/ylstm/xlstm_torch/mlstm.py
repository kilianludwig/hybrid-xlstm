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

from .ops import mlstm_cell, mlstm, mlstm_parallel


class Layer(nn.Module):

    def __init__(self, in_features: int, out_features: int, hid_features: int | None = None,
                 return_state: bool = False, eps: float = 1e-6):
        super().__init__()
        self.cell = Cell(in_features, out_features, hid_features, eps=eps)
        self.return_state = return_state
        self.reset_parameters()

    @property
    def in_features(self):
        return self.cell.in_features

    @property
    def out_features(self):
        return self.cell.out_features

    @property
    def hid_features(self):
        return self.cell.hid_features

    def reset_parameters(self):
        self.cell.reset_parameters()

    def forward(self, x, state=None):
        q, k, v, s_i, s_f = self.cell.project(x)
        if state is None:
            if not self.return_state:
                return mlstm_parallel(q, k, v, s_i, s_f, self.cell.eps)

            state = self.cell.initial_state(x.shape[:1])

        out, state = mlstm(q, k, v, s_i, s_f, state, self.cell.eps)
        return (out, state) if self.return_state else out


class Cell(nn.Module):

    def __init__(self, in_features: int, out_features: int, 
                 hid_features: int | None = None, eps: float = 1e-6):
        super().__init__()
        if hid_features is None:
            hid_features = out_features

        self.q_proj = nn.Linear(in_features, hid_features, bias=False)
        self.k_proj = nn.Linear(in_features, hid_features, bias=False)
        self.v_proj = nn.Linear(in_features, out_features, bias=False)
        self.if_gate = nn.Linear(in_features, 2)
        self.eps = eps

        self.reset_parameters()

    @property
    def in_features(self):
        return self.q_proj.in_features

    @property
    def out_features(self):
        return self.v_proj.out_features

    @property
    def hid_features(self):
        return self.q_proj.out_features

    def reset_parameters(self, fgate_bias_init: float = 3.):
        nn.init.kaiming_uniform_(self.q_proj.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.k_proj.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.v_proj.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.if_gate.weight, nonlinearity="linear")
        nn.init.zeros_(self.if_gate.bias)
        nn.init.constant_(self.if_gate.bias[1], fgate_bias_init)

    def initial_state(self, batch_shape: tuple):
        ref = self.q_proj.weight
        return (
            ref.new_zeros(*batch_shape, self.out_features, self.hid_features),
            ref.new_zeros(*batch_shape, self.hid_features),
            ref.new_full(batch_shape + (1, ), -torch.inf)
        )

    def project(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        s_if = self.if_gate(x)
        return q, k, v, *torch.chunk(s_if, 2, dim=-1)

    def forward(self, x, state=None):
        if state is None:
            state = self.initial_state(x.shape[:1])

        q, k, v, s_i, s_f = self.project(x)
        out, state = mlstm_cell(q, k, v, s_i, s_f, state, self.eps)
        return out, state
