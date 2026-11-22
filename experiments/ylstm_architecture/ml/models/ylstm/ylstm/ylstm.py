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

from xlstm_torch.ops import mlstm

class yLSTM(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 hid_features: int | None = None, chunk_size: int = 1,
                 norm: str = "none", feedback_init_std: float = 0.0):
        super().__init__()
        if hid_features is None:
            hid_features = out_features
        if norm not in ("none", "feedback"):
            raise ValueError(f"norm must be 'none' or 'feedback', got {norm!r}")
        if feedback_init_std < 0:
            raise ValueError(
                f"feedback_init_std must be >= 0, got {feedback_init_std!r}")

        num_qkv_features = 2 * hid_features + out_features
        self.qkv_proj = nn.Linear(in_features, num_qkv_features)
        self.if_proj = nn.Linear(in_features, 2)
        self.qkv_recurrent = nn.Parameter(torch.zeros(num_qkv_features, out_features))
        self.if_recurrent = nn.Parameter(torch.zeros(2, out_features))
        self.chunk_size = chunk_size
        self.norm = norm
        self.feedback_init_std = float(feedback_init_std)
        self.reset_parameters()

    @property
    def in_features(self):
        return self.qkv_proj.in_features

    @property
    def out_features(self):
        return self.if_recurrent.shape[1]

    @property
    def hid_features(self):
        return (self.qkv_proj.out_features - self.out_features) // 2

    def reset_parameters(self, fgate_bias_init: float = 3.):
        nn.init.kaiming_uniform_(self.qkv_proj.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.if_proj.weight, nonlinearity="linear")
        if self.feedback_init_std > 0:
            nn.init.normal_(self.qkv_recurrent, std=self.feedback_init_std)
            nn.init.normal_(self.if_recurrent, std=self.feedback_init_std)
        else:
            nn.init.zeros_(self.qkv_recurrent)
            nn.init.zeros_(self.if_recurrent)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        if self.if_proj.bias is not None:
            nn.init.zeros_(self.if_proj.bias)
            nn.init.constant_(self.if_proj.bias[1], fgate_bias_init)

    def initial_state(self, batch_shape: tuple[int, ...]):
        ref = self.qkv_proj.weight
        return (
            ref.new_zeros(*batch_shape, self.out_features, self.hid_features),
            ref.new_zeros(*batch_shape, self.hid_features),
            ref.new_full(batch_shape + (1, ), -torch.inf),
        )

    def forward(self, x, state=None):
        qkv_split = [self.hid_features, self.hid_features, self.out_features]
        qkv = self.qkv_proj(x)
        s_if = self.if_proj(x)
        if state is None:
            state = self.initial_state(x.shape[:1])

        qkv_chunks = torch.split(qkv, self.chunk_size, dim=-2)
        if_chunks = torch.split(s_if, self.chunk_size, dim=-2)
        h = self.qkv_proj.weight.new_zeros(*x.shape[:1], 1, self.out_features)
        hs = []
        for qkvt, s_ift in zip(qkv_chunks, if_chunks):
            # A shorter final chunk keeps the leading, position-aligned
            # slice of the previous chunk's hidden states (strided feedback).
            if h.shape[-2] > qkvt.shape[-2]:
                h = h[..., :qkvt.shape[-2], :]
            if self.norm == "feedback":
                h = torch.nn.functional.layer_norm(
                    h, h.shape[-1:], None, None, eps=1e-6
                )
            qkvt = qkvt + h @ self.qkv_recurrent.T
            s_ift = s_ift + h @ self.if_recurrent.T

            q, k, v = torch.split(qkvt, qkv_split, dim=-1)
            si, sf = torch.chunk(s_ift, 2, dim=-1)
            h, state = mlstm(q, k, v, si, sf, state)
            hs.append(h)

        return torch.cat(hs, dim=-2)
