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

from ml.models.base import BaseSequenceModel, Tensor

from xlstm_torch.mlstm import Layer as _MLSTMLayer
from xlstm_torch.slstm import Layer as _SLSTMLayer


def _causal_depthwise_conv(channels: int, kernel_size: int) -> nn.Conv1d:
    return nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        groups=channels,
        padding=kernel_size - 1,
    )


class _SwiGLU(nn.Module):

    def __init__(self, dim: int, proj_factor: float = 4.0 / 3.0) -> None:
        super().__init__()
        hidden = max(1, int(round(dim * proj_factor)))
        self.gate = nn.Linear(dim, hidden)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class ParallelHybridBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        *,
        d_proj: Optional[int] = None,
        split_ratio: float = 0.5,
        num_heads: int = 1,
        hid_features: Optional[int] = None,
        fusion: str = "gate",
        ffn: Optional[str] = "swiglu",
        conv_kernel: int = 4,
        fgate_bias_init: Optional[float] = 3.0,
    ) -> None:
        super().__init__()
        if fusion not in ("sum", "mean", "gate"):
            raise ValueError(f"unknown fusion mode: {fusion!r}")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")

        d_proj = d_proj if d_proj is not None else d_model
        d_fuse = d_model

        d_s = int(round(d_proj * split_ratio))
        d_s = max(num_heads, (d_s // num_heads) * num_heads)
        d_m = d_proj - d_s
        if d_m < num_heads or d_s < num_heads:
            raise ValueError(
                f"d_proj={d_proj} too small to split across {num_heads} heads "
                f"per branch (got d_s={d_s}, d_m={d_m})"
            )
        if d_s % num_heads or d_m % num_heads or d_fuse % num_heads:
            raise ValueError(
                "num_heads must divide the sLSTM share, mLSTM share, and "
                f"d_model (d_s={d_s}, d_m={d_m}, d_model={d_fuse}, "
                f"num_heads={num_heads})"
            )

        self.d_s = d_s
        self.d_m = d_m
        self.num_heads = num_heads
        self.fusion = fusion

        self.norm_in = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_proj)

        self.s_conv = (
            _causal_depthwise_conv(d_s, conv_kernel) if conv_kernel else None
        )
        s_head_in = d_s // num_heads
        s_head_out = d_fuse // num_heads
        self.slstm_heads = nn.ModuleList(
            _SLSTMLayer(
                in_features=s_head_in,
                out_features=s_head_out,
                return_state=False,
            )
            for _ in range(num_heads)
        )

        m_head_in = d_m // num_heads
        m_head_out = d_fuse // num_heads
        m_head_hid = None if hid_features is None else hid_features // num_heads
        self.mlstm_heads = nn.ModuleList(
            _MLSTMLayer(
                in_features=m_head_in,
                out_features=m_head_out,
                hid_features=m_head_hid,
                return_state=False,
            )
            for _ in range(num_heads)
        )
        self.m_out_gate = nn.Linear(d_m, d_fuse)

        if fgate_bias_init is not None:
            for layer in self.slstm_heads:
                layer.reset_parameters(fgate_bias_init=float(fgate_bias_init))
            for layer in self.mlstm_heads:
                layer.cell.reset_parameters(fgate_bias_init=float(fgate_bias_init))

        self.norm_s = nn.LayerNorm(d_fuse)
        self.norm_m = nn.LayerNorm(d_fuse)
        self.beta_s = nn.Parameter(torch.ones(()))
        self.beta_m = nn.Parameter(torch.ones(()))

        if fusion == "gate":
            self.fuse_gate = nn.Linear(d_model, d_fuse)
        self.out_proj = nn.Linear(d_fuse, d_model)

        if ffn == "swiglu":
            self.norm_ffn = nn.LayerNorm(d_model)
            self.ffn = _SwiGLU(d_model)
        elif ffn is None:
            self.ffn = None
        else:
            raise ValueError(f"unknown ffn mode: {ffn!r}")

    def _run_branch(self, heads: nn.ModuleList, x: Tensor, head_in: int) -> Tensor:
        if self.num_heads == 1:
            return heads[0](x)
        outs = [head(chunk) for head, chunk in zip(heads, x.split(head_in, dim=-1))]
        return torch.cat(outs, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = self.norm_in(x)
        proj = self.in_proj(h)
        xs, xm = proj[..., : self.d_s], proj[..., self.d_s :]

        if self.s_conv is not None:
            seq_len = xs.shape[1]
            conv = self.s_conv(xs.transpose(1, 2))[..., :seq_len].transpose(1, 2)
            xs = torch.nn.functional.silu(conv)
        out_s = self._run_branch(self.slstm_heads, xs, self.d_s // self.num_heads)
        out_s = self.norm_s(out_s) * self.beta_s

        out_m = self._run_branch(self.mlstm_heads, xm, self.d_m // self.num_heads)
        out_m = out_m * torch.nn.functional.silu(self.m_out_gate(xm))
        out_m = self.norm_m(out_m) * self.beta_m

        if self.fusion == "gate":
            g = torch.sigmoid(self.fuse_gate(h))
            fused = g * out_s + (1.0 - g) * out_m
        elif self.fusion == "mean":
            fused = 0.5 * (out_s + out_m)
        else:
            fused = out_s + out_m

        x = residual + self.out_proj(fused)
        if self.ffn is not None:
            x = x + self.ffn(self.norm_ffn(x))
        return x


class HLSTMSequenceModel(BaseSequenceModel):

    model_name = "hlstm"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        d_proj: Optional[int] = None,
        split_ratio: float = 0.5,
        num_heads: int = 1,
        hid_features: Optional[int] = None,
        fusion: str = "gate",
        ffn: Optional[str] = "swiglu",
        conv_kernel: int = 4,
        fgate_bias_init: Optional[float] = 3.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ParallelHybridBlock(
                d_model=hidden_dim,
                d_proj=d_proj,
                split_ratio=split_ratio,
                num_heads=num_heads,
                hid_features=hid_features,
                fusion=fusion,
                ffn=ffn,
                conv_kernel=conv_kernel,
                fgate_bias_init=fgate_bias_init,
            )
            for _ in range(num_layers)
        )
        self.output_dim = hidden_dim

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Any] = None,
        **_: Any,
    ) -> Tuple[Tensor, None]:
        del state
        x = self.embed(inputs)
        for block in self.blocks:
            x = block(x)
        return x, None
