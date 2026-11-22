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

from ylstm.ylstm import yLSTM as _yLSTM


class _LayerNorm(nn.Module):

    def __init__(self, ndim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim))
        self.ndim = ndim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.layer_norm(
            x, (self.ndim,), 1.0 + self.weight, None, self.eps
        )


class _CausalConv1d(nn.Module):

    def __init__(self, feature_dim: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = (
            nn.Conv1d(
                feature_dim,
                feature_dim,
                kernel_size=kernel_size,
                padding=self.pad,
                groups=feature_dim,
            )
            if kernel_size > 0
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.conv is None:
            return x
        y = self.conv(x.transpose(2, 1))
        return y[:, :, : -self.pad].transpose(2, 1)


class _GatedFeedForward(nn.Module):

    def __init__(self, dim: int, proj_factor: float = 4.0 / 3.0) -> None:
        super().__init__()
        hidden = max(1, int(round(dim * proj_factor)))
        self.hidden = hidden
        self.proj_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.proj_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.proj_up(x).split(self.hidden, dim=-1)
        return self.proj_down(torch.nn.functional.gelu(gate) * up)


class _YLSTMCellMixin:

    def _make_cell(
        self,
        in_features: int,
        out_features: int,
        *,
        hid_features: Optional[int],
        chunk_size: int,
        norm: str,
        feedback_init_std: float,
        fgate_bias_init: Optional[float],
    ) -> _yLSTM:
        cell = _yLSTM(
            in_features=in_features,
            out_features=out_features,
            hid_features=hid_features,
            chunk_size=max(1, chunk_size),
            norm=norm,
            feedback_init_std=feedback_init_std,
        )
        if fgate_bias_init is not None:
            cell.reset_parameters(fgate_bias_init=float(fgate_bias_init))
        return cell

    def _bind_chunk_size(self, seq_len: int) -> None:
        cs = seq_len if self.chunk_size == 0 else self.chunk_size
        if self.cell.chunk_size != cs:
            self.cell.chunk_size = cs


class YLSTMMBlock(nn.Module, _YLSTMCellMixin):

    def __init__(
        self,
        d_model: int,
        *,
        proj_factor: float = 2.0,
        chunk_size: int = 1,
        hid_features: Optional[int] = None,
        conv_kernel: int = 4,
        fgate_bias_init: Optional[float] = 3.0,
        norm: str = "feedback",
        feedback_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        inner = max(1, int(round(proj_factor * d_model)))
        self.inner = inner
        self.chunk_size = int(chunk_size)

        self.norm_in = _LayerNorm(d_model)
        self.proj_up = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv = _CausalConv1d(inner, conv_kernel)
        self.cell = self._make_cell(
            inner,
            inner,
            hid_features=hid_features,
            chunk_size=self.chunk_size,
            norm=norm,
            feedback_init_std=feedback_init_std,
            fgate_bias_init=fgate_bias_init,
        )
        self.out_norm = _LayerNorm(inner)
        self.learnable_skip = nn.Parameter(torch.ones(inner))
        self.proj_down = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        self._bind_chunk_size(x.shape[1])
        residual = x

        x_cell, z = self.proj_up(self.norm_in(x)).split(self.inner, dim=-1)
        conv_act = torch.nn.functional.silu(self.conv(x_cell))

        out = self.out_norm(self.cell(conv_act))
        out = out + self.learnable_skip * conv_act
        out = out * torch.nn.functional.silu(z)

        return residual + self.proj_down(out)


class YLSTMSBlock(nn.Module, _YLSTMCellMixin):

    def __init__(
        self,
        d_model: int,
        *,
        chunk_size: int = 1,
        hid_features: Optional[int] = None,
        conv_kernel: int = 4,
        ffn_proj_factor: float = 4.0 / 3.0,
        fgate_bias_init: Optional[float] = 3.0,
        norm: str = "feedback",
        feedback_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)

        self.norm_in = _LayerNorm(d_model)
        self.conv = _CausalConv1d(d_model, conv_kernel)
        self.cell = self._make_cell(
            d_model,
            d_model,
            hid_features=hid_features,
            chunk_size=self.chunk_size,
            norm=norm,
            feedback_init_std=feedback_init_std,
            fgate_bias_init=fgate_bias_init,
        )
        self.group_norm = _LayerNorm(d_model)

        self.norm_ffn = _LayerNorm(d_model)
        self.ffn = _GatedFeedForward(d_model, ffn_proj_factor)

    def forward(self, x: Tensor) -> Tensor:
        self._bind_chunk_size(x.shape[1])

        conv_act = torch.nn.functional.silu(self.conv(self.norm_in(x)))
        x = x + self.group_norm(self.cell(conv_act))
        return x + self.ffn(self.norm_ffn(x))


class _YLSTMVanillaBlockModel(BaseSequenceModel):

    def __init__(self, input_dim: int, hidden_dim: int, blocks) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(blocks)
        self.post_blocks_norm = _LayerNorm(hidden_dim)
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
        return self.post_blocks_norm(x), None


class YLSTMMBSequenceModel(_YLSTMVanillaBlockModel):

    model_name = "ylstm_mb"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        proj_factor: float = 2.0,
        chunk_size: int = 1,
        hid_features: Optional[int] = None,
        conv_kernel: int = 4,
        fgate_bias_init: Optional[float] = 3.0,
        norm: str = "feedback",
        feedback_init_std: float = 0.0,
    ) -> None:
        blocks = [
            YLSTMMBlock(
                hidden_dim,
                proj_factor=proj_factor,
                chunk_size=chunk_size,
                hid_features=hid_features,
                conv_kernel=conv_kernel,
                fgate_bias_init=fgate_bias_init,
                norm=norm,
                feedback_init_std=feedback_init_std,
            )
            for _ in range(num_layers)
        ]
        super().__init__(input_dim, hidden_dim, blocks)


class YLSTMSBSequenceModel(_YLSTMVanillaBlockModel):

    model_name = "ylstm_sb"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        ffn_proj_factor: float = 4.0 / 3.0,
        chunk_size: int = 1,
        hid_features: Optional[int] = None,
        conv_kernel: int = 4,
        fgate_bias_init: Optional[float] = 3.0,
        norm: str = "feedback",
        feedback_init_std: float = 0.0,
    ) -> None:
        blocks = [
            YLSTMSBlock(
                hidden_dim,
                ffn_proj_factor=ffn_proj_factor,
                chunk_size=chunk_size,
                hid_features=hid_features,
                conv_kernel=conv_kernel,
                fgate_bias_init=fgate_bias_init,
                norm=norm,
                feedback_init_std=feedback_init_std,
            )
            for _ in range(num_layers)
        ]
        super().__init__(input_dim, hidden_dim, blocks)
