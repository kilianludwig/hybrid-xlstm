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

from typing import Any, List, Optional, Tuple

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


class XLSTMHBBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        *,
        cell_type: str,
        d_proj: Optional[int] = None,
        num_heads: int = 1,
        hid_features: Optional[int] = None,
        ffn: Optional[str] = "swiglu",
        conv_kernel: int = 4,
        fgate_bias_init: Optional[float] = 3.0,
    ) -> None:
        super().__init__()
        if cell_type not in ("slstm", "mlstm"):
            raise ValueError(f"unknown cell type: {cell_type!r}")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")

        d_proj = d_proj if d_proj is not None else d_model
        if d_proj % num_heads or d_model % num_heads:
            raise ValueError(
                "num_heads must divide d_proj and d_model "
                f"(d_proj={d_proj}, d_model={d_model}, num_heads={num_heads})"
            )
        if hid_features is not None and hid_features % num_heads:
            raise ValueError(
                "num_heads must divide hid_features "
                f"(hid_features={hid_features}, num_heads={num_heads})"
            )

        self.cell_type = cell_type
        self.num_heads = num_heads
        self.head_in = d_proj // num_heads

        self.norm_in = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_proj)

        head_out = d_model // num_heads
        if cell_type == "slstm":
            self.conv = (
                _causal_depthwise_conv(d_proj, conv_kernel) if conv_kernel else None
            )
            self.heads = nn.ModuleList(
                _SLSTMLayer(
                    in_features=self.head_in,
                    out_features=head_out,
                    return_state=False,
                )
                for _ in range(num_heads)
            )
            self.out_gate = None
        else:
            self.conv = None
            head_hid = (
                None if hid_features is None else hid_features // num_heads
            )
            self.heads = nn.ModuleList(
                _MLSTMLayer(
                    in_features=self.head_in,
                    out_features=head_out,
                    hid_features=head_hid,
                    return_state=False,
                )
                for _ in range(num_heads)
            )
            self.out_gate = nn.Linear(d_proj, d_model)

        if fgate_bias_init is not None:
            for layer in self.heads:
                if cell_type == "slstm":
                    layer.reset_parameters(fgate_bias_init=float(fgate_bias_init))
                else:
                    layer.cell.reset_parameters(
                        fgate_bias_init=float(fgate_bias_init)
                    )

        self.norm_branch = nn.LayerNorm(d_model)
        self.beta = nn.Parameter(torch.ones(()))
        self.out_proj = nn.Linear(d_model, d_model)

        if ffn == "swiglu":
            self.norm_ffn = nn.LayerNorm(d_model)
            self.ffn = _SwiGLU(d_model)
        elif ffn is None:
            self.ffn = None
        else:
            raise ValueError(f"unknown ffn mode: {ffn!r}")

    def _run_heads(self, x: Tensor) -> Tensor:
        if self.num_heads == 1:
            return self.heads[0](x)
        outputs = [
            head(chunk)
            for head, chunk in zip(
                self.heads, x.split(self.head_in, dim=-1)
            )
        ]
        return torch.cat(outputs, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        projected = self.in_proj(self.norm_in(x))

        if self.conv is not None:
            seq_len = projected.shape[1]
            convolved = self.conv(projected.transpose(1, 2))
            projected = torch.nn.functional.silu(
                convolved[..., :seq_len].transpose(1, 2)
            )

        out = self._run_heads(projected)
        if self.out_gate is not None:
            out = out * torch.nn.functional.silu(self.out_gate(projected))
        out = self.norm_branch(out) * self.beta

        x = residual + self.out_proj(out)
        if self.ffn is not None:
            x = x + self.ffn(self.norm_ffn(x))
        return x


class XLSTMHBSequenceModel(BaseSequenceModel):

    model_name = "xlstm_hb"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        slstm_at: Optional[List[int]] = None,
        d_proj: Optional[int] = None,
        num_heads: int = 1,
        hid_features: Optional[int] = None,
        ffn: Optional[str] = "swiglu",
        conv_kernel: int = 4,
        fgate_bias_init: Optional[float] = 3.0,
    ) -> None:
        super().__init__()
        if slstm_at is None:
            slstm_at = list(range(1, num_layers, 2))
        slstm_idx = set(int(i) for i in slstm_at)

        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            XLSTMHBBlock(
                d_model=hidden_dim,
                cell_type="slstm" if i in slstm_idx else "mlstm",
                d_proj=d_proj,
                num_heads=num_heads,
                hid_features=hid_features,
                ffn=ffn,
                conv_kernel=conv_kernel,
                fgate_bias_init=fgate_bias_init,
            )
            for i in range(num_layers)
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


class MLSTMHBSequenceModel(XLSTMHBSequenceModel):

    model_name = "mlstm_hb"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        d_proj: Optional[int] = None,
        num_heads: int = 1,
        hid_features: Optional[int] = None,
        ffn: Optional[str] = "swiglu",
        conv_kernel: int = 4,
        fgate_bias_init: Optional[float] = 3.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            slstm_at=[],
            d_proj=d_proj,
            num_heads=num_heads,
            hid_features=hid_features,
            ffn=ffn,
            conv_kernel=conv_kernel,
            fgate_bias_init=fgate_bias_init,
        )


class SLSTMHBSequenceModel(XLSTMHBSequenceModel):

    model_name = "slstm_hb"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        d_proj: Optional[int] = None,
        num_heads: int = 1,
        hid_features: Optional[int] = None,
        ffn: Optional[str] = "swiglu",
        conv_kernel: int = 4,
        fgate_bias_init: Optional[float] = 3.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            slstm_at=list(range(num_layers)),
            d_proj=d_proj,
            num_heads=num_heads,
            hid_features=hid_features,
            ffn=ffn,
            conv_kernel=conv_kernel,
            fgate_bias_init=fgate_bias_init,
        )
