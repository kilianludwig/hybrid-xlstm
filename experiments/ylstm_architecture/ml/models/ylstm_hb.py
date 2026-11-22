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


class _SwiGLU(nn.Module):

    def __init__(self, dim: int, proj_factor: float = 4.0 / 3.0) -> None:
        super().__init__()
        hidden = max(1, int(round(dim * proj_factor)))
        self.gate = nn.Linear(dim, hidden)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class YLSTMHBBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        *,
        d_proj: Optional[int] = None,
        num_heads: int = 1,
        chunk_size: int = 1,
        hid_features: Optional[int] = None,
        ffn: Optional[str] = "swiglu",
        fgate_bias_init: Optional[float] = 3.0,
        norm: str = "feedback",
        feedback_init_std: float = 0.0,
    ) -> None:
        super().__init__()
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

        self.num_heads = num_heads
        self.head_in = d_proj // num_heads
        self.chunk_size = int(chunk_size)

        self.norm_in = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_proj)

        head_out = d_model // num_heads
        head_hid = None if hid_features is None else hid_features // num_heads
        self.heads = nn.ModuleList(
            _yLSTM(
                in_features=self.head_in,
                out_features=head_out,
                hid_features=head_hid,
                chunk_size=max(1, self.chunk_size),
                norm=norm,
                feedback_init_std=feedback_init_std,
            )
            for _ in range(num_heads)
        )
        self.out_gate = nn.Linear(d_proj, d_model)

        if fgate_bias_init is not None:
            for layer in self.heads:
                layer.reset_parameters(fgate_bias_init=float(fgate_bias_init))

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

    def _bind_chunk_size(self, seq_len: int) -> None:
        cs = seq_len if self.chunk_size == 0 else self.chunk_size
        for layer in self.heads:
            if layer.chunk_size != cs:
                layer.chunk_size = cs

    def _run_heads(self, x: Tensor) -> Tensor:
        if self.num_heads == 1:
            return self.heads[0](x)
        outputs = [
            head(chunk)
            for head, chunk in zip(self.heads, x.split(self.head_in, dim=-1))
        ]
        return torch.cat(outputs, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        self._bind_chunk_size(x.shape[1])
        residual = x
        projected = self.in_proj(self.norm_in(x))

        out = self._run_heads(projected)
        out = out * torch.nn.functional.silu(self.out_gate(projected))
        out = self.norm_branch(out) * self.beta

        x = residual + self.out_proj(out)
        if self.ffn is not None:
            x = x + self.ffn(self.norm_ffn(x))
        return x


class YLSTMHBSequenceModel(BaseSequenceModel):

    model_name = "ylstm_hb"

    DEFAULT_CHUNK_SIZE = 1

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        d_proj: Optional[int] = None,
        num_heads: int = 1,
        chunk_size: Optional[int] = None,
        hid_features: Optional[int] = None,
        ffn: Optional[str] = "swiglu",
        fgate_bias_init: Optional[float] = 3.0,
        norm: str = "feedback",
        feedback_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if chunk_size is None:
            chunk_size = self.DEFAULT_CHUNK_SIZE
        self.chunk_size = int(chunk_size)

        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            YLSTMHBBlock(
                d_model=hidden_dim,
                d_proj=d_proj,
                num_heads=num_heads,
                chunk_size=self.chunk_size,
                hid_features=hid_features,
                ffn=ffn,
                fgate_bias_init=fgate_bias_init,
                norm=norm,
                feedback_init_std=feedback_init_std,
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


# Each chunk size is its own model name. ``__module__`` must be this module.
_CHUNK_SIZES = [0, 1, 2, 4, 8]
for _cs in _CHUNK_SIZES:
    _name = f"ylstm_hb_c{_cs}"
    _cls_name = f"YLSTMHB_C{_cs}"
    globals()[_cls_name] = type(
        _cls_name,
        (YLSTMHBSequenceModel,),
        {"model_name": _name, "DEFAULT_CHUNK_SIZE": _cs, "__module__": __name__},
    )
