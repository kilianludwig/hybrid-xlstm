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

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .base import BaseSequenceModel, Tensor


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(
        self, d_model: int, max_len: int = 5000, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerSequenceModel(BaseSequenceModel):

    model_name = "transformer"

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_cls_token: bool = False,
        max_seq_len: int = 5000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.use_cls_token = use_cls_token

        self.input_proj = nn.Linear(input_dim, d_model)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        if use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model, max_len=max_seq_len, dropout=dropout
            )
        else:
            self.pos_encoder = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Tensor] = None,
        **kwargs: object,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        del state
        src_key_padding_mask = kwargs.get("src_key_padding_mask")
        causal = kwargs.get("causal", False)

        B = inputs.size(0)

        embedded = self.input_proj(inputs)

        if self.use_cls_token and self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, 1, self.d_model)
            embedded = torch.cat([cls_tokens, embedded], dim=1)

            if src_key_padding_mask is not None:
                cls_pad = torch.zeros(
                    B, 1, dtype=src_key_padding_mask.dtype, device=embedded.device
                )
                src_key_padding_mask = torch.cat(
                    [cls_pad, src_key_padding_mask], dim=1
                )

        if self.use_positional_encoding and self.pos_encoder is not None:
            embedded = self.pos_encoder(embedded)

        attn_mask = None
        if causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                embedded.size(1), device=embedded.device
            )

        outputs = self.encoder(
            embedded,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        return outputs, None

    def default_training_config(self) -> Dict[str, Any]:
        return {
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "betas": [0.9, 0.98],
            "scheduler": "cosine",
            "warmup_pct": 0.05,
        }

