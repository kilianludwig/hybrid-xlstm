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

from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
from torch import nn

from .base import BaseSequenceModel, Tensor

try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig,
    )
    from xlstm.components.init import bias_linspace_init_
    XLSTM_AVAILABLE = True
    XLSTM_LOAD_ERROR = None
except (ImportError, OSError, RuntimeError) as _e:
    xLSTMBlockStack = None
    xLSTMBlockStackConfig = None
    mLSTMBlockConfig = None
    mLSTMLayerConfig = None
    sLSTMBlockConfig = None
    sLSTMLayerConfig = None
    FeedForwardConfig = None
    bias_linspace_init_ = None
    XLSTM_AVAILABLE = False
    XLSTM_LOAD_ERROR = _e


class XLSTMSequenceModel(BaseSequenceModel):

    model_name = "xlstm"
    _cuda_backend_failed = False

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        num_blocks: int = 4,
        context_length: int = 512,
        slstm_at: Optional[List[int]] = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        add_post_blocks_norm: bool = True,
        bias_init: str = "powerlaw_blockdependent",
        fgate_bias_range: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        
        if not XLSTM_AVAILABLE:
            raise ImportError(
                "xlstm is not installed. Install with: pip install xlstm\n"
                "Repository: https://github.com/NX-AI/xlstm"
            )
        
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.context_length = context_length

        self.input_proj = nn.Linear(input_dim, embedding_dim)
        
        if slstm_at is None:
            slstm_at = list(range(1, num_blocks, 2))
        
        self._slstm_indices = set(slstm_at)
        
        mlstm_config = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                num_heads=num_heads,
            )
        )
        
        def _make_slstm_config(backend: str) -> Optional[sLSTMBlockConfig]:
            if not slstm_at:
                return None
            return sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_heads,
                    backend=backend,
                    bias_init=bias_init,
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=4.0 / 3.0,
                ),
            )

        def _make_stack_config(slstm_cfg: Optional[sLSTMBlockConfig]) -> xLSTMBlockStackConfig:
            return xLSTMBlockStackConfig(
                mlstm_block=mlstm_config,
                slstm_block=slstm_cfg,
                num_blocks=num_blocks,
                embedding_dim=embedding_dim,
                context_length=context_length,
                add_post_blocks_norm=add_post_blocks_norm,
                dropout=dropout,
                slstm_at=slstm_at,
            )

        use_cuda = (
            slstm_at
            and torch.cuda.is_available()
            and not XLSTMSequenceModel._cuda_backend_failed
        )

        if use_cuda:
            try:
                config = _make_stack_config(_make_slstm_config("cuda"))
                self.xlstm = xLSTMBlockStack(config)
            except (RuntimeError, ImportError) as exc:
                XLSTMSequenceModel._cuda_backend_failed = True
                warnings.warn(
                    f"sLSTM CUDA backend failed ({exc}); "
                    "falling back to vanilla (pure PyTorch) backend"
                )
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                config = _make_stack_config(_make_slstm_config("vanilla"))
                self.xlstm = xLSTMBlockStack(config)
        else:
            config = _make_stack_config(_make_slstm_config("vanilla"))
            self.xlstm = xLSTMBlockStack(config)
        
        if fgate_bias_range is not None:
            self._override_mlstm_fgate_bias(fgate_bias_range)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_dim = embedding_dim

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Tensor] = None,
        **_: object,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        del state
        x = self.input_proj(inputs)
        x = self.xlstm(x)
        x = self.dropout_layer(x)
        
        return x, None

    def default_training_config(self) -> Dict[str, Any]:
        return {
            "optimizer": "adamw",
            "learning_rate": 8e-4,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_pct": 0.08,
        }
    
    def get_block_config_summary(self) -> str:
        blocks = []
        for i in range(self.num_blocks):
            block_type = "sLSTM" if i in self._slstm_indices else "mLSTM"
            blocks.append(f"Block {i}: {block_type}")
        return "\n".join(blocks)

    def _override_mlstm_fgate_bias(self, fgate_bias_range: List[float]) -> None:
        start, end = fgate_bias_range
        for name, param in self.xlstm.named_parameters():
            if "mlstm_cell.fgate.bias" in name:
                bias_linspace_init_(param, start=start, end=end)


def _adamw_cosine(lr: float, warmup_pct: float = 0.05) -> Dict[str, Any]:
    return {
        "optimizer": "adamw",
        "learning_rate": lr,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "warmup_pct": warmup_pct,
    }


class XLSTMSLSTMOnly(XLSTMSequenceModel):
    model_name = "slstm"

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        num_blocks: int = 2,
        context_length: int = 512,
        num_heads: int = 4,
        dropout: float = 0.0,
        add_post_blocks_norm: bool = True,
        bias_init: str = "powerlaw_blockdependent",
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            context_length=context_length,
            slstm_at=list(range(num_blocks)),
            num_heads=num_heads,
            dropout=dropout,
            add_post_blocks_norm=add_post_blocks_norm,
            bias_init=bias_init,
        )

    def default_training_config(self) -> Dict[str, Any]:
        return _adamw_cosine(1.5e-3)


class XLSTMMLSTMOnly(XLSTMSequenceModel):
    model_name = "mlstm"

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        num_blocks: int = 2,
        context_length: int = 512,
        num_heads: int = 4,
        dropout: float = 0.0,
        add_post_blocks_norm: bool = True,
        fgate_bias_range: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            context_length=context_length,
            slstm_at=[],
            num_heads=num_heads,
            dropout=dropout,
            add_post_blocks_norm=add_post_blocks_norm,
            fgate_bias_range=fgate_bias_range,
        )

    def default_training_config(self) -> Dict[str, Any]:
        return _adamw_cosine(5e-4, warmup_pct=0.08)


class XLSTM_1_1(XLSTMSequenceModel):
    model_name = "xlstm"

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        num_blocks: int = 2,
        context_length: int = 512,
        num_heads: int = 4,
        dropout: float = 0.0,
        add_post_blocks_norm: bool = True,
        bias_init: str = "powerlaw_blockdependent",
        fgate_bias_range: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            context_length=context_length,
            slstm_at=list(range(1, num_blocks, 2)),
            num_heads=num_heads,
            dropout=dropout,
            add_post_blocks_norm=add_post_blocks_norm,
            bias_init=bias_init,
            fgate_bias_range=fgate_bias_range,
        )


class XLSTM_7_1(XLSTMSequenceModel):
    model_name = "xlstm_7_1"

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        num_blocks: int = 8,
        context_length: int = 512,
        num_heads: int = 4,
        dropout: float = 0.0,
        add_post_blocks_norm: bool = True,
        bias_init: str = "powerlaw_blockdependent",
        fgate_bias_range: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            context_length=context_length,
            slstm_at=[num_blocks - 1],
            num_heads=num_heads,
            dropout=dropout,
            add_post_blocks_norm=add_post_blocks_norm,
            bias_init=bias_init,
            fgate_bias_range=fgate_bias_range,
        )
