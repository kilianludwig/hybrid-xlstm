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

from typing import Any, Dict

ALLOWED_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
MIN_DIM = 8


CHOMSKY_SEQUENCE_LENGTH = 32

MQAR_INPUT_SEQ_LEN = 32

MQAR_VOCAB_SIZE = 256
MQAR_NUM_KV_PAIRS = 8
MQAR_POWER_A = 0.01
MQAR_RANDOM_NON_QUERIES = True


TRAINING_DEFAULTS = {
    "epochs": 256,
    "batch_size": 128,
    "dataset_size": 100000,
    "val_split": 0.10, 
    "device": "auto",
    "log_every": 20,
    "early_stop_at_accuracy": 0.95,
    "seed": 42,
}

OUTPUT_DEFAULTS = {
    "dir": "./experiments",
    "save_checkpoint": True,
    "save_metrics": True,
    "checkpoint_save_last": False,
    "checkpoint_save_best": False, 
    "checkpoint_interval": None,
    "plots": {"enabled": True, "format": "pdf", "dpi": 300},
}

WANDB_DEFAULTS = {
    "enabled": False,
    "project": "hybrid-sequence-models",
}

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "lstm": {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "bidirectional": False,
        "forget_gate_bias_init": 3.0,
    },
    "gru": {
        "hidden_dim": 74,
        "num_layers": 2,
        "dropout": 0.0,
        "bidirectional": False,
    },
    "transformer": {
        "hidden_dim": 64,
        "d_model": 64,
        "num_layers": 2,
        "nhead": 4,
        "dropout": 0.0,
        "dim_feedforward": 256,
    },
    # One defaults block: mamba picks mamba-ssm or mambapy at runtime.
    "mamba": {
        "hidden_dim": 64,
        "d_model": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    },
    "mamba_py": {
        "hidden_dim": 64,
        "d_model": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    },
    "mamba_ssm": {
        "hidden_dim": 64,
        "d_model": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    },
    "slstm": {
        "hidden_dim": 64,
        "embedding_dim": 64,
        "num_blocks": 2,
        "dropout": 0.0,
        "context_length": 512,
        "num_heads": 4,
        "add_post_blocks_norm": True,
        "bias_init": "powerlaw_blockdependent",
    },
    "mlstm": {
        "hidden_dim": 64,
        "embedding_dim": 64,
        "num_blocks": 2,
        "dropout": 0.0,
        "context_length": 512,
        "num_heads": 4,
        "add_post_blocks_norm": True,
        "fgate_bias_range": None,
    },
    "xlstm": {
        "hidden_dim": 64,
        "embedding_dim": 64,
        "num_blocks": 2,
        "dropout": 0.0,
        "context_length": 512,
        "num_heads": 4,
        "add_post_blocks_norm": True,
        "bias_init": "powerlaw_blockdependent", 
        "fgate_bias_range": None, 
    },
    "ylstm": {
        "hidden_dim": 64,
        "num_layers": 2,
        "chunk_size": 1,
        "hid_features": None,
        "fgate_bias_init": None,
        "norm": "feedback",
        "stack_norm": True,
        "feedback_init_std": 0.0,
    },
    "slstm_bare": {
        "hidden_dim": 64,
        "num_layers": 2,
        "fgate_bias_init": None,
    },
    "mlstm_bare": {
        "hidden_dim": 64,
        "num_layers": 2,
        "hid_features": None,
        "fgate_bias_init": None,
    },
    "yhlstm": {
        "hidden_dim": 64,
        "num_layers": 2,
        "split_ratio": 0.5,
        "num_heads": 1,
        "fusion": "gate",
        "ffn": "swiglu",
        "conv_kernel": 4,
        "chunk_size": 1,
        "hid_features": None,
        "fgate_bias_init": 3.0,
        "norm": "feedback",
        "feedback_init_std": 0.0,
    },
    "hlstm": {
        "hidden_dim": 64,
        "num_layers": 2,
        "split_ratio": 0.5,
        "num_heads": 1,
        "fusion": "gate",
        "ffn": "swiglu",
        "conv_kernel": 4,
        "hid_features": None,
        "fgate_bias_init": 3.0,
    },
    "xlstm_hb": {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 1,
        "ffn": "swiglu",
        "conv_kernel": 4,
        "hid_features": None,
        "fgate_bias_init": 3.0,
    },
    "ylstm_hb": {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 1,
        "chunk_size": 1,
        "ffn": "swiglu",
        "hid_features": None,
        "fgate_bias_init": 3.0,
        "norm": "feedback",
        "feedback_init_std": 0.0,
    },
    "mlstm_hb": {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 1,
        "ffn": "swiglu",
        "conv_kernel": 4,
        "hid_features": None,
        "fgate_bias_init": 3.0,
    },
    "slstm_hb": {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 1,
        "ffn": "swiglu",
        "conv_kernel": 4,
        "hid_features": None,
        "fgate_bias_init": 3.0,
    },
    "ylstm_mb": {
        "hidden_dim": 64,
        "num_layers": 2,
        "proj_factor": 2.0,
        "chunk_size": 1,
        "conv_kernel": 4,
        "hid_features": None,
        "fgate_bias_init": 3.0,
        "norm": "feedback",
        "feedback_init_std": 0.0,
    },
    "ylstm_sb": {
        "hidden_dim": 64,
        "num_layers": 2,
        "ffn_proj_factor": 1.3333333333333333,
        "chunk_size": 1,
        "conv_kernel": 4,
        "hid_features": None,
        "fgate_bias_init": 3.0,
        "norm": "feedback",
        "feedback_init_std": 0.0,
    },
}


MODEL_OPTIM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "lstm": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_max_norm": 1.0,

        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "gru": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "transformer": {
        "optimizer": "adamw",
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "grad_clip_max_norm": 1.0,
        "betas": [0.9, 0.98],
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "mamba": {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "mamba_py": {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "mamba_ssm": {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "slstm": {
        "optimizer": "adamw",
        "learning_rate": 1.5e-3,
        "weight_decay": 0.01,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "mlstm": {
        "optimizer": "adamw",
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "xlstm": {
        "optimizer": "adamw",
        "learning_rate": 8e-4,
        "weight_decay": 0.01,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "ylstm": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "yhlstm": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "hlstm": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "xlstm_hb": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
    "ylstm_hb": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_max_norm": 1.0,
        "scheduler": None,
        "warmup_pct": None,
        "cosine_eta_min_ratio": None,
    },
}

for _vname in (
    "mlstm_hb", "slstm_hb", "ylstm_mb", "ylstm_sb",
    "slstm_bare", "mlstm_bare",
):
    MODEL_OPTIM_DEFAULTS[_vname] = dict(MODEL_OPTIM_DEFAULTS["ylstm_hb"])

_Y_CHUNK_SIZES = [0, 1, 2, 4, 8]
for _cs in _Y_CHUNK_SIZES:
    _vname = f"ylstm_hb_c{_cs}"
    _arch = dict(MODEL_DEFAULTS["ylstm_hb"])
    _arch["chunk_size"] = _cs
    MODEL_DEFAULTS[_vname] = _arch
    MODEL_OPTIM_DEFAULTS[_vname] = dict(MODEL_OPTIM_DEFAULTS["ylstm_hb"])

