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

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

_GLOBAL_FALLBACK: Dict[str, Any] = {
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "grad_clip_max_norm": None,
    "betas": None,
    "scheduler": None,
    "warmup_pct": 0.0,
    "warmup_epochs": 0,
    "cosine_eta_min_ratio": 0.1,
    "step_size": 30,
    "gamma": 0.5,
    "patience": 5,
    "factor": 0.5,
}


def _warmup_pct_to_epochs(pct: float, total_epochs: int) -> int:
    if pct <= 0:
        return 0
    return max(2, 2 * round(pct * total_epochs / 2))


def _resolve_warmup_epochs(
    train_cfg: Dict[str, Any],
    model_defaults: Dict[str, Any],
    total_epochs: int,
) -> int:
    explicit = train_cfg.get("warmup_epochs")
    if explicit is None:
        explicit = model_defaults.get("warmup_epochs")
    if explicit is not None:
        return int(explicit)

    pct = float(_resolve("warmup_pct", train_cfg, model_defaults))
    return _warmup_pct_to_epochs(pct, total_epochs)


_OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def _resolve(key: str, train_cfg: Dict[str, Any], model_defaults: Dict[str, Any]) -> Any:
    if key in train_cfg and train_cfg[key] is not None:
        return train_cfg[key]
    if key in model_defaults and model_defaults[key] is not None:
        return model_defaults[key]
    return _GLOBAL_FALLBACK[key]


def create_optimizer(
    model: nn.Module,
    train_cfg: Dict[str, Any],
    model_defaults: Dict[str, Any],
) -> Optimizer:
    opt_name: str = _resolve("optimizer", train_cfg, model_defaults).lower()
    lr: float = float(_resolve("learning_rate", train_cfg, model_defaults))
    wd: float = float(_resolve("weight_decay", train_cfg, model_defaults))
    betas = _resolve("betas", train_cfg, model_defaults)

    opt_cls = _OPTIMIZER_MAP.get(opt_name)
    if opt_cls is None:
        raise ValueError(
            f"Unknown optimizer '{opt_name}'. "
            f"Supported: {list(_OPTIMIZER_MAP.keys())}"
        )

    kwargs: Dict[str, Any] = dict(lr=lr, weight_decay=wd)
    if betas is not None and opt_name in ("adam", "adamw"):
        kwargs["betas"] = tuple(betas)

    return opt_cls(model.parameters(), **kwargs)


def create_scheduler(
    optimizer: Optimizer,
    train_cfg: Dict[str, Any],
    model_defaults: Dict[str, Any],
    epochs: int,
) -> Optional[LRScheduler]:
    sched_name = train_cfg.get("scheduler", model_defaults.get("scheduler"))
    if sched_name is None:
        return None

    sched_name = sched_name.lower()
    warmup_epochs: int = _resolve_warmup_epochs(train_cfg, model_defaults, epochs)

    def _maybe_with_warmup(base_scheduler: LRScheduler) -> LRScheduler:
        if warmup_epochs > 0 and warmup_epochs < epochs:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-2,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup, base_scheduler],
                milestones=[warmup_epochs],
            )
        return base_scheduler

    if sched_name == "cosine":
        lr = float(_resolve("learning_rate", train_cfg, model_defaults))
        eta_min_ratio = float(_resolve("cosine_eta_min_ratio", train_cfg, model_defaults))
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=epochs - max(warmup_epochs, 0),
            eta_min=lr * eta_min_ratio,
        )
        return _maybe_with_warmup(cosine)

    if sched_name == "step_decay":
        step_size: int = int(_resolve("step_size", train_cfg, model_defaults))
        gamma: float = float(_resolve("gamma", train_cfg, model_defaults))
        step = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return _maybe_with_warmup(step)

    if sched_name == "reduce_on_plateau":
        patience: int = int(_resolve("patience", train_cfg, model_defaults))
        factor: float = float(_resolve("factor", train_cfg, model_defaults))
        return ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor,
        )

    raise ValueError(
        f"Unknown scheduler '{sched_name}'. "
        f"Supported: 'cosine', 'step_decay', 'reduce_on_plateau', null."
    )


def get_resolved_optim_info(
    train_cfg: Dict[str, Any],
    model_defaults: Dict[str, Any],
    epochs: int = 100,
) -> Dict[str, Any]:
    betas = _resolve("betas", train_cfg, model_defaults)
    sched = train_cfg.get("scheduler", model_defaults.get("scheduler"))
    grad_clip = _resolve("grad_clip_max_norm", train_cfg, model_defaults)
    warmup_pct = float(_resolve("warmup_pct", train_cfg, model_defaults))
    warmup_epochs = _resolve_warmup_epochs(train_cfg, model_defaults, epochs)
    info: Dict[str, Any] = {
        "optimizer": _resolve("optimizer", train_cfg, model_defaults),
        "learning_rate": float(_resolve("learning_rate", train_cfg, model_defaults)),
        "weight_decay": float(_resolve("weight_decay", train_cfg, model_defaults)),
        "grad_clip_max_norm": float(grad_clip) if grad_clip is not None else None,
        "betas": list(betas) if betas is not None else None,
        "scheduler": sched,
        "warmup_pct": warmup_pct,
        "warmup_epochs": warmup_epochs,
    }
    if sched == "cosine":
        info["cosine_eta_min_ratio"] = float(_resolve("cosine_eta_min_ratio", train_cfg, model_defaults))
    elif sched == "step_decay":
        info["step_size"] = int(_resolve("step_size", train_cfg, model_defaults))
        info["gamma"] = float(_resolve("gamma", train_cfg, model_defaults))
    elif sched == "reduce_on_plateau":
        info["patience"] = int(_resolve("patience", train_cfg, model_defaults))
        info["factor"] = float(_resolve("factor", train_cfg, model_defaults))
    return info
