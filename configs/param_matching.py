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

import inspect
from typing import Any, Callable, Dict, Optional, Tuple

from configs.defaults import (
    MODEL_DEFAULTS,
    MQAR_VOCAB_SIZE,
)
from utils.parameters import count_parameters


REFERENCE_INPUT_DIM = 8

MQAR_NUM_KV_PAIRS = 4
MQAR_INPUT_SEQ_LEN = 16

DEFAULT_TOLERANCE = 0.05

FF_RATIO = 4

_MAX_WIDTH = 256

_MATCHED_MODELS = (
    "transformer", "mamba", "slstm", "mlstm", "xlstm",
    "ylstm", "yhlstm", "hlstm",
    "xlstm_hb", "ylstm_hb",
    "mlstm_hb", "slstm_hb", "ylstm_mb", "ylstm_sb",
    "slstm_bare", "mlstm_bare",
)

_WIDTH_KEY: Dict[str, str] = {
    "transformer": "d_model",
    "mamba": "d_model",
    "slstm": "embedding_dim",
    "mlstm": "embedding_dim",
    "xlstm": "embedding_dim",
    "ylstm": "hidden_dim",
    "yhlstm": "hidden_dim",
    "hlstm": "hidden_dim",
    "xlstm_hb": "hidden_dim",
    "ylstm_hb": "hidden_dim",
    "mlstm_hb": "hidden_dim",
    "slstm_hb": "hidden_dim",
    "ylstm_mb": "hidden_dim",
    "ylstm_sb": "hidden_dim",
    "slstm_bare": "hidden_dim",
    "mlstm_bare": "hidden_dim",
}

ParamsFn = Callable[[str, Dict[str, Any]], int]


def _width_step(model: str, base_cfg: Dict[str, Any]) -> int:
    if model == "transformer":
        return int(base_cfg.get("nhead", 4))
    if model in ("slstm", "mlstm", "xlstm"):
        return int(base_cfg.get("num_heads", 4))
    if model in (
        "xlstm_hb", "ylstm_hb",
        "mlstm_hb", "slstm_hb", "ylstm_mb", "ylstm_sb",
    ) or model.startswith("ylstm_hb_c"):
        return int(base_cfg.get("num_heads", 1))
    if model.startswith("hlstm") or model.startswith("yhlstm"):
        return 2 * int(base_cfg.get("num_heads", 1))
    return 2


def _build_backbone(model_name: str, input_dim: int, model_cfg: Dict[str, Any]):
    from ml.models import MODEL_REGISTRY

    cls = MODEL_REGISTRY[model_name]
    params = inspect.signature(cls.__init__).parameters

    kwargs = dict(model_cfg)
    kwargs.pop("name", None)
    default_hidden = kwargs.pop("hidden_dim", 128)

    if "input_dim" in params:
        kwargs.setdefault("input_dim", input_dim)
    if "hidden_dim" in params:
        kwargs.setdefault("hidden_dim", default_hidden)
    if "d_model" in params:
        kwargs.setdefault("d_model", default_hidden)
    if "use_positional_encoding" in params:
        kwargs.setdefault("use_positional_encoding", True)
    if "use_cls_token" in params:
        kwargs.setdefault("use_cls_token", False)
    if "max_seq_len" in params:
        kwargs.setdefault("max_seq_len", 5000)

    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return cls(**kwargs)


def backbone_params(
    model_name: str, model_cfg: Dict[str, Any], input_dim: int = REFERENCE_INPUT_DIM,
) -> int:
    model = _build_backbone(model_name, input_dim, model_cfg)
    return count_parameters(model).trainable


_MQAR_DIMS: Optional[Tuple[int, int]] = None


def _mqar_dims() -> Tuple[int, int]:
    global _MQAR_DIMS
    if _MQAR_DIMS is None:
        from ml.training.pipeline import build_experiment_components

        payload = {
            "model": "lstm",
            "dataset_size": 16,
            "model_kwargs": {k: v for k, v in MODEL_DEFAULTS["lstm"].items()
                             if k != "name"},
            "task_kwargs": {
                "input_seq_len": MQAR_INPUT_SEQ_LEN,
                "vocab_size": MQAR_VOCAB_SIZE,
                "num_kv_pairs": MQAR_NUM_KV_PAIRS,
                "power_a": 0.01,
                "random_non_queries": True,
                "seed": 42,
            },
        }
        _, _, _, metadata, extra = build_experiment_components("mqar", payload)
        _MQAR_DIMS = (metadata["input_dim"], extra["num_classes"])
    return _MQAR_DIMS


def mqar_params(model_name: str, model_cfg: Dict[str, Any]) -> int:
    from ml.models import TokenClassifier

    input_dim, num_classes = _mqar_dims()
    backbone = _build_backbone(model_name, input_dim, model_cfg)
    return count_parameters(
        TokenClassifier(backbone, num_classes=num_classes)
    ).trainable


def _resolve_params_fn(family: str) -> Tuple[ParamsFn, int]:
    if family == "chomsky":
        def pfn(model: str, cfg: Dict[str, Any]) -> int:
            return backbone_params(model, cfg, REFERENCE_INPUT_DIM)
    elif family == "mqar":
        pfn = mqar_params
    else:
        raise ValueError(f"Unknown family {family!r} (use 'chomsky' or 'mqar')")
    ref = pfn("lstm", MODEL_DEFAULTS["lstm"])
    return pfn, ref


def _width_overlay(model: str, width: int) -> Dict[str, Any]:
    key = _WIDTH_KEY[model]
    overlay: Dict[str, Any] = {key: width, "hidden_dim": width}
    if model == "transformer":
        overlay["dim_feedforward"] = FF_RATIO * width
    return overlay


def _search_single_knob(
    model: str, ref_params: int, params_fn: ParamsFn,
) -> Tuple[int, int]:
    base = dict(MODEL_DEFAULTS[model])
    step = _width_step(model, base)

    best: Optional[Tuple[int, int]] = None
    for width in range(step, _MAX_WIDTH + 1, step):
        cfg = dict(base)
        cfg.update(_width_overlay(model, width))
        try:
            params = params_fn(model, cfg)
        except Exception:
            continue
        if best is None or abs(params - ref_params) < abs(best[1] - ref_params):
            best = (width, params)
    if best is None:
        raise RuntimeError(f"Could not build any backbone for model {model!r}")
    return best


def _search_transformer(
    ref_params: int, params_fn: ParamsFn, tolerance: float,
) -> Tuple[int, int, int]:
    base = dict(MODEL_DEFAULTS["transformer"])
    nhead = int(base.get("nhead", 4))

    def params_for(d_model: int, ff: int) -> int:
        cfg = dict(base)
        cfg.update({"d_model": d_model, "hidden_dim": d_model,
                    "dim_feedforward": int(ff)})
        return params_fn("transformer", cfg)

    candidates = []
    for d_model in range(nhead, _MAX_WIDTH + 1, nhead):
        ff0 = FF_RATIO * d_model
        try:
            p0 = params_for(d_model, ff0)
            p1 = params_for(d_model, ff0 + 8)
        except Exception:
            continue
        slope = (p1 - p0) / 8.0
        if slope <= 0:
            continue
        ff = int(round(ff0 + (ref_params - p0) / slope))
        ff = max(d_model, min(8 * d_model, ff))
        try:
            params = params_for(d_model, ff)
        except Exception:
            continue
        candidates.append((d_model, ff, params))

    if not candidates:
        raise RuntimeError("Could not build any Transformer backbone")

    within = [c for c in candidates
              if abs(c[2] - ref_params) / ref_params <= tolerance]
    if within:
        return min(within, key=lambda c: abs(c[1] / c[0] - FF_RATIO))
    return min(candidates, key=lambda c: abs(c[2] - ref_params))


def matched_model_overrides(
    family: str = "chomsky",
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, Dict[str, Any]]:
    import warnings

    params_fn, ref = _resolve_params_fn(family)
    overrides: Dict[str, Dict[str, Any]] = {}
    for model in _MATCHED_MODELS:
        if model == "transformer":
            d_model, ff, params = _search_transformer(ref, params_fn, tolerance)
            overlay: Dict[str, Any] = {
                "d_model": d_model, "hidden_dim": d_model, "dim_feedforward": ff,
            }
            width = d_model
        else:
            width, params = _search_single_knob(model, ref, params_fn)
            overlay = _width_overlay(model, width)

        delta = (params - ref) / ref
        if abs(delta) > tolerance:
            warnings.warn(
                f"[{family}] {model}: best width {width} gives {delta:+.1%} "
                f"vs LSTM (> +/-{tolerance:.0%} tolerance)",
                RuntimeWarning,
                stacklevel=2,
            )
        overrides[model] = overlay
    return overrides


def _print_family_table(family: str, tolerance: float) -> None:
    params_fn, ref = _resolve_params_fn(family)
    overrides = matched_model_overrides(family, tolerance)

    resolved: Dict[str, Dict[str, Any]] = {}
    for model in ("lstm", "gru", *_MATCHED_MODELS):
        cfg = dict(MODEL_DEFAULTS[model])
        cfg.update(overrides.get(model, {}))
        resolved[model] = cfg

    label = "backbone (input_dim=%d)" % REFERENCE_INPUT_DIM if family == "chomsky" \
        else "full model (vocab=%d)" % MQAR_VOCAB_SIZE
    print(f"\n=== {family.upper()} :: {label} ===")
    print(f"LSTM reference params: {ref:,}   tolerance: +/-{tolerance:.0%}\n")

    header = f"{'model':<12}{'width':>8}{'ff':>8}{'params':>12}{'delta%':>9}"
    print(header)
    print("-" * len(header))
    worst = 0.0
    for model in ("lstm", "gru", *_MATCHED_MODELS):
        cfg = resolved[model]
        params = params_fn(model, cfg)
        delta = (params - ref) / ref * 100.0
        if model in _MATCHED_MODELS:
            worst = max(worst, abs(delta))
        key = _WIDTH_KEY.get(model, "hidden_dim")
        ff = cfg.get("dim_feedforward", "-") if model == "transformer" else "-"
        print(f"{model:<12}{cfg.get(key, '-'):>8}{str(ff):>8}{params:>12,}"
              f"{delta:>8.1f}%")

    if worst > tolerance * 100.0:
        print(f"\nWARNING: worst matched |delta| = {worst:.1f}% exceeds tolerance.")
    else:
        print(f"\nOK: all matched models within +/-{tolerance:.0%} "
              f"(worst {worst:.1f}%).")


def verify(tolerance: float = DEFAULT_TOLERANCE) -> None:
    for family in ("chomsky", "mqar"):
        _print_family_table(family, tolerance)


if __name__ == "__main__":
    verify()
