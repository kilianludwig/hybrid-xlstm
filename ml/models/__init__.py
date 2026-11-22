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

from typing import Optional

from .base import BaseSequenceModel
from .gru import GRUSequenceModel
from .lstm import LSTMSequenceModel
from .transformer import TransformerSequenceModel
from .wrappers import TokenClassifier
from .seq2seq_wrapper import AutoregressiveWrapper


try:
    from .mamba import (
        MambaSequenceModel,
        MambaSSMSequenceModel,
        MambaUnifiedSequenceModel,
        MAMBA_AVAILABLE,
        MAMBAPY_AVAILABLE,
        MAMBA_SSM_AVAILABLE,
    )
except ImportError:
    MambaSequenceModel = None  # type: ignore[misc, assignment]
    MambaSSMSequenceModel = None  # type: ignore[misc, assignment]
    MambaUnifiedSequenceModel = None  # type: ignore[misc, assignment]
    MAMBA_AVAILABLE = False
    MAMBAPY_AVAILABLE = False
    MAMBA_SSM_AVAILABLE = False

try:
    from .xlstm import (
        XLSTMSequenceModel,
        XLSTMSLSTMOnly,
        XLSTMMLSTMOnly,
        XLSTM_1_1,
        XLSTM_7_1,
        XLSTM_AVAILABLE,
        XLSTM_LOAD_ERROR,
    )
except (ImportError, OSError, RuntimeError) as e:
    XLSTMSequenceModel = None  # type: ignore[misc, assignment]
    XLSTMSLSTMOnly = None  # type: ignore[misc, assignment]
    XLSTMMLSTMOnly = None  # type: ignore[misc, assignment]
    XLSTM_1_1 = None  # type: ignore[misc, assignment]
    XLSTM_7_1 = None  # type: ignore[misc, assignment]
    XLSTM_AVAILABLE = False
    XLSTM_LOAD_ERROR = e


_CORE_MODEL_CLASSES = [
    LSTMSequenceModel,
    GRUSequenceModel,
    TransformerSequenceModel,
]

_MODEL_CLASSES = list(_CORE_MODEL_CLASSES)

if MAMBAPY_AVAILABLE and MambaSequenceModel is not None:
    _MODEL_CLASSES.append(MambaSequenceModel)

if MAMBA_SSM_AVAILABLE and MambaSSMSequenceModel is not None:
    _MODEL_CLASSES.append(MambaSSMSequenceModel)

# Unified mamba: available whenever *either* backend is installed.  Picks
# mamba-ssm when CUDA + kernels are present, else falls back to mambapy.
if MAMBA_AVAILABLE and MambaUnifiedSequenceModel is not None:
    _MODEL_CLASSES.append(MambaUnifiedSequenceModel)

if XLSTM_AVAILABLE and XLSTMSLSTMOnly is not None:
    _MODEL_CLASSES.append(XLSTMSLSTMOnly)
    _MODEL_CLASSES.append(XLSTMMLSTMOnly)
    _MODEL_CLASSES.append(XLSTM_1_1)

MODEL_REGISTRY = {cls.model_name: cls for cls in _MODEL_CLASSES}


_GROUP_LOADER_SKIP_FILES = frozenset({
    "base.py",
    "wrappers.py",
    "seq2seq_wrapper.py",
})


def _load_group_model_overrides(group: Optional[str] = None) -> dict:
    import importlib.util
    import sys as _sys
    import inspect
    import types

    try:
        from configs.paths import group_models_dir
    except Exception:
        return {}

    try:
        models_dir = group_models_dir(group)
    except Exception:
        return {}

    if not models_dir.is_dir():
        return {}

    group_name = models_dir.parent.parent.name
    pkg_root = "_group_models"
    pkg_full = f"{pkg_root}.{group_name}"

    if pkg_root not in _sys.modules:
        root_pkg = types.ModuleType(pkg_root)
        root_pkg.__path__ = []
        _sys.modules[pkg_root] = root_pkg
    if pkg_full not in _sys.modules:
        group_pkg = types.ModuleType(pkg_full)
        group_pkg.__path__ = [str(models_dir)]
        _sys.modules[pkg_full] = group_pkg

    allowlist: Optional[set] = None
    init_py = models_dir / "__init__.py"
    if init_py.is_file():
        init_mod_name = pkg_full
        spec = importlib.util.spec_from_file_location(
            init_mod_name, init_py,
            submodule_search_locations=[str(models_dir)],
        )
        if spec is not None and spec.loader is not None:
            try:
                init_module = importlib.util.module_from_spec(spec)
                _sys.modules[init_mod_name] = init_module
                spec.loader.exec_module(init_module)
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    f"[ml.models] Failed to load per-group __init__.py {init_py}: {exc}",
                    file=_sys.stderr,
                )
            else:
                declared = getattr(init_module, "MODEL_NAMES", None)
                if declared is not None:
                    try:
                        allowlist = {str(n) for n in declared}
                    except TypeError:
                        allowlist = None

    overrides: dict = {}
    for path in sorted(models_dir.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        if path.name in _GROUP_LOADER_SKIP_FILES:
            continue
        mod_name = f"{pkg_full}.{path.stem}"
        spec = importlib.util.spec_from_file_location(
            mod_name, path, submodule_search_locations=[],
        )
        if spec is None or spec.loader is None:
            continue
        try:
            module = importlib.util.module_from_spec(spec)
            _sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"[ml.models] Failed to load per-group module {path}: {exc}",
                file=_sys.stderr,
            )
            continue
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, "__module__", None) != mod_name:
                continue
            if inspect.isabstract(obj):
                continue
            model_name = getattr(obj, "model_name", None)
            if not (isinstance(model_name, str) and model_name):
                continue
            if allowlist is not None and model_name not in allowlist:
                continue
            overrides[model_name] = obj
    return overrides


_GROUP_OVERRIDES = _load_group_model_overrides()
if _GROUP_OVERRIDES:
    MODEL_REGISTRY.update(_GROUP_OVERRIDES)


def refresh_group_overrides(group: Optional[str] = None) -> list:
    overrides = _load_group_model_overrides(group)
    if not overrides:
        return []
    MODEL_REGISTRY.update(overrides)
    return list(overrides.keys())


def get_available_models() -> list:
    return list(MODEL_REGISTRY.keys())


def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        available = get_available_models()
        raise KeyError(
            f"Model '{name}' not found. Available models: {available}"
        )
    return MODEL_REGISTRY[name]


def is_model_available(name: str) -> bool:
    return name in MODEL_REGISTRY


__all__ = [
    "BaseSequenceModel",
    "MODEL_REGISTRY",
    "get_available_models",
    "get_model_class",
    "is_model_available",
    "refresh_group_overrides",
    "GRUSequenceModel",
    "LSTMSequenceModel",
    "TransformerSequenceModel",
    "TokenClassifier",
    "AutoregressiveWrapper",
    "MambaSequenceModel",
    "MambaSSMSequenceModel",
    "MambaUnifiedSequenceModel",
    "XLSTMSequenceModel",
    "XLSTMSLSTMOnly",
    "XLSTMMLSTMOnly",
    "XLSTM_1_1",
    "XLSTM_7_1",
    "MAMBA_AVAILABLE",
    "MAMBAPY_AVAILABLE",
    "MAMBA_SSM_AVAILABLE",
    "XLSTM_AVAILABLE",
    "XLSTM_LOAD_ERROR",
]
