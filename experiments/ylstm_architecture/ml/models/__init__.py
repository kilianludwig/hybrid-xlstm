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

import os as _os
import sys as _sys
from pathlib import Path as _Path

# Eager mode: compiled kernels fail on pre-Ampere GPUs.
_os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_CELL_PKG_ROOT = _Path(__file__).resolve().parent / "ylstm"
if _CELL_PKG_ROOT.is_dir():
    _p = str(_CELL_PKG_ROOT)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

_Y_CHUNK_SIZES = [0, 1, 2, 4, 8]
MODEL_NAMES = [
    "ylstm",
    "yhlstm",
    "hlstm",
    "xlstm_hb",
    "ylstm_hb",
    "mlstm_hb",
    "slstm_hb",
    "ylstm_mb",
    "ylstm_sb",
    "slstm_bare",
    "mlstm_bare",
]
MODEL_NAMES += [f"ylstm_hb_c{_cs}" for _cs in _Y_CHUNK_SIZES]

EXTRA_MODELS = list(MODEL_NAMES)
