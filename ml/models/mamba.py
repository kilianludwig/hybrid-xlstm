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

import os
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .base import BaseSequenceModel, Tensor

MAMBAPY_AVAILABLE = False
try:
    from mambapy.mamba import Mamba as _MambaPy, MambaConfig as _MambaConfig
    MAMBAPY_AVAILABLE = True
except ImportError:
    _MambaPy = None
    _MambaConfig = None

MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm import Mamba as _MambaSSM
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    _MambaSSM = None

MAMBA_AVAILABLE = MAMBAPY_AVAILABLE or MAMBA_SSM_AVAILABLE


class MambaSequenceModel(BaseSequenceModel):

    model_name = "mamba_py"

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not MAMBAPY_AVAILABLE:
            raise ImportError(
                "mambapy is not installed. Install with:\n"
                "  pip install mambapy"
            )

        self.d_model = d_model
        self.num_layers = num_layers
        self.backend = "mambapy"

        self.input_proj = nn.Linear(input_dim, d_model)

        config = _MambaConfig(
            d_model=d_model,
            n_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
            use_cuda=False,
        )
        self.mamba = _MambaPy(config)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_dim = d_model

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Tensor] = None,
        **_: object,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        del state
        x = self.input_proj(inputs)
        x = self.mamba(x)
        x = self.dropout(x)
        return x, None

    def default_training_config(self) -> Dict[str, Any]:
        return _MAMBA_TRAIN_CFG.copy()


_MAMBA_TRAIN_CFG = {
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "scheduler": "cosine",
    "warmup_pct": 0.05,
}


# Use mamba-ssm only when a kernel probe succeeds.
_MAMBA_SSM_MIN_CUDA_CAPABILITY: Tuple[int, int] = (7, 0)

# Optional escape hatch for testing / overrides:
#   MAMBA_BACKEND=ssm   force mamba-ssm (skip probe)
#   MAMBA_BACKEND=py    force mambapy   (skip probe)
_FORCE_BACKEND_ENV = "MAMBA_BACKEND"

_BACKEND_DECISION: Optional[Tuple[bool, str]] = None


def _force_backend_from_env() -> Optional[str]:
    val = os.environ.get(_FORCE_BACKEND_ENV, "").strip().lower()
    if val in ("ssm", "py"):
        return val
    return None


# Inline subprocess body kept as a module-level constant so the spawn
# call stays readable.  The script is intentionally minimal: import,
# build a tiny block, do a 1-step forward, synchronise.  Anything
# longer would risk false negatives from unrelated kernel launches.
_MAMBA_SSM_PROBE_SCRIPT = (
    "import torch\n"
    "from mamba_ssm import Mamba\n"
    "torch.cuda.set_device(0)\n"
    "m = Mamba(d_model=16, d_state=4, d_conv=2, expand=1).to('cuda')\n"
    "y = m(torch.zeros(1, 4, 16, device='cuda'))\n"
    "torch.cuda.synchronize()\n"
    "print('OK')\n"
)


def _probe_mamba_ssm_runtime() -> Tuple[bool, str]:
    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", _MAMBA_SSM_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "probe timed out (>120s)"
    except OSError as e:
        return False, f"probe spawn error: {e}"

    if result.returncode == 0 and "OK" in result.stdout:
        return True, "ok"

    stderr_lines = [ln for ln in (result.stderr or "").splitlines() if ln.strip()]
    for line in reversed(stderr_lines):
        low = line.lower()
        if "no kernel image" in low:
            return False, "wheel has no kernel image for active GPU arch"
        if "cuda" in low and "error" in low:
            return False, line.strip()[:140]
    if stderr_lines:
        return False, stderr_lines[-1].strip()[:140]
    return False, f"probe exited {result.returncode} with no stderr"


def _decide_backend() -> Tuple[bool, str]:
    global _BACKEND_DECISION
    if _BACKEND_DECISION is not None:
        return _BACKEND_DECISION

    if not MAMBA_SSM_AVAILABLE:
        _BACKEND_DECISION = (False, "mamba-ssm not installed")
        return _BACKEND_DECISION

    forced = _force_backend_from_env()
    if forced == "ssm":
        _BACKEND_DECISION = (True, f"forced via {_FORCE_BACKEND_ENV}=ssm")
        return _BACKEND_DECISION
    if forced == "py":
        _BACKEND_DECISION = (False, f"forced via {_FORCE_BACKEND_ENV}=py")
        return _BACKEND_DECISION

    try:
        if not torch.cuda.is_available():
            _BACKEND_DECISION = (False, "no CUDA available")
            return _BACKEND_DECISION
        cap = torch.cuda.get_device_capability()
    except Exception as e:
        _BACKEND_DECISION = (False, f"CUDA capability probe error: {e}")
        return _BACKEND_DECISION

    if cap < _MAMBA_SSM_MIN_CUDA_CAPABILITY:
        _BACKEND_DECISION = (False, f"GPU sm_{cap[0]}{cap[1]} < sm_70")
        return _BACKEND_DECISION

    ok, reason = _probe_mamba_ssm_runtime()
    _BACKEND_DECISION = (ok, reason)
    return _BACKEND_DECISION


def _prefer_ssm_over_py() -> bool:
    use_ssm, _ = _decide_backend()
    return use_ssm


def _resolve_mambapy_backend_label() -> str:
    if not MAMBA_SSM_AVAILABLE:
        return "mambapy"
    _, reason = _decide_backend()
    return f"mambapy (fallback: {reason})"


_MAMBA_FALLBACK_WARNED = False


def _warn_mamba_fallback_once(backend_label: str) -> None:
    global _MAMBA_FALLBACK_WARNED
    if _MAMBA_FALLBACK_WARNED:
        return
    _MAMBA_FALLBACK_WARNED = True
    print(
        f"[mamba] WARNING: using pure-PyTorch backend ({backend_label}). "
        f"Set {_FORCE_BACKEND_ENV}=ssm to override the probe.",
        flush=True,
    )


class MambaSSMSequenceModel(BaseSequenceModel):

    model_name = "mamba_ssm"

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "mamba-ssm is not installed. Install with:\n"
                "  pip install mamba-ssm\n"
                "Requires CUDA, Triton, and causal-conv1d."
            )

        self.d_model = d_model
        self.num_layers = num_layers
        self.backend = "mamba-ssm"

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            _MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_dim = d_model

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Tensor] = None,
        **_: object,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        del state
        x = self.input_proj(inputs)
        for layer in self.layers:
            x = layer(x) + x
        x = self.norm(x)
        x = self.dropout(x)
        return x, None

    def default_training_config(self) -> Dict[str, Any]:
        return _MAMBA_TRAIN_CFG.copy()


class MambaUnifiedSequenceModel(BaseSequenceModel):

    model_name = "mamba"

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.0,
        backend: Optional[str] = None,
        fallback: bool = True,
    ) -> None:
        super().__init__()

        if backend is not None and backend not in ("ssm", "py"):
            raise ValueError(
                f"Invalid mamba backend '{backend}'. "
                "Choose 'ssm', 'py', or omit for automatic detection."
            )

        use_ssm: bool
        if backend == "ssm":
            probe_ok = _prefer_ssm_over_py()
            if probe_ok:
                use_ssm = True
            elif fallback and MAMBAPY_AVAILABLE:
                _, reason = _decide_backend()
                _warn_mamba_fallback_once(f"mambapy (fallback from ssm: {reason})")
                use_ssm = False
            else:
                _, reason = _decide_backend()
                raise ImportError(
                    f"mamba-ssm backend requested (fallback=False) but unavailable: {reason}\n"
                    "Install with:  pip install mamba-ssm\n"
                    "Or set  fallback: true  in the model config."
                )
        elif backend == "py":
            if MAMBAPY_AVAILABLE:
                use_ssm = False
            elif fallback and _prefer_ssm_over_py():
                print(
                    "[mamba] WARNING: mambapy not installed; falling back to mamba-ssm. "
                    "Install mambapy or set  fallback: false  to make this an error.",
                    flush=True,
                )
                use_ssm = True
            else:
                raise ImportError(
                    "mambapy backend requested (fallback=False) but not installed.\n"
                    "Install with:  pip install mambapy\n"
                    "Or set  fallback: true  in the model config."
                )
        else:
            use_ssm = _prefer_ssm_over_py()

        if use_ssm:
            self._impl: BaseSequenceModel = MambaSSMSequenceModel(
                input_dim,
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.backend = "mamba-ssm"
        elif MAMBAPY_AVAILABLE:
            self._impl = MambaSequenceModel(
                input_dim,
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.backend = _resolve_mambapy_backend_label()
            if backend is None:
                _warn_mamba_fallback_once(self.backend)
        else:
            raise ImportError(
                "Neither mamba-ssm nor mambapy is installed. Install one:\n"
                "  pip install mambapy          # pure PyTorch, CPU/MPS/CUDA\n"
                "  pip install mamba-ssm        # CUDA only, faster"
            )

        self.d_model = d_model
        self.num_layers = num_layers
        self.output_dim = self._impl.output_dim

    def forward(
        self,
        inputs: Tensor,
        *,
        state: Optional[Tensor] = None,
        **kwargs: object,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self._impl(inputs, state=state, **kwargs)

    def default_training_config(self) -> Dict[str, Any]:
        return self._impl.default_training_config()
