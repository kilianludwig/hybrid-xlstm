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

import os
import platform
from typing import Any, Dict, List, Optional, Tuple


def _safe_import_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
    except Exception:
        return None
    return getattr(mod, "__version__", None) or "unknown"


def gpu_signature() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "reason": f"torch not importable: {exc}"}

    try:
        if not torch.cuda.is_available():
            return {
                "available": False,
                "reason": "torch.cuda.is_available() == False",
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
    except Exception as exc:
        return {"available": False, "reason": f"is_available() raised: {exc}"}

    out: Dict[str, Any] = {
        "available": True,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        out["device_count"] = int(torch.cuda.device_count())
    except Exception:
        pass

    try:
        out["index"] = int(torch.cuda.current_device())
    except Exception:
        out["index"] = 0

    try:
        out["name"] = torch.cuda.get_device_name(out.get("index", 0))
    except Exception:
        pass

    try:
        cap = torch.cuda.get_device_capability(out.get("index", 0))
        out["capability_tuple"] = [int(cap[0]), int(cap[1])]
        out["capability"] = f"sm_{cap[0]}{cap[1]}"
        out["arch"] = _arch_name_from_capability(cap)
    except Exception:
        pass

    try:
        free, total = torch.cuda.mem_get_info()
        out["memory_free_mb"] = round(free / (1024 ** 2), 1)
        out["memory_total_mb"] = round(total / (1024 ** 2), 1)
    except Exception:
        pass

    return out


def _arch_name_from_capability(cap: Tuple[int, int]) -> Optional[str]:
    major, minor = cap
    table = [
        ((6, 0), "pascal"),
        ((6, 1), "pascal"),
        ((6, 2), "pascal"),
        ((7, 0), "volta"),
        ((7, 2), "volta"),
        ((7, 5), "turing"),
        ((8, 0), "ampere"),
        ((8, 6), "ampere"),
        ((8, 7), "ampere"),
        ((8, 9), "ada"),
        ((9, 0), "hopper"),
        ((10, 0), "blackwell"),
        ((12, 0), "blackwell"),
    ]
    for (m, n), name in table:
        if (major, minor) == (m, n):
            return name
    return f"sm_{major}{minor}"


def collect_session_env() -> Dict[str, Any]:
    from utils.env import describe_python_env

    env: Dict[str, Any] = {
        "python": describe_python_env(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }

    torch_info: Dict[str, Any] = {}
    try:
        import torch
        torch_info["version"] = torch.__version__
        torch_info["cuda"] = torch.version.cuda
        try:
            torch_info["cudnn"] = torch.backends.cudnn.version()
        except Exception:
            pass
    except Exception as exc:
        torch_info["error"] = f"torch not importable: {exc}"
    env["torch"] = torch_info

    env["gpu"] = gpu_signature()

    env["packages"] = {
        "mamba_ssm": _safe_import_version("mamba_ssm"),
        "causal_conv1d": _safe_import_version("causal_conv1d"),
        "mambapy": _safe_import_version("mambapy"),
        "xlstm": _safe_import_version("xlstm"),
        "triton": _safe_import_version("triton"),
        "numpy": _safe_import_version("numpy"),
    }

    env["env_vars"] = {
        k: os.environ.get(k, "")
        for k in (
            "CUDA_VISIBLE_DEVICES",
            "CUDA_HOME",
            "TORCH_CUDA_ARCH_LIST",
            "CUDA_LAUNCH_BLOCKING",
            "CUDA_DEVICE_ORDER",
            "OMP_NUM_THREADS",
            "MKL_THREADING_LAYER",
            "MAMBA_BACKEND",
        )
        if os.environ.get(k) is not None
    }

    return env


def format_session_env_lines(env: Optional[Dict[str, Any]] = None) -> List[str]:
    if env is None:
        env = collect_session_env()

    lines: List[str] = []
    py = env.get("python") or {}
    pyver = py.get("version", "?")
    pykind = py.get("kind", "?")
    pyname = py.get("name", "")
    host = py.get("host", "?")
    py_part = f"{pykind}:{pyname}" if pyname else pykind
    lines.append(
        f"[label]Host:[/label]      [value]{host}[/value]   "
        f"[label]Env:[/label] [value]{py_part}[/value]   "
        f"[label]Python:[/label] [value]{pyver}[/value]"
    )

    torch_info = env.get("torch") or {}
    torch_ver = torch_info.get("version", "?")
    cuda_ver = torch_info.get("cuda") or "n/a"
    lines.append(
        f"[label]Torch:[/label]     [value]{torch_ver}[/value]   "
        f"[label]CUDA:[/label] [value]{cuda_ver}[/value]"
    )

    gpu = env.get("gpu") or {}
    if gpu.get("available"):
        name = gpu.get("name", "?")
        cap = gpu.get("capability", "?")
        arch = gpu.get("arch", "?")
        free = gpu.get("memory_free_mb")
        total = gpu.get("memory_total_mb")
        mem_part = ""
        if free is not None and total is not None:
            mem_part = f"   [label]Mem:[/label] [value]{free:.0f}/{total:.0f} MB[/value]"
        lines.append(
            f"[label]GPU:[/label]       [value]{name}[/value]   "
            f"[value]{cap}[/value] [muted]({arch})[/muted]"
            f"{mem_part}"
        )
    else:
        reason = gpu.get("reason", "no CUDA")
        lines.append(f"[label]GPU:[/label]       [muted]not available -- {reason}[/muted]")

    pkgs = env.get("packages") or {}
    interesting = [
        (k, v) for k, v in pkgs.items()
        if v is not None and k in ("mamba_ssm", "causal_conv1d", "mambapy", "xlstm", "triton")
    ]
    if interesting:
        parts = [f"{k}={v}" for k, v in interesting]
        lines.append(f"[label]DL libs:[/label]   [muted]{'  '.join(parts)}[/muted]")

    return lines


__all__ = [
    "collect_session_env",
    "format_session_env_lines",
    "gpu_signature",
]
