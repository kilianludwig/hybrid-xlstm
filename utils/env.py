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
import sys
from pathlib import Path
from typing import Dict, Optional


def describe_python_env() -> Dict[str, str]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    virtual_env = os.environ.get("VIRTUAL_ENV")

    if conda_env or conda_prefix:
        kind = "conda"
        name = conda_env or (Path(conda_prefix).name if conda_prefix else "")
    elif virtual_env:
        kind = "venv"
        name = Path(virtual_env).name
    elif sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        kind = "venv"
        name = Path(sys.prefix).name
    else:
        kind = "system"
        name = ""

    try:
        host = os.uname().nodename.split(".")[0]
    except AttributeError:  # pragma: no cover - non-POSIX platforms
        import platform

        host = platform.node().split(".")[0]

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return {
        "kind": kind,
        "name": name,
        "python": sys.executable,
        "prefix": sys.prefix,
        "version": version,
        "host": host,
    }


def format_env(info: Optional[Dict[str, str]] = None, *, location: str = "local") -> str:
    if info is None:
        info = describe_python_env()

    kind = info.get("kind", "system")
    name = info.get("name") or ""
    version = info.get("version") or ""
    host = info.get("host") or ""

    if kind in ("conda", "venv") and name:
        env_part = f"{kind}:{name}"
    else:
        env_part = kind or "system"

    where = location
    if location == "remote" and host:
        where = f"remote@{host}"

    suffix = f" (py {version})" if version else ""
    return f"{where} {env_part}{suffix}"


__all__ = ["describe_python_env", "format_env"]
