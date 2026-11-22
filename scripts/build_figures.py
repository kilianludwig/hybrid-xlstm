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

"""Rebuild the thesis plots and tables under figures/."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "thesis_headline_figures.py",
    "thesis_cell_figures.py",
    "thesis_cross_chunk_figures.py",
    "thesis_paired_effects.py",
    "thesis_appendix_tables.py",
    "thesis_hybrid_radars.py",
    "thesis_split_noise.py",
    "thesis_throughput_extension.py",
]


def main() -> None:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not previous else str(ROOT) + os.pathsep + previous
    for name in SCRIPTS:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name)],
            cwd=ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
