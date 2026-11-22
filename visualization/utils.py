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

import math
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt


def is_valid_number(val) -> bool:
    if val is None:
        return False
    return not (isinstance(val, float) and math.isnan(val))


def save_figure(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white' if not transparent else 'none',
        edgecolor='none',
        pad_inches=0.1,
    )

    return output_path


def save_or_show(
    fig: plt.Figure,
    output_path: Optional[Union[str, Path]],
    dpi: int = 300,
    show: bool = False,
) -> Optional[plt.Figure]:
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return None
    if show:
        plt.show()
    return fig


def add_legend(
    ax: plt.Axes,
    loc: str = "best",
    ncol: int = 1,
    frameon: bool = True,
    fontsize: int = 10,
) -> None:
    ax.legend(
        loc=loc,
        ncol=ncol,
        frameon=frameon,
        fontsize=fontsize,
        framealpha=0.9,
        facecolor='white',
        edgecolor='#CCCCCC',
    )
