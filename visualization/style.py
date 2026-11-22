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
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl

if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg", force=True)

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

THESIS_TEXT_WIDTH_IN = 5.315
THESIS_TEXT_HEIGHT_IN = 9.33

HEATMAP_CELL_HEIGHT_IN = 0.50

SIDEWAYS_HEATMAP_MAX_HEIGHT_IN = 4.50


def heatmap_figsize(
    n_rows: int, width: float, tick_rotation: float,
) -> tuple:
    margin = 1.78 if tick_rotation >= 40 else 1.52
    height = n_rows * HEATMAP_CELL_HEIGHT_IN + margin
    if abs(width - THESIS_TEXT_HEIGHT_IN) < 0.05:
        height = min(height, SIDEWAYS_HEATMAP_MAX_HEIGHT_IN)
    return (width, height)


SCHEMATIC_STROKE = "#969693"
SCHEMATIC_TEXT = "#5A5854"
SCHEMATIC_FILL = "#F9F8F7"
SCHEMATIC_BLUE_FILL = "#E7F2FB"
SCHEMATIC_BLUE = "#4481D7"
SCHEMATIC_BLUE_DARK = "#3569B2"
SCHEMATIC_RED = "#D56C5E"
SCHEMATIC_PURPLE = "#8E6FC5"
SCHEMATIC_PURPLE_FILL = "#EFE9F7"
CONVERGENCE_CELL_COLOR = "#E8C4B8"

CHUNK_SIZE_COLORS: Dict[str, str] = {
    "ylstm_hb_c1": "#332288",
    "ylstm_hb_c2": "#004488",
    "ylstm_hb_c4": "#228833",
    "ylstm_hb_c8": "#009988",
    "ylstm_hb_c0": SCHEMATIC_RED,
}
CHUNK_SIZE_MARKERS: Dict[str, str] = {
    "ylstm_hb_c1": "o",
    "ylstm_hb_c2": "s",
    "ylstm_hb_c4": "^",
    "ylstm_hb_c8": "v",
    "ylstm_hb_c0": "D",
}
SEQ_LEN_COLORS: Dict[int, str] = {
    16: "#46353A",
    64: "#765166",
    256: "#92668C",
    1024: "#9C7AAE",
    2048: "#9890CA",
    4096: "#88A5DD",
}
HEADLINE_ARCH_COLORS: Dict[str, str] = {
    "transformer": "#661100",
    "mamba": "#E69F00",
    "slstm": "#EE3377",
    "mlstm": "#6699CC",
    "xlstm": "#882255",
    "hlstm": "#999933",
    "xlstm_hb": "#CC79A7",
    "yhlstm": "#D55E00",
}

THESIS_HEATMAP_CMAP = "thesis_blues"


def _register_thesis_cmap() -> None:
    if THESIS_HEATMAP_CMAP in mpl.colormaps:
        return
    cmap = mcolors.LinearSegmentedColormap.from_list(
        THESIS_HEATMAP_CMAP,
        [
            (0.0, SCHEMATIC_FILL),
            (0.55, SCHEMATIC_BLUE_FILL),
            (0.85, SCHEMATIC_BLUE),
            (1.0, SCHEMATIC_BLUE_DARK),
        ],
    )
    mpl.colormaps.register(cmap)


_register_thesis_cmap()

_FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
_MERRIWEATHER_REGISTERED = False


def _register_merriweather() -> None:
    global _MERRIWEATHER_REGISTERED
    if _MERRIWEATHER_REGISTERED:
        return
    for name in ("Merriweather-Regular.ttf", "Merriweather-Bold.ttf"):
        path = _FONT_DIR / name
        if path.is_file():
            fm.fontManager.addfont(str(path))
    _MERRIWEATHER_REGISTERED = True


MODEL_COLORS: Dict[str, str] = {
    "lstm": "#1f77b4",
    "gru": "#2ca02c",
    "transformer": "#e15759",
    "llama": "#e15759",
    "mamba": "#ff7f0e",
    "mamba_py": "#ff7f0e",       # Orange (mambapy)
    "mamba_ssm": "#d62728",      # Red (official mamba-ssm)
    "xlstm": "#8b008b",
    "slstm": "#da70d6",
    "mlstm": "#9370db",
    "xlstm_7_1": "#ba55d3",
    "xlstm_1_0": "#8b0a50",
    "xlstm_0_1": "#8b0a50",
    
    "hlstm": "#4682B4",
    "yhlstm": "#20B2AA",
    "xlstm_hb": "#CD853F",
    "ylstm_hb": "#B8860B",
    "ylstm_hb_c0": "#EE7733",
    "ylstm_hb_c1": "#0077BB",
    "ylstm_hb_c2": "#009988",
    "ylstm_hb_c4": "#CC3311",
    "ylstm_hb_c8": "#AA3377",
    "ylstm": "#708090",
    
    "rwkv": "#2ca02c",
    "rwkv_4": "#006400",
    "rwkv_5": "#228b22",
    "rwkv_6": "#32cd32",
    
    "retention": "#17becf",
    "hyena": "#bcbd22",
    "hgrn": "#7f7f7f",
    "hgrn2": "#7f7f7f",
    
    "default": "#333333",
}

METRIC_COLORS: Dict[str, str] = {
    "loss": "#1f77b4",
    "val_loss": "#ff7f0e",
    "accuracy": "#2ca02c",
    "val_accuracy": "#d62728",
    "val/accuracy": "#d62728",
    "train_loss": "#1f77b4",
    "train_accuracy": "#2ca02c",
}

COMPARISON_PALETTE: List[str] = [
    "#0077BB",
    "#EE7733",
    "#009988",
    "#CC3311",
    "#33BBEE",
    "#EE3377",
    "#117733",
    "#AA3377",
    "#DDCC77",
    "#882255",
]

MODEL_MARKERS: Dict[str, str] = {
    "lstm": "o",
    "gru": "s",
    "transformer": "^",
    "llama": "^",
    "mamba": "D",
    "mamba_py": "D",
    "mamba_ssm": "d",
    "xlstm": "P",
    "slstm": "v",
    "mlstm": "h",
    "xlstm_7_1": "H",
    "default": "o",
}


HEATMAP_CMAP = "Blues"
HEATMAP_CMAP_SCALED = "Blues"
HEATMAP_CMAP_CB = "viridis"
HEATMAP_BEST_BORDER_COLOR = "#F4F5F6"
HEATMAP_BEST_BORDER_WIDTH = 2.5


def get_model_color(model_name: str) -> str:
    normalized = model_name.lower().replace("-", "_").replace(" ", "_")
    
    if normalized in MODEL_COLORS:
        return MODEL_COLORS[normalized]
    
    for key, color in MODEL_COLORS.items():
        if key in normalized or normalized in key:
            return color
    
    return MODEL_COLORS["default"]


def get_metric_color(metric_name: str) -> str:
    normalized = metric_name.lower().replace("-", "_").replace(" ", "_")
    return METRIC_COLORS.get(normalized, "#333333")


def get_model_marker(model_name: str) -> str:
    normalized = model_name.lower().replace("-", "_").replace(" ", "_")
    if normalized in MODEL_MARKERS:
        return MODEL_MARKERS[normalized]
    for key, marker in MODEL_MARKERS.items():
        if key != "default" and (key in normalized or normalized in key):
            return marker
    return MODEL_MARKERS["default"]


def apply_paper_style() -> None:
    _register_merriweather()
    plt.style.use("seaborn-v0_8-whitegrid")

    paper_params = {
        "figure.figsize": (THESIS_TEXT_WIDTH_IN, 3.2),
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",

        "font.family": "serif",
        "font.serif": ["Merriweather", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        "text.color": SCHEMATIC_TEXT,
        "axes.labelcolor": SCHEMATIC_TEXT,
        "xtick.color": SCHEMATIC_TEXT,
        "ytick.color": SCHEMATIC_TEXT,

        "axes.facecolor": "white",
        "axes.edgecolor": SCHEMATIC_STROKE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "grid.alpha": 0.7,

        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "lines.markeredgewidth": 1.2,
        "lines.markeredgecolor": "white",

        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CCCCCC",
        "legend.loc": "best",

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,

        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }

    mpl.rcParams.update(paper_params)


def reset_style() -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)


def has_both_mamba_variants(all_models: Optional[List[str]] = None) -> bool:
    if all_models is None:
        return True
    norms = {m.lower().replace("-", "_").replace(" ", "_") for m in all_models}
    return "mamba_py" in norms and "mamba_ssm" in norms


def format_model_label(model_name: str, all_models: Optional[List[str]] = None) -> str:
    normalized = model_name.lower().replace("-", "_").replace(" ", "_")

    _HYBRID_LABELS = {
        "hlstm": "HLSTM",
        "yhlstm": "yHLSTM",
        "xlstm_hb": "xLSTM-hb",
        "ylstm_hb": "yLSTM-hb",
        "ylstm_hb_c0": "c=T (no feedback)",
        "ylstm_hb_c1": "c=1 (every step)",
        "ylstm_hb_c2": "c=2",
        "ylstm_hb_c4": "c=4",
        "ylstm_hb_c8": "c=8",
        "ylstm": "yLSTM",
    }
    if normalized in _HYBRID_LABELS:
        return _HYBRID_LABELS[normalized]

    if normalized == "slstm":
        return "sLSTM"
    if normalized == "mlstm":
        return "mLSTM"
    if normalized == "xlstm_7_1":
        return "xLSTM_7_1"
    if normalized == "xlstm":
        return "xLSTM"
    if "xlstm" in model_name.lower():
        if "7_1" in model_name or "[7:1]" in model_name:
            return "xLSTM_7_1"
        return "xLSTM"

    if "rwkv" in model_name.lower():
        if "4" in model_name:
            return "RWKV-4"
        elif "5" in model_name:
            return "RWKV-5"
        elif "6" in model_name:
            return "RWKV-6"
        return "RWKV"

    if normalized in ("mamba_py", "mamba_ssm"):
        if has_both_mamba_variants(all_models):
            return "Mamba (py)" if normalized == "mamba_py" else "Mamba (SSM)"
        return "Mamba"

    _NAME_MAP = {
        "lstm": "LSTM",
        "gru": "GRU",
        "transformer": "Transformer",
        "mamba": "Mamba",
        "llama": "Llama",
        "retention": "Retention",
        "hyena": "Hyena",
        "hgrn": "HGRN",
        "hgrn2": "HGRN2",
    }
    return _NAME_MAP.get(normalized, model_name)


_TASK_ABBREVS: Dict[str, str] = {
    "parity_check": "Parity",
    "even_pairs": "Even Pairs",
    "cycle_navigation": "Cycle Nav",
    "modular_arithmetic": "Mod Arith",
    "modular_arithmetic_brackets": "Mod Arith (w Brackets)",
    "reverse_string": "Reverse String",
    "solve_equation": "Solve Equation",
    "stack_manipulation": "Stack Manip",
    "binary_addition": "Binary Add",
    "binary_multiplication": "Binary Mult",
    "bucket_sort": "Bucket Sort",
    "compute_sqrt": "Compute Sqrt",
    "duplicate_string": "Duplicate",
    "missing_duplicate_string": "Missing Duplicate",
    "odds_first": "Odds First",
    "mqar": "MQAR",
}


def format_task_label(task_name: str) -> str:
    normalized = task_name.lower().replace("-", "_").replace(" ", "_")
    if normalized in _TASK_ABBREVS:
        return _TASK_ABBREVS[normalized]
    return task_name.replace("_", " ").title()
