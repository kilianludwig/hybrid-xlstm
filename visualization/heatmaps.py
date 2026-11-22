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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.patches as mpl_patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from visualization.style import (
    apply_paper_style,
    CONVERGENCE_CELL_COLOR,
    HEATMAP_CMAP,
    HEATMAP_BEST_BORDER_COLOR,
    SCHEMATIC_TEXT,
    format_model_label as _format_model_label,
    format_task_label as _format_task_label,
)
from visualization.utils import is_valid_number as _is_valid, save_or_show as _save_or_show

HEATMAP_MISSING_COLOR = "#E8E8E8"

_MODEL_ORDER = [
    "gru", "lstm", "mamba", "mamba_py", "mamba_ssm",
    "transformer", "slstm", "mlstm", "xlstm", "xlstm_7_1",
    "hlstm", "yhlstm", "xlstm_hb", "ylstm_hb",
    "ylstm_hb_c0", "ylstm_hb_c8", "ylstm_hb_c4", "ylstm_hb_c2", "ylstm_hb_c1",
    "ylstm",
]

_BEST_PATH_EFFECTS = [
    path_effects.Stroke(linewidth=2, foreground="#2B2F33"),
    path_effects.Normal(),
]

_GREY_OVERLAY_COLOR = (0.55, 0.55, 0.55, 0.45)
_GROUP_BOUNDARY_COLOR = "#9AA0A6"


def _annotate_cells(
    ax, data: np.ndarray, std_data: np.ndarray, *,
    cmap: str, vmin: float, vmax: float,
    decimals: int = 2, fontsize: int = 8,
    highlight_best: bool = True,
    best_per_task: Optional[Dict[int, Set[int]]] = None,
    annotate_std: bool = True,
    greyed_columns: Optional[Set[int]] = None,
) -> None:
    if best_per_task is None:
        best_per_task = {}
    if greyed_columns is None:
        greyed_columns = set()
    n_models, n_tasks = data.shape
    for i in range(n_models):
        for j in range(n_tasks):
            value = data[i, j]
            std_val = std_data[i, j]
            is_greyed = j in greyed_columns
            if np.isnan(value):
                ax.text(j, i, "\u2014", ha="center", va="center",
                        color="#AAAAAA" if is_greyed else "#555555", fontsize=fontsize)
                continue
            is_best = highlight_best and i in best_per_task.get(j, set())
            if is_greyed:
                text_color, fw = "#888888", "normal"
            else:
                text_color = _text_color_for_value(value, cmap, vmin, vmax)
                fw = "bold" if is_best else "normal"
            if annotate_std and std_val > 0:
                text = f"{value:.{decimals}f}\n\u00b1{std_val:.{decimals}f}"
            else:
                text = f"{value:.{decimals}f}"
            ax.text(j, i, text, ha="center", va="center",
                    color=text_color, fontsize=fontsize, fontweight=fw)

def _sort_models(model_names: List[str]) -> List[str]:
    order_map = {name: idx for idx, name in enumerate(_MODEL_ORDER)}

    def _key(m: str) -> Tuple[int, str]:
        norm = m.lower().replace("-", "_")
        return (order_map.get(norm, len(_MODEL_ORDER)), norm)

    return sorted(model_names, key=_key)


def _text_color_for_value(value: float, cmap_name: str, vmin: float, vmax: float) -> str:
    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl_cm.get_cmap(cmap_name)
    rgba = cmap(norm(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if luminance < 0.5 else SCHEMATIC_TEXT


def _best_indices_per_column(data: np.ndarray) -> Dict[int, Set[int]]:
    best_per_task: Dict[int, Set[int]] = {}
    for j in range(data.shape[1]):
        col = data[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        col_max = np.nanmax(col)
        best_per_task[j] = set(np.where(np.abs(col - col_max) < 1e-9)[0])
    return best_per_task


def _best_indices_convergence(
    data: np.ndarray,
    epoch_data: np.ndarray,
    solved_mask: np.ndarray,
) -> Dict[int, Set[int]]:
    best: Dict[int, Set[int]] = {}
    for j in range(data.shape[1]):
        col = data[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        solved_rows = [
            i for i in range(data.shape[0])
            if solved_mask[i, j] and not np.isnan(data[i, j])
        ]
        if not solved_rows:
            col_max = np.nanmax(col)
            best[j] = set(np.where(np.abs(col - col_max) < 1e-9)[0])
            continue
        ep_vals = np.array([
            epoch_data[i, j] if not np.isnan(epoch_data[i, j]) else np.inf
            for i in solved_rows
        ])
        min_ep = np.min(ep_vals)
        best[j] = {
            solved_rows[k] for k, v in enumerate(ep_vals)
            if np.abs(v - min_ep) < 0.5
        }
    return best


@dataclass
class _GroupedHeatmapScaffold:
    fig: Any
    ax: Any
    im: Any
    model_names: List[str]
    all_tasks: List[str]
    group_boundaries: List[int]
    group_labels: List[str]
    data: np.ndarray
    std_data: np.ndarray
    solved_mask: np.ndarray
    n_models: int = field(init=False)
    n_tasks: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_models = len(self.model_names)
        self.n_tasks = len(self.all_tasks)


def _prepare_grouped_heatmap(
    results: Dict[str, Dict[str, float]],
    task_groups: Dict[str, List[str]],
    std_results: Optional[Dict[str, Dict[str, float]]],
    cmap: str,
    vmin: float,
    vmax: float,
    figsize: Optional[Tuple[float, float]],
    *,
    default_cell_width: float = 1.0,
    default_min_width: float = 10,
    default_cell_height: float = 0.7,
    default_min_height: float = 5,
    solved_threshold: Optional[float] = None,
    solved_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> _GroupedHeatmapScaffold:
    apply_paper_style()
    model_names = _sort_models(list(results.keys()))

    all_tasks: List[str] = []
    group_boundaries: List[int] = []
    group_labels: List[str] = []
    for group_name, tasks in task_groups.items():
        group_boundaries.append(len(all_tasks))
        group_labels.append(group_name)
        all_tasks.extend(tasks)
    group_boundaries.append(len(all_tasks))

    n_models = len(model_names)
    n_tasks = len(all_tasks)

    data = np.full((n_models, n_tasks), np.nan)
    std_data = np.zeros((n_models, n_tasks))
    for i, model in enumerate(model_names):
        for j, task in enumerate(all_tasks):
            val = results[model].get(task, np.nan)
            if _is_valid(val):
                data[i, j] = float(val)
            if std_results and model in std_results:
                std_data[i, j] = std_results[model].get(task, 0.0)

    solved_mask = np.zeros((n_models, n_tasks), dtype=bool)
    if solved_threshold is not None:
        if solved_results is not None:
            for i, model in enumerate(model_names):
                model_solved = solved_results.get(model, {})
                for j, task in enumerate(all_tasks):
                    sv = model_solved.get(task, np.nan)
                    if _is_valid(sv) and float(sv) >= solved_threshold:
                        solved_mask[i, j] = True
        else:
            valid_data = ~np.isnan(data)
            solved_mask = valid_data & (data >= solved_threshold)

    display_data = data.copy()
    if solved_threshold is not None:
        display_data[solved_mask] = vmax
    to_show = np.ma.masked_invalid(display_data)

    if figsize is None:
        figsize = (
            max(default_min_width, n_tasks * default_cell_width),
            max(default_min_height, n_models * default_cell_height),
        )

    cmap_obj = mpl_cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=HEATMAP_MISSING_COLOR)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        to_show,
        cmap=cmap_obj,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    return _GroupedHeatmapScaffold(
        fig=fig, ax=ax, im=im,
        model_names=model_names,
        all_tasks=all_tasks,
        group_boundaries=group_boundaries,
        group_labels=group_labels,
        data=data,
        std_data=std_data,
        solved_mask=solved_mask,
    )


def _finalize_grouped_axes(
    s: _GroupedHeatmapScaffold,
    title: Optional[str],
    *,
    task_labels: Optional[List[str]] = None,
    label_fontsize: Optional[int] = None,
    tick_fontsize: Optional[int] = None,
    group_fontsize: Optional[int] = None,
    title_fontsize: Optional[int] = None,
    boundary_color: str = _GROUP_BOUNDARY_COLOR,
    tick_rotation: int = 45,
) -> None:
    ax = s.ax

    for boundary in s.group_boundaries[1:-1]:
        ax.axvline(x=boundary - 0.5, color=boundary_color, linewidth=1.2)

    label_kw = {"fontsize": label_fontsize} if label_fontsize else {}
    tick_kw = {"fontsize": tick_fontsize} if tick_fontsize else {}

    ax.set_xlabel("Task", **label_kw)
    ax.set_ylabel("Model", **label_kw)
    ax.set_xticks(np.arange(s.n_tasks))
    ax.set_yticks(np.arange(s.n_models))

    if task_labels is None:
        task_labels = [_format_task_label(t) for t in s.all_tasks]
    ax.set_xticklabels(
        task_labels, rotation=tick_rotation, ha="right",
        rotation_mode="anchor", **tick_kw,
    )
    ax.set_yticklabels(
        [_format_model_label(m, s.model_names) for m in s.model_names],
        **tick_kw,
    )

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    group_centers = [
        (s.group_boundaries[k] + s.group_boundaries[k + 1]) / 2 - 0.5
        for k in range(len(s.group_labels))
    ]
    group_kw = {"fontsize": group_fontsize} if group_fontsize else {}
    ax2.set_xticks(group_centers)
    ax2.set_xticklabels(s.group_labels, fontweight="bold", **group_kw)
    ax2.tick_params(length=0)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(boundary_color)
        spine.set_linewidth(0.8)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.grid(False)
    ax2.grid(False)

    if title:
        title_kw = {"fontsize": title_fontsize} if title_fontsize else {}
        ax.set_title(title, pad=40, **title_kw)

    plt.tight_layout()


def _add_compact_colorbar(
    s: _GroupedHeatmapScaffold,
    label: str,
    *,
    fontsize: int = 9,
) -> None:
    cbar = s.fig.colorbar(s.im, ax=s.ax, fraction=0.025, pad=0.02)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.outline.set_edgecolor(_GROUP_BOUNDARY_COLOR)
    cbar.outline.set_linewidth(0.8)


def plot_task_heatmap(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
    std_results: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,
    highlight_best: bool = True,
    annotate: bool = True,
    annotate_std: bool = True,
    cmap: str = HEATMAP_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar_label: str = "Accuracy",
    decimals: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    show: bool = False,
) -> Optional[plt.Figure]:
    if not results:
        return None

    apply_paper_style()

    model_names = _sort_models(list(results.keys()))
    task_names = sorted(set(
        task for tasks in results.values() for task in tasks.keys()
    ))

    data = np.full((len(model_names), len(task_names)), np.nan)
    std_data = np.zeros((len(model_names), len(task_names)))
    for i, model in enumerate(model_names):
        for j, task in enumerate(task_names):
            val = results[model].get(task, np.nan)
            if _is_valid(val):
                data[i, j] = float(val)
            if std_results and model in std_results:
                std_data[i, j] = std_results[model].get(task, 0.0)

    to_show = np.ma.masked_invalid(data)

    if figsize is None:
        width = max(8, len(task_names) * 1.2)
        height = max(4, len(model_names) * 0.6)
        figsize = (width, height)

    cmap_obj = mpl_cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=HEATMAP_MISSING_COLOR)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        to_show,
        cmap=cmap_obj,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    if annotate:
        best_per_task = _best_indices_per_column(data) if highlight_best else {}
        _annotate_cells(ax, data, std_data, cmap=cmap, vmin=vmin, vmax=vmax,
                        decimals=decimals, fontsize=8, highlight_best=highlight_best,
                        best_per_task=best_per_task, annotate_std=annotate_std)

    ax.set_xlabel("Task")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(len(task_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([_format_task_label(t) for t in task_names], rotation=45, ha="right")
    ax.set_yticklabels([_format_model_label(m, model_names) for m in model_names])
    if title:
        ax.set_title(title, pad=20)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(colorbar_label)
    plt.tight_layout()

    return _save_or_show(fig, output_path, dpi, show)


def plot_grouped_heatmap(
    results: Dict[str, Dict[str, float]],
    task_groups: Dict[str, List[str]],
    output_path: Optional[Union[str, Path]] = None,
    std_results: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,
    highlight_best: bool = True,
    annotate_std: bool = True,
    cmap: str = HEATMAP_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar_label: str = "Accuracy",
    decimals: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    show: bool = False,
    greyed_columns: Optional[Set[int]] = None,
    annotation_fontsize: int = 7,
    show_colorbar: bool = False,
    boundary_color: str = _GROUP_BOUNDARY_COLOR,
    tick_rotation: int = 45,
) -> Optional[plt.Figure]:
    if not results or not task_groups:
        return None
    if greyed_columns is None:
        greyed_columns = set()

    s = _prepare_grouped_heatmap(
        results, task_groups, std_results, cmap, vmin, vmax, figsize,
        default_cell_height=0.95,
    )
    best_per_task = _best_indices_per_column(s.data) if highlight_best else {}

    _annotate_cells(s.ax, s.data, s.std_data, cmap=cmap, vmin=vmin, vmax=vmax,
                    decimals=decimals, fontsize=annotation_fontsize,
                    highlight_best=highlight_best,
                    best_per_task=best_per_task, annotate_std=annotate_std,
                    greyed_columns=greyed_columns)

    for j in greyed_columns:
        if 0 <= j < s.n_tasks:
            rect = mpl_patches.Rectangle(
                (j - 0.5, -0.5), 1.0, s.n_models,
                facecolor=_GREY_OVERLAY_COLOR, edgecolor="none", zorder=3,
            )
            s.ax.add_patch(rect)

    task_labels = None
    if greyed_columns:
        task_labels = [
            f"({_format_task_label(t)})" if j in greyed_columns else _format_task_label(t)
            for j, t in enumerate(s.all_tasks)
        ]

    if show_colorbar:
        _add_compact_colorbar(s, colorbar_label)

    _finalize_grouped_axes(
        s, title, task_labels=task_labels,
        label_fontsize=10, tick_fontsize=9,
        group_fontsize=10, title_fontsize=10,
        boundary_color=boundary_color, tick_rotation=tick_rotation,
    )

    if greyed_columns:
        for j, tick_label in enumerate(s.ax.get_xticklabels()):
            if j in greyed_columns:
                tick_label.set_color("#999999")

    return _save_or_show(s.fig, output_path, dpi, show)


def plot_grouped_heatmap_with_insight(
    results: Dict[str, Dict[str, float]],
    task_groups: Dict[str, List[str]],
    greyed_columns: Optional[Set[int]] = None,
    output_path: Optional[Union[str, Path]] = None,
    std_results: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,
    highlight_best: bool = True,
    annotate_std: bool = True,
    cmap: str = HEATMAP_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar_label: str = "Accuracy",
    decimals: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    show: bool = False,
) -> Optional[plt.Figure]:
    return plot_grouped_heatmap(
        results, task_groups, output_path=output_path,
        std_results=std_results, title=title, highlight_best=highlight_best,
        annotate_std=annotate_std, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar_label=colorbar_label, decimals=decimals, figsize=figsize,
        dpi=dpi, show=show, greyed_columns=greyed_columns,
    )


def plot_grouped_heatmap_convergence(
    results: Dict[str, Dict[str, float]],
    task_groups: Dict[str, List[str]],
    epoch_results: Dict[str, Dict[str, float]],
    early_stop_threshold: float = 0.95,
    output_path: Optional[Union[str, Path]] = None,
    std_results: Optional[Dict[str, Dict[str, float]]] = None,
    epoch_std_results: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,
    highlight_best: bool = True,
    cmap: str = HEATMAP_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar_label: str = "Accuracy",
    decimals: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    show: bool = False,
    solved_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[plt.Figure]:
    if not results or not task_groups:
        return None

    s = _prepare_grouped_heatmap(
        results, task_groups, std_results, cmap, vmin, vmax, figsize,
        default_cell_width=1.2, default_cell_height=1.3,
        default_min_height=6,
        solved_threshold=early_stop_threshold,
        solved_results=solved_results,
    )

    epoch_data = np.full((s.n_models, s.n_tasks), np.nan)
    epoch_std_data = np.zeros((s.n_models, s.n_tasks))
    for i, model in enumerate(s.model_names):
        for j, task in enumerate(s.all_tasks):
            ep = (epoch_results.get(model) or {}).get(task, np.nan)
            if _is_valid(ep):
                epoch_data[i, j] = float(ep)
            if epoch_std_results and model in epoch_std_results:
                epoch_std_data[i, j] = epoch_std_results[model].get(task, 0.0)

    best_per_task = (
        _best_indices_convergence(s.data, epoch_data, s.solved_mask)
        if highlight_best else {}
    )

    for i in range(s.n_models):
        for j in range(s.n_tasks):
            value = s.data[i, j]
            ep_val = epoch_data[i, j]
            ep_std = epoch_std_data[i, j]
            if np.isnan(value):
                s.ax.text(j, i, "\u2014", ha="center", va="center",
                          color="#555555", fontsize=10)
                continue

            solved = bool(s.solved_mask[i, j])
            is_best = highlight_best and i in best_per_task.get(j, set())

            has_epoch = not np.isnan(ep_val)
            if has_epoch:
                if ep_std > 0.5:
                    epoch_str = f"{ep_val:.0f}\u00b1{ep_std:.0f} ep"
                else:
                    epoch_str = f"{ep_val:.0f} ep"

            text_color = (
                HEATMAP_BEST_BORDER_COLOR
                if is_best
                else _text_color_for_value(value, cmap, vmin, vmax)
            )
            y_top = i - 0.18 if has_epoch else i
            y_bot = i + 0.22

            if solved:
                top_obj = s.ax.text(
                    j, y_top, r"$\checkmark$", ha="center", va="center",
                    color=text_color, fontsize=16, fontweight="bold",
                )
            else:
                top_obj = s.ax.text(
                    j, y_top, f"{value:.{decimals}f}", ha="center", va="center",
                    color=text_color, fontsize=10,
                    fontweight="bold" if is_best else "normal",
                )

            if has_epoch:
                ep_obj = s.ax.text(
                    j, y_bot, epoch_str, ha="center", va="center",
                    color=text_color, fontsize=8,
                    fontweight="bold" if is_best else "normal",
                )

            if is_best:
                top_obj.set_path_effects(_BEST_PATH_EFFECTS)
                if has_epoch:
                    ep_obj.set_path_effects(_BEST_PATH_EFFECTS)

    _finalize_grouped_axes(
        s, title,
        label_fontsize=10, tick_fontsize=9,
        group_fontsize=10, title_fontsize=10,
    )
    return _save_or_show(s.fig, output_path, dpi, show)


def plot_grouped_heatmap_solved(
    results: Dict[str, Dict[str, float]],
    task_groups: Dict[str, List[str]],
    threshold: float = 0.95,
    output_path: Optional[Union[str, Path]] = None,
    std_results: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,
    highlight_best: bool = True,
    cmap: str = HEATMAP_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar_label: str = "Accuracy",
    decimals: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    show: bool = False,
    solved_results: Optional[Dict[str, Dict[str, float]]] = None,
    binary_only: bool = False,
    annotation_fontsize: int = 8,
    check_fontsize: int = 16,
    show_colorbar: bool = False,
    boundary_color: str = _GROUP_BOUNDARY_COLOR,
    tick_rotation: int = 45,
    annotation_suffix: Optional[Dict[str, Dict[str, str]]] = None,
    linespacing: Optional[float] = None,
) -> Optional[plt.Figure]:
    if not results or not task_groups:
        return None

    cell_height = 0.45 if binary_only else 1.3
    s = _prepare_grouped_heatmap(
        results, task_groups, std_results, cmap, vmin, vmax, figsize,
        default_cell_height=cell_height,
        default_min_height=6,
        solved_threshold=threshold,
        solved_results=solved_results,
    )

    best_per_task: Dict[int, Set[int]] = {}
    if highlight_best:
        for j in range(s.n_tasks):
            col = s.data[:, j]
            valid_mask = ~np.isnan(col)
            if not valid_mask.any():
                continue
            solved_col = s.solved_mask[:, j]
            solved_rows = set(np.where(valid_mask & solved_col)[0])
            all_solved = bool(np.all(solved_col[valid_mask]))
            if solved_rows and not all_solved:
                best_per_task[j] = solved_rows
            elif not solved_rows:
                col_max = np.nanmax(col[valid_mask])
                best_per_task[j] = set(
                    np.where(valid_mask & (np.abs(col - col_max) < 1e-9))[0]
                )

    for i in range(s.n_models):
        for j in range(s.n_tasks):
            value = s.data[i, j]
            std_val = s.std_data[i, j]
            if np.isnan(value):
                s.ax.text(j, i, "\u2014", ha="center", va="center",
                          color="#555555", fontsize=8)
                continue

            solved = bool(s.solved_mask[i, j])
            is_best = highlight_best and i in best_per_task.get(j, set())
            text_color = _text_color_for_value(value, cmap, vmin, vmax)

            if solved:
                check_fs = 12 if binary_only else check_fontsize
                s.ax.text(
                    j, i, r"$\checkmark$", ha="center", va="center",
                    color=text_color, fontsize=check_fs, fontweight="bold",
                )
            elif not binary_only:
                if std_val > 0:
                    text = f"{value:.{decimals}f}\n\u00b1{std_val:.{decimals}f}"
                else:
                    text = f"{value:.{decimals}f}"
                if annotation_suffix is not None:
                    suffix = annotation_suffix.get(
                        s.model_names[i], {}
                    ).get(s.all_tasks[j])
                    if suffix:
                        text = f"{text}\n{suffix}"
                text_kw = {}
                if linespacing is not None:
                    text_kw["linespacing"] = linespacing
                s.ax.text(
                    j, i, text, ha="center", va="center",
                    color=text_color, fontsize=annotation_fontsize,
                    fontweight="bold" if is_best else "normal",
                    **text_kw,
                )

    if show_colorbar:
        _add_compact_colorbar(s, colorbar_label)

    _finalize_grouped_axes(
        s, title,
        label_fontsize=10, tick_fontsize=9,
        group_fontsize=10, title_fontsize=10,
        boundary_color=boundary_color, tick_rotation=tick_rotation,
    )
    return _save_or_show(s.fig, output_path, dpi, show)


def plot_grouped_heatmap_solved_epochs(
    results: Dict[str, Dict[str, float]],
    task_groups: Dict[str, List[str]],
    epoch_results: Dict[str, Dict[str, float]],
    threshold: float = 0.95,
    output_path: Optional[Union[str, Path]] = None,
    epoch_std_results: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,
    cmap: str = HEATMAP_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    show: bool = False,
    solved_results: Optional[Dict[str, Dict[str, float]]] = None,
    annotation_fontsize: int = 8,
    check_fontsize: int = 14,
    boundary_color: str = _GROUP_BOUNDARY_COLOR,
    tick_rotation: int = 45,
) -> Optional[plt.Figure]:
    if not results or not task_groups:
        return None

    s = _prepare_grouped_heatmap(
        results, task_groups, None, cmap, vmin, vmax, figsize,
        default_cell_height=1.3,
        default_min_height=6,
        solved_threshold=threshold,
        solved_results=solved_results,
    )

    epoch_data = np.full((s.n_models, s.n_tasks), np.nan)
    epoch_std_data = np.zeros((s.n_models, s.n_tasks))
    for i, model in enumerate(s.model_names):
        for j, task in enumerate(s.all_tasks):
            ep = (epoch_results.get(model) or {}).get(task, np.nan)
            if _is_valid(ep):
                epoch_data[i, j] = float(ep)
            if epoch_std_results and model in epoch_std_results:
                epoch_std_data[i, j] = float(
                    epoch_std_results[model].get(task, 0.0) or 0.0
                )

    text_color = "black"
    for i in range(s.n_models):
        for j in range(s.n_tasks):
            s.ax.add_patch(mpl_patches.Rectangle(
                (j - 0.5, i - 0.5), 1.0, 1.0,
                facecolor=(
                    CONVERGENCE_CELL_COLOR
                    if s.solved_mask[i, j]
                    else HEATMAP_MISSING_COLOR
                ),
                edgecolor="none",
                zorder=1,
            ))
            if not s.solved_mask[i, j]:
                continue

            ep_val = epoch_data[i, j]
            ep_std = epoch_std_data[i, j]
            if np.isnan(ep_val):
                number = "\u2014"
            else:
                number = f"{ep_val:.1f}\n\u00b1{ep_std:.1f}"

            s.ax.text(
                j, i - 0.22, r"$\checkmark$", ha="center", va="center",
                color=text_color, fontsize=check_fontsize, fontweight="bold",
                zorder=4,
            )
            s.ax.text(
                j, i + 0.18, number, ha="center", va="center",
                color=text_color, fontsize=annotation_fontsize,
                linespacing=0.9, zorder=4,
            )

    _finalize_grouped_axes(
        s, title,
        label_fontsize=10, tick_fontsize=9,
        group_fontsize=10, title_fontsize=10,
        boundary_color=boundary_color, tick_rotation=tick_rotation,
    )
    return _save_or_show(s.fig, output_path, dpi, show)
