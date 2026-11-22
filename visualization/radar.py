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
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from visualization.style import (
    apply_paper_style,
    COMPARISON_PALETTE,
    format_model_label as _format_model_label,
    format_task_label as _format_task_label,
    get_model_color,
    get_model_marker,
)
from visualization.utils import is_valid_number as _is_valid, save_or_show as _save_or_show

_RADIAL_TICKS = [0.2, 0.4, 0.6, 0.8, 1.0]


def _radar_angles(n: int) -> List[float]:
    return [2 * math.pi * i / n for i in range(n)]


def _values_for_tasks(
    task_to_value: Dict[str, float],
    task_order: List[str],
) -> List[float]:
    out: List[float] = []
    for task in task_order:
        val = task_to_value.get(task)
        out.append(float(val) if _is_valid(val) else 0.0)
    return out


def _setup_polar_axes(
    ax: plt.Axes,
    task_order: List[str],
    *,
    tick_fontsize: float = 9,
    rlabel_position: Optional[float] = None,
) -> List[float]:
    angles = _radar_angles(len(task_order))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(
        [_format_task_label(t) for t in task_order], fontsize=tick_fontsize
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(_RADIAL_TICKS)
    ax.set_yticklabels([f"{t:.1f}" for t in _RADIAL_TICKS], fontsize=8)
    if rlabel_position is None:
        rlabel_position = 180 / len(task_order)
    ax.set_rlabel_position(rlabel_position)
    ax.grid(True, alpha=0.4)
    return angles


def _align_angular_labels(ax: plt.Axes, pad: float) -> None:
    ax.tick_params(axis="x", pad=pad)
    offset = ax.get_theta_offset()
    direction = ax.get_theta_direction()
    for label, angle in zip(ax.get_xticklabels(), ax.get_xticks()):
        label.set_clip_on(False)
        screen = offset + direction * float(angle)
        cos_s = math.cos(screen)
        sin_s = math.sin(screen)
        if abs(cos_s) < 0.35:
            label.set_horizontalalignment("center")
        elif cos_s > 0:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")
        if abs(sin_s) < 0.35:
            label.set_verticalalignment("center")
        elif sin_s > 0:
            label.set_verticalalignment("bottom")
        else:
            label.set_verticalalignment("top")


def _closed(seq: List[float]) -> List[float]:
    return list(seq) + [seq[0]]


def plot_radar_single(
    model: str,
    task_to_value: Dict[str, float],
    task_order: List[str],
    *,
    std: Optional[Dict[str, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 8),
    dpi: int = 300,
    show: bool = False,
) -> Optional[plt.Figure]:
    if not task_order or not task_to_value:
        return None

    apply_paper_style()

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    angles = _setup_polar_axes(ax, task_order)
    color = get_model_color(model)

    values = _values_for_tasks(task_to_value, task_order)
    ang_c = _closed(angles)
    val_c = _closed(values)

    ax.plot(
        ang_c, val_c, "-",
        color=color,
        linewidth=2,
        marker=get_model_marker(model),
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label=_format_model_label(model),
    )
    ax.fill(ang_c, val_c, color=color, alpha=0.15)

    if std:
        lo = _closed([max(0.0, v - (std.get(t, 0.0) or 0.0)) for v, t in zip(values, task_order)])
        hi = _closed([min(1.0, v + (std.get(t, 0.0) or 0.0)) for v, t in zip(values, task_order)])
        ax.fill_between(ang_c, lo, hi, color=color, alpha=0.10, linewidth=0)

    ax.set_title(title or f"Scaled accuracy — {_format_model_label(model)}", pad=24)

    return _save_or_show(fig, output_path, dpi, show)


def plot_radar_overlay(
    results: Dict[str, Dict[str, float]],
    task_order: List[str],
    *,
    model_order: Optional[List[str]] = None,
    color_overrides: Optional[Dict[str, str]] = None,
    linestyle_overrides: Optional[Dict[str, str]] = None,
    marker_overrides: Optional[Dict[str, str]] = None,
    label_overrides: Optional[Dict[str, str]] = None,
    std: Optional[Dict[str, Dict[str, float]]] = None,
    legend_below: bool = False,
    legend_ncol: int = 1,
    tick_fontsize: float = 9,
    tick_pad: Optional[float] = None,
    align_labels: bool = False,
    rlabel_position: Optional[float] = None,
    lift_outer_rlabel: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: tuple = (9, 8),
    dpi: int = 300,
    show: bool = False,
) -> Optional[plt.Figure]:
    if not task_order or not results:
        return None

    apply_paper_style()

    if model_order:
        ordered = [m for m in model_order if m in results]
        ordered += sorted(m for m in results if m not in ordered)
    else:
        ordered = sorted(results)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    angles = _setup_polar_axes(
        ax, task_order,
        tick_fontsize=tick_fontsize,
        rlabel_position=rlabel_position,
    )
    ang_c = _closed(angles)

    for i, model in enumerate(ordered):
        values = _values_for_tasks(results[model], task_order)
        val_c = _closed(values)
        color = (color_overrides or {}).get(model)
        if color is None:
            color = get_model_color(model)
            if color == "#333333":
                color = COMPARISON_PALETTE[i % len(COMPARISON_PALETTE)]
        linestyle = (linestyle_overrides or {}).get(model, "-")
        marker = (marker_overrides or {}).get(model, get_model_marker(model))
        label = (label_overrides or {}).get(model)
        if label is None:
            label = _format_model_label(model, ordered)
        ax.plot(
            ang_c, val_c, linestyle,
            color=color,
            linewidth=1.8,
            marker=marker,
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
        ax.fill(ang_c, val_c, color=color, alpha=0.08)

        model_std = (std or {}).get(model)
        if model_std:
            lo = _closed([
                max(0.0, v - (model_std.get(t, 0.0) or 0.0))
                for v, t in zip(values, task_order)
            ])
            hi = _closed([
                min(1.0, v + (model_std.get(t, 0.0) or 0.0))
                for v, t in zip(values, task_order)
            ])
            ax.fill_between(ang_c, lo, hi, color=color, alpha=0.10, linewidth=0)

    if align_labels:
        _align_angular_labels(ax, 2.0 if tick_pad is None else tick_pad)
    elif tick_pad is not None:
        ax.tick_params(axis="x", pad=tick_pad)
    if lift_outer_rlabel:
        for lab in ax.get_yticklabels():
            lab.set_ha("center")
            lab.set_va("center")
            if lab.get_text() == "1.0":
                lab.set_visible(False)
        ray = math.radians(
            ax.get_rlabel_position() if rlabel_position is None else rlabel_position
        )
        screen = ax.get_theta_offset() + ax.get_theta_direction() * ray
        ax.text(
            ray,
            1.0,
            "1.0",
            ha="center",
            va="center",
            fontsize=8,
            clip_on=False,
            zorder=5,
            transform=ax.transData + mtransforms.ScaledTranslation(
                10 * math.cos(screen) / 72,
                10 * math.sin(screen) / 72,
                fig.dpi_scale_trans,
            ),
        )
    if title is None:
        ax.set_title("Scaled accuracy by task", pad=24)
    elif title:
        ax.set_title(title, pad=24)
    if legend_below:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=legend_ncol,
            frameon=True,
        )
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.05), frameon=True)

    return _save_or_show(fig, output_path, dpi, show)
