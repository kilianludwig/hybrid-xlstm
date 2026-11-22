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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt

from visualization.style import (
    apply_paper_style,
    COMPARISON_PALETTE,
    THESIS_TEXT_WIDTH_IN,
    format_model_label as _format_model_label,
    get_model_color,
)
from visualization.utils import save_or_show as _save_or_show


def plot_model_comparison(
    results: Dict[str, List[Dict[str, Any]]],
    metric: str = "val_accuracy",
    output_path: Optional[Union[str, Path]] = None,
    x_axis: str = "epoch",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    show: bool = False,
) -> Optional[plt.Figure]:
    if not results:
        return None
    
    apply_paper_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    all_model_keys = list(results.keys())
    for i, (model_name, history) in enumerate(results.items()):
        if not history:
            continue
        
        if x_axis == "epoch":
            x_values = list(range(1, len(history) + 1))
        else:
            x_values = [h.get(x_axis, j) for j, h in enumerate(history)]
        
        y_values = [h.get(metric) for h in history]
        
        if not all(v is not None for v in y_values):
            continue
        
        color = COMPARISON_PALETTE[i % len(COMPARISON_PALETTE)]
        label = _format_model_label(model_name, all_model_keys)
        
        ax.plot(x_values, y_values, 'o-', 
                label=label, 
                color=color,
                markersize=6,
                markeredgecolor='white',
                markeredgewidth=1.2)
    
    ax.set_xlabel(xlabel or _format_axis_label(x_axis))
    ax.set_ylabel(ylabel or _format_metric_label(metric))
    
    if title:
        ax.set_title(title)
    
    n_series = sum(1 for h in results.values() if h)
    ncol = 2 if n_series > 15 else 1
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=ncol, frameon=True)

    fig.subplots_adjust(left=0.08, right=0.72 if n_series > 15 else 0.85, top=0.92, bottom=0.1)

    return _save_or_show(fig, output_path, dpi, show)


def plot_bar_comparison(
    results: Dict[str, float],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    ylabel: str = "Score",
    figsize: tuple = (THESIS_TEXT_WIDTH_IN, 3.2),
    dpi: int = 300,
    sort_by_value: bool = True,
    show_values: bool = True,
    show: bool = False,
    yerr: Optional[Dict[str, float]] = None,
) -> Optional[plt.Figure]:
    if not results:
        return None
    
    apply_paper_style()
    
    items = list(results.items())
    if sort_by_value:
        items = sorted(items, key=lambda x: x[1], reverse=True)
    
    all_model_keys = [m for m, _ in items]
    model_names = [_format_model_label(m, all_model_keys) for m in all_model_keys]
    values = [v for _, v in items]
    colors = [get_model_color(m) for m in all_model_keys]
    errors = [yerr.get(m, 0.0) for m in all_model_keys] if yerr else None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(model_names, values, color=colors, edgecolor='white', linewidth=1.2)
    if errors is not None:
        ax.errorbar(
            model_names, values, yerr=errors,
            fmt='none', ecolor='#333333', elinewidth=1.2, capsize=4, capthick=1.2,
        )
    
    if show_values:
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            y_anchor = height + (errors[i] if errors is not None else 0.0)
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, y_anchor),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return _save_or_show(fig, output_path, dpi, show)


def _format_axis_label(axis: str) -> str:
    labels = {
        "epoch": "Epoch",
        "parameters": "Number of Parameters",
        "model_dim": "Model Dimension",
        "d_model": "Model Dimension",
        "num_layers": "Number of Layers",
        "batch_size": "Batch Size",
    }
    return labels.get(axis.lower(), axis.replace("_", " ").title())


def _format_metric_label(metric: str) -> str:
    labels = {
        "val_accuracy": "Validation Accuracy",
        "val_loss": "Validation Loss",
        "accuracy": "Accuracy",
        "loss": "Loss",
        "perplexity": "Perplexity",
        "val_perplexity": "Validation Perplexity",
    }
    return labels.get(metric.lower(), metric.replace("_", " ").title())
