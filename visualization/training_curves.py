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
import numpy as np

from visualization.style import apply_paper_style, get_metric_color
from visualization.utils import save_or_show as _save_or_show


def plot_training_curves(
    history: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    include_val: bool = True,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    show: bool = False,
) -> Optional[plt.Figure]:
    if not history:
        return None
    
    apply_paper_style()
    
    if metrics is None:
        available = set()
        for h in history:
            available.update(h.keys())
        
        default_order = ["loss", "val_loss", "accuracy", "val_accuracy"]
        metrics = [m for m in default_order if m in available]
        
        if not include_val:
            metrics = [m for m in metrics if not m.startswith("val_")]
    
    if not metrics:
        return None
    
    loss_metrics = [m for m in metrics if "loss" in m.lower()]
    acc_metrics = [m for m in metrics if "accuracy" in m.lower() or "acc" in m.lower()]
    
    has_loss = len(loss_metrics) > 0
    has_acc = len(acc_metrics) > 0
    
    if has_loss and has_acc:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] // 2 + 2, figsize[1]))
        ax1 = ax2 = ax
    
    epochs = list(range(1, len(history) + 1))
    
    if has_loss:
        for metric in loss_metrics:
            values = [h.get(metric) for h in history]
            if all(v is not None for v in values):
                label = _format_metric_label(metric)
                color = get_metric_color(metric)
                ax1.plot(epochs, values, 'o-', label=label, color=color,
                        markersize=5, markeredgecolor='white', markeredgewidth=1)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.set_title("Training Loss" if title is None else f"{title} - Loss")
    
    if has_acc:
        for metric in acc_metrics:
            values = [h.get(metric) for h in history]
            if all(v is not None for v in values):
                label = _format_metric_label(metric)
                color = get_metric_color(metric)
                ax2.plot(epochs, values, 'o-', label=label, color=color,
                        markersize=5, markeredgecolor='white', markeredgewidth=1)
        
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")
        ax2.set_title("Training Accuracy" if title is None else f"{title} - Accuracy")
        ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()

    return _save_or_show(fig, output_path, dpi, show)


def _format_metric_label(metric: str) -> str:
    replacements = {
        "loss": "Train Loss",
        "val_loss": "Val Loss",
        "accuracy": "Train Acc",
        "val_accuracy": "Val Acc",
        "val/accuracy": "Val Acc",
        "train_loss": "Train Loss",
        "train_accuracy": "Train Acc",
    }
    return replacements.get(metric, metric.replace("_", " ").title())
