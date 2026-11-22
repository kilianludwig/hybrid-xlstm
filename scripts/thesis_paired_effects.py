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

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from reporting.heatmap_data import build_heatmap_scaled_matrix
from reporting.thesis_manifest import labels, load_manifest, model_of, runs_for, seed_of
from scripts.thesis_headline_figures import FULL_TASK_GROUPS, _task_label
from visualization.style import (
    SCHEMATIC_BLUE_DARK, SCHEMATIC_BLUE_FILL, SCHEMATIC_STROKE,
    THESIS_TEXT_WIDTH_IN, apply_paper_style,
)

PLOTS = Path(__file__).resolve().parents[1] / "figures" / "plots"
TABLES = Path(__file__).resolve().parents[1] / "figures" / "tables"
TASKS = [task for group in FULL_TASK_GROUPS.values() for task in group]
SEEDS = sorted(load_manifest()["seeds"])

Side = Tuple[str, str]
COMPARISONS: List[Tuple[str, List[Tuple[Side, Side]]]] = [
    (
        "Block architectures",
        [
            (("E1_headline", "hlstm"), ("E1_headline", "xlstm")),
            (("E1_headline", "hlstm"), ("E1_headline", "xlstm_hb")),
            (("E1_headline", "ylstm_hb"), ("E1_headline", "xlstm")),
            (("E1_headline", "ylstm_hb"), ("E1_headline", "xlstm_hb")),
            (("E1_headline", "hlstm"), ("E1_headline", "ylstm_hb")),
            (("E1_headline", "slstm"), ("E1_headline", "xlstm")),
        ],
    ),
    (
        "Bare cell stacks",
        [
            (("E1_cell_stacks", "ylstm"), ("E1_cell_stacks", "mlstm_bare")),
            (("E1_cell_stacks", "ylstm"), ("E1_cell_stacks", "slstm_bare")),
            (("E1_cell_stacks", "slstm_bare"), ("E1_cell_stacks", "mlstm_bare")),
            (("E1_headline", "ylstm_hb"), ("E1_cell_stacks", "ylstm")),
            (("E2_block_cross", "mlstm_hb"), ("E1_cell_stacks", "mlstm_bare")),
            (("E2_block_cross", "slstm_hb"), ("E1_cell_stacks", "slstm_bare")),
        ],
    ),
    (
        "Block comparison",
        [
            (("E2_block_cross", "ylstm_hb"), ("E2_block_cross", "ylstm_sb")),
            (("E2_block_cross", "ylstm_hb"), ("E2_block_cross", "ylstm_mb")),
            (("E2_block_cross", "ylstm_hb"), ("E2_block_cross", "slstm_hb")),
            (("E2_block_cross", "ylstm_hb"), ("E2_block_cross", "mlstm_hb")),
        ],
    ),
    (
        "Chunk size",
        [
            (("E3_chunk", "ylstm_hb_c1"), ("E3_chunk", "ylstm_hb_c2")),
            (("E3_chunk", "ylstm_hb_c1"), ("E3_chunk", "ylstm_hb_c4")),
            (("E3_chunk", "ylstm_hb_c1"), ("E3_chunk", "ylstm_hb_c8")),
            (("E3_chunk", "ylstm_hb_c1"), ("E3_chunk", "ylstm_hb_c0")),
            (("E3_chunk", "ylstm_hb_c8"), ("E3_chunk", "ylstm_hb_c0")),
        ],
    ),
]


@lru_cache(maxsize=None)
def scores(experiment):
    values = {}
    for run in runs_for(experiment):
        model, seed, task = model_of(run), seed_of(run), run["task"]
        key = (model, seed, task)
        if key in values or run["best"].get("test_accuracy") is None:
            raise ValueError(f"Duplicate or missing held-out score for {experiment}, {key}")
        matrix, _ = build_heatmap_scaled_matrix([run])
        value = matrix[model][task]
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite score for {experiment}, {key}")
        values[key] = value
    return values


def score_grid(side):
    experiment, model = side
    data = scores(experiment)
    return np.array([[data[model, seed, task] for task in TASKS] for seed in SEEDS])


def comparison_label(side):
    model = side[1]
    if model.startswith("ylstm_hb_c"):
        return labels()[model]
    symbols = {
        "hlstm": "h", "xlstm": "x", "slstm": "s", "xlstm_hb": "x_h",
        "ylstm_hb": "y_h", "mlstm_hb": "m_h", "slstm_hb": "s_h",
        "ylstm_sb": "y_s", "ylstm_mb": "y_m", "ylstm": "y_b",
        "mlstm_bare": "m_b", "slstm_bare": "s_b",
    }
    return f"${symbols[model]}$"


def task_heatmap(rows, contrasts, group_starts):
    fig, ax = plt.subplots(figsize=(8.2, 6.1), layout="constrained")
    data = np.stack([delta.mean(axis=0) for delta in rows])
    im = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_yticks(range(len(rows)), contrasts, fontsize=9)
    ax.set_xticks(range(len(TASKS)), [_task_label(task) for task in TASKS],
                  rotation=48, ha="right", rotation_mode="anchor", fontsize=9)
    for start in group_starts[1:]:
        ax.axhline(start - 0.5, color="black", linewidth=1)
    for boundary in np.cumsum([len(tasks) for tasks in FULL_TASK_GROUPS.values()])[:-1]:
        ax.axvline(boundary - 0.5, color="0.6", linewidth=0.6)
    bar = fig.colorbar(im, ax=ax, shrink=0.8)
    bar.set_label("Mean paired difference in scaled accuracy")
    fig.savefig(PLOTS / "paired_task_differences.pdf")
    plt.close(fig)


def paired_panel(ax, delta, left_label, right_label):
    values = np.column_stack([delta.mean(axis=1), delta])
    y = np.arange(values.shape[1])
    ax.axvline(0, color=SCHEMATIC_STROKE, lw=0.8)
    ax.axhspan(-0.5, 0.5, color=SCHEMATIC_BLUE_FILL)
    for seed, offset in zip(values, np.linspace(-0.28, 0.28, len(SEEDS))):
        ax.scatter(seed, y + offset, s=9, color="0.45", linewidths=0)
    ax.scatter(values.mean(axis=0), y, marker="D", s=20, color=SCHEMATIC_BLUE_DARK)
    ax.set_yticks(y, ["Sixteen-task mean"] + [_task_label(task) for task in TASKS])
    ax.set_ylim(len(y) - 0.45, -0.55)
    ax.set_xlim(-1.1, 1.1)
    ax.tick_params(labelsize=8)
    ax.set_title(f"← {left_label} higher", loc="left", fontsize=8)
    ax.set_title(f"{right_label} higher →", loc="right", fontsize=8)
    ax.yaxis.grid(False)


def feedback_plot(delta):
    fig, ax = plt.subplots(figsize=(THESIS_TEXT_WIDTH_IN, 4.5), layout="constrained")
    paired_panel(ax, delta, "mLSTM", "yLSTM")
    ax.set_xlabel("Paired difference in scaled test accuracy (yLSTM − mLSTM)")
    fig.savefig(PLOTS / "paired_feedback.pdf")
    plt.close(fig)


def headline_control_plot():
    control = score_grid(("E1_headline", "xlstm_hb"))
    fig, axes = plt.subplots(1, 2, figsize=(THESIS_TEXT_WIDTH_IN, 4.6), sharex=True, sharey=True,
                             layout="constrained")
    for ax, model, title in zip(axes, ("hlstm", "ylstm_hb"), ("hLSTM", "yLSTM")):
        paired_panel(ax, score_grid(("E1_headline", model)) - control, "Control", title)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["−1", "−0.5", "0", "0.5", "1"])
    fig.supxlabel("Paired difference in scaled test accuracy", fontsize=10)
    fig.savefig(PLOTS / "paired_headline_control.pdf")
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    rows, contrasts, group_starts = [], [], []
    table = [r"% Generated by scripts/thesis_paired_effects.py",
             r"\begin{tabular}{rllrrrrrrr}\toprule",
             r"No. & First & Second & $d_1$ & $d_2$ & $d_3$ & $d_4$ & $d_5$ & Mean & $d_s>0$ \\"]
    for group, comparisons in COMPARISONS:
        group_starts.append(len(rows))
        table.append(r"\midrule\multicolumn{10}{l}{\textit{" + group + r"}} \\")
        for first, second in comparisons:
            delta = score_grid(first) - score_grid(second)
            rows.append(delta)
            paired = delta.mean(axis=1)
            fields = [str(len(rows)), comparison_label(first), comparison_label(second)]
            contrasts.append(f"{len(rows)}.  {fields[1]} − {fields[2]}")
            fields.extend(f"${value:+.3f}$" for value in [*paired, paired.mean()])
            fields.append(str(np.count_nonzero(paired > 0)))
            table.append(" & ".join(fields) + r" \\")
            if (first[1], second[1]) == ("ylstm_hb", "mlstm_hb"):
                feedback_plot(delta)
    table.append(r"\bottomrule\end{tabular}")
    (TABLES / "paired_effects.tex").write_text("\n".join(table) + "\n")
    task_heatmap(rows, contrasts, group_starts)
    headline_control_plot()


if __name__ == "__main__":
    main()
