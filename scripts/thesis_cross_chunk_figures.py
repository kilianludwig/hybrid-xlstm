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

import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from configs.registry import CHOMSKY_BY_HIERARCHY, MQAR_TASKS
from reporting.heatmap_data import (
    average_performance_per_model,
    build_heatmap_raw_matrix,
    build_heatmap_scaled_matrix,
    build_heatmap_solved_matrix,
    extract_early_stop_accuracy,
    heatmap_task_groups,
    is_solved,
    mean_scaled_accuracy_per_model,
)
from reporting.thesis_manifest import (
    benchmark_csv,
    entry,
    labels,
    model_of as _model_of,
    models_for,
    runs_for,
    seed_of as _seed_of,
)
from visualization.style import (
    CHUNK_SIZE_COLORS,
    CHUNK_SIZE_MARKERS,
    SCHEMATIC_BLUE_DARK,
    SCHEMATIC_STROKE,
    SEQ_LEN_COLORS,
    THESIS_HEATMAP_CMAP,
    THESIS_TEXT_HEIGHT_IN,
    THESIS_TEXT_WIDTH_IN,
    apply_paper_style,
    heatmap_figsize,
)
from visualization.utils import save_or_show

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CSV = benchmark_csv("E3_chunk")
IMAGES_DIR = REPO_ROOT / "figures/plots"

_TASK_LABELS = {
    "parity_check": "Parity",
    "even_pairs": "Even pairs",
    "cycle_navigation": "Cycle nav.",
    "modular_arithmetic": "Mod. arith.",
    "modular_arithmetic_brackets": "Mod. arith. (brackets)",
    "reverse_string": "Reverse string",
    "solve_equation": "Solve equation",
    "stack_manipulation": "Stack manip.",
    "binary_addition": "Binary add.",
    "binary_multiplication": "Binary mult.",
    "bucket_sort": "Bucket sort",
    "compute_sqrt": "Compute sqrt",
    "duplicate_string": "Duplicate string",
    "missing_duplicate_string": "Missing dup. string",
    "odds_first": "Odds first",
    "mqar": "MQAR",
}

CROSS_MODELS = models_for("E2_block_cross")
CROSS_LABELS = labels()
CROSS_LABELS["ylstm_hb"] = "yLSTM, hLSTM block"
N_YLSTM_ROWS = len(entry("E2_block_cross")["models"]["ylstm_cell"])
GROUP_REGULAR, GROUP_DCF, GROUP_CS, GROUP_MQAR = "Regular", "DCF", "CS", "MQAR"
SELECTED_TASK_GROUPS = {
    GROUP_REGULAR: [
        "parity_check",
        "even_pairs",
        "cycle_navigation",
        "modular_arithmetic",
    ],
    GROUP_DCF: [
        "modular_arithmetic_brackets",
        "solve_equation",
    ],
    GROUP_CS: ["binary_multiplication"],
    GROUP_MQAR: list(MQAR_TASKS),
}
SELECTED_TASKS = [
    task for tasks in SELECTED_TASK_GROUPS.values() for task in tasks
]
CHUNK_MODELS = models_for("E3_chunk")
CHUNK_LABELS = {m: labels()[m] for m in CHUNK_MODELS}

GROUP_TASKS = {
    "Regular": CHOMSKY_BY_HIERARCHY["regular"],
    "DCF": CHOMSKY_BY_HIERARCHY["dcf"],
    "CS": CHOMSKY_BY_HIERARCHY["cs"],
    "MQAR": MQAR_TASKS,
}
FORMAL_TASKS = [
    task
    for group in ("regular", "dcf", "cs")
    for task in CHOMSKY_BY_HIERARCHY[group]
]


def _mean_std(vals: List[float]) -> tuple:
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)


def _aggregates(runs: List[Dict[str, Any]], models: List[str]) -> Dict[str, Dict[str, tuple]]:
    runs = [r for r in runs if _model_of(r) in models]
    by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in runs:
        by_seed.setdefault(_seed_of(r), []).append(r)
    per_seed: Dict[str, Dict[str, List[float]]] = {m: {} for m in models}
    for seed_runs in by_seed.values():
        matrix, _ = build_heatmap_scaled_matrix(seed_runs)
        overall = average_performance_per_model(matrix)
        formal = mean_scaled_accuracy_per_model(matrix, FORMAL_TASKS)
        groups = {g: mean_scaled_accuracy_per_model(matrix, t) for g, t in GROUP_TASKS.items()}
        for m in models:
            if m in overall:
                per_seed[m].setdefault("Overall", []).append(overall[m])
                per_seed[m].setdefault("Formal", []).append(formal[m])
                for g in GROUP_TASKS:
                    per_seed[m].setdefault(g, []).append(groups[g][m])
    return {m: {k: _mean_std(v) for k, v in d.items()} for m, d in per_seed.items()}


def _print_aggregates(title: str, agg: Dict[str, Dict[str, tuple]]) -> None:
    cols = ["Overall", "Formal", *GROUP_TASKS]
    print(f"\n=== {title}: mean scaled accuracy +/- sample std over seeds ===")
    print(f"  {'model':14s}" + "".join(f"{c:>18s}" for c in cols))
    for m, d in agg.items():
        if not d:
            continue
        print(f"  {m:14s}" + "".join(f"{d[c][0]:9.3f} +/- {d[c][1]:.3f}" for c in cols))


def _print_per_task(title: str, runs: List[Dict[str, Any]], models: List[str]) -> Dict[str, Dict[str, tuple]]:
    runs = [r for r in runs if _model_of(r) in models]
    n_seeds = len({_seed_of(r) for r in runs})
    ddof = float(np.sqrt(n_seeds / (n_seeds - 1))) if n_seeds > 1 else 1.0
    raw, raw_std = build_heatmap_raw_matrix(runs)
    scaled, scaled_std = build_heatmap_scaled_matrix(runs)
    thr = extract_early_stop_accuracy(runs)
    tasks = [t for ts in heatmap_task_groups().values() for t in ts]
    solved = _solved_seed_counts(runs, models, thr)
    print(f"\n=== {title}: per task (scaled | raw, mean +/- sample std, solved seeds of {n_seeds}) ===")
    out: Dict[str, Dict[str, tuple]] = {}
    for m in models:
        if m not in scaled:
            continue
        print(f"\n  {m}")
        out[m] = {}
        for t in tasks:
            if t not in scaled[m]:
                continue
            s, sd = scaled[m][t], scaled_std[m].get(t, 0.0) * ddof
            rv, rsd = raw[m][t], raw_std[m].get(t, 0.0) * ddof
            out[m][t] = (s, sd)
            print(f"    {t:28s} scaled {s:.3f} +/- {sd:.3f}   raw {rv:.3f} +/- {rsd:.3f}   ({solved[m].get(t, 0)}/{n_seeds})")
    return out


def _solved_seed_counts(
    runs: List[Dict[str, Any]], models: List[str], thr: float
) -> Dict[str, Dict[str, int]]:
    solved: Dict[str, Dict[str, int]] = {m: {} for m in models}
    for r in runs:
        m = _model_of(r)
        if m not in solved:
            continue
        t = r.get("run_info", {}).get("task") or r["config"]["task"]["name"]
        solved[m][t] = solved[m].get(t, 0) + int(is_solved(r, thr))
    return solved


def _chunk_model_label(model_name: str, all_models=None) -> str:
    del all_models
    return CHUNK_LABELS[model_name]


def _write_chunk_heatmap(runs: List[Dict[str, Any]]) -> None:
    import visualization.heatmaps
    from visualization.heatmaps import plot_grouped_heatmap_solved

    saved = (
        visualization.heatmaps._format_model_label,
        visualization.heatmaps._format_task_label,
        visualization.heatmaps._MODEL_ORDER,
    )
    visualization.heatmaps._format_model_label = _chunk_model_label
    visualization.heatmaps._format_task_label = _cross_task_label
    visualization.heatmaps._MODEL_ORDER = list(CHUNK_MODELS)
    try:
        runs = [r for r in runs if _model_of(r) in set(CHUNK_MODELS)]
        n_seeds = len({_seed_of(run) for run in runs})

        scaled_results, scaled_std = build_heatmap_scaled_matrix(runs)
        ddof_factor = float(np.sqrt(n_seeds / (n_seeds - 1))) if n_seeds > 1 else 1.0
        scaled_std = {
            model: {task: value * ddof_factor for task, value in task_values.items()}
            for model, task_values in scaled_std.items()
        }
        early_stop = extract_early_stop_accuracy(runs)
        solved, solved_counts = build_heatmap_solved_matrix(runs, early_stop)

        fig = plot_grouped_heatmap_solved(
            scaled_results,
            GROUP_TASKS,
            threshold=1.0,
            std_results=scaled_std,
            title=None,
            cmap=THESIS_HEATMAP_CMAP,
            colorbar_label="Scaled accuracy",
            solved_results=solved,
            annotation_suffix=solved_counts,
            figsize=heatmap_figsize(len(CHUNK_MODELS), THESIS_TEXT_HEIGHT_IN, 30),
            annotation_fontsize=6,
            check_fontsize=12,
            show_colorbar=True,
            boundary_color=SCHEMATIC_STROKE,
            tick_rotation=30,
        )
        fig.axes[0].set_xlabel("")
        fig.axes[0].set_ylabel("")
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        output = IMAGES_DIR / "chunk_full_scaled_accuracy_heatmap.pdf"
        save_or_show(fig, output)
        print(f"  Wrote {output}")
    finally:
        (
            visualization.heatmaps._format_model_label,
            visualization.heatmaps._format_task_label,
            visualization.heatmaps._MODEL_ORDER,
        ) = saved


def _load_benchmark() -> List[Dict[str, Any]]:
    with open(BENCHMARK_CSV) as f:
        return list(csv.DictReader(f))


_SEQ_LEN_MARKERS = ["o", "s", "^", "v", "D", "P"]
_SEQ_LEN_OFFSET_STEP = 0.08
_SEQ_LEN_MARKER_SIZE = 5
_SEQ_LEN_ALPHA = 0.55

def _thousand_token_linear_axis(
    ax, values: List[float], tick_step: float
) -> None:
    upper = tick_step * np.ceil((max(values) * 1.08) / tick_step)
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))


def _fig_throughput(rows: List[Dict[str, Any]]) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(THESIS_TEXT_WIDTH_IN, 3.0), sharex=True)
    seq_lens = sorted({int(r["seq_len"]) for r in rows})
    x_of = {m: i for i, m in enumerate(CHUNK_MODELS)}
    center = (len(seq_lens) - 1) / 2
    offset_of = {
        n: (i - center) * _SEQ_LEN_OFFSET_STEP for i, n in enumerate(seq_lens)
    }
    style_of = {
        n: (SEQ_LEN_COLORS[n], _SEQ_LEN_MARKERS[i])
        for i, n in enumerate(seq_lens)
    }
    marker_pad = 0.22
    x_pad = abs(offset_of[seq_lens[0]]) + marker_pad
    for ax, col, title, tick_step in zip(
        axes,
        ("inference_tokens_per_second", "train_tokens_per_second"),
        ("Inference", "Training step"),
        (200, 100),
    ):
        plotted: List[float] = []
        for seq_len in seq_lens:
            color, marker = style_of[seq_len]
            measured = []
            for m in CHUNK_MODELS:
                match = next(
                    (r for r in rows if r["model"] == m and int(r["seq_len"]) == seq_len),
                    None,
                )
                if match is not None and match[col]:
                    measured.append((x_of[m], float(match[col]) / 1000.0))
            measured.sort()
            start = 0
            labeled = False
            for i in range(1, len(measured) + 1):
                gap = i < len(measured) and measured[i][0] != measured[i - 1][0] + 1
                if i < len(measured) and not gap:
                    continue
                xs = [p[0] + offset_of[seq_len] for p in measured[start:i]]
                ys = [p[1] for p in measured[start:i]]
                plotted.extend(ys)
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    color=color,
                    alpha=_SEQ_LEN_ALPHA,
                    markersize=_SEQ_LEN_MARKER_SIZE,
                    markeredgewidth=0.6,
                    label=(rf"$T = {seq_len}$" if not labeled else None),
                )
                labeled = True
                start = i
        _thousand_token_linear_axis(ax, plotted, tick_step)
        ax.set_xticks(range(len(CHUNK_MODELS)))
        ax.set_xticklabels([CHUNK_LABELS[m] for m in CHUNK_MODELS])
        ax.set_xlabel(r"Chunk size $\ell$")
        ax.set_xlim(-x_pad, len(CHUNK_MODELS) - 1 + x_pad)
        ax.set_title(title)
    axes[0].set_ylabel("Thousand tokens per second")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(seq_lens),
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    save_or_show(fig, IMAGES_DIR / "chunk_throughput.pdf")


def _fig_pareto(agg: Dict[str, Dict[str, tuple]], rows: List[Dict[str, Any]]) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(THESIS_TEXT_WIDTH_IN, 3.2))
    xs, ys = [], []
    for m in CHUNK_MODELS:
        r = next(r for r in rows if r["model"] == m and r["seq_len"] == "16")
        tps = float(r["inference_tokens_per_second"]) / 1000.0
        mu, sd = agg[m]["Formal"]
        xs.append(tps)
        ys.append(mu)
        ax.errorbar(tps, mu, yerr=sd, fmt=CHUNK_SIZE_MARKERS[m], color=CHUNK_SIZE_COLORS[m], capsize=4, markersize=8, label=CHUNK_LABELS[m])
    ax.plot(xs, ys, color=SCHEMATIC_STROKE, linewidth=1, zorder=0)
    ax.set_xlabel("Inference throughput at backbone length $T = 16$ (k tokens/s)")
    ax.set_ylabel("Mean scaled formal-language accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    save_or_show(fig, IMAGES_DIR / "chunk_pareto.pdf")


def _fig_chunk_tasks(per_task: Dict[str, Dict[str, tuple]]) -> None:
    from matplotlib.colors import to_rgba

    apply_paper_style()
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(THESIS_TEXT_WIDTH_IN, 5.6),
        sharex=True,
        sharey=True,
    )
    xs = list(range(len(CHUNK_MODELS)))
    tick_labels = [CHUNK_LABELS[m] for m in CHUNK_MODELS]
    y_lo, y_hi = 0.0, 1.0
    series = []
    for task in SELECTED_TASKS:
        ys, sds = [], []
        for model in CHUNK_MODELS:
            mu, sd = per_task[model][task]
            ys.append(mu)
            sds.append(sd)
        y_lo = min(y_lo, *(y - s for y, s in zip(ys, sds)))
        y_hi = max(y_hi, *(y + s for y, s in zip(ys, sds)))
        series.append((ys, sds))
    for ax, task, (ys, sds) in zip(axes.ravel(), SELECTED_TASKS, series):
        ax.errorbar(
            xs,
            ys,
            yerr=sds,
            fmt="none",
            ecolor=to_rgba(SCHEMATIC_BLUE_DARK, 0.45),
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            color=SCHEMATIC_BLUE_DARK,
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            zorder=3,
        )
        ax.set_title(_TASK_LABELS[task], fontsize=9, pad=3)
        ax.set_xlim(-0.35, len(CHUNK_MODELS) - 0.65)
        ax.set_ylim(y_lo - 0.04, y_hi + 0.04)
    for ax in axes[-1, :]:
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(r"Chunk size $\ell$")
    fig.supylabel("Mean scaled accuracy")
    fig.tight_layout()
    save_or_show(fig, IMAGES_DIR / "chunk_tasks.pdf")


def _cross_model_label(model_name: str, all_models=None) -> str:
    del all_models
    return CROSS_LABELS[model_name]


def _cross_task_label(task_name: str) -> str:
    return _TASK_LABELS.get(task_name, task_name)


def _write_cross_heatmaps(runs: List[Dict[str, Any]]) -> None:
    import visualization.heatmaps
    from visualization.heatmaps import plot_grouped_heatmap_solved

    saved = (
        visualization.heatmaps._format_model_label,
        visualization.heatmaps._format_task_label,
        visualization.heatmaps._MODEL_ORDER,
    )
    visualization.heatmaps._format_model_label = _cross_model_label
    visualization.heatmaps._format_task_label = _cross_task_label
    visualization.heatmaps._MODEL_ORDER = list(CROSS_MODELS)
    try:
        runs = [r for r in runs if _model_of(r) in set(CROSS_MODELS)]
        n_seeds = len({_seed_of(run) for run in runs})

        scaled_results, scaled_std = build_heatmap_scaled_matrix(runs)
        ddof_factor = float(np.sqrt(n_seeds / (n_seeds - 1))) if n_seeds > 1 else 1.0
        scaled_std = {
            model: {task: value * ddof_factor for task, value in task_values.items()}
            for model, task_values in scaled_std.items()
        }
        early_stop = extract_early_stop_accuracy(runs)
        solved, solved_counts = build_heatmap_solved_matrix(runs, early_stop)

        def _draw(task_groups, figsize, annotation_fontsize, check_fontsize, name, tick_rotation=45):
            fig = plot_grouped_heatmap_solved(
                scaled_results,
                task_groups,
                threshold=1.0,
                std_results=scaled_std,
                title=None,
                cmap=THESIS_HEATMAP_CMAP,
                colorbar_label="Scaled accuracy",
                solved_results=solved,
                annotation_suffix=solved_counts,
                figsize=figsize,
                annotation_fontsize=annotation_fontsize,
                check_fontsize=check_fontsize,
                show_colorbar=True,
                boundary_color=SCHEMATIC_STROKE,
                tick_rotation=tick_rotation,
            )
            fig.axes[0].axhline(
                y=N_YLSTM_ROWS - 0.5, color=SCHEMATIC_STROKE, linewidth=1.2
            )
            fig.axes[0].set_xlabel("")
            fig.axes[0].set_ylabel("")
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            output = IMAGES_DIR / name
            save_or_show(fig, output)
            print(f"  Wrote {output}")

        _draw(
            SELECTED_TASK_GROUPS,
            heatmap_figsize(len(CROSS_MODELS), THESIS_TEXT_WIDTH_IN, 45),
            8,
            14,
            "cross_selected_scaled_accuracy_heatmap.pdf",
        )
        _draw(
            GROUP_TASKS,
            heatmap_figsize(len(CROSS_MODELS), THESIS_TEXT_HEIGHT_IN, 30),
            6,
            12,
            "cross_full_scaled_accuracy_heatmap.pdf",
            tick_rotation=30,
        )
    finally:
        (
            visualization.heatmaps._format_model_label,
            visualization.heatmaps._format_task_label,
            visualization.heatmaps._MODEL_ORDER,
        ) = saved


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    cross_runs = runs_for("E2_block_cross")
    cross_agg = _aggregates(cross_runs, CROSS_MODELS)
    _print_aggregates("Block cross", cross_agg)
    _print_per_task("Block cross", cross_runs, CROSS_MODELS)
    _write_cross_heatmaps(cross_runs)

    chunk_runs = runs_for("E3_chunk")
    chunk_agg = _aggregates(chunk_runs, CHUNK_MODELS)
    _print_aggregates("Chunk sweep", chunk_agg)
    chunk_per_task = _print_per_task("Chunk sweep", chunk_runs, CHUNK_MODELS)
    _write_chunk_heatmap(chunk_runs)
    _fig_chunk_tasks(chunk_per_task)

    rows = _load_benchmark()
    missing = [m for m in CHUNK_MODELS if not any(r["model"] == m for r in rows)]
    if missing:
        raise SystemExit(f"{BENCHMARK_CSV} has no rows for {missing}; point the manifest at a benchmark with all chunk arms")
    print(f"\n=== Benchmark at T = 16 (tokens per second), {BENCHMARK_CSV.relative_to(REPO_ROOT)} ===")
    for m in CHUNK_MODELS:
        r = next(r for r in rows if r["model"] == m and r["seq_len"] == "16")
        print(f"  {m:12s} inference {float(r['inference_tokens_per_second']):9.0f}   train {float(r['train_tokens_per_second']):9.0f}")

    _fig_throughput(rows)
    _fig_pareto(chunk_agg, rows)
    print(f"\nWrote chunk figures to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
