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
from typing import Any, Dict, List, Optional

import numpy as np

from configs.registry import CHOMSKY_BY_HIERARCHY, MQAR_TASKS
from reporting.heatmap_data import (
    average_performance_per_model,
    build_heatmap_epoch_matrix,
    build_heatmap_raw_matrix,
    build_heatmap_scaled_matrix,
    build_heatmap_solved_matrix,
    extract_early_stop_accuracy,
    mean_scaled_accuracy_per_model,
)
from reporting.loading import group_mamba_variants
from reporting.thesis_manifest import entry, labels, runs_for, seed_of as _seed_of
import visualization.comparison
import visualization.heatmaps
from visualization.heatmaps import (
    _sort_models as _canonical_sort_models,
    plot_grouped_heatmap,
    plot_grouped_heatmap_solved,
    plot_grouped_heatmap_solved_epochs,
)
from visualization.comparison import plot_bar_comparison
from visualization import style
from visualization.style import (
    SCHEMATIC_BLUE,
    SCHEMATIC_STROKE,
    THESIS_HEATMAP_CMAP,
    THESIS_TEXT_HEIGHT_IN,
    THESIS_TEXT_WIDTH_IN,
    heatmap_figsize,
)
from visualization.utils import save_or_show

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "E1_headline"
APPENDIX_EXPERIMENT = "E1_yhlstm_appendix"
IMAGES_DIR = REPO_ROOT / "figures/plots"
TABLES_DIR = REPO_ROOT / "figures/tables"

_COMPACT_LABELS = {
    m: labels()[m]
    for m in (
        list(entry(EXPERIMENT)["models"]["contributed"])
        + list(entry(APPENDIX_EXPERIMENT)["models"]["exploratory"])
    )
}

PUBLISHED = list(entry(EXPERIMENT)["models"]["published"])
N_PUBLISHED = len(PUBLISHED)
CORE_MODELS = PUBLISHED + list(entry(EXPERIMENT)["models"]["contributed"])
APPENDIX_MODELS = list(entry(APPENDIX_EXPERIMENT)["models"]["exploratory"])
N_CORE = len(CORE_MODELS)
N_COMPLETE = N_CORE + len(APPENDIX_MODELS)
CORE_MODEL_ORDER = _canonical_sort_models(CORE_MODELS)
COMPLETE_MODEL_ORDER = CORE_MODEL_ORDER + APPENDIX_MODELS
SEPARATOR_COLOR = SCHEMATIC_STROKE
BAR_COLOR_PUBLISHED = SCHEMATIC_STROKE
BAR_COLOR_CONTRIBUTED = SCHEMATIC_BLUE

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
FULL_TASK_GROUPS = {
    GROUP_REGULAR: CHOMSKY_BY_HIERARCHY["regular"],
    GROUP_DCF: CHOMSKY_BY_HIERARCHY["dcf"],
    GROUP_CS: CHOMSKY_BY_HIERARCHY["cs"],
    GROUP_MQAR: list(MQAR_TASKS),
}

_NUMBER_WORDS = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def _compact_label(model_name: str, all_models=None) -> str:
    return _COMPACT_LABELS.get(
        model_name, style.format_model_label(model_name, all_models)
    )


def _bar_label(model_name: str, all_models=None) -> str:
    if model_name == "xlstm_hb":
        return "Layer-alternating control"
    return _compact_label(model_name, all_models)


def _headline_sort_models(model_names: List[str]) -> List[str]:
    order = {model: i for i, model in enumerate(COMPLETE_MODEL_ORDER)}
    return sorted(model_names, key=lambda model: order.get(model, len(order)))


def _bar_color(model_name: str) -> str:
    return BAR_COLOR_PUBLISHED if model_name in PUBLISHED else BAR_COLOR_CONTRIBUTED


def _strip_axis_titles(fig) -> None:
    fig.axes[0].set_xlabel("")
    fig.axes[0].set_ylabel("")


def _mean_std_over_seeds(
    per_seed: Dict[int, Dict[str, float]],
) -> tuple:
    models = sorted({m for vals in per_seed.values() for m in vals})
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for m in models:
        vals = np.array([per_seed[s][m] for s in per_seed if m in per_seed[s]])
        means[m] = float(np.mean(vals))
        stds[m] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return means, stds


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


def _task_label(task_name: str) -> str:
    return _TASK_LABELS.get(task_name, task_name)


def _write_aggregate_table(
    filepath: Path,
    *,
    label: str,
    caption_short: str,
    caption_long: str,
    means: Dict[str, float],
    stds: Dict[str, float],
    model_order: List[str],
    solved: Dict[str, Dict[str, float]],
    group_stats: Optional[List[
        tuple[str, Dict[str, float], Dict[str, float]]
    ]] = None,
) -> None:
    group_stats = group_stats or []
    existing = sorted(
        model_order[:N_PUBLISHED], key=lambda model: means[model], reverse=True
    )
    contributed = sorted(
        model_order[N_PUBLISHED:], key=lambda model: means[model], reverse=True
    )
    n_metrics = 1 + len(group_stats)
    total_cols = n_metrics + 2

    lines = [
        r"% Generated by scripts/thesis_headline_figures.py. Do not edit by hand.",
        r"\begin{table}[!htbp]",
        r"    \centering",
        f"    \\caption[{caption_short}]{{{caption_long}}}",
        f"    \\label{{{label}}}",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{2pt}",
        f"    \\begin{{tabularx}}{{\\linewidth}}{{l*{{{n_metrics}}}"
        r"{>{\centering\arraybackslash}X}c}",
        r"        \toprule",
    ]
    headers = ["", "Overall"] + [name for name, _, _ in group_stats]
    headers.append(r"\shortstack{Tasks\\solved}")
    lines.append("        " + " & ".join(headers) + r" \\")
    lines.append(r"        \midrule")

    def add_group(title: str, models: List[str]) -> None:
        lines.append(
            f"        \\multicolumn{{{total_cols}}}{{l}}"
            f"{{\\emph{{{title}}}}} \\\\"
        )
        for model in models:
            model_label = _compact_label(model)
            if ", " in model_label:
                model_name, block = model_label.split(", ", 1)
                model_label = rf"\shortstack[l]{{{model_name},\\{block}}}"
            row = [
                model_label,
                f"${means[model]:.3f}\\,{{\\pm}}\\,{stds[model]:.3f}$",
            ]
            for _, group_means, group_stds in group_stats:
                row.append(
                    f"${group_means[model]:.3f}\\,{{\\pm}}\\,"
                    f"{group_stds[model]:.3f}$"
                )
            tasks = solved[model]
            row.append(f"{sum(value == 1.0 for value in tasks.values())}/{len(tasks)}")
            lines.append("        " + " & ".join(row) + r" \\")

    add_group("Published block architectures", existing)
    lines.append(r"        \midrule")
    add_group(
        "Architectural contributions and the layer-alternating control",
        contributed,
    )
    lines += [
        r"        \bottomrule",
        r"    \end{tabularx}",
        r"\end{table}",
    ]

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {filepath}")


def _write_exploratory_table(
    filepath: Path,
    *,
    means: Dict[str, float],
    stds: Dict[str, float],
    group_stats: List[tuple[str, Dict[str, float], Dict[str, float]]],
) -> None:
    model = APPENDIX_MODELS[0]
    headers = ["Architecture", "Overall"] + [
        name for name, _group_means, _group_stds in group_stats
    ]
    row = [
        _compact_label(model),
        f"${means[model]:.3f} \\pm {stds[model]:.3f}$",
    ]
    for _name, group_means, group_stds in group_stats:
        row.append(
            f"${group_means[model]:.3f} \\pm {group_stds[model]:.3f}$"
        )
    lines = [
        r"% Generated by scripts/thesis_headline_figures.py. Do not edit by hand.",
        r"\begin{tabularx}{\linewidth}{l*{5}{>{\centering\arraybackslash}X}}",
        r"    \toprule",
        "    " + " & ".join(headers) + r" \\",
        r"    \midrule",
        "    " + " & ".join(row) + r" \\",
        r"    \bottomrule",
        r"\end{tabularx}",
    ]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {filepath}")


def main() -> None:
    visualization.comparison._format_model_label = _bar_label
    visualization.comparison.get_model_color = _bar_color
    visualization.heatmaps._format_model_label = _compact_label
    visualization.heatmaps._format_task_label = _task_label
    visualization.heatmaps._sort_models = _headline_sort_models

    runs = runs_for(EXPERIMENT)
    appendix_runs = runs_for(APPENDIX_EXPERIMENT)
    complete_runs = runs + appendix_runs
    if not runs or not appendix_runs:
        raise SystemExit("No runs found")
    group_mamba_variants(complete_runs)

    runs_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in runs:
        runs_by_seed.setdefault(_seed_of(r), []).append(r)
    complete_runs_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in complete_runs:
        complete_runs_by_seed.setdefault(_seed_of(r), []).append(r)
    appendix_runs_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in appendix_runs:
        appendix_runs_by_seed.setdefault(_seed_of(r), []).append(r)
    print(
        f"Loaded {len(runs)} core runs and {len(appendix_runs)} exploratory "
        f"runs across seeds {sorted(complete_runs_by_seed)}"
    )
    n_seeds_word = _NUMBER_WORDS.get(len(runs_by_seed), str(len(runs_by_seed)))

    group_tasks = {
        "type3": CHOMSKY_BY_HIERARCHY["regular"],
        "type2": CHOMSKY_BY_HIERARCHY["dcf"],
        "type1": CHOMSKY_BY_HIERARCHY["cs"],
        "mqar": MQAR_TASKS,
    }

    per_seed_full: Dict[int, Dict[str, float]] = {}
    per_seed_selected: Dict[int, Dict[str, float]] = {}
    per_seed_group: Dict[str, Dict[int, Dict[str, float]]] = {
        k: {} for k in group_tasks
    }
    for seed, seed_runs in runs_by_seed.items():
        matrix, _ = build_heatmap_scaled_matrix(seed_runs)
        per_seed_full[seed] = average_performance_per_model(matrix)
        per_seed_selected[seed] = mean_scaled_accuracy_per_model(
            matrix, SELECTED_TASKS
        )
        for key, tasks in group_tasks.items():
            per_seed_group[key][seed] = mean_scaled_accuracy_per_model(
                matrix, tasks
            )

    per_seed_appendix_full: Dict[int, Dict[str, float]] = {}
    per_seed_appendix_group: Dict[str, Dict[int, Dict[str, float]]] = {
        k: {} for k in group_tasks
    }
    for seed, seed_runs in appendix_runs_by_seed.items():
        matrix, _ = build_heatmap_scaled_matrix(seed_runs)
        per_seed_appendix_full[seed] = average_performance_per_model(matrix)
        for key, tasks in group_tasks.items():
            per_seed_appendix_group[key][seed] = (
                mean_scaled_accuracy_per_model(matrix, tasks)
            )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    full_means, full_stds = _mean_std_over_seeds(per_seed_full)
    selected_means, selected_stds = _mean_std_over_seeds(per_seed_selected)
    print("\n=== Complete 16-task mean scaled accuracy +/- sample std ===")
    for m in sorted(full_means, key=full_means.get, reverse=True):
        print(f"  {m:12s} {full_means[m]:.3f} +/- {full_stds[m]:.3f}")
    print("\n=== Eight displayed tasks, mean scaled accuracy (display subset only) ===")
    for m in sorted(selected_means, key=selected_means.get, reverse=True):
        print(
            f"  {m:12s} {selected_means[m]:.3f} "
            f"+/- {selected_stds[m]:.3f}"
        )

    plot_bar_comparison(
        full_means,
        output_path=IMAGES_DIR / "headline_average_performance.pdf",
        title=None,
        ylabel="Mean scaled accuracy",
        yerr=full_stds,
        figsize=(THESIS_TEXT_WIDTH_IN, 3.2),
    )

    group_labels = {
        "type3": GROUP_REGULAR,
        "type2": GROUP_DCF,
        "type1": GROUP_CS,
        "mqar": GROUP_MQAR,
    }
    full_group_stats = []
    for key in ("type3", "type2", "type1", "mqar"):
        g_means, g_stds = _mean_std_over_seeds(per_seed_group[key])
        full_group_stats.append((group_labels[key], g_means, g_stds))
        print(f"\n=== {group_labels[key]} ===")
        for m in sorted(g_means, key=g_means.get, reverse=True):
            print(f"  {m:12s} {g_means[m]:.3f} +/- {g_stds[m]:.3f}")

    appendix_means, appendix_stds = _mean_std_over_seeds(
        per_seed_appendix_full
    )
    appendix_group_stats = []
    for key in ("type3", "type2", "type1", "mqar"):
        g_means, g_stds = _mean_std_over_seeds(
            per_seed_appendix_group[key]
        )
        appendix_group_stats.append((group_labels[key], g_means, g_stds))

    n_seeds = len(runs_by_seed)
    ddof_factor = float(np.sqrt(n_seeds / (n_seeds - 1)))

    def _to_sample_std(std: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        return {m: {t: v * ddof_factor for t, v in d.items()} for m, d in std.items()}

    raw_results_complete, raw_std_complete = build_heatmap_raw_matrix(
        complete_runs
    )
    scaled_results_complete, scaled_std_complete = build_heatmap_scaled_matrix(
        complete_runs
    )
    raw_std_complete = _to_sample_std(raw_std_complete)
    scaled_std_complete = _to_sample_std(scaled_std_complete)
    raw_results = {m: raw_results_complete[m] for m in CORE_MODELS}
    scaled_results = {m: scaled_results_complete[m] for m in CORE_MODELS}
    raw_std = {m: raw_std_complete[m] for m in CORE_MODELS}
    scaled_std = {m: scaled_std_complete[m] for m in CORE_MODELS}
    early_stop = extract_early_stop_accuracy(complete_runs)
    all_tasks_flat = [t for tasks in FULL_TASK_GROUPS.values() for t in tasks]

    solved_complete, solved_counts_complete = build_heatmap_solved_matrix(
        complete_runs, early_stop
    )
    solved = {m: solved_complete[m] for m in CORE_MODELS}
    solved_counts = {m: solved_counts_complete[m] for m in CORE_MODELS}

    full_tick_rotation = 30
    full_figsize = heatmap_figsize(
        N_COMPLETE, THESIS_TEXT_HEIGHT_IN, full_tick_rotation,
    )
    selected_figsize = heatmap_figsize(N_CORE, THESIS_TEXT_WIDTH_IN, 45)

    fig = plot_grouped_heatmap(
        raw_results_complete,
        FULL_TASK_GROUPS,
        std_results=raw_std_complete,
        title=None,
        highlight_best=True,
        annotate_std=True,
        cmap=THESIS_HEATMAP_CMAP,
        colorbar_label="Raw accuracy",
        figsize=full_figsize,
        annotation_fontsize=7,
        show_colorbar=True,
        boundary_color=SEPARATOR_COLOR,
        tick_rotation=full_tick_rotation,
    )
    fig.axes[0].axhline(
        y=N_PUBLISHED - 0.5, color=SEPARATOR_COLOR, linewidth=1.2
    )
    fig.axes[0].axhline(
        y=N_CORE - 0.5, color=SEPARATOR_COLOR, linewidth=1.2
    )
    _strip_axis_titles(fig)
    save_or_show(fig, IMAGES_DIR / "headline_full_raw_accuracy_heatmap.pdf")

    fig = plot_grouped_heatmap_solved(
        scaled_results_complete,
        FULL_TASK_GROUPS,
        threshold=1.0,
        std_results=scaled_std_complete,
        title=None,
        cmap=THESIS_HEATMAP_CMAP,
        colorbar_label="Scaled accuracy",
        solved_results=solved_complete,
        annotation_suffix=solved_counts_complete,
        figsize=full_figsize,
        annotation_fontsize=6,
        check_fontsize=12,
        linespacing=0.9,
        show_colorbar=True,
        boundary_color=SEPARATOR_COLOR,
        tick_rotation=full_tick_rotation,
    )
    fig.axes[0].axhline(
        y=N_PUBLISHED - 0.5, color=SEPARATOR_COLOR, linewidth=1.2
    )
    fig.axes[0].axhline(
        y=N_CORE - 0.5, color=SEPARATOR_COLOR, linewidth=1.2
    )
    _strip_axis_titles(fig)
    save_or_show(fig, IMAGES_DIR / "headline_full_scaled_accuracy_heatmap.pdf")

    fig = plot_grouped_heatmap_solved(
        scaled_results,
        SELECTED_TASK_GROUPS,
        threshold=1.0,
        std_results=scaled_std,
        title=None,
        cmap=THESIS_HEATMAP_CMAP,
        colorbar_label="Scaled accuracy",
        solved_results=solved,
        annotation_suffix=solved_counts,
        figsize=selected_figsize,
        annotation_fontsize=8,
        check_fontsize=14,
        show_colorbar=True,
        boundary_color=SEPARATOR_COLOR,
    )
    fig.axes[0].axhline(
        y=N_PUBLISHED - 0.5, color=SEPARATOR_COLOR, linewidth=1.2
    )
    _strip_axis_titles(fig)
    save_or_show(
        fig, IMAGES_DIR / "headline_selected_scaled_accuracy_heatmap.pdf"
    )

    epoch_results_complete, epoch_std_complete = build_heatmap_epoch_matrix(
        complete_runs
    )
    epoch_std_complete = _to_sample_std(epoch_std_complete)
    epoch_results = {m: epoch_results_complete[m] for m in CORE_MODELS}
    epoch_std = {m: epoch_std_complete[m] for m in CORE_MODELS}
    print("\n=== Epochs to threshold (cells solved on every seed) ===")
    for model in _headline_sort_models(list(epoch_results.keys())):
        for task in all_tasks_flat:
            if solved.get(model, {}).get(task, 0.0) < 1.0:
                continue
            ep = epoch_results[model].get(task)
            sd = epoch_std.get(model, {}).get(task, 0.0)
            if ep is None:
                continue
            print(f"  {model:12s} {task:28s} {ep:.1f} +/- {sd:.1f}")

    def _write_solved_epoch_heatmap(
        results,
        epochs,
        epoch_stds,
        solved,
        task_groups,
        figsize,
        tick_rotation,
        filename,
        fontsize,
        check_fontsize,
        *,
        show_appendix_boundary=False,
    ):
        fig_ep = plot_grouped_heatmap_solved_epochs(
            results,
            task_groups,
            epochs,
            threshold=1.0,
            epoch_std_results=epoch_stds,
            solved_results=solved,
            cmap=THESIS_HEATMAP_CMAP,
            figsize=figsize,
            annotation_fontsize=fontsize,
            check_fontsize=check_fontsize,
            boundary_color=SEPARATOR_COLOR,
            tick_rotation=tick_rotation,
        )
        fig_ep.axes[0].axhline(
            y=N_PUBLISHED - 0.5, color=SEPARATOR_COLOR, linewidth=1.2
        )
        if show_appendix_boundary:
            fig_ep.axes[0].axhline(
                y=N_CORE - 0.5, color=SEPARATOR_COLOR, linewidth=1.2
            )
        _strip_axis_titles(fig_ep)
        save_or_show(fig_ep, IMAGES_DIR / filename)

    _write_solved_epoch_heatmap(
        scaled_results,
        epoch_results,
        epoch_std,
        solved,
        SELECTED_TASK_GROUPS,
        selected_figsize,
        45,
        "headline_selected_solved_epochs.pdf",
        8,
        14,
    )
    _write_solved_epoch_heatmap(
        scaled_results_complete,
        epoch_results_complete,
        epoch_std_complete,
        solved_complete,
        FULL_TASK_GROUPS,
        full_figsize,
        full_tick_rotation,
        "headline_full_solved_epochs.pdf",
        6,
        11,
        show_appendix_boundary=True,
    )

    model_order = _headline_sort_models(list(scaled_results.keys()))
    _write_aggregate_table(
        TABLES_DIR / "headline_aggregate.tex",
        label="tab:experiments:headline:aggregate",
        caption_short="Mean scaled accuracy of the block architectures",
        caption_long=(
            "Mean scaled accuracy over all sixteen tasks and by task group "
            f"for the {_NUMBER_WORDS[N_CORE]} block architectures. "
            r"Accuracy entries give the mean $\pm$ sample standard deviation of the "
            f"{n_seeds_word} seed-level task averages. "
            r"A task counts as solved when all five seeds meet the criterion in Section~\ref{sec:method:techniques:metrics}."
            " Rows are ordered within each architecture group by the all-task mean."
        ),
        means=full_means,
        stds=full_stds,
        model_order=model_order,
        group_stats=full_group_stats,
        solved=solved,
    )
    _write_exploratory_table(
        TABLES_DIR / "yhlstm_aggregate.tex",
        means=appendix_means,
        stds=appendix_stds,
        group_stats=appendix_group_stats,
    )

    print(f"\n=== Per task scaled accuracy (mean +/- sample std; * = every "
          f"seed solved; k/n = solved seeds) ===")
    for model in _headline_sort_models(list(scaled_results_complete.keys())):
        print(f"\n  {model}")
        for task in all_tasks_flat:
            sv = scaled_results_complete[model].get(task)
            if sv is None:
                continue
            rv = raw_results_complete[model].get(task, float("nan"))
            flag = "*" if solved_complete[model][task] >= 1.0 else " "
            print(f"    {flag} {task:28s} scaled {sv:.3f} +/- "
                  f"{scaled_std_complete[model].get(task, 0.0):.3f}   raw {rv:.3f} +/- "
                  f"{raw_std_complete[model].get(task, 0.0):.3f}   "
                  f"({solved_counts_complete[model][task]})")

    print(f"\nWrote 6 figures to {IMAGES_DIR}")
    print(f"Wrote 2 tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
