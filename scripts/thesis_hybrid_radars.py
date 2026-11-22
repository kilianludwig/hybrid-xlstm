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
from typing import Dict, List, Tuple

import numpy as np

from configs.registry import CHOMSKY_BY_HIERARCHY, MQAR_TASKS
from evaluation.task_metadata import TASK_SCALING_REGISTRY
from reporting.heatmap_data import (
    _extract_best_accuracy,
    build_heatmap_scaled_matrix,
)
from reporting.thesis_manifest import labels, model_of, runs_for, seed_of
from visualization.radar import plot_radar_overlay
from visualization.style import (
    SCHEMATIC_BLUE,
    SCHEMATIC_PURPLE,
    SCHEMATIC_RED,
    THESIS_TEXT_WIDTH_IN,
)
from visualization.utils import is_valid_number

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "figures/plots"
N_SEEDS = 5

TASK_ORDER = [
    task
    for group in ("regular", "dcf", "cs")
    for task in CHOMSKY_BY_HIERARCHY[group]
] + list(MQAR_TASKS)

_SCALAR = (SCHEMATIC_BLUE, ":", "o")
_MATRIX = (SCHEMATIC_RED, ":", "s")
_HYBRID = (SCHEMATIC_PURPLE, "-", "D")

TRIOS: List[Tuple[str, List[Tuple[str, str, str, str]]]] = [
    (
        "hybrid_radar_xlstm",
        [
            ("slstm", *_SCALAR),
            ("mlstm", *_MATRIX),
            ("xlstm", *_HYBRID),
        ],
    ),
    (
        "hybrid_radar_xlstm_hb",
        [
            ("slstm_hb", *_SCALAR),
            ("mlstm_hb", *_MATRIX),
            ("xlstm_hb", *_HYBRID),
        ],
    ),
    (
        "hybrid_radar_hlstm",
        [
            ("slstm_hb", *_SCALAR),
            ("mlstm_hb", *_MATRIX),
            ("hlstm", *_HYBRID),
        ],
    ),
    (
        "hybrid_radar_ylstm",
        [
            ("slstm_hb", *_SCALAR),
            ("mlstm_hb", *_MATRIX),
            ("ylstm_hb", *_HYBRID),
        ],
    ),
    (
        "hybrid_radar_yhlstm",
        [
            ("slstm_hb", *_SCALAR),
            ("ylstm_hb", *_MATRIX),
            ("yhlstm", *_HYBRID),
        ],
    ),
]


def _has_scaled_value(run: dict) -> bool:
    task = run.get("task", "?")
    if task not in TASK_SCALING_REGISTRY:
        return False
    scaling = run.get("scaling") or {}
    if "scaled_accuracy" in scaling and is_valid_number(scaling["scaled_accuracy"]):
        return True
    return is_valid_number(_extract_best_accuracy(run))


def _deduped_runs() -> List[dict]:
    seen: Dict[Tuple[str, str, int], dict] = {}
    for experiment_id in (
        "E1_headline",
        "E1_yhlstm_appendix",
        "E2_block_cross",
    ):
        for run in runs_for(experiment_id):
            key = (model_of(run), run.get("task", "?"), seed_of(run))
            seen.setdefault(key, run)
    return list(seen.values())


def _require_five_seeds(runs: List[dict], models: List[str]) -> None:
    counts: Dict[Tuple[str, str], int] = {}
    for run in runs:
        model = model_of(run)
        task = run.get("task", "?")
        if model not in models or task not in TASK_ORDER:
            continue
        if _has_scaled_value(run):
            counts[(model, task)] = counts.get((model, task), 0) + 1
    missing = [
        f"{model} / {task}: {counts.get((model, task), 0)}"
        for model in models
        for task in TASK_ORDER
        if counts.get((model, task), 0) != N_SEEDS
    ]
    if missing:
        raise SystemExit(
            "expected "
            f"{N_SEEDS} scaled values per model and task:\n"
            + "\n".join(missing)
        )


def _sample_std(
    std: Dict[str, Dict[str, float]],
    models: List[str],
) -> Dict[str, Dict[str, float]]:
    factor = float(np.sqrt(N_SEEDS / (N_SEEDS - 1)))
    return {
        model: {task: value * factor for task, value in std[model].items()}
        for model in models
    }


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    name_of = labels()
    runs = _deduped_runs()
    needed = sorted({model for _, arms in TRIOS for model, *_rest in arms})
    _require_five_seeds(runs, needed)

    scaled, population_std = build_heatmap_scaled_matrix(runs)
    for model in needed:
        absent = [task for task in TASK_ORDER if task not in scaled.get(model, {})]
        if absent:
            raise SystemExit(f"{model} has no scaled accuracy for {absent}")
    sample_std = _sample_std(population_std, needed)

    for stem, arms in TRIOS:
        models = [model for model, *_rest in arms]
        plot_radar_overlay(
            {model: scaled[model] for model in models},
            TASK_ORDER,
            model_order=models,
            color_overrides={model: color for model, color, *_rest in arms},
            linestyle_overrides={model: style for model, _color, style, _marker in arms},
            marker_overrides={model: marker for model, *_rest, marker in arms},
            label_overrides={model: name_of[model] for model in models},
            std={model: sample_std[model] for model in models},
            legend_below=True,
            legend_ncol=1,
            tick_fontsize=7,
            align_labels=True,
            tick_pad=8,
            rlabel_position=360.0 * (1.0 - 0.5 / len(TASK_ORDER)),
            lift_outer_rlabel=True,
            title="",
            figsize=(THESIS_TEXT_WIDTH_IN, THESIS_TEXT_WIDTH_IN),
            output_path=IMAGES_DIR / f"{stem}.pdf",
        )
        print(f"wrote {stem}.pdf")


if __name__ == "__main__":
    main()
