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
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from visualization.style import (
    CHUNK_SIZE_COLORS,
    CHUNK_SIZE_MARKERS,
    HEADLINE_ARCH_COLORS,
    SCHEMATIC_STROKE,
    apply_paper_style,
)
from visualization.utils import save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    REPO_ROOT
    / "experiments/ylstm_architecture/20260930_0130_headline_throughput_matched/results"
)
IMAGES = REPO_ROOT / "figures/plots"

CHUNK_ORDER = ["ylstm_hb_c1", "ylstm_hb_c2", "ylstm_hb_c4", "ylstm_hb_c8", "ylstm_hb_c0"]
HEADLINE_ORDER = [
    "transformer", "mamba", "slstm", "mlstm", "xlstm",
    "hlstm", "xlstm_hb", "yhlstm",
]
LABELS = {
    "ylstm_hb_c1": r"yLSTM $\ell=1$",
    "ylstm_hb_c2": r"yLSTM $\ell=2$",
    "ylstm_hb_c4": r"yLSTM $\ell=4$",
    "ylstm_hb_c8": r"yLSTM $\ell=8$",
    "ylstm_hb_c0": r"yLSTM $\ell=T$",
    "transformer": "Transformer",
    "mamba": "Mamba",
    "slstm": "sLSTM",
    "mlstm": "mLSTM",
    "xlstm": "xLSTM",
    "hlstm": "hLSTM",
    "xlstm_hb": "xLSTM, hLSTM block",
    "yhlstm": "yhLSTM",
}
HEADLINE_MARKER = {
    "transformer": "^",
    "mamba": "D",
    "slstm": "s",
    "mlstm": "P",
    "xlstm": "X",
    "hlstm": "o",
    "xlstm_hb": "v",
    "yhlstm": "h",
}


def _load_csv(paths: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No benchmark rows under {RESULTS}")
    return rows


def _float(row: Dict[str, str], key: str) -> Optional[float]:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    return float(raw)


def _series(rows, model, key):
    pts = []
    for row in rows:
        if row["model"] != model:
            continue
        value = _float(row, key)
        if value is None:
            continue
        pts.append((int(row["seq_len"]), value / 1000.0))
    pts.sort()
    return pts


def fig_vs_length(rows) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True)
    columns = (
        ("inference_tokens_per_second", "Inference"),
        ("train_tokens_per_second", "Training step"),
    )
    for ax, (key, title) in zip(axes, columns):
        for model in CHUNK_ORDER:
            pts = _series(rows, model, key)
            if not pts:
                continue
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=CHUNK_SIZE_COLORS[model],
                marker=CHUNK_SIZE_MARKERS[model],
                linewidth=2.0,
                markersize=4.5,
                label=LABELS[model],
                zorder=3,
            )
        for model in HEADLINE_ORDER:
            pts = _series(rows, model, key)
            if not pts:
                continue
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=HEADLINE_ARCH_COLORS[model],
                marker=HEADLINE_MARKER[model],
                linewidth=1.15,
                markersize=5,
                linestyle=(0, (1, 1.2)),
                label=LABELS[model],
                zorder=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([16, 64, 256, 1024, 2048, 4096])
        ax.set_xticklabels(["16", "64", "256", "1024", "2048", "4096"])
        ax.set_xlabel(r"Bare-backbone length $T$")
        ax.set_title(title)
        ax.grid(True, which="both", axis="y", color=SCHEMATIC_STROKE, alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Thousand tokens per second")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save_figure(fig, IMAGES / "throughput_vs_length.pdf")
    plt.close(fig)


def fig_memory(rows) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True)
    columns = (
        ("inference_peak_mib", "inference_status", "Inference"),
        ("train_peak_mib", "train_status", "Training step"),
    )
    totals = [
        float(row["device_total_mib"])
        for row in rows
        if row.get("device_total_mib")
    ]
    capacity_gib = (totals[0] / 1024.0) if totals else 11.0
    for ax, (key, status_key, title) in zip(axes, columns):
        for model, color, marker, width, style in (
            *((m, CHUNK_SIZE_COLORS[m], CHUNK_SIZE_MARKERS[m], 2.0, "-") for m in CHUNK_ORDER),
            *((m, HEADLINE_ARCH_COLORS[m], HEADLINE_MARKER[m], 1.15, (0, (1, 1.2))) for m in HEADLINE_ORDER),
        ):
            ok, oom = [], []
            for row in rows:
                if row["model"] != model:
                    continue
                value = _float(row, key)
                if value is None:
                    continue
                point = (int(row["seq_len"]), value / 1024.0)
                if row.get(status_key) == "oom":
                    oom.append(point)
                else:
                    ok.append(point)
            ok.sort()
            if ok:
                ax.plot(
                    [p[0] for p in ok],
                    [p[1] for p in ok],
                    color=color,
                    marker=marker,
                    linewidth=width,
                    markersize=5,
                    linestyle=style,
                    label=LABELS[model],
                    zorder=3,
                )
            if oom:
                ax.scatter(
                    [p[0] for p in oom],
                    [p[1] for p in oom],
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    s=28,
                    linewidths=1.2,
                    zorder=4,
                )
        ax.axhline(capacity_gib, color=SCHEMATIC_STROKE, linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([16, 64, 256, 1024, 2048, 4096])
        ax.set_xticklabels(["16", "64", "256", "1024", "2048", "4096"])
        ax.set_xlabel(r"Bare-backbone length $T$")
        ax.set_title(title)
        ax.grid(True, which="both", axis="y", color=SCHEMATIC_STROKE, alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Peak memory (GiB)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=SCHEMATIC_STROKE, linestyle="--", linewidth=0.8))
    labels.append(f"{capacity_gib:.0f} GiB device")
    handles.append(
        Line2D([0], [0], marker="o", color="black", markerfacecolor="none", linestyle="none", markersize=5)
    )
    labels.append("out of memory")
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    save_figure(fig, IMAGES / "memory_vs_length.pdf")
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig_vs_length(_load_csv(sorted(RESULTS.glob("gpu[0-9].csv"))))
    fig_memory(_load_csv(sorted(RESULTS.glob("memory_gpu[0-9].csv"))))
    print(f"Wrote throughput_vs_length.pdf and memory_vs_length.pdf to {IMAGES}")


if __name__ == "__main__":
    main()
