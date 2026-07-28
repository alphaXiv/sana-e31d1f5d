"""Collect terminal ORX metrics and render the Sol-Attn report figures.

The published CSV and SVG files are generated from successful Kubernetes run
logs.  This script requires the local ``orx`` CLI and matplotlib, but the
generated artifacts are committed so readers do not need either dependency.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sol_attn"
IMAGES = ROOT / "reports" / "sol_attn" / "images"

RUNS = {
    "final_random_32k": "246b7bba-c79f-4dd0-9e9c-afca5da7ee22",
    "final_smooth_32k": "3a61b20d-6895-4446-8157-27f5be6c6749",
    "final_temporal_32k": "b5c76a93-64ee-4ee3-8464-09959982f19a",
    "final_random_64k": "92386117-2672-4264-a8d5-b286a5bfa5e0",
    "density_calibration": "0e6093c3-adfc-43fc-b9b9-047221bbf5c7",
    "c32_scaling": "0c05f385-11e0-4d0e-a02e-d6c14908afb7",
    "memory_profile": "1547da46-860a-4b6b-adc1-a20df4af922b",
    "heavy_tail_control": "83c6cf03-d173-4612-8189-d46cc0a5eb5b",
    "released_sana_probe": "e261a579-bccb-47f0-9784-a008c5187370",
}

COLORS = {
    "random": "#276FBF",
    "smooth": "#F28E2B",
    "temporal": "#59A14F",
    "heavy_tail": "#B07AA1",
    "sol": "#19A7A0",
    "exact": "#9AA0A6",
    "accent": "#E45756",
}


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def collect() -> list[dict]:
    rows: list[dict] = []
    for source, run_id in RUNS.items():
        text = subprocess.run(
            ["orx", "logs", run_id], check=True, capture_output=True, text=True
        ).stdout
        found = 0
        for line in text.splitlines():
            if not line.startswith("ORX_METRIC "):
                continue
            payload = json.loads(line.removeprefix("ORX_METRIC "))
            payload["source"] = source
            rows.append(payload)
            found += 1
        if not found:
            raise RuntimeError(f"No ORX_METRIC rows in {run_id}")
    return rows


def write_csv(rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (RESULTS / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def target_label(row: dict) -> str:
    return f"{100 * float(row['target_density_gaussian']):.0f}%"


def style_ax(ax, title: str, ylabel: str, xlabel: str = "") -> None:
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def headline(rows: list[dict]) -> None:
    final = [
        row
        for row in rows
        if row["source"]
        in {"final_random_32k", "final_smooth_32k", "final_temporal_32k"}
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in final:
        grouped[(row["family"], target_label(row))].append(row)

    families = ["random", "smooth", "temporal"]
    targets = ["15%", "10%"]
    x = list(range(len(families)))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for index, target in enumerate(targets):
        offset = (index - 0.5) * width
        error = [
            100 * mean([r["error_reduction_fraction"] for r in grouped[(family, target)]])
            for family in families
        ]
        speed = [
            mean([r["sol_speedup_over_dense"] for r in grouped[(family, target)]])
            for family in families
        ]
        error_sd = [
            100 * sd([r["error_reduction_fraction"] for r in grouped[(family, target)]])
            for family in families
        ]
        speed_sd = [
            sd([r["sol_speedup_over_dense"] for r in grouped[(family, target)]])
            for family in families
        ]
        axes[0].bar(
            [v + offset for v in x],
            error,
            width,
            yerr=error_sd,
            capsize=3,
            label=f"{target} exact blocks",
            color=("#3E7CB1", "#73BFB8")[index],
        )
        axes[1].bar(
            [v + offset for v in x],
            speed,
            width,
            yerr=speed_sd,
            capsize=3,
            label=f"{target} exact blocks",
            color=("#3E7CB1", "#73BFB8")[index],
        )
    style_ax(
        axes[0],
        "Correction recovers discarded-block signal",
        "L2 error reduction vs exact-only (%)",
    )
    style_ax(axes[1], "Useful end-to-end latency remains", "Speedup over dense")
    axes[0].set_ylim(0, 105)
    axes[1].axhline(1, color="#555", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.6)
    for ax in axes:
        ax.set_xticks(x, ["Random", "Smooth", "Temporal"])
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "32K-token Blackwell reproduction · mean ± seed SD (n=4)",
        x=0.08,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "headline_tradeoff.png")


def calibration(rows: list[dict]) -> None:
    chosen = [row for row in rows if row["source"] == "density_calibration"]
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in chosen:
        grouped[round(float(row["target_density_gaussian"]), 3)].append(row)
    targets = sorted(grouped)
    observed = [mean([r["density_mean"] for r in grouped[t]]) for t in targets]
    low = [mean([r["density_p10"] for r in grouped[t]]) for t in targets]
    high = [mean([r["density_p90"] for r in grouped[t]]) for t in targets]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([0, 0.28], [0, 0.28], color="#777", linestyle="--", label="ideal")
    ax.errorbar(
        targets,
        observed,
        yerr=[
            [value - lower for value, lower in zip(observed, low)],
            [upper - value for upper, value in zip(high, observed)],
        ],
        fmt="o-",
        color=COLORS["random"],
        capsize=4,
        linewidth=2,
        label="observed mean; query p10–p90",
    )
    style_ax(
        ax,
        "Mean/variance thresholds track requested sparsity",
        "Observed selected-block density",
        "Gaussian target density",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100*value:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100*value:.0f}%"))
    ax.set_xlim(0.03, 0.27)
    ax.set_ylim(0.03, 0.27)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "density_calibration.png")


def scaling(rows: list[dict]) -> None:
    chosen = [row for row in rows if row["source"] == "c32_scaling"]
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in chosen:
        grouped[(int(row["length"]), target_label(row))].append(row)
    lengths = sorted({key[0] for key in grouped})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for target, color in [("15%", "#3E7CB1"), ("10%", "#19A7A0")]:
        sol = [mean([r["sol_speedup_over_dense"] for r in grouped[(n, target)]]) for n in lengths]
        exact = [
            mean([r["exact_speedup_over_dense"] for r in grouped[(n, target)]])
            for n in lengths
        ]
        ax.plot(
            [n / 1024 for n in lengths],
            sol,
            "o-",
            color=color,
            linewidth=2,
            label=f"corrected, {target}",
        )
        ax.plot(
            [n / 1024 for n in lengths],
            exact,
            ":",
            color=color,
            alpha=0.65,
            label=f"exact-only, {target}",
        )
    ax.axhline(1, color="#555", linestyle="--", linewidth=1)
    style_ax(
        ax,
        "Correction preserves a practical long-sequence advantage",
        "End-to-end speedup over dense",
        "Sequence length (K tokens)",
    )
    ax.set_xticks([n / 1024 for n in lengths])
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "latency_scaling.png")


def routing_memory(rows: list[dict]) -> None:
    chosen = [row for row in rows if row["source"] == "memory_profile"]
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in chosen:
        grouped[int(row["length"])].append(row)
    lengths = sorted(grouped)
    reduction = [mean([r["routing_aux_reduction"] for r in grouped[n]]) for n in lengths]
    memory = [
        mean([r["sol_kernel_memory_ratio_to_dense"] for r in grouped[n]]) for n in lengths
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax2 = ax.twinx()
    line1 = ax.plot(
        [n / 1024 for n in lengths],
        reduction,
        "o-",
        color=COLORS["random"],
        linewidth=2,
        label="routing-state reduction",
    )
    line2 = ax2.plot(
        [n / 1024 for n in lengths],
        memory,
        "s-",
        color=COLORS["accent"],
        linewidth=2,
        label="kernel memory / dense",
    )
    ax.set_yscale("log")
    style_ax(
        ax,
        "Routing map disappears; kernel memory stays near dense",
        "Explicit proxy map / moment state (×, log)",
        "Sequence length (K tokens)",
    )
    ax2.set_ylabel("Kernel incremental memory / dense")
    ax2.axhline(1, color="#777", linestyle="--", linewidth=1)
    ax2.set_ylim(0.998, 1.003)
    ax2.spines["top"].set_visible(False)
    ax.set_xticks([n / 1024 for n in lengths])
    lines = line1 + line2
    ax.legend(lines, [line.get_label() for line in lines], frameon=False)
    fig.tight_layout()
    save(fig, "routing_memory.png")


def robustness(rows: list[dict]) -> None:
    sources = {
        "Random": "final_random_32k",
        "Smooth": "final_smooth_32k",
        "Temporal": "final_temporal_32k",
        "Heavy-tail": "heavy_tail_control",
    }
    values: dict[str, list[float]] = {}
    for label, source in sources.items():
        values[label] = [
            100 * float(row["error_reduction_fraction"])
            for row in rows
            if row["source"] == source
            and isinstance(row.get("error_reduction_fraction"), (float, int))
            and math.isfinite(float(row["error_reduction_fraction"]))
        ]
    labels = list(values)
    averages = [mean(values[label]) for label in labels]
    errors = [sd(values[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(
        labels,
        averages,
        yerr=errors,
        capsize=4,
        color=[
            COLORS["random"],
            COLORS["smooth"],
            COLORS["temporal"],
            COLORS["heavy_tail"],
        ],
    )
    style_ax(
        ax,
        "The zeroth-order correction depends on block-mean structure",
        "L2 error reduction vs exact-only (%)",
    )
    ax.set_ylim(0, 105)
    ax.text(
        3,
        averages[3] + 6,
        "negative control",
        ha="center",
        color="#6D4C72",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "robustness_control.png")


def write_summary(rows: list[dict]) -> None:
    final = [
        row
        for row in rows
        if row["source"]
        in {
            "final_random_32k",
            "final_smooth_32k",
            "final_temporal_32k",
            "final_random_64k",
        }
    ]
    sana = [row for row in rows if row["source"] == "released_sana_probe"]
    summary = {
        "terminal_metric_rows": len(rows),
        "final_replication_conditions": len(final),
        "final_error_reduction_by_family": {
            family: mean(
                [
                    row["error_reduction_fraction"]
                    for row in final
                    if row["family"] == family
                ]
            )
            for family in ["random", "smooth", "temporal"]
        },
        "final_32k_speedup_by_family": {
            family: mean(
                [
                    row["sol_speedup_over_dense"]
                    for row in final
                    if row["family"] == family and row["length"] == 32768
                ]
            )
            for family in ["random", "smooth", "temporal"]
        },
        "sana_observed_density_by_layer": {
            str(layer): mean(
                [row["density_mean"] for row in sana if row["layer"] == layer]
            )
            for layer in sorted({row["layer"] for row in sana})
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#25313C",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    rows = collect()
    write_csv(rows)
    write_summary(rows)
    headline(rows)
    calibration(rows)
    scaling(rows)
    routing_memory(rows)
    robustness(rows)
    print(f"Wrote {len(rows)} metric rows, 1 summary, and 5 figures")


if __name__ == "__main__":
    main()
