#!/usr/bin/env python3
"""Plot BabyAI means and SD across three independent runs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
OUTPUT_DATA = ROOT / "data" / "three_independent_runs"

LABELS = {
    "rlonly": "RL-only",
    "echo005": "ECHO 0.05",
    "echo050": "ECHO 0.5",
    "echo100": "ECHO 1.0",
}
COLORS = {
    "rlonly": "#333B45",
    "echo005": "#2778A5",
    "echo050": "#D1842D",
    "echo100": "#43805E",
}
OLD_TRAIN_NAMES = {
    "always_on/rl_only": "rlonly",
    "always_on/echo_0.05": "echo005",
    "always_on/echo_0.5": "echo050",
    "always_on/echo_1.0": "echo100",
}
TEXT = "#20252B"
MUTED = "#66717D"
GRID = "#DDE3E8"
SPINE = "#AEB8C2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_training_runs() -> dict[str, dict[str, list[tuple[int, float]]]]:
    runs: dict[str, dict[str, list[tuple[int, float]]]] = {
        "independent_1": defaultdict(list),
        "independent_2": defaultdict(list),
        "independent_3": defaultdict(list),
    }
    for row in read_csv(ROOT / "data" / "training_behavior_metrics.csv"):
        if row["population"] != "retained_trainable":
            continue
        variant = OLD_TRAIN_NAMES.get(row["run"])
        if variant:
            runs["independent_1"][variant].append(
                (int(row["step"]), float(row["reward_mean"]))
            )
    for row in read_csv(ROOT / "data" / "repeat2" / "training_curves.csv"):
        runs["independent_2"][row["variant"]].append(
            (int(row["step"]), float(row["reward_mean"]))
        )
    for row in read_csv(ROOT / "data" / "repeat3" / "training_curves.csv"):
        runs["independent_3"][row["variant"]].append(
            (int(row["step"]), float(row["reward_mean"]))
        )
    for run in runs.values():
        for points in run.values():
            points.sort()
    return runs


def load_eval_runs() -> dict[str, dict[str, list[tuple[int, float]]]]:
    runs: dict[str, dict[str, list[tuple[int, float]]]] = {
        "independent_1": defaultdict(list),
        "independent_2": defaultdict(list),
        "independent_3": defaultdict(list),
    }
    for row in read_csv(ROOT / "data" / "checkpoint_eval_metrics.csv"):
        runs["independent_1"][row["variant"]].append(
            (int(row["step"]), float(row["reward_mean"]))
        )
    repeat2_rows = read_csv(ROOT / "data" / "repeat2" / "eval_curves.csv")
    base = next(row for row in repeat2_rows if row["variant"] == "base")
    for variant in LABELS:
        runs["independent_2"][variant].append((0, float(base["reward_mean"])))
    for row in repeat2_rows:
        if row["variant"] != "base":
            runs["independent_2"][row["variant"]].append(
                (int(row["step"]), float(row["reward_mean"]))
            )
    repeat3_rows = read_csv(ROOT / "data" / "repeat3" / "eval_curves.csv")
    base = next(row for row in repeat3_rows if row["variant"] == "base")
    for variant in LABELS:
        runs["independent_3"][variant].append((0, float(base["reward_mean"])))
    for row in repeat3_rows:
        if row["variant"] != "base":
            runs["independent_3"][row["variant"]].append(
                (int(row["step"]), float(row["reward_mean"]))
            )
    for run in runs.values():
        for points in run.values():
            points.sort()
    return runs


def moving_mean(values: list[float], window: int = 5) -> list[float]:
    return [
        fmean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def combine(
    runs: dict[str, dict[str, list[tuple[int, float]]]],
    *,
    smooth: bool,
) -> dict[str, list[tuple[int, float, float]]]:
    combined: dict[str, list[tuple[int, float, float]]] = {}
    run_names = list(runs)
    for variant in LABELS:
        by_run: list[dict[int, float]] = []
        for run_name in run_names:
            points = runs[run_name][variant]
            steps = [point[0] for point in points]
            values = [point[1] for point in points]
            if smooth:
                values = moving_mean(values)
            by_run.append(dict(zip(steps, values, strict=True)))
        steps = sorted(set.intersection(*(set(values) for values in by_run)))
        combined[variant] = []
        for step in steps:
            values = [run[step] for run in by_run]
            combined[variant].append((step, fmean(values), stdev(values)))
    return combined


def write_combined(
    path: Path,
    combined: dict[str, list[tuple[int, float, float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "step", "reward_mean", "training_run_sd", "num_training_runs"])
        for variant, points in combined.items():
            for step, mean, deviation in points:
                writer.writerow([variant, step, mean, deviation, 3])


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 100)
    axis.set_ylim(0.35, 0.95)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE)
    axis.spines["bottom"].set_color(SPINE)
    axis.tick_params(colors=MUTED, labelsize=9.5)
    axis.set_xlabel("Training step", color=TEXT, fontsize=10)


def save_eval_summary(combined: dict[str, list[tuple[int, float, float]]]) -> None:
    rows: list[list[str]] = []
    for variant, label in LABELS.items():
        points = combined[variant]
        peak = max(points, key=lambda point: point[1])
        final = max(points, key=lambda point: point[0])
        rows.append(
            [
                label,
                f"{peak[1]:.3f} ± {peak[2]:.3f}",
                str(peak[0]),
                f"{final[1]:.3f} ± {final[2]:.3f}",
            ]
        )

    figure, axis = plt.subplots(figsize=(12.6, 3.7))
    axis.axis("off")
    axis.set_position([0.015, 0.04, 0.97, 0.67])
    figure.text(
        0.015,
        0.94,
        "Held-out Evaluation Summary",
        fontsize=22,
        fontweight="bold",
        color="#1f2d3d",
    )
    figure.text(
        0.015,
        0.84,
        "Mean ± sample SD across three independent training runs",
        fontsize=12.5,
        color=MUTED,
    )
    table = axis.table(
        cellText=rows,
        colLabels=[
            "Objective",
            "Peak held-out reward",
            "Peak step",
            "Final held-out reward",
        ],
        cellLoc="right",
        colLoc="right",
        bbox=[0, 0, 1, 1],
        colWidths=[0.24, 0.30, 0.18, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dbe2ea")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#203043")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f4f7fa" if row % 2 else "#ffffff")
            cell.get_text().set_color("#253447")
            if column == 0:
                cell.get_text().set_weight("bold")
        cell.PAD = 0.04
    for row in range(1, len(rows) + 1):
        table[(row, 0)].get_text().set_ha("left")
    table[(0, 0)].get_text().set_ha("left")
    table[(4, 1)].get_text().set_weight("bold")
    table[(4, 3)].get_text().set_weight("bold")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_DIR / "always_on_eval_summary.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(figure)


def plot_panel(
    axis: plt.Axes,
    runs: dict[str, dict[str, list[tuple[int, float]]]],
    combined: dict[str, list[tuple[int, float, float]]],
    *,
    smooth: bool,
) -> None:
    for variant in LABELS:
        for run in runs.values():
            points = run[variant]
            steps = [point[0] for point in points]
            values = [point[1] for point in points]
            if smooth:
                values = moving_mean(values)
            axis.plot(steps, values, color=COLORS[variant], linewidth=0.9, alpha=0.20)
        points = combined[variant]
        steps = [point[0] for point in points]
        means = [point[1] for point in points]
        deviations = [point[2] for point in points]
        axis.fill_between(
            steps,
            [mean - deviation for mean, deviation in zip(means, deviations, strict=True)],
            [mean + deviation for mean, deviation in zip(means, deviations, strict=True)],
            color=COLORS[variant],
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(
            steps,
            means,
            color=COLORS[variant],
            linewidth=2.35,
            marker=None if smooth else "o",
            markersize=3.0,
            label=LABELS[variant],
        )


def main() -> None:
    training_runs = load_training_runs()
    eval_runs = load_eval_runs()
    training_combined = combine(training_runs, smooth=True)
    eval_combined = combine(eval_runs, smooth=False)
    write_combined(OUTPUT_DATA / "training_reward_mean_sd.csv", training_combined)
    write_combined(OUTPUT_DATA / "eval_reward_mean_sd.csv", eval_combined)
    save_eval_summary(eval_combined)

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    figure.subplots_adjust(top=0.72, bottom=0.12, left=0.08, right=0.98, wspace=0.18)
    plot_panel(axes[0], training_runs, training_combined, smooth=True)
    plot_panel(axes[1], eval_runs, eval_combined, smooth=False)
    for axis in axes:
        style_axis(axis)
    axes[0].set_title("Training reward", loc="left", color=TEXT, fontsize=13, weight="bold")
    axes[0].set_ylabel("Mean progress reward", color=TEXT, fontsize=10)
    axes[1].set_title("Held-out evaluation reward", loc="left", color=TEXT, fontsize=13, weight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.82),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle("BabyAI", fontsize=18, fontweight="bold", color=TEXT, y=0.985)
    figure.text(
        0.5,
        0.91,
        "Qwen3.5-9B | bold: mean | band: +/- SD across runs | faint: individual runs",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURE_DIR / f"three_independent_runs_training_eval.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor="white",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
