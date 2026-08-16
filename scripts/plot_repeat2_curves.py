#!/usr/bin/env python3
"""Plot the four matched BabyAI repeat runs without replacing blog figures."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "repeat2"
FIGURE_DIR = ROOT / "figures"

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
TEXT = "#20252B"
MUTED = "#66717D"
GRID = "#DDE3E8"
SPINE = "#AEB8C2"


def moving_mean(values: list[float], window: int = 5) -> list[float]:
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / len(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def load_training() -> dict[str, list[tuple[int, float]]]:
    curves: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with (DATA_DIR / "training_curves.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            curves[row["variant"]].append((int(row["step"]), float(row["reward_mean"])))
    for points in curves.values():
        points.sort()
    return curves


def load_eval() -> tuple[tuple[int, float, float], dict[str, list[tuple[int, float, float]]]]:
    curves: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    base: tuple[int, float, float] | None = None
    with (DATA_DIR / "eval_curves.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            point = (int(row["step"]), float(row["reward_mean"]), float(row["reward_sd"]))
            if row["variant"] == "base":
                base = point
            else:
                curves[row["variant"]].append(point)
    if base is None:
        raise ValueError("Missing shared step-0 baseline")
    for points in curves.values():
        points.sort()
    return base, curves


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


def main() -> None:
    training = load_training()
    base, evaluation = load_eval()

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    figure.subplots_adjust(top=0.77, bottom=0.16, left=0.08, right=0.98, wspace=0.18)

    for variant in LABELS:
        train_points = training[variant]
        train_steps = [point[0] for point in train_points]
        train_rewards = [point[1] for point in train_points]
        axes[0].plot(
            train_steps,
            moving_mean(train_rewards),
            color=COLORS[variant],
            linewidth=2.25,
            label=LABELS[variant],
        )

        eval_points = [base, *evaluation[variant]]
        eval_steps = [point[0] for point in eval_points]
        eval_means = [point[1] for point in eval_points]
        eval_sd = [point[2] for point in eval_points]
        axes[1].fill_between(
            eval_steps,
            [mean - deviation for mean, deviation in zip(eval_means, eval_sd, strict=True)],
            [mean + deviation for mean, deviation in zip(eval_means, eval_sd, strict=True)],
            color=COLORS[variant],
            alpha=0.10,
            linewidth=0,
        )
        axes[1].plot(
            eval_steps,
            eval_means,
            color=COLORS[variant],
            linewidth=2.05,
            marker="o",
            markersize=3.1,
            label=LABELS[variant],
        )

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
        bbox_to_anchor=(0.5, 0.865),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle("BabyAI repeat runs", fontsize=18, fontweight="bold", color=TEXT, y=0.97)
    figure.text(
        0.5,
        0.915,
        "Qwen3.5-9B | 100 steps | training: 5-step moving mean | eval: mean +/- SD across 3 replicates",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURE_DIR / f"repeat2_always_on_curves_preview.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor="white",
        )
    plt.close(figure)

    summary_rows = []
    for variant in LABELS:
        points = evaluation[variant]
        peak = max(points, key=lambda point: point[1])
        final = points[-1]
        summary_rows.append(
            [
                LABELS[variant],
                f"{peak[1]:.3f} +/- {peak[2]:.3f}",
                str(peak[0]),
                f"{final[1]:.3f} +/- {final[2]:.3f}",
            ]
        )

    table_figure, table_axis = plt.subplots(figsize=(11.6, 4.2))
    table_axis.axis("off")
    table = table_axis.table(
        cellText=summary_rows,
        colLabels=["Run", "Peak eval", "Peak step", "Final eval (step 100)"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.22, 0.27, 0.18, 0.29],
        bbox=[0.03, 0.18, 0.94, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.65)
    variants = list(LABELS)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.9)
        if row == 0:
            cell.set_facecolor(TEXT)
            cell.set_text_props(color="white", weight="bold")
        elif column == 0:
            cell.set_facecolor(COLORS[variants[row - 1]])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F5F7F8" if row % 2 == 0 else "white")

    table_figure.suptitle(
        "BabyAI repeat-run evaluation summary",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
        y=0.95,
    )
    table_figure.text(
        0.5,
        0.865,
        "Qwen3.5-9B | mean +/- SD across 3 evaluation replicates",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    table_figure.savefig(
        FIGURE_DIR / "repeat2_eval_summary_table.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.16,
        facecolor="white",
    )
    plt.close(table_figure)


if __name__ == "__main__":
    main()
