#!/usr/bin/env python3
"""Plot ECHO 0.5 and ECHO 1.0 curves for BabyAI repeat three."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "repeat3"
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
        raise ValueError("Missing step-0 base evaluation")
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

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.05))
    figure.subplots_adjust(top=0.72, bottom=0.16, left=0.08, right=0.98, wspace=0.18)

    for variant in LABELS:
        train_points = training[variant]
        axes[0].plot(
            [point[0] for point in train_points],
            moving_mean([point[1] for point in train_points]),
            color=COLORS[variant],
            linewidth=2.35,
            label=LABELS[variant],
        )

        eval_points = [base, *evaluation[variant]]
        steps = [point[0] for point in eval_points]
        means = [point[1] for point in eval_points]
        deviations = [point[2] for point in eval_points]
        axes[1].fill_between(
            steps,
            [mean - deviation for mean, deviation in zip(means, deviations, strict=True)],
            [mean + deviation for mean, deviation in zip(means, deviations, strict=True)],
            color=COLORS[variant],
            alpha=0.12,
            linewidth=0,
        )
        axes[1].plot(
            steps,
            means,
            color=COLORS[variant],
            linewidth=2.15,
            marker="o",
            markersize=3.2,
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
        bbox_to_anchor=(0.5, 0.82),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle("BabyAI repeat 3", fontsize=18, fontweight="bold", color=TEXT, y=0.985)
    figure.text(
        0.5,
        0.91,
        "Qwen3.5-9B | 100 steps | training: 5-step moving mean | eval: mean +/- SD across 3 evaluation replicates",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURE_DIR / f"repeat3_always_on_training_eval.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor="white",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
