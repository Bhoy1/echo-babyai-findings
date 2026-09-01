#!/usr/bin/env python3
"""Render the standard 200-step ECHO figures in the blog's shared style."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "standard_echo_200"
FIGURE_DIR = ROOT / "figures"

RL_COLOR = "#333B45"
ECHO_COLOR = "#43805E"
TEXT = "#20252B"
MUTED = "#66717D"
GRID = "#DDE3E8"
SPINE = "#AEB8C2"
SWITCH = "#929AA4"


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    marker: str
    train_steps: np.ndarray
    train_rewards: np.ndarray
    eval_steps: np.ndarray
    eval_means: np.ndarray
    eval_sds: np.ndarray


def moving_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = values[start : index + 1].mean()
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def series_from_csvs(
    *,
    label: str,
    color: str,
    marker: str,
    training_path: Path,
    eval_path: Path,
) -> Series:
    training = read_csv(training_path)
    evaluation = read_csv(eval_path)
    return Series(
        label=label,
        color=color,
        marker=marker,
        train_steps=np.asarray([int(row["step"]) for row in training]),
        train_rewards=np.asarray([float(row["train_reward"]) for row in training]),
        eval_steps=np.asarray([int(row["step"]) for row in evaluation]),
        eval_means=np.asarray([float(row["eval_mean"]) for row in evaluation]),
        eval_sds=np.asarray([float(row["eval_sd"]) for row in evaluation]),
    )


def grouped_series(
    *,
    schedule: str,
    label: str,
    color: str,
    marker: str,
) -> Series:
    training = [
        row
        for row in read_csv(DATA_DIR / "switch50_training.csv")
        if row["schedule"] == schedule
    ]
    evaluation = [
        row
        for row in read_csv(DATA_DIR / "switch50_eval.csv")
        if row["schedule"] == schedule
    ]
    return Series(
        label=label,
        color=color,
        marker=marker,
        train_steps=np.asarray([int(row["step"]) for row in training]),
        train_rewards=np.asarray([float(row["train_reward"]) for row in training]),
        eval_steps=np.asarray([int(row["step"]) for row in evaluation]),
        eval_means=np.asarray([float(row["eval_mean"]) for row in evaluation]),
        eval_sds=np.asarray([float(row["eval_sd"]) for row in evaluation]),
    )


def always_on_series() -> list[Series]:
    variants = json.loads(
        (DATA_DIR / "always_on.json").read_text(encoding="utf-8")
    )["variants"]
    result: list[Series] = []
    for key, label, color, marker in (
        ("rlonly", "RL", RL_COLOR, "o"),
        ("echo100", "ECHO 1.0", ECHO_COLOR, "D"),
    ):
        values = variants[key]
        result.append(
            Series(
                label=label,
                color=color,
                marker=marker,
                train_steps=np.asarray(values["train_steps"], dtype=int),
                train_rewards=np.asarray(values["train_rewards"], dtype=float),
                eval_steps=np.asarray(values["eval_steps"], dtype=int),
                eval_means=np.asarray(values["eval_means"], dtype=float),
                eval_sds=np.asarray(values["eval_sds"], dtype=float),
            )
        )
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 200)
    axis.set_ylim(0.35, 0.95)
    axis.set_xticks([0, 40, 80, 120, 160, 200])
    axis.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE)
    axis.spines["bottom"].set_color(SPINE)
    axis.tick_params(colors=MUTED, labelsize=9.5)
    axis.set_xlabel("Training step", color=TEXT, fontsize=10)


def plot_figure(
    series: list[Series],
    output_stem: str,
    *,
    switch_step: int | None = None,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.7))
    figure.subplots_adjust(top=0.79, bottom=0.13, left=0.08, right=0.98, wspace=0.18)

    for item in series:
        axes[0].plot(
            item.train_steps,
            item.train_rewards,
            color=item.color,
            linewidth=0.9,
            alpha=0.18,
        )
        axes[0].plot(
            item.train_steps,
            moving_mean(item.train_rewards),
            color=item.color,
            linewidth=2.35,
            label=item.label,
        )
        axes[1].fill_between(
            item.eval_steps,
            np.clip(item.eval_means - item.eval_sds, 0, 1),
            np.clip(item.eval_means + item.eval_sds, 0, 1),
            color=item.color,
            alpha=0.12,
            linewidth=0,
        )
        axes[1].plot(
            item.eval_steps,
            item.eval_means,
            color=item.color,
            linewidth=2.35,
            marker=item.marker,
            markersize=3.4,
            label=item.label,
        )

    for axis in axes:
        style_axis(axis)
        if switch_step is not None:
            axis.axvline(
                switch_step,
                color=SWITCH,
                linestyle="--",
                linewidth=1.15,
                zorder=1,
            )
            axis.text(
                switch_step + 3,
                0.97,
                "Switch",
                transform=axis.get_xaxis_transform(),
                color=MUTED,
                fontsize=8.5,
                va="top",
            )

    axes[0].set_title("Training reward", loc="left", color=TEXT, fontsize=13, weight="bold")
    axes[0].set_ylabel("Mean progress reward", color=TEXT, fontsize=10)
    axes[1].set_title(
        "Held-out evaluation reward", loc="left", color=TEXT, fontsize=13, weight="bold"
    )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=len(series),
        frameon=False,
        fontsize=10,
    )

    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURE_DIR / f"{output_stem}.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    plot_figure(always_on_series(), "standard_echo_200_always_on")

    switch100 = [
        series_from_csvs(
            label="RL100 → ECHO100",
            color=RL_COLOR,
            marker="o",
            training_path=DATA_DIR / "switch100_rl_echo_training.csv",
            eval_path=DATA_DIR / "switch100_rl_echo_eval.csv",
        ),
        series_from_csvs(
            label="ECHO100 → RL100",
            color=ECHO_COLOR,
            marker="D",
            training_path=DATA_DIR / "switch100_echo_rl_training.csv",
            eval_path=DATA_DIR / "switch100_echo_rl_eval.csv",
        ),
    ]
    plot_figure(switch100, "standard_echo_200_switch100", switch_step=100)

    switch50 = [
        grouped_series(
            schedule="rl50_echo150",
            label="RL50 → ECHO150",
            color=RL_COLOR,
            marker="o",
        ),
        grouped_series(
            schedule="echo50_rl150",
            label="ECHO50 → RL150",
            color=ECHO_COLOR,
            marker="D",
        ),
    ]
    plot_figure(switch50, "standard_echo_200_switch50", switch_step=50)


if __name__ == "__main__":
    main()
