#!/usr/bin/env python3
"""Render pure ECHO (SFT) switch figures in the blog's shared style."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pure_sft_200"
FIGURE_DIR = ROOT / "figures"

RL_COLOR = "#333B45"
SFT_COLOR = "#6F5BD3"
TEXT = "#20252B"
MUTED = "#66717D"
GRID = "#DDE3E8"
SPINE = "#AEB8C2"
SWITCH = "#929AA4"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def moving_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = values[start : index + 1].mean()
    return result


def style_axis(axis: plt.Axes, *, y_min: float = 0.30) -> None:
    axis.set_xlim(0, 200)
    axis.set_ylim(y_min, 0.95)
    axis.set_xticks([0, 40, 80, 120, 160, 200])
    axis.set_yticks(
        np.arange(0.3, 0.91, 0.1)
        if y_min <= 0.30
        else np.arange(0.4, 0.91, 0.1)
    )
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE)
    axis.spines["bottom"].set_color(SPINE)
    axis.tick_params(colors=MUTED, labelsize=9.5)
    axis.set_xlabel("Training step", color=TEXT, fontsize=10)


def add_phases(
    axis: plt.Axes,
    phases: list[tuple[int, int, str]],
) -> None:
    for start, end, label in phases:
        color = SFT_COLOR if "SFT" in label else RL_COLOR
        axis.text(
            (start + end) / 2,
            0.975,
            label,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color=color,
            fontsize=8.5,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
            zorder=5,
        )
    for _, end, _ in phases[:-1]:
        axis.axvline(end, color=SWITCH, linestyle="--", linewidth=1.15, zorder=1)


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURE_DIR / f"{stem}.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.16,
            facecolor="white",
        )
    plt.close(figure)


def load_switch_training() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(DATA_DIR / "switch100_training.csv"):
        grouped.setdefault(row["schedule"], []).append(row)
    return {
        key: (
            np.asarray([int(row["step"]) for row in rows]),
            np.asarray([float(row["train_reward"]) for row in rows]),
        )
        for key, rows in grouped.items()
    }


def load_switch_eval() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(DATA_DIR / "switch100_eval.csv"):
        grouped.setdefault(row["schedule"], []).append(row)
    return {
        key: (
            np.asarray([int(row["step"]) for row in rows]),
            np.asarray([float(row["reward_mean"]) for row in rows]),
            np.asarray([float(row["reward_sd"]) for row in rows]),
        )
        for key, rows in grouped.items()
    }


def plot_switch_training_eval() -> None:
    training = load_switch_training()
    evaluation = load_switch_eval()
    series = (
        (
            "echo_to_rl",
            "echo_sft100_to_rl100",
            "ECHO (SFT) → RL",
            SFT_COLOR,
            "D",
            "-",
        ),
        (
            "rl_to_echo",
            "rl100_to_echo_sft100",
            "RL → ECHO (SFT)",
            RL_COLOR,
            "o",
            "--",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), sharey=True)
    figure.subplots_adjust(
        top=0.79,
        bottom=0.13,
        left=0.08,
        right=0.98,
        wspace=0.18,
    )
    axes[0].set_title(
        "Training Curve", loc="left", color=TEXT, fontsize=13, weight="bold", pad=12
    )
    axes[1].set_title(
        "Held-Out Evaluation",
        loc="left",
        color=TEXT,
        fontsize=13,
        weight="bold",
        pad=12,
    )

    for (
        training_key,
        evaluation_key,
        label,
        color,
        marker,
        linestyle,
    ) in series:
        train_steps, train_rewards = training[training_key]
        eval_steps, eval_means, eval_sds = evaluation[evaluation_key]

        axes[0].plot(
            train_steps,
            train_rewards,
            color=color,
            linewidth=0.9,
            alpha=0.18,
        )
        axes[0].plot(
            train_steps,
            moving_mean(train_rewards),
            color=color,
            linewidth=2.35,
            linestyle=linestyle,
            label=label,
        )
        axes[1].fill_between(
            eval_steps,
            np.clip(eval_means - eval_sds, 0, 1),
            np.clip(eval_means + eval_sds, 0, 1),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        axes[1].plot(
            eval_steps,
            eval_means,
            color=color,
            linewidth=2.35,
            linestyle=linestyle,
            marker=marker,
            markersize=3.4,
            label=label,
        )

    for axis in axes:
        style_axis(axis)
        axis.axvline(
            100,
            color=SWITCH,
            linestyle="--",
            linewidth=1.15,
            zorder=1,
        )
        axis.text(
            103,
            0.97,
            "Switch",
            transform=axis.get_xaxis_transform(),
            color=MUTED,
            fontsize=8.5,
            va="top",
        )
    axes[0].set_ylabel("Mean progress reward", color=TEXT, fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    save_figure(figure, "sft_switch_training_eval")


def load_four_phase() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    training = read_csv(DATA_DIR / "four_phase_training.csv")
    evaluation = [
        row
        for row in read_csv(DATA_DIR / "four_phase_eval.csv")
        if row["schedule"] == "four_phase"
    ]
    return (
        np.asarray([int(row["step"]) for row in training]),
        np.asarray([float(row["train_reward"]) for row in training]),
        np.asarray([int(float(row["step"])) for row in evaluation]),
        np.asarray([float(row["reward_mean"]) for row in evaluation]),
        np.asarray([float(row["reward_sd"]) for row in evaluation]),
    )


def plot_four_phase() -> None:
    train_steps, train_rewards, eval_steps, eval_means, eval_sds = load_four_phase()
    phases = [
        (0, 50, "RL"),
        (50, 100, "ECHO (SFT)"),
        (100, 150, "RL"),
        (150, 200, "ECHO (SFT)"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), sharey=True)
    figure.subplots_adjust(top=0.82, bottom=0.13, left=0.08, right=0.98, wspace=0.18)

    axes[0].plot(train_steps, train_rewards, color=SFT_COLOR, linewidth=0.9, alpha=0.18)
    axes[0].plot(
        train_steps,
        moving_mean(train_rewards),
        color=SFT_COLOR,
        linewidth=2.35,
    )
    axes[1].fill_between(
        eval_steps,
        np.clip(eval_means - eval_sds, 0, 1),
        np.clip(eval_means + eval_sds, 0, 1),
        color=SFT_COLOR,
        alpha=0.12,
        linewidth=0,
    )
    axes[1].plot(
        eval_steps,
        eval_means,
        color=SFT_COLOR,
        linewidth=2.35,
        marker="D",
        markersize=3.4,
    )
    for axis in axes:
        add_phases(axis, phases)
        style_axis(axis, y_min=0.35)
    axes[0].set_title("Training reward", loc="left", color=TEXT, fontsize=13, weight="bold")
    axes[0].set_ylabel("Mean progress reward", color=TEXT, fontsize=10)
    axes[1].set_title(
        "Held-out evaluation reward",
        loc="left",
        color=TEXT,
        fontsize=13,
        weight="bold",
    )
    save_figure(figure, "sft_four_phase_training_eval")


def main() -> None:
    plot_switch_training_eval()
    plot_four_phase()


if __name__ == "__main__":
    main()
