#!/usr/bin/env python3
"""Generate publication-ready BabyAI always-on and switch curve figures."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CSV = ROOT / "data" / "training_behavior_metrics.csv"
ALWAYS_EVAL_CSV = ROOT / "data" / "checkpoint_eval_metrics.csv"
SWITCH_EVAL_CSV = ROOT / "data" / "babyai_switch50_eval_curve.csv"
FIGURE_DIR = ROOT / "figures"

TEXT = "#20252B"
MUTED = "#66717D"
GRID = "#DDE3E8"
SPINE = "#AEB8C2"
SWITCH = "#7D8790"

COLORS = {
    "rl": "#333B45",
    "echo005": "#2778A5",
    "echo050": "#D1842D",
    "echo100": "#43805E",
    "rl_echo": "#2778A5",
    "echo_rl": "#C65368",
}

ALWAYS_TRAIN = {
    "always_on/rl_only": ("RL-only", "rl"),
    "always_on/echo_0.05": ("ECHO 0.05", "echo005"),
    "always_on/echo_0.5": ("ECHO 0.5", "echo050"),
    "always_on/echo_1.0": ("ECHO 1.0", "echo100"),
}

ALWAYS_EVAL = {
    "rlonly": ("RL-only", "rl"),
    "echo005": ("ECHO 0.05", "echo005"),
    "echo050": ("ECHO 0.5", "echo050"),
    "echo100": ("ECHO 1.0", "echo100"),
}

SWITCH_TRAIN = {
    "0.05": {
        "switch50/rl50_then_echo_0.05": ("RL to ECHO", "rl_echo"),
        "switch50/echo_0.05_then_rl50": ("ECHO to RL", "echo_rl"),
    },
    "0.5": {
        "switch50/rl50_then_echo_0.5": ("RL to ECHO", "rl_echo"),
        "switch50/echo_0.5_then_rl50": ("ECHO to RL", "echo_rl"),
    },
    "1.0": {
        "switch50/rl50_then_echo_1.0": ("RL to ECHO", "rl_echo"),
        "switch50/echo_1.0_then_rl50": ("ECHO to RL", "echo_rl"),
    },
}

SWITCH_EVAL = {
    "0.05": {
        "rl50_echo005": ("RL to ECHO", "rl_echo"),
        "echo005_rl50": ("ECHO to RL", "echo_rl"),
    },
    "0.5": {
        "rl50_echo050": ("RL to ECHO", "rl_echo"),
        "echo050_rl50": ("ECHO to RL", "echo_rl"),
    },
    "1.0": {
        "rl50_echo100": ("RL to ECHO", "rl_echo"),
        "echo100_rl50": ("ECHO to RL", "echo_rl"),
    },
}


def moving_mean(values: list[float], window: int = 5) -> list[float]:
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / len(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def load_training() -> dict[str, list[tuple[int, float]]]:
    curves: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with TRAINING_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["population"] != "retained_trainable":
                continue
            curves[row["run"]].append((int(row["step"]), float(row["reward_mean"])))
    for points in curves.values():
        points.sort()
    return curves


def load_always_eval() -> dict[str, list[tuple[int, float, float]]]:
    curves: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with ALWAYS_EVAL_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            curves[row["variant"]].append(
                (int(row["step"]), float(row["reward_mean"]), float(row["reward_sd"]))
            )
    for points in curves.values():
        points.sort()
    return curves


def load_switch_eval() -> tuple[
    tuple[int, float, float], dict[str, list[tuple[int, float, float]]]
]:
    curves: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    base: tuple[int, float, float] | None = None
    with SWITCH_EVAL_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            point = (
                int(row["step"]),
                float(row["mean_reward"]),
                float(row["replicate_sd"]),
            )
            if row["variant"] == "base":
                base = point
            else:
                curves[row["variant"]].append(point)
    if base is None:
        raise ValueError("Switch evaluation CSV is missing the step-0 baseline")
    for points in curves.values():
        points.sort()
    return base, curves


def style_axis(axis: plt.Axes, *, switch: bool = False) -> None:
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
    if switch:
        axis.axvline(50, color=SWITCH, linestyle="--", linewidth=1.2, zorder=1)
        axis.text(
            50,
            0.98,
            "switch",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color=MUTED,
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )


def draw_training(
    axis: plt.Axes,
    points: list[tuple[int, float]],
    *,
    label: str,
    color: str,
) -> None:
    steps = [point[0] for point in points]
    rewards = [point[1] for point in points]
    axis.plot(
        steps,
        moving_mean(rewards),
        color=color,
        linewidth=2.35,
        label=label,
    )


def draw_eval(
    axis: plt.Axes,
    points: list[tuple[int, float, float]],
    *,
    label: str,
    color: str,
) -> None:
    steps = [point[0] for point in points]
    means = [point[1] for point in points]
    deviations = [point[2] for point in points]
    axis.fill_between(
        steps,
        [mean - deviation for mean, deviation in zip(means, deviations, strict=True)],
        [mean + deviation for mean, deviation in zip(means, deviations, strict=True)],
        color=color,
        alpha=0.12,
        linewidth=0,
    )
    axis.plot(
        steps,
        means,
        color=color,
        linewidth=2.15,
        marker="o",
        markersize=3.2,
        label=label,
    )


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        figure.savefig(
            FIGURE_DIR / f"{name}{suffix}",
            dpi=220 if suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.18,
            facecolor="white",
        )


def plot_always_on(
    training: dict[str, list[tuple[int, float]]],
    evaluation: dict[str, list[tuple[int, float, float]]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), sharey=True)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.70, bottom=0.14, wspace=0.12)

    for run, (label, color_key) in ALWAYS_TRAIN.items():
        draw_training(
            axes[0], training[run], label=label, color=COLORS[color_key]
        )
    for variant, (label, color_key) in ALWAYS_EVAL.items():
        draw_eval(
            axes[1], evaluation[variant], label=label, color=COLORS[color_key]
        )

    for axis in axes:
        style_axis(axis)
    axes[0].set_title("Training reward", fontsize=14, fontweight="bold", color=TEXT)
    axes[1].set_title(
        "Held-out evaluation", fontsize=14, fontweight="bold", color=TEXT
    )
    axes[0].set_ylabel("Progress reward", fontsize=10.5, color=TEXT)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.815),
        ncol=4,
        frameon=False,
        fontsize=10,
        handlelength=2.8,
        columnspacing=1.8,
    )
    figure.suptitle(
        "Always-on objectives",
        x=0.075,
        y=0.97,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.985,
        0.967,
        "Train: 5-step moving mean   |   Eval: mean ± 1 SD",
        ha="right",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    save_figure(figure, "always_on_curves")
    plt.close(figure)


def plot_switches(
    training: dict[str, list[tuple[int, float]]],
    base: tuple[int, float, float],
    evaluation: dict[str, list[tuple[int, float, float]]],
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(13.2, 13.1), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.075, right=0.985, top=0.87, bottom=0.07, hspace=0.34, wspace=0.12
    )

    for row, weight in enumerate(("0.05", "0.5", "1.0")):
        train_axis, eval_axis = axes[row]
        for run, (label, color_key) in SWITCH_TRAIN[weight].items():
            draw_training(
                train_axis,
                training[run],
                label=label,
                color=COLORS[color_key],
            )
        for variant, (label, color_key) in SWITCH_EVAL[weight].items():
            draw_eval(
                eval_axis,
                [base, *evaluation[variant]],
                label=label,
                color=COLORS[color_key],
            )

        style_axis(train_axis, switch=True)
        style_axis(eval_axis, switch=True)
        train_axis.set_title(
            f"ECHO {weight} - training reward",
            fontsize=13.5,
            fontweight="bold",
            color=TEXT,
        )
        eval_axis.set_title(
            f"ECHO {weight} - held-out evaluation",
            fontsize=13.5,
            fontweight="bold",
            color=TEXT,
        )
        train_axis.set_ylabel("Progress reward", fontsize=10.5, color=TEXT)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=3.0,
        columnspacing=2.5,
    )
    figure.suptitle(
        "Switch schedules at step 50",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.985,
        0.972,
        "Train: 5-step moving mean   |   Eval: mean ± 1 SD",
        ha="right",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    save_figure(figure, "switch_curves")
    plt.close(figure)


def main() -> None:
    training = load_training()
    always_eval = load_always_eval()
    base, switch_eval = load_switch_eval()
    plot_always_on(training, always_eval)
    plot_switches(training, base, switch_eval)


if __name__ == "__main__":
    main()
