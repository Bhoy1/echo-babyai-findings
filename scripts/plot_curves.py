#!/usr/bin/env python3
"""Generate publication-ready BabyAI always-on and switch curve figures."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CSV = ROOT / "data" / "training_behavior_metrics.csv"
ALWAYS_EVAL_CSV = ROOT / "data" / "checkpoint_eval_metrics.csv"
SWITCH_EVAL_CSV = ROOT / "data" / "babyai_switch50_eval_curve.csv"
THREE_RUN_TRAIN_CSV = (
    ROOT / "data" / "three_independent_runs" / "training_reward_mean_sd.csv"
)
THREE_RUN_EVAL_CSV = (
    ROOT / "data" / "three_independent_runs" / "eval_reward_mean_sd.csv"
)
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
    "rl_echo": "#0072B2",
    "echo_rl": "#D55E00",
    "reference_rl": "#6B7280",
    "reference_echo": "#80658B",
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

ALWAYS_TRAIN_BY_WEIGHT = {
    "0.05": "always_on/echo_0.05",
    "0.5": "always_on/echo_0.5",
    "1.0": "always_on/echo_1.0",
}

ALWAYS_EVAL_BY_WEIGHT = {
    "0.05": "echo005",
    "0.5": "echo050",
    "1.0": "echo100",
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


def load_three_run_means(path: Path) -> dict[str, list[tuple[int, float]]]:
    curves: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            curves[row["variant"]].append(
                (int(row["step"]), float(row["reward_mean"]))
            )
    for points in curves.values():
        points.sort()
    return curves


def style_axis(
    axis: plt.Axes,
    *,
    switch: bool = False,
    xlim: tuple[float, float] = (0, 100),
    ylim: tuple[float, float] = (0.35, 0.95),
    xticks: list[int] | None = None,
    yticks: list[float] | None = None,
) -> None:
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xticks(xticks or [0, 20, 40, 60, 80, 100])
    axis.set_yticks(yticks or [0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
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


def draw_reference(
    axis: plt.Axes,
    points: list[tuple[int, float]] | list[tuple[int, float, float]],
    *,
    color: str,
    smooth: bool,
    linestyle: str | tuple[int, tuple[int, ...]],
    marker: str | None = None,
    markevery: int | None = None,
) -> None:
    steps = [point[0] for point in points]
    values = [point[1] for point in points]
    if smooth:
        values = moving_mean(values)
    axis.plot(
        steps,
        values,
        color=color,
        linewidth=1.4,
        linestyle=linestyle,
        marker=marker,
        markersize=4.2 if marker else None,
        markerfacecolor="white" if marker else None,
        markeredgewidth=1.2 if marker else None,
        markevery=markevery,
        alpha=0.68,
        zorder=1,
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
    *,
    reference_training: dict[str, list[tuple[int, float]]] | None = None,
    reference_evaluation: dict[str, list[tuple[int, float]]] | None = None,
    view: str = "default",
) -> None:
    with_references = reference_training is not None and reference_evaluation is not None
    share_y = view != "second_half_zoom"
    figure, axes = plt.subplots(3, 2, figsize=(13.2, 13.1), sharex=True, sharey=share_y)
    figure.subplots_adjust(
        left=0.075, right=0.985, top=0.84, bottom=0.07, hspace=0.38, wspace=0.12
    )

    for row, weight in enumerate(("0.05", "0.5", "1.0")):
        train_axis, eval_axis = axes[row]
        if with_references:
            draw_reference(
                train_axis,
                reference_training["rlonly"],
                color=COLORS["reference_rl"],
                smooth=False,
                linestyle=(0, (6, 3)),
                marker="o",
                markevery=5,
            )
            draw_reference(
                train_axis,
                reference_training[ALWAYS_EVAL_BY_WEIGHT[weight]],
                color=COLORS["reference_echo"],
                smooth=False,
                linestyle=(0, (2, 3)),
                marker="D",
                markevery=5,
            )
            draw_reference(
                eval_axis,
                reference_evaluation["rlonly"],
                color=COLORS["reference_rl"],
                smooth=False,
                linestyle=(0, (6, 3)),
                marker="o",
                markevery=1,
            )
            draw_reference(
                eval_axis,
                reference_evaluation[ALWAYS_EVAL_BY_WEIGHT[weight]],
                color=COLORS["reference_echo"],
                smooth=False,
                linestyle=(0, (2, 3)),
                marker="D",
                markevery=1,
            )
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

        if view == "zero_to_one":
            axis_options = {
                "switch": True,
                "ylim": (0, 1),
                "yticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            }
            style_axis(train_axis, **axis_options)
            style_axis(eval_axis, **axis_options)
        elif view == "second_half_zoom":
            style_axis(
                train_axis,
                switch=False,
                xlim=(50, 100),
                ylim=(0.50, 0.80),
                xticks=[50, 60, 70, 80, 90, 100],
                yticks=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
            )
            style_axis(
                eval_axis,
                switch=False,
                xlim=(50, 100),
                ylim=(0.68, 0.90),
                xticks=[50, 60, 70, 80, 90, 100],
                yticks=[0.70, 0.75, 0.80, 0.85, 0.90],
            )
        else:
            style_axis(train_axis, switch=True)
            style_axis(eval_axis, switch=True)
        train_axis.text(
            -0.115,
            0.5,
            f"ECHO {weight}",
            transform=train_axis.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color=TEXT,
            clip_on=False,
        )
        if row < 2:
            train_axis.set_xlabel("")
            eval_axis.set_xlabel("")

    axes[0, 0].set_title(
        "Training reward",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        pad=10,
    )
    axes[0, 1].set_title(
        "Held-out evaluation",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        pad=10,
    )

    handles = [
        Line2D([0], [0], color=COLORS["rl_echo"], linewidth=2.35),
        Line2D([0], [0], color=COLORS["echo_rl"], linewidth=2.35),
    ]
    labels = ["RL to ECHO", "ECHO to RL"]
    if with_references:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=COLORS["reference_rl"],
                    linewidth=1.4,
                    linestyle=(0, (6, 3)),
                    marker="o",
                    markersize=4.2,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    alpha=0.68,
                ),
                Line2D(
                    [0],
                    [0],
                    color=COLORS["reference_echo"],
                    linewidth=1.4,
                    linestyle=(0, (2, 3)),
                    marker="D",
                    markersize=4.2,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    alpha=0.68,
                ),
            ]
        )
        labels.extend(
            ["Always RL (3-run mean)", "Always ECHO (3-run mean)"]
        )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=4 if with_references else 2,
        frameon=False,
        fontsize=10.5,
        handlelength=3.0,
        columnspacing=1.8 if with_references else 2.5,
    )
    title = "Switch schedules at step 50"
    if view == "second_half_zoom":
        title += " · steps 50–100"
    figure.suptitle(
        title,
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
    if with_references:
        figure.text(
            0.5,
            0.882,
            "Gray circles: always RL · Purple diamonds: always ECHO at the matching weight",
            ha="center",
            va="top",
            fontsize=9.2,
            color=MUTED,
        )
    name = "switch_curves_with_three_run_means" if with_references else "switch_curves"
    if view == "zero_to_one":
        name += "_zero_to_one"
    elif view == "second_half_zoom":
        name += "_second_half_zoom"
    save_figure(figure, name)
    plt.close(figure)


def save_switch_eval_summary(
    evaluation: dict[str, list[tuple[int, float, float]]],
) -> None:
    rows: list[list[str]] = []
    for weight in ("0.05", "0.5", "1.0"):
        for variant, (label, _) in SWITCH_EVAL[weight].items():
            points = evaluation[variant]
            best_after = max(
                (point for point in points if point[0] > 50),
                key=lambda point: point[1],
            )
            final = max(points, key=lambda point: point[0])
            rows.append(
                [
                    weight,
                    label.replace(" to ", " → "),
                    f"{best_after[1]:.3f}",
                    str(best_after[0]),
                    f"{final[1]:.3f}",
                ]
            )

    figure, axis = plt.subplots(figsize=(12.8, 5.3))
    axis.axis("off")
    axis.set_position([0.015, 0.09, 0.97, 0.665])
    figure.text(
        0.015,
        0.94,
        "Switch Schedule Evaluation Summary",
        fontsize=22,
        fontweight="bold",
        color="#1f2d3d",
    )
    figure.text(
        0.015,
        0.86,
        "Eval = mean of three rollouts",
        fontsize=12.5,
        color=MUTED,
    )
    table = axis.table(
        cellText=rows,
        colLabels=[
            "ECHO weight",
            "Schedule",
            "Best eval",
            "Best step",
            "Final eval",
        ],
        cellLoc="right",
        colLoc="right",
        bbox=[0, 0, 1, 1],
        colWidths=[0.17, 0.24, 0.21, 0.17, 0.21],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dbe2ea")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#203043")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            group_fill = {
                1: "#edf5fa",
                2: "#edf5fa",
                3: "#fdf4e8",
                4: "#fdf4e8",
                5: "#edf6f0",
                6: "#edf6f0",
            }
            cell.set_facecolor(group_fill[row])
            cell.get_text().set_color("#253447")
        cell.PAD = 0.04
    table[(0, 1)].get_text().set_ha("left")
    for row in range(1, len(rows) + 1):
        table[(row, 1)].get_text().set_ha("left")
    for row, column in ((4, 2), (4, 4), (5, 2), (5, 4)):
        table[(row, column)].get_text().set_weight("bold")

    figure.text(
        0.015,
        0.025,
        "Note: ECHO 0.05 schedules are effectively tied; both best and final eval differ by only 0.002.",
        fontsize=10.5,
        color=MUTED,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_DIR / "switch_eval_summary.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    training = load_training()
    always_eval = load_always_eval()
    base, switch_eval = load_switch_eval()
    three_run_training = load_three_run_means(THREE_RUN_TRAIN_CSV)
    three_run_eval = load_three_run_means(THREE_RUN_EVAL_CSV)
    save_switch_eval_summary(switch_eval)
    plot_always_on(training, always_eval)
    plot_switches(training, base, switch_eval)
    plot_switches(
        training,
        base,
        switch_eval,
        reference_training=three_run_training,
        reference_evaluation=three_run_eval,
    )
    plot_switches(
        training,
        base,
        switch_eval,
        reference_training=three_run_training,
        reference_evaluation=three_run_eval,
        view="zero_to_one",
    )
    plot_switches(
        training,
        base,
        switch_eval,
        reference_training=three_run_training,
        reference_evaluation=three_run_eval,
        view="second_half_zoom",
    )


if __name__ == "__main__":
    main()
