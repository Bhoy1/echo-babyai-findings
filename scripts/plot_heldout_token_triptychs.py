#!/usr/bin/env python3
"""Plot held-out observation, output, and total token usage."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/eval_token_metrics"
OUTPUT_DIR = ROOT / "figures/token_metrics"

COLORS = {
    "RL": "#333B45",
    "ECHO 0.05": "#2778A5",
    "ECHO 0.5": "#D1842D",
    "ECHO 1.0": "#43805E",
}
TEXT = "#20252B"
MUTED = "#66717D"
GRID = "#DDE3E8"
SPINE = "#AEB8C2"
PANELS = (
    ("observation_tokens", "Environment-observation tokens"),
    ("visible_output_tokens", "Assistant-output tokens"),
    ("total_tokens", "Total tokens processed"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE)
    axis.spines["bottom"].set_color(SPINE)
    axis.tick_params(colors=MUTED, labelsize=9.2)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def render(
    *,
    rows: list[dict[str, str]],
    objectives: list[str],
    periods: list[str],
    period_column: str,
    replicated: bool,
    title: str,
    subtitle: str,
    output_stem: str,
) -> None:
    indexed = {(row["objective"], row[period_column]): row for row in rows}
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), facecolor="white")
    x = np.arange(len(periods))
    width = min(0.8 / len(objectives), 0.28)

    for axis, (metric, panel_title) in zip(axes, PANELS, strict=True):
        for index, objective in enumerate(objectives):
            objective_rows = [indexed[(objective, period)] for period in periods]
            mean_key = f"{metric}_mean" if replicated else metric
            means = [float(row[mean_key]) for row in objective_rows]
            errors = (
                [float(row[f"{metric}_sd"]) for row in objective_rows]
                if replicated
                else None
            )
            offset = (index - (len(objectives) - 1) / 2) * width
            axis.bar(
                x + offset,
                means,
                width,
                yerr=errors,
                capsize=2.5 if replicated else 0,
                color=COLORS[objective],
                edgecolor="white",
                linewidth=0.5,
                error_kw={"elinewidth": 0.8, "ecolor": "#59636D"},
                label=objective,
            )
        axis.set_title(panel_title, fontsize=12, fontweight="bold", color=TEXT, pad=9)
        axis.set_xticks(x, periods)
        if len(periods) > 4:
            axis.tick_params(axis="x", labelrotation=35)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
        axis.set_xlabel("Training steps", fontsize=9.8, color=MUTED, labelpad=7)
        axis.set_ylabel("Mean tokens per held-out trajectory", fontsize=9.8, color=MUTED)
        axis.set_ylim(bottom=0)
        style_axis(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(objectives),
        frameon=False,
        fontsize=9.5,
    )
    figure.suptitle(title, y=0.995, fontsize=16, fontweight="bold", color=TEXT)
    figure.text(
        0.5,
        0.948,
        subtitle,
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    bottom = 0.19 if len(periods) > 4 else 0.13
    figure.subplots_adjust(left=0.055, right=0.99, bottom=bottom, top=0.80, wspace=0.29)

    for suffix in ("png", "svg"):
        figure.savefig(
            OUTPUT_DIR / f"{output_stem}.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render(
        rows=read_csv(DATA_DIR / "heldout_token_metrics_three_run_by_quarter.csv"),
        objectives=["RL", "ECHO 0.05", "ECHO 0.5", "ECHO 1.0"],
        periods=["0-25", "26-50", "51-75", "76-100"],
        period_column="quarter",
        replicated=True,
        title="Held-Out Token Usage",
        subtitle="Mean per trajectory; error bars show +/- 1 SD across three independent training runs",
        output_stem="heldout_observation_output_total_tokens",
    )
    render(
        rows=read_csv(DATA_DIR / "extended_200_token_metrics_by_quarter.csv"),
        objectives=["RL", "ECHO 1.0"],
        periods=[
            "0-25",
            "26-50",
            "51-75",
            "76-100",
            "101-125",
            "126-150",
            "151-175",
            "176-200",
        ],
        period_column="steps",
        replicated=False,
        title="Held-Out Token Usage Through 200 Steps",
        subtitle="Mean per trajectory; one training run per objective",
        output_stem="extended_200_observation_output_total_tokens",
    )


if __name__ == "__main__":
    main()
