#!/usr/bin/env python3
"""Render compact token-usage tables from the checked-in summary CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/eval_token_metrics"
FIGURE_DIR = ROOT / "figures/token_metrics"

COLUMNS = [
    ("Objective", None),
    ("Steps", None),
    ("Mean\nturns", "turns"),
    ("System", "system_tokens"),
    ("Initial\nprompt", "initial_prompt_tokens"),
    ("Environment\nobservations", "observation_tokens"),
    ("Visible\noutput", "visible_output_tokens"),
    ("Total tokens\nprocessed", "total_tokens"),
]
COL_WIDTHS = [0.12, 0.075, 0.09, 0.105, 0.145, 0.18, 0.12, 0.165]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_table(
    *,
    rows: list[list[str]],
    fills: dict[str, str],
    title: str,
    subtitle: str,
    output_stem: str,
) -> None:
    figure = plt.figure(figsize=(15.2, 6.05), facecolor="white")
    axis = figure.add_axes([0.012, 0.015, 0.976, 0.825])
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=[column[0] for column in COLUMNS],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=COL_WIDTHS,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(1, 1.62)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD3DA")
        cell.set_linewidth(0.55)
        if row_index == 0:
            cell.set_facecolor("#27313A")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.088)
            continue

        objective = rows[row_index - 1][0]
        cell.set_facecolor(fills[objective])
        if col_index == 0:
            cell.set_text_props(weight="bold")

    figure.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#20252B",
    )
    figure.text(
        0.5,
        0.905,
        subtitle,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#66717D",
    )

    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURE_DIR / f"{output_stem}.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    plt.close(figure)


def render_output_limit_table() -> None:
    source_rows = read_csv(DATA_DIR / "heldout_output_limit_behavior_by_quarter.csv")
    quarters = ["0-25", "26-50", "51-75", "76-100"]
    objectives = ["RL", "ECHO 0.05", "ECHO 0.5", "ECHO 1.0"]
    rates = {
        (row["objective"], row["quarter"]): 100 * float(row["limit_hit_rate_mean"])
        for row in source_rows
    }
    rows = [
        [objective, *[f"{rates[(objective, quarter)]:.1f}%" for quarter in quarters]]
        for objective in objectives
    ]

    figure = plt.figure(figsize=(11.8, 2.85), facecolor="white")
    axis = figure.add_axes([0.055, 0.025, 0.89, 0.66])
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Objective", *quarters],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.24, 0.19, 0.19, 0.19, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1, 1.75)

    fills = {
        "RL": "#F0F2F4",
        "ECHO 0.05": "#E7F2F8",
        "ECHO 0.5": "#FAEFE1",
        "ECHO 1.0": "#E9F2EC",
    }
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD3DA")
        cell.set_linewidth(0.65)
        if row_index == 0:
            cell.set_facecolor("#27313A")
            cell.set_text_props(color="white", weight="bold")
            continue
        objective = rows[row_index - 1][0]
        cell.set_facecolor(fills[objective])
        if col_index == 0:
            cell.set_text_props(weight="bold")

    figure.text(
        0.5,
        0.965,
        "Held-Out Trajectories Hitting the Output Limit",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#20252B",
    )
    figure.text(
        0.5,
        0.815,
        "Percentage containing at least one 1,024-token completion; mean across three independent training runs",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#66717D",
    )
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURE_DIR / f"heldout_output_limit_behavior.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    plt.close(figure)


def primary_rows() -> list[list[str]]:
    rows = read_csv(DATA_DIR / "heldout_token_metrics_three_run_by_quarter.csv")
    rendered: list[list[str]] = []
    for row in rows:
        values = [row["objective"], row["quarter"]]
        for _, metric in COLUMNS[2:]:
            precision = 1 if metric == "turns" else 0
            values.append(
                f"{float(row[f'{metric}_mean']):,.{precision}f} +/- "
                f"{float(row[f'{metric}_sd']):,.{precision}f}"
            )
        rendered.append(values)
    return rendered


def extended_rows() -> list[list[str]]:
    rows = read_csv(DATA_DIR / "extended_200_token_metrics_by_quarter.csv")
    rendered: list[list[str]] = []
    for row in rows:
        values = [row["objective"], row["steps"]]
        for _, metric in COLUMNS[2:]:
            precision = 1 if metric == "turns" else 0
            values.append(f"{float(row[metric]):,.{precision}f}")
        rendered.append(values)
    return rendered


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    render_table(
        rows=primary_rows(),
        fills={
            "RL": "#F0F2F4",
            "ECHO 0.05": "#E7F2F8",
            "ECHO 0.5": "#FAEFE1",
            "ECHO 1.0": "#E9F2EC",
        },
        title="Held-Out Evaluation Token Usage",
        subtitle="Mean tokens processed per trajectory +/- SD across three independent training runs",
        output_stem="heldout_token_metrics_three_run_by_quarter",
    )
    render_table(
        rows=extended_rows(),
        fills={"RL": "#F0F2F4", "ECHO 1.0": "#EEE9F5"},
        title="Held-Out Evaluation Token Usage Through 200 Steps",
        subtitle="Mean tokens processed per trajectory; one training run per objective",
        output_stem="extended_200_token_metrics_by_quarter",
    )
    render_output_limit_table()


if __name__ == "__main__":
    main()
