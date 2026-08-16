#!/usr/bin/env python3
"""Build BabyAI behavior tables for three independent always-on runs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


VARIANTS = [
    "always_on/rl_only",
    "always_on/echo_0.05",
    "always_on/echo_0.5",
    "always_on/echo_1.0",
]
LABELS = {
    "always_on/rl_only": "RL-only",
    "always_on/echo_0.05": "ECHO 0.05",
    "always_on/echo_0.5": "ECHO 0.5",
    "always_on/echo_1.0": "ECHO 1.0",
}
COLORS = {
    "always_on/rl_only": "#333B45",
    "always_on/echo_0.05": "#2778A5",
    "always_on/echo_0.5": "#D1842D",
    "always_on/echo_1.0": "#43805E",
}
PHASES = [(1, 25), (26, 50), (51, 75), (76, 100)]
PHASE_LABELS = ["1-25", "26-50", "51-75", "76-100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--runs23", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path, independent_run: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row["run"] not in VARIANTS:
            continue
        normalized = dict(row)
        normalized["independent_run"] = int(
            independent_run if independent_run is not None else row["independent_run"]
        )
        normalized["step"] = int(row["step"])
        normalized["rollouts"] = int(row["rollouts"])
        for key, value in row.items():
            if key in {"run", "population", "step", "rollouts", "independent_run"}:
                continue
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                normalized[key] = math.nan
        selected.append(normalized)
    return selected


def weighted_mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    materialized = list(rows)
    denominator = sum(int(row["rollouts"]) for row in materialized)
    if not denominator:
        return math.nan
    return sum(float(row[key]) * int(row["rollouts"]) for row in materialized) / denominator


def invalid_action_rate(rows: Iterable[dict[str, Any]]) -> float:
    invalid_actions = 0.0
    turns = 0.0
    for row in rows:
        count = int(row["rollouts"])
        invalid_actions += float(row["invalid_actions_mean"]) * count
        turns += float(row["turns_mean"]) * count
    return invalid_actions / turns if turns else math.nan


def summarize_run(rows: list[dict[str, Any]], independent_run: int, run: str) -> dict[str, Any]:
    matching = [
        row
        for row in rows
        if row["independent_run"] == independent_run and row["run"] == run
    ]
    all_rows = [row for row in matching if row["population"] == "all_candidates"]
    retained_rows = [row for row in matching if row["population"] == "retained_trainable"]
    if len(all_rows) != 100 or len(retained_rows) != 100:
        raise ValueError(
            f"Expected 100 rows/population for run={independent_run} variant={run}; "
            f"got all={len(all_rows)} retained={len(retained_rows)}"
        )

    generated = sum(row["rollouts"] for row in all_rows)
    retained = sum(row["rollouts"] for row in retained_rows)
    summary: dict[str, Any] = {
        "independent_run": independent_run,
        "run": run,
        "generated_candidates": generated,
        "retained_trainable": retained,
        "retained_fraction": retained / generated,
        "extra_rollouts": generated - retained,
        "extra_per_step": (generated - retained) / 100,
        "candidates_per_retained": generated / retained,
        "mean_turns_all": weighted_mean(all_rows, "turns_mean"),
        "mean_turns_retained": weighted_mean(retained_rows, "turns_mean"),
        "turn_limit_rate_all": weighted_mean(all_rows, "ran_out_of_turns_rate"),
        "turn_limit_rate_retained": weighted_mean(
            retained_rows, "ran_out_of_turns_rate"
        ),
        "invalid_action_rate_all": invalid_action_rate(all_rows),
        "invalid_action_rate_retained": invalid_action_rate(retained_rows),
    }
    all_by_step = {row["step"]: row for row in all_rows}
    retained_by_step = {row["step"]: row for row in retained_rows}
    for (start, end), label in zip(PHASES, PHASE_LABELS):
        phase_all = [all_by_step[step] for step in range(start, end + 1)]
        phase_retained = [retained_by_step[step] for step in range(start, end + 1)]
        phase_generated = sum(row["rollouts"] for row in phase_all)
        phase_kept = sum(row["rollouts"] for row in phase_retained)
        summary[f"turns_{label}"] = weighted_mean(phase_retained, "turns_mean")
        summary[f"turn_limit_{label}"] = weighted_mean(
            phase_retained, "ran_out_of_turns_rate"
        )
        summary[f"extra_per_step_{label}"] = (phase_generated - phase_kept) / 25
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def moving_mean(values: list[float], window: int = 5) -> list[float]:
    return [
        statistics.fmean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def combined_summary(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for run in VARIANTS:
        matching = [row for row in run_summaries if row["run"] == run]
        output: dict[str, Any] = {"run": run, "independent_runs": len(matching)}
        for key in matching[0]:
            if key in {"independent_run", "run"}:
                continue
            mean, sd = mean_sd([float(row[key]) for row in matching])
            output[f"{key}_mean"] = mean
            output[f"{key}_sd"] = sd
        combined.append(output)
    return combined


def style_table(table: Any, row_count: int, *, font_size: float = 12.5) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
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
        cell.PAD = 0.035
    for row in range(1, row_count + 1):
        table[(row, 0)].get_text().set_ha("left")


def figure_header(figure: Any, title: str, subtitle: str) -> None:
    figure.text(0.015, 0.94, title, fontsize=22, fontweight="bold", color="#1f2d3d")
    figure.text(0.015, 0.875, subtitle, fontsize=12.5, color="#667085")


def save_turn_table(
    path: Path,
    title_suffix: str,
    summaries: list[dict[str, Any]],
    combined: bool,
) -> None:
    cell_rows: list[list[str]] = []
    for row in summaries:
        label = LABELS[row["run"]]
        if combined:
            turns = [
                f"{row[f'turns_{phase}_mean']:.2f} ± {row[f'turns_{phase}_sd']:.2f}"
                for phase in PHASE_LABELS
            ]
            turns.append(
                f"{row['mean_turns_retained_mean']:.2f} ± {row['mean_turns_retained_sd']:.2f}"
            )
            limits = [
                f"{row[f'turn_limit_{phase}_mean']:.1%} ± {row[f'turn_limit_{phase}_sd']:.1%}"
                for phase in PHASE_LABELS
            ]
            limits.append(
                f"{row['turn_limit_rate_retained_mean']:.1%} ± "
                f"{row['turn_limit_rate_retained_sd']:.1%}"
            )
        else:
            turns = [f"{row[f'turns_{phase}']:.2f}" for phase in PHASE_LABELS]
            turns.append(f"{row['mean_turns_retained']:.2f}")
            limits = [f"{row[f'turn_limit_{phase}']:.1%}" for phase in PHASE_LABELS]
            limits.append(f"{row['turn_limit_rate_retained']:.1%}")
        cell_rows.append([label, "Mean turns", *turns])
        cell_rows.append([label, "Turn-limit rate", *limits])

    figure, axis = plt.subplots(figsize=(16, 7.4))
    axis.axis("off")
    axis.set_position([0.015, 0.035, 0.97, 0.72])
    figure_header(
        figure,
        "Turn Length and Turn Limit Rate",
        "Max Turn Limit = 20 steps",
    )
    table = axis.table(
        cellText=cell_rows,
        colLabels=["Run", "Metric", "Steps 1-25", "26-50", "51-75", "76-100", "Overall"],
        cellLoc="right",
        colLoc="right",
        bbox=[0, 0, 1, 1],
        colWidths=[0.16, 0.18, 0.14, 0.14, 0.14, 0.14, 0.14],
    )
    style_table(table, len(cell_rows), font_size=11.5 if combined else 12.5)
    table[(0, 0)].get_text().set_ha("left")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_turn_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.4, 6.7))
    figure_header(
        figure,
        "Turn Length and Turn Limit Rate",
        "Max Turn Limit = 20 steps",
    )

    metric_specs = [
        ("turns_mean", "Mean turn length", "Mean turns", (10, 20), [10, 12, 14, 16, 18, 20]),
        (
            "ran_out_of_turns_rate",
            "Turn-limit rate",
            "Rollouts reaching turn limit (%)",
            (0, 100),
            [0, 20, 40, 60, 80, 100],
        ),
    ]
    legend_handles = []
    legend_labels = []
    for axis, (metric, panel_title, ylabel, ylim, yticks) in zip(axes, metric_specs):
        for run in VARIANTS:
            run_series: list[list[float]] = []
            for independent_run in (1, 2, 3):
                matching = sorted(
                    (
                        row
                        for row in rows
                        if row["independent_run"] == independent_run
                        and row["run"] == run
                        and row["population"] == "retained_trainable"
                    ),
                    key=lambda row: row["step"],
                )
                values = [float(row[metric]) for row in matching]
                if metric == "ran_out_of_turns_rate":
                    values = [100.0 * value for value in values]
                run_series.append(moving_mean(values))

            steps = list(range(1, 101))
            means = [statistics.fmean(values) for values in zip(*run_series)]
            deviations = [statistics.stdev(values) for values in zip(*run_series)]
            lower = [max(ylim[0], mean - deviation) for mean, deviation in zip(means, deviations)]
            upper = [min(ylim[1], mean + deviation) for mean, deviation in zip(means, deviations)]
            (line,) = axis.plot(
                steps,
                means,
                color=COLORS[run],
                linewidth=2.3,
                label=LABELS[run],
            )
            axis.fill_between(
                steps,
                lower,
                upper,
                color=COLORS[run],
                alpha=0.13,
                linewidth=0,
            )
            if axis is axes[0]:
                legend_handles.append(line)
                legend_labels.append(LABELS[run])

        axis.set_title(panel_title, fontsize=13, fontweight="bold", color="#253447", pad=10)
        axis.set_xlim(1, 100)
        axis.set_ylim(*ylim)
        axis.set_xticks([1, 20, 40, 60, 80, 100])
        axis.set_yticks(yticks)
        axis.set_xlabel("Training step", fontsize=10.5, color="#253447")
        axis.set_ylabel(ylabel, fontsize=10.5, color="#253447")
        axis.grid(axis="y", color="#dbe2ea", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#aeb8c2")
        axis.spines["bottom"].set_color("#aeb8c2")
        axis.tick_params(colors="#667085", labelsize=9.5)

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.80),
        ncol=4,
        frameon=False,
        fontsize=10.5,
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.68, wspace=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_filter_table(
    path: Path,
    title_suffix: str,
    summaries: list[dict[str, Any]],
    combined: bool,
) -> None:
    cell_rows: list[list[str]] = []
    for row in summaries:
        if combined:
            phases = [
                f"{row[f'extra_per_step_{phase}_mean']:.1f} ± "
                f"{row[f'extra_per_step_{phase}_sd']:.1f}"
                for phase in PHASE_LABELS
            ]
            total = f"{row['extra_rollouts_mean']:,.0f} ± {row['extra_rollouts_sd']:,.0f}"
            retained = (
                f"{row['retained_fraction_mean']:.1%} ± "
                f"{row['retained_fraction_sd']:.1%}"
            )
        else:
            phases = [f"{row[f'extra_per_step_{phase}']:.1f}" for phase in PHASE_LABELS]
            total = f"{row['extra_rollouts']:,.0f}"
            retained = f"{row['retained_fraction']:.1%}"
        cell_rows.append([LABELS[row["run"]], *phases, total, retained])

    figure, axis = plt.subplots(figsize=(16, 5.4))
    axis.axis("off")
    figure_header(
        figure,
        f"BabyAI · Rollout filtering overhead · {title_suffix}",
        "Extra rollouts = generated candidates − retained trainable rollouts; phase columns are per update",
    )
    table = axis.table(
        cellText=cell_rows,
        colLabels=["Run", "1-25", "26-50", "51-75", "76-100", "Total extra", "Retained %"],
        cellLoc="right",
        colLoc="right",
        bbox=[0.015, 0.05, 0.97, 0.65],
        colWidths=[0.18, 0.13, 0.13, 0.13, 0.13, 0.18, 0.16],
    )
    style_table(table, len(cell_rows), font_size=11.5 if combined else 12.5)
    table[(0, 0)].get_text().set_ha("left")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_filter_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(13.6, 6.8))
    figure_header(
        figure,
        "Rollout Filtering",
        "Usable samples = retained trainable rollouts ÷ generated candidates · 5-step moving mean",
    )

    for run in VARIANTS:
        run_series: list[list[float]] = []
        for independent_run in (1, 2, 3):
            matching = [
                row
                for row in rows
                if row["independent_run"] == independent_run and row["run"] == run
            ]
            all_by_step = {
                row["step"]: row["rollouts"]
                for row in matching
                if row["population"] == "all_candidates"
            }
            retained_by_step = {
                row["step"]: row["rollouts"]
                for row in matching
                if row["population"] == "retained_trainable"
            }
            usable = [
                100.0 * retained_by_step[step] / all_by_step[step]
                for step in range(1, 101)
            ]
            run_series.append(moving_mean(usable))

        steps = list(range(1, 101))
        means = [statistics.fmean(values) for values in zip(*run_series)]
        deviations = [statistics.stdev(values) for values in zip(*run_series)]
        lower = [max(0.0, mean - deviation) for mean, deviation in zip(means, deviations)]
        upper = [min(100.0, mean + deviation) for mean, deviation in zip(means, deviations)]
        axis.plot(
            steps,
            means,
            color=COLORS[run],
            linewidth=2.4,
            label=LABELS[run],
        )
        axis.fill_between(steps, lower, upper, color=COLORS[run], alpha=0.13, linewidth=0)

    axis.set_xlim(1, 100)
    axis.set_ylim(0, 100)
    axis.set_xticks([1, 20, 40, 60, 80, 100])
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.set_xlabel("Training step", fontsize=11, color="#253447")
    axis.set_ylabel("Usable samples (%)", fontsize=11, color="#253447")
    axis.grid(axis="y", color="#dbe2ea", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#aeb8c2")
    axis.spines["bottom"].set_color("#aeb8c2")
    axis.tick_params(colors="#667085", labelsize=10)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=4,
        frameon=False,
        fontsize=10.5,
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.76)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_behavior_table(path: Path, summaries: list[dict[str, Any]]) -> None:
    cell_rows: list[list[str]] = []
    for row in summaries:
        cell_rows.append(
            [
                LABELS[row["run"]],
                f"{row['invalid_action_rate_all_mean']:.1%} ± {row['invalid_action_rate_all_sd']:.1%}",
                f"{row['invalid_action_rate_retained_mean']:.1%} ± {row['invalid_action_rate_retained_sd']:.1%}",
                f"{row['mean_turns_all_mean']:.2f} ± {row['mean_turns_all_sd']:.2f}",
                f"{row['mean_turns_retained_mean']:.2f} ± {row['mean_turns_retained_sd']:.2f}",
                f"{row['turn_limit_rate_all_mean']:.1%} ± {row['turn_limit_rate_all_sd']:.1%}",
                f"{row['turn_limit_rate_retained_mean']:.1%} ± {row['turn_limit_rate_retained_sd']:.1%}",
            ]
        )
    figure, axis = plt.subplots(figsize=(17.2, 5.5))
    axis.axis("off")
    figure_header(
        figure,
        "BabyAI · Rollout behavior · three independent runs",
        "Mean ± ordinary sample SD across independent runs; metrics pool all 100 steps within each run first",
    )
    table = axis.table(
        cellText=cell_rows,
        colLabels=[
            "Run",
            "Invalid action\nall",
            "Invalid action\nretained",
            "Mean turns\nall",
            "Mean turns\nretained",
            "Turn limit\nall",
            "Turn limit\nretained",
        ],
        cellLoc="right",
        colLoc="right",
        bbox=[0.015, 0.04, 0.97, 0.68],
        colWidths=[0.17, 0.15, 0.16, 0.14, 0.16, 0.14, 0.16],
    )
    style_table(table, len(cell_rows), font_size=11.25)
    table[(0, 0)].get_text().set_ha("left")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.run1, independent_run=1) + read_rows(args.runs23)
    run_summaries = [
        summarize_run(rows, independent_run, run)
        for independent_run in (1, 2, 3)
        for run in VARIANTS
    ]
    combined = combined_summary(run_summaries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_independent_run_behavior_summary.csv", run_summaries)
    write_csv(args.output_dir / "three_run_behavior_mean_sd.csv", combined)
    write_csv(args.output_dir / "independent_runs_2_3_per_step_metrics.csv", read_rows(args.runs23))

    for independent_run in (2, 3):
        selected = [
            row for row in run_summaries if row["independent_run"] == independent_run
        ]
        save_turn_table(
            args.figure_dir / f"independent_{independent_run}_turns_table.png",
            f"independent run {independent_run}",
            selected,
            combined=False,
        )
        save_filter_table(
            args.figure_dir / f"independent_{independent_run}_filtering_table.png",
            f"independent run {independent_run}",
            selected,
            combined=False,
        )

    save_turn_plot(
        args.figure_dir / "three_independent_runs_turns_plot.png",
        rows,
    )
    save_filter_plot(
        args.figure_dir / "three_independent_runs_filtering_plot.png",
        rows,
    )
    save_behavior_table(
        args.figure_dir / "three_independent_runs_behavior_table.png", combined
    )
    print(f"Wrote summaries to {args.output_dir}")
    print(f"Wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
