#!/usr/bin/env python3
"""Plot held-out output-limit rates and summarize the observed failure mode."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
REMOTE_ROOT = Path("/tmp/babyai_eval_token_traces/evals")
OUTPUT_LIMIT = 1024

VARIANTS = {
    "rlonly": "RL",
    "echo005": "ECHO 0.05",
    "echo050": "ECHO 0.5",
    "echo100": "ECHO 1.0",
}
WINDOWS = (
    ("0-25", 0, 25),
    ("26-50", 26, 50),
    ("51-75", 51, 75),
    ("76-100", 76, 100),
)


def trace_roots() -> dict[int, dict[str, Path]]:
    run1_a = PROJECTS / "Echo_Jericho/artifacts/babyai_matched_checkpoint_eval_r3/runs"
    run1_b = PROJECTS / "Echo_Jericho/artifacts/babyai_matched_echo050_echo100_eval_r3/runs"
    roots: dict[int, dict[str, Path]] = {
        1: {
            "base": run1_b / "base",
            "rlonly": run1_a / "rlonly",
            "echo005": run1_a / "echo005",
            "echo050": run1_b / "echo050",
            "echo100": run1_b / "echo100",
        }
    }
    for run in (2, 3):
        prefix = REMOTE_ROOT / f"independent_{run}"
        roots[run] = {
            "base": prefix / "echo050_echo100/runs/base",
            "rlonly": prefix / "rlonly_echo005/runs/rlonly",
            "echo005": prefix / "rlonly_echo005/runs/echo005",
            "echo050": prefix / "echo050_echo100/runs/echo050",
            "echo100": prefix / "echo050_echo100/runs/echo100",
        }
    return roots


def iter_traces(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield from json.loads(line).get("traces", [])


def completion_lengths(trace: dict[str, Any]) -> list[int]:
    return [
        int(call.get("usage", {}).get("completion_tokens", 0))
        for call in trace.get("calls", [])
    ]


def main() -> None:
    roots = trace_roots()
    per_run: dict[tuple[int, str, str], dict[str, float]] = {}

    for run in (1, 2, 3):
        for variant in VARIANTS:
            for window, low, high in WINDOWS:
                paths = []
                if low == 0:
                    paths.append(roots[run]["base"] / "step_0/traces.jsonl")
                paths.extend(
                    roots[run][variant] / f"step_{step}/traces.jsonl"
                    for step in range(5, 101, 5)
                    if low <= step <= high
                )

                trajectory_outputs: list[int] = []
                hit_outputs: list[int] = []
                normal_outputs: list[int] = []
                for path in paths:
                    if not path.exists():
                        raise FileNotFoundError(path)
                    for trace in iter_traces(path):
                        lengths = completion_lengths(trace)
                        output_total = sum(lengths)
                        trajectory_outputs.append(output_total)
                        if any(length >= OUTPUT_LIMIT for length in lengths):
                            hit_outputs.append(output_total)
                        else:
                            normal_outputs.append(output_total)

                per_run[(run, variant, window)] = {
                    "trajectories": float(len(trajectory_outputs)),
                    "limit_hit_rate": len(hit_outputs) / len(trajectory_outputs),
                    "visible_output_mean": statistics.mean(trajectory_outputs),
                    "normal_output_mean": statistics.mean(normal_outputs),
                    "limit_hit_output_mean": (
                        statistics.mean(hit_outputs) if hit_outputs else float("nan")
                    ),
                }

    summary: list[dict[str, Any]] = []
    for variant, objective in VARIANTS.items():
        for window, _, _ in WINDOWS:
            row: dict[str, Any] = {
                "variant": variant,
                "objective": objective,
                "quarter": window,
            }
            for metric in (
                "limit_hit_rate",
                "visible_output_mean",
                "normal_output_mean",
            ):
                values = [per_run[(run, variant, window)][metric] for run in (1, 2, 3)]
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_sd"] = statistics.stdev(values)

            pooled_hit_means = [
                per_run[(run, variant, window)]["limit_hit_output_mean"]
                for run in (1, 2, 3)
                if per_run[(run, variant, window)]["limit_hit_output_mean"]
                == per_run[(run, variant, window)]["limit_hit_output_mean"]
            ]
            row["limit_hit_output_mean"] = (
                statistics.mean(pooled_hit_means) if pooled_hit_means else float("nan")
            )
            summary.append(row)

    data_dir = ROOT / "data/eval_token_metrics"
    figure_dir = ROOT / "figures/token_metrics"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "heldout_output_limit_behavior_by_quarter.csv"
    fields = list(summary[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    rates = {
        (row["objective"], row["quarter"]): row["limit_hit_rate_mean"] * 100
        for row in summary
    }
    rows = []
    for objective in VARIANTS.values():
        rows.append(
            [objective]
            + [f"{rates[(objective, window)]:.1f}%" for window, _, _ in WINDOWS]
        )

    figure = plt.figure(figsize=(12.2, 8.1), facecolor="white")
    table_axis = figure.add_axes([0.055, 0.47, 0.89, 0.36])
    table_axis.axis("off")
    table = table_axis.table(
        cellText=rows,
        colLabels=["Objective", *[window for window, _, _ in WINDOWS]],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.24, 0.19, 0.19, 0.19, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1, 1.85)

    header = "#27313A"
    fills = {
        "RL": "#F0F2F4",
        "ECHO 0.05": "#E7F2F8",
        "ECHO 0.5": "#FAEFE1",
        "ECHO 1.0": "#E9F2EC",
    }
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD3DA")
        cell.set_linewidth(0.7)
        if row_index == 0:
            cell.set_facecolor(header)
            cell.set_text_props(color="white", weight="bold")
        else:
            objective = rows[row_index - 1][0]
            cell.set_facecolor(fills[objective])
            if col_index == 0:
                cell.set_text_props(weight="bold")

    figure.text(
        0.5,
        0.945,
        "Held-Out Trajectories Hitting the Output Limit",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#20252B",
    )
    figure.text(
        0.5,
        0.895,
        "Percentage containing at least one 1,024-token completion; mean across three independent training runs",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#66717D",
    )

    figure.add_artist(
        plt.Line2D([0.07, 0.93], [0.415, 0.415], color="#CBD3DA", linewidth=1.0)
    )
    figure.add_artist(
        plt.Line2D([0.075, 0.075], [0.215, 0.365], color="#43805E", linewidth=4.0)
    )
    figure.text(
        0.105,
        0.365,
        "Interpretation",
        fontsize=11.5,
        fontweight="bold",
        color="#20252B",
        ha="left",
        va="top",
    )
    figure.text(
        0.105,
        0.325,
        "At high ECHO weights, the model sometimes generated a plausible-looking internal rollout, including both actions and observations.\n"
        "However, its imagined observations rarely matched the simulator, and trajectories exhibiting this behavior performed substantially\n"
        "worse. This appears to be transcript imitation or role confusion rather than accurate environment prediction.",
        fontsize=10.4,
        color="#30363C",
        ha="left",
        va="top",
        linespacing=1.45,
    )
    figure.text(
        0.085,
        0.155,
        "\u2022",
        fontsize=17,
        color="#D1842D",
        ha="center",
        va="center",
    )
    figure.text(
        0.11,
        0.155,
        "At ECHO 1.0 during steps 51-75, normal trajectories averaged 70 output tokens; limit-hit trajectories averaged 6,327.",
        fontsize=10.1,
        color="#30363C",
        ha="left",
        va="center",
    )
    figure.text(
        0.085,
        0.105,
        "\u2022",
        fontsize=17,
        color="#2778A5",
        ha="center",
        va="center",
    )
    figure.text(
        0.11,
        0.105,
        "The parser executed the first line, while the complete response remained in the trajectory and later context.",
        fontsize=10.1,
        color="#30363C",
        ha="left",
        va="center",
    )

    figure.text(
        0.5,
        0.035,
        "Each checkpoint evaluates 28 held-out tasks x 3 rollout replicates. Output limits are measured from exact API usage records.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#66717D",
    )

    for suffix in ("png", "svg"):
        figure.savefig(
            figure_dir / f"heldout_output_limit_behavior.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    print(f"wrote {csv_path}")
    print(f"wrote {figure_dir / 'heldout_output_limit_behavior.png'}")


if __name__ == "__main__":
    main()
