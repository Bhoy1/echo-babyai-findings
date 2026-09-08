#!/usr/bin/env python3
"""Summarize held-out behavior for the 200-step RL and ECHO 1.0 runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_ROOT = Path(
    "/tmp/babyai_alwayson_200/evaluation/posthoc_r3/runs"
)
OUTPUT_LIMIT = 1024

VARIANTS = {
    "rlonly": {"label": "RL", "color": "#4D555E", "marker": "o"},
    "echo100": {"label": "ECHO 1.0", "color": "#7651A8", "marker": "D"},
}

WINDOWS = tuple(
    ("0-25" if start == 0 else f"{start + 1}-{start + 25}", start + (start > 0), start + 25)
    for start in range(0, 200, 25)
)

TRANSCRIPT_RE = re.compile(r"(?im)^\s*(?:action|observation)\s*:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    return parser.parse_args()


def iter_traces(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "traces" in record:
                yield from record["traces"]
            else:
                yield record


def assistant_nodes(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in trace.get("nodes", [])
        if node.get("message", {}).get("role") == "assistant"
    ]


def has_transcript_like_output(nodes: list[dict[str, Any]]) -> bool:
    for node in nodes:
        content = str(node.get("message", {}).get("content", ""))
        lines = content.splitlines()
        if len(lines) > 1 and TRANSCRIPT_RE.search("\n".join(lines[1:])):
            return True
    return False


def trace_row(variant: str, step: int, trace: dict[str, Any]) -> dict[str, Any]:
    nodes = assistant_nodes(trace)
    completion_lengths = [
        int(node.get("usage", {}).get("completion_tokens", 0)) for node in nodes
    ]
    prompt_lengths = [
        int(node.get("usage", {}).get("prompt_tokens", 0)) for node in nodes
    ]
    metrics = trace.get("metrics", {})
    turns = float(metrics.get("babyai_turns", len(nodes)))
    return {
        "variant": variant,
        "step": step,
        "reward": float(trace.get("rewards", {}).get("progress_reward", 0.0)),
        "success": float(metrics.get("success", 0.0)),
        "turns": turns,
        "turn_limit_hit": float(turns >= 20),
        "completion_tokens": sum(completion_lengths),
        "prompt_tokens": sum(prompt_lengths),
        "total_tokens": sum(completion_lengths) + sum(prompt_lengths),
        "output_limit_hit": float(any(length >= OUTPUT_LIMIT for length in completion_lengths)),
        "transcript_like": float(has_transcript_like_output(nodes)),
    }


def mean(rows: list[dict[str, Any]], metric: str) -> float:
    return statistics.mean(float(row[metric]) for row in rows)


def main() -> None:
    args = parse_args()
    base_path = args.trace_root / "base/step_0/traces.jsonl"
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        step_paths = [(0, base_path)]
        step_paths.extend(
            (step, args.trace_root / variant / f"step_{step}/traces.jsonl")
            for step in range(5, 201, 5)
        )
        for step, path in step_paths:
            if not path.exists():
                raise FileNotFoundError(path)
            traces = list(iter_traces(path))
            if len(traces) != 84:
                raise ValueError(f"Expected 84 traces in {path}; found {len(traces)}")
            records.extend(trace_row(variant, step, trace) for trace in traces)

    metrics = (
        "reward",
        "success",
        "turns",
        "turn_limit_hit",
        "completion_tokens",
        "prompt_tokens",
        "total_tokens",
        "output_limit_hit",
        "transcript_like",
    )
    checkpoint_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for step in range(0, 201, 5):
            selected = [
                row for row in records if row["variant"] == variant and row["step"] == step
            ]
            checkpoint_rows.append(
                {
                    "variant": variant,
                    "objective": VARIANTS[variant]["label"],
                    "step": step,
                    **{metric: mean(selected, metric) for metric in metrics},
                }
            )

    window_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for label, low, high in WINDOWS:
            selected = [
                row
                for row in records
                if row["variant"] == variant and low <= row["step"] <= high
            ]
            window_rows.append(
                {
                    "variant": variant,
                    "objective": VARIANTS[variant]["label"],
                    "steps": label,
                    "trajectory_count": len(selected),
                    **{metric: mean(selected, metric) for metric in metrics},
                }
            )

    output_dir = ROOT / "data/eval_token_metrics"
    figure_dir = ROOT / "figures/token_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for path, rows in (
        (output_dir / "extended_200_behavior_by_checkpoint.csv", checkpoint_rows),
        (output_dir / "extended_200_behavior_by_quarter.csv", window_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=True)
    figure.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.98, hspace=0.30, wspace=0.23)
    panels = (
        ("reward", "Held-out reward", (0.45, 0.85), False),
        ("turns", "Mean turns", (8, 20.5), False),
        ("turn_limit_hit", "Reached 20-turn limit", (0, 1), True),
        ("output_limit_hit", "Hit 1,024-token output limit", (0, 0.55), True),
    )
    for axis, (metric, title, ylim, percent) in zip(axes.flat, panels):
        for variant, style in VARIANTS.items():
            rows = [row for row in checkpoint_rows if row["variant"] == variant]
            y = [100 * row[metric] if percent else row[metric] for row in rows]
            panel_ylim = (100 * ylim[0], 100 * ylim[1]) if percent else ylim
            axis.plot(
                [row["step"] for row in rows],
                y,
                color=style["color"],
                marker=style["marker"],
                markevery=2,
                markersize=3.5,
                linewidth=2.0,
                label=style["label"],
            )
            axis.set_ylim(*panel_ylim)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        axis.set_xlim(0, 200)
        axis.set_xticks([0, 50, 100, 150, 200])
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#AEB8C2")
        axis.spines["bottom"].set_color("#AEB8C2")
        axis.tick_params(colors="#66717D")
        if percent:
            axis.set_ylabel("Percent of trajectories")
        else:
            axis.set_ylabel("Reward" if metric == "reward" else "Turns")
        axis.set_xlabel("Training step")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=2, frameon=False)
    figure.suptitle("Extended 200-Step Behavior", fontsize=18, fontweight="bold", y=0.97)
    figure.text(
        0.5,
        0.925,
        "One training run per objective; each checkpoint contains 28 held-out tasks x 3 rollout replicates",
        ha="center",
        fontsize=10,
        color="#66717D",
    )
    for suffix in ("png", "svg"):
        figure.savefig(
            figure_dir / f"extended_200_behavior.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    print(output_dir / "extended_200_behavior_by_checkpoint.csv")
    print(output_dir / "extended_200_behavior_by_quarter.csv")
    print(figure_dir / "extended_200_behavior.png")


if __name__ == "__main__":
    main()
