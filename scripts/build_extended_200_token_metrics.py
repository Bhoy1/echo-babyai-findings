#!/usr/bin/env python3
"""Build the held-out token-usage table for the 200-step RL/ECHO runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_ROOT = Path(
    "/tmp/babyai_alwayson_200/evaluation/posthoc_r3/runs"
)
DEFAULT_TOKENIZER = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots"
    / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)

VARIANTS = {
    "rlonly": "RL",
    "echo100": "ECHO 1.0",
}
WINDOWS = tuple(
    (
        "0-25" if start == 0 else f"{start + 1}-{start + 25}",
        start + (start > 0),
        start + 25,
    )
    for start in range(0, 200, 25)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
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


def ancestors(nodes: list[dict[str, Any]], node_index: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    parent = nodes[node_index].get("parent")
    while parent is not None:
        result.append(nodes[parent])
        parent = nodes[parent].get("parent")
    result.reverse()
    return result


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    @lru_cache(maxsize=None)
    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

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

            for trace in traces:
                nodes = trace.get("nodes", [])
                totals: dict[str, int] = defaultdict(int)
                turns = 0
                for node_index, node in enumerate(nodes):
                    message = node.get("message", {})
                    if message.get("role") != "assistant":
                        continue
                    usage = node.get("usage") or {}
                    if not usage:
                        continue
                    turns += 1
                    prompt_total = int(usage.get("prompt_tokens", 0))
                    completion_total = int(usage.get("completion_tokens", 0))
                    prompt_content = 0
                    first_user_seen = False
                    for ancestor in ancestors(nodes, node_index):
                        ancestor_message = ancestor.get("message", {})
                        role = ancestor_message.get("role")
                        content = str(ancestor_message.get("content", ""))
                        count = token_count(content)
                        prompt_content += count
                        if role == "system":
                            totals["system_tokens"] += count
                        elif role == "user" and not first_user_seen:
                            totals["initial_prompt_tokens"] += count
                            first_user_seen = True
                        elif role == "user":
                            totals["observation_tokens"] += count
                        elif role == "assistant":
                            totals["prior_action_tokens"] += count

                    totals["format_tokens"] += max(0, prompt_total - prompt_content)
                    totals["visible_output_tokens"] += completion_total
                    totals["prompt_tokens"] += prompt_total
                    totals["completion_tokens"] += completion_total

                records.append(
                    {
                        "variant": variant,
                        "step": step,
                        "turns": turns,
                        "system_tokens": totals["system_tokens"],
                        "initial_prompt_tokens": totals["initial_prompt_tokens"],
                        "observation_tokens": totals["observation_tokens"],
                        "prior_action_tokens": totals["prior_action_tokens"],
                        "format_tokens": totals["format_tokens"],
                        "visible_output_tokens": totals["visible_output_tokens"],
                        "prompt_tokens": totals["prompt_tokens"],
                        "completion_tokens": totals["completion_tokens"],
                        "total_tokens": totals["prompt_tokens"]
                        + totals["completion_tokens"],
                    }
                )

    metric_names = (
        "turns",
        "system_tokens",
        "initial_prompt_tokens",
        "observation_tokens",
        "prior_action_tokens",
        "format_tokens",
        "visible_output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )
    summary_rows: list[dict[str, Any]] = []
    for variant, objective in VARIANTS.items():
        for window, low, high in WINDOWS:
            selected = [
                row
                for row in records
                if row["variant"] == variant and low <= row["step"] <= high
            ]
            summary_rows.append(
                {
                    "variant": variant,
                    "objective": objective,
                    "steps": window,
                    "trajectory_count": len(selected),
                    **{
                        metric: statistics.mean(float(row[metric]) for row in selected)
                        for metric in metric_names
                    },
                }
            )

    output_dir = ROOT / "data/eval_token_metrics"
    figure_dir = ROOT / "figures/token_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "extended_200_token_metrics_by_quarter.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    columns = [
        ("Objective", None),
        ("Steps", None),
        ("Mean\nturns", "turns"),
        ("System", "system_tokens"),
        ("Initial\nprompt", "initial_prompt_tokens"),
        ("Environment\nobservations", "observation_tokens"),
        ("Visible\noutput", "visible_output_tokens"),
        ("Total tokens\nprocessed", "total_tokens"),
    ]
    table_rows: list[list[str]] = []
    for row in summary_rows:
        rendered = [row["objective"], row["steps"]]
        for _, metric in columns[2:]:
            precision = 1 if metric == "turns" else 0
            rendered.append(f"{row[metric]:,.{precision}f}")
        table_rows.append(rendered)

    figure, axis = plt.subplots(figsize=(15.2, 6.25))
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=[column[0] for column in columns],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.075, 0.09, 0.105, 0.145, 0.18, 0.12, 0.165],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(1, 1.72)
    fills = {"RL": "#F0F2F4", "ECHO 1.0": "#EEE9F5"}
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD3DA")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#27313A")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.085)
        else:
            objective = table_rows[row_index - 1][0]
            cell.set_facecolor(fills[objective])
            if col_index == 0:
                cell.set_text_props(weight="bold")

    figure.text(
        0.5,
        0.975,
        "Held-Out Evaluation Token Usage Through 200 Steps",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#20252B",
    )
    figure.text(
        0.5,
        0.925,
        "Mean tokens processed per trajectory; one training run per objective",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#66717D",
    )
    figure.subplots_adjust(left=0.015, right=0.985, top=0.89, bottom=0.015)
    for suffix in ("png", "svg"):
        figure.savefig(
            figure_dir / f"extended_200_token_metrics_by_quarter.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    print(csv_path)
    print(figure_dir / "extended_200_token_metrics_by_quarter.png")


if __name__ == "__main__":
    main()
