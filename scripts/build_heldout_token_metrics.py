#!/usr/bin/env python3
"""Summarize held-out evaluation token usage across three independent runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
DEFAULT_REMOTE_ROOT = Path("/tmp/babyai_eval_token_traces/evals")
DEFAULT_TOKENIZER = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots"
    / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)

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
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    return parser.parse_args()


def trace_roots(remote_root: Path) -> dict[int, dict[str, Path]]:
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
        prefix = remote_root / f"independent_{run}"
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
            record = json.loads(line)
            yield from record.get("traces", [])


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

    records: list[dict[str, Any]] = []
    roots = trace_roots(args.remote_root)
    for independent_run, variants in roots.items():
        for variant in VARIANTS:
            step_paths = [(0, variants["base"] / "step_0/traces.jsonl")]
            step_paths.extend(
                (step, variants[variant] / f"step_{step}/traces.jsonl")
                for step in range(5, 101, 5)
            )
            for step, path in step_paths:
                if not path.exists():
                    raise FileNotFoundError(path)
                for trace in iter_traces(path):
                    nodes = trace.get("nodes", [])
                    totals = defaultdict(int)
                    for call in trace.get("calls", []):
                        usage = call.get("usage", {})
                        prompt_total = int(usage.get("prompt_tokens", 0))
                        completion_total = int(usage.get("completion_tokens", 0))
                        prompt_content = 0
                        first_user_seen = False
                        for node in ancestors(nodes, int(call["node"])):
                            message = node.get("message", {})
                            role = message.get("role")
                            content = str(message.get("content", ""))
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

                        completion_node = nodes[int(call["node"])]
                        completion = str(completion_node.get("message", {}).get("content", ""))
                        thinking_text = "".join(THINK_RE.findall(completion))
                        thinking_tokens = min(token_count(thinking_text), completion_total)
                        totals["thinking_tokens"] += thinking_tokens
                        totals["visible_output_tokens"] += completion_total - thinking_tokens
                        totals["format_tokens"] += max(0, prompt_total - prompt_content)
                        totals["prompt_tokens"] += prompt_total
                        totals["completion_tokens"] += completion_total

                    token_metrics = {
                        key: totals[key]
                        for key in (
                            "system_tokens",
                            "initial_prompt_tokens",
                            "observation_tokens",
                            "prior_action_tokens",
                            "format_tokens",
                            "thinking_tokens",
                            "visible_output_tokens",
                            "prompt_tokens",
                            "completion_tokens",
                        )
                    }
                    records.append(
                        {
                            "independent_run": independent_run,
                            "variant": variant,
                            "step": step,
                            "reward": float(trace.get("rewards", {}).get("progress_reward", 0.0)),
                            "turns": len(trace.get("calls", [])),
                            **token_metrics,
                            "total_tokens": totals["prompt_tokens"] + totals["completion_tokens"],
                        }
                    )

    metric_names = [
        "system_tokens",
        "initial_prompt_tokens",
        "observation_tokens",
        "prior_action_tokens",
        "format_tokens",
        "visible_output_tokens",
        "thinking_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "turns",
        "reward",
    ]
    run_means: dict[tuple[int, str, str], dict[str, float]] = {}
    for independent_run in (1, 2, 3):
        for variant in VARIANTS:
            for window, low, high in WINDOWS:
                selected = [
                    row
                    for row in records
                    if row["independent_run"] == independent_run
                    and row["variant"] == variant
                    and low <= row["step"] <= high
                ]
                if not selected:
                    raise ValueError((independent_run, variant, window))
                run_means[(independent_run, variant, window)] = {
                    metric: statistics.mean(float(row[metric]) for row in selected)
                    for metric in metric_names
                }

    output_dir = ROOT / "data/eval_token_metrics"
    figure_dir = ROOT / "figures/token_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    run_csv = output_dir / "heldout_token_metrics_per_run_by_quarter.csv"
    with run_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["independent_run", "variant", "objective", "quarter", *metric_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (run, variant, window), metrics in sorted(run_means.items()):
            writer.writerow(
                {
                    "independent_run": run,
                    "variant": variant,
                    "objective": VARIANTS[variant],
                    "quarter": window,
                    **{key: f"{value:.6f}" for key, value in metrics.items()},
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for variant, objective in VARIANTS.items():
        for window, _, _ in WINDOWS:
            row: dict[str, Any] = {
                "variant": variant,
                "objective": objective,
                "quarter": window,
            }
            for metric in metric_names:
                values = [run_means[(run, variant, window)][metric] for run in (1, 2, 3)]
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_sd"] = statistics.stdev(values)
            summary_rows.append(row)

    summary_csv = output_dir / "heldout_token_metrics_three_run_by_quarter.csv"
    fields = ["variant", "objective", "quarter"]
    fields.extend(f"{metric}_{suffix}" for metric in metric_names for suffix in ("mean", "sd"))
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

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
    table_data = []
    for row in summary_rows:
        rendered = [row["objective"], row["quarter"]]
        for _, metric in columns[2:]:
            mean = row[f"{metric}_mean"]
            sd = row[f"{metric}_sd"]
            precision = 1 if metric == "turns" else 0
            rendered.append(f"{mean:,.{precision}f} +/- {sd:,.{precision}f}")
        table_data.append(rendered)

    figure, axis = plt.subplots(figsize=(15.4, 6.25))
    axis.axis("off")
    table = axis.table(
        cellText=table_data,
        colLabels=[column[0] for column in columns],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.075, 0.09, 0.105, 0.145, 0.18, 0.12, 0.165],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1, 1.75)
    header = "#27313A"
    fills = {
        "RL": "#F0F2F4",
        "ECHO 0.05": "#E7F2F8",
        "ECHO 0.5": "#FAEFE1",
        "ECHO 1.0": "#E9F2EC",
    }
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD3DA")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor(header)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.09)
        else:
            objective = table_data[row_index - 1][0]
            cell.set_facecolor(fills[objective])
            if col_index == 0:
                cell.set_text_props(weight="bold")

    figure.text(
        0.5,
        0.975,
        "Held-Out Evaluation Token Usage",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#20252B",
    )
    figure.text(
        0.5,
        0.925,
        "Mean tokens processed per trajectory +/- SD across three independent training runs",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#66717D",
    )
    figure.subplots_adjust(left=0.015, right=0.985, top=0.89, bottom=0.015)
    for suffix in ("png", "svg"):
        figure.savefig(
            figure_dir / f"heldout_token_metrics_three_run_by_quarter.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    print(f"wrote {run_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {figure_dir / 'heldout_token_metrics_three_run_by_quarter.png'}")


if __name__ == "__main__":
    main()
