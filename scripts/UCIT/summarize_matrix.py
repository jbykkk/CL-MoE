#!/usr/bin/env python3
"""Summarize the lower-triangular UCIT continual-learning evaluation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = ["imagenet_r", "arxivqa", "vizwiz", "iconqa", "clevr_math", "flickr30k"]
DISPLAY = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR-Math", "Flickr30k"]
STAGE_LABELS = [f"After_{task}" for task in TASKS[:-1]] + ["Final"]


def read_trailing_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = [index for index, char in enumerate(text) if char == "{"]
    for index in reversed(starts):
        try:
            value = json.loads(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "task" in value:
            return value
    raise ValueError(f"No final JSON metric object found in {path}")


def task_score(metric: dict[str, Any]) -> float:
    if "accuracy" in metric:
        return float(metric["accuracy"])
    return float(metric["average"])


def rounded(value: float) -> float:
    return round(value, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    matrix: list[list[float | None]] = [[None] * len(TASKS) for _ in TASKS]
    for stage_index, label in enumerate(STAGE_LABELS):
        for task_index in range(stage_index + 1):
            metric_path = args.eval_root / TASKS[task_index] / label / "metrics.json"
            matrix[stage_index][task_index] = task_score(read_trailing_json(metric_path))

    task_avg = [
        sum(float(matrix[stage][task]) for stage in range(task, len(TASKS))) / (len(TASKS) - task)
        for task in range(len(TASKS))
    ]
    last = [float(value) for value in matrix[-1] if value is not None]
    stage_avg = [
        sum(float(value) for value in row if value is not None) / (stage + 1)
        for stage, row in enumerate(matrix)
    ]
    forgetting = [
        max(float(matrix[stage][task]) for stage in range(task, len(TASKS))) - last[task]
        for task in range(len(TASKS))
    ]
    summary = {
        "tasks": TASKS,
        "stage_labels": STAGE_LABELS,
        "matrix": [[rounded(value) if value is not None else None for value in row] for row in matrix],
        "task_avg": [rounded(value) for value in task_avg],
        "avg": rounded(sum(task_avg) / len(task_avg)),
        "last": [rounded(value) for value in last],
        "last_average": rounded(sum(last) / len(last)),
        "stage_average": [rounded(value) for value in stage_avg],
        "forgetting": [rounded(value) for value in forgetting],
        "average_forgetting_all_tasks": rounded(sum(forgetting) / len(forgetting)),
        "average_forgetting_excluding_last_task": rounded(sum(forgetting[:-1]) / (len(forgetting) - 1)),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    header = "| Stage | " + " | ".join(DISPLAY) + " | Seen-task mean |"
    separator = "|---|" + "---:|" * (len(TASKS) + 1)
    lines = ["# CL-MoE UCIT continual-learning matrix", "", header, separator]
    for stage, row in enumerate(matrix):
        cells = ["—" if value is None else f"{value:.2f}" for value in row]
        lines.append(f"| {TASKS[stage]} | " + " | ".join(cells) + f" | {stage_avg[stage]:.2f} |")
    lines.extend([
        "",
        "| Summary | " + " | ".join(DISPLAY) + " | Macro average |",
        separator,
        "| Avg | " + " | ".join(f"{value:.2f}" for value in task_avg) + f" | {summary['avg']:.2f} |",
        "| Last | " + " | ".join(f"{value:.2f}" for value in last) + f" | {summary['last_average']:.2f} |",
        "| Forgetting | " + " | ".join(f"{value:.2f}" for value in forgetting)
        + f" | {summary['average_forgetting_all_tasks']:.2f} |",
        "",
        "Avg is the macro-average of per-task means from the stage where each task is learned through the final stage.",
        "Last is the final-stage score on all six tasks.",
    ])
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
