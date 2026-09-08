#!/usr/bin/env python3
"""Score CL-MoE JSONL predictions using the metrics used by Octopus UCIT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACCURACY_TASKS = {"imagenet_r", "arxivqa", "iconqa", "clevr_math"}
CAPTION_TASKS = {"vizwiz", "flickr30k"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cleaned_prediction(value: str, task: str) -> str:
    value = value.replace("</s>", "").strip()
    if task == "arxivqa":
        value = value.replace("(", "").replace(")", "")
    return value


def score_accuracy(task: str, annotations: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> None:
    try:
        from mathruler.grader import grade_answer
    except ImportError as exc:
        raise SystemExit("accuracy scoring requires: pip install mathruler") from exc

    gold = {str(row["question_id"]): row["answer"] for row in annotations}
    prediction_map = {str(row["question_id"]): row["text"] for row in predictions}
    missing = sorted(set(gold) - set(prediction_map))
    extra = sorted(set(prediction_map) - set(gold))
    if missing or extra:
        raise SystemExit(f"prediction IDs mismatch: missing={len(missing)}, extra={len(extra)}")
    correct = sum(
        bool(grade_answer(cleaned_prediction(prediction_map[qid], task), answer))
        for qid, answer in gold.items()
    )
    print(json.dumps({"task": task, "correct": correct, "total": len(gold), "accuracy": 100 * correct / len(gold)}, indent=2))


def score_caption(
    task: str,
    annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    predictions_path: Path,
    coco_path: Path,
) -> None:
    try:
        from pycocoevalcap.eval import COCOEvalCap
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise SystemExit("caption scoring requires pycocotools and pycocoevalcap") from exc

    prediction_map = {str(row["question_id"]): row["text"] for row in predictions}
    ordered = []
    for row in annotations:
        qid = str(row["question_id"])
        if qid not in prediction_map:
            raise SystemExit(f"missing prediction for question_id={qid}")
        ordered.append(cleaned_prediction(prediction_map[qid], task))

    coco = COCO(str(coco_path))
    image_ids = [row["id"] for row in coco.dataset["images"]]
    if len(image_ids) > len(ordered):
        raise SystemExit(f"COCO annotations contain {len(image_ids)} images but only {len(ordered)} predictions")
    results = [
        {"image_id": image_id, "caption": ordered[index]}
        for index, image_id in enumerate(image_ids)
    ]
    result_path = predictions_path.with_suffix(predictions_path.suffix + ".coco.json")
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False)
    coco_result = coco.loadRes(str(result_path))
    evaluator = COCOEvalCap(coco, coco_result)
    evaluator.evaluate()
    metrics = {key: 100 * value for key, value in evaluator.eval.items() if key in {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"}}
    metrics["average"] = sum(metrics.values()) / len(metrics)
    metrics["task"] = task
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(ACCURACY_TASKS | CAPTION_TASKS), required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path)
    args = parser.parse_args()

    annotations = read_json(args.annotations)
    predictions = read_jsonl(args.predictions)
    if args.task in ACCURACY_TASKS:
        score_accuracy(args.task, annotations, predictions)
    else:
        if args.coco_annotations is None:
            raise SystemExit("--coco-annotations is required for caption tasks")
        score_caption(args.task, annotations, predictions, args.predictions, args.coco_annotations)


if __name__ == "__main__":
    main()
