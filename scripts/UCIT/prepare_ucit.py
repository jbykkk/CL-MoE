#!/usr/bin/env python3
"""Convert Octopus UCIT instructions to the legacy LLaVA/CL-MoE format."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


TASKS = (
    ("imagenet_r", "ImageNet-R"),
    ("arxivqa", "ArxivQA"),
    ("vizwiz", "VizWiz"),
    ("iconqa", "IconQA"),
    ("clevr_math", "CLEVR-Math"),
    ("flickr30k", "Flickr30k"),
)
SPLITS = ("train", "test")
OCTOPUS_TRAIN_SAMPLES = {
    "imagenet_r": 9600,
    "arxivqa": 9600,
    "vizwiz": 9600,
    "iconqa": 9600,
    "clevr_math": 16000,
    "flickr30k": 3200,
}


class DatasetError(ValueError):
    pass


def first_string(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def message_pair(row: dict[str, Any]) -> tuple[str | None, str | None]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, None
    question = answer = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from")
        content = message.get("content") or message.get("value")
        if not isinstance(content, str):
            continue
        if role in ("user", "human") and question is None:
            question = content
        elif role in ("assistant", "gpt") and answer is None:
            answer = content
    return question, answer


def image_value(row: dict[str, Any]) -> str | None:
    image = row.get("image")
    if not isinstance(image, str):
        images = row.get("images")
        if isinstance(images, list) and len(images) == 1 and isinstance(images[0], str):
            image = images[0]
    return image.strip() if isinstance(image, str) and image.strip() else None


def normalized_image(image: str, octopus_root: Path) -> str:
    path = Path(os.path.expanduser(image))
    if path.is_absolute():
        try:
            path = path.relative_to(octopus_root)
        except ValueError as exc:
            raise DatasetError(
                f"absolute image path is outside Octopus root: {image}"
            ) from exc
    normalized = Path(os.path.normpath(str(path)))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise DatasetError(f"unsafe image path: {image}")
    return normalized.as_posix()


def convert_row(row: Any, index: int, octopus_root: Path) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise DatasetError(f"row {index} is not a JSON object")

    question = first_string(row, ("text", "question", "query", "prompt", "input"))
    answer = first_string(row, ("answer", "response", "output"))
    message_question, message_answer = message_pair(row)
    question = question or message_question
    answer = answer or message_answer
    image = image_value(row)

    if question is None:
        raise DatasetError(f"row {index} has no question/text field")
    if answer is None:
        raise DatasetError(f"row {index} has no answer/response field")
    if image is None:
        raise DatasetError(f"row {index} has no image field")

    question = question.replace("<image>", "").strip()
    sample_id = row.get("question_id", row.get("id", index))
    converted: dict[str, Any] = {
        "id": sample_id,
        "question_id": sample_id,
        "image": normalized_image(image, octopus_root),
        "text": question,
        "answer": answer,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }
    return converted


def load_rows(path: Path) -> list[Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise DatasetError(f"expected a JSON array in {path}")
    return rows


def prepare_file(
    source: Path,
    destination: Path,
    octopus_root: Path,
    check_images: bool,
    validate_only: bool,
    sample_limit: int | None = None,
    data_seed: int = 42,
) -> tuple[int, int]:
    rows = load_rows(source)
    if sample_limit is not None:
        if sample_limit > len(rows):
            raise DatasetError(
                f"requested {sample_limit} samples from a dataset with {len(rows)} rows"
            )
        # Match ms-swift's `path.json#N` sampling exactly. Octopus launches
        # every task separately with the default data seed (42), so each task
        # starts from a fresh RandomState rather than sharing RNG state.
        indices = np.random.RandomState(data_seed).permutation(len(rows))[:sample_limit]
        rows = [rows[int(index)] for index in indices]
    converted = [convert_row(row, index, octopus_root) for index, row in enumerate(rows)]
    missing_images = 0
    if check_images:
        missing_images = sum(
            not (octopus_root / row["image"]).is_file() for row in converted
        )
    if not validate_only:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(converted, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return len(converted), missing_images


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_octopus = project_root.parent / "Octopus"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--octopus-root",
        type=Path,
        default=Path(os.environ.get("OCTOPUS_ROOT", default_octopus)),
        help="Octopus checkout containing data/UCIT_instructions",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "UCIT",
        help="destination for CL-MoE-formatted instruction JSON",
    )
    parser.add_argument(
        "--instructions-root",
        type=Path,
        default=project_root / "data" / "UCIT_instructions",
        help="directory containing the Octopus train/ and test/ instruction files",
    )
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--task", choices=("all", *(name for name, _ in TASKS)), default="all")
    parser.add_argument(
        "--train-sampling",
        choices=("octopus", "full"),
        default="octopus",
        help="use Octopus's fixed UCIT subsets (default) or all training rows",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="sampling seed used by Octopus/ms-swift (default: 42)",
    )
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    octopus_root = args.octopus_root.expanduser().resolve()
    source_root = args.instructions_root.expanduser().resolve()
    splits = SPLITS if args.split == "all" else (args.split,)
    tasks = TASKS if args.task == "all" else tuple(
        task for task in TASKS if task[0] == args.task
    )

    if not source_root.is_dir():
        print(
            f"UCIT instructions are not ready: {source_root} does not exist",
            file=sys.stderr,
        )
        return 2

    failed = False
    for split in splits:
        for task_name, source_name in tasks:
            source = source_root / split / f"{source_name}.json"
            destination = args.output_root / split / f"{task_name}.json"
            if not source.is_file():
                print(f"MISSING {source}", file=sys.stderr)
                failed = True
                continue
            try:
                sample_limit = (
                    OCTOPUS_TRAIN_SAMPLES[task_name]
                    if split == "train" and args.train_sampling == "octopus"
                    else None
                )
                count, missing_images = prepare_file(
                    source,
                    destination,
                    octopus_root,
                    args.check_images,
                    args.validate_only,
                    sample_limit,
                    args.data_seed,
                )
            except DatasetError as exc:
                print(f"INVALID {source}: {exc}", file=sys.stderr)
                failed = True
                continue
            verb = "validated" if args.validate_only else "converted"
            details = []
            if sample_limit is not None:
                details.append(f"Octopus subset, seed={args.data_seed}")
            if args.check_images:
                details.append(f"missing images: {missing_images}")
            suffix = f" ({', '.join(details)})" if details else ""
            print(f"{verb} {split}/{source_name}: {count} samples{suffix}")
            failed = failed or missing_images > 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
