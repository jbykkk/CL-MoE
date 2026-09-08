import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate CL-MoE router counts and select the most-used experts."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--runtime-dir", default=".")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def calculate_sums(file_path):
    index_sums = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                index_text, count_text = line.split(":", maxsplit=1)
                index = int(index_text)
                count = int(count_text)
            except ValueError as error:
                raise SystemExit(
                    f"Invalid router statistic at {file_path}:{line_number}: {line!r}"
                ) from error
            index_sums[index] = index_sums.get(index, 0) + count
    return index_sums


def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.task):
        raise SystemExit(f"Invalid task name: {args.task!r}")
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")

    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    input_path = runtime_dir / f"value_counts_{args.task}.txt"
    output_path = runtime_dir / f"index_{args.task}.txt"

    if not input_path.is_file():
        raise SystemExit(f"Router statistics file does not exist: {input_path}")

    index_sums = calculate_sums(input_path)
    selected = [
        index
        for index, _ in sorted(index_sums.items(), key=lambda item: item[1], reverse=True)
        if index != -1
    ][: args.top_k]
    if not selected:
        raise SystemExit(f"No valid expert indices found in: {input_path}")

    output_path.write_text(
        "".join(f"{index}\n" for index in selected),
        encoding="utf-8",
    )
    sorted_counts = dict(
        sorted(index_sums.items(), key=lambda item: item[1], reverse=True)
    )
    print(f"Router counts: {sorted_counts}")
    print(f"Selected experts: {selected}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
