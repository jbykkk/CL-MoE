import argparse
import json
from collections import Counter
from pathlib import Path


TASKS = [
    "recognition",
    "location",
    "judge",
    "commonsense",
    "count",
    "action",
    "color",
    "type",
    "subcategory",
    "causal",
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_answer(value):
    return str(value).strip().upper()


def summarize_stage(results_root, data_root, stage):
    accuracies = []
    print(f"\n[{stage}]")
    for task in TASKS:
        annotation_file = data_root / "test" / f"test_q_{task}.json"
        result_file = results_root / task / stage / "merge.jsonl"
        if not annotation_file.exists() or not result_file.exists():
            print(f"{task:12s} missing annotation/result")
            continue

        annotations = {item["question_id"]: item for item in load_json(annotation_file)}
        results = load_jsonl(result_file)
        correct = 0
        for item in results:
            gold = annotations[item["question_id"]]["answer"]
            correct += norm_answer(item["text"]) == norm_answer(gold)

        acc = 100.0 * correct / len(results) if results else 0.0
        accuracies.append(acc)
        model_id = results[0].get("model_id", "") if results else ""
        print(f"{task:12s} samples={len(results):5d} acc={acc:6.2f} model_id={model_id}")

    if accuracies:
        print(f"{'AP(mean)':12s} {'':13s} acc={sum(accuracies) / len(accuracies):6.2f}")


def summarize_data(data_root):
    print("\n[data checks]")
    for task in TASKS:
        train_file = data_root / "train" / f"train_q_{task}.json"
        test_file = data_root / "test" / f"test_q_{task}.json"
        if not train_file.exists() or not test_file.exists():
            print(f"{task:12s} missing train/test json")
            continue

        train_rows = load_json(train_file)
        test_rows = load_json(test_file)
        train_qids = {row.get("question_id") for row in train_rows}
        test_qids = {row.get("question_id") for row in test_rows}
        overlap_qids = train_qids & test_qids

        test_answers = [norm_answer(row.get("answer", "")) for row in test_rows]
        answer_counts = Counter(test_answers)
        majority_answer, majority_count = answer_counts.most_common(1)[0]
        majority_acc = 100.0 * majority_count / len(test_rows) if test_rows else 0.0

        print(
            f"{task:12s} train={len(train_rows):5d} test={len(test_rows):5d} "
            f"qid_overlap={len(overlap_qids):5d} "
            f"majority={majority_answer!r}:{majority_acc:5.2f}% "
            f"unique_answers={len(answer_counts):5d}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("CL4VQA"))
    parser.add_argument("--results-root", type=Path, default=Path("results/CLMoE"))
    parser.add_argument("--stage", action="append")
    args = parser.parse_args()

    summarize_data(args.data_root)
    for stage in args.stage or ["BaseModel", "Finetune"]:
        summarize_stage(args.results_root, args.data_root, stage)


if __name__ == "__main__":
    main()
