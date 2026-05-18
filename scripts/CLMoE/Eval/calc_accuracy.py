import json
import os

tasks = ['recognition', 'location', 'judge', 'commonsense', 'count', 'action', 'color', 'type', 'subcategory', 'causal']

for task in tasks:
    annotation_file = f'CL4VQA/test/test_q_{task}.json'
    result_file = f'results/CLMoE/{task}/Finetune/merge.jsonl'
    output_file = f'results/CLMoE/{task}/Finetune/output_result.jsonl'

    if not os.path.exists(annotation_file) or not os.path.exists(result_file):
        print(f'{task}: missing files, skipping')
        continue

    annotations = {a['question_id']: a for a in json.load(open(annotation_file))}
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = sum(1 for r in results if r['text'].upper() == annotations[r['question_id']]['answer'].upper())
    accuracy = 100. * right / total

    print(f'{task}: Samples={total}, Accuracy={accuracy:.2f}%')

    with open(output_file, 'w') as f:
        f.write(f'Samples: {total}\nAccuracy: {accuracy:.2f}%\n')
