#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 MODEL_PATH [STAGE]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_PATH=$1
STAGE=${2:-Finetune}
TASKS=(imagenet_r arxivqa vizwiz iconqa clevr_math flickr30k)

for task in "${TASKS[@]}"; do
    bash "$SCRIPT_DIR/evaluate_task.sh" "$task" "$MODEL_PATH" "$STAGE"
done
