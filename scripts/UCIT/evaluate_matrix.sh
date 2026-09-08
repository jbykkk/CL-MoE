#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLMOE_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$CLMOE_ROOT"

UCIT_RUN_NAME=${UCIT_RUN_NAME:-octopus_matched_seed42}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$CLMOE_ROOT/checkpoints/UCIT/$UCIT_RUN_NAME"}
RESULT_ROOT=${RESULT_ROOT:-"$CLMOE_ROOT/results/UCIT/$UCIT_RUN_NAME/eval"}
EXPECTED_TEST_SAMPLES=${EXPECTED_TEST_SAMPLES:-3000}

TASKS=(imagenet_r arxivqa vizwiz iconqa clevr_math flickr30k)

cell_complete() {
    local task=$1
    local label=$2
    local output_dir="$RESULT_ROOT/$task/$label"
    local merged="$output_dir/merge.jsonl"
    local metrics="$output_dir/metrics.json"
    [[ -f "$merged" && -f "$metrics" ]] || return 1
    [[ $(wc -l < "$merged") -eq "$EXPECTED_TEST_SAMPLES" ]] || return 1
    grep -q '"task"' "$metrics"
}

for stage_index in "${!TASKS[@]}"; do
    stage_task=${TASKS[$stage_index]}
    model_path="$OUTPUT_ROOT/$stage_task/llava-1.5-7b-lora"
    if [[ ! -f "$model_path/adapter_model.bin" || ! -f "$model_path/non_lora_trainables.bin" ]]; then
        echo "Incomplete stage checkpoint: $model_path" >&2
        exit 1
    fi

    if [[ "$stage_index" -eq $((${#TASKS[@]} - 1)) ]]; then
        label=Final
    else
        label="After_${stage_task}"
    fi

    for ((task_index = 0; task_index <= stage_index; task_index++)); do
        task=${TASKS[$task_index]}
        if cell_complete "$task" "$label"; then
            echo "Skipping complete matrix cell: stage=$stage_task task=$task"
            continue
        fi
        echo "Evaluating matrix cell: stage=$stage_task task=$task"
        bash "$SCRIPT_DIR/evaluate_task.sh" "$task" "$model_path" "$label"
    done
done

python3 "$SCRIPT_DIR/summarize_matrix.py" \
    --eval-root "$RESULT_ROOT" \
    --output-json "$CLMOE_ROOT/results/UCIT/$UCIT_RUN_NAME/matrix_summary.json" \
    --output-markdown "$CLMOE_ROOT/results/UCIT/$UCIT_RUN_NAME/matrix_summary.md"
