#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [START_TASK]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLMOE_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$CLMOE_ROOT"

UCIT_RUN_NAME=${UCIT_RUN_NAME:-octopus_matched_seed42}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$CLMOE_ROOT/checkpoints/UCIT/$UCIT_RUN_NAME"}
CLMOE_RUNTIME_DIR=${CLMOE_RUNTIME_DIR:-"$CLMOE_ROOT/results/UCIT/$UCIT_RUN_NAME/router_stats"}
TEMP_FOLDER=${TEMP_FOLDER:-Only_Pretrain_1.5_MOE_2}
export UCIT_RUN_NAME OUTPUT_ROOT CLMOE_RUNTIME_DIR TEMP_FOLDER

mkdir -p "$CLMOE_RUNTIME_DIR"

TASKS=(imagenet_r arxivqa vizwiz iconqa clevr_math flickr30k)
START_TASK=${1:-${UCIT_START_TASK:-imagenet_r}}
START_INDEX=-1

for index in "${!TASKS[@]}"; do
    if [[ "${TASKS[$index]}" == "$START_TASK" ]]; then
        START_INDEX=$index
        break
    fi
done
if [[ "$START_INDEX" -lt 0 ]]; then
    echo "Unknown UCIT start task: $START_TASK" >&2
    echo "Expected one of: ${TASKS[*]}" >&2
    exit 2
fi

# A task-level restart keeps completed task models and router indices, but the
# selected task itself starts from step zero. Verify the preceding chain before
# skipping it so a partial model is never treated as complete.
for ((index = 0; index < START_INDEX; index++)); do
    task=${TASKS[$index]}
    model_dir="$OUTPUT_ROOT/$task/llava-1.5-7b-lora"
    for required in \
        "$model_dir/adapter_model.bin" \
        "$model_dir/non_lora_trainables.bin" \
        "$CLMOE_RUNTIME_DIR/index_$task.txt"; do
        if [[ ! -f "$required" ]]; then
            echo "Cannot skip incomplete task $task; missing: $required" >&2
            exit 1
        fi
    done
done

for ((index = START_INDEX; index < ${#TASKS[@]}; index++)); do
    task=${TASKS[$index]}
    if [[ -e "$CLMOE_RUNTIME_DIR/value_counts_$task.txt" || -e "$CLMOE_RUNTIME_DIR/index_$task.txt" ]]; then
        echo "Refusing to mix existing router statistics for task: $task" >&2
        echo "Move the old value_counts/index files aside before restarting from $START_TASK." >&2
        exit 1
    fi
done

if [[ ${UCIT_CHAIN_DRY_RUN:-0} == 1 ]]; then
    echo "UCIT task-level continuation preflight passed"
    echo "run=$UCIT_RUN_NAME"
    echo "start_task=$START_TASK"
    echo "output_root=$OUTPUT_ROOT"
    echo "router_stats=$CLMOE_RUNTIME_DIR"
    exit 0
fi

if [[ "$START_INDEX" -eq 0 ]]; then
    previous=""
else
    previous=${TASKS[$((START_INDEX - 1))]}
fi

for ((index = START_INDEX; index < ${#TASKS[@]}; index++)); do
    task=${TASKS[$index]}
    bash "$SCRIPT_DIR/train_task.sh" "$task" ${previous:+"$previous"}
    python3 statistic.py \
        --task "$task" \
        --runtime-dir "$CLMOE_RUNTIME_DIR"
    if [[ -n "$previous" ]]; then
        python3 params.py \
            --task1 "$previous" \
            --task2 "$task" \
            --base-path "$OUTPUT_ROOT" \
            --temp-folder "$TEMP_FOLDER" \
            --runtime-dir "$CLMOE_RUNTIME_DIR"
    fi
    previous=$task
done
