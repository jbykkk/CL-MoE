#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 TASK MODEL_PATH [STAGE]" >&2
    exit 2
fi

TASK=$1
MODEL_PATH=$2
STAGE=${3:-Finetune}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLMOE_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$CLMOE_ROOT"
export PYTHONPATH="$CLMOE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "$TASK" in
    imagenet_r) SOURCE_NAME=ImageNet-R ;;
    arxivqa) SOURCE_NAME=ArxivQA ;;
    vizwiz) SOURCE_NAME=VizWiz ;;
    iconqa) SOURCE_NAME=IconQA ;;
    clevr_math) SOURCE_NAME=CLEVR-Math ;;
    flickr30k) SOURCE_NAME=Flickr30k ;;
    *) echo "Unsupported UCIT task: $TASK" >&2; exit 2 ;;
esac

OCTOPUS_ROOT=${OCTOPUS_ROOT:-"$CLMOE_ROOT/../Octopus"}
UCIT_DATA_ROOT=${UCIT_DATA_ROOT:-"$CLMOE_ROOT/data/UCIT"}
UCIT_INSTRUCTIONS_ROOT=${UCIT_INSTRUCTIONS_ROOT:-"$CLMOE_ROOT/data/UCIT_instructions"}
MODEL_ROOT=${MODEL_ROOT:-"$CLMOE_ROOT/checkpoint"}
CLMOE_GPUS=${CLMOE_GPUS:-0,1,2,3}
UCIT_RUN_NAME=${UCIT_RUN_NAME:-octopus_matched_seed42}
RESULT_ROOT=${RESULT_ROOT:-"$CLMOE_ROOT/results/UCIT/$UCIT_RUN_NAME/eval"}
UCIT_SCORER_PYTHON=${UCIT_SCORER_PYTHON:-python3}
UCIT_MAX_NEW_TOKENS=${UCIT_MAX_NEW_TOKENS:-16}

QUESTION_FILE="$UCIT_DATA_ROOT/test/$TASK.json"
OUTPUT_DIR="$RESULT_ROOT/$TASK/$STAGE"

if [[ ! -f "$QUESTION_FILE" ]]; then
    echo "Question file does not exist: $QUESTION_FILE" >&2
    exit 1
fi
if [[ ! -e "$MODEL_PATH" ]]; then
    echo "Model path does not exist: $MODEL_PATH" >&2
    exit 1
fi
if [[ ${UCIT_DRY_RUN:-0} == 1 ]]; then
    printf 'UCIT evaluation preflight passed\n'
    printf 'task=%s\nquestions=%s\nimages=%s\nmodel=%s\nresults=%s\n' \
        "$TASK" "$QUESTION_FILE" "$OCTOPUS_ROOT" "$MODEL_PATH" "$OUTPUT_DIR"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
IFS=',' read -r -a GPU_ARRAY <<< "$CLMOE_GPUS"
CHUNKS=${#GPU_ARRAY[@]}

for index in "${!GPU_ARRAY[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[$index]} python3 -m llava.eval.CLMoE.model_vqa_cc_instruction \
        --model-path "$MODEL_PATH" \
        --model-base "$MODEL_ROOT/vicuna-7b-v1.5" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$OCTOPUS_ROOT" \
        --answers-file "$OUTPUT_DIR/${CHUNKS}_${index}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$index" \
        --temperature 0 \
        --max-new-tokens "$UCIT_MAX_NEW_TOKENS" \
        --conv-mode vicuna_v1 &
done
wait

MERGED="$OUTPUT_DIR/merge.jsonl"
: > "$MERGED"
for index in "${!GPU_ARRAY[@]}"; do
    cat "$OUTPUT_DIR/${CHUNKS}_${index}.jsonl" >> "$MERGED"
done

SCORE_ARGS=(
    --task "$TASK"
    --annotations "$QUESTION_FILE"
    --predictions "$MERGED"
)
if [[ "$TASK" == "vizwiz" || "$TASK" == "flickr30k" ]]; then
    SCORE_ARGS+=(--coco-annotations "$UCIT_INSTRUCTIONS_ROOT/test/${SOURCE_NAME}_coco_type_3000.json")
fi
"$UCIT_SCORER_PYTHON" "$SCRIPT_DIR/score_ucit.py" "${SCORE_ARGS[@]}" | tee "$OUTPUT_DIR/metrics.json"
