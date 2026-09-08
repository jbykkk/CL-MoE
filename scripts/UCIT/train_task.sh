#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 TASK [PREVIOUS_TASK]" >&2
    exit 2
fi

TASK=$1
PREVIOUS_TASK=${2:-}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLMOE_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$CLMOE_ROOT"
export PYTHONPATH="$CLMOE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

UCIT_TRAIN_SAMPLING=${UCIT_TRAIN_SAMPLING:-octopus}
case "$UCIT_TRAIN_SAMPLING:$TASK" in
    octopus:imagenet_r|octopus:arxivqa|octopus:vizwiz|octopus:iconqa) EXPECTED_TRAIN_SAMPLES=9600 ;;
    octopus:clevr_math) EXPECTED_TRAIN_SAMPLES=16000 ;;
    octopus:flickr30k) EXPECTED_TRAIN_SAMPLES=3200 ;;
    full:imagenet_r) EXPECTED_TRAIN_SAMPLES=23998 ;;
    full:arxivqa|full:vizwiz|full:clevr_math|full:flickr30k) EXPECTED_TRAIN_SAMPLES=40000 ;;
    full:iconqa) EXPECTED_TRAIN_SAMPLES=29859 ;;
    octopus:*|full:*) echo "Unsupported UCIT task: $TASK" >&2; exit 2 ;;
    *) echo "Unsupported UCIT_TRAIN_SAMPLING: $UCIT_TRAIN_SAMPLING (expected octopus or full)" >&2; exit 2 ;;
esac

OCTOPUS_ROOT=${OCTOPUS_ROOT:-"$CLMOE_ROOT/../Octopus"}
if [[ "$UCIT_TRAIN_SAMPLING" == "full" ]]; then
    DEFAULT_UCIT_DATA_ROOT="$CLMOE_ROOT/data/UCIT_full"
    DEFAULT_UCIT_RUN_NAME=full_train
else
    DEFAULT_UCIT_DATA_ROOT="$CLMOE_ROOT/data/UCIT"
    DEFAULT_UCIT_RUN_NAME=octopus_matched_seed42
fi
UCIT_DATA_ROOT=${UCIT_DATA_ROOT:-$DEFAULT_UCIT_DATA_ROOT}
MODEL_ROOT=${MODEL_ROOT:-"$CLMOE_ROOT/checkpoint"}
UCIT_RUN_NAME=${UCIT_RUN_NAME:-$DEFAULT_UCIT_RUN_NAME}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$CLMOE_ROOT/checkpoints/UCIT/$UCIT_RUN_NAME"}
CLMOE_GPUS=${CLMOE_GPUS:-0,1}
MASTER_PORT=${MASTER_PORT:-29600}
CLMOE_RUNTIME_DIR=${CLMOE_RUNTIME_DIR:-"$CLMOE_ROOT/results/UCIT/$UCIT_RUN_NAME/router_stats"}
TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$CLMOE_ROOT/.cache/torch_extensions_clmoe"}
MAX_JOBS=${MAX_JOBS:-8}
if [[ -z ${CC:-} && -x /usr/bin/gcc ]]; then
    CC=/usr/bin/gcc
fi
if [[ -z ${CXX:-} && -x /usr/bin/g++ ]]; then
    CXX=/usr/bin/g++
fi
export UCIT_RUN_NAME CLMOE_RUNTIME_DIR TORCH_EXTENSIONS_DIR MAX_JOBS CC CXX

DATA_PATH="$UCIT_DATA_ROOT/train/$TASK.json"
BASE_MODEL="$MODEL_ROOT/vicuna-7b-v1.5"
VISION_TOWER="$MODEL_ROOT/clip-vit-large-patch14-336"
PROJECTOR="$MODEL_ROOT/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"

for required in "$DATA_PATH" "$BASE_MODEL" "$VISION_TOWER"; do
    if [[ ! -e "$required" ]]; then
        echo "Required path does not exist: $required" >&2
        exit 1
    fi
done

ACTUAL_TRAIN_SAMPLES=$(python3 -c \
    'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
    "$DATA_PATH")
if [[ "$ACTUAL_TRAIN_SAMPLES" -ne "$EXPECTED_TRAIN_SAMPLES" ]]; then
    echo "UCIT $UCIT_TRAIN_SAMPLING train-data mismatch for $TASK: expected $EXPECTED_TRAIN_SAMPLES, found $ACTUAL_TRAIN_SAMPLES" >&2
    if [[ "$UCIT_TRAIN_SAMPLING" == "full" ]]; then
        echo "Regenerate full data with: python scripts/UCIT/prepare_ucit.py --split train --train-sampling full --output-root data/UCIT_full" >&2
    else
        echo "Regenerate matched subsets with: python scripts/UCIT/prepare_ucit.py --split train" >&2
    fi
    exit 1
fi

MODEL_ARGS=()
if [[ -z "$PREVIOUS_TASK" ]]; then
    if [[ "$TASK" != "imagenet_r" ]]; then
        echo "Only imagenet_r can run without PREVIOUS_TASK in the UCIT order." >&2
        exit 2
    fi
    if [[ ! -f "$PROJECTOR" ]]; then
        echo "Required projector does not exist: $PROJECTOR" >&2
        exit 1
    fi
    MODEL_ARGS+=(--pretrain_mm_mlp_adapter "$PROJECTOR")
    OUTPUT_DIR="$OUTPUT_ROOT/$TASK/llava-1.5-7b-lora"
else
    PREVIOUS_MODEL="$OUTPUT_ROOT/$PREVIOUS_TASK/llava-1.5-7b-lora"
    if [[ ! -d "$PREVIOUS_MODEL" ]]; then
        echo "Previous task model does not exist: $PREVIOUS_MODEL" >&2
        exit 1
    fi
    MODEL_ARGS+=(--previous_task_model_path "$PREVIOUS_MODEL")
    OUTPUT_DIR="$OUTPUT_ROOT/Only_Pretrain_1.5_MOE_2/$TASK/llava-1.5-7b-lora"
fi

if [[ ${CLMOE_DRY_RUN:-0} == 1 ]]; then
    printf 'UCIT task preflight passed\n'
    printf 'task=%s\ntrain_sampling=%s\ndata=%s\ntrain_samples=%s\nimages=%s\nbase_model=%s\nvision_tower=%s\noutput=%s\nrouter_stats=%s\ntorch_extensions=%s\ncc=%s\ncxx=%s\n' \
        "$TASK" "$UCIT_TRAIN_SAMPLING" "$DATA_PATH" "$ACTUAL_TRAIN_SAMPLES" "$OCTOPUS_ROOT" "$BASE_MODEL" "$VISION_TOWER" "$OUTPUT_DIR" "$CLMOE_RUNTIME_DIR" \
        "$TORCH_EXTENSIONS_DIR" "${CC:-default}" "${CXX:-default}"
    exit 0
fi

deepspeed --include "localhost:$CLMOE_GPUS" --master_port "$MASTER_PORT" \
    llava/train/train_mem_MOE.py \
    --deepspeed ./scripts/zero3_offload.json \
    --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --expert_num 4 \
    --model_name_or_path "$BASE_MODEL" \
    "${MODEL_ARGS[@]}" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$OCTOPUS_ROOT" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --task "$TASK"
