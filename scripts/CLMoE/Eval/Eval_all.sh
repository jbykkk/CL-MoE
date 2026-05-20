#!/bin/bash

# Usage:
#   bash Eval_all.sh           # eval CL-MoE (default)
#   bash Eval_all.sh base      # eval base LLaVA-1.5 (no LoRA)
#   bash Eval_all.sh base shuffle
#   bash Eval_all.sh base blank
#   bash Eval_all.sh finetune shuffle

export CUDA_VISIBLE_DEVICES=0,1

IMAGE_EVAL_MODE=${2:-normal}

if [ "$1" = "base" ]; then
    STAGE='BaseModel'
    BASE_ADAPTER_DIR='/home/data1/lyk/Experiments/CL-MoE/checkpoint/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5'
    LORA_CONFIG_DIR='/home/data1/lyk/Experiments/CL-MoE/checkpoints/CL4VQA/recognition/llava-1.5-7b-lora'

    if [ ! -f "$BASE_ADAPTER_DIR/config.json" ]; then
        cp "$LORA_CONFIG_DIR/config.json" "$BASE_ADAPTER_DIR/config.json"
        echo "Copied config.json to $BASE_ADAPTER_DIR"
    fi
    MODELPATH=$BASE_ADAPTER_DIR
else
    STAGE='Finetune'
    MODELPATH='/home/data1/lyk/Experiments/CL-MoE/checkpoints/CL4VQA/causal/llava-1.5-7b-lora'
fi

if [ "$IMAGE_EVAL_MODE" != "normal" ]; then
    STAGE="${STAGE}_${IMAGE_EVAL_MODE}"
fi

bash ./scripts/CLMoE/Eval/1_eval_recognition.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
bash ./scripts/CLMoE/Eval/2_eval_location.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
bash ./scripts/CLMoE/Eval/3_eval_judge.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
bash ./scripts/CLMoE/Eval/4_eval_commonsense.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
bash ./scripts/CLMoE/Eval/5_eval_count.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
bash ./scripts/CLMoE/Eval/6_eval_action.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
bash ./scripts/CLMoE/Eval/7_eval_color.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
bash ./scripts/CLMoE/Eval/8_eval_type.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
bash ./scripts/CLMoE/Eval/9_eval_subcategory.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
bash ./scripts/CLMoE/Eval/10_eval_causal.sh $STAGE $MODELPATH $IMAGE_EVAL_MODE
