#!/bin/bash

# Evaluate base LLaVA-1.5 model (no LoRA) on all tasks
# Usage: bash Eval_all_base.sh

export CUDA_VISIBLE_DEVICES=0,1

STAGE='BaseModel'
MODELPATH='/home/data1/lyk/Experiments/CL-MoE/checkpoint/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5'

bash ./scripts/CLMoE/Eval/1_eval_recognition.sh $STAGE $MODELPATH
bash ./scripts/CLMoE/Eval/2_eval_location.sh $STAGE $MODELPATH
bash ./scripts/CLMoE/Eval/3_eval_judge.sh $STAGE $MODELPATH
bash ./scripts/CLMoE/Eval/4_eval_commonsense.sh $STAGE $MODELPATH
bash ./scripts/CLMoE/Eval/5_eval_count.sh $STAGE $MODELPATH
bash ./scripts/CLMoE/Eval/6_eval_action.sh $STAGE $MODELPATH
bash ./scripts/CLMoE/Eval/7_eval_color.sh $STAGE $MODELPATH
bash ./scripts/CLMoE/Eval/8_eval_type.sh $STAGE $MODELPATH
bash ./scripts/CLMoE/Eval/9_eval_subcategory.sh $STAGE $MODELPATH
bash ./scripts/CLMoE/Eval/10_eval_causal.sh $STAGE $MODELPATH
