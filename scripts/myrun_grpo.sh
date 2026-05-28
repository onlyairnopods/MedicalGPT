#!/usr/bin/env bash
set -e
set -o xtrace

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HOME="/cephfs/songyue/zzlai/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-reason-lora-Qwen3-4B-Instruct-2507-v3-v1/merged_model"
model_name="Qwen3-4B-Instruct-2507-v3"

TRAIN_FILE_DIR="myreasondata/mygrpodata/"

OUTPUT_DIR="./outputs-grpo-lora-$model_name-v1"

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node 4 --nnodes 1 training/mygrpo_training.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL \
    --train_file_dir $TRAIN_FILE_DIR \
    --train_samples 2500 \
    --max_steps -1 --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 3 \
    --output_dir $OUTPUT_DIR \
    --dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --beta 0.001 \
    --learning_rate 5.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm False \
    --attn_implementation "flash_attention_2" \
    --logging_steps 10 \
    \
    `# QLoRA配置` \
    --use_peft True \
    --qlora False \
    --load_in_4bit False \
    --lora_target_modules q_proj v_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    \
    `# 显存优化配置` \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 1024 \
    2>&1 | tee $OUTPUT_DIR/train.log

