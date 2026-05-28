#!/usr/bin/env bash
set -e
set -o xtrace

export TOKENIZERS_PARALLELISM=true
export WANDB_DISABLED=true
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/cephfs/songyue/hf_cache"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2,3,4,5

MODEL="/cephfs/songyue/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/"
model_name="${MODEL%/}"
model_name="${model_name##*/}"
TRAIN_FILE_DIR="mydata"
OUTPUT_DIR="./outputs-sft-full-$model_name-v1"

mkdir -p $OUTPUT_DIR

accelerate launch --num_processes=4 training/supervised_finetuning_accelerate.py \
    --model_name_or_path $MODEL \
    --train_file_dir $TRAIN_FILE_DIR \
    --validation_file_dir $TRAIN_FILE_DIR \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft False \
    --max_train_samples -1 \
    --max_eval_samples 400 \
    --model_max_length 512 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --warmup_steps 5 \
    --weight_decay 0.00 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 4 \
    --output_dir $OUTPUT_DIR \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --cache_dir ./cache --flash_attn False \
    2>&1 | tee $OUTPUT_DIR/train.log
