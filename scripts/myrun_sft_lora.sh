#!/usr/bin/env bash
set -e
set -o xtrace

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HOME="/cephfs/songyue/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# MODEL="/cephfs/songyue/hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
# model_name="Llama-3.2-3B-Instruct"
# MODEL="/cephfs/songyue/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
# model_name="Qwen3-4B-Instruct-2507"

# TRAIN_FILE_DIR="mydata"

# MODEL="/cephfs/songyue/hf_cache/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
# model_name="Llama-3.2-3B"

# OUTPUT_DIR="./outputs-sft-lora-$model_name-v3"

# mkdir -p $OUTPUT_DIR

#torchrun --nproc_per_node 4 --nnodes 1 training/supervised_finetuning.py \
#    --model_name_or_path $MODEL \
#    --train_file_dir $TRAIN_FILE_DIR \
#    --validation_file_dir $TRAIN_FILE_DIR \
#    --per_device_train_batch_size 32 \
#    --per_device_eval_batch_size 2 \
#    --do_train \
#    --do_eval \
#    --use_peft True \
#    --max_train_samples -1 \
#    --max_eval_samples 400 \
#    --model_max_length 512 \
#    --num_train_epochs 1 \
#    --learning_rate 2e-5 \
#    --warmup_ratio 0.03 \
#    --warmup_steps 5 \
#    --weight_decay 0.00 \
#    --logging_strategy steps \
#    --logging_steps 200 \
#    --eval_steps 400 \
#    --eval_strategy steps \
#    --save_steps 400 \
#    --save_strategy steps \
#    --save_total_limit 3 \
#    --gradient_accumulation_steps 1 \
#    --preprocessing_num_workers 8 \
#    --output_dir $OUTPUT_DIR \
#    --ddp_timeout 30000 \
#    --logging_first_step True \
#    --target_modules all \
#    --lora_rank 32 \
#    --lora_alpha 64 \
#    --lora_dropout 0.05 \
#    --torch_dtype bfloat16 \
#    --bf16 \
#    --report_to tensorboard \
#    --ddp_find_unused_parameters False \
#    --cache_dir ./cache --flash_attn True \
#    2>&1 | tee $OUTPUT_DIR/train.log

# --template_name llama3
# qwen3_nothink

# mv ./selected_train_zh_0_share_v1_filtered.jsonl mydata/
# TRAIN_FILE_DIR="mydata"


# MODEL="/cephfs/songyue/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
# model_name="Qwen3-4B-Instruct-2507"

# OUTPUT_DIR="./outputs-sft-lora-$model_name-v3"

# mkdir -p $OUTPUT_DIR

# torchrun --nproc_per_node 4 --nnodes 1 training/supervised_finetuning.py \
#     --model_name_or_path $MODEL \
#     --train_file_dir $TRAIN_FILE_DIR \
#     --validation_file_dir $TRAIN_FILE_DIR \
#     --per_device_train_batch_size 26 \
#     --per_device_eval_batch_size 4 \
#     --do_train \
#     --do_eval \
#     --use_peft True \
#     --max_train_samples -1 \
#     --max_eval_samples 400 \
#     --model_max_length 512 \
#     --num_train_epochs 1 \
#     --learning_rate 2e-5 \
#     --warmup_ratio 0.03 \
#     --warmup_steps 5 \
#     --weight_decay 0.00 \
#     --logging_strategy steps \
#     --logging_steps 200 \
#     --eval_steps 400 \
#     --eval_strategy steps \
#     --save_steps 400 \
#     --save_strategy steps \
#     --save_total_limit 3 \
#     --gradient_accumulation_steps 1 \
#     --preprocessing_num_workers 8 \
#     --output_dir $OUTPUT_DIR \
#     --ddp_timeout 30000 \
#     --logging_first_step True \
#     --target_modules all \
#     --lora_rank 32 \
#     --lora_alpha 64 \
#     --lora_dropout 0.05 \
#     --torch_dtype bfloat16 \
#     --bf16 \
#     --report_to tensorboard \
#     --ddp_find_unused_parameters False \
#     --cache_dir ./cache --flash_attn True \
#     2>&1 | tee $OUTPUT_DIR/train.log



#################################################### base model
TRAIN_FILE_DIR="mydata_v1"

MODEL="/cephfs/songyue/hf_cache/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
model_name="Llama-3.2-3B"

OUTPUT_DIR="./outputs-sft-lora-$model_name-v1"

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node 4 --nnodes 1 training/supervised_finetuning.py \
    --model_name_or_path $MODEL \
    --train_file_dir $TRAIN_FILE_DIR \
    --validation_file_dir $TRAIN_FILE_DIR \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 400 \
    --model_max_length 512 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --warmup_steps 5 \
    --weight_decay 0.00 \
    --logging_strategy steps \
    --logging_steps 200 \
    --eval_steps 400 \
    --eval_strategy steps \
    --save_steps 400 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 8 \
    --output_dir $OUTPUT_DIR \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --cache_dir ./cache --flash_attn True \
    --template_name llama3 \
    2>&1 | tee $OUTPUT_DIR/train.log



MODEL="/cephfs/songyue/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
model_name="Qwen3-4B"

OUTPUT_DIR="./outputs-sft-lora-$model_name-v1"

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node 4 --nnodes 1 training/supervised_finetuning.py \
    --model_name_or_path $MODEL \
    --train_file_dir $TRAIN_FILE_DIR \
    --validation_file_dir $TRAIN_FILE_DIR \
    --per_device_train_batch_size 26 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 400 \
    --model_max_length 512 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --warmup_steps 5 \
    --weight_decay 0.00 \
    --logging_strategy steps \
    --logging_steps 200 \
    --eval_steps 400 \
    --eval_strategy steps \
    --save_steps 400 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 8 \
    --output_dir $OUTPUT_DIR \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --cache_dir ./cache --flash_attn True \
    --template_name qwen3_nothink \
    2>&1 | tee $OUTPUT_DIR/train.log


##################################################
TRAIN_FILE_DIR="mydata"

MODEL="/cephfs/songyue/hf_cache/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
model_name="Llama-3.2-3B"

OUTPUT_DIR="./outputs-sft-lora-$model_name-v2"

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node 4 --nnodes 1 training/supervised_finetuning.py \
    --model_name_or_path $MODEL \
    --train_file_dir $TRAIN_FILE_DIR \
    --validation_file_dir $TRAIN_FILE_DIR \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 400 \
    --model_max_length 512 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --warmup_steps 5 \
    --weight_decay 0.00 \
    --logging_strategy steps \
    --logging_steps 200 \
    --eval_steps 400 \
    --eval_strategy steps \
    --save_steps 400 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 8 \
    --output_dir $OUTPUT_DIR \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --cache_dir ./cache --flash_attn True \
    --template_name llama3 \
    2>&1 | tee $OUTPUT_DIR/train.log



MODEL="/cephfs/songyue/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
model_name="Qwen3-4B"

OUTPUT_DIR="./outputs-sft-lora-$model_name-v2"

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node 4 --nnodes 1 training/supervised_finetuning.py \
    --model_name_or_path $MODEL \
    --train_file_dir $TRAIN_FILE_DIR \
    --validation_file_dir $TRAIN_FILE_DIR \
    --per_device_train_batch_size 26 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 400 \
    --model_max_length 512 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --warmup_steps 5 \
    --weight_decay 0.00 \
    --logging_strategy steps \
    --logging_steps 200 \
    --eval_steps 400 \
    --eval_strategy steps \
    --save_steps 400 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 8 \
    --output_dir $OUTPUT_DIR \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --cache_dir ./cache --flash_attn True \
    --template_name qwen3_nothink \
    2>&1 | tee $OUTPUT_DIR/train.log
