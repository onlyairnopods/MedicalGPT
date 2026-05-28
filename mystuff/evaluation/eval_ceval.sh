# use lm-evaluation-harness to evaluate the model on C-Eval dataset

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
# export HF_HOME="/cephfs/songyue/hf_cache"
export HF_HOME="/cephfs/songyue/zzlai/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3

conda activate /cephfs/songyue/zzlai/my_conda_env/llava


# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Llama-3.2-3B-Instruct-v3/merged_model"
# model_name="sft-lora-Llama-3.2-3B-Instruct-v3"
# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Llama-3.2-3B-Instruct-v2_2e/merged_model"
# model_name="sft-lora-Llama-3.2-3B-Instruct-v2_2e"

# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-Instruct-2507-v2/merged_model"
# model_name="sft-lora-Qwen3-4B-Instruct-2507-v2"
# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-Instruct-2507-v3/merged_model"
# model_name="sft-lora-Qwen3-4B-Instruct-2507-v3"



# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Llama-3.2-3B-v1/merged_model"
# model_name="sft-lora-Llama-3.2-3B-v1"

# SAVE_DIR="./results/$model_name"

# mkdir -p $SAVE_DIR

# lm-eval run --config lm-eval_config.yaml \
#     --model_args pretrained=$MODEL dtype=bfloat16 \
#     --output_path $SAVE_DIR \
#     2>&1 | tee -a $SAVE_DIR/eval_ceval.log


# DATASET_NAME="/cephfs/songyue/zzlai/hf_cache/hub/datasets--FreedomIntelligence--Huatuo26M-Lite/snapshots/90ce61699e90568db82cdce4c4035c6918d70aa6"

# python eval_ppl.py --model $MODEL --dataset_name $DATASET_NAME --output_dir $SAVE_DIR



# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Llama-3.2-3B-v2/merged_model"
# model_name="sft-lora-Llama-3.2-3B-v2"

# SAVE_DIR="./results/$model_name"

# mkdir -p $SAVE_DIR

# lm-eval run --config lm-eval_config.yaml \
#     --model_args pretrained=$MODEL dtype=bfloat16 \
#     --output_path $SAVE_DIR \
#     2>&1 | tee -a $SAVE_DIR/eval_ceval.log


# DATASET_NAME="/cephfs/songyue/zzlai/hf_cache/hub/datasets--FreedomIntelligence--Huatuo26M-Lite/snapshots/90ce61699e90568db82cdce4c4035c6918d70aa6"

# python eval_ppl.py --model $MODEL --dataset_name $DATASET_NAME --output_dir $SAVE_DIR



# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-v1/merged_model"
# model_name="sft-lora-Qwen3-4B-v1"

# SAVE_DIR="./results/$model_name"

# mkdir -p $SAVE_DIR

# lm-eval run --config lm-eval_config.yaml \
#     --model_args pretrained=$MODEL dtype=bfloat16 \
#     --output_path $SAVE_DIR \
#     2>&1 | tee -a $SAVE_DIR/eval_ceval.log


# DATASET_NAME="/cephfs/songyue/zzlai/hf_cache/hub/datasets--FreedomIntelligence--Huatuo26M-Lite/snapshots/90ce61699e90568db82cdce4c4035c6918d70aa6"

# python eval_ppl.py --model $MODEL --dataset_name $DATASET_NAME --output_dir $SAVE_DIR



# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-v2/merged_model"
# model_name="sft-lora-Qwen3-4B-v2"

# SAVE_DIR="./results/$model_name"

# mkdir -p $SAVE_DIR

# lm-eval run --config lm-eval_config.yaml \
#     --model_args pretrained=$MODEL dtype=bfloat16 \
#     --output_path $SAVE_DIR \
#     2>&1 | tee -a $SAVE_DIR/eval_ceval.log


# DATASET_NAME="/cephfs/songyue/zzlai/hf_cache/hub/datasets--FreedomIntelligence--Huatuo26M-Lite/snapshots/90ce61699e90568db82cdce4c4035c6918d70aa6"

# python eval_ppl.py --model $MODEL --dataset_name $DATASET_NAME --output_dir $SAVE_DIR



# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-grpo-lora-Qwen3-4B-Instruct-2507-v3-v1/merged_model"
# model_name="grpo-lora-Qwen3-4B-Instruct-2507-v3-v1"

# SAVE_DIR="./results/$model_name"

# mkdir -p $SAVE_DIR

# lm-eval run --config lm-eval_config.yaml \
#     --model_args pretrained=$MODEL dtype=bfloat16 \
#     --output_path $SAVE_DIR \
#     2>&1 | tee -a $SAVE_DIR/eval_ceval.log


# DATASET_NAME="/cephfs/songyue/zzlai/hf_cache/hub/datasets--FreedomIntelligence--Huatuo26M-Lite/snapshots/90ce61699e90568db82cdce4c4035c6918d70aa6"

# python eval_ppl.py --model $MODEL --dataset_name $DATASET_NAME --output_dir $SAVE_DIR


MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-reason-lora-Qwen3-4B-Instruct-2507-v3-v1/merged_model"
model_name="sft-reason-lora-Qwen3-4B-Instruct-2507-v3-v1"

SAVE_DIR="./results/$model_name"

mkdir -p $SAVE_DIR

lm-eval run --config lm-eval_config.yaml \
    --model_args pretrained=$MODEL dtype=bfloat16 \
    --output_path $SAVE_DIR \
    2>&1 | tee -a $SAVE_DIR/eval_ceval.log


DATASET_NAME="/cephfs/songyue/zzlai/hf_cache/hub/datasets--FreedomIntelligence--Huatuo26M-Lite/snapshots/90ce61699e90568db82cdce4c4035c6918d70aa6"

python eval_ppl.py --model $MODEL --dataset_name $DATASET_NAME --output_dir $SAVE_DIR