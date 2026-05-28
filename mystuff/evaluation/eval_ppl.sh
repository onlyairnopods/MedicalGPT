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
export CUDA_VISIBLE_DEVICES=1

conda activate /cephfs/songyue/zzlai/my_conda_env/llava

MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Llama-3.2-3B-Instruct-v1/merged_model"
model_name="sft-lora-Llama-3.2-3B-Instruct-v1"

# MODEL="/cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-Instruct-2507-v1/merged_model"
# model_name="sft-lora-Qwen3-4B-Instruct-2507-v1"

DATASET_NAME="/cephfs/songyue/zzlai/hf_cache/hub/datasets--FreedomIntelligence--Huatuo26M-Lite/snapshots/90ce61699e90568db82cdce4c4035c6918d70aa6"

SAVE_DIR="./results/$model_name"

mkdir -p $SAVE_DIR

python eval_ppl.py --model $MODEL --dataset_name $DATASET_NAME --output_dir $SAVE_DIR