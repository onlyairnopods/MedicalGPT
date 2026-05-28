export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HOME="/cephfs/songyue/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3

# python ../tools/merge_peft_adapter.py \
#    --base_model Qwen/Qwen3-4B-Instruct-2507 \
#    --tokenizer_path Qwen/Qwen3-4B-Instruct-2507 \
#    --lora_model /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-Instruct-2507-v3 \
#    --output_dir /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-Instruct-2507-v3/merged_model

# python ../tools/merge_peft_adapter.py \
#     --base_model meta-llama/Llama-3.2-3B-Instruct \
#     --tokenizer_path meta-llama/Llama-3.2-3B-Instruct \
#     --lora_model /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Llama-3.2-3B-Instruct-v3 \
#     --output_dir /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Llama-3.2-3B-Instruct-v3/merged_model

# python ../tools/merge_peft_adapter.py \
#     --base_model meta-llama/Llama-3.2-3B \
#     --tokenizer_path meta-llama/Llama-3.2-3B \
#     --lora_model /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Llama-3.2-3B-v2 \
#     --output_dir /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Llama-3.2-3B-v2/merged_model

# python ../tools/merge_peft_adapter.py \
#    --base_model Qwen/Qwen3-4B \
#    --tokenizer_path Qwen/Qwen3-4B \
#    --lora_model /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-v2 \
#    --output_dir /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-v2/merged_model

# python ../tools/merge_peft_adapter.py \
#    --base_model /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-Instruct-2507-v3/merged_model \
#    --tokenizer_path /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-lora-Qwen3-4B-Instruct-2507-v3/merged_model \
#    --lora_model /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-reason-lora-Qwen3-4B-Instruct-2507-v3-v1 \
#    --output_dir /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-reason-lora-Qwen3-4B-Instruct-2507-v3-v1/merged_model

python ../tools/merge_peft_adapter.py \
   --base_model /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-reason-lora-Qwen3-4B-Instruct-2507-v3-v1/merged_model \
   --tokenizer_path /cephfs/songyue/zzlai/MedicalGPT/outputs-sft-reason-lora-Qwen3-4B-Instruct-2507-v3-v1/merged_model \
   --lora_model /cephfs/songyue/zzlai/MedicalGPT/outputs-grpo-lora-Qwen3-4B-Instruct-2507-v3-v1 \
   --output_dir /cephfs/songyue/zzlai/MedicalGPT/outputs-grpo-lora-Qwen3-4B-Instruct-2507-v3-v1/merged_model
