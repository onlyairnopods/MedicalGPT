export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HOME="/cephfs/songyue/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


CUDA_VISIBLE_DEVICES=1 python build_candidate_embeddings_v3.py \
  --input split_shards/train_zh_0_shard_0000.jsonl \
  --output-dir emb_shards/emb_0000 \
  --batch-size 2048 \
  --max-context-turns 6 \
  --max-message-chars 512 \
  --assistant-prefix-chars 300 \
  --max-total-chars 1200 \
  --max-seq-length 1024 \
  --normalize-embeddings \
  --fp16 \
  --inference-fp16 &

CUDA_VISIBLE_DEVICES=2 python build_candidate_embeddings_v3.py \
  --input split_shards/train_zh_0_shard_0001.jsonl \
  --output-dir emb_shards/emb_0001 \
  --batch-size 2048 \
  --max-context-turns 6 \
  --max-message-chars 512 \
  --assistant-prefix-chars 300 \
  --max-total-chars 1200 \
  --max-seq-length 1024 \
  --normalize-embeddings \
  --fp16 \
  --inference-fp16 &

CUDA_VISIBLE_DEVICES=3 python build_candidate_embeddings_v3.py \
  --input split_shards/train_zh_0_shard_0002.jsonl \
  --output-dir emb_shards/emb_0002 \
  --batch-size 2048 \
  --max-context-turns 6 \
  --max-message-chars 512 \
  --assistant-prefix-chars 300 \
  --max-total-chars 1200 \
  --max-seq-length 1024 \
  --normalize-embeddings \
  --fp16 \
  --inference-fp16


CUDA_VISIBLE_DEVICES=1 python build_candidate_embeddings_v3.py \
  --input split_shards/train_zh_0_shard_0003.jsonl \
  --output-dir emb_shards/emb_0003 \
  --batch-size 2048 \
  --max-context-turns 6 \
  --max-message-chars 512 \
  --assistant-prefix-chars 300 \
  --max-total-chars 1200 \
  --max-seq-length 1024 \
  --normalize-embeddings \
  --fp16 \
  --inference-fp16 &

CUDA_VISIBLE_DEVICES=2 python build_candidate_embeddings_v3.py \
  --input split_shards/train_zh_0_shard_0004.jsonl \
  --output-dir emb_shards/emb_0004 \
  --batch-size 2048 \
  --max-context-turns 6 \
  --max-message-chars 512 \
  --assistant-prefix-chars 300 \
  --max-total-chars 1200 \
  --max-seq-length 1024 \
  --normalize-embeddings \
  --fp16 \
  --inference-fp16 &

CUDA_VISIBLE_DEVICES=3 python build_candidate_embeddings_v3.py \
  --input split_shards/train_zh_0_shard_0005.jsonl \
  --output-dir emb_shards/emb_0005 \
  --batch-size 2048 \
  --max-context-turns 6 \
  --max-message-chars 512 \
  --assistant-prefix-chars 300 \
  --max-total-chars 1200 \
  --max-seq-length 1024 \
  --normalize-embeddings \
  --fp16 \
  --inference-fp16

CUDA_VISIBLE_DEVICES=1 python build_candidate_embeddings_v3.py \
  --input split_shards/train_zh_0_shard_0006.jsonl \
  --output-dir emb_shards/emb_0006 \
  --batch-size 2048 \
  --max-context-turns 6 \
  --max-message-chars 512 \
  --assistant-prefix-chars 300 \
  --max-total-chars 1200 \
  --max-seq-length 1024 \
  --normalize-embeddings \
  --fp16 \
  --inference-fp16 &

CUDA_VISIBLE_DEVICES=2 python build_candidate_embeddings_v3.py \
  --input split_shards/train_zh_0_shard_0007.jsonl \
  --output-dir emb_shards/emb_0007 \
  --batch-size 2048 \
  --max-context-turns 6 \
  --max-message-chars 512 \
  --assistant-prefix-chars 300 \
  --max-total-chars 1200 \
  --max-seq-length 1024 \
  --normalize-embeddings \
  --fp16 \
  --inference-fp16

wait

echo "All shards finished."