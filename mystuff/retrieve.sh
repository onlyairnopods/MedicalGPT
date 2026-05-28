export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HOME="/cephfs/songyue/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1


OUTPUT_FILE="./selected_train_zh_0_share_v3.jsonl"

python retrieve_by_prototype_faiss_v2.py \
    --input train_zh_0_share.jsonl \
    --prototypes prototype_expanded.jsonl \
    --emb-dir emb_cache_merged/ \
    --index emb_cache/faiss_ivf_flat.index \
    --output $OUTPUT_FILE \
    --normalize-queries \
    --top-high 80 --top-mid 15 --top-tail 10 --mid-width 10000 --tail-width 1000 --nprobe 4096 \
    --high-ratio 0.70 --mid-ratio 0.10 --tail-ratio 0.01 \
    --overwrite
    # --debug-output ./selected_train_zh_0_share_test_debug.jsonl \
    # --warnings-output ./selected_train_zh_0_share_test_warn.jsonl \
    # --model /cephfs/songyue/zzlai/hf_cache/hub/models--lastmass--Qwen3-Embedding-Medical-0.6B/snapshots/23e6fb533f46b3372efc2935b6d2c0715f83cfe4


ANALYSIS_OUT=./analysis_out_v3
mkdir -p $ANALYSIS_OUT

python analyze_retrieved_data.py \
    --input $OUTPUT_FILE \
    --output-dir $ANALYSIS_OUT \
    --write-filtered \
    --min-score 0.45 \
    --min-assistant-chars 5 \
    --max-total-chars 4000 \
    --dedupe normalized \
    --per-prototype-topk 3000 \
    --per-prototype-min-quantile 0.10

python plot_prototype_stats.py \
    --input $ANALYSIS_OUT/prototype_stats.csv \
    --output-dir $ANALYSIS_OUT/prototype_plots