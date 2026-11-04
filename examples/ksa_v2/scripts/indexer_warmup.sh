#!/usr/bin/env bash
# C-KSA group128 + S-KSA block4: train aggregation and Indexer only.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Paths are relative to the repository root.
MODEL_CONFIG=${MODEL_CONFIG:-examples/ksa_v2/configs/cksa_g128_sksa_block4_1b9_128k.json}
DATASET_CONFIG=${DATASET_CONFIG:-examples/ksa_v2/dataset_config/demo_128k.json}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-output/ksa_v2/full_cpt}
CHECKPOINT_ID=${CHECKPOINT_ID:-global_step2400}
OUTPUT_DIR=${OUTPUT_DIR:-output/ksa_v2/indexer}

# Training
GBS=${GBS:-32}
LR=${LR:-1e-3}
WARMUP_STEPS=${WARMUP_STEPS:-200}
TOTAL_STEPS=${TOTAL_STEPS:-4400}
SAVE_EVERY=${SAVE_EVERY:-200}
SAVE_STEPS=${SAVE_STEPS:-10}
RESUME=${RESUME:-0}
RESUME_DATALOADER=${RESUME_DATALOADER:-1}
SOURCE_GBS=${SOURCE_GBS:-}

# One copy of this command runs on each node.
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
[[ "$WORLD_SIZE" -gt 0 && "$GBS" -ge "$WORLD_SIZE" && $((GBS % WORLD_SIZE)) -eq 0 ]] || {
    echo "GBS must be a positive multiple of NNODES * GPUS_PER_NODE" >&2
    exit 1
}
GRAD_ACC=$((GBS / WORLD_SIZE))

# Keep caches local; select the V2 kernels before importing the recipe.
export PYTHONPATH="$REPO_ROOT/summary_attention_kernel/ksa_v2/v0.1:$REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKSA_LIGHTNING_BACKEND=cute
export SKSA_SUMMARY_SPARSE_CORE=cute_mask
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/output/ksa_v2/.cache}"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export HF_MODULES_CACHE="$XDG_CACHE_HOME/modules"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export CUTE_DSL_CACHE_DIR="$XDG_CACHE_HOME/cute"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_SIZE

resume_flags=()
if [[ "$RESUME" == 1 ]]; then
    [[ "$RESUME_DATALOADER" == 1 ]] || {
        echo "RESUME=1 also requires RESUME_DATALOADER=1" >&2
        exit 1
    }
    resume_flags+=(--resume-optimizer)
else
    resume_flags+=(--allow-random-init-params "sksa_indexer.query_down_proj.weight,sksa_indexer.query_up_proj.weight,sksa_indexer.key_proj.weight,sksa_indexer.head_weight_proj.weight,sksa_indexer.chunk_q_proj.weight,sksa_indexer.chunk_k_proj.weight,sksa_indexer.chunk_q_norm.scale,sksa_indexer.chunk_k_norm.scale,sksa_indexer.chunk_v_proj.weight")
fi
if [[ "$RESUME_DATALOADER" == 1 ]]; then
    resume_flags+=(--resume-dataloader)
    if [[ "$RESUME" != 1 && -n "$SOURCE_GBS" ]]; then
        resume_flags+=(--resume-dataloader-source-global-batch-size "$SOURCE_GBS")
    fi
fi

command=(
    python -m torch.distributed.run
    --nnodes "$NNODES"
    --nproc-per-node "$GPUS_PER_NODE"
    --node-rank "${NODE_RANK:-0}"
    --master-addr "${MASTER_ADDR:-127.0.0.1}"
    --master-port "${MASTER_PORT:-29500}"
    recipes/ksa_v2.py
    --model-config "$MODEL_CONFIG"
    --dataset-config "$DATASET_CONFIG"
    --output-dir "$OUTPUT_DIR"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --checkpoint-id "$CHECKPOINT_ID"
    --resume-weights
    --context-parallel-size 1
    --gradient-accumulation-steps "$GRAD_ACC"
    --chunked-loss-minibatch-size 1024
    --learning-rate "$LR"
    --warmup-start-lr 1e-6
    --min-lr "$LR"
    --lr-scheduler-type wsd
    --num-warmup-steps "$WARMUP_STEPS"
    --num-decay-steps 0
    --num-training-steps "$TOTAL_STEPS"
    --save-checkpoint-per-step "$SAVE_EVERY"
    --save-checkpoint-steps "$SAVE_STEPS,$TOTAL_STEPS"
    --weight-decay 0.01
    --beta1 0.9
    --beta2 0.95
    --clip-range 1.0
    --fp32-weight
    --use-flash-attention-2
    --enable-gradient-checkpointing
    --logging-per-step 1
    --seed 42
    --comment "indexer_warmup"
    --commit-id "${SOURCE_COMMIT:-$(git rev-parse --short HEAD)}"
    --train-sksa-indexer-only
    --enable-sksa-indexer-distill
    --sksa-dense-warmup
    --sksa-indexer-kl-factor 1.0
    --sksa-indexer-monitor-every 10
    --sksa-indexer-monitor-topk-blocks 1024
    "${resume_flags[@]}"
)
if [[ "${1:-}" == --dry-run ]]; then
    shift
    printf '%q ' "${command[@]}" "$@"
    printf '\n'
    exit 0
fi
mkdir -p "$TMPDIR"
exec "${command[@]}" "$@"
