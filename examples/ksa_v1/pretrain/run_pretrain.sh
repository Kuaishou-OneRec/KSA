#!/usr/bin/env bash
# Shared torchrun launcher; sourced by run_pretrain_{8,32,64,128}k.sh.
KSA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$KSA_REPO"
source "$KSA_REPO/set_env.sh"

export PYTHONPATH="$KSA_REPO/summary_attention_kernel/ksa_v2/v0.1:$KSA_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TMPDIR="${TMPDIR:-$KSA_REPO/.tmp}"
export HF_HOME="${HF_HOME:-$KSA_REPO/output/cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$KSA_REPO/output/cache/triton}"
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_RANK

OUTPUT_DIR=${OUTPUT_DIR:-output/ksa_v1/1b9_hybrid_$CONTEXT}
MODEL_CONFIG=${MODEL_CONFIG:-examples/ksa_v1/pretrain/model_config/model_config_1b9_hybrid.json}
DATASET_CONFIG=${DATASET_CONFIG:-examples/ksa_v1/pretrain/dataset_config/pretrain_kai_mmap_$CONTEXT.json}
extra=()
if [[ "$CONTEXT" != 8k || -n "${CHECKPOINT_DIR:-}" ]]; then
    extra+=(
        --checkpoint-dir "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the parent checkpoint run}"
        --resume-weights
    )
fi
if [[ "${RESUME:-0}" == 1 ]]; then
    extra+=(
        --resume-optimizer
        --resume-dataloader
    )
fi
if [[ "$CONTEXT" != 8k ]]; then
    extra+=(--enable-gradient-checkpointing)
fi

command=(
    python -m torch.distributed.run
    --nnodes "${NNODES:-1}"
    --nproc-per-node "${GPUS_PER_NODE:-8}"
    --node-rank "${NODE_RANK:-0}"
    --master-addr "${MASTER_ADDR:-127.0.0.1}"
    --master-port "${MASTER_PORT:-29500}"
    recipes/ksa_v1.py
    --model-config "$MODEL_CONFIG"
    --output-dir "$OUTPUT_DIR"
    --dataset-config "$DATASET_CONFIG"
    --context-parallel-size "${CP_SIZE:-1}"
    --gradient-accumulation-steps "${GRAD_ACC:-1}"
    --chunked-loss-minibatch-size 1024
    --disable-chunked-loss
    --learning-rate "$LR"
    --min-lr "${MIN_LR:-1e-7}"
    --weight-decay 0.01
    --beta1 0.9
    --beta2 0.95
    --fp32-weight
    --lr-scheduler-type wsd
    --num-warmup-steps "$WARMUP_STEPS"
    --num-decay-steps "$DECAY_STEPS"
    --num-training-steps "$TOTAL_STEPS"
    --save-checkpoint-per-step "$SAVE_STEPS"
    --enable-dataset-checkpointing
    --use-flash-attention-2
    --logging-per-step 1
    --clip-range 1.0
    --seed 42
    --comment "ksa_v1_1b9_hybrid_$CONTEXT"
    --commit-id "$(git rev-parse --short HEAD)"
    "${extra[@]}"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "$TMPDIR"
exec "${command[@]}" "$@"
