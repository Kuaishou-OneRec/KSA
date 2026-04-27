#!/bin/bash
# Qwen3-1.6B Summary Attention — 32k pretrain continuing from 8k DCP checkpoint.
#
# Two modes (controlled by RESUME_DATALOADER below):
#   - RESUME_DATALOADER=0: first launch. Load weights only from 8k DCP,
#                          start 32k training from step 0 with a fresh dataloader.
#   - RESUME_DATALOADER=1: mid-run resume. CHECKPOINT_DIR must point to this
#                          32k run's own output; weights + dataloader state both restored.
#
# Usage: bash examples/pretrain/run_pretrain_32k.sh

set -e

# ============================================
# Paths — ADJUST THESE
# ============================================
CHECKPOINT_DIR=/path/to/muse_outputs/1b6_sa_hybrid_8k   # 8k DCP on first launch; this 32k output on resume
MODEL_CONFIG=examples/pretrain/model_config/model_config_1b6_hybrid.json
DATASET_CONFIG=examples/pretrain/dataset_config/pretrain_kai_mmap_32k.json
OUTPUT_DIR=/path/to/muse_outputs/1b6_sa_hybrid_32k

# ============================================
# Training hyperparameters
# ============================================
CP_SIZE=1
GRAD_ACC=1
LR=1e-5
MIN_LR=1e-7
WARMUP_STEPS=100
DECAY_STEPS=100
TOTAL_STEPS=1600
CHUNK_SIZE=1024
USE_CHUNKED_CE=0          # 1 = chunked CE (memory-efficient), 0 = standard CE
RESUME_DATALOADER=1       # 0 = first launch from 8k; 1 = resume this 32k run

# ============================================
# Infrastructure
# ============================================
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source set_env.sh

mkdir -p $OUTPUT_DIR

git_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
comment="pretrain_1b6_sa_hybrid_32k"

hostfile=/etc/mpi/hostfile_seq
sed 's/=1/=8/g' /etc/mpi/hostfile > $hostfile 2>/dev/null || true
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
TCP_NIC=$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*")

MASTER_ADDR=$MY_NODE_IP
MASTER_PORT=8499

set -x

nohup mpirun --allow-run-as-root \
        -hostfile $hostfile \
        -mca btl self,tcp -mca pml ob1 \
        -mca plm_rsh_num_concurrent 600 \
        -mca routed_radix 600 \
        -mca btl_tcp_if_include $TCP_NIC \
        -mca oob_tcp_if_include $TCP_NIC \
        -mca btl_openib_allow_ib false \
        -mca opal_set_max_sys_limits 1 \
        -x OMPI_MCA_btl=self,tcp \
        -x OMPI_MCA_pml=ob1 \
        -x OMPI_MCA_btl_tcp_if_include=$TCP_NIC \
        -x OMPI_MCA_oob_tcp_if_include=$TCP_NIC \
        -x OMPI_MCA_btl_openib_allow_ib=false \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_SOCKET_IFNAME=$TCP_NIC \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_DEBUG=WARN \
        -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_NET_OVERHEAD=1000 \
        -x NCCL_IB_TIMEOUT=20 \
        -x PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
        -x LD_PRELOAD=$LD_PRELOAD \
        -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
        -x PATH \
        -x PYTHONPATH=$PYTHONPATH \
        -x TOKENIZERS_PARALLELISM=false \
        -x MASTER_ADDR=$MASTER_ADDR \
        -x MASTER_PORT=$MASTER_PORT \
        -x http_proxy= \
        -x https_proxy= \
        with_nccl_local_env \
        bash -c "python3 recipes/pretrain_kai_summary_unified.py \
                --model-config $MODEL_CONFIG \
                --output-dir $OUTPUT_DIR \
                --dataset-config $DATASET_CONFIG \
                --checkpoint-dir $CHECKPOINT_DIR \
                --resume-weights \
                --enable-dataset-checkpointing \
                $([ $RESUME_DATALOADER -eq 1 ] && echo "--resume-dataloader" || echo "") \
                --context-parallel-size $CP_SIZE \
                --gradient-accumulation-steps $GRAD_ACC \
                --chunked-loss-minibatch-size $CHUNK_SIZE \
                $([ $USE_CHUNKED_CE -eq 0 ] && echo "--disable-chunked-loss" || echo "") \
                --learning-rate $LR \
                --min-lr $MIN_LR \
                --weight-decay 0.01 \
                --beta1 0.9 \
                --beta2 0.95 \
                --fp32-weight \
                --lr-scheduler-type wsd \
                --num-warmup-steps $WARMUP_STEPS \
                --num-decay-steps $DECAY_STEPS \
                --num-training-steps $TOTAL_STEPS \
                --save-checkpoint-per-step 250 \
                --use-flash-attention-2 \
                --enable-gradient-checkpointing \
                --logging-per-step 1 \
                --clip-range 1.0 \
                --seed 42 \
                --comment $comment \
                --commit-id $git_hash" > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &

echo "pretrain 32k started. Logs:"
echo "  tail -f $OUTPUT_DIR/stdout.log"
echo "  tail -f $OUTPUT_DIR/stderr.log"
