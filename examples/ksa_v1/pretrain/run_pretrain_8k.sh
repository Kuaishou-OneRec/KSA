#!/usr/bin/env bash
set -euo pipefail

CONTEXT=8k
LR=${LR:-4e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
DECAY_STEPS=${DECAY_STEPS:-1000}
TOTAL_STEPS=${TOTAL_STEPS:-127200}
SAVE_STEPS=${SAVE_STEPS:-5000}
source "$(dirname "${BASH_SOURCE[0]}")/run_pretrain.sh"
