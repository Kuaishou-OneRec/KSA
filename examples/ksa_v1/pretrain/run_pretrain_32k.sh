#!/usr/bin/env bash
set -euo pipefail

CONTEXT=32k
LR=${LR:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-100}
DECAY_STEPS=${DECAY_STEPS:-100}
TOTAL_STEPS=${TOTAL_STEPS:-1600}
SAVE_STEPS=${SAVE_STEPS:-250}
source "$(dirname "${BASH_SOURCE[0]}")/run_pretrain.sh"
