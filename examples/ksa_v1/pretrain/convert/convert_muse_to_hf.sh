#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <RUN_DIR> <STEP> <V1_HF_TEMPLATE> [DTYPE] [TRAINING_CONFIG]" >&2
    exit 1
fi
extra=()
if [[ -n "${5:-}" ]]; then
    extra+=(--model-config "$5")
fi
python3 "$(dirname "${BASH_SOURCE[0]}")/convert.py" \
    --dcp_path "$1" \
    --step "$2" \
    --base_model_path "$3" \
    --dtype "${4:-bf16}" \
    --remap muse2hf \
    "${extra[@]}"
