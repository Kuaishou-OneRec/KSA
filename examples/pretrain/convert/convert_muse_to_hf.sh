#!/usr/bin/env bash
# Convert a Muse DCP checkpoint to a HuggingFace-compatible directory for
# inference / evaluation. The converted HF folder is written under
# <OUTPUT_DIR>/<STEP>/hf.
#
# Prereq: a base HF model (same architecture family, e.g. Qwen3-1.7B) whose
# tokenizer / configs / modeling_*.py will be copied alongside the converted
# weights. Only the weights come from the DCP — everything else is borrowed
# from BASE_MODEL.
#
# Usage:
#   bash examples/pretrain/convert/convert_muse_to_hf.sh \
#       /path/to/muse_outputs/1b9_sa_hybrid_8k \
#       global_step5000 \
#       /path/to/hf/Qwen3-1.7B

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <DCP_OUTPUT_DIR> <STEP> <BASE_HF_MODEL> [DTYPE]"
  echo "  DCP_OUTPUT_DIR : training output dir that contains global_step* subdirs"
  echo "  STEP           : e.g. global_step5000"
  echo "  BASE_HF_MODEL  : HF directory to copy tokenizer + config templates from"
  echo "  DTYPE          : bf16 (default) | fp16 | fp32"
  exit 1
fi

DCP_OUTPUT_DIR="$1"
STEP="$2"
BASE_HF_MODEL="$3"
DTYPE="${4:-bf16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/convert.py" \
  --dcp_path "${DCP_OUTPUT_DIR}" \
  --step "${STEP}" \
  --base_model_path "${BASE_HF_MODEL}" \
  --dtype "${DTYPE}" \
  --remap muse2hf

echo
echo "Converted HF model: ${DCP_OUTPUT_DIR}/${STEP}/hf"
