#!/usr/bin/env bash
set -euo pipefail

# Run inside a dedicated Python 3.12 environment on a CUDA 12.6 development host.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.cache/pip}"
export TMPDIR="${TMPDIR:-$PWD/../../.tmp}"
mkdir -p "$TMPDIR"

python -m pip install \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install \
    packaging \
    ninja \
    setuptools \
    wheel
python -m pip install \
    flash-attn==2.7.4.post1 \
    --no-build-isolation
python -m pip install \
    -r requirements.txt
python -m pip install \
    --no-deps \
    ../../summary_attention_kernel/ksa_v2/v0.1/wheels/flash_attn_cute-0.1.0+sksa.n64bwd1-py3-none-any.whl
python -m pip install \
    --no-deps \
    --no-build-isolation \
    ./../../summary_attention_kernel/ksa_v2/v0.1
