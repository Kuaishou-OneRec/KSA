# Kwai Summary Attention — reference training / inference image.
#
# This Dockerfile builds a reproducible software stack using public upstream
# wheels (torch 2.6.0 + cu126) so the image builds on any host with a recent
# enough NVIDIA driver (>= 535).
#
# Build:
#   docker build -t ksa-train -f dockerfile/Dockerfile .
#
# Pinned versions:
#   Ubuntu 24.04 / CUDA 12.6 / Python 3.12 /
#   torch 2.6.0 / flash-attn 2.7.4.post1 / transformers 4.57.1

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates tzdata \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        openssh-client openssh-server openmpi-bin libopenmpi-dev pdsh \
        tmux vim lsof iputils-ping \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel ninja build packaging

# PyTorch 2.6.0 matched to CUDA 12.6 (public index).
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.6.0 \
        torchvision==0.21.0

# FlashAttention-2 (builds against the torch installed above).
RUN MAX_JOBS=16 pip install flash-attn==2.7.4.post1 --no-build-isolation

# Full training / inference / serving stack — see dockerfile/requirements.txt.
COPY dockerfile/requirements.txt /tmp/ksa_requirements.txt
RUN pip install -r /tmp/ksa_requirements.txt && rm -f /tmp/ksa_requirements.txt

# Summary Attention block-sparse kernel + flash-attn-cute (wheels shipped in the repo).
COPY summary_attention_kernel/*.whl /tmp/ksa/
RUN pip install /tmp/ksa/*.whl && rm -rf /tmp/ksa

WORKDIR /workspace
