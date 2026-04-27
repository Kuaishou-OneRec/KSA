"""
Model FLOPs Utilization (MFU) Calculator.

Computes per-step FLOPs from model config and reports MFU as a fraction of
GPU theoretical peak. Independent of model code — only reads config values.

Two metrics:
  - MFU (Model FLOPs Utilization): uses minimum necessary FLOPs (3x fwd),
    standard definition from PaLM paper. Grad ckpt / distill overhead
    shows up as lower MFU (more time for same model FLOPs).
  - HFU (Hardware FLOPs Utilization): uses actual FLOPs including
    recomputation and distillation. Measures raw hardware efficiency.

Usage:
    from muse.utils.mfu import estimate_flops, get_gpu_peak_tflops

    model_flops, hardware_flops = estimate_flops(
        num_layers=36, hidden_size=2560, head_dim=128,
        num_heads=32, num_kv_heads=8, intermediate_size=9728,
        vocab_size=151936, seq_length=32768,
        gradient_checkpointing=True, distill=True,
    )
    peak = get_gpu_peak_tflops()
    mfu = model_flops / (step_time * 1e12 * peak)
    hfu = hardware_flops / (step_time * 1e12 * peak)
"""

import torch


# BF16 peak TFLOP/s per GPU (vendor specs, tensor core)
_GPU_PEAK_TFLOPS_BF16 = {
    "A100": 312,
    "A800": 312,
    "H100": 989,
    "H800": 989,
    "H20":  148,
}


def get_gpu_peak_tflops(override: float = None) -> float:
    """Auto-detect GPU peak BF16 TFLOP/s, or use override.

    Returns 0.0 if detection fails and no override is given (MFU will be skipped).
    """
    if override is not None and override > 0:
        return override

    if not torch.cuda.is_available():
        return 0.0

    name = torch.cuda.get_device_name(0)
    for key, val in _GPU_PEAK_TFLOPS_BF16.items():
        if key in name:
            return float(val)
    return 0.0


def _compute_fwd_flops(
    num_layers: int,
    hidden_size: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    seq_length: int,
    batch_size: int = 1,
) -> float:
    """Forward FLOPs for a single pass through the transformer."""
    b, s, h = batch_size, seq_length, hidden_size
    h_q = num_heads * head_dim
    h_kv = num_kv_heads * head_dim

    # Per-layer FLOPs (forward, per token)
    qkv = 2 * h * (h_q + 2 * h_kv)        # QKV projection
    attn_core = 2 * s * h_q                 # QK^T + softmax*V (causal: s/2 avg positions)
    out_proj = 2 * h_q * h                  # Output projection
    mlp = 2 * h * intermediate_size * 2 + 2 * intermediate_size * h  # SwiGLU

    per_layer_per_token = qkv + attn_core + out_proj + mlp
    fwd_layers = per_layer_per_token * num_layers * b * s
    fwd_logit = 2 * h * vocab_size * b * s

    return float(fwd_layers + fwd_logit)


def estimate_flops(
    num_layers: int,
    hidden_size: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    seq_length: int,
    batch_size: int = 1,
    gradient_checkpointing: bool = False,
    distill: bool = False,
    summary_chunk_size: int = 0,
    summary_token_num: int = 0,
) -> tuple:
    """Estimate training FLOPs for one micro-batch.

    Args:
        summary_chunk_size: If >0, every chunk_size tokens gets summary_token_num
            summary tokens inserted. Student forward uses expanded seq_length,
            teacher forward (distill) uses original seq_length.
        summary_token_num: Number of summary tokens per chunk.

    Returns:
        (model_flops, hardware_flops):
          - model_flops: minimum necessary FLOPs (3x student fwd), for MFU
          - hardware_flops: actual FLOPs including recomputation/distill, for HFU
    """
    import math

    # Student seq_length: original + summary tokens
    if summary_chunk_size > 0 and summary_token_num > 0:
        num_chunks = math.ceil(seq_length / summary_chunk_size)
        student_seq = seq_length + num_chunks * summary_token_num
    else:
        student_seq = seq_length

    common_args = dict(
        num_layers=num_layers, hidden_size=hidden_size, head_dim=head_dim,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size, vocab_size=vocab_size,
        batch_size=batch_size,
    )

    student_fwd = _compute_fwd_flops(seq_length=student_seq, **common_args)

    # Model FLOPs (MFU): 3x student fwd (fwd + bwd), no recomputation
    model_flops = student_fwd * 3

    # Hardware FLOPs (HFU): actual compute
    hw_multiplier = 4 if gradient_checkpointing else 3  # +1x recomputation
    hardware_flops = student_fwd * hw_multiplier

    if distill:
        # Teacher forward uses original seq_length (no summary tokens, no backward)
        teacher_fwd = _compute_fwd_flops(seq_length=seq_length, **common_args)
        hardware_flops += teacher_fwd

    return float(model_flops), float(hardware_flops)
