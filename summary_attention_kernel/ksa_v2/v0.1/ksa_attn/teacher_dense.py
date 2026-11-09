"""Exact dense-attention teacher kernels for the S-KSA Lightning indexer.

The warm-up teacher needs ordinary Full-layer causal attention probabilities,
but materializing ``[batch, heads, sequence, sequence]`` is not viable at long
context. These inference-only kernels make two streaming QK passes instead:
one for per-head causal log-sum-exp and one for head-mean, 8-token block scores.
Only ``[batch, query_tile, summary_blocks]`` is materialized.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_causal_lse_kernel(
    q_ptr, k_ptr, lse_ptr,
    q_stride_b, q_stride_s, q_stride_h,
    k_stride_b, k_stride_s, k_stride_h,
    lse_stride_b, lse_stride_h,
    sequence_length,
    softmax_scale: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    query_offsets = query_block * block_m + tl.arange(0, block_m)
    dim_offsets = tl.arange(0, block_d)
    query_valid = query_offsets < sequence_length
    dim_valid = dim_offsets < head_dim
    query = tl.load(
        q_ptr + batch * q_stride_b + query_offsets[:, None] * q_stride_s
        + head * q_stride_h + dim_offsets[None, :],
        mask=query_valid[:, None] & dim_valid[None, :], other=0.0,
    )
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((block_m,), dtype=tl.float32)
    key_limit = tl.minimum((query_block + 1) * block_m, sequence_length)
    for key_start in range(0, key_limit, block_n):
        key_offsets = key_start + tl.arange(0, block_n)
        key_valid = key_offsets < sequence_length
        key = tl.load(
            k_ptr + batch * k_stride_b + key_offsets[:, None] * k_stride_s
            + head * k_stride_h + dim_offsets[None, :],
            mask=key_valid[:, None] & dim_valid[None, :], other=0.0,
        )
        logits = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
        logits *= softmax_scale
        visible = query_valid[:, None] & key_valid[None, :] & (
            key_offsets[None, :] <= query_offsets[:, None]
        )
        logits = tl.where(visible, logits, -float("inf"))
        tile_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        old_scale = tl.exp(running_max - new_max)
        probability = tl.where(
            visible, tl.exp(logits - new_max[:, None]), 0.0
        )
        running_sum = running_sum * old_scale + tl.sum(probability, axis=1)
        running_max = new_max
    tl.store(
        lse_ptr + batch * lse_stride_b + head * lse_stride_h + query_offsets,
        tl.log(running_sum) + running_max,
        mask=query_valid,
    )


@triton.jit
def _dense_teacher_block_score_kernel(
    q_ptr, k_ptr, lse_ptr, target_ptr,
    q_stride_b, q_stride_s, q_stride_h,
    k_stride_b, k_stride_s, k_stride_h,
    lse_stride_b, lse_stride_h,
    target_stride_b, target_stride_q,
    query_start, query_count, sequence_length, summary_count,
    local_window_tokens,
    softmax_scale: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    summary_block_size: tl.constexpr,
    use_max_pool: tl.constexpr,
    block_m: tl.constexpr,
    block_meta: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    query_block = tl.program_id(0)
    meta_block = tl.program_id(1)
    batch = tl.program_id(2)
    local_query_offsets = query_block * block_m + tl.arange(0, block_m)
    query_offsets = query_start + local_query_offsets
    query_valid = local_query_offsets < query_count
    meta_offsets = meta_block * block_meta + tl.arange(0, block_meta)

    max_query = tl.minimum(
        query_start + (query_block + 1) * block_m - 1,
        query_start + query_count - 1,
    )
    max_local_start = tl.maximum(0, max_query - local_window_tokens + 1)
    max_eligible = max_local_start // summary_block_size
    if meta_block * block_meta >= max_eligible:
        return

    raw_offsets = meta_block * block_n + tl.arange(0, block_n)
    raw_valid = (raw_offsets < sequence_length) & (
        (raw_offsets // summary_block_size) < summary_count
    )
    dim_offsets = tl.arange(0, block_d)
    dim_valid = dim_offsets < head_dim
    token_probability = tl.zeros((block_m, block_n), dtype=tl.float32)
    for head in tl.static_range(0, num_heads):
        query = tl.load(
            q_ptr + batch * q_stride_b + query_offsets[:, None] * q_stride_s
            + head * q_stride_h + dim_offsets[None, :],
            mask=query_valid[:, None] & dim_valid[None, :], other=0.0,
        )
        key = tl.load(
            k_ptr + batch * k_stride_b + raw_offsets[:, None] * k_stride_s
            + head * k_stride_h + dim_offsets[None, :],
            mask=raw_valid[:, None] & dim_valid[None, :], other=0.0,
        )
        logits = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
        row_lse = tl.load(
            lse_ptr + batch * lse_stride_b + head * lse_stride_h
            + query_offsets,
            mask=query_valid, other=0.0,
        )
        probability = tl.exp(logits * softmax_scale - row_lse[:, None])
        token_probability += tl.where(
            query_valid[:, None] & raw_valid[None, :], probability, 0.0
        )
    token_probability *= 1.0 / num_heads
    grouped = tl.reshape(
        token_probability, (block_m, block_meta, summary_block_size)
    )
    if use_max_pool:
        block_score = tl.max(grouped, axis=2)
    else:
        block_score = tl.sum(grouped, axis=2)
    local_start = tl.maximum(0, query_offsets - local_window_tokens + 1)
    eligible_count = tl.minimum(
        summary_count, local_start // summary_block_size
    )
    block_valid = query_valid[:, None] & (
        meta_offsets[None, :] < summary_count
    ) & (meta_offsets[None, :] < eligible_count[:, None])
    tl.store(
        target_ptr + batch * target_stride_b
        + local_query_offsets[:, None] * target_stride_q
        + meta_offsets[None, :],
        block_score,
        mask=block_valid,
    )


def _validate_qk(q: torch.Tensor, k: torch.Tensor) -> None:
    if q.ndim != 4 or tuple(q.shape) != tuple(k.shape):
        raise ValueError("dense teacher q/k must share [B,text,H,D] shape")
    if not q.is_cuda or not k.is_cuda:
        raise ValueError("dense teacher Triton kernels require CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        raise ValueError("dense teacher requires matching CUDA fp16/bf16 q/k")
    if q.shape[-1] not in (64, 128):
        raise ValueError("dense teacher supports head_dim 64 or 128")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("dense teacher q/k head dimensions must be contiguous")


@torch.no_grad()
def dense_causal_attention_lse(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Return causal QK log-sum-exp in ``[B,H,S]`` without dense P."""
    _validate_qk(q, k)
    batch, sequence_length, heads, head_dim = q.shape
    lse = torch.empty(
        (batch, heads, sequence_length), dtype=torch.float32, device=q.device
    )
    block_m = block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(sequence_length, block_m), heads, batch)
    _dense_causal_lse_kernel[grid](
        q, k, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1), sequence_length,
        softmax_scale=head_dim ** -0.5,
        head_dim=head_dim, block_m=block_m, block_n=block_n,
        block_d=block_d, num_warps=4, num_stages=2,
    )
    return lse


@torch.no_grad()
def dense_teacher_block_distribution(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    *,
    query_start: int,
    query_end: int,
    summary_block_size: int,
    summary_count: int,
    local_window_tokens: int,
    pooling: str,
) -> torch.Tensor:
    """Return normalized summary-block teacher targets for one query tile."""
    _validate_qk(q, k)
    if summary_block_size not in (2, 4, 8):
        raise ValueError("dense teacher block size must be 2, 4, or 8")
    if pooling not in ("max", "sum"):
        raise ValueError(f"unsupported S-KSA teacher pooling: {pooling}")
    batch, sequence_length, heads, head_dim = q.shape
    if tuple(lse.shape) != (batch, heads, sequence_length):
        raise ValueError("dense teacher LSE must have [B,H,text] shape")
    if lse.dtype != torch.float32 or not lse.is_cuda:
        raise ValueError("dense teacher LSE must be CUDA float32")
    if not (0 <= query_start <= query_end <= sequence_length):
        raise ValueError("invalid dense teacher query range")
    if not (0 <= summary_count <= sequence_length // summary_block_size):
        raise ValueError("invalid dense teacher summary count")
    query_count = query_end - query_start
    target = torch.zeros(
        (batch, query_count, summary_count), dtype=torch.float32, device=q.device
    )
    if query_count == 0 or summary_count == 0:
        return target
    block_m, block_n = 16, 64
    block_meta = block_n // summary_block_size
    grid = (
        triton.cdiv(query_count, block_m),
        triton.cdiv(summary_count, block_meta),
        batch,
    )
    _dense_teacher_block_score_kernel[grid](
        q, k, lse, target,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1),
        target.stride(0), target.stride(1),
        query_start, query_count, sequence_length, summary_count,
        local_window_tokens,
        softmax_scale=head_dim ** -0.5,
        num_heads=heads, head_dim=head_dim,
        summary_block_size=summary_block_size,
        use_max_pool=pooling == "max",
        block_m=block_m, block_meta=block_meta, block_n=block_n,
        block_d=triton.next_power_of_2(head_dim),
        num_warps=8, num_stages=2,
    )
    denominator = target.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > 0,
        target / denominator.clamp_min(1e-30),
        target,
    )


__all__ = [
    "dense_causal_attention_lse",
    "dense_teacher_block_distribution",
]
