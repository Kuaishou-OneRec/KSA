"""Direct-indirect sparse attention forward prototype.

Unlike the generic Flash/CuTe mask path, this kernel iterates the compact
``selected_block_indices`` directly.  It therefore computes at most
``TopK * span + local_window + alignment - 1`` raw K/V slots per query instead
of taking a union inside a 128-query scheduling tile.  The extra
``alignment - 1`` slots cover the aligned local-window boundary.

Backward is intentionally provided by the established CuTe implementation in
``interface.py`` for v0.1.  This keeps training gradients exact while the
direct sparse dK/dV kernel (which needs an efficient inverse route rather than
per-score atomics) is developed independently.
"""

from __future__ import annotations


import torch
import triton
import triton.language as tl


@triton.jit
def _summary_block_sparse_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    lse_ptr,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    v_stride_b,
    v_stride_s,
    v_stride_h,
    route_stride_b,
    route_stride_s,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    lse_stride_b,
    lse_stride_h,
    sequence_length,
    q_head_count: tl.constexpr,
    kv_head_count: tl.constexpr,
    head_dim: tl.constexpr,
    topk_blocks: tl.constexpr,
    block_size_tokens: tl.constexpr,
    local_alignment_tokens: tl.constexpr,
    local_window_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    total_slots_padded: tl.constexpr,
):
    program = tl.program_id(0)
    query_head = program % q_head_count
    batch_query = program // q_head_count
    query_index = batch_query % sequence_length
    batch_index = batch_query // sequence_length
    q_per_kv = q_head_count // kv_head_count
    kv_head = query_head // q_per_kv

    dim_offsets = tl.arange(0, block_d)
    dim_mask = dim_offsets < head_dim
    q = tl.load(
        q_ptr
        + batch_index * q_stride_b
        + query_index * q_stride_s
        + query_head * q_stride_h
        + dim_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    remote_slots = topk_blocks * block_size_tokens
    raw_local_start = tl.maximum(
        query_index - local_window_tokens + 1,
        0,
    )
    local_start = (
        raw_local_start // local_alignment_tokens
    ) * local_alignment_tokens
    local_token_count = query_index - local_start + 1
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_d,), dtype=tl.float32)

    for slot_start in tl.range(0, total_slots_padded, block_n):
        slot_offsets = slot_start + tl.arange(0, block_n)
        is_remote = slot_offsets < remote_slots
        route_slot = slot_offsets // block_size_tokens
        raw_offset = slot_offsets % block_size_tokens
        route_id = tl.load(
            selected_ptr
            + batch_index * route_stride_b
            + query_index * route_stride_s
            + route_slot,
            mask=is_remote & (route_slot < topk_blocks),
            other=-1,
        )
        remote_key = route_id * block_size_tokens + raw_offset
        local_offset = slot_offsets - remote_slots
        local_key = local_start + local_offset
        key_index = tl.where(is_remote, remote_key, local_key)

        # Selected blocks are a set union with the local window.  Masking the
        # selected copy once it reaches local_start prevents double counting
        # even if a diagnostic caller supplies an overlapping route.
        remote_valid = (
            is_remote
            & (route_id >= 0)
            & (remote_key >= 0)
            & (remote_key < local_start)
            & (remote_key <= query_index)
            & (remote_key < sequence_length)
        )
        local_valid = (
            (~is_remote)
            & (local_offset >= 0)
            & (local_offset < local_token_count)
            & (local_key >= 0)
            & (local_key <= query_index)
            & (local_key < sequence_length)
        )
        key_valid = remote_valid | local_valid
        safe_key = tl.where(key_valid, key_index, 0)
        key = tl.load(
            k_ptr
            + batch_index * k_stride_b
            + safe_key[:, None] * k_stride_s
            + kv_head * k_stride_h
            + dim_offsets[None, :],
            mask=key_valid[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key * q[None, :], axis=1) * softmax_scale
        scores = tl.where(key_valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        has_finite = new_max != -float("inf")
        old_scale = tl.where(
            has_finite,
            tl.exp(running_max - new_max),
            1.0,
        )
        probability = tl.where(
            key_valid & has_finite,
            tl.exp(scores - new_max),
            0.0,
        )
        value = tl.load(
            v_ptr
            + batch_index * v_stride_b
            + safe_key[:, None] * v_stride_s
            + kv_head * v_stride_h
            + dim_offsets[None, :],
            mask=key_valid[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * old_scale + tl.sum(
            probability[:, None] * value, axis=0
        )
        running_sum = running_sum * old_scale + tl.sum(probability, axis=0)
        running_max = new_max

    output = accumulator / running_sum
    tl.store(
        out_ptr
        + batch_index * out_stride_b
        + query_index * out_stride_s
        + query_head * out_stride_h
        + dim_offsets,
        output,
        mask=dim_mask,
    )
    tl.store(
        lse_ptr
        + batch_index * lse_stride_b
        + query_head * lse_stride_h
        + query_index,
        tl.log(running_sum) + running_max,
    )


def summary_block_sparse_direct_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_block_indices: torch.Tensor,
    *,
    block_size_tokens: int,
    local_window_tokens: int,
    local_alignment_tokens: int | None = None,
    softmax_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct sparse output and LSE in CuTe-compatible layout."""
    if block_size_tokens not in (1, 2, 4, 8):
        raise ValueError("direct v0.1 kernel supports 1/2/4/8-token routes")
    if local_alignment_tokens is None:
        local_alignment_tokens = block_size_tokens
    if local_alignment_tokens <= 0:
        raise ValueError("local_alignment_tokens must be positive")
    if local_alignment_tokens % block_size_tokens:
        raise ValueError("local alignment must be divisible by route span")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have [batch, text, heads, head_dim]")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError("direct sparse attention requires matching batch/text KV")
    if q.shape[2] % k.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("direct sparse attention requires equal Q/K/V head dims")
    if q.shape[-1] not in (64, 128):
        raise ValueError("direct v0.1 kernel supports head_dim 64 or 128")
    if tuple(selected_block_indices.shape[:2]) != tuple(q.shape[:2]):
        raise ValueError("selected block routes must match batch/query dimensions")
    if selected_block_indices.dtype != torch.int32:
        selected_block_indices = selected_block_indices.to(torch.int32)
    selected_block_indices = selected_block_indices.contiguous()

    batch, sequence_length, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]
    topk_blocks = selected_block_indices.shape[-1]
    scale = float(softmax_scale) if softmax_scale is not None else head_dim ** -0.5
    block_n = 32
    block_d = triton.next_power_of_2(head_dim)
    # Aligning the local start down to a raw block boundary can expose up to
    # ``block_size_tokens - 1`` additional keys.
    total_slots = (
        topk_blocks * block_size_tokens
        + local_window_tokens
        + local_alignment_tokens
        - 1
    )
    total_slots_padded = triton.cdiv(total_slots, block_n) * block_n
    output = torch.empty_like(q)
    lse = torch.empty(
        (batch, q_heads, sequence_length),
        dtype=torch.float32,
        device=q.device,
    )
    grid = (batch * sequence_length * q_heads,)
    _summary_block_sparse_fwd_kernel[grid](
        q,
        k,
        v,
        selected_block_indices,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        selected_block_indices.stride(0),
        selected_block_indices.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        sequence_length,
        q_head_count=q_heads,
        kv_head_count=kv_heads,
        head_dim=head_dim,
        topk_blocks=topk_blocks,
        block_size_tokens=block_size_tokens,
        local_alignment_tokens=local_alignment_tokens,
        local_window_tokens=local_window_tokens,
        softmax_scale=scale,
        block_n=block_n,
        block_d=block_d,
        total_slots_padded=total_slots_padded,
        num_warps=4,
        num_stages=2,
    )
    return output, lse


def direct_sparse_slot_count(
    topk_blocks: int,
    local_window_tokens: int,
    *,
    block_size_tokens: int = 8,
    local_alignment_tokens: int | None = None,
) -> int:
    """Maximum raw K/V slots computed per query before causal padding."""
    alignment = (
        block_size_tokens
        if local_alignment_tokens is None
        else local_alignment_tokens
    )
    return topk_blocks * block_size_tokens + local_window_tokens + alignment - 1


__all__ = [
    "summary_block_sparse_direct_forward",
    "direct_sparse_slot_count",
]
