"""Fused selected-set teacher kernels for Lightning-Indexer distillation.

The ordinary PyTorch expression first gathers a tensor shaped like
``[batch, query, topk, block, heads, head_dim]``.  At the production setting
(``topk=256``, ``block=8``) that temporary is several GiB even for a modest
query tile.  This Triton kernel follows the indirection directly and emits
only the QK logits.  Softmax/head aggregation/block pooling remain ordinary
PyTorch operations, which keeps their exact, easy-to-audit semantics.

"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_teacher_qk_kernel(
    q_ptr,
    k_ptr,
    route_ptr,
    logits_ptr,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    route_stride_b,
    route_stride_s,
    logits_stride_b,
    logits_stride_s,
    logits_stride_h,
    key_sequence_length,
    query_count: tl.constexpr,
    head_count: tl.constexpr,
    head_dim: tl.constexpr,
    topk_blocks: tl.constexpr,
    block_size_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    program = tl.program_id(0)
    head_index = program % head_count
    batch_query = program // head_count
    query_index = batch_query % query_count
    batch_index = batch_query // query_count

    dim_offsets = tl.arange(0, block_d)
    dim_mask = dim_offsets < head_dim
    query = tl.load(
        q_ptr
        + batch_index * q_stride_b
        + query_index * q_stride_s
        + head_index * q_stride_h
        + dim_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    token_count = topk_blocks * block_size_tokens
    for token_start in tl.range(0, token_count, block_n):
        token_slots = token_start + tl.arange(0, block_n)
        block_slots = token_slots // block_size_tokens
        raw_offsets = token_slots % block_size_tokens
        in_range = token_slots < token_count
        route_ids = tl.load(
            route_ptr
            + batch_index * route_stride_b
            + query_index * route_stride_s
            + block_slots,
            mask=in_range & (block_slots < topk_blocks),
            other=-1,
        )
        raw_indices = route_ids * block_size_tokens + raw_offsets
        valid = (
            in_range
            & (route_ids >= 0)
            & (raw_indices >= 0)
            & (raw_indices < key_sequence_length)
        )
        safe_indices = tl.where(valid, raw_indices, 0)
        keys = tl.load(
            k_ptr
            + batch_index * k_stride_b
            + safe_indices[:, None] * k_stride_s
            + head_index * k_stride_h
            + dim_offsets[None, :],
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        tl.store(
            logits_ptr
            + batch_index * logits_stride_b
            + query_index * logits_stride_s
            + head_index * logits_stride_h
            + token_slots,
            scores,
            mask=in_range,
        )


def selected_teacher_qk_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    selected_block_indices: torch.Tensor,
    *,
    block_size_tokens: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Return selected raw-token QK logits in ``[B,Q,H,TopK*block]``.

    ``q`` may be a non-contiguous query slice.  ``k`` is the complete raw Full
    key sequence.  Invalid ``-1`` routes produce ``-inf`` logits.
    """
    if not q.is_cuda or not k.is_cuda or not selected_block_indices.is_cuda:
        raise RuntimeError("selected teacher Triton kernel requires CUDA tensors")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("selected teacher q/k must be [batch, sequence, heads, dim]")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("selected teacher q/k batch, heads, and dim must match")
    if tuple(selected_block_indices.shape[:2]) != tuple(q.shape[:2]):
        raise ValueError("selected teacher routes must match batch/query dimensions")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        raise ValueError("selected teacher Triton kernel requires matching fp16/bf16 q/k")
    if q.shape[-1] not in (64, 128):
        raise ValueError("selected teacher Triton kernel supports head_dim 64 or 128")
    if block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be positive")

    route = selected_block_indices.to(torch.int32).contiguous()
    batch, query_count, heads, head_dim = q.shape
    topk_blocks = route.shape[-1]
    token_count = topk_blocks * block_size_tokens
    logits = torch.empty(
        (batch, query_count, heads, token_count),
        dtype=torch.float32,
        device=q.device,
    )
    if query_count == 0 or token_count == 0:
        return logits

    block_n = 128
    block_d = triton.next_power_of_2(head_dim)
    scale = float(softmax_scale) if softmax_scale is not None else head_dim ** -0.5
    _selected_teacher_qk_kernel[(batch * query_count * heads,)](
        q,
        k,
        route,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        route.stride(0),
        route.stride(1),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        k.shape[1],
        query_count=query_count,
        head_count=heads,
        head_dim=head_dim,
        topk_blocks=topk_blocks,
        block_size_tokens=block_size_tokens,
        softmax_scale=scale,
        block_n=block_n,
        block_d=block_d,
        num_warps=4,
        num_stages=2,
    )
    return logits


__all__ = ["selected_teacher_qk_logits"]
