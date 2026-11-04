"""Differentiable multi-head scorer over selected S-KSA routes.

Read keys through route indices without materializing a ``[B, Q, K, D]``
selected-key tensor. Forward preserves model-dtype dot-product rounding before
FP32 activation and head reduction. Backward recomputes selected dot products
for the query, key and head-weight gradients.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_indexer_score_forward_kernel(
    query_ptr,
    key_ptr,
    weight_ptr,
    route_ptr,
    output_ptr,
    query_stride_b,
    query_stride_q,
    query_stride_h,
    key_stride_b,
    key_stride_m,
    weight_stride_b,
    weight_stride_q,
    route_stride_b,
    route_stride_q,
    output_stride_b,
    output_stride_q,
    query_count,
    candidate_count,
    selected_count: tl.constexpr,
    score_scale,
    head_count: tl.constexpr,
    head_dim: tl.constexpr,
    input_is_bf16: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    selected_block = tl.program_id(0)
    query_index = tl.program_id(1)
    batch_index = tl.program_id(2)
    selected_offsets = selected_block * block_k + tl.arange(0, block_k)
    head_offsets = tl.arange(0, block_h)
    dim_offsets = tl.arange(0, head_dim)
    selected_valid = selected_offsets < selected_count
    head_valid = head_offsets < head_count

    route = tl.load(
        route_ptr
        + batch_index * route_stride_b
        + query_index * route_stride_q
        + selected_offsets,
        mask=selected_valid,
        other=-1,
    )
    # Match ``selected.clamp_min(0)`` in the PyTorch oracle.  Positive route
    # IDs are produced internally by TopK and are therefore trusted; the
    # bounds mask is defensive and prevents an invalid memory access.
    safe_route = tl.maximum(route, 0)
    route_in_bounds = safe_route < candidate_count
    selected_key = tl.load(
        key_ptr
        + batch_index * key_stride_b
        + safe_route[:, None] * key_stride_m
        + dim_offsets[None, :],
        mask=(selected_valid & route_in_bounds)[:, None],
        other=0.0,
    )
    query = tl.load(
        query_ptr
        + batch_index * query_stride_b
        + query_index * query_stride_q
        + head_offsets[:, None] * query_stride_h
        + dim_offsets[None, :],
        mask=head_valid[:, None],
        other=0.0,
    )
    dot = tl.dot(query, tl.trans(selected_key))
    # ``torch.einsum`` writes a model-dtype result before ``.float()``.
    # Preserve that quantization boundary; the reduction order can still
    # differ slightly and is covered by the oracle tolerance tests.
    if input_is_bf16:
        dot = dot.to(tl.bfloat16).to(tl.float32)
    else:
        dot = dot.to(tl.float16).to(tl.float32)
    activated = tl.maximum(dot * score_scale, 0.0)
    weight = tl.load(
        weight_ptr
        + batch_index * weight_stride_b
        + query_index * weight_stride_q
        + head_offsets,
        mask=head_valid,
        other=0.0,
    ).to(tl.float32)
    score = tl.sum(activated * weight[:, None], axis=0)
    tl.store(
        output_ptr
        + batch_index * output_stride_b
        + query_index * output_stride_q
        + selected_offsets,
        score,
        mask=selected_valid,
    )


@triton.jit
def _selected_indexer_score_query_weight_backward_kernel(
    query_ptr,
    key_ptr,
    weight_ptr,
    route_ptr,
    grad_score_ptr,
    grad_query_ptr,
    grad_weight_ptr,
    grad_dot_ptr,
    query_stride_b,
    query_stride_q,
    query_stride_h,
    key_stride_b,
    key_stride_m,
    weight_stride_b,
    weight_stride_q,
    route_stride_b,
    route_stride_q,
    grad_score_stride_b,
    grad_score_stride_q,
    grad_query_stride_b,
    grad_query_stride_q,
    grad_query_stride_h,
    grad_weight_stride_b,
    grad_weight_stride_q,
    grad_dot_stride_b,
    grad_dot_stride_q,
    grad_dot_stride_h,
    candidate_count,
    selected_count: tl.constexpr,
    score_scale,
    head_count: tl.constexpr,
    head_dim: tl.constexpr,
    input_is_bf16: tl.constexpr,
    need_query: tl.constexpr,
    need_weight: tl.constexpr,
    need_key: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    query_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    head_offsets = tl.arange(0, block_h)
    dim_offsets = tl.arange(0, head_dim)
    head_valid = head_offsets < head_count
    query = tl.load(
        query_ptr
        + batch_index * query_stride_b
        + query_index * query_stride_q
        + head_offsets[:, None] * query_stride_h
        + dim_offsets[None, :],
        mask=head_valid[:, None],
        other=0.0,
    )
    weight = tl.load(
        weight_ptr
        + batch_index * weight_stride_b
        + query_index * weight_stride_q
        + head_offsets,
        mask=head_valid,
        other=0.0,
    ).to(tl.float32)
    grad_query = tl.zeros((block_h, head_dim), dtype=tl.float32)
    grad_weight = tl.zeros((block_h,), dtype=tl.float32)

    for selected_start in tl.static_range(0, selected_count, block_k):
        selected_offsets = selected_start + tl.arange(0, block_k)
        selected_valid = selected_offsets < selected_count
        route = tl.load(
            route_ptr
            + batch_index * route_stride_b
            + query_index * route_stride_q
            + selected_offsets,
            mask=selected_valid,
            other=-1,
        )
        safe_route = tl.maximum(route, 0)
        route_in_bounds = safe_route < candidate_count
        selected_key = tl.load(
            key_ptr
            + batch_index * key_stride_b
            + safe_route[:, None] * key_stride_m
            + dim_offsets[None, :],
            mask=(selected_valid & route_in_bounds)[:, None],
            other=0.0,
        )
        dot = tl.dot(query, tl.trans(selected_key))
        if input_is_bf16:
            dot = dot.to(tl.bfloat16).to(tl.float32)
        else:
            dot = dot.to(tl.float16).to(tl.float32)
        scaled_dot = dot * score_scale
        activated = tl.maximum(scaled_dot, 0.0)
        grad_score = tl.load(
            grad_score_ptr
            + batch_index * grad_score_stride_b
            + query_index * grad_score_stride_q
            + selected_offsets,
            mask=selected_valid,
            other=0.0,
        ).to(tl.float32)
        grad_dot = (
            grad_score[None, :]
            * weight[:, None]
            * score_scale
            * (scaled_dot > 0.0)
        )
        if input_is_bf16:
            grad_dot_low = grad_dot.to(tl.bfloat16)
        else:
            grad_dot_low = grad_dot.to(tl.float16)
        if need_query:
            grad_query += tl.dot(grad_dot_low, selected_key)
        if need_weight:
            grad_weight += tl.sum(
                activated * grad_score[None, :], axis=1
            )
        if need_key:
            tl.store(
                grad_dot_ptr
                + batch_index * grad_dot_stride_b
                + query_index * grad_dot_stride_q
                + head_offsets[:, None] * grad_dot_stride_h
                + selected_offsets[None, :],
                grad_dot_low,
                mask=head_valid[:, None] & selected_valid[None, :],
            )

    if need_query:
        tl.store(
            grad_query_ptr
            + batch_index * grad_query_stride_b
            + query_index * grad_query_stride_q
            + head_offsets[:, None] * grad_query_stride_h
            + dim_offsets[None, :],
            grad_query,
            mask=head_valid[:, None],
        )
    if need_weight:
        tl.store(
            grad_weight_ptr
            + batch_index * grad_weight_stride_b
            + query_index * grad_weight_stride_q
            + head_offsets,
            grad_weight,
            mask=head_valid,
        )


@triton.jit
def _selected_indexer_score_key_backward_kernel(
    query_ptr,
    route_ptr,
    grad_dot_ptr,
    grad_key_ptr,
    query_stride_b,
    query_stride_q,
    query_stride_h,
    route_stride_b,
    route_stride_q,
    grad_dot_stride_b,
    grad_dot_stride_q,
    grad_dot_stride_h,
    grad_key_stride_b,
    grad_key_stride_m,
    candidate_count,
    selected_count,
    head_count: tl.constexpr,
    head_dim: tl.constexpr,
    input_is_bf16: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    selected_block = tl.program_id(0)
    query_index = tl.program_id(1)
    batch_index = tl.program_id(2)
    selected_offsets = selected_block * block_k + tl.arange(0, block_k)
    head_offsets = tl.arange(0, block_h)
    dim_offsets = tl.arange(0, head_dim)
    selected_valid = selected_offsets < selected_count
    head_valid = head_offsets < head_count
    route = tl.load(
        route_ptr
        + batch_index * route_stride_b
        + query_index * route_stride_q
        + selected_offsets,
        mask=selected_valid,
        other=-1,
    )
    safe_route = tl.maximum(route, 0)
    route_in_bounds = safe_route < candidate_count
    query = tl.load(
        query_ptr
        + batch_index * query_stride_b
        + query_index * query_stride_q
        + head_offsets[:, None] * query_stride_h
        + dim_offsets[None, :],
        mask=head_valid[:, None],
        other=0.0,
    )
    grad_dot = tl.load(
        grad_dot_ptr
        + batch_index * grad_dot_stride_b
        + query_index * grad_dot_stride_q
        + head_offsets[:, None] * grad_dot_stride_h
        + selected_offsets[None, :],
        mask=head_valid[:, None] & selected_valid[None, :],
        other=0.0,
    )
    contribution = tl.dot(tl.trans(grad_dot), query)
    # The oracle's selected-key bmm writes model dtype before its indexing
    # backward.  Preserve that boundary, then accumulate routes in FP32 to
    # avoid low-precision atomics under repeated keys (especially the sink).
    if input_is_bf16:
        contribution = contribution.to(tl.bfloat16).to(tl.float32)
    else:
        contribution = contribution.to(tl.float16).to(tl.float32)
    tl.atomic_add(
        grad_key_ptr
        + batch_index * grad_key_stride_b
        + safe_route[:, None] * grad_key_stride_m
        + dim_offsets[None, :],
        contribution,
        mask=(selected_valid & route_in_bounds)[:, None],
    )


def _validate_selected_indexer_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weight: torch.Tensor,
    selected: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 3 or head_weight.ndim != 3:
        raise ValueError(
            "selected indexer scorer expects query[B,Q,H,D], key[B,M,D], "
            "and weight[B,Q,H]"
        )
    batch, query_count, head_count, head_dim = query.shape
    if key.shape[0] != batch or key.shape[-1] != head_dim:
        raise ValueError("selected indexer query/key batch and head_dim must match")
    if tuple(head_weight.shape) != (batch, query_count, head_count):
        raise ValueError("selected indexer weights must match query B/Q/H")
    if selected.ndim != 3 or tuple(selected.shape[:2]) != (batch, query_count):
        raise ValueError("selected indexer routes must match query B/Q")
    if not (query.is_cuda and key.is_cuda and head_weight.is_cuda and selected.is_cuda):
        raise RuntimeError("selected indexer scorer requires CUDA tensors")
    if not (query.device == key.device == head_weight.device == selected.device):
        raise ValueError("selected indexer tensors must share one CUDA device")
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype:
        raise ValueError("selected indexer scorer requires matching fp16/bf16 query/key")
    if head_weight.dtype != torch.float32:
        raise ValueError("selected indexer scorer requires FP32 head weights")
    if selected.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected indexer routes must be int32 or int64")
    if head_dim not in (64, 128):
        raise ValueError("selected indexer scorer supports head_dim 64 or 128")
    if not 1 <= head_count <= 8:
        raise ValueError("selected indexer scorer supports 1 to 8 heads")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("selected indexer query/key head dimensions must be contiguous")
    if head_weight.stride(-1) != 1:
        raise ValueError("selected indexer weights must be contiguous over heads")
    if selected.shape[-1] and key.shape[1] == 0:
        raise ValueError("non-empty selected routes require at least one key")


def _launch_selected_indexer_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weight: torch.Tensor,
    selected: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    batch, query_count, head_count, head_dim = query.shape
    selected_count = selected.shape[-1]
    output = torch.empty(
        (batch, query_count, selected_count),
        dtype=torch.float32,
        device=query.device,
    )
    if query_count == 0 or selected_count == 0:
        return output
    block_h = 16
    block_k = 64
    grid = (triton.cdiv(selected_count, block_k), query_count, batch)
    _selected_indexer_score_forward_kernel[grid](
        query,
        key,
        head_weight,
        selected,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        head_weight.stride(0),
        head_weight.stride(1),
        selected.stride(0),
        selected.stride(1),
        output.stride(0),
        output.stride(1),
        query_count,
        key.shape[1],
        selected_count,
        float(score_scale),
        head_count=head_count,
        head_dim=head_dim,
        input_is_bf16=query.dtype == torch.bfloat16,
        block_h=block_h,
        block_k=block_k,
        num_warps=8,
        num_stages=3,
    )
    return output


class _FusedSelectedIndexerScores(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        head_weight: torch.Tensor,
        selected: torch.Tensor,
        score_scale: float,
    ) -> torch.Tensor:
        _validate_selected_indexer_inputs(query, key, head_weight, selected)
        route = selected.to(torch.int32).contiguous()
        ctx.save_for_backward(query, key, head_weight, route)
        ctx.score_scale = float(score_scale)
        return _launch_selected_indexer_forward(
            query, key, head_weight, route, ctx.score_scale
        )

    @staticmethod
    def backward(ctx, grad_scores: torch.Tensor):
        query, key, head_weight, selected = ctx.saved_tensors
        need_query, need_key, need_weight = ctx.needs_input_grad[:3]
        batch, query_count, head_count, head_dim = query.shape
        selected_count = selected.shape[-1]
        grad_query = torch.empty_like(query) if need_query else None
        grad_weight = torch.empty_like(head_weight) if need_weight else None
        grad_key = None

        if query_count == 0 or selected_count == 0:
            if need_query:
                grad_query.zero_()
            if need_key:
                grad_key = torch.zeros_like(key)
            if need_weight:
                grad_weight.zero_()
            return grad_query, grad_key, grad_weight, None, None

        grad_scores = grad_scores.contiguous()
        grad_dot = (
            torch.empty(
                (batch, query_count, head_count, selected_count),
                dtype=query.dtype,
                device=query.device,
            )
            if need_key
            else torch.empty(0, dtype=query.dtype, device=query.device)
        )
        block_h = 16
        block_k = 64
        _selected_indexer_score_query_weight_backward_kernel[
            (query_count, batch)
        ](
            query,
            key,
            head_weight,
            selected,
            grad_scores,
            grad_query if need_query else query,
            grad_weight if need_weight else head_weight,
            grad_dot,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            head_weight.stride(0),
            head_weight.stride(1),
            selected.stride(0),
            selected.stride(1),
            grad_scores.stride(0),
            grad_scores.stride(1),
            grad_query.stride(0) if need_query else 0,
            grad_query.stride(1) if need_query else 0,
            grad_query.stride(2) if need_query else 0,
            grad_weight.stride(0) if need_weight else 0,
            grad_weight.stride(1) if need_weight else 0,
            grad_dot.stride(0) if need_key else 0,
            grad_dot.stride(1) if need_key else 0,
            grad_dot.stride(2) if need_key else 0,
            key.shape[1],
            selected_count,
            ctx.score_scale,
            head_count=head_count,
            head_dim=head_dim,
            input_is_bf16=query.dtype == torch.bfloat16,
            need_query=need_query,
            need_weight=need_weight,
            need_key=need_key,
            block_h=block_h,
            block_k=block_k,
            num_warps=8,
            num_stages=3,
        )
        if need_key:
            grad_key_fp32 = torch.zeros(
                key.shape, dtype=torch.float32, device=key.device
            )
            _selected_indexer_score_key_backward_kernel[
                (triton.cdiv(selected_count, block_k), query_count, batch)
            ](
                query,
                selected,
                grad_dot,
                grad_key_fp32,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                selected.stride(0),
                selected.stride(1),
                grad_dot.stride(0),
                grad_dot.stride(1),
                grad_dot.stride(2),
                grad_key_fp32.stride(0),
                grad_key_fp32.stride(1),
                key.shape[1],
                selected_count,
                head_count=head_count,
                head_dim=head_dim,
                input_is_bf16=query.dtype == torch.bfloat16,
                block_h=block_h,
                block_k=block_k,
                num_warps=8,
                num_stages=2,
            )
            grad_key = grad_key_fp32.to(key.dtype)
        return grad_query, grad_key, grad_weight, None, None


def fused_selected_indexer_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weight: torch.Tensor,
    selected_block_indices: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    """Differentiably score only selected summary keys without gathering them.

    Routes follow the model's ``-1`` padding convention.  As in the PyTorch
    oracle, negative entries read key zero; the KL validity mask subsequently
    removes those score slots.  Positive route IDs are trusted internal TopK
    output and must be smaller than ``key.shape[1]``.
    """
    return _FusedSelectedIndexerScores.apply(
        query,
        key,
        head_weight,
        selected_block_indices,
        float(score_scale),
    )


__all__ = ["fused_selected_indexer_scores"]
