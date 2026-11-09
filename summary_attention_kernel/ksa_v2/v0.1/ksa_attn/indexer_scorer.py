"""Fused no-grad candidate scorer for the S-KSA Lightning indexer.

The hard routing path only consumes the final ``[B,Q,M]`` score matrix, but
the ordinary expression first materializes ``[B,Q,H,M]`` in the model dtype
and then another FP32 activation of the same shape.  This Triton kernel keeps
one output tile in registers, evaluates every indexer head with tensor-core
dot products, and writes only the final FP32 score matrix.

The hard-route API remains inference-only.  Dense warm-up additionally uses
an autograd wrapper around the same forward kernel.  Its backward streams one
Indexer head at a time, recomputing the low-precision dot product before the
ReLU derivative.  This preserves the PyTorch expression's dtype boundaries
without retaining ``[B,Q,H,M]`` dot/ReLU tensors.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_indexer_candidate_scorer_kernel(
    query_ptr,
    key_ptr,
    head_weight_ptr,
    output_ptr,
    query_stride_b,
    query_stride_q,
    query_stride_h,
    key_stride_b,
    key_stride_m,
    weight_stride_b,
    weight_stride_q,
    output_stride_b,
    output_stride_q,
    query_count,
    candidate_count,
    score_scale,
    head_count: tl.constexpr,
    head_dim: tl.constexpr,
    input_is_bf16: tl.constexpr,
    block_q: tl.constexpr,
    block_m: tl.constexpr,
):
    query_block = tl.program_id(0)
    candidate_block = tl.program_id(1)
    batch_index = tl.program_id(2)
    query_offsets = query_block * block_q + tl.arange(0, block_q)
    candidate_offsets = candidate_block * block_m + tl.arange(0, block_m)
    dim_offsets = tl.arange(0, head_dim)
    query_valid = query_offsets < query_count
    candidate_valid = candidate_offsets < candidate_count

    # Summary/index keys are shared by all indexer heads.
    key = tl.load(
        key_ptr
        + batch_index * key_stride_b
        + candidate_offsets[:, None] * key_stride_m
        + dim_offsets[None, :],
        mask=candidate_valid[:, None],
        other=0.0,
    )
    score = tl.zeros((block_q, block_m), dtype=tl.float32)
    for head_index in tl.static_range(0, head_count):
        query = tl.load(
            query_ptr
            + batch_index * query_stride_b
            + query_offsets[:, None] * query_stride_q
            + head_index * query_stride_h
            + dim_offsets[None, :],
            mask=query_valid[:, None],
            other=0.0,
        )
        dot = tl.dot(query, tl.trans(key))
        # torch.einsum returns the model dtype before `.float()`.  Reproduce
        # that explicit storage round-trip; accumulation order can still
        # differ slightly from cuBLAS and is tested with a tight tolerance.
        if input_is_bf16:
            dot = dot.to(tl.bfloat16).to(tl.float32)
        else:
            dot = dot.to(tl.float16).to(tl.float32)
        head_weight = tl.load(
            head_weight_ptr
            + batch_index * weight_stride_b
            + query_offsets * weight_stride_q
            + head_index,
            mask=query_valid,
            other=0.0,
        ).to(tl.float32)
        score += tl.maximum(dot * score_scale, 0.0) * head_weight[:, None]

    tl.store(
        output_ptr
        + batch_index * output_stride_b
        + query_offsets[:, None] * output_stride_q
        + candidate_offsets[None, :],
        score,
        mask=query_valid[:, None] & candidate_valid[None, :],
    )


def fused_indexer_candidate_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weight: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    """Return hard-route scores in ``[B,Q,M]`` without a head intermediate.

    Expected layouts are ``query[B,Q,H,D]``, ``key[B,M,D]`` and
    ``head_weight[B,Q,H]``.  Candidate and query extents remain runtime
    values, so causal-prefix changes do not create one Triton compilation per
    indexer tile.

    The caller must retain the existing PyTorch implementation as fallback
    whenever gradients are enabled or the tensors are not CUDA fp16/bf16.
    """
    if torch.is_grad_enabled():
        raise RuntimeError(
            "fused indexer candidate scoring is only valid inside torch.no_grad()"
        )
    _validate_indexer_inputs(query, key, head_weight)
    return _launch_fused_indexer_candidate_scores(
        query, key, head_weight, float(score_scale)
    )


class _FusedDenseIndexerScores(torch.autograd.Function):
    """Differentiable dense scores with bounded saved activations.

    Only ``query``, ``key`` and ``head_weight`` are retained.  In backward we
    recompute one head's dot matrix at a time, immediately contract its query,
    key and weight gradients, and release it before moving to the next head.
    The public result and all gradient formulas are identical to::

        dot = einsum("bqhd,bmd->bqhm", query, key)
        relu(dot.float() * scale).mul(weight[..., None]).sum(dim=2)
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        head_weight: torch.Tensor,
        score_scale: float,
    ) -> torch.Tensor:
        _validate_indexer_inputs(query, key, head_weight)
        ctx.save_for_backward(query, key, head_weight)
        ctx.score_scale = float(score_scale)
        return _launch_fused_indexer_candidate_scores(
            query, key, head_weight, ctx.score_scale
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_scores: torch.Tensor):
        query, key, head_weight = ctx.saved_tensors
        need_query, need_key, need_weight = ctx.needs_input_grad[:3]
        grad_query = torch.empty_like(query) if need_query else None
        grad_key_fp32 = (
            torch.zeros(key.shape, dtype=torch.float32, device=key.device)
            if need_key
            else None
        )
        grad_weight = (
            torch.empty_like(head_weight) if need_weight else None
        )
        grad_scores = grad_scores.contiguous()
        scale = ctx.score_scale

        # Reproduce the reference's BF16/FP16 dot storage boundary and its
        # cast-back boundary on grad_dot.  Keeping the head loop in Python is
        # intentional: at 128K it caps the largest live activation at one
        # head instead of four while each contraction remains a large GEMM.
        for head_index in range(query.shape[2]):
            query_head = query[:, :, head_index, :]
            dot = torch.bmm(query_head, key.transpose(1, 2))
            activated = dot.float()
            activated.mul_(scale).relu_()

            if need_weight:
                grad_weight[:, :, head_index].copy_(
                    (activated * grad_scores).sum(dim=-1)
                )

            if need_query or need_key:
                grad_dot = grad_scores * head_weight[
                    :, :, head_index, None
                ]
                grad_dot.mul_(scale)
                grad_dot.masked_fill_(activated <= 0.0, 0.0)
                grad_dot_low = grad_dot.to(query.dtype)
                if need_query:
                    grad_query[:, :, head_index, :].copy_(
                        torch.bmm(grad_dot_low, key)
                    )
                if need_key:
                    # Each individual head contraction has model dtype in
                    # the reference graph.  Accumulate those results in FP32
                    # before the final cast, matching the surrounding
                    # precomputed-gradient path's prefix accumulation.
                    grad_key_fp32.add_(
                        torch.bmm(
                            grad_dot_low.transpose(1, 2), query_head
                        ).float()
                    )
                del dot, activated, grad_dot, grad_dot_low

        grad_key = (
            grad_key_fp32.to(key.dtype) if need_key else None
        )
        return grad_query, grad_key, grad_weight, None


def _validate_indexer_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weight: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 3 or head_weight.ndim != 3:
        raise ValueError(
            "indexer scorer expects query[B,Q,H,D], key[B,M,D], weight[B,Q,H]"
        )
    batch, query_count, head_count, head_dim = query.shape
    if key.shape[0] != batch or key.shape[-1] != head_dim:
        raise ValueError("indexer scorer query/key batch and head_dim must match")
    if tuple(head_weight.shape) != (batch, query_count, head_count):
        raise ValueError("indexer scorer head weights must match query B/Q/H")
    if not (query.is_cuda and key.is_cuda and head_weight.is_cuda):
        raise RuntimeError("fused indexer candidate scoring requires CUDA tensors")
    if not (query.device == key.device == head_weight.device):
        raise ValueError("indexer scorer tensors must share one CUDA device")
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype:
        raise ValueError("fused indexer scorer requires matching CUDA fp16/bf16 query/key")
    if head_weight.dtype != torch.float32:
        raise ValueError("fused indexer scorer requires FP32 head weights")
    if head_dim not in (64, 128):
        raise ValueError("fused indexer scorer supports head_dim 64 or 128")
    if not 1 <= head_count <= 8:
        raise ValueError("fused indexer scorer supports 1 to 8 indexer heads")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("indexer scorer query/key head dimensions must be contiguous")
    if head_weight.stride(-1) != 1:
        raise ValueError("indexer scorer head weights must be contiguous over heads")


def _launch_fused_indexer_candidate_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weight: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    batch, query_count, head_count, head_dim = query.shape
    candidate_count = key.shape[1]
    output = torch.empty(
        (batch, query_count, candidate_count),
        dtype=torch.float32,
        device=query.device,
    )
    if query_count == 0 or candidate_count == 0:
        return output

    block_q = 64
    block_m = 64
    grid = (
        triton.cdiv(query_count, block_q),
        triton.cdiv(candidate_count, block_m),
        batch,
    )
    _fused_indexer_candidate_scorer_kernel[grid](
        query,
        key,
        head_weight,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        head_weight.stride(0),
        head_weight.stride(1),
        output.stride(0),
        output.stride(1),
        query_count,
        candidate_count,
        float(score_scale),
        head_count=head_count,
        head_dim=head_dim,
        input_is_bf16=query.dtype == torch.bfloat16,
        block_q=block_q,
        block_m=block_m,
        num_warps=8,
        num_stages=3,
    )
    return output


def fused_dense_indexer_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weight: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    """Return differentiable dense scores without head-sized activations."""
    _validate_indexer_inputs(query, key, head_weight)
    return _FusedDenseIndexerScores.apply(
        query, key, head_weight, float(score_scale)
    )


__all__ = [
    "fused_dense_indexer_scores",
    "fused_indexer_candidate_scores",
]
