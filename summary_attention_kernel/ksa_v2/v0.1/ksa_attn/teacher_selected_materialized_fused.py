"""Experimental fused postprocess for the materialized selected teacher.

Unlike the two-pass streaming candidate, this path reads selected raw K only
once and retains the established FP32 QK-logit tensor.  Triton then fuses the
remaining per-head softmax, head average, raw8 max/sum pooling and block L1
normalization without creating a token-probability tensor.

An optional route-locality experiment sorts block IDs before QK and scatters
the final target back to the original TopK slot order.  Sorting only permutes
the common per-head softmax domain, so this is mathematically equivalent;
the inverse scatter is required because the student KL consumes the original
TopK ordering.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .teacher_selected import selected_teacher_qk_logits


@triton.jit
def _materialized_selected_lse_kernel(
    logits_ptr,
    lse_ptr,
    logits_stride_b,
    logits_stride_s,
    logits_stride_h,
    lse_stride_b,
    lse_stride_s,
    query_count,
    head_count: tl.constexpr,
    token_count: tl.constexpr,
    block_t: tl.constexpr,
):
    program = tl.program_id(0)
    head_index = program % head_count
    batch_query = program // head_count
    query_index = batch_query % query_count
    batch_index = batch_query // query_count
    offsets = tl.arange(0, block_t)
    in_range = offsets < token_count
    score = tl.load(
        logits_ptr
        + batch_index * logits_stride_b
        + query_index * logits_stride_s
        + head_index * logits_stride_h
        + offsets,
        mask=in_range,
        other=-float("inf"),
    ).to(tl.float32)
    finite = in_range & (score != -float("inf"))
    maximum = tl.max(score, axis=0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    denominator = tl.sum(
        tl.where(finite, tl.exp(score - safe_maximum), 0.0), axis=0
    )
    lse = tl.where(
        denominator > 0.0,
        tl.log(denominator) + maximum,
        -float("inf"),
    )
    tl.store(
        lse_ptr
        + batch_index * lse_stride_b
        + query_index * lse_stride_s
        + head_index,
        lse,
    )


@triton.jit
def _materialized_selected_block_score_kernel(
    logits_ptr,
    route_ptr,
    lse_ptr,
    target_ptr,
    logits_stride_b,
    logits_stride_s,
    logits_stride_h,
    route_stride_b,
    route_stride_s,
    lse_stride_b,
    lse_stride_s,
    target_stride_b,
    target_stride_s,
    query_count,
    head_count: tl.constexpr,
    topk_blocks: tl.constexpr,
    block_size_tokens: tl.constexpr,
    use_max_pool: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    batch_query = tl.program_id(0)
    query_index = batch_query % query_count
    batch_index = batch_query // query_count
    block_start = tl.program_id(1) * block_k
    token_lanes = tl.arange(0, block_n)
    block_lanes = token_lanes // block_size_tokens
    block_slots = block_start + block_lanes
    token_in_range = block_slots < topk_blocks
    route_ids = tl.load(
        route_ptr
        + batch_index * route_stride_b
        + query_index * route_stride_s
        + block_slots,
        mask=token_in_range,
        other=-1,
    )
    token_valid = token_in_range & (route_ids >= 0)
    token_slots = block_slots * block_size_tokens + (
        token_lanes % block_size_tokens
    )
    token_probability = tl.zeros((block_n,), dtype=tl.float32)

    for head_index in tl.static_range(0, head_count):
        score = tl.load(
            logits_ptr
            + batch_index * logits_stride_b
            + query_index * logits_stride_s
            + head_index * logits_stride_h
            + token_slots,
            mask=token_in_range,
            other=-float("inf"),
        ).to(tl.float32)
        head_lse = tl.load(
            lse_ptr
            + batch_index * lse_stride_b
            + query_index * lse_stride_s
            + head_index
        )
        finite_lse = head_lse != -float("inf")
        safe_lse = tl.where(finite_lse, head_lse, 0.0)
        probability = tl.exp(score - safe_lse)
        token_probability += tl.where(
            token_valid & finite_lse, probability, 0.0
        )

    token_probability *= 1.0 / head_count
    grouped = tl.reshape(
        token_probability, (block_k, block_size_tokens)
    )
    if use_max_pool:
        block_score = tl.max(grouped, axis=1)
    else:
        block_score = tl.sum(grouped, axis=1)
    output_blocks = block_start + tl.arange(0, block_k)
    output_in_range = output_blocks < topk_blocks
    output_route = tl.load(
        route_ptr
        + batch_index * route_stride_b
        + query_index * route_stride_s
        + output_blocks,
        mask=output_in_range,
        other=-1,
    )
    block_score = tl.where(
        output_in_range & (output_route >= 0), block_score, 0.0
    )
    tl.store(
        target_ptr
        + batch_index * target_stride_b
        + query_index * target_stride_s
        + output_blocks,
        block_score,
        mask=output_in_range,
    )


@triton.jit
def _normalize_materialized_selected_kernel(
    target_ptr,
    target_stride_b,
    target_stride_s,
    query_count,
    topk_blocks: tl.constexpr,
    block_k: tl.constexpr,
):
    batch_query = tl.program_id(0)
    query_index = batch_query % query_count
    batch_index = batch_query // query_count
    offsets = tl.arange(0, block_k)
    valid = offsets < topk_blocks
    pointer = (
        target_ptr
        + batch_index * target_stride_b
        + query_index * target_stride_s
        + offsets
    )
    score = tl.load(pointer, mask=valid, other=0.0).to(tl.float32)
    denominator = tl.sum(score, axis=0)
    normalized = tl.where(denominator > 0.0, score / denominator, 0.0)
    tl.store(pointer, normalized, mask=valid)


def _sort_routes_for_locality(route: torch.Tensor):
    """Sort valid IDs ascending, place -1 last, return original-slot indices."""
    sentinel = torch.iinfo(torch.int32).max
    sort_key = torch.where(route >= 0, route, sentinel)
    _, sorted_to_original = torch.sort(sort_key, dim=-1, stable=True)
    sorted_route = torch.gather(route, -1, sorted_to_original)
    return sorted_route.contiguous(), sorted_to_original


def _validate_inputs(q, k, route, block_size_tokens, pooling):
    if not q.is_cuda or not k.is_cuda or not route.is_cuda:
        raise RuntimeError("materialized fused selected teacher requires CUDA")
    if q.ndim != 4 or k.ndim != 4 or route.ndim != 3:
        raise ValueError("q/k/routes must be [B,Q,H,D], [B,S,H,D], [B,Q,K]")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("selected teacher q/k batch, heads, and dim must match")
    if tuple(route.shape[:2]) != tuple(q.shape[:2]):
        raise ValueError("selected teacher routes must match batch/query dimensions")
    if route.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected teacher routes must use int32 or int64")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        raise ValueError("selected teacher requires matching fp16/bf16 q/k")
    if q.shape[-1] not in (64, 128):
        raise ValueError("selected teacher supports head_dim 64 or 128")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("selected teacher head dimensions must be contiguous")
    if not (q.device == k.device == route.device):
        raise ValueError("selected teacher q/k/routes must share a CUDA device")
    if block_size_tokens not in (1, 2, 4, 8):
        raise ValueError("selected teacher route span must be 1, 2, 4, or 8 tokens")
    if pooling not in ("max", "sum"):
        raise ValueError(f"unsupported S-KSA teacher pooling: {pooling}")


@torch.no_grad()
def selected_teacher_block_distribution_materialized_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    selected_block_indices: torch.Tensor,
    *,
    block_size_tokens: int,
    pooling: str,
    softmax_scale: float | None = None,
    sort_routes: bool = False,
) -> torch.Tensor:
    """Return the established selected target with a fused postprocess.

    ``sort_routes=True`` is an isolated locality experiment.  Results are
    always scattered back to the caller's original TopK slot order.
    """
    _validate_inputs(q, k, selected_block_indices, block_size_tokens, pooling)
    route = selected_block_indices.to(torch.int32).contiguous()
    batch, query_count, heads, head_dim = q.shape
    topk_blocks = route.shape[-1]
    token_count = topk_blocks * block_size_tokens
    if token_count > 4096:
        raise ValueError(
            "materialized fused selected teacher supports at most 4096 "
            "selected raw-token entries"
        )
    target = torch.empty(
        (batch, query_count, topk_blocks), dtype=torch.float32, device=q.device
    )
    if batch == 0 or query_count == 0 or topk_blocks == 0:
        return target

    sorted_to_original = None
    compute_route = route
    if sort_routes:
        compute_route, sorted_to_original = _sort_routes_for_locality(route)
    logits = selected_teacher_qk_logits(
        q,
        k,
        compute_route,
        block_size_tokens=block_size_tokens,
        softmax_scale=softmax_scale,
    )
    lse = torch.empty(
        (batch, query_count, heads), dtype=torch.float32, device=q.device
    )
    lse_block = triton.next_power_of_2(token_count)
    _materialized_selected_lse_kernel[(batch * query_count * heads,)](
        logits,
        lse,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        lse.stride(0),
        lse.stride(1),
        query_count,
        head_count=heads,
        token_count=token_count,
        block_t=lse_block,
        num_warps=min(8, max(1, lse_block // 256)),
    )
    # Raw-token routing has no within-block reduction. A wider block cuts
    # grid overhead for TopK=2048; raw8 keeps its register-bounded schedule.
    block_k = 64 if block_size_tokens == 1 else 8
    _materialized_selected_block_score_kernel[
        (batch * query_count, triton.cdiv(topk_blocks, block_k))
    ](
        logits,
        compute_route,
        lse,
        target,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        compute_route.stride(0),
        compute_route.stride(1),
        lse.stride(0),
        lse.stride(1),
        target.stride(0),
        target.stride(1),
        query_count,
        head_count=heads,
        topk_blocks=topk_blocks,
        block_size_tokens=block_size_tokens,
        use_max_pool=pooling == "max",
        block_k=block_k,
        block_n=block_k * block_size_tokens,
        num_warps=4,
        num_stages=2,
    )
    normalize_block = triton.next_power_of_2(topk_blocks)
    _normalize_materialized_selected_kernel[(batch * query_count,)](
        target,
        target.stride(0),
        target.stride(1),
        query_count,
        topk_blocks=topk_blocks,
        block_k=normalize_block,
        num_warps=min(8, max(1, normalize_block // 32)),
    )
    if sorted_to_original is not None:
        original_order = torch.empty_like(target)
        original_order.scatter_(-1, sorted_to_original, target)
        return original_order
    return target


__all__ = ["selected_teacher_block_distribution_materialized_fused"]
