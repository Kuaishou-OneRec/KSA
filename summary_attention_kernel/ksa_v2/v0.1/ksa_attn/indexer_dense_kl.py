"""Streaming dense-warmup KL for the raw-token Lightning Indexer.

The Full-attention teacher kernel emits unnormalised per-token probability
mass.  The former PyTorch path then normalised that ``[Q,K]`` tensor, built a
prefix mask, ran log_softmax, and materialised several more full-size FP32
tensors.  These kernels reduce each causal-prefix row directly and retain
only its student LSE and inverse teacher mass for backward.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_indexer_kl_forward_kernel(
    score_ptr,
    teacher_ptr,
    eligible_ptr,
    row_lse_ptr,
    row_inv_teacher_ptr,
    row_loss_ptr,
    score_stride_b,
    score_stride_q,
    teacher_stride_b,
    teacher_stride_q,
    query_count,
    candidate_count,
    block_n: tl.constexpr,
):
    query_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    eligible = tl.minimum(
        tl.load(eligible_ptr + query_index), candidate_count
    )
    running_max = tl.full((), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((), dtype=tl.float32)
    teacher_sum = tl.zeros((), dtype=tl.float32)
    teacher_log_sum = tl.zeros((), dtype=tl.float32)
    teacher_score_sum = tl.zeros((), dtype=tl.float32)

    for candidate_start in range(0, candidate_count, block_n):
        offsets = candidate_start + tl.arange(0, block_n)
        valid = offsets < eligible
        loaded_scores = tl.load(
            score_ptr
            + batch_index * score_stride_b
            + query_index * score_stride_q
            + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scores = tl.where(valid, loaded_scores, -float("inf"))
        teacher = tl.load(
            teacher_ptr
            + batch_index * teacher_stride_b
            + query_index * teacher_stride_q
            + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)

        tile_has_candidate = candidate_start < eligible
        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        new_sum = (
            running_sum * tl.exp(running_max - new_max)
            + tl.sum(tl.exp(scores - new_max), axis=0)
        )
        running_sum = tl.where(
            tile_has_candidate, new_sum, running_sum
        )
        running_max = tl.where(
            tile_has_candidate, new_max, running_max
        )
        teacher_sum += tl.sum(teacher, axis=0)
        teacher_log_sum += tl.sum(
            tl.where(
                teacher > 0.0,
                teacher * tl.log(tl.maximum(teacher, 1.0e-30)),
                0.0,
            ),
            axis=0,
        )
        teacher_score_sum += tl.sum(teacher * loaded_scores, axis=0)

    has_teacher = teacher_sum > 0.0
    has_candidate = eligible > 0
    row_lse = tl.where(
        has_candidate,
        tl.log(running_sum) + running_max,
        0.0,
    )
    inverse_teacher = tl.where(has_teacher, 1.0 / teacher_sum, 0.0)
    loss = tl.where(
        has_teacher,
        teacher_log_sum * inverse_teacher
        - tl.log(teacher_sum)
        - teacher_score_sum * inverse_teacher
        + row_lse,
        0.0,
    )
    row_offset = batch_index * query_count + query_index
    tl.store(row_lse_ptr + row_offset, row_lse)
    tl.store(row_inv_teacher_ptr + row_offset, inverse_teacher)
    tl.store(row_loss_ptr + row_offset, loss)


@triton.jit
def _dense_indexer_kl_backward_kernel(
    score_ptr,
    teacher_ptr,
    eligible_ptr,
    row_lse_ptr,
    row_inv_teacher_ptr,
    grad_output_ptr,
    grad_score_ptr,
    score_stride_b,
    score_stride_q,
    teacher_stride_b,
    teacher_stride_q,
    grad_stride_b,
    grad_stride_q,
    query_count,
    candidate_count,
    block_n: tl.constexpr,
):
    candidate_block = tl.program_id(0)
    query_index = tl.program_id(1)
    batch_index = tl.program_id(2)
    offsets = candidate_block * block_n + tl.arange(0, block_n)
    eligible = tl.minimum(
        tl.load(eligible_ptr + query_index), candidate_count
    )
    valid = offsets < eligible
    row_offset = batch_index * query_count + query_index
    row_lse = tl.load(row_lse_ptr + row_offset)
    inverse_teacher = tl.load(row_inv_teacher_ptr + row_offset)
    scores = tl.load(
        score_ptr
        + batch_index * score_stride_b
        + query_index * score_stride_q
        + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    teacher = tl.load(
        teacher_ptr
        + batch_index * teacher_stride_b
        + query_index * teacher_stride_q
        + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    upstream = tl.load(grad_output_ptr).to(tl.float32)
    gradient = (
        tl.exp(scores - row_lse) - teacher * inverse_teacher
    ) * upstream
    tl.store(
        grad_score_ptr
        + batch_index * grad_stride_b
        + query_index * grad_stride_q
        + offsets,
        gradient,
        mask=valid & (inverse_teacher > 0.0),
    )
    # Invalid slots are consumed by neither scorer nor downstream code, but
    # returning an explicitly zero dense gradient keeps the autograd contract
    # robust for arbitrary callers.
    tl.store(
        grad_score_ptr
        + batch_index * grad_stride_b
        + query_index * grad_stride_q
        + offsets,
        0.0,
        mask=(offsets < candidate_count)
        & (~valid | (inverse_teacher <= 0.0)),
    )


def _validate_inputs(
    scores: torch.Tensor,
    teacher_mass: torch.Tensor,
    eligible_count: torch.Tensor,
) -> None:
    if scores.ndim != 3 or tuple(teacher_mass.shape) != tuple(scores.shape):
        raise ValueError("dense Indexer KL expects matching [B,Q,K] tensors")
    if scores.dtype != torch.float32 or teacher_mass.dtype != torch.float32:
        raise ValueError("dense Indexer KL scores/teacher must be FP32")
    if not (scores.is_cuda and teacher_mass.is_cuda and eligible_count.is_cuda):
        raise RuntimeError("dense Indexer KL requires CUDA tensors")
    if not (scores.device == teacher_mass.device == eligible_count.device):
        raise ValueError("dense Indexer KL tensors must share one device")
    if eligible_count.ndim != 1 or eligible_count.shape[0] != scores.shape[1]:
        raise ValueError("dense Indexer KL eligible counts must be per query")
    if eligible_count.dtype not in (torch.int32, torch.int64):
        raise ValueError("dense Indexer KL eligible counts must be integers")
    if scores.stride(-1) != 1 or teacher_mass.stride(-1) != 1:
        raise ValueError("dense Indexer KL candidate dimensions must be contiguous")


class _FusedDenseIndexerKL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        teacher_mass: torch.Tensor,
        eligible_count: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(scores, teacher_mass, eligible_count)
        batch, query_count, candidate_count = scores.shape
        row_lse = torch.empty(
            (batch, query_count), dtype=torch.float32, device=scores.device
        )
        row_inv_teacher = torch.empty_like(row_lse)
        row_loss = torch.empty_like(row_lse)
        if query_count and candidate_count:
            _dense_indexer_kl_forward_kernel[(query_count, batch)](
                scores,
                teacher_mass,
                eligible_count,
                row_lse,
                row_inv_teacher,
                row_loss,
                scores.stride(0),
                scores.stride(1),
                teacher_mass.stride(0),
                teacher_mass.stride(1),
                query_count,
                candidate_count,
                block_n=256,
                num_warps=8,
                num_stages=2,
            )
        else:
            row_lse.zero_()
            row_inv_teacher.zero_()
            row_loss.zero_()
        ctx.save_for_backward(
            scores,
            teacher_mass,
            eligible_count,
            row_lse,
            row_inv_teacher,
        )
        return row_loss.sum()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        scores, teacher, eligible, row_lse, row_inv_teacher = (
            ctx.saved_tensors
        )
        batch, query_count, candidate_count = scores.shape
        grad_scores = torch.zeros_like(scores)
        if query_count and candidate_count:
            block_n = 256
            _dense_indexer_kl_backward_kernel[
                (triton.cdiv(candidate_count, block_n), query_count, batch)
            ](
                scores,
                teacher,
                eligible,
                row_lse,
                row_inv_teacher,
                grad_output,
                grad_scores,
                scores.stride(0),
                scores.stride(1),
                teacher.stride(0),
                teacher.stride(1),
                grad_scores.stride(0),
                grad_scores.stride(1),
                query_count,
                candidate_count,
                block_n=block_n,
                num_warps=8,
                num_stages=2,
            )
        return grad_scores, None, None


def fused_dense_indexer_kl(
    scores: torch.Tensor,
    teacher_mass: torch.Tensor,
    eligible_count: torch.Tensor,
) -> torch.Tensor:
    """Summed exact ``KL(normalize(teacher) || softmax(scores))``."""
    return _FusedDenseIndexerKL.apply(scores, teacher_mass, eligible_count)


__all__ = ["fused_dense_indexer_kl"]
