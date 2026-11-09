"""Summary-keyed Lightning Indexer for sparse Full attention.

Each inserted summary token builds one or more index keys for its preceding
raw-token chunk. Routing never exposes summary K/V to a Full-layer text query:
selected block IDs are converted to raw token indices before attention.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from muse.layers.rms_norm import RMSNorm


_INDEXER_SCORER_BACKENDS = {"torch", "fused", "auto"}
_FUSED_INDEXER_SCORER = None
_FUSED_INDEXER_SCORER_LOAD_ATTEMPTED = False
_FUSED_INDEXER_SCORER_IMPORT_ERROR: Optional[Exception] = None
_FUSED_SELECTED_INDEXER_SCORER = None
_FUSED_SELECTED_INDEXER_SCORER_LOAD_ATTEMPTED = False
_FUSED_SELECTED_INDEXER_SCORER_IMPORT_ERROR: Optional[Exception] = None


def _resolve_indexer_scorer_backend(config_backend: str) -> str:
    """Resolve the persisted scorer backend and its explicit env override."""
    backend = os.environ.get(
        "SKSA_INDEXER_SCORER_BACKEND", config_backend
    )
    if not isinstance(backend, str):
        raise ValueError("S-KSA indexer scorer backend must be a string")
    backend = backend.strip().lower()
    if backend not in _INDEXER_SCORER_BACKENDS:
        raise ValueError(
            "SKSA_INDEXER_SCORER_BACKEND must be 'torch', 'fused', or 'auto'"
        )
    return backend


def _load_fused_indexer_scorer():
    """Load the optional hksa scorer once without weakening strict dispatch."""
    global _FUSED_INDEXER_SCORER
    global _FUSED_INDEXER_SCORER_LOAD_ATTEMPTED
    global _FUSED_INDEXER_SCORER_IMPORT_ERROR
    if not _FUSED_INDEXER_SCORER_LOAD_ATTEMPTED:
        _FUSED_INDEXER_SCORER_LOAD_ATTEMPTED = True
        try:
            from ksa_attn import fused_indexer_candidate_scores
        except ImportError as error:
            _FUSED_INDEXER_SCORER_IMPORT_ERROR = error
        else:
            _FUSED_INDEXER_SCORER = fused_indexer_candidate_scores
    return _FUSED_INDEXER_SCORER


def _load_fused_selected_indexer_scorer():
    """Load the differentiable selected-route scorer once."""
    global _FUSED_SELECTED_INDEXER_SCORER
    global _FUSED_SELECTED_INDEXER_SCORER_LOAD_ATTEMPTED
    global _FUSED_SELECTED_INDEXER_SCORER_IMPORT_ERROR
    if not _FUSED_SELECTED_INDEXER_SCORER_LOAD_ATTEMPTED:
        _FUSED_SELECTED_INDEXER_SCORER_LOAD_ATTEMPTED = True
        try:
            from ksa_attn import fused_selected_indexer_scores
        except ImportError as error:
            _FUSED_SELECTED_INDEXER_SCORER_IMPORT_ERROR = error
        else:
            _FUSED_SELECTED_INDEXER_SCORER = fused_selected_indexer_scores
    return _FUSED_SELECTED_INDEXER_SCORER


def _fused_indexer_scorer_unavailable_reason(
    query: Tensor,
    key: Tensor,
    head_weight: Tensor,
) -> Optional[str]:
    """Return why this call cannot use the fused no-grad scorer."""
    if torch.is_grad_enabled():
        return "gradient mode is enabled"
    if query.ndim != 4 or key.ndim != 3 or head_weight.ndim != 3:
        return "expected query[B,Q,H,D], key[B,M,D], and weight[B,Q,H]"
    if not (query.is_cuda and key.is_cuda and head_weight.is_cuda):
        return "all scorer tensors must be CUDA tensors"
    if not (query.device == key.device == head_weight.device):
        return "all scorer tensors must share one CUDA device"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return "query must use fp16 or bf16"
    if key.dtype != query.dtype:
        return "query and key dtypes must match"
    if head_weight.dtype != torch.float32:
        return "head weights must use fp32"
    if query.shape[-1] not in (64, 128):
        return "indexer head_dim must be 64 or 128"
    if not 1 <= query.shape[2] <= 8:
        return "indexer head count must be between 1 and 8"
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        return "query/key head dimensions must be contiguous"
    if head_weight.stride(-1) != 1:
        return "head weights must be contiguous over heads"
    if _load_fused_indexer_scorer() is None:
        detail = (
            f": {_FUSED_INDEXER_SCORER_IMPORT_ERROR}"
            if _FUSED_INDEXER_SCORER_IMPORT_ERROR is not None
            else ""
        )
        return f"ksa_attn fused scorer is unavailable{detail}"
    return None


def _fused_selected_indexer_scorer_unavailable_reason(
    query: Tensor,
    key: Tensor,
    head_weight: Tensor,
    selected: Tensor,
) -> Optional[str]:
    """Return why selected-route scoring cannot use its fused kernel."""
    if query.ndim != 4 or key.ndim != 3 or head_weight.ndim != 3:
        return "expected query[B,Q,H,D], key[B,M,D], and weight[B,Q,H]"
    if selected.ndim != 3 or selected.shape[:2] != query.shape[:2]:
        return "selected routes must have shape [B,Q,TopK]"
    if not (
        query.is_cuda
        and key.is_cuda
        and head_weight.is_cuda
        and selected.is_cuda
    ):
        return "all selected scorer tensors must be CUDA tensors"
    if not (query.device == key.device == head_weight.device == selected.device):
        return "all selected scorer tensors must share one CUDA device"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return "query must use fp16 or bf16"
    if key.dtype != query.dtype:
        return "query and key dtypes must match"
    if head_weight.dtype != torch.float32:
        return "head weights must use fp32"
    if selected.dtype not in (torch.int32, torch.int64):
        return "selected routes must use int32 or int64"
    if query.shape[-1] not in (64, 128):
        return "indexer head_dim must be 64 or 128"
    if not 1 <= query.shape[2] <= 8:
        return "indexer head count must be between 1 and 8"
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        return "query/key head dimensions must be contiguous"
    if head_weight.stride(-1) != 1:
        return "head weights must be contiguous over heads"
    if _load_fused_selected_indexer_scorer() is None:
        detail = (
            f": {_FUSED_SELECTED_INDEXER_SCORER_IMPORT_ERROR}"
            if _FUSED_SELECTED_INDEXER_SCORER_IMPORT_ERROR is not None
            else ""
        )
        return f"ksa_attn fused selected scorer is unavailable{detail}"
    return None


@dataclass(frozen=True)
class SKSASummaryIndexerResult:
    """Per-query routes and the summary Indexer training diagnostics.

    ``indexer_loss`` is the differentiable, weighted objective consumed by
    the model.  The remaining fields are detached diagnostics; keeping them
    separate lets a continuation monitor calibration, TopK recall, and the
    teacher attention mass captured by the selected set independently from the
    optimized target.
    """

    selected_block_indices: Tensor
    indexer_loss: Optional[Tensor]
    checkpoint_cache: Optional[Union[
        "SKSASummaryIndexerCheckpointCache", "SKSASummaryIndexerDenseCheckpointCache"
    ]] = None
    kl_loss: Optional[Tensor] = None
    oracle_topk_recall: Optional[Tensor] = None
    selected_attention_mass: Optional[Tensor] = None
    oracle_topk_attention_mass: Optional[Tensor] = None
    selected_oracle_attention_mass_ratio: Optional[Tensor] = None
    # Kept for compatibility with callers that still inspect the old result
    # field.  The training recipe no longer registers this as a TB series;
    # it is derived as ``1 - oracle_topk_recall`` when TopK diagnostics run.


@dataclass(frozen=True)
class SKSASummaryIndexerCheckpointCache:
    """Detached work reused by an enclosing layer checkpoint recomputation.

    Hard routes and Full-QK teacher targets carry no gradient.  Retaining
    their compact ``[B,Q,TopK]`` forms avoids repeating the two most expensive
    no-grad parts of the indexer.  The detached per-tile KL values let an
    enclosing checkpoint recomputation reproduce the exact forward without a
    redundant selected-score pass; custom autograd still rebuilds that pass
    once in backward to produce exact indexer gradients.
    """

    selected_block_indices: Tensor
    teacher_targets: Tuple[Tensor, ...]
    selected_kl_losses: Tuple[Tensor, ...]
    topk_metrics: Optional[Tuple[Tensor, ...]] = None


@dataclass(frozen=True)
class SKSASummaryIndexerDenseCheckpointCache:
    """Detached losses, diagnostics, and compact LSE/route caches.

    Reuse during the enclosing layer's recomputation; do not retain dense
    score matrices or teacher distributions across the full sequence.
    """

    tile_losses: Tuple[Tensor, ...]
    tile_metrics: Tuple[Tensor, ...]
    selected_block_indices: Optional[Tensor] = None
    dense_lse: Optional[Tensor] = None


def _apply_partial_rope(
    tensor: Tensor,
    position_ids: Tensor,
    rope: Optional[RotaryPositionalEmbeddings],
    rope_dim: int,
) -> Tensor:
    """Apply RoPE to the trailing ``rope_dim`` dimensions only."""
    if rope is None or rope_dim == 0 or tensor.shape[1] == 0:
        return tensor
    rotated = rope(tensor[..., -rope_dim:], input_pos=position_ids)
    if rope_dim == tensor.shape[-1]:
        return rotated
    return torch.cat((tensor[..., :-rope_dim], rotated), dim=-1)


def _masked_kl(
    scores: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    return_count: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Return summed ``KL(target || softmax(scores))`` and valid row count."""
    has_candidate = valid.any(dim=-1)
    masked_scores = scores.float().masked_fill(~valid, float("-inf"))
    # Avoid synchronizing once per tile merely to special-case an empty early
    # prefix. Those rows receive a harmless finite softmax and zero target.
    masked_scores.masked_fill_(~has_candidate[..., None], 0.0)
    log_probability = torch.log_softmax(masked_scores, dim=-1)
    safe_target = target.float().clamp_min(1e-12)
    per_row = (
        target.float() * (safe_target.log() - log_probability)
    ).masked_fill(~valid, 0.0).sum(dim=-1)
    # Invalid rows are already exactly zero.  Summing the fixed-shape tensor
    # avoids CUDA boolean indexing, whose dynamically sized output can force a
    # host synchronization once per query tile.
    count = has_candidate.sum().float() if return_count else None
    return per_row.sum(), count


def _masked_kl_gradient_surrogate(
    scores: Tensor,
    target: Tensor,
    valid: Tensor,
) -> Tensor:
    """Return a scalar with the exact KL gradient with respect to ``scores``.

    The selected teacher is detached, so its entropy is constant with respect
    to all trainable indexer inputs.  Custom backward only needs
    ``-target * log_softmax(scores)``; rebuilding ``target * log(target)``
    would add clamp/log/elementwise work without contributing any gradient.
    The real forward still evaluates :func:`_masked_kl`, so the reported loss
    value is unchanged.
    """
    has_candidate = valid.any(dim=-1)
    masked_scores = scores.float().masked_fill(~valid, float("-inf"))
    masked_scores.masked_fill_(~has_candidate[..., None], 0.0)
    log_probability = torch.log_softmax(masked_scores, dim=-1)
    per_row = (-(target.float() * log_probability)).masked_fill(
        ~valid, 0.0
    ).sum(dim=-1)
    return per_row.sum()






@torch.no_grad()
def _dense_summary_topk_attention_mass_tile(
    scores: Tensor,
    teacher: Tensor,
    eligible_count: Tensor,
    *,
    topk_blocks: int,
    sink_block_count: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Measure TopK recall and teacher mass captured by the student set.

    The metric is computed over the same remote summary-block set as the
    Oracle TopK diagnostic: forced sink blocks are excluded, as are rows that
    do not yet have enough eligible remote candidates.  ``teacher`` is the
    already pooled dense-teacher block distribution, so the returned mass
    fractions are relative to all eligible non-sink remote teacher mass.

    The five returned scalars are sums over valid query rows followed by the
    valid-row count: recall sum, student mass-fraction sum, Oracle mass-
    fraction sum, student/Oracle mass-ratio sum, and row count.  Keeping sums
    here lets the caller aggregate tiles without materializing per-query
    diagnostics.
    """
    if scores.ndim != 3 or teacher.shape != scores.shape:
        raise ValueError(
            "summary TopK attention diagnostics scores and teacher must share [B,Q,M]"
        )
    if eligible_count.shape != (scores.shape[1],):
        raise ValueError(
            "summary TopK attention diagnostic eligible counts must match query tile"
        )
    if topk_blocks <= 0:
        zero = scores.new_zeros((), dtype=torch.float32)
        return zero, zero, zero, zero, zero
    if sink_block_count < 0 or sink_block_count >= topk_blocks:
        raise ValueError("summary TopK attention diagnostic sink count must be below TopK")

    batch, _, candidate_count = scores.shape
    remote_topk = topk_blocks - sink_block_count
    if candidate_count < remote_topk or candidate_count == 0:
        zero = scores.new_zeros((), dtype=torch.float32)
        return zero, zero, zero, zero, zero

    candidate_ids = torch.arange(candidate_count, device=scores.device)
    valid = (
        candidate_ids[None, None, :] < eligible_count[None, :, None]
    ) & (candidate_ids[None, None, :] >= sink_block_count)
    eligible_remote = (eligible_count - sink_block_count).clamp_min(0)
    valid_rows = eligible_remote >= remote_topk

    # Use unperturbed scores, as in hard routing. Equal-score membership is
    # left to torch.topk; it need not be invariant to candidate tensor shape.
    teacher_key = torch.where(
        valid, teacher.detach(), torch.full_like(teacher, float("-inf"))
    )
    student_key = torch.where(
        valid, scores.detach(), torch.full_like(scores, float("-inf"))
    )
    _, teacher_ids = torch.topk(
        teacher_key, remote_topk, dim=-1, largest=True, sorted=False
    )
    _, student_ids = torch.topk(
        student_key, remote_topk, dim=-1, largest=True, sorted=False
    )

    teacher_ids_sorted = torch.sort(teacher_ids, dim=-1).values
    positions = torch.searchsorted(teacher_ids_sorted, student_ids)
    safe_positions = positions.clamp_max(remote_topk - 1)
    student_correct = positions < remote_topk
    student_correct &= torch.gather(
        teacher_ids_sorted, -1, safe_positions
    ).eq(student_ids)
    row_mask = valid_rows[None, :]
    recall = (
        (student_correct & row_mask[..., None]).to(torch.float32).sum(dim=-1)
        / float(remote_topk)
    )

    # ``teacher`` is already normalized over all eligible blocks.  Removing
    # the sink and dividing by the remaining mass gives the same remote-mass
    # fraction as using the unnormalized dense attention scores, while avoiding
    # another large teacher-probability materialization.
    remote_teacher = teacher[..., sink_block_count:]
    remote_mass = remote_teacher.sum(dim=-1)
    student_values = torch.gather(
        teacher,
        -1,
        student_ids.clamp_min(0).clamp_max(candidate_count - 1),
    )
    oracle_values = torch.gather(
        teacher,
        -1,
        teacher_ids.clamp_min(0).clamp_max(candidate_count - 1),
    )
    student_mass = student_values.sum(dim=-1)
    oracle_mass = oracle_values.sum(dim=-1)
    has_mass = row_mask & (remote_mass > 0)
    student_fraction = torch.where(
        has_mass,
        student_mass / remote_mass.clamp_min(1e-30),
        torch.zeros_like(student_mass),
    )
    oracle_fraction = torch.where(
        has_mass,
        oracle_mass / remote_mass.clamp_min(1e-30),
        torch.zeros_like(oracle_mass),
    )
    ratio = torch.where(
        has_mass & (oracle_mass > 0),
        student_mass / oracle_mass.clamp_min(1e-30),
        torch.zeros_like(student_mass),
    )

    return (
        recall.masked_fill(~row_mask, 0.0).sum(),
        student_fraction.masked_fill(~row_mask, 0.0).sum(),
        oracle_fraction.masked_fill(~row_mask, 0.0).sum(),
        ratio.masked_fill(~row_mask, 0.0).sum(),
        row_mask.to(torch.float32).sum(),
    )


def _score_all_indexer_candidates(
    query: Tensor,
    key: Tensor,
    head_weight: Tensor,
    score_scale: float,
    *,
    scorer_backend: str = "torch",
    dense_head_order: bool = False,
) -> Tensor:
    """Score every summary key for route selection or dense warm-up KL."""
    backend = _resolve_indexer_scorer_backend(scorer_backend)
    # Dense warm-up needs this differentiable PyTorch graph regardless of the
    # configured hard-route backend.  Only the no-grad TopK path may dispatch
    # to the fused scorer.
    if not torch.is_grad_enabled() and not dense_head_order:
        if backend != "torch":
            unavailable = _fused_indexer_scorer_unavailable_reason(
                query, key, head_weight
            )
            if unavailable is None:
                # Do not catch execution/compilation errors here.  A requested
                # fused scorer must never silently change routes by falling
                # back to a numerically different implementation.
                return _load_fused_indexer_scorer()(
                    query, key, head_weight, score_scale
                )
            if backend == "fused":
                raise RuntimeError(
                    f"strict fused S-KSA indexer scorer is unavailable: {unavailable}"
                )
    if not torch.is_grad_enabled() and not dense_head_order:
        dot = torch.einsum("bqhd,bmd->bqhm", query, key)
        activated = dot.float()
        activated.mul_(score_scale).relu_()
        # Hard routing consumes these scores under no_grad.  There is no
        # backward state to preserve, so the head weighting can reuse the
        # large B x Q x H x candidates activation as well.
        activated.mul_(head_weight[..., None])
        return activated.sum(dim=2)
    # Dense warm-up is the only differentiable all-candidate path.  Computing
    # all heads in one einsum leaves both the B x Q x H x M activation and a
    # second tensor of the same size for head weighting alive at once.  Accrue
    # one head at a time into the compact B x Q x M result instead.  The head
    # order and operations are unchanged, while the peak temporary storage is
    # reduced by roughly ``indexer_heads``.
    scores = torch.zeros(
        (query.shape[0], query.shape[1], key.shape[1]),
        dtype=torch.float32,
        device=query.device,
    )
    for head_index in range(query.shape[2]):
        dot_head = torch.einsum(
            "bqd,bmd->bqm", query[:, :, head_index], key
        )
        activated_head = dot_head.float()
        activated_head.mul_(score_scale).relu_()
        # ``addcmul_`` fuses the weighting and accumulation without a
        # temporary B x Q x M product tensor.  It remains differentiable for
        # both the activated QK term and the per-head weight.
        scores.addcmul_(
            activated_head, head_weight[:, :, head_index, None]
        )
    return scores


def _dense_indexer_kl_tile(
    query: Tensor,
    key: Tensor,
    head_weight: Tensor,
    main_q: Tensor,
    main_k: Tensor,
    lse: Tensor,
    eligible_count: Tensor,
    *,
    query_start: int,
    summary_block_size: int,
    local_window_tokens: int,
    teacher_pooling: str,
    teacher_query_block_size: int,
    scorer_backend: str,
    diagnostic_topk: int,
    sink_block_count: int,
    gradient_only: bool = False,
) -> Tuple[Tensor, Tensor]:
    """One complete KL tile, with no large score/teacher outputs escaping.

    Checkpoint the *score and loss together*. Rebuild masks and detached
    teacher targets here instead of saving a sequence's worth of dense masks
    or probabilities as checkpoint inputs. Main Q/K/LSE and projected keys
    are shared storage, not copies per tile. Bind scalar tile arguments with
    functools.partial so backward cannot see the last loop iteration's values.
    """
    count = query.shape[1]
    scores = _score_all_indexer_candidates(
        query, key, head_weight, query.shape[-1] ** -0.5,
        scorer_backend=scorer_backend,
        dense_head_order=True,
    )
    candidate_ids = torch.arange(key.shape[1], device=key.device)
    valid = candidate_ids[None, None] < eligible_count[None, :, None]
    valid = valid.expand(query.shape[0], -1, -1)
    step = (
        count if main_q.is_cuda and main_q.dtype in (torch.float16, torch.bfloat16)
        else teacher_query_block_size
    )
    loss = query.reshape(-1)[:1].sum() * 0.
    metrics = scores.new_zeros(5)
    for offset in range(0, count, step):
        end = min(count, offset + step)
        tile_valid = valid[:, offset:end]
        target = dense_teacher_summary_block_distribution(
            main_q, main_k, lse, query_start=query_start + offset,
            query_end=query_start + end, summary_block_size=summary_block_size,
            summary_count=key.shape[1], valid=tile_valid,
            pooling=teacher_pooling, local_window_tokens=local_window_tokens,
        )
        if gradient_only:
            tile_loss = _masked_kl_gradient_surrogate(
                scores[:, offset:end], target, tile_valid
            )
        else:
            tile_loss, _ = _masked_kl(
                scores[:, offset:end], target, tile_valid, return_count=False
            )
        loss = loss + tile_loss
        if diagnostic_topk:
            diagnostics = _dense_summary_topk_attention_mass_tile(
                scores[:, offset:end].detach(), target, eligible_count[offset:end],
                topk_blocks=diagnostic_topk, sink_block_count=sink_block_count,
            )
            metrics = metrics + torch.stack(diagnostics)
    return loss, metrics.detach()


class _DenseIndexerKLLoss(torch.autograd.Function):
    """Rebuild one tile in backward, omitting detached forward diagnostics.

    As in the selected-set loss, identical save order lets the enclosing
    non-reentrant checkpoint restore inputs while reusing cached scalars.
    Native PyTorch scoring and its head accumulation order are preserved.
    """

    @staticmethod
    def forward(ctx, query, key, head_weight, main_q, main_k, lse,
                eligible_count, tile_fn, cached):
        ctx.save_for_backward(query, key, head_weight, main_q, main_k, lse,
                              eligible_count)
        ctx.tile_fn = tile_fn
        if cached is None:
            loss, metrics = tile_fn(query, key, head_weight, main_q, main_k,
                                    lse, eligible_count)
        else:
            loss, metrics = (tensor.clone() for tensor in cached)
        ctx.mark_non_differentiable(metrics)
        return loss, metrics

    @staticmethod
    def backward(ctx, grad_output, grad_metrics):
        query, key, head_weight, *teacher_inputs = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        rebuilt = [tensor.detach().requires_grad_(needed)
                   for tensor, needed in zip((query, key, head_weight), needs)]
        differentiable = [tensor for tensor, needed in zip(rebuilt, needs) if needed]
        gradients = iter(())
        if differentiable:
            with torch.enable_grad():
                loss, _ = ctx.tile_fn(
                    *rebuilt, *teacher_inputs, diagnostic_topk=0,
                    gradient_only=True,
                )
            gradients = iter(torch.autograd.grad(
                loss, differentiable, grad_outputs=grad_output,
            ))
        return (*(next(gradients) if needed else None for needed in needs),
                None, None, None, None, None, None)


def _score_selected_indexer_candidates(
    query: Tensor,
    key: Tensor,
    head_weight: Tensor,
    selected: Tensor,
    score_scale: float,
    *,
    scorer_backend: str = "torch",
) -> Tensor:
    """Differentiably rescore only the hard-selected summary keys.

    Hard TopK has no route gradient.  The selected-set KL therefore has the
    same gradient as gathering these entries from the dense score matrix, but
    does not need an autograd graph for the unselected ``summary_count-TopK``
    candidates.
    """
    backend = _resolve_indexer_scorer_backend(scorer_backend)
    if backend != "torch":
        unavailable = _fused_selected_indexer_scorer_unavailable_reason(
            query, key, head_weight, selected
        )
        if unavailable is None:
            return _load_fused_selected_indexer_scorer()(
                query, key, head_weight, selected, score_scale
            )
        if backend == "fused":
            raise RuntimeError(
                "strict fused S-KSA selected scorer is unavailable: "
                f"{unavailable}"
            )
    batch = query.shape[0]
    safe_selected = selected.clamp_min(0).long()
    batch_index = torch.arange(batch, device=key.device)[:, None, None]
    selected_key = key[batch_index, safe_selected]
    dot = torch.einsum("bqhd,bqkd->bqhk", query, selected_key)
    activated = dot.float()
    activated.mul_(score_scale).relu_()
    return (activated * head_weight[..., None]).sum(dim=2)


def _checkpointed_selected_indexer_scores(
    query: Tensor,
    key: Tensor,
    head_weight: Tensor,
    selected: Tensor,
    score_scale: float,
) -> Tensor:
    """Selected scores with bounded saved activations at 128K context."""
    score_fn = lambda q, k, w, s: _score_selected_indexer_candidates(
        q, k, w, s, score_scale
    )
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, head_weight)
    ):
        # Recompute the selected-key gather in backward rather than retaining
        # one large copy per query tile across a 128K layer. Non-reentrant
        # checkpointing also composes with the outer layer checkpoint.
        return checkpoint(
            score_fn,
            query,
            key,
            head_weight,
            selected,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return score_fn(query, key, head_weight, selected)


class _SelectedIndexerKLLoss(torch.autograd.Function):
    """Memory-bounded selected-set KL with recomputation-aware forward reuse.

    An enclosing non-reentrant layer checkpoint executes this Function twice:
    once in the real forward and once while recovering saved tensors during
    backward.  Both executions save the exact same tensor sequence.  The real
    forward computes and caches the detached scalar KL; the outer recompute
    can return that scalar without repeating the large selected-key gather.
    The original Function's backward then rebuilds the score and KL once with
    gradients enabled, retaining only one query tile at a time.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        head_weight: Tensor,
        selected: Tensor,
        target: Tensor,
        valid: Tensor,
        cached_loss: Optional[Tensor],
        score_scale: float,
        scorer_backend: str,
    ) -> Tensor:
        # Keep this save order identical in the original and checkpoint
        # recompute phases.  PyTorch's non-reentrant checkpoint pairs these
        # saves with the original autograd context by position and metadata.
        ctx.save_for_backward(
            query,
            key,
            head_weight,
            selected,
            target,
            valid,
        )
        ctx.score_scale = score_scale
        ctx.scorer_backend = scorer_backend
        if cached_loss is not None:
            # Do not return an input tensor as-is from a custom Function: a
            # scalar clone makes the aliasing contract unambiguous while
            # preserving its value bit-for-bit.
            return cached_loss.clone()
        scores = _score_selected_indexer_candidates(
            query,
            key,
            head_weight,
            selected,
            score_scale,
            scorer_backend=scorer_backend,
        )
        loss, _ = _masked_kl(
            scores, target, valid, return_count=False
        )
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, head_weight, selected, target, valid = ctx.saved_tensors
        needs_gradient = ctx.needs_input_grad[:3]
        differentiable = []
        rebuilt_inputs = []
        for tensor, needed in zip(
            (query, key, head_weight), needs_gradient
        ):
            rebuilt = tensor.detach().requires_grad_(needed)
            rebuilt_inputs.append(rebuilt)
            if needed:
                differentiable.append(rebuilt)

        gradients = []
        if differentiable:
            with torch.enable_grad():
                scores = _score_selected_indexer_candidates(
                    rebuilt_inputs[0],
                    rebuilt_inputs[1],
                    rebuilt_inputs[2],
                    selected,
                    ctx.score_scale,
                    scorer_backend=ctx.scorer_backend,
                )
                loss = _masked_kl_gradient_surrogate(scores, target, valid)
            gradients = list(
                torch.autograd.grad(
                    loss,
                    differentiable,
                    grad_outputs=grad_output,
                )
            )

        mapped_gradients = []
        gradient_index = 0
        for needed in needs_gradient:
            if needed:
                mapped_gradients.append(gradients[gradient_index])
                gradient_index += 1
            else:
                mapped_gradients.append(None)
        return (*mapped_gradients, None, None, None, None, None, None)


def _selected_indexer_kl_loss(
    query: Tensor,
    key: Tensor,
    head_weight: Tensor,
    selected: Tensor,
    target: Tensor,
    valid: Tensor,
    score_scale: float,
    *,
    cached_loss: Optional[Tensor] = None,
    scorer_backend: str = "torch",
) -> Tensor:
    """Return selected-set KL while bounding saved activation memory."""
    return _SelectedIndexerKLLoss.apply(
        query,
        key,
        head_weight,
        selected,
        target,
        valid,
        cached_loss,
        score_scale,
        scorer_backend,
    )


@torch.no_grad()
def _dense_causal_lse(q: Tensor, k: Tensor) -> Tensor:
    """Exact dense raw-token causal attention LSE in ``[B,H,Q]`` layout."""
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("dense teacher q/k must share [B,text,H,D] shape")
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            from ksa_attn import dense_causal_attention_lse
        except ImportError as exc:
            raise RuntimeError(
                "S-KSA dense warm-up requires kernel/v0.1 to be installed"
            ) from exc
        # The dedicated kernel returns only LSE.  In particular it never sets
        # FlashAttention's diagnostic ``return_attn_probs=True`` flag, which
        # would materialize an O(sequence^2) probability tensor.
        return dense_causal_attention_lse(q, k)

    sequence = q.shape[1]
    if sequence > 4096:
        raise RuntimeError(
            "The fp32/CPU dense teacher is a small-sequence oracle; use CUDA "
            "fp16/bf16 for long-context warm-up"
        )
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float())
    logits.mul_(q.shape[-1] ** -0.5)
    causal = torch.ones(sequence, sequence, dtype=torch.bool, device=q.device).tril_()
    return torch.logsumexp(logits.masked_fill(~causal, float("-inf")), dim=-1)


def _can_reuse_full_attention_lse_for_dense_teacher(
    *,
    summary_attention_mode: str,
    summary_layout_validated: bool,
    summary_layout_signature: Optional[Tuple[int, bool]],
    expanded_sequence_length: int,
    text_count: int,
    summary_chunk_size: int,
    summary_token_num: int,
    sliding_chunk_num: int,
) -> bool:
    """Whether Full-attention text rows are exactly dense raw causal rows.

    The copied Summary-Lightning kernel exposes the LSE already maintained by
    normal Full attention.  It is a valid dense-teacher denominator only for a
    complete, regular summary layout whose sliding raw-token window covers the
    entire sequence.  Otherwise old summary keys can replace remote raw keys,
    so callers must retain the dedicated dense-causal LSE fallback.

    This predicate intentionally uses static layout metadata only.  It adds no
    CUDA synchronization to the six Full layers in the 128K warm-up.
    """
    if summary_attention_mode != "kernel":
        return False
    if not summary_layout_validated or summary_layout_signature is None:
        return False
    if summary_chunk_size <= 0 or summary_token_num <= 0 or text_count <= 0:
        return False
    summary_count, all_chunks_summarized = summary_layout_signature
    if not all_chunks_summarized or summary_count <= 0:
        return False
    if summary_count % summary_token_num:
        return False
    chunk_count = summary_count // summary_token_num
    if chunk_count * summary_chunk_size != text_count:
        return False
    if expanded_sequence_length != text_count + summary_count:
        return False
    return sliding_chunk_num >= chunk_count


def _extract_regular_summary_text_lse(
    expanded_lse: Tensor,
    *,
    summary_chunk_size: int,
    summary_token_num: int,
    expected_text_count: int,
) -> Tensor:
    """Extract regular-layout text rows from [B,H,expanded] Full LSE."""
    if expanded_lse.ndim != 3:
        raise ValueError("Full-attention LSE must have [B,H,expanded] shape")
    if expanded_lse.dtype != torch.float32:
        raise ValueError("Full-attention LSE must be float32")
    block_size = summary_chunk_size + summary_token_num
    if summary_chunk_size <= 0 or summary_token_num <= 0:
        raise ValueError("summary chunk/token counts must be positive")
    if expanded_lse.shape[-1] % block_size:
        raise ValueError("Full-attention LSE does not have a regular summary layout")
    chunk_count = expanded_lse.shape[-1] // block_size
    if chunk_count * summary_chunk_size != expected_text_count:
        raise ValueError("Full-attention LSE text count mismatch")
    # view(..., block) exposes the builder's [raw..., summary...] layout.  The
    # compact contiguous copy is only B*H*text fp32 (8 MiB at the formal 128K
    # shape) and can safely be retained across one checkpoint recomputation.
    return (
        expanded_lse.view(
            expanded_lse.shape[0],
            expanded_lse.shape[1],
            chunk_count,
            block_size,
        )[..., :summary_chunk_size]
        .reshape(expanded_lse.shape[0], expanded_lse.shape[1], expected_text_count)
        .contiguous()
    )


@torch.no_grad()
def dense_teacher_summary_block_distribution(
    q: Tensor,
    k: Tensor,
    lse: Tensor,
    *,
    query_start: int,
    query_end: int,
    summary_block_size: int,
    summary_count: int,
    valid: Tensor,
    pooling: str,
    local_window_tokens: int,
) -> Tensor:
    """Pool dense Full attention into QSA-style summary-block targets.

    Probabilities are first aggregated across ordinary Full-attention heads,
    then max- or sum-pooled over each summary token's raw block, and finally
    L1-normalized over causally eligible remote blocks.
    """
    batch = q.shape[0]
    query_count = query_end - query_start
    if tuple(valid.shape) != (batch, query_count, summary_count):
        raise ValueError("dense teacher valid-mask shape mismatch")
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            from ksa_attn import dense_teacher_block_distribution
        except ImportError as exc:
            raise RuntimeError(
                "S-KSA dense warm-up requires kernel/v0.1 to be installed"
            ) from exc
        target = dense_teacher_block_distribution(
            q,
            k,
            lse,
            query_start=query_start,
            query_end=query_end,
            summary_block_size=summary_block_size,
            summary_count=summary_count,
            local_window_tokens=local_window_tokens,
            pooling=pooling,
        )
        # The kernel derives the regular causal eligibility directly. Keep the
        # caller mask authoritative so this helper retains its exact public
        # semantics for diagnostic/custom masks as well.
        target.masked_fill_(~valid, 0.0)
        denominator = target.sum(dim=-1, keepdim=True)
        return torch.where(
            denominator > 0,
            target / denominator.clamp_min(1e-30),
            target,
        )
    target = torch.zeros(
        (batch, query_count, summary_count), dtype=torch.float32, device=q.device
    )
    if summary_count == 0:
        return target

    # Only contract against the longest complete remote prefix needed by this
    # query tile. Per-row eligibility is restored by ``valid`` below.
    max_eligible = min(
        summary_count,
        max(0, query_end - local_window_tokens) // summary_block_size,
    )
    key_end = min(k.shape[1], summary_count * summary_block_size,
                  max_eligible * summary_block_size)
    if key_end == 0:
        return target
    complete_blocks = key_end // summary_block_size
    key_end = complete_blocks * summary_block_size
    logits = torch.einsum(
        "bqhd,bkhd->bqhk",
        q[:, query_start:query_end].detach(),
        k[:, :key_end].detach(),
    ).float()
    logits.mul_(q.shape[-1] ** -0.5)
    query_lse = lse[:, :, query_start:query_end].permute(0, 2, 1)
    logits.sub_(query_lse[..., None]).exp_()
    token_probability = logits.mean(dim=2)
    del logits
    grouped = token_probability.view(
        batch, query_count, complete_blocks, summary_block_size
    )
    if pooling == "max":
        block_score = grouped.amax(dim=-1)
    elif pooling == "sum":
        block_score = grouped.sum(dim=-1)
    else:
        raise ValueError(f"unsupported S-KSA teacher pooling: {pooling}")
    target[:, :, :complete_blocks] = block_score
    target.masked_fill_(~valid, 0.0)
    denominator = target.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > 0,
        target / denominator.clamp_min(1e-30),
        target,
    )


@torch.no_grad()
def selected_teacher_summary_block_distribution(
    q: Tensor,
    k: Tensor,
    selected: Tensor,
    *,
    summary_block_size: int,
    pooling: str,
    all_rows_have_candidates: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Dense Full-attention teacher restricted to the hard selected set."""
    batch, query_count, heads, head_dim = q.shape
    if tuple(selected.shape[:2]) != (batch, query_count):
        raise ValueError("selected teacher route does not match q")
    topk = selected.shape[-1]
    valid = selected >= 0
    if topk == 0:
        return q.new_zeros((batch, query_count, 0), dtype=torch.float32), valid

    fused_cuda_logits = q.is_cuda and q.dtype in (
        torch.float16, torch.bfloat16
    )
    if fused_cuda_logits and (
        (summary_block_size in (2, 4, 8) and topk <= 512)
        or (summary_block_size == 1 and topk <= 4096)
    ):
        # Preserve the established one-pass selected QK, but fuse its FP32
        # LSE, per-head softmax, head average, raw-block pooling and final L1
        # normalization.  The returned slots retain the caller's original
        # route order, so the selected student KL remains unchanged.
        from ksa_attn import (
            selected_teacher_block_distribution_materialized_fused,
        )

        target = selected_teacher_block_distribution_materialized_fused(
            q.detach(),
            k.detach(),
            selected,
            block_size_tokens=summary_block_size,
            pooling=pooling,
            sort_routes=False,
        )
        return target, valid
    if fused_cuda_logits:
        # Follow the selected raw-block indirection directly instead of
        # materializing [B,Q,TopK,block,H,D] gathered keys.
        from ksa_attn import selected_teacher_qk_logits

        logits = selected_teacher_qk_logits(
            q.detach(),
            k.detach(),
            selected,
            block_size_tokens=summary_block_size,
            softmax_scale=head_dim ** -0.5,
        ).view(batch, query_count, heads, topk, summary_block_size)
    else:
        safe = selected.clamp_min(0).to(torch.long)
        offsets = torch.arange(summary_block_size, device=k.device)
        token_indices = safe[..., None] * summary_block_size + offsets
        batch_indices = torch.arange(
            batch, device=k.device
        )[:, None, None, None, None]
        head_indices = torch.arange(heads, device=k.device)[None, None, None, None]
        selected_keys = k[
            batch_indices,
            token_indices[..., None].expand(-1, -1, -1, -1, heads),
            head_indices,
        ]
        logits = torch.einsum(
            "bqhd,bqkshd->bqhks", q.detach(), selected_keys.detach()
        ).float()
        logits.mul_(head_dim ** -0.5)
    if not fused_cuda_logits:
        # The gather oracle reads block zero for -1 padding.  The CUDA kernel
        # already writes -inf for those exact slots, so repeating this 256-MiB
        # pass on every production tile would be a pure no-op.
        logits.masked_fill_(~valid[:, :, None, :, None], float("-inf"))
    if not all_rows_have_candidates:
        any_valid = valid.any(dim=-1)
        logits.masked_fill_(
            ~any_valid[:, :, None, None, None], 0.0
        )
    # This teacher is no_grad and logits have no later consumer.  Reuse their
    # storage for softmax instead of allocating another
    # B x Q x H x (TopK*block) FP32 tensor (512 MiB at Q=4096).
    flat_logits = logits.flatten(-2)
    torch.softmax(flat_logits, dim=-1, out=flat_logits)
    token_probability = logits
    token_probability = token_probability.mean(dim=2)
    if pooling == "max":
        block_score = token_probability.amax(dim=-1)
    elif pooling == "sum":
        block_score = token_probability.sum(dim=-1)
    else:
        raise ValueError(f"unsupported S-KSA teacher pooling: {pooling}")
    block_score.masked_fill_(~valid, 0.0)
    denominator = block_score.sum(dim=-1, keepdim=True)
    target = torch.where(
        denominator > 0,
        block_score / denominator.clamp_min(1e-30),
        block_score,
    )
    return target, valid


class SKSASummaryLightningIndexer(nn.Module):
    """Layer-local DeepSeek-style indexer over inserted summary hidden states."""

    def __init__(
        self,
        *,
        embed_dim: int,
        query_rank: int,
        indexer_heads: int,
        indexer_head_dim: int,
        rope_dim: int,
        max_seq_len: int,
        rope_base: int,
        query_block_size: int,
        scorer_backend: str = "torch",
        chunk_attention_qk_source: str = "backbone",
        chunk_num_heads: Optional[int] = None,
        chunk_num_kv_heads: Optional[int] = None,
        chunk_head_dim: Optional[int] = None,
        chunk_norm_eps: float = 1e-6,
        chunk_q_bias: bool = False,
        chunk_k_bias: bool = False,
        chunk_attention_v_source: str = "backbone",
        chunk_v_bias: bool = False,
        chunk_attention_include_summary: bool = True,
    ) -> None:
        super().__init__()
        self.indexer_heads = indexer_heads
        self.indexer_head_dim = indexer_head_dim
        self.rope_dim = rope_dim
        self.query_block_size = query_block_size
        if scorer_backend not in _INDEXER_SCORER_BACKENDS:
            raise ValueError(
                "sksa_indexer_scorer_backend must be 'torch', 'fused', or 'auto'"
            )
        self.scorer_backend = scorer_backend
        if chunk_attention_qk_source not in {"backbone", "independent"}:
            raise ValueError(
                "chunk_attention_qk_source must be 'backbone' or 'independent'"
            )
        self.chunk_attention_qk_source = chunk_attention_qk_source
        if chunk_attention_v_source not in {"backbone", "independent"}:
            raise ValueError("chunk_attention_v_source must be backbone or independent")
        if chunk_attention_v_source in {"independent"} and chunk_attention_qk_source != "independent":
            raise ValueError("independent chunk V requires independent chunk Q/K")
        self.chunk_attention_v_source = chunk_attention_v_source
        self.chunk_attention_include_summary = chunk_attention_include_summary
        if not chunk_attention_include_summary and chunk_attention_qk_source != "independent":
            raise ValueError("raw-only chunk attention requires independent chunk Q/K")
        self.chunk_v_proj = None
        self.query_down_proj = nn.Linear(embed_dim, query_rank, bias=False)
        self.query_up_proj = nn.Linear(
            query_rank, indexer_heads * indexer_head_dim, bias=False
        )
        self.key_proj = nn.Linear(embed_dim, indexer_head_dim, bias=False)
        self.head_weight_proj = nn.Linear(embed_dim, indexer_heads, bias=False)
        self.chunk_num_heads = None
        self.chunk_num_kv_heads = None
        self.chunk_head_dim = None
        self.chunk_q_proj = None
        self.chunk_k_proj = None
        self.chunk_q_norm = None
        self.chunk_k_norm = None
        if chunk_attention_qk_source == "independent":
            if (
                chunk_num_heads is None
                or chunk_num_kv_heads is None
                or chunk_head_dim is None
                or min(chunk_num_heads, chunk_num_kv_heads, chunk_head_dim) <= 0
            ):
                raise ValueError(
                    "independent chunk Q/K requires positive head geometry"
                )
            if chunk_num_heads % chunk_num_kv_heads:
                raise ValueError(
                    "chunk query heads must be divisible by chunk KV heads"
                )
            self.chunk_num_heads = int(chunk_num_heads)
            self.chunk_num_kv_heads = int(chunk_num_kv_heads)
            self.chunk_head_dim = int(chunk_head_dim)
            self.chunk_q_proj = nn.Linear(
                embed_dim,
                self.chunk_num_heads * self.chunk_head_dim,
                bias=chunk_q_bias,
            )
            self.chunk_k_proj = nn.Linear(
                embed_dim,
                self.chunk_num_kv_heads * self.chunk_head_dim,
                bias=chunk_k_bias,
            )
            self.chunk_q_norm = RMSNorm(self.chunk_head_dim, eps=chunk_norm_eps)
            self.chunk_k_norm = RMSNorm(self.chunk_head_dim, eps=chunk_norm_eps)
        if chunk_attention_v_source == "independent":
            self.chunk_v_proj = nn.Linear(
                embed_dim, self.chunk_num_kv_heads * self.chunk_head_dim,
                bias=chunk_v_bias,
            )
        self.rope = (
            RotaryPositionalEmbeddings(
                dim=rope_dim, max_seq_len=max_seq_len, base=rope_base
            )
            if rope_dim
            else None
        )

    def build_chunk_attention_summary(
        self,
        hidden_chunks: Tensor,
        value_chunks: Tensor,
        chunk_position_ids: Tensor,
        *,
        summary_chunk_size: int,
        summary_token_num: int,
        pos_embeddings: Optional[nn.Module],
        indexer_block_size: Optional[int] = None,
    ) -> Tensor:
        """Build local summary states with the independent chunk Q/K path.

        The hidden states and values are detached at this boundary.  The
        resulting graph contains the Indexer-local Q/K, Q/K norms, optional
        independent V, and the downstream Indexer key projection.
        """
        if self.chunk_attention_qk_source != "independent":
            raise RuntimeError(
                "independent chunk Q/K is not enabled for this Indexer"
            )
        if (
            hidden_chunks.ndim != 4
            or value_chunks.ndim != 5
            or chunk_position_ids.ndim != 3
        ):
            raise ValueError(
                "chunk attention expects hidden[B,N,W,E], value[B,N,W,H,D], "
                "and position[B,N,W]"
            )
        batch, chunk_count, chunk_width, _ = hidden_chunks.shape
        if tuple(value_chunks.shape[:3]) != (batch, chunk_count, chunk_width):
            raise ValueError("chunk attention hidden/value geometry mismatch")
        if tuple(chunk_position_ids.shape) != (batch, chunk_count, chunk_width):
            raise ValueError("chunk attention position geometry mismatch")
        if (
            summary_chunk_size <= 0
            or summary_token_num <= 0
            or summary_chunk_size + summary_token_num != chunk_width
        ):
            raise ValueError("chunk attention chunk width is inconsistent")
        if (
            self.chunk_q_proj is None
            or self.chunk_k_proj is None
            or self.chunk_q_norm is None
            or self.chunk_k_norm is None
        ):
            raise RuntimeError("independent chunk Q/K modules are incomplete")

        hidden_chunks = hidden_chunks.detach()
        if self.chunk_v_proj is not None:
            value_chunks = self.chunk_v_proj(hidden_chunks).view(
                batch, chunk_count, chunk_width,
                self.chunk_num_kv_heads, self.chunk_head_dim,
            )
            if self.chunk_num_heads != self.chunk_num_kv_heads:
                groups = self.chunk_num_heads // self.chunk_num_kv_heads
                value_chunks = value_chunks.repeat_interleave(groups, dim=3)
        else:
            value_chunks = value_chunks.detach()
        q = self.chunk_q_proj(
            hidden_chunks[:, :, summary_chunk_size:, :]
        ).view(
            batch,
            chunk_count,
            summary_token_num,
            self.chunk_num_heads,
            self.chunk_head_dim,
        )
        k = self.chunk_k_proj(hidden_chunks).view(
            batch,
            chunk_count,
            chunk_width,
            self.chunk_num_kv_heads,
            self.chunk_head_dim,
        )
        q = self.chunk_q_norm(q)
        k = self.chunk_k_norm(k)

        if pos_embeddings is not None:
            q = pos_embeddings(
                q.reshape(
                    batch,
                    chunk_count * summary_token_num,
                    self.chunk_num_heads,
                    self.chunk_head_dim,
                ),
                input_pos=chunk_position_ids[:, :, summary_chunk_size:].reshape(
                    batch, chunk_count * summary_token_num
                ),
            ).view(
                batch,
                chunk_count,
                summary_token_num,
                self.chunk_num_heads,
                self.chunk_head_dim,
            )
            k = pos_embeddings(
                k.reshape(
                    batch,
                    chunk_count * chunk_width,
                    self.chunk_num_kv_heads,
                    self.chunk_head_dim,
                ),
                input_pos=chunk_position_ids.reshape(
                    batch, chunk_count * chunk_width
                ),
            ).view(
                batch,
                chunk_count,
                chunk_width,
                self.chunk_num_kv_heads,
                self.chunk_head_dim,
            )

        if self.chunk_num_heads != self.chunk_num_kv_heads:
            groups = self.chunk_num_heads // self.chunk_num_kv_heads
            k = (
                k.unsqueeze(4)
                .expand(
                    batch,
                    chunk_count,
                    chunk_width,
                    self.chunk_num_kv_heads,
                    groups,
                    self.chunk_head_dim,
                )
                .reshape(
                    batch,
                    chunk_count,
                    chunk_width,
                    self.chunk_num_heads,
                    self.chunk_head_dim,
                )
            )
        if tuple(value_chunks.shape[-2:]) != (
            self.chunk_num_heads,
            self.chunk_head_dim,
        ):
            raise ValueError("chunk attention values do not match Q/K geometry")

        block_size = indexer_block_size or summary_chunk_size
        if block_size != summary_chunk_size or not self.chunk_attention_include_summary:
            if (
                summary_token_num != 1
                or block_size not in (2, 4, 8)
                or summary_chunk_size % block_size
            ):
                raise ValueError("raw-block attention requires one summary and blocks of 2, 4 or 8")
            # Summary supplies Q only; normalize over each raw-text block.
            groups = summary_chunk_size // block_size
            raw_k = k[:, :, :summary_chunk_size].reshape(
                batch, chunk_count, groups, block_size,
                self.chunk_num_heads, self.chunk_head_dim,
            )
            raw_v = value_chunks[:, :, :summary_chunk_size].reshape_as(raw_k)
            logits = torch.einsum("bnhd,bngshd->bnghs", q[:, :, 0], raw_k).float()
            probability = torch.softmax(logits * (self.chunk_head_dim ** -0.5), dim=-1)
            summary = torch.einsum(
                "bnghs,bngshd->bnghd", probability.to(raw_v.dtype), raw_v
            )
            return summary.reshape(batch, chunk_count * groups, -1)

        logits = torch.einsum("bnthd,bnshd->bnhts", q, k).float()
        logits.mul_(self.chunk_head_dim ** -0.5)
        probability = torch.softmax(logits, dim=-1)
        summary = torch.einsum(
            "bnhts,bnshd->bnthd",
            probability.to(value_chunks.dtype),
            value_chunks,
        )
        return summary.reshape(batch, chunk_count * summary_token_num, -1)

    @staticmethod
    def _select_topk(
        scores: Tensor,
        valid: Tensor,
        topk: int,
        force_sink: bool,
        *,
        consume_scores: bool = False,
        sink_block_count: int = 1,
    ) -> Tensor:
        batch, query, candidates = scores.shape
        selected = torch.full(
            (batch, query, topk), -1, dtype=torch.int32, device=scores.device
        )
        if topk == 0 or candidates == 0:
            return selected
        take = min(topk, candidates)
        if consume_scores:
            if scores.requires_grad:
                raise ValueError("consumed TopK scores must not require gradients")
            # Sparse CPT creates this matrix under no_grad solely for hard
            # routing.  Reusing its storage avoids another B x Q x candidates
            # allocation before TopK; the caller discards it immediately.
            selection_scores = scores.detach()
            selection_scores.masked_fill_(~valid, float("-inf"))
        else:
            selection_scores = scores.detach().masked_fill(
                ~valid, float("-inf")
            )
        if force_sink:
            sink_count = min(sink_block_count, candidates, topk)
            sink_valid = valid[..., :sink_count]
            selection_scores[..., :sink_count] = torch.where(
                sink_valid,
                selection_scores.new_full((), float("inf")),
                selection_scores[..., :sink_count],
            )
        # Routes are sorted by block ID below, so TopK need not sort by score.
        values, indices = torch.topk(
            selection_scores, take, dim=-1, sorted=False
        )
        # ``+inf`` is the deliberately forced sink; only masked ``-inf``
        # candidates are padding.
        indices.masked_fill_(values == float("-inf"), -1)
        if take < topk:
            indices = torch.nn.functional.pad(indices, (0, topk - take), value=-1)
        # Sort the chosen IDs for the kernel; this does not break score ties.
        sort_key = torch.where(indices >= 0, indices, candidates + 1)
        return torch.gather(indices, -1, torch.argsort(sort_key, dim=-1)).to(
            torch.int32
        )

    def forward(
        self,
        text_hidden: Tensor,
        summary_hidden: Tensor,
        text_position_ids: Tensor,
        summary_position_ids: Tensor,
        *,
        summary_block_size: int,
        topk_blocks: int,
        local_window_tokens: int,
        force_sink: bool,
        distill: bool,
        dense_warmup: bool,
        teacher_pooling: str,
        full_candidate_kl: bool = False,
        main_q: Optional[Tensor] = None,
        main_k: Optional[Tensor] = None,
        teacher_query_block_size: int = 32,
        precomputed_dense_lse: Optional[Tensor] = None,
        sink_block_count: int = 1,
        preserve_summary_grad: bool = False,
        kl_loss_factor: float = 1.0,
        monitor_topk_blocks: int = 0,
        recompute_cache: Optional[
            Union[SKSASummaryIndexerCheckpointCache,
                  SKSASummaryIndexerDenseCheckpointCache]
        ] = None,
    ) -> SKSASummaryIndexerResult:
        if text_hidden.ndim != 3 or summary_hidden.ndim != 3:
            raise ValueError("S-KSA indexer hidden states must be rank-3")
        batch, text_count, _ = text_hidden.shape
        if summary_hidden.shape[0] != batch:
            raise ValueError("text and summary indexer batches must match")
        summary_count = summary_hidden.shape[1]
        if tuple(text_position_ids.shape) != (batch, text_count):
            raise ValueError("text indexer position shape mismatch")
        if tuple(summary_position_ids.shape) != (batch, summary_count):
            raise ValueError("summary indexer position shape mismatch")
        if distill and (main_q is None or main_k is None):
            raise ValueError("S-KSA indexer distillation requires Full q/k")
        if full_candidate_kl and not distill:
            raise ValueError("full-candidate KL requires indexer distillation")
        dense_supervision = dense_warmup or full_candidate_kl
        if any(
            not math.isfinite(value) or value < 0
            for value in (kl_loss_factor,)
        ):
            raise ValueError(
                "summary Indexer loss factors must be finite and non-negative"
            )
        # Monitoring is detached and does not change the KL objective.
        monitor_active = distill and monitor_topk_blocks > 0
        sparse_diagnostics = monitor_active and not dense_supervision
        checkpoint_dense_kl = (
            (distill and dense_supervision and (kl_loss_factor > 0) and (os.environ.get('SKSA_DENSE_WARMUP_CHECKPOINT', '1') == '1'))
        )
        custom_dense_kl = (
            checkpoint_dense_kl
            and os.environ.get("SKSA_DENSE_WARMUP_CUSTOM_BACKWARD", "1") == "1"
        )
        dense_cache = isinstance(recompute_cache, SKSASummaryIndexerDenseCheckpointCache)
        if monitor_active:
            if sparse_diagnostics and monitor_topk_blocks != topk_blocks:
                raise ValueError("sparse TopK diagnostics must match the actual route budget")
            if monitor_topk_blocks <= (sink_block_count if force_sink else 0):
                raise ValueError(
                    "summary TopK diagnostics require a TopK "
                    "larger than forced sinks"
                )
        elif monitor_topk_blocks < 0:
            raise ValueError("monitor TopK must be non-negative")
        if precomputed_dense_lse is not None and not (distill and dense_supervision):
            raise ValueError(
                "precomputed dense LSE requires all-candidate supervision"
            )
        if dense_cache:
            if not custom_dense_kl:
                raise ValueError("dense KL cache requires the custom KL-only path")
            expected_tiles = math.ceil(text_count / self.query_block_size)
            if (len(recompute_cache.tile_losses) != expected_tiles
                    or len(recompute_cache.tile_metrics) != expected_tiles):
                raise ValueError("S-KSA dense checkpoint cache tile count mismatch")
            if not dense_warmup and (
                recompute_cache.selected_block_indices is None
                or tuple(recompute_cache.selected_block_indices.shape)
                != (batch, text_count, topk_blocks)
            ):
                raise ValueError("S-KSA full-candidate checkpoint route cache shape mismatch")
        elif dense_supervision and recompute_cache is not None:
            raise ValueError("all-candidate supervision cannot consume a selected-only KL cache")
        if recompute_cache is not None and not dense_cache:
            expected_route_shape = (batch, text_count, topk_blocks)
            if tuple(recompute_cache.selected_block_indices.shape) != (
                expected_route_shape
            ):
                raise ValueError("S-KSA checkpoint route cache shape mismatch")
            expected_tiles = math.ceil(text_count / self.query_block_size)
            if (
                distill
                and len(recompute_cache.teacher_targets) != expected_tiles
            ):
                raise ValueError("S-KSA checkpoint teacher cache tile count mismatch")
            if (
                distill
                and len(recompute_cache.selected_kl_losses) != expected_tiles
            ):
                raise ValueError("S-KSA checkpoint KL cache tile count mismatch")

        detached_text = text_hidden.detach()
        detached_summary = (
            summary_hidden
            if preserve_summary_grad
            else summary_hidden.detach()
        )
        query = self.query_up_proj(self.query_down_proj(detached_text)).view(
            batch, text_count, self.indexer_heads, self.indexer_head_dim
        )
        query = _apply_partial_rope(
            query, text_position_ids, self.rope, self.rope_dim
        )
        key = self.key_proj(detached_summary).unsqueeze(2)
        key = _apply_partial_rope(
            key, summary_position_ids, self.rope, self.rope_dim
        ).squeeze(2)
        head_weight = self.head_weight_proj(detached_text).float()
        head_weight.mul_(self.indexer_heads ** -0.5)

        dense_lse = None
        if distill and (dense_supervision or (sparse_diagnostics and recompute_cache is None)):
            if dense_cache and recompute_cache.dense_lse is not None:
                precomputed_dense_lse = recompute_cache.dense_lse
            if precomputed_dense_lse is None:
                dense_lse = _dense_causal_lse(main_q.detach(), main_k.detach())
            else:
                expected_lse_shape = (
                    batch,
                    main_q.shape[2],
                    text_count,
                )
                if tuple(precomputed_dense_lse.shape) != expected_lse_shape:
                    raise ValueError("precomputed dense LSE shape mismatch")
                if (
                    precomputed_dense_lse.dtype != torch.float32
                    or precomputed_dense_lse.device != main_q.device
                ):
                    raise ValueError(
                        "precomputed dense LSE must be device-local float32"
                    )
                dense_lse = precomputed_dense_lse.detach()

        loss_sum: Optional[Tensor] = None
        kl_sum_total: Optional[Tensor] = None
        topk_recall_sum_total: Optional[Tensor] = None
        selected_attention_mass_sum_total: Optional[Tensor] = None
        oracle_topk_attention_mass_sum_total: Optional[Tensor] = None
        selected_oracle_attention_mass_ratio_sum_total: Optional[Tensor] = None
        topk_metric_row_count_total: Optional[Tensor] = None
        selected_tiles = []
        teacher_target_tiles = []
        selected_kl_loss_tiles = []
        dense_kl_loss_tiles = []
        dense_metric_tiles = []
        candidate_ids = torch.arange(summary_count, device=query.device)
        text_ids = torch.arange(text_count, device=query.device)
        # Sparse Full attention operates on raw-token blocks.  Align the
        # causal local-window boundary to that same block grid so the block
        # straddling an exact W-token boundary cannot develop a 1--(B-1)
        # token visibility hole.  Dividing the aligned start by B produces
        # the same complete-remote-block prefix as floor(raw_start / B), but
        # spelling the alignment out keeps routing/teacher semantics tied to
        # the sparse attention mask below.
        local_start = _aligned_local_window_start(
            text_ids,
            local_window_tokens=local_window_tokens,
            raw_block_size=summary_block_size,
        )
        eligible_count = torch.div(
            local_start, summary_block_size, rounding_mode="floor"
        ).clamp_max(summary_count)
        first_eligible_query = local_window_tokens + summary_block_size - 1
        valid_row_count = (
            batch * max(text_count - first_eligible_query, 0)
            if summary_count > 0 and (dense_supervision or topk_blocks > 0)
            else 0
        )
        score_scale = self.indexer_head_dim ** -0.5
        forced_sink_count = sink_block_count if force_sink else 0

        for tile_index, query_start in enumerate(
            range(0, text_count, self.query_block_size)
        ):
            query_end = min(text_count, query_start + self.query_block_size)
            query_tile = query[:, query_start:query_end]
            head_weight_tile = head_weight[:, query_start:query_end]
            selected = None
            if recompute_cache is not None and not dense_cache:
                # The original checkpointed forward already made this hard,
                # detached decision. Reusing it is exact and also ensures the
                # recomputation cannot diverge on a tied TopK boundary.
                selected = recompute_cache.selected_block_indices[
                    :, query_start:query_end
                ]
            else:
                # Eligibility is always a prefix of summary/raw-block IDs.  At
                # 128K, slicing off the impossible future suffix removes half
                # of all-candidate QK work on average without changing routes.
                tile_candidate_count = min(
                    summary_count,
                    max(0, query_end - local_window_tokens)
                    // summary_block_size,
                )
                if dense_supervision and tile_candidate_count:
                    # Bucket the differentiable warm-up GEMM so long-context
                    # backward sees a small set of efficient N dimensions.
                    candidate_bucket = 1024
                    tile_candidate_count = min(
                        summary_count,
                        math.ceil(tile_candidate_count / candidate_bucket)
                        * candidate_bucket,
                    )
                tile_candidate_ids = candidate_ids[:tile_candidate_count]
                tile_key = key[:, :tile_candidate_count]
                if not checkpoint_dense_kl or not dense_warmup:
                    valid = tile_candidate_ids[None, None, :] < eligible_count[
                        None, query_start:query_end, None
                    ]
                    valid = valid.expand(batch, -1, -1)
                if dense_supervision:
                    # Every eligible candidate participates in KL, independently
                    # of whether the main attention path is dense or sparse.
                    if not checkpoint_dense_kl:
                        scores = _score_all_indexer_candidates(
                            query_tile,
                            tile_key,
                            head_weight_tile,
                            score_scale,
                            scorer_backend=self.scorer_backend,
                        )
                if not dense_warmup and dense_cache:
                    selected = recompute_cache.selected_block_indices[:, query_start:query_end]
                elif not dense_warmup:
                    # Hard TopK is non-differentiable. Build its dense score
                    # matrix under no_grad, independently of KL supervision.
                    with torch.no_grad():
                        routing_scores = _score_all_indexer_candidates(
                            query_tile,
                            tile_key,
                            head_weight_tile,
                            score_scale,
                            scorer_backend=self.scorer_backend,
                        )
                        if sparse_diagnostics:
                            # Diagnostic-only full teacher. Run before TopK
                            # consumes scores; never retain a graph or reuse this
                            # distribution as the selected-only KL target.
                            diagnostic_teacher = dense_teacher_summary_block_distribution(
                                main_q.detach(), main_k.detach(), dense_lse,
                                query_start=query_start, query_end=query_end,
                                summary_block_size=summary_block_size,
                                summary_count=tile_candidate_count, valid=valid,
                                pooling=teacher_pooling,
                                local_window_tokens=local_window_tokens,
                            )
                            diagnostic_metrics = _dense_summary_topk_attention_mass_tile(
                                routing_scores, diagnostic_teacher,
                                eligible_count[query_start:query_end],
                                topk_blocks=topk_blocks,
                                sink_block_count=forced_sink_count,
                            )
                            accumulators = (
                                topk_recall_sum_total, selected_attention_mass_sum_total,
                                oracle_topk_attention_mass_sum_total,
                                selected_oracle_attention_mass_ratio_sum_total,
                                topk_metric_row_count_total,
                            )
                            (
                                topk_recall_sum_total, selected_attention_mass_sum_total,
                                oracle_topk_attention_mass_sum_total,
                                selected_oracle_attention_mass_ratio_sum_total,
                                topk_metric_row_count_total,
                            ) = tuple(value if total is None else total + value
                                      for total, value in zip(accumulators, diagnostic_metrics))
                            del diagnostic_teacher
                    selected = self._select_topk(
                        routing_scores,
                        valid,
                        topk_blocks,
                        force_sink,
                        sink_block_count=sink_block_count,
                        consume_scores=True,
                    )
                    selected_tiles.append(selected)
                    del routing_scores

            if checkpoint_dense_kl:
                tile_fn = partial(
                    _dense_indexer_kl_tile, query_start=query_start,
                    summary_block_size=summary_block_size,
                    local_window_tokens=local_window_tokens,
                    teacher_pooling=teacher_pooling,
                    teacher_query_block_size=teacher_query_block_size,
                    scorer_backend=self.scorer_backend,
                    diagnostic_topk=monitor_topk_blocks,
                    sink_block_count=forced_sink_count,
                )
                tile_args = (
                    query_tile, tile_key, head_weight_tile,
                    main_q.detach(), main_k.detach(), dense_lse,
                    eligible_count[query_start:query_end],
                )
                if custom_dense_kl:
                    cached = (
                        (recompute_cache.tile_losses[tile_index],
                         recompute_cache.tile_metrics[tile_index])
                        if dense_cache else None
                    )
                    tile_loss, tile_metrics = _DenseIndexerKLLoss.apply(
                        *tile_args, tile_fn, cached,
                    )
                    if recompute_cache is None:
                        dense_kl_loss_tiles.append(tile_loss.detach())
                        dense_metric_tiles.append(tile_metrics.detach())
                elif torch.is_grad_enabled():
                    tile_loss, tile_metrics = checkpoint(
                        tile_fn, *tile_args, use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    tile_loss, tile_metrics = tile_fn(*tile_args)
                kl_sum_total = (
                    tile_loss.detach() if kl_sum_total is None
                    else kl_sum_total + tile_loss.detach()
                )
                tile_objective = (
                    kl_loss_factor * tile_loss / max(float(valid_row_count), 1.)
                )
                loss_sum = (
                    tile_objective if loss_sum is None
                    else loss_sum + tile_objective
                )
                if monitor_active:
                    accumulators = (
                        topk_recall_sum_total, selected_attention_mass_sum_total,
                        oracle_topk_attention_mass_sum_total,
                        selected_oracle_attention_mass_ratio_sum_total,
                        topk_metric_row_count_total,
                    )
                    (
                        topk_recall_sum_total, selected_attention_mass_sum_total,
                        oracle_topk_attention_mass_sum_total,
                        selected_oracle_attention_mass_ratio_sum_total,
                        topk_metric_row_count_total,
                    ) = tuple(
                        value if total is None else total + value
                        for total, value in zip(accumulators, tile_metrics)
                    )
            elif distill and dense_supervision:
                # The CUDA teacher streams QK inside its kernel and only
                # materializes the compact block target.  Consume the whole
                # indexer tile at once; the smaller knob remains useful for
                # the PyTorch CPU/fp32 oracle, whose logits are explicit.
                dense_teacher_step = (
                    query_end - query_start
                    if main_q.is_cuda
                    and main_q.dtype in (torch.float16, torch.bfloat16)
                    else teacher_query_block_size
                )
                for teacher_start in range(
                    query_start, query_end, dense_teacher_step
                ):
                    teacher_end = min(
                        query_end, teacher_start + dense_teacher_step
                    )
                    offset_start = teacher_start - query_start
                    offset_end = teacher_end - query_start
                    teacher_valid = valid[:, offset_start:offset_end]
                    target = dense_teacher_summary_block_distribution(
                        main_q,
                        main_k,
                        dense_lse,
                        query_start=teacher_start,
                        query_end=teacher_end,
                        summary_block_size=summary_block_size,
                        summary_count=tile_candidate_count,
                        valid=teacher_valid,
                        pooling=teacher_pooling,
                        local_window_tokens=local_window_tokens,
                    )
                    if kl_loss_factor > 0:
                        tile_loss, _ = _masked_kl(
                            scores[:, offset_start:offset_end],
                            target,
                            teacher_valid,
                            return_count=False,
                        )
                    else:
                        # Rank-only continuation: KL is a detached diagnostic
                        # and must not retain a second long-context autograd
                        # graph or contribute a hidden gradient.
                        with torch.no_grad():
                            tile_loss, _ = _masked_kl(
                                scores[:, offset_start:offset_end].detach(),
                                target,
                                teacher_valid,
                                return_count=False,
                            )
                    if monitor_active:
                        (
                            tile_recall_sum,
                            tile_selected_mass_sum,
                            tile_oracle_mass_sum,
                            tile_mass_ratio_sum,
                            tile_metric_row_count,
                        ) = _dense_summary_topk_attention_mass_tile(
                            scores[:, offset_start:offset_end].detach(),
                            target,
                            eligible_count[teacher_start:teacher_end],
                            topk_blocks=monitor_topk_blocks,
                            sink_block_count=forced_sink_count,
                        )
                        topk_recall_sum_total = (
                            tile_recall_sum
                            if topk_recall_sum_total is None
                            else topk_recall_sum_total + tile_recall_sum
                        )
                        selected_attention_mass_sum_total = (
                            tile_selected_mass_sum
                            if selected_attention_mass_sum_total is None
                            else selected_attention_mass_sum_total
                            + tile_selected_mass_sum
                        )
                        oracle_topk_attention_mass_sum_total = (
                            tile_oracle_mass_sum
                            if oracle_topk_attention_mass_sum_total is None
                            else oracle_topk_attention_mass_sum_total
                            + tile_oracle_mass_sum
                        )
                        selected_oracle_attention_mass_ratio_sum_total = (
                            tile_mass_ratio_sum
                            if selected_oracle_attention_mass_ratio_sum_total
                            is None
                            else selected_oracle_attention_mass_ratio_sum_total
                            + tile_mass_ratio_sum
                        )
                        topk_metric_row_count_total = (
                            tile_metric_row_count
                            if topk_metric_row_count_total is None
                            else topk_metric_row_count_total
                            + tile_metric_row_count
                        )
                    kl_sum_total = (
                        tile_loss.detach()
                        if kl_sum_total is None
                        else kl_sum_total + tile_loss.detach()
                    )
                    tile_objective = (
                        kl_loss_factor
                        * tile_loss
                        / max(float(valid_row_count), 1.0)
                    )
                    loss_sum = (
                        tile_objective
                        if loss_sum is None
                        else loss_sum + tile_objective
                    )
            elif distill:
                if selected is None:
                    raise RuntimeError("selected-set distillation requires hard routes")
                if recompute_cache is not None:
                    target = recompute_cache.teacher_targets[tile_index]
                    selected_valid = selected >= 0
                else:
                    # The teacher is detached. Cache the compact target when
                    # an enclosing layer checkpoint asks for recomputation.
                    target, selected_valid = (
                        selected_teacher_summary_block_distribution(
                            main_q[:, query_start:query_end],
                            main_k,
                            selected,
                            summary_block_size=summary_block_size,
                            pooling=teacher_pooling,
                            all_rows_have_candidates=(
                                summary_count > 0
                                and topk_blocks > 0
                                and query_start
                                >= local_window_tokens
                                + summary_block_size
                                - 1
                            ),
                        )
                    )
                    teacher_target_tiles.append(target)
                cached_tile_loss = (
                    recompute_cache.selected_kl_losses[tile_index]
                    if recompute_cache is not None
                    else None
                )
                tile_loss = _selected_indexer_kl_loss(
                    query_tile,
                    key,
                    head_weight_tile,
                    selected,
                    target,
                    selected_valid,
                    score_scale,
                    cached_loss=cached_tile_loss,
                    scorer_backend=self.scorer_backend,
                )
                if recompute_cache is None:
                    selected_kl_loss_tiles.append(tile_loss.detach())
                loss_sum = (
                    tile_loss if loss_sum is None else loss_sum + tile_loss
                )

        if recompute_cache is not None and (not dense_cache or not dense_warmup):
            selected_indices = recompute_cache.selected_block_indices
        else:
            selected_indices = (
                torch.cat(selected_tiles, dim=1)
                if selected_tiles
                else torch.full(
                    (batch, text_count, 0),
                    -1,
                    dtype=torch.int32,
                    device=query.device,
                )
            )
        checkpoint_cache = None
        sparse_topk_metrics = None
        if sparse_diagnostics:
            if recompute_cache is not None:
                if recompute_cache.topk_metrics is None:
                    raise ValueError("sparse checkpoint cache is missing TopK diagnostics")
                (
                    topk_recall_sum_total, selected_attention_mass_sum_total,
                    oracle_topk_attention_mass_sum_total,
                    selected_oracle_attention_mass_ratio_sum_total,
                    topk_metric_row_count_total,
                ) = recompute_cache.topk_metrics
            sparse_topk_metrics = (
                topk_recall_sum_total, selected_attention_mass_sum_total,
                oracle_topk_attention_mass_sum_total,
                selected_oracle_attention_mass_ratio_sum_total,
                topk_metric_row_count_total,
            )
        if custom_dense_kl and recompute_cache is None:
            checkpoint_cache = SKSASummaryIndexerDenseCheckpointCache(
                tile_losses=tuple(dense_kl_loss_tiles),
                tile_metrics=tuple(dense_metric_tiles),
                selected_block_indices=(selected_indices.detach() if not dense_warmup else None),
                dense_lse=dense_lse.detach(),
            )
        elif not dense_supervision and recompute_cache is None:
            checkpoint_cache = SKSASummaryIndexerCheckpointCache(
                selected_block_indices=selected_indices.detach(),
                teacher_targets=tuple(
                    target.detach() for target in teacher_target_tiles
                ),
                selected_kl_losses=tuple(selected_kl_loss_tiles),
                topk_metrics=sparse_topk_metrics,
            )
        indexer_loss = None
        kl_metric = None
        oracle_topk_recall_metric = None
        selected_attention_mass_metric = None
        oracle_topk_attention_mass_metric = None
        selected_oracle_attention_mass_ratio_metric = None
        if distill:
            if loss_sum is None:
                # Preserve the old differentiable-zero behavior for tiny
                # no-candidate inputs without reducing the full query tensor.
                loss_sum = query.reshape(-1)[:1].sum() * 0.0
            if dense_supervision:
                indexer_loss = loss_sum
                if kl_sum_total is None:
                    kl_sum_total = query.reshape(-1)[:1].sum() * 0.0
                kl_metric = (
                    kl_sum_total / max(float(valid_row_count), 1.0)
                ).detach()
            else:
                raw_kl = loss_sum / max(float(valid_row_count), 1.0)
                indexer_loss = kl_loss_factor * raw_kl
                kl_metric = raw_kl.detach()
            if monitor_active:
                zero_metric = query.reshape(-1)[:1].sum() * 0.0
                if topk_recall_sum_total is None:
                    topk_recall_sum_total = zero_metric
                if selected_attention_mass_sum_total is None:
                    selected_attention_mass_sum_total = zero_metric
                if oracle_topk_attention_mass_sum_total is None:
                    oracle_topk_attention_mass_sum_total = zero_metric
                if selected_oracle_attention_mass_ratio_sum_total is None:
                    selected_oracle_attention_mass_ratio_sum_total = zero_metric
                if topk_metric_row_count_total is None:
                    topk_metric_row_count_total = zero_metric
                metric_denominator = topk_metric_row_count_total.clamp_min(1.0)
                oracle_topk_recall_metric = (
                    topk_recall_sum_total / metric_denominator
                ).detach()
                selected_attention_mass_metric = (
                    selected_attention_mass_sum_total / metric_denominator
                ).detach()
                oracle_topk_attention_mass_metric = (
                    oracle_topk_attention_mass_sum_total / metric_denominator
                ).detach()
                selected_oracle_attention_mass_ratio_metric = (
                    selected_oracle_attention_mass_ratio_sum_total
                    / metric_denominator
                ).detach()
                # Preserve the old in-memory result contract for callers
                # that have not migrated their diagnostics yet.  The
                # recipe deliberately does not expose this as a TB series.
        return SKSASummaryIndexerResult(
            selected_block_indices=selected_indices,
            indexer_loss=indexer_loss,
            checkpoint_cache=checkpoint_cache,
            kl_loss=kl_metric,
            oracle_topk_recall=oracle_topk_recall_metric,
            selected_attention_mass=selected_attention_mass_metric,
            oracle_topk_attention_mass=oracle_topk_attention_mass_metric,
            selected_oracle_attention_mass_ratio=(
                selected_oracle_attention_mass_ratio_metric
            ),
        )


def _split_regular_layout(
    tensor: Tensor,
    summary_mask: Tensor,
    *,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_layout_signature: Optional[Tuple[int, bool]] = None,
    summary_layout_validated: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return text, summary and chunk views for the regular inserted layout."""
    if summary_token_num != 1:
        raise ValueError("summary-indexed S-KSA currently requires one summary per block")
    batch, expanded = tensor.shape[:2]
    chunk_width = summary_chunk_size + summary_token_num
    if expanded % chunk_width:
        raise ValueError("summary-indexed S-KSA requires a complete summary layout")
    if tuple(summary_mask.shape) != (batch, expanded):
        raise ValueError("summary-indexed S-KSA requires the regular summary layout")
    chunk_count = expanded // chunk_width
    expected_summary_count = chunk_count * summary_token_num
    if summary_layout_validated:
        if summary_layout_signature is None:
            raise ValueError(
                "validated summary layout requires its static signature"
            )
        summary_count, all_chunks_summarized = summary_layout_signature
        if (
            not all_chunks_summarized
            or summary_count != expected_summary_count
        ):
            raise ValueError(
                "validated summary layout signature does not match tensor shape"
            )
    else:
        expected = torch.cat((
            torch.zeros(
                summary_chunk_size, dtype=torch.bool, device=tensor.device
            ),
            torch.ones(
                summary_token_num, dtype=torch.bool, device=tensor.device
            ),
        )).repeat(chunk_count)
        if not torch.all(summary_mask == expected[None]):
            raise ValueError(
                "summary-indexed S-KSA requires the regular summary layout"
            )
    chunks = tensor.view(batch, -1, chunk_width, *tensor.shape[2:])
    text = chunks[:, :, :summary_chunk_size].reshape(
        batch, -1, *tensor.shape[2:]
    )
    summary = chunks[:, :, summary_chunk_size:].reshape(
        batch, -1, *tensor.shape[2:]
    )
    return text, summary, chunks


def split_summary_indexer_inputs(
    hidden: Tensor,
    position_ids: Tensor,
    summary_mask: Tensor,
    *,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_layout_signature: Optional[Tuple[int, bool]] = None,
    summary_layout_validated: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extract direct Full-layer text/summary hidden states and positions."""
    text_hidden, summary_hidden, _ = _split_regular_layout(
        hidden,
        summary_mask,
        summary_chunk_size=summary_chunk_size,
        summary_token_num=summary_token_num,
        summary_layout_signature=summary_layout_signature,
        summary_layout_validated=summary_layout_validated,
    )
    # The same mask/layout applies to positions.  Once the hidden-state split
    # has validated it, repeating a CUDA-wide mask reduction would only add a
    # host synchronization.
    trusted_signature = (
        summary_layout_signature
        if summary_layout_validated
        else (
            hidden.shape[1]
            // (summary_chunk_size + summary_token_num)
            * summary_token_num,
            True,
        )
    )
    text_pos, summary_pos, _ = _split_regular_layout(
        position_ids,
        summary_mask,
        summary_chunk_size=summary_chunk_size,
        summary_token_num=summary_token_num,
        summary_layout_signature=trusted_signature,
        summary_layout_validated=True,
    )
    return text_hidden, summary_hidden, text_pos, summary_pos


def _summary_chunk_local_attention(
    q_chunks: Tensor,
    k_chunks: Tensor,
    v_chunks: Tensor,
    *,
    summary_chunk_size: int,
) -> Tensor:
    """Preserve the baseline chunk-local output of inserted summary queries."""
    q_summary = q_chunks[:, :, summary_chunk_size:]
    logits = torch.einsum("bnthd,bnshd->bnhts", q_summary, k_chunks).float()
    logits.mul_(q_chunks.shape[-1] ** -0.5)
    probability = torch.softmax(logits, dim=-1)
    return torch.einsum(
        "bnhts,bnshd->bnthd", probability.to(v_chunks.dtype), v_chunks
    )


def _aligned_local_window_start(
    query_ids: Tensor,
    *,
    local_window_tokens: int,
    raw_block_size: int,
) -> Tensor:
    """Return the causal local-window start aligned down to a raw block.

    The effective local span is therefore in
    ``[local_window_tokens, local_window_tokens + raw_block_size - 1]`` once
    the sequence is long enough.  This is intentional: remote routing is
    block-granular, so retaining an exact token-granular left boundary would
    otherwise hide the prefix of the block that straddles that boundary.
    """
    if local_window_tokens <= 0:
        raise ValueError("S-KSA local window must be positive")
    if raw_block_size <= 0:
        raise ValueError("S-KSA raw block size must be positive")
    unaligned = (query_ids - local_window_tokens + 1).clamp_min(0)
    return torch.div(
        unaligned, raw_block_size, rounding_mode="floor"
    ) * raw_block_size


def _summary_block_sparse_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    *,
    block_size: int,
    local_window_tokens: int,
) -> Tensor:
    """Small exact raw-token oracle used by CPU and unit tests."""
    batch, text_count, heads, head_dim = q.shape
    raw_blocks = math.ceil(text_count / block_size)
    route_count = torch.zeros(
        (batch, text_count, raw_blocks), dtype=torch.int32, device=q.device
    )
    valid = selected >= 0
    route_count.scatter_add_(
        2, selected.clamp_min(0).long(), valid.to(torch.int32)
    )
    route = route_count > 0
    key_ids = torch.arange(text_count, device=q.device)
    query_ids = torch.arange(text_count, device=q.device)
    selected_visible = route[:, :, key_ids // block_size]
    causal = key_ids[None, :] <= query_ids[:, None]
    aligned_local_start = _aligned_local_window_start(
        query_ids,
        local_window_tokens=local_window_tokens,
        raw_block_size=block_size,
    )
    local = key_ids[None, :] >= aligned_local_start[:, None]
    visibility = causal[None] & (local[None] | selected_visible)
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float())
    logits.mul_(head_dim ** -0.5)
    probability = torch.softmax(
        logits.masked_fill(~visibility[:, None], float("-inf")), dim=-1
    )
    return torch.einsum("bhqk,bkhd->bqhd", probability.to(v.dtype), v)


def sksa_summary_indexer_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    summary_mask: Tensor,
    selected_block_indices: Tensor,
    *,
    summary_chunk_size: int,
    summary_token_num: int,
    local_window_tokens: int,
    indexer_block_size: Optional[int] = None,
    summary_layout_signature: Optional[Tuple[int, bool]] = None,
    summary_layout_validated: bool = False,
    precomputed_q_text: Optional[Tensor] = None,
    precomputed_k_text: Optional[Tensor] = None,
) -> Tensor:
    """Apply per-query sparse Full attention selected through summary keys."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("S-KSA Full q/k/v must share [B,expanded,H,D] shape")
    chunk_width = summary_chunk_size + summary_token_num
    raw_block_size = indexer_block_size or summary_chunk_size
    text_count = q.shape[1] // chunk_width * summary_chunk_size
    if (precomputed_q_text is None) != (precomputed_k_text is None):
        raise ValueError("precomputed S-KSA text Q/K must be provided together")
    if precomputed_q_text is None:
        q_text, _, q_chunks = _split_regular_layout(
            q,
            summary_mask,
            summary_chunk_size=summary_chunk_size,
            summary_token_num=summary_token_num,
            summary_layout_signature=summary_layout_signature,
            summary_layout_validated=summary_layout_validated,
        )
        trusted_signature = (
            summary_layout_signature
            if summary_layout_validated
            else (
                q.shape[1] // chunk_width * summary_token_num,
                True,
            )
        )
        k_text, _, k_chunks = _split_regular_layout(
            k,
            summary_mask,
            summary_chunk_size=summary_chunk_size,
            summary_token_num=summary_token_num,
            summary_layout_signature=trusted_signature,
            summary_layout_validated=True,
        )
    else:
        if not summary_layout_validated:
            raise ValueError(
                "precomputed S-KSA text Q/K require a validated summary layout"
            )
        # Q/K text copies were already made for the layer-local indexer teacher.
        # Keep their existing lifetime and only recover the cheap expanded-layout
        # chunk views needed by summary-query attention.
        expected_shape = (q.shape[0], text_count, *q.shape[2:])
        if (
            tuple(precomputed_q_text.shape) != expected_shape
            or tuple(precomputed_k_text.shape) != expected_shape
            or precomputed_q_text.dtype != q.dtype
            or precomputed_k_text.dtype != k.dtype
            or precomputed_q_text.device != q.device
            or precomputed_k_text.device != k.device
        ):
            raise ValueError("precomputed S-KSA text Q/K do not match Full Q/K")
        q_text = precomputed_q_text
        k_text = precomputed_k_text
        q_chunks = q.view(q.shape[0], -1, chunk_width, *q.shape[2:])
        k_chunks = k.view(k.shape[0], -1, chunk_width, *k.shape[2:])
        trusted_signature = summary_layout_signature
    v_text, _, v_chunks = _split_regular_layout(
        v,
        summary_mask,
        summary_chunk_size=summary_chunk_size,
        summary_token_num=summary_token_num,
        summary_layout_signature=trusted_signature,
        summary_layout_validated=True,
    )
    q_text = q_text.contiguous()
    k_text = k_text.contiguous()
    v_text = v_text.contiguous()
    backend = os.environ.get("SKSA_LIGHTNING_BACKEND", "cute").lower()
    if q.is_cuda and backend == "cute":
        try:
            from ksa_attn import summary_block_sparse_attn_func
        except ImportError as exc:
            raise RuntimeError(
                "S-KSA Lightning training requires kernel/v0.1 to be installed"
            ) from exc
        text_output = summary_block_sparse_attn_func(
            q_text,
            k_text,
            v_text,
            selected_block_indices,
            raw_block_size,
            local_window_tokens,
        )
    elif backend in {"reference", "torch"} or not q.is_cuda:
        if q_text.shape[1] > 4096:
            raise RuntimeError(
                "The S-KSA Lightning torch backend is a small-sequence oracle; "
                "install/use the CuTe backend for long sequences"
            )
        text_output = _summary_block_sparse_reference(
            q_text,
            k_text,
            v_text,
            selected_block_indices,
            block_size=raw_block_size,
            local_window_tokens=local_window_tokens,
        )
    else:
        raise ValueError("SKSA_LIGHTNING_BACKEND must be 'cute' or 'reference'")

    summary_output = _summary_chunk_local_attention(
        q_chunks,
        k_chunks,
        v_chunks,
        summary_chunk_size=summary_chunk_size,
    )
    output_chunks = torch.empty_like(q_chunks)
    output_chunks[:, :, :summary_chunk_size] = text_output.view_as(
        output_chunks[:, :, :summary_chunk_size]
    )
    output_chunks[:, :, summary_chunk_size:] = summary_output
    return output_chunks.reshape_as(q)
