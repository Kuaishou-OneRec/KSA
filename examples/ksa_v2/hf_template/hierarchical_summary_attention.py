"""Reference prefill attention for hierarchical KSA summary pooling.

This module intentionally uses PyTorch SDPA over compact per-query-block KV sets.
It is the correctness path used before the fused CUDA kernel grows support for
meta-summary KV entries and their additive logit bias.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


_LAYOUT_CACHE: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
_SUMMARY_LAYOUT_CACHE: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}


def _summary_layout(
    *,
    seq_len: int,
    q_start: int,
    q_end: int,
    chunk_size: int,
    sliding_chunk_num: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return compact base-KV positions and the original-KSA block mask."""
    block = chunk_size + 1
    q_pos = torch.arange(q_start, q_end, device=device, dtype=torch.long)
    q_chunk = torch.div(q_pos, block, rounding_mode="floor")
    q_offset = q_pos.remainder(block)
    q_is_text = q_offset < chunk_size
    window_start = torch.clamp(q_chunk - sliding_chunk_num, min=0)

    candidates = []
    if bool(q_is_text.any()):
        old_summary_count = int(window_start[q_is_text].max().item())
        if old_summary_count:
            candidates.append(
                torch.arange(old_summary_count, device=device) * block + chunk_size
            )
        first_text_chunk = int(window_start[q_is_text].min().item())
    else:
        first_text_chunk = int(q_chunk.min().item())

    last_text_chunk = int(q_chunk.max().item())
    text_chunks = torch.arange(
        first_text_chunk,
        last_text_chunk + 1,
        device=device,
        dtype=torch.long,
    )
    text_offsets = torch.arange(chunk_size, device=device, dtype=torch.long)
    text_candidates = (text_chunks[:, None] * block + text_offsets).reshape(-1)
    candidates.append(text_candidates[text_candidates < seq_len])

    if bool((~q_is_text).any()):
        current_summary_chunks = torch.unique(q_chunk[~q_is_text], sorted=True)
        candidates.append(current_summary_chunks * block + chunk_size)

    base_pos = torch.unique(torch.cat(candidates), sorted=True)
    k_chunk = torch.div(base_pos, block, rounding_mode="floor")
    k_offset = base_pos.remainder(block)
    k_is_text = k_offset < chunk_size
    k_is_summary = k_offset == chunk_size

    qc = q_chunk[:, None]
    qp = q_pos[:, None]
    ws = window_start[:, None]
    kc = k_chunk[None, :]
    kp = base_pos[None, :]
    kt = k_is_text[None, :]
    ks = k_is_summary[None, :]
    text_allowed = (ks & (kc < ws)) | (
        kt & (((kc >= ws) & (kc < qc)) | ((kc == qc) & (kp <= qp)))
    )
    summary_allowed = (kc == qc) & (kt | (ks & (kp <= qp)))
    allowed = torch.where(q_is_text[:, None], text_allowed, summary_allowed)
    used = allowed.any(dim=0)
    return base_pos[used], allowed[:, used]


def summary_attention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    summary_mask: torch.Tensor,
    *,
    chunk_size: int,
    sliding_chunk_num: int,
    block_size: int = 512,
) -> torch.Tensor:
    """Pure-PyTorch batch-uniform prefill for original (unpooled) KSA.

    This is the portable correctness path used when the optional ``ksa_attn``
    package is unavailable or explicitly disabled. Inputs use
    ``[batch, heads, sequence, head_dim]`` layout; the output uses
    ``[batch, sequence, query_heads, head_dim]``.
    """
    if chunk_size <= 0 or sliding_chunk_num <= 0 or block_size <= 0:
        raise ValueError("chunk_size, sliding_chunk_num, and block_size must be positive")
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("query/key/value states must be rank-4 tensors")

    batch, query_heads, seq_len, head_dim = query_states.shape
    remainder = seq_len % (chunk_size + 1)
    if remainder > chunk_size:
        raise ValueError("invalid deferred partial-chunk layout")
    if key_states.shape[:1] != (batch,) or key_states.shape[2:] != (seq_len, head_dim):
        raise ValueError("key_states shape is incompatible with query_states")
    if value_states.shape != key_states.shape:
        raise ValueError("key_states and value_states must have identical shapes")
    kv_heads = key_states.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")

    if summary_mask.ndim == 2:
        if summary_mask.shape != (batch, seq_len):
            raise ValueError("summary_mask shape is incompatible with query_states")
        if not torch.all(summary_mask == summary_mask[:1]):
            raise ValueError("reference prefill requires a batch-uniform summary mask")
        shared_summary_mask = summary_mask[0]
    elif summary_mask.ndim == 1 and summary_mask.shape[0] == seq_len:
        shared_summary_mask = summary_mask
    else:
        raise ValueError("summary_mask must have shape [sequence] or [batch, sequence]")
    expected_summary_mask = torch.zeros(seq_len, dtype=torch.bool, device=query_states.device)
    expected_summary_mask[chunk_size:: chunk_size + 1] = True
    if not torch.equal(shared_summary_mask.to(dtype=torch.bool), expected_summary_mask):
        raise ValueError("summary_mask does not match the configured chunk layout")

    repeat_factor = query_heads // kv_heads
    output = torch.empty_like(query_states)
    for q_start in range(0, seq_len, block_size):
        q_end = min(seq_len, q_start + block_size)
        cache_key = (
            seq_len,
            q_start,
            q_end,
            chunk_size,
            sliding_chunk_num,
            device_key(query_states.device),
        )
        cached = _SUMMARY_LAYOUT_CACHE.get(cache_key)
        if cached is None:
            base_pos, allowed = _summary_layout(
                seq_len=seq_len,
                q_start=q_start,
                q_end=q_end,
                chunk_size=chunk_size,
                sliding_chunk_num=sliding_chunk_num,
                device=query_states.device,
            )
            _SUMMARY_LAYOUT_CACHE[cache_key] = (base_pos, allowed)
        else:
            base_pos, allowed = cached

        compact_k = key_states[:, :, base_pos, :]
        compact_v = value_states[:, :, base_pos, :]
        if repeat_factor != 1:
            compact_k = compact_k.repeat_interleave(repeat_factor, dim=1)
            compact_v = compact_v.repeat_interleave(repeat_factor, dim=1)
        output[:, :, q_start:q_end, :] = F.scaled_dot_product_attention(
            query_states[:, :, q_start:q_end, :],
            compact_k,
            compact_v,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
        )
    return output.transpose(1, 2).contiguous()


def _compact_layout(
    *,
    seq_len: int,
    q_start: int,
    q_end: int,
    chunk_size: int,
    sliding_chunk_num: int,
    group_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return compact base-KV positions and an additive [Q, meta+base] mask."""
    block = chunk_size + 1
    q_pos = torch.arange(q_start, q_end, device=device, dtype=torch.long)
    q_chunk = torch.div(q_pos, block, rounding_mode="floor")
    q_offset = q_pos.remainder(block)
    q_is_text = q_offset < chunk_size
    window_start = torch.clamp(q_chunk - sliding_chunk_num, min=0)
    complete_group_count = torch.div(window_start, group_size, rounding_mode="floor")
    complete_end = complete_group_count * group_size

    # Only a local interval of original K/V can remain visible. Older history is
    # represented exclusively by meta KV, so it is omitted from the compact set.
    if bool(q_is_text.any()):
        first_chunk = int(complete_end[q_is_text].min().item())
    else:
        first_chunk = int(q_chunk.min().item())
    base_start = first_chunk * block
    base_pos = torch.arange(base_start, q_end, device=device, dtype=torch.long)
    k_chunk = torch.div(base_pos, block, rounding_mode="floor")
    k_offset = base_pos.remainder(block)
    k_is_text = k_offset < chunk_size
    k_is_summary = k_offset == chunk_size

    qc = q_chunk[:, None]
    qp = q_pos[:, None]
    ws = window_start[:, None]
    ce = complete_end[:, None]
    kc = k_chunk[None, :]
    kp = base_pos[None, :]
    kt = k_is_text[None, :]
    ks = k_is_summary[None, :]

    text_to_raw = kt & (((kc >= ws) & (kc < qc)) | ((kc == qc) & (kp <= qp)))
    text_to_residual = ks & (kc >= ce) & (kc < ws)
    summary_to_current = (kc == qc) & (kt | (ks & (kp <= qp)))
    base_allowed = torch.where(
        q_is_text[:, None],
        text_to_raw | text_to_residual,
        summary_to_current,
    )

    max_meta = int(complete_group_count[q_is_text].max().item()) if bool(q_is_text.any()) else 0
    if max_meta:
        meta_idx = torch.arange(max_meta, device=device, dtype=torch.long)
        meta_allowed = q_is_text[:, None] & (
            meta_idx[None, :] < complete_group_count[:, None]
        )
        allowed = torch.cat((meta_allowed, base_allowed), dim=1)
    else:
        meta_idx = torch.empty(0, device=device, dtype=torch.long)
        allowed = base_allowed

    return torch.cat((meta_idx, base_pos)), allowed


def hierarchical_summary_attention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    meta_key_states: torch.Tensor,
    meta_value_states: torch.Tensor,
    *,
    chunk_size: int,
    sliding_chunk_num: int,
    group_size: int,
    block_size: int = 512,
    mass_correction: bool = True,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute hierarchical KSA attention for a batch-uniform prefill.

    Args:
        query_states: [batch, query_heads, seq, head_dim]
        key_states/value_states: [batch, kv_heads, seq, head_dim]
        meta_key_states/meta_value_states: [batch, kv_heads, groups, head_dim]

    Returns:
        Attention output in [batch, seq, query_heads, head_dim] layout.  When
        ``return_lse`` is true, also returns fp32 LSE in [batch, heads, seq].
    """
    if group_size <= 1:
        raise ValueError(f"group_size must be greater than one, got {group_size}")
    if chunk_size <= 0 or sliding_chunk_num <= 0 or block_size <= 0:
        raise ValueError("chunk_size, sliding_chunk_num, and block_size must be positive")
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("query/key/value states must be rank-4 tensors")

    batch, query_heads, seq_len, head_dim = query_states.shape
    if key_states.shape[:1] != (batch,) or key_states.shape[2:] != (seq_len, head_dim):
        raise ValueError("key_states shape is incompatible with query_states")
    if value_states.shape != key_states.shape:
        raise ValueError("key_states and value_states must have identical shapes")
    kv_heads = key_states.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    if meta_key_states.shape != meta_value_states.shape:
        raise ValueError("meta key/value shapes must match")
    if meta_key_states.shape[:2] != (batch, kv_heads) or meta_key_states.shape[3] != head_dim:
        raise ValueError("meta key/value shape is incompatible with base key/value")

    repeat_factor = query_heads // kv_heads
    output = torch.empty_like(query_states)
    softmax_lse = (
        torch.empty(
            batch, query_heads, seq_len,
            dtype=torch.float32, device=query_states.device,
        )
        if return_lse else None
    )
    log_group_size = math.log(group_size) if mass_correction else 0.0

    for q_start in range(0, seq_len, block_size):
        q_end = min(seq_len, q_start + block_size)
        cache_key = (
            seq_len,
            q_start,
            q_end,
            chunk_size,
            sliding_chunk_num,
            group_size,
            device_key(query_states.device),
        )
        cached = _LAYOUT_CACHE.get(cache_key)
        if cached is None:
            packed_pos, allowed = _compact_layout(
                seq_len=seq_len,
                q_start=q_start,
                q_end=q_end,
                chunk_size=chunk_size,
                sliding_chunk_num=sliding_chunk_num,
                group_size=group_size,
                device=query_states.device,
            )
            _LAYOUT_CACHE[cache_key] = (packed_pos, allowed)
        else:
            packed_pos, allowed = cached

        # packed_pos stores meta indices first and sequence positions second.
        q_pos = torch.arange(q_start, q_end, device=query_states.device)
        q_chunk = torch.div(q_pos, chunk_size + 1, rounding_mode="floor")
        q_is_text = q_pos.remainder(chunk_size + 1) < chunk_size
        window_start = torch.clamp(q_chunk - sliding_chunk_num, min=0)
        meta_len = (
            int(torch.div(window_start[q_is_text], group_size, rounding_mode="floor").max().item())
            if bool(q_is_text.any())
            else 0
        )
        meta_idx = packed_pos[:meta_len]
        base_pos = packed_pos[meta_len:]

        base_k = key_states[:, :, base_pos, :]
        base_v = value_states[:, :, base_pos, :]
        if meta_len:
            compact_k = torch.cat((meta_key_states[:, :, meta_idx, :], base_k), dim=2)
            compact_v = torch.cat((meta_value_states[:, :, meta_idx, :], base_v), dim=2)
        else:
            compact_k, compact_v = base_k, base_v
        if repeat_factor != 1:
            compact_k = compact_k.repeat_interleave(repeat_factor, dim=1)
            compact_v = compact_v.repeat_interleave(repeat_factor, dim=1)

        additive_mask = torch.full(
            allowed.shape,
            float("-inf"),
            dtype=query_states.dtype,
            device=query_states.device,
        )
        additive_mask.masked_fill_(allowed, 0.0)
        if meta_len and log_group_size:
            additive_mask[:, :meta_len].masked_fill_(allowed[:, :meta_len], log_group_size)

        output[:, :, q_start:q_end, :] = F.scaled_dot_product_attention(
            query_states[:, :, q_start:q_end, :],
            compact_k,
            compact_v,
            attn_mask=additive_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        if return_lse:
            logits = torch.einsum(
                "bhqd,bhkd->bhqk",
                query_states[:, :, q_start:q_end, :],
                compact_k,
            ).float()
            logits.mul_(head_dim ** -0.5)
            logits.add_(additive_mask.float()[None, None])
            softmax_lse[:, :, q_start:q_end] = torch.logsumexp(logits, dim=-1)

    output = output.transpose(1, 2).contiguous()
    return (output, softmax_lse) if return_lse else output


def device_key(device: torch.device) -> tuple:
    return device.type, device.index
