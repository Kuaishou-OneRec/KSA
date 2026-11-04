"""
Summary Attention Context Data Structures.

Ported from kai-megatron/megatron/core/models/qwen3_next/summary_context.py,
adapted for muse's [b, s] tensor layout. No Megatron dependencies.

Provides:
- SummaryChunkMeta: per-chunk position metadata
- SummarySampleContext: per-sample chunk list
- SummaryBatchContext: batch-level container passed through the model
- Cache utilities for chunk positions (debugging/testing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import torch


# ---------------------------------------------------------------------------
# Global cache for attention masks and chunk positions
# Key: (seq_len, chunk_size, summary_num, sliding_num, device_type, device_index)
# ---------------------------------------------------------------------------

_SUMMARY_ATTENTION_CACHE: Dict[Tuple, Dict] = {}


def get_summary_attention_cache() -> Dict[Tuple, Dict]:
    """Get the global attention mask cache."""
    return _SUMMARY_ATTENTION_CACHE


def clear_summary_attention_cache():
    """Clear the global attention mask cache."""
    global _SUMMARY_ATTENTION_CACHE
    _SUMMARY_ATTENTION_CACHE.clear()


def get_or_create_cached_mask(
    seq_len: int,
    summary_chunk_size: int,
    summary_num: int,
    summary_sliding_chunk_num: int,
    device: torch.device,
    build_mask_fn,
) -> torch.Tensor:
    """Get cached mask or create and cache it if not exists."""
    cache_key = (
        seq_len,
        summary_chunk_size,
        summary_num,
        summary_sliding_chunk_num,
        device.type,
        getattr(device, 'index', None),
    )

    if cache_key not in _SUMMARY_ATTENTION_CACHE:
        _SUMMARY_ATTENTION_CACHE[cache_key] = {}

    cache = _SUMMARY_ATTENTION_CACHE[cache_key]
    if 'mask' not in cache:
        cache['mask'] = build_mask_fn(
            seq_len,
            summary_chunk_size=summary_chunk_size,
            summary_num=summary_num,
            summary_sliding_chunk_num=summary_sliding_chunk_num,
            device=device,
        )

    return cache['mask']


def get_or_create_chunk_positions(
    seq_len: int,
    summary_chunk_size: int,
    summary_num: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Get cached chunk positions or create and cache them if not exists."""
    cache_key = (
        seq_len,
        summary_chunk_size,
        summary_num,
        0,  # sliding_num not used for positions
        device.type,
        getattr(device, 'index', None),
    )

    if cache_key not in _SUMMARY_ATTENTION_CACHE:
        _SUMMARY_ATTENTION_CACHE[cache_key] = {}

    cache = _SUMMARY_ATTENTION_CACHE[cache_key]
    if 'text_positions' not in cache or 'summary_positions' not in cache:
        block = summary_chunk_size + summary_num
        chunks = seq_len // block
        text_positions = []
        summary_positions = []
        for c in range(chunks):
            start = c * block
            text_positions.append(torch.arange(start, start + summary_chunk_size, device=device))
            summary_positions.append(torch.arange(start + summary_chunk_size, start + block, device=device))
        cache['text_positions'] = text_positions
        cache['summary_positions'] = summary_positions

    return cache['text_positions'], cache['summary_positions']


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SummaryChunkMeta:
    """Metadata describing a single chunk within a sample."""

    text_positions: torch.Tensor
    summary_positions: torch.Tensor
    prefix_summary_positions: torch.Tensor

    @property
    def window_positions(self) -> torch.Tensor:
        if self.prefix_summary_positions.numel() == 0:
            if self.summary_positions.numel() == 0:
                return self.text_positions
            return torch.cat((self.text_positions, self.summary_positions), dim=0)
        if self.summary_positions.numel() == 0:
            return torch.cat((self.prefix_summary_positions, self.text_positions), dim=0)
        return torch.cat(
            (self.prefix_summary_positions, self.text_positions, self.summary_positions),
            dim=0,
        )


@dataclass
class SummarySampleContext:
    """Holds chunk metadata for a single sample/batch element."""

    chunks: List[SummaryChunkMeta]


@dataclass
class SummaryBatchContext:
    """Container passed to the model so layers can access summary layout.

    Tensor shapes use muse's [b, s] convention.

    Attributes:
        samples: per-sample chunk metadata (for debugging / future chunk-loop mode).
        position_ids: [b, local_seq_len] — RoPE position IDs (CP-split if CP > 1).
        summary_mask: [b, local_seq_len] bool — True at summary token positions (CP-split if CP > 1).
        full_summary_mask: [b, full_seq_len] bool — True at summary token positions (never CP-split).
            Used by the CUDA kernel which operates on full-sequence tensors after CP all-to-all.
            When CP=1, this equals summary_mask.
        summary_sliding_mask: optional [seq, seq] attention mask (for PyTorch fallback, not used with CUDA kernel).
        attention_mode: "kernel" for CUDA summary_attn_func, "mask"/"chunk" for fallback.
    """

    samples: List[SummarySampleContext]
    position_ids: torch.Tensor
    summary_mask: torch.Tensor
    full_summary_mask: Optional[torch.Tensor] = None
    summary_sliding_mask: Optional[torch.Tensor] = None
    attention_mode: str = "mask"  # "mask" or "chunk"
    curr_iteration: int = 0

    # Training-only Summary Indexer objective controls.  These live on the
    # batch context rather than the model config because they are an
    # experiment-specific loss choice and do not change parameter shapes or
    # exported runtime behavior.
    sksa_indexer_kl_factor: float = 1.0
    sksa_indexer_full_candidate_kl: bool = False
    sksa_indexer_monitor_topk_blocks: int = 0

    # Detached per-target-layer diagnostics populated during the original
    # forward.  Activation-checkpoint recomputation deliberately does not
    # append a second copy.  The training recipe averages these values for
    # logging while only the scalar structural objective is backpropagated.
    _sksa_indexer_kl_diagnostics: List[torch.Tensor] = field(
        default_factory=list, repr=False
    )
    _sksa_indexer_oracle_topk_recall_diagnostics: List[torch.Tensor] = field(
        default_factory=list, repr=False
    )
    _sksa_indexer_selected_attention_mass_diagnostics: List[torch.Tensor] = field(
        default_factory=list, repr=False
    )
    _sksa_indexer_oracle_topk_attention_mass_diagnostics: List[torch.Tensor] = field(
        default_factory=list, repr=False
    )
    _sksa_indexer_selected_oracle_attention_mass_ratio_diagnostics: List[
        torch.Tensor
    ] = field(default_factory=list, repr=False)

    # Training-only per-microbatch override.  ``None`` preserves the model
    # config, while ``False`` skips the expensive Indexer teacher/KL path but
    # still computes hard routes and sparse attention.  Runtime callers never
    # set this field.
    sksa_indexer_distill_active: Optional[bool] = None

    # Static, batch-uniform layout metadata computed while inserting summaries.
    # Passing this through the model avoids synchronizing CUDA tensors merely to
    # rediscover the summary count and whether the final token is a summary in
    # every attention layer (and again during checkpoint recomputation).
    summary_layout_signature: Optional[Tuple[int, bool]] = None
    summary_layout_validated: bool = False

    # Cached data for block-wise attention (to avoid recomputation)
    _block_attn_cache: Dict = field(default_factory=dict)

    # Detached Summary-Lightning routes/teacher targets (sparse), or compact
    # per-tile KL/diagnostic scalars (dense), retained only between
    # the original forward of a checkpointed Full layer and its backward
    # recomputation.  Values are stacks keyed by layer index so repeated
    # forwards over the same context remain LIFO-correct.  This cache is never
    # serialized and ordinary non-checkpointed forwards do not populate it.
    _sksa_indexer_checkpoint_cache: Dict = field(
        default_factory=dict, repr=False
    )

    # Optional training-only speed cache for token-sparse Full layers under
    # non-reentrant activation checkpointing.  Each entry retains the
    # detached BF16 attention output and FP32 LSE from the real forward until
    # that layer is recomputed in backward, avoiding a duplicate sparse
    # attention forward at a modest, bounded memory cost.

    # All-token Indexer supervision can reuse the exact text-row LSE produced
    # by its detached Full-attention teacher.  During dense warm-up that teacher
    # is also the main path; during sparse CPT its output is discarded.  Keep
    # only the compact, detached [B,H,text] LSE (or a None fallback sentinel)
    # until the enclosing layer's checkpoint recomputation consumes it.  As
    # above, each layer owns a LIFO stack and ordinary forwards retain nothing.
    _sksa_dense_teacher_lse_checkpoint_cache: Dict = field(
        default_factory=dict, repr=False
    )

    @property
    def enabled(self) -> bool:
        return any(sample.chunks for sample in self.samples)
