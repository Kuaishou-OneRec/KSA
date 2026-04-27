"""
Summary Attention Computation Functions.

Provides:
- summary_attention_forward(): CUDA kernel wrapper (requires `summary_attn` package)
- summary_attention_forward_native(): Pure PyTorch implementation (no external deps)
  - "mask" mode: full [s, s] attention mask + SDPA
  - "block" mode: block-wise compact masks with caching + SDPA
- build_summary_sliding_attention_mask(): mask builder for testing/debugging

All functions use muse's [b, s, h, d] tensor layout.
Native functions expect already GQA-expanded q/k/v (h_q == h_kv).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CUDA kernel wrapper (lazy import)
# ---------------------------------------------------------------------------

_summary_attn_func = None


def _get_summary_attn_func():
    """Lazy import of the CUDA kernel to avoid import error when not installed."""
    global _summary_attn_func
    if _summary_attn_func is None:
        from summary_attn import summary_attn_func
        _summary_attn_func = summary_attn_func
    return _summary_attn_func


def summary_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    summary_pos: torch.Tensor,
) -> torch.Tensor:
    """Compute summary attention using the CUDA kernel.

    The kernel processes each sample independently. For batch_size > 1,
    we loop over samples.

    Args:
        q: [b, s, h, d] query tensor.
        k: [b, s, h, d] key tensor.
        v: [b, s, h, d] value tensor.
        summary_chunk_size: number of text tokens per chunk.
        summary_token_num: number of summary tokens per chunk.
        summary_sliding_chunk_num: number of previous chunks visible to text tokens.
        summary_pos: [s] bool tensor marking summary token positions.

    Returns:
        attn_out: [b, s, h * d] attention output.
    """
    kernel_fn = _get_summary_attn_func()
    b, s, h, d = q.shape

    if b == 1:
        output, _ = kernel_fn(
            q, k, v,
            summary_chunk_size,
            summary_token_num,
            summary_sliding_chunk_num,
            summary_pos=summary_pos,
        )
        return output.reshape(b, s, h * d)
    else:
        outputs = []
        for i in range(b):
            out_i, _ = kernel_fn(
                q[i:i+1], k[i:i+1], v[i:i+1],
                summary_chunk_size,
                summary_token_num,
                summary_sliding_chunk_num,
                summary_pos=summary_pos,
            )
            outputs.append(out_i.reshape(1, s, h * d))
        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Native PyTorch implementation
# ---------------------------------------------------------------------------

# Global cache for block mode attention masks.
# Structure: { pos_cache_key: { 'text_positions': ..., block_cache_key: (k_pos, mask), ... } }
_CHUNK_ATTN_MASK_CACHE: Dict[tuple, dict] = {}


def get_chunk_attn_mask_cache() -> Dict[tuple, dict]:
    """Get the global chunk attention mask cache."""
    return _CHUNK_ATTN_MASK_CACHE


def clear_chunk_attn_mask_cache():
    """Clear the global chunk attention mask cache."""
    _CHUNK_ATTN_MASK_CACHE.clear()


def _validate_native_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
) -> Tuple[int, int, int, int]:
    """Validate q/k/v shapes for native attention. Layout: [b, s, h, d]."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be 4D: [batch, seq_len, heads, head_dim].")
    if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
        raise ValueError("query/key/value must have the same seq_len.")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("query/key/value must have the same batch size.")
    if query.shape[2] != key.shape[2] or query.shape[2] != value.shape[2]:
        raise ValueError("query/key/value must have the same number of heads (already GQA expanded).")
    if query.shape[3] != key.shape[3] or query.shape[3] != value.shape[3]:
        raise ValueError("query/key/value must have the same head_dim.")
    if summary_chunk_size <= 0 or summary_token_num <= 0:
        raise ValueError("summary_chunk_size and summary_token_num must be positive.")
    if summary_sliding_chunk_num < 0:
        raise ValueError("summary_sliding_chunk_num must be >= 0.")

    b, seq_len, h, d = query.shape
    block = summary_chunk_size + summary_token_num
    if seq_len % block != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be a multiple of "
            f"(summary_chunk_size + summary_token_num) ({block})."
        )
    return b, seq_len, h, d


def summary_sliding_attention_mask_mode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full-mask native implementation of summary sliding attention.

    Builds a complete [s, s] bool mask and runs a single SDPA call.
    Simple but O(n²) memory for the mask.

    Args:
        q/k/v: [b, s, h, d] — already GQA expanded (h_q == h_kv).
        summary_chunk_size: text tokens per chunk.
        summary_token_num: summary tokens per chunk.
        summary_sliding_chunk_num: sliding window size in chunks.
        attn_mask: optional pre-computed [s, s] bool mask (True = allowed).

    Returns:
        [b, s, h*d] attention output.
    """
    b, seq_len, num_heads, head_dim = _validate_native_qkv(
        q, k, v, summary_chunk_size, summary_token_num, summary_sliding_chunk_num
    )

    if attn_mask is None:
        attn_mask = build_summary_sliding_attention_mask(
            seq_len,
            summary_chunk_size=summary_chunk_size,
            summary_num=summary_token_num,
            summary_sliding_chunk_num=summary_sliding_chunk_num,
            device=q.device,
        )
    else:
        attn_mask = attn_mask.to(q.device)

    # [b, s, h, d] → [b, h, s, d]
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)

    # Normal forward via SDPA (training graph unchanged)
    out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    # [b, h, s, d] → [b, s, h, d] → [b, s, h*d]
    out = out.transpose(1, 2).contiguous().reshape(b, seq_len, num_heads * head_dim)

    return out


def _build_summary_sliding_block_mask(
    q_start: int,
    q_end: int,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    text_positions: list,
    summary_positions: list,
    text_flat: torch.Tensor,
    summary_flat: torch.Tensor,
    text_tri: torch.Tensor,
    summary_tri: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a compact (k_pos, mask) for a query block [q_start, q_end).

    Returns:
        k_pos: [k_len] sorted unique key position indices.
        mask: [q_len, k_len] bool mask (True = allowed).
    """
    q_len = q_end - q_start
    if q_len <= 0:
        raise ValueError("block length must be positive.")

    block = summary_chunk_size + summary_token_num
    c_start = q_start // block
    c_end = (q_end - 1) // block

    k_pos_list: list = []
    chunk_q: list = []

    # First pass: collect key positions needed by each chunk's queries
    for c in range(c_start, c_end + 1):
        text_start = c * block
        text_end = text_start + summary_chunk_size
        summary_start = text_end
        summary_end = summary_start + summary_token_num

        q_text_start = max(q_start, text_start)
        q_text_end = min(q_end, text_end)
        q_text_pos = (
            torch.arange(q_text_start, q_text_end, device=device)
            if q_text_end > q_text_start
            else None
        )

        q_summary_start = max(q_start, summary_start)
        q_summary_end = min(q_end, summary_end)
        q_summary_pos = (
            torch.arange(q_summary_start, q_summary_end, device=device)
            if q_summary_end > q_summary_start
            else None
        )

        if q_text_pos is not None or q_summary_pos is not None:
            chunk_q.append((c, q_text_pos, q_summary_pos))

        # Text queries need: old summaries + prev window text + current chunk text
        if q_text_pos is not None:
            window_start = max(0, c - summary_sliding_chunk_num)
            if window_start > 0:
                old_summary_pos = summary_flat[: window_start * summary_token_num]
                if old_summary_pos.numel() > 0:
                    k_pos_list.append(old_summary_pos)
            if window_start < c:
                prev_text_pos = text_flat[
                    window_start * summary_chunk_size : c * summary_chunk_size
                ]
                if prev_text_pos.numel() > 0:
                    k_pos_list.append(prev_text_pos)
            k_pos_list.append(text_positions[c])

        # Summary queries need: current chunk text + current chunk summary
        if q_summary_pos is not None:
            k_pos_list.append(text_positions[c])
            k_pos_list.append(summary_positions[c])

    if not k_pos_list:
        k_pos = torch.empty(0, dtype=torch.long, device=device)
        mask = torch.empty((q_len, 0), dtype=torch.bool, device=device)
        return k_pos, mask

    k_pos = torch.unique(torch.cat(k_pos_list), sorted=True)
    k_len = k_pos.numel()
    mask = torch.zeros((q_len, k_len), dtype=torch.bool, device=device)

    # Second pass: fill mask values using searchsorted for column mapping
    for c, q_text_pos, q_summary_pos in chunk_q:
        if q_text_pos is not None:
            row_idx = q_text_pos - q_start
            window_start = max(0, c - summary_sliding_chunk_num)

            # Old summaries: full visibility
            if window_start > 0:
                old_summary_pos = summary_flat[: window_start * summary_token_num]
                if old_summary_pos.numel() > 0:
                    cols = torch.searchsorted(k_pos, old_summary_pos)
                    mask[row_idx[:, None], cols[None, :]] = True

            # Previous window text: full visibility
            if window_start < c:
                prev_text_pos = text_flat[
                    window_start * summary_chunk_size : c * summary_chunk_size
                ]
                if prev_text_pos.numel() > 0:
                    cols = torch.searchsorted(k_pos, prev_text_pos)
                    mask[row_idx[:, None], cols[None, :]] = True

            # Current chunk text: causal (lower triangular)
            cur_text_pos = text_positions[c]
            cur_text_cols = torch.searchsorted(k_pos, cur_text_pos)
            q_text_offsets = q_text_pos - cur_text_pos[0]
            mask_rows = text_tri[q_text_offsets]
            mask[row_idx[:, None], cur_text_cols[None, :]] = mask_rows

        if q_summary_pos is not None:
            row_idx = q_summary_pos - q_start

            # Current chunk text: full visibility
            cur_text_pos = text_positions[c]
            cur_text_cols = torch.searchsorted(k_pos, cur_text_pos)
            mask[row_idx[:, None], cur_text_cols[None, :]] = True

            # Current chunk summary: causal (lower triangular)
            cur_summary_pos = summary_positions[c]
            cur_summary_cols = torch.searchsorted(k_pos, cur_summary_pos)
            q_summary_offsets = q_summary_pos - cur_summary_pos[0]
            mask_rows = summary_tri[q_summary_offsets]
            mask[row_idx[:, None], cur_summary_cols[None, :]] = mask_rows

    return k_pos, mask


def summary_sliding_attention_block_mode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    block_size: int = 4096,
    attn_mask_cache: Optional[Dict] = None,
) -> torch.Tensor:
    """Block-wise native implementation of summary sliding attention with caching.

    Processes the sequence in contiguous query blocks of `block_size`.
    For each block, builds a compact mask covering only the needed key positions.
    Masks are cached globally so they are computed only once across training iterations.

    Args:
        q/k/v: [b, s, h, d] — already GQA expanded (h_q == h_kv).
        summary_chunk_size: text tokens per chunk.
        summary_token_num: summary tokens per chunk.
        summary_sliding_chunk_num: sliding window size in chunks.
        block_size: query block size for processing.
        attn_mask_cache: optional external cache dict.

    Returns:
        [b, s, h*d] attention output.
    """
    b, seq_len, num_heads, head_dim = _validate_native_qkv(
        q, k, v, summary_chunk_size, summary_token_num, summary_sliding_chunk_num
    )
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    # [b, s, h, d] → [b, h, s, d]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    block = summary_chunk_size + summary_token_num
    chunks = seq_len // block
    device = q.device

    # Use global cache if not provided
    if attn_mask_cache is None:
        attn_mask_cache = _CHUNK_ATTN_MASK_CACHE

    # Build or retrieve cached chunk positions
    pos_cache_key = (
        seq_len,
        summary_chunk_size,
        summary_token_num,
        device.type,
        getattr(device, 'index', None),
    )

    if pos_cache_key not in attn_mask_cache:
        attn_mask_cache[pos_cache_key] = {}

    pos_cache = attn_mask_cache[pos_cache_key]

    if 'text_positions' not in pos_cache:
        text_positions = []
        summary_positions = []
        for c in range(chunks):
            start = c * block
            text_positions.append(torch.arange(start, start + summary_chunk_size, device=device))
            summary_positions.append(torch.arange(start + summary_chunk_size, start + block, device=device))

        pos_cache['text_positions'] = text_positions
        pos_cache['summary_positions'] = summary_positions
        pos_cache['text_flat'] = torch.cat(text_positions, dim=0)
        pos_cache['summary_flat'] = torch.cat(summary_positions, dim=0)
        pos_cache['text_tri'] = torch.tril(
            torch.ones((summary_chunk_size, summary_chunk_size), dtype=torch.bool, device=device)
        )
        pos_cache['summary_tri'] = torch.tril(
            torch.ones((summary_token_num, summary_token_num), dtype=torch.bool, device=device)
        )

    text_positions = pos_cache['text_positions']
    summary_positions = pos_cache['summary_positions']
    text_flat = pos_cache['text_flat']
    summary_flat = pos_cache['summary_flat']
    text_tri = pos_cache['text_tri']
    summary_tri = pos_cache['summary_tri']

    out = torch.zeros((b, num_heads, seq_len, head_dim), dtype=q.dtype, device=device)

    for blk_start in range(0, seq_len, block_size):
        blk_end = min(seq_len, blk_start + block_size)
        blk_len = blk_end - blk_start

        # Build cache key for this specific block
        block_cache_key = (
            'block_mask',
            summary_sliding_chunk_num,
            block_size,
            blk_start,
            blk_len,
        )

        if block_cache_key not in pos_cache:
            k_pos, local_mask = _build_summary_sliding_block_mask(
                blk_start,
                blk_end,
                summary_chunk_size,
                summary_token_num,
                summary_sliding_chunk_num,
                text_positions,
                summary_positions,
                text_flat,
                summary_flat,
                text_tri,
                summary_tri,
                device,
            )
            pos_cache[block_cache_key] = (k_pos, local_mask)
        else:
            k_pos, local_mask = pos_cache[block_cache_key]

        q_block = q[:, :, blk_start:blk_end, :]
        k_block = k[:, :, k_pos, :]
        v_block = v[:, :, k_pos, :]
        attn_out = F.scaled_dot_product_attention(
            q_block, k_block, v_block, attn_mask=local_mask, dropout_p=0.0, is_causal=False
        )
        out[:, :, blk_start:blk_end, :] = attn_out.to(out.dtype)

    # [b, h, s, d] → [b, s, h, d] → [b, s, h*d]
    out = out.transpose(1, 2).contiguous().reshape(b, seq_len, num_heads * head_dim)
    return out


def summary_attention_forward_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    attention_mode: str = "mask",
    block_size: int = 4096,
) -> torch.Tensor:
    """Unified native PyTorch forward for summary sliding attention.

    No external kernel dependencies. Uses F.scaled_dot_product_attention.

    Args:
        q/k/v: [b, s, h, d] — already GQA expanded (h_q == h_kv).
        summary_chunk_size: text tokens per chunk.
        summary_token_num: summary tokens per chunk.
        summary_sliding_chunk_num: sliding window size in chunks.
        attention_mode: "mask" for full-mask, "block" for block-wise with caching.
        block_size: query block size (only used in block mode).

    Returns:
        [b, s, h*d] attention output.
    """
    if attention_mode == "block":
        return summary_sliding_attention_block_mode(
            q, k, v,
            summary_chunk_size=summary_chunk_size,
            summary_token_num=summary_token_num,
            summary_sliding_chunk_num=summary_sliding_chunk_num,
            block_size=block_size,
        )
    else:
        return summary_sliding_attention_mask_mode(
            q, k, v,
            summary_chunk_size=summary_chunk_size,
            summary_token_num=summary_token_num,
            summary_sliding_chunk_num=summary_sliding_chunk_num,
        )


# ---------------------------------------------------------------------------
# Mask builder (used by full-mask mode and for testing/debugging)
# ---------------------------------------------------------------------------

def build_summary_sliding_attention_mask(
    seq_len: int,
    summary_chunk_size: int = 32,
    summary_num: int = 1,
    summary_sliding_chunk_num: int = 0,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a boolean mask for the sliding summary attention rule (True = allowed).

    Rules:
    1) Summary tokens see current-chunk text tokens and same-chunk summary tokens (causal).
    2) Normal tokens see current-chunk text (causal), previous `summary_sliding_chunk_num`
       chunks' text, and summary tokens from older chunks.
    3) Causal order is respected within chunk and within summary tokens.

    Args:
        seq_len: expanded sequence length (must be multiple of chunk_size + summary_num).
        summary_chunk_size: text tokens per chunk.
        summary_num: summary tokens per chunk.
        summary_sliding_chunk_num: sliding window size in chunks.
        device: torch device.

    Returns:
        [seq_len, seq_len] bool tensor.
    """
    if summary_chunk_size <= 0 or summary_num <= 0:
        raise ValueError("summary_chunk_size and summary_num must be positive.")
    if summary_sliding_chunk_num < 0:
        raise ValueError("summary_sliding_chunk_num must be >= 0.")
    block = summary_chunk_size + summary_num
    if seq_len % block != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be a multiple of "
            f"(summary_chunk_size + summary_num) ({block})."
        )

    device = device or torch.device("cpu")
    chunks = seq_len // block
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    text_tri = torch.tril(
        torch.ones((summary_chunk_size, summary_chunk_size), dtype=torch.bool, device=device)
    )
    summary_tri = torch.tril(
        torch.ones((summary_num, summary_num), dtype=torch.bool, device=device)
    )

    text_positions: List[torch.Tensor] = []
    summary_positions: List[torch.Tensor] = []
    for c in range(chunks):
        start = c * block
        text_positions.append(torch.arange(start, start + summary_chunk_size, device=device))
        summary_positions.append(torch.arange(start + summary_chunk_size, start + block, device=device))

    for c in range(chunks):
        text_pos = text_positions[c]
        summary_pos = summary_positions[c]
        window_start = max(0, c - summary_sliding_chunk_num)

        # Text tokens see old summaries (before sliding window)
        if window_start > 0:
            old_summary_pos = torch.cat(summary_positions[:window_start])
            mask[text_pos[:, None], old_summary_pos] = True

        # Text tokens see previous window text
        if window_start < c:
            prev_text_pos = torch.cat(text_positions[window_start:c])
            mask[text_pos[:, None], prev_text_pos] = True

        # Text tokens see current chunk text (causal)
        mask[text_pos[:, None], text_pos[None, :]] = text_tri

        # Summary tokens see current chunk text (all)
        mask[summary_pos[:, None], text_pos] = True

        # Summary tokens see current chunk summary (causal)
        mask[summary_pos[:, None], summary_pos[None, :]] = summary_tri

    return mask
