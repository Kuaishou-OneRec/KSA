"""
Summary Attention Computation Functions.

Provides:
- summary_attention_forward(): CUDA kernel wrapper (requires `ksa_attn` package)
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
_hierarchical_summary_attn_func = None


def _get_summary_attn_func():
    """Lazy import of the CUDA kernel to avoid import error when not installed."""
    global _summary_attn_func
    if _summary_attn_func is None:
        from ksa_attn import summary_attn_func
        _summary_attn_func = summary_attn_func
    return _summary_attn_func


def _get_hierarchical_summary_attn_func():
    """Lazy import of the trainable Compressed-KSA CUDA kernel."""
    global _hierarchical_summary_attn_func
    if _hierarchical_summary_attn_func is None:
        from ksa_attn import hierarchical_summary_attn_func
        _hierarchical_summary_attn_func = hierarchical_summary_attn_func
    return _hierarchical_summary_attn_func


def summary_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    summary_pos: torch.Tensor,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        attn_out: [b, s, h * d] attention output.  When ``return_lse`` is
            true, also return the optional fp32 LSE in [b, h, s]. The Flex
            fallback returns ``None`` for LSE.
    """
    kernel_fn = _get_summary_attn_func()
    b, s, h, d = q.shape
    kernel_kwargs = {}
    if return_lse:
        kernel_kwargs["force_return_lse"] = True

    if b == 1:
        output, lse = kernel_fn(
            q, k, v,
            summary_chunk_size,
            summary_token_num,
            summary_sliding_chunk_num,
            summary_pos=summary_pos,
            **kernel_kwargs,
        )
        output = output.reshape(b, s, h * d)
        return (output, lse) if return_lse else output
    else:
        outputs = []
        lses = []
        for i in range(b):
            out_i, lse_i = kernel_fn(
                q[i:i+1], k[i:i+1], v[i:i+1],
                summary_chunk_size,
                summary_token_num,
                summary_sliding_chunk_num,
                summary_pos=summary_pos,
                **kernel_kwargs,
            )
            outputs.append(out_i.reshape(1, s, h * d))
            if return_lse:
                lses.append(lse_i)
        output = torch.cat(outputs, dim=0)
        if not return_lse:
            return output
        lse = None if any(item is None for item in lses) else torch.cat(lses, dim=0)
        return output, lse


def hierarchical_summary_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta_k: torch.Tensor,
    meta_v: torch.Tensor,
    summary_chunk_size: int,
    summary_sliding_chunk_num: int,
    meta_pool_group_size: int,
    mass_correction: bool,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute trainable Compressed-KSA attention with the CUDA kernel."""
    kernel_fn = _get_hierarchical_summary_attn_func()
    b, s, h, d = q.shape
    kernel_kwargs = {"mass_correction": mass_correction}
    if return_lse:
        # Routing can request LSE without changing ordinary C-KSA calls.
        kernel_kwargs["force_return_lse"] = True

    if b == 1:
        output, lse = kernel_fn(
            q,
            k,
            v,
            meta_k,
            meta_v,
            summary_chunk_size,
            summary_sliding_chunk_num,
            meta_pool_group_size,
            **kernel_kwargs,
        )
        output = output.reshape(b, s, h * d)
        return (output, lse) if return_lse else output

    outputs = []
    lses = []
    for i in range(b):
        output_i, lse_i = kernel_fn(
            q[i:i + 1],
            k[i:i + 1],
            v[i:i + 1],
            meta_k[i:i + 1],
            meta_v[i:i + 1],
            summary_chunk_size,
            summary_sliding_chunk_num,
            meta_pool_group_size,
            **kernel_kwargs,
        )
        outputs.append(output_i.reshape(1, s, h * d))
        if return_lse:
            lses.append(lse_i)
    output = torch.cat(outputs, dim=0)
    return (output, torch.cat(lses, dim=0)) if return_lse else output


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
    summary_mask: Optional[torch.Tensor] = None,
    summary_layout_validated: bool = False,
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
    if summary_mask is None and seq_len % block != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be a multiple of "
            f"(summary_chunk_size + summary_token_num) ({block}) when "
            "summary_mask is not provided."
        )
    if summary_mask is not None:
        if summary_mask.ndim == 2:
            if summary_mask.shape != (b, seq_len):
                raise ValueError("summary_mask shape is incompatible with q/k/v")
            if (
                not summary_layout_validated
                and not torch.all(summary_mask == summary_mask[:1])
            ):
                raise ValueError("native attention requires a batch-uniform summary_mask")
        elif summary_mask.ndim != 1 or summary_mask.shape[0] != seq_len:
            raise ValueError("summary_mask must have shape [sequence] or [batch, sequence]")
    return b, seq_len, h, d


def _build_chunk_positions(
    seq_len: int,
    summary_chunk_size: int,
    summary_token_num: int,
    device: torch.device,
    summary_mask: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build chunk positions, including an optional final text-only partial chunk."""
    if summary_mask is None:
        block = summary_chunk_size + summary_token_num
        chunks = seq_len // block
        return (
            [
                torch.arange(c * block, c * block + summary_chunk_size, device=device)
                for c in range(chunks)
            ],
            [
                torch.arange(
                    c * block + summary_chunk_size, (c + 1) * block, device=device
                )
                for c in range(chunks)
            ],
        )

    shared_mask = summary_mask[0] if summary_mask.ndim == 2 else summary_mask
    shared_mask = shared_mask.to(device=device, dtype=torch.bool)
    summary_indices = torch.nonzero(shared_mask, as_tuple=False).flatten()
    text_positions: List[torch.Tensor] = []
    summary_positions: List[torch.Tensor] = []
    cursor = 0
    summary_cursor = 0
    while summary_cursor < summary_indices.numel():
        summary_start = int(summary_indices[summary_cursor].item())
        summary_end = summary_start + summary_token_num
        expected = torch.arange(summary_start, summary_end, device=device)
        actual = summary_indices[summary_cursor: summary_cursor + summary_token_num]
        if actual.numel() != summary_token_num or not torch.equal(actual, expected):
            raise ValueError("summary_mask must mark contiguous summary-token groups")
        text_len = summary_start - cursor
        if text_len <= 0 or text_len > summary_chunk_size:
            raise ValueError("summary_mask does not match the configured chunk layout")
        is_final_summarized_chunk = summary_end == seq_len
        if text_len != summary_chunk_size and not is_final_summarized_chunk:
            raise ValueError("only the final summarized chunk may be shorter than chunk_size")
        text_positions.append(torch.arange(cursor, summary_start, device=device))
        summary_positions.append(expected)
        cursor = summary_end
        summary_cursor += summary_token_num

    if cursor < seq_len:
        trailing_len = seq_len - cursor
        if trailing_len > summary_chunk_size:
            raise ValueError("trailing partial chunk exceeds summary_chunk_size")
        text_positions.append(torch.arange(cursor, seq_len, device=device))
        summary_positions.append(torch.empty(0, dtype=torch.long, device=device))
    if cursor > seq_len or not text_positions:
        raise ValueError("summary_mask does not describe a non-empty chunk layout")
    return text_positions, summary_positions


def _summary_mask_cache_signature(
    summary_mask: Optional[torch.Tensor],
) -> Optional[Tuple[int, bool]]:
    if summary_mask is None:
        return None
    shared_mask = summary_mask[0] if summary_mask.ndim == 2 else summary_mask
    return int(shared_mask.sum().item()), bool(shared_mask[-1].item())


def summary_sliding_attention_mask_mode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    attn_mask: Optional[torch.Tensor] = None,
    summary_mask: Optional[torch.Tensor] = None,
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
        q, k, v, summary_chunk_size, summary_token_num,
        summary_sliding_chunk_num, summary_mask
    )

    if attn_mask is None:
        attn_mask = build_summary_sliding_attention_mask(
            seq_len,
            summary_chunk_size=summary_chunk_size,
            summary_num=summary_token_num,
            summary_sliding_chunk_num=summary_sliding_chunk_num,
            device=q.device,
            summary_mask=summary_mask,
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
    c_end = min((q_end - 1) // block, len(text_positions) - 1)

    k_pos_list: list = []
    chunk_q: list = []

    # First pass: collect key positions needed by each chunk's queries
    for c in range(c_start, c_end + 1):
        cur_text = text_positions[c]
        cur_summary = summary_positions[c]
        q_text_pos = cur_text[(cur_text >= q_start) & (cur_text < q_end)]
        q_summary_pos = cur_summary[
            (cur_summary >= q_start) & (cur_summary < q_end)
        ]
        q_text_pos = q_text_pos if q_text_pos.numel() else None
        q_summary_pos = q_summary_pos if q_summary_pos.numel() else None

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
            mask_rows = text_tri[q_text_offsets, :cur_text_pos.numel()]
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


def _build_summary_pooled_block_mask(
    q_start: int,
    q_end: int,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    meta_pool_group_size: int,
    text_positions: list,
    summary_positions: list,
    text_flat: torch.Tensor,
    summary_flat: torch.Tensor,
    text_tri: torch.Tensor,
    summary_tri: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compact meta/regular KV indices and mask for a pooled query block."""
    if summary_token_num != 1:
        raise ValueError("Pooled summary attention currently requires summary_token_num=1.")
    if meta_pool_group_size <= 1:
        raise ValueError("meta_pool_group_size must be greater than one.")

    q_len = q_end - q_start
    if q_len <= 0:
        raise ValueError("block length must be positive.")

    chunk_block = summary_chunk_size + summary_token_num
    c_start = q_start // chunk_block
    c_end = min((q_end - 1) // chunk_block, len(text_positions) - 1)
    meta_idx_list: list = []
    k_pos_list: list = []
    chunk_q: list = []

    for c in range(c_start, c_end + 1):
        cur_text = text_positions[c]
        cur_summary = summary_positions[c]
        q_text_pos = cur_text[(cur_text >= q_start) & (cur_text < q_end)]
        q_summary_pos = cur_summary[
            (cur_summary >= q_start) & (cur_summary < q_end)
        ]
        q_text_pos = q_text_pos if q_text_pos.numel() else None
        q_summary_pos = q_summary_pos if q_summary_pos.numel() else None
        if q_text_pos is not None or q_summary_pos is not None:
            chunk_q.append((c, q_text_pos, q_summary_pos))

        if q_text_pos is not None:
            window_start = max(0, c - summary_sliding_chunk_num)
            complete_group_count = window_start // meta_pool_group_size
            complete_end = complete_group_count * meta_pool_group_size
            if complete_group_count > 0:
                meta_idx_list.append(
                    torch.arange(complete_group_count, device=device)
                )
            if complete_end < window_start:
                k_pos_list.append(summary_flat[complete_end:window_start])
            if window_start < c:
                k_pos_list.append(
                    text_flat[
                        window_start * summary_chunk_size : c * summary_chunk_size
                    ]
                )
            k_pos_list.append(text_positions[c])

        if q_summary_pos is not None:
            k_pos_list.append(text_positions[c])
            k_pos_list.append(summary_positions[c])

    meta_idx = (
        torch.unique(torch.cat(meta_idx_list), sorted=True)
        if meta_idx_list
        else torch.empty(0, dtype=torch.long, device=device)
    )
    k_pos = (
        torch.unique(torch.cat(k_pos_list), sorted=True)
        if k_pos_list
        else torch.empty(0, dtype=torch.long, device=device)
    )
    meta_len = meta_idx.numel()
    mask = torch.zeros(
        (q_len, meta_len + k_pos.numel()), dtype=torch.bool, device=device
    )

    for c, q_text_pos, q_summary_pos in chunk_q:
        if q_text_pos is not None:
            row_idx = q_text_pos - q_start
            window_start = max(0, c - summary_sliding_chunk_num)
            complete_group_count = window_start // meta_pool_group_size
            complete_end = complete_group_count * meta_pool_group_size

            if complete_group_count > 0:
                allowed_meta = torch.arange(complete_group_count, device=device)
                meta_cols = torch.searchsorted(meta_idx, allowed_meta)
                mask[row_idx[:, None], meta_cols[None, :]] = True

            if complete_end < window_start:
                residual_summary_pos = summary_flat[complete_end:window_start]
                cols = meta_len + torch.searchsorted(k_pos, residual_summary_pos)
                mask[row_idx[:, None], cols[None, :]] = True

            if window_start < c:
                prev_text_pos = text_flat[
                    window_start * summary_chunk_size : c * summary_chunk_size
                ]
                cols = meta_len + torch.searchsorted(k_pos, prev_text_pos)
                mask[row_idx[:, None], cols[None, :]] = True

            cur_text_pos = text_positions[c]
            cur_text_cols = meta_len + torch.searchsorted(k_pos, cur_text_pos)
            q_text_offsets = q_text_pos - cur_text_pos[0]
            mask[row_idx[:, None], cur_text_cols[None, :]] = text_tri[
                q_text_offsets, :cur_text_pos.numel()
            ]

        if q_summary_pos is not None:
            row_idx = q_summary_pos - q_start
            cur_text_pos = text_positions[c]
            cur_text_cols = meta_len + torch.searchsorted(k_pos, cur_text_pos)
            mask[row_idx[:, None], cur_text_cols[None, :]] = True

            cur_summary_pos = summary_positions[c]
            cur_summary_cols = meta_len + torch.searchsorted(k_pos, cur_summary_pos)
            q_summary_offsets = q_summary_pos - cur_summary_pos[0]
            mask[row_idx[:, None], cur_summary_cols[None, :]] = summary_tri[
                q_summary_offsets
            ]

    return meta_idx, k_pos, mask


def summary_sliding_attention_block_mode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    block_size: int = 4096,
    attn_mask_cache: Optional[Dict] = None,
    summary_mask: Optional[torch.Tensor] = None,
    summary_layout_signature: Optional[Tuple[int, bool]] = None,
    summary_layout_validated: bool = False,
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
        q, k, v, summary_chunk_size, summary_token_num,
        summary_sliding_chunk_num, summary_mask, summary_layout_validated
    )
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    # [b, s, h, d] → [b, h, s, d]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    block = summary_chunk_size + summary_token_num
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
        summary_layout_signature
        if summary_layout_signature is not None
        else _summary_mask_cache_signature(summary_mask),
    )

    if pos_cache_key not in attn_mask_cache:
        attn_mask_cache[pos_cache_key] = {}

    pos_cache = attn_mask_cache[pos_cache_key]

    if 'text_positions' not in pos_cache:
        text_positions, summary_positions = _build_chunk_positions(
            seq_len,
            summary_chunk_size,
            summary_token_num,
            device,
            summary_mask,
        )

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


def summary_sliding_attention_pooled_block_mode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta_k: torch.Tensor,
    meta_v: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    meta_pool_group_size: int,
    meta_logit_bias: float = 0.0,
    block_size: int = 4096,
    attn_mask_cache: Optional[Dict] = None,
    summary_mask: Optional[torch.Tensor] = None,
    summary_layout_signature: Optional[Tuple[int, bool]] = None,
    summary_layout_validated: bool = False,
) -> torch.Tensor:
    """Native block attention with grouped meta KV for old summaries.

    ``meta_k`` must already have RoPE applied using each group's final summary
    position. ``meta_v`` is pooled from the corresponding pre-attention values.
    Both tensors use the same [batch, groups, heads, head_dim] layout as q/k/v.
    """
    b, seq_len, num_heads, head_dim = _validate_native_qkv(
        q, k, v, summary_chunk_size, summary_token_num,
        summary_sliding_chunk_num, summary_mask, summary_layout_validated
    )
    if summary_token_num != 1:
        raise ValueError("Pooled summary attention currently requires summary_token_num=1.")
    if meta_pool_group_size <= 1:
        raise ValueError("meta_pool_group_size must be greater than one.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    meta_k = meta_k.transpose(1, 2)
    meta_v = meta_v.transpose(1, 2)
    device = q.device

    if attn_mask_cache is None:
        attn_mask_cache = _CHUNK_ATTN_MASK_CACHE
    pos_cache_key = (
        seq_len,
        summary_chunk_size,
        summary_token_num,
        device.type,
        getattr(device, "index", None),
        summary_layout_signature
        if summary_layout_signature is not None
        else _summary_mask_cache_signature(summary_mask),
    )
    if pos_cache_key not in attn_mask_cache:
        attn_mask_cache[pos_cache_key] = {}
    pos_cache = attn_mask_cache[pos_cache_key]

    if "text_positions" not in pos_cache:
        text_positions, summary_positions = _build_chunk_positions(
            seq_len,
            summary_chunk_size,
            summary_token_num,
            device,
            summary_mask,
        )
        pos_cache["text_positions"] = text_positions
        pos_cache["summary_positions"] = summary_positions
        pos_cache["text_flat"] = torch.cat(text_positions, dim=0)
        pos_cache["summary_flat"] = torch.cat(summary_positions, dim=0)
        pos_cache["text_tri"] = torch.tril(
            torch.ones(
                (summary_chunk_size, summary_chunk_size),
                dtype=torch.bool,
                device=device,
            )
        )
        pos_cache["summary_tri"] = torch.tril(
            torch.ones(
                (summary_token_num, summary_token_num),
                dtype=torch.bool,
                device=device,
            )
        )

    text_positions = pos_cache["text_positions"]
    summary_positions = pos_cache["summary_positions"]
    text_flat = pos_cache["text_flat"]
    summary_flat = pos_cache["summary_flat"]
    text_tri = pos_cache["text_tri"]
    summary_tri = pos_cache["summary_tri"]
    summarized_chunks = sum(
        int(position.numel() > 0) for position in summary_positions
    )
    expected_groups = summarized_chunks // meta_pool_group_size
    expected_meta_shape = (b, num_heads, expected_groups, head_dim)
    if (
        tuple(meta_k.shape) != expected_meta_shape
        or tuple(meta_v.shape) != expected_meta_shape
    ):
        raise ValueError(
            f"meta_k/meta_v must both have shape {expected_meta_shape}; "
            f"got {tuple(meta_k.shape)} and {tuple(meta_v.shape)}"
        )

    # Concatenate the complete KV domains once per layer. The old path gathered
    # meta/regular KV independently and concatenated both tensors for every query
    # block, multiplying data movement by the number of blocks.
    combined_k = torch.cat((meta_k, k), dim=2)
    combined_v = torch.cat((meta_v, v), dim=2)
    out = torch.zeros(
        (b, num_heads, seq_len, head_dim), dtype=q.dtype, device=device
    )

    for blk_start in range(0, seq_len, block_size):
        blk_end = min(seq_len, blk_start + block_size)
        blk_len = blk_end - blk_start
        block_cache_key = (
            "pooled_block_mask",
            summary_sliding_chunk_num,
            meta_pool_group_size,
            block_size,
            blk_start,
            blk_len,
        )
        if block_cache_key not in pos_cache:
            meta_idx, k_pos, local_mask = _build_summary_pooled_block_mask(
                blk_start,
                blk_end,
                summary_chunk_size,
                summary_token_num,
                summary_sliding_chunk_num,
                meta_pool_group_size,
                text_positions,
                summary_positions,
                text_flat,
                summary_flat,
                text_tri,
                summary_tri,
                device,
            )
            combined_pos = torch.cat((meta_idx, expected_groups + k_pos))
            pos_cache[block_cache_key] = (
                meta_idx,
                k_pos,
                combined_pos,
                local_mask,
            )
        else:
            meta_idx, k_pos, combined_pos, local_mask = pos_cache[block_cache_key]

        q_block = q[:, :, blk_start:blk_end, :]
        k_block = combined_k[:, :, combined_pos, :]
        v_block = combined_v[:, :, combined_pos, :]
        attn_mask = local_mask
        if meta_logit_bias != 0.0:
            bias_cache_key = block_cache_key + (
                "meta_logit_bias",
                float(meta_logit_bias),
                q.dtype,
            )
            if bias_cache_key not in pos_cache:
                attn_mask = torch.full(
                    local_mask.shape,
                    float("-inf"),
                    dtype=q.dtype,
                    device=device,
                )
                attn_mask.masked_fill_(local_mask, 0.0)
                meta_len = meta_idx.numel()
                if meta_len:
                    attn_mask[:, :meta_len].masked_fill_(
                        local_mask[:, :meta_len], meta_logit_bias
                    )
                pos_cache[bias_cache_key] = attn_mask
            else:
                attn_mask = pos_cache[bias_cache_key]

        attn_out = F.scaled_dot_product_attention(
            q_block,
            k_block,
            v_block,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out[:, :, blk_start:blk_end, :] = attn_out.to(out.dtype)

    return out.transpose(1, 2).contiguous().reshape(
        b, seq_len, num_heads * head_dim
    )


def summary_attention_forward_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_sliding_chunk_num: int,
    attention_mode: str = "mask",
    block_size: int = 4096,
    summary_mask: Optional[torch.Tensor] = None,
    summary_layout_signature: Optional[Tuple[int, bool]] = None,
    summary_layout_validated: bool = False,
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
            summary_mask=summary_mask,
            summary_layout_signature=summary_layout_signature,
            summary_layout_validated=summary_layout_validated,
        )
    else:
        return summary_sliding_attention_mask_mode(
            q, k, v,
            summary_chunk_size=summary_chunk_size,
            summary_token_num=summary_token_num,
            summary_sliding_chunk_num=summary_sliding_chunk_num,
            summary_mask=summary_mask,
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
    summary_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a boolean mask for the sliding summary attention rule (True = allowed).

    Ported from kai-megatron/megatron/core/models/gpt/summary_attention_func.py.

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
    if summary_mask is None and seq_len % block != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be a multiple of "
            f"(summary_chunk_size + summary_num) ({block})."
        )

    device = device or torch.device("cpu")
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    text_tri = torch.tril(
        torch.ones((summary_chunk_size, summary_chunk_size), dtype=torch.bool, device=device)
    )
    summary_tri = torch.tril(
        torch.ones((summary_num, summary_num), dtype=torch.bool, device=device)
    )

    text_positions, summary_positions = _build_chunk_positions(
        seq_len,
        summary_chunk_size,
        summary_num,
        device,
        summary_mask,
    )
    chunks = len(text_positions)

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
        mask[text_pos[:, None], text_pos[None, :]] = text_tri[
            :text_pos.numel(), :text_pos.numel()
        ]

        # Summary tokens see current chunk text (all)
        if summary_pos.numel() == 0:
            continue
        mask[summary_pos[:, None], text_pos] = True

        # Summary tokens see current chunk summary (causal)
        mask[summary_pos[:, None], summary_pos[None, :]] = summary_tri

    return mask
