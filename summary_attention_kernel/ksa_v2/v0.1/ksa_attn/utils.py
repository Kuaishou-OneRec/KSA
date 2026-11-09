import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

# Global cache for chunk mode attention masks
_CHUNK_ATTN_MASK_CACHE: Dict[tuple, tuple] = {}

def get_chunk_attn_mask_cache() -> Dict[tuple, tuple]:
    """Get the global chunk attention mask cache."""
    return _CHUNK_ATTN_MASK_CACHE

def clear_chunk_attn_mask_cache():
    """Clear the global chunk attention mask cache."""
    global _CHUNK_ATTN_MASK_CACHE
    _CHUNK_ATTN_MASK_CACHE.clear()

def build_summary_attn_mask(
    seq_len: int,
    summary_chunk_size: int = 32,
    summary_num: int = 1,
    summary_sliding_chunk_num: int = 0,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a boolean mask for the sliding summary attention rule (True = allowed).

    Rules:
    1) Summary tokens see current-chunk text tokens and same-chunk summary tokens (causal).
    2) Normal tokens see current-chunk text (causal), previous `summary_sliding_chunk_num`
       chunks' text, and summary tokens from older chunks.
    3) Causal order is respected within chunk and within summary tokens.
    """
    if summary_chunk_size <= 0 or summary_num <= 0:
        raise ValueError("summary_chunk_size and summary_num must be positive.")
    if summary_sliding_chunk_num < 0:
        raise ValueError("summary_sliding_chunk_num must be >= 0.")
    block = summary_chunk_size + summary_num
    if seq_len % block != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be a multiple of (summary_chunk_size + summary_num) ({block})."
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

    text_positions = []
    summary_positions = []
    for c in range(chunks):
        start = c * block
        text_positions.append(torch.arange(start, start + summary_chunk_size, device=device))
        summary_positions.append(torch.arange(start + summary_chunk_size, start + block, device=device))

    for c in range(chunks):
        text_pos = text_positions[c]
        summary_pos = summary_positions[c]
        window_start = max(0, c - summary_sliding_chunk_num)

        if window_start > 0:
            old_summary_pos = torch.cat(summary_positions[:window_start])
            mask[text_pos[:, None], old_summary_pos] = True

        if window_start < c:
            prev_text_pos = torch.cat(text_positions[window_start:c])
            mask[text_pos[:, None], prev_text_pos] = True

        mask[text_pos[:, None], text_pos[None, :]] = text_tri
        mask[summary_pos[:, None], text_pos] = True
        mask[summary_pos[:, None], summary_pos[None, :]] = summary_tri

    return mask

def summary_attn_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    summary_chunk_size: int = 32,
    summary_num: int = 1,
    summary_sliding_chunk_num: int = 0,
) -> torch.Tensor:

    batch_size, seqlen_q, num_heads, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    qhead_per_kvhead = num_heads // num_heads_kv
    if qhead_per_kvhead > 1:
        key = key.repeat_interleave(qhead_per_kvhead, dim=2)
        value = value.repeat_interleave(qhead_per_kvhead, dim=2)

    cache_key = (
        seqlen_q,
        summary_chunk_size,
        summary_num,
        summary_sliding_chunk_num,
        "mask",
    )
    if cache_key not in _CHUNK_ATTN_MASK_CACHE:
        _CHUNK_ATTN_MASK_CACHE[cache_key] = build_summary_attn_mask(
            seqlen_q,
            summary_chunk_size=summary_chunk_size,
            summary_num=summary_num,
            summary_sliding_chunk_num=summary_sliding_chunk_num,
            device=query.device,
        )
    attn_mask = _CHUNK_ATTN_MASK_CACHE[cache_key]

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    out = out.to(query.dtype).permute(0, 2, 1, 3)
    return out

def build_summary_attn_block_mask(
    q_start: int,
    q_end: int,
    summary_chunk_size: int,
    summary_num: int,
    summary_sliding_chunk_num: int,
    text_positions: list,
    summary_positions: list,
    text_flat: torch.Tensor,
    summary_flat: torch.Tensor,
    text_tri: torch.Tensor,
    summary_tri: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_len = q_end - q_start
    if q_len <= 0:
        raise ValueError("block length must be positive.")

    block = summary_chunk_size + summary_num
    c_start = q_start // block
    c_end = (q_end - 1) // block

    k_pos_list: list = []
    chunk_q: list = []

    for c in range(c_start, c_end + 1):
        text_start = c * block
        text_end = text_start + summary_chunk_size
        summary_start = text_end
        summary_end = summary_start + summary_num

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

        if q_text_pos is not None:
            window_start = max(0, c - summary_sliding_chunk_num)
            if window_start > 0:
                old_summary_pos = summary_flat[: window_start * summary_num]
                if old_summary_pos.numel() > 0:
                    k_pos_list.append(old_summary_pos)
            if window_start < c:
                prev_text_pos = text_flat[
                    window_start * summary_chunk_size : c * summary_chunk_size
                ]
                if prev_text_pos.numel() > 0:
                    k_pos_list.append(prev_text_pos)
            k_pos_list.append(text_positions[c])

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

    for c, q_text_pos, q_summary_pos in chunk_q:
        if q_text_pos is not None:
            row_idx = q_text_pos - q_start
            window_start = max(0, c - summary_sliding_chunk_num)
            if window_start > 0:
                old_summary_pos = summary_flat[: window_start * summary_num]
                if old_summary_pos.numel() > 0:
                    cols = torch.searchsorted(k_pos, old_summary_pos)
                    mask[row_idx[:, None], cols[None, :]] = True
            if window_start < c:
                prev_text_pos = text_flat[
                    window_start * summary_chunk_size : c * summary_chunk_size
                ]
                if prev_text_pos.numel() > 0:
                    cols = torch.searchsorted(k_pos, prev_text_pos)
                    mask[row_idx[:, None], cols[None, :]] = True

            cur_text_pos = text_positions[c]
            cur_text_cols = torch.searchsorted(k_pos, cur_text_pos)
            q_text_offsets = q_text_pos - cur_text_pos[0]
            mask_rows = text_tri[q_text_offsets]
            mask[row_idx[:, None], cur_text_cols[None, :]] = mask_rows

        if q_summary_pos is not None:
            row_idx = q_summary_pos - q_start
            cur_text_pos = text_positions[c]
            cur_text_cols = torch.searchsorted(k_pos, cur_text_pos)
            mask[row_idx[:, None], cur_text_cols[None, :]] = True

            cur_summary_pos = summary_positions[c]
            cur_summary_cols = torch.searchsorted(k_pos, cur_summary_pos)
            q_summary_offsets = q_summary_pos - cur_summary_pos[0]
            mask_rows = summary_tri[q_summary_offsets]
            mask[row_idx[:, None], cur_summary_cols[None, :]] = mask_rows

    return k_pos, mask

def summary_attn_block(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    summary_chunk_size: int = 32,
    summary_num: int = 1,
    summary_sliding_chunk_num: int = 0,
    block_size: int = 256,
    attn_mask_cache: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Blocked implementation of the sliding summary attention with caching.

    The sequence is processed in contiguous query blocks of size `block_size`.
    For each block, we build a compact mask and key positions list.
    `attn_mask_cache` can be used to reuse (k_pos, mask) per block across iterations.

    Args:
        query/key/value: [seq_len, batch, heads, head_dim]
        summary_chunk_size: Number of text tokens per summary chunk
        summary_num: Number of summary tokens per chunk
        summary_sliding_chunk_num: Number of sliding chunks for text tokens
        block_size: Size of query blocks for processing
        attn_mask_cache: Optional dict to cache masks (reused across training iterations)

    Returns:
        attention output: [seq_len, batch, heads * head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    qhead_per_kvhead = num_heads // num_heads_kv
    if qhead_per_kvhead > 1:
        key = key.repeat_interleave(qhead_per_kvhead, dim=2)
        value = value.repeat_interleave(qhead_per_kvhead, dim=2)

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    block = summary_chunk_size + summary_num
    chunks = seq_len // block
    device = query.device

    # Use global cache if not provided
    if attn_mask_cache is None:
        attn_mask_cache = _CHUNK_ATTN_MASK_CACHE

    # Build or retrieve cached chunk positions
    pos_cache_key = (
        seq_len,
        summary_chunk_size,
        summary_num,
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
            torch.ones((summary_num, summary_num), dtype=torch.bool, device=device)
        )

    text_positions = pos_cache['text_positions']
    summary_positions = pos_cache['summary_positions']
    text_flat = pos_cache['text_flat']
    summary_flat = pos_cache['summary_flat']
    text_tri = pos_cache['text_tri']
    summary_tri = pos_cache['summary_tri']

    out = torch.zeros((batch_size, num_heads, seq_len, head_dim), dtype=query.dtype, device=device)

    for block_start in range(0, seq_len, block_size):
        block_end = min(seq_len, block_start + block_size)
        block_len = block_end - block_start

        # Build cache key for this specific block
        block_cache_key = (
            'block_mask',
            summary_sliding_chunk_num,
            block_size,
            block_start,
            block_len,
        )

        if block_cache_key not in pos_cache:
            k_pos, local_mask = build_summary_attn_block_mask(
                block_start,
                block_end,
                summary_chunk_size,
                summary_num,
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

        q = query[:, :, block_start:block_end, :]
        k = key[:, :, k_pos, :]
        v = value[:, :, k_pos, :]
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=local_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out[:, :, block_start:block_end, :] = attn_out.to(out.dtype)

    out = out.permute(0, 2, 1, 3)
    return out

def summary_attn_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    summary_chunk_size: int = 32,
    summary_num: int = 1,
    summary_sliding_chunk_num: int = 0,
    attention_mode: str = "mask",
    block_size: int = 4096,
    attn_mask_cache: Optional[Dict] = None,
):
    if attention_mode == "block":
        return summary_attn_block(
            query,
            key,
            value,
            summary_chunk_size=summary_chunk_size,
            summary_num=summary_num,
            summary_sliding_chunk_num=summary_sliding_chunk_num,
            block_size=block_size,
            attn_mask_cache=attn_mask_cache,
        )
    else:
        return summary_attn_mask(
            query,
            key,
            value,
            summary_chunk_size=summary_chunk_size,
            summary_num=summary_num,
            summary_sliding_chunk_num=summary_sliding_chunk_num,
        )
