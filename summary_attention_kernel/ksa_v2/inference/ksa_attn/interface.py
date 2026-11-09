import math
import os
import torch
from typing import Tuple, Optional, Dict
from functools import partial

# Lazy import: CuTe kernel only available on Hopper (sm90+)
_CUTE_AVAILABLE = None


def _check_cute_available():
    global _CUTE_AVAILABLE
    if _CUTE_AVAILABLE is not None:
        return _CUTE_AVAILABLE
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
        from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 9:  # sm90+ (Hopper)
            _CUTE_AVAILABLE = True
        else:
            _CUTE_AVAILABLE = False
    except (ImportError, ModuleNotFoundError):
        _CUTE_AVAILABLE = False
    return _CUTE_AVAILABLE


SUMMARY_ATTN_CACHE: Dict[Tuple, Tuple] = {}
HIERARCHICAL_SUMMARY_ATTN_CACHE: Dict[Tuple, Tuple] = {}


# ---------------------------------------------------------------------------
# CuTe kernel path (Hopper only)
# ---------------------------------------------------------------------------

class SummaryAttnFunc(torch.autograd.Function):
    @staticmethod
    def _get_cached_value(
        seqlen_q, seqlen_k, chunk_size, summary_num, sliding_chunk_num,
        skip_old_summary=False, is_fwd=True, m_block_size=128, n_block_size=128,
    ):
        from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
        from .mask import (
            summary_attn_mask_func, summary_attn_contiguous_mask_func,
            sliding_chunk_mask_func, sliding_chunk_contiguous_mask_func,
        )

        prefix_summary_length = seqlen_k - seqlen_q
        cache_key = (seqlen_q, chunk_size, summary_num, sliding_chunk_num,
                     prefix_summary_length, skip_old_summary, is_fwd)
        if cache_key not in SUMMARY_ATTN_CACHE:
            if skip_old_summary:
                func = sliding_chunk_contiguous_mask_func if prefix_summary_length > 0 else sliding_chunk_mask_func
            else:
                func = summary_attn_contiguous_mask_func if prefix_summary_length > 0 else summary_attn_mask_func
            mask_mod = partial(func, chunk_size=chunk_size, summary_num=summary_num,
                               sliding_chunk_num=sliding_chunk_num,
                               prefix_summary_length=prefix_summary_length, swap_indices=False)
            mask_swap_mod = mask_mod
            if not is_fwd:
                mask_swap_mod = partial(func, chunk_size=chunk_size, summary_num=summary_num,
                                        sliding_chunk_num=sliding_chunk_num,
                                        prefix_summary_length=prefix_summary_length, swap_indices=True)
                m_block_size, n_block_size = n_block_size, m_block_size
                seqlen_q, seqlen_k = seqlen_k, seqlen_q
            _, block_sparse = compute_block_sparsity(
                m_block_size, n_block_size, 1, 1, seqlen_q, seqlen_k,
                mask_swap_mod, None, "cuda",
            )
            SUMMARY_ATTN_CACHE[cache_key] = (block_sparse, mask_mod)
        return SUMMARY_ATTN_CACHE[cache_key]

    @staticmethod
    def forward(ctx, q, k, v, chunk_size, summary_num, sliding_chunk_num,
                softmax_scale=None, causal=False, pack_gqa=None,
                deterministic=False, skip_old_summary=False,
                force_return_lse=False):
        from flash_attn.cute.interface import _flash_attn_fwd
        block_sparse, mask_mod = SummaryAttnFunc._get_cached_value(
            q.shape[1], k.shape[1], chunk_size, summary_num, sliding_chunk_num,
            skip_old_summary=skip_old_summary, is_fwd=True, m_block_size=128, n_block_size=64,
        )
        out, lse = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal,
                                     pack_gqa=pack_gqa, mask_mod=mask_mod,
                                     block_sparse_tensors=block_sparse,
                                     m_block_size=128, n_block_size=64,
                                     return_lse=force_return_lse)
        return out, lse



# ---------------------------------------------------------------------------
# Compressed-KSA CuTe path (Hopper only)
# ---------------------------------------------------------------------------

class HierarchicalSummaryAttnFunc(torch.autograd.Function):
    @staticmethod
    def _get_cached_value(
        device_index, seqlen_q, seqlen_k, chunk_size, sliding_chunk_num,
        group_size, meta_length, *, is_fwd,
        m_block_size, n_block_size,
    ):
        from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
        from .mask import hierarchical_summary_attn_mask_func

        cache_key = (
            device_index, seqlen_q, seqlen_k, chunk_size, sliding_chunk_num,
            group_size, meta_length, is_fwd,
        )
        cached = HIERARCHICAL_SUMMARY_ATTN_CACHE.get(cache_key)
        if cached is not None:
            return cached

        mask_mod = partial(
            hierarchical_summary_attn_mask_func,
            chunk_size=chunk_size,
            sliding_chunk_num=sliding_chunk_num,
            group_size=group_size,
            meta_length=meta_length,
            swap_indices=False,
        )
        sparsity_mask_mod = mask_mod
        sparsity_seqlen_q, sparsity_seqlen_k = seqlen_q, seqlen_k
        if not is_fwd:
            sparsity_mask_mod = partial(
                hierarchical_summary_attn_mask_func,
                chunk_size=chunk_size,
                sliding_chunk_num=sliding_chunk_num,
                group_size=group_size,
                meta_length=meta_length,
                swap_indices=True,
            )
            m_block_size, n_block_size = n_block_size, m_block_size
            sparsity_seqlen_q, sparsity_seqlen_k = seqlen_k, seqlen_q

        _, block_sparse = compute_block_sparsity(
            m_block_size,
            n_block_size,
            1,
            1,
            sparsity_seqlen_q,
            sparsity_seqlen_k,
            sparsity_mask_mod,
            None,
            "cuda",
        )
        cached = (block_sparse, mask_mod)
        HIERARCHICAL_SUMMARY_ATTN_CACHE[cache_key] = cached
        return cached

    @staticmethod
    def forward(
        ctx, q, k, v, meta_k, meta_v,
        chunk_size, sliding_chunk_num, group_size,
        softmax_scale=None, deterministic=False, mass_correction=True,
        force_return_lse=False,
    ):
        from flash_attn.cute.interface import _flash_attn_fwd

        from .mask import (
            hierarchical_summary_attn_score_mod,
        )

        # Keep one physical entry per meta K/V and apply log(G) directly to
        # its score. This preserves the mass correction without undoing the
        # sequence compression inside the attention kernel.
        needs_backward = any(
            tensor.requires_grad for tensor in (q, k, v, meta_k, meta_v)
        )
        meta_length = meta_k.shape[1]
        full_k = torch.cat((meta_k, k), dim=1).contiguous()
        full_v = torch.cat((meta_v, v), dim=1).contiguous()
        meta_logit_bias = math.log(group_size)
        score_mod = None
        if mass_correction:
            score_mod = partial(
                hierarchical_summary_attn_score_mod,
                meta_length=meta_length,
                meta_logit_bias=meta_logit_bias,
            )
        block_sparse, mask_mod = HierarchicalSummaryAttnFunc._get_cached_value(
            q.device.index,
            q.shape[1],
            full_k.shape[1],
            chunk_size,
            sliding_chunk_num,
            group_size,
            meta_length,
            is_fwd=True,
            m_block_size=128,
            n_block_size=64,
        )
        out, lse = _flash_attn_fwd(
            q,
            full_k,
            full_v,
            softmax_scale=softmax_scale,
            causal=False,
            pack_gqa=None,
            score_mod=score_mod,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse,
            m_block_size=128,
            n_block_size=64,
            # torch.autograd.Function.forward runs with grad recording
            # disabled.  Therefore full_k/full_v created above do not carry
            # requires_grad even when meta_k/meta_v are the only trainable
            # inputs (the meta-pool-only warm-up case).  Force LSE retention
            # based on the original inputs so custom backward has the state
            # it needs without adding inference-only storage.
            return_lse=needs_backward or force_return_lse,
        )
        return out, lse



# ---------------------------------------------------------------------------
# Per-query attention-reuse S-KSA CuTe path (Hopper only)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Summary-indexed, 8-token block sparse Full attention (Hopper only)
# ---------------------------------------------------------------------------

def _validate_summary_block_sparse_inputs(
    q,
    k,
    v,
    selected_block_indices,
    block_size_tokens,
    local_window_tokens,
    local_alignment_tokens=None,
):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have [batch, text, heads, head_dim]")
    if q.shape[0] != k.shape[0] or k.shape != v.shape:
        raise ValueError("q, k, and v batch/KV shapes must agree")
    if q.shape[1] != k.shape[1]:
        raise ValueError(
            "summary_block_sparse_attn_func currently requires equal Q/K text lengths"
        )
    if selected_block_indices.ndim != 3:
        raise ValueError("selected_block_indices must have [batch, query, topk]")
    if tuple(selected_block_indices.shape[:2]) != tuple(q.shape[:2]):
        raise ValueError("one selected-block row is required per text query")
    if selected_block_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected_block_indices must be int32 or int64")
    if not isinstance(block_size_tokens, int) or block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be a positive integer")
    if block_size_tokens not in (1, 2, 4, 8):
        raise ValueError("route span must be 1, 2, 4, or 8 tokens")
    if not isinstance(local_window_tokens, int) or local_window_tokens <= 0:
        raise ValueError("local_window_tokens must be a positive integer")
    if local_alignment_tokens is None:
        local_alignment_tokens = block_size_tokens
    if (
        not isinstance(local_alignment_tokens, int)
        or local_alignment_tokens <= 0
        or local_alignment_tokens % block_size_tokens
    ):
        raise ValueError(
            "local_alignment_tokens must be a positive multiple of the "
            "route span"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K head dimensions must agree")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("the number of Q heads must be divisible by KV heads")
    if not all(t.is_cuda for t in (q, k, v, selected_block_indices)):
        raise RuntimeError("summary_block_sparse_attn_func requires CUDA tensors")
    if not (q.device == k.device == v.device == selected_block_indices.device):
        raise ValueError("q, k, v, and selected_block_indices must share a device")

    if os.environ.get("SKSA_VALIDATE_BLOCK_INDICES", "0") == "1":
        raw_block_count = math.ceil(k.shape[1] / block_size_tokens)
        # This diagnostic intentionally synchronises.  The production path
        # relies on the indexer's TopK contract and remains asynchronous.
        invalid = (selected_block_indices < -1) | (
            selected_block_indices >= raw_block_count
        )
        if bool(invalid.any().item()):
            raise ValueError(
                f"selected block IDs must be -1 or in [0, {raw_block_count})"
            )


def _pack_summary_block_indices(
    selected_block_indices,
    *,
    raw_block_count,
    max_workspace_elements=16 * 1024 * 1024,
):
    """Pack ``[B,Q,K]`` unique block IDs into an 8x smaller bit route.

    Top-k is expected to produce unique non-negative IDs per row.  Repeated
    IDs would make scatter-add carry into a neighbouring bit and are rejected
    by the public wrapper before this helper is called.
    """
    batch, query_count, _ = selected_block_indices.shape
    # CuTe's DLPack bridge cannot infer a unique leading dimension for a
    # rank-3 auxiliary tensor whose final extent is 1 (both trailing strides
    # then equal one).  One unused padding byte keeps short smoke/decode-like
    # sequences on the same code path without changing any route bit.
    route_byte_count = max(2, math.ceil(raw_block_count / 8))
    packed = torch.zeros(
        (batch, query_count, route_byte_count),
        dtype=torch.uint8,
        device=selected_block_indices.device,
    )
    query_chunk = max(
        1,
        max_workspace_elements // max(batch * route_byte_count, 1),
    )
    for query_start in range(0, query_count, query_chunk):
        query_end = min(query_start + query_chunk, query_count)
        indices = selected_block_indices[:, query_start:query_end]
        valid = indices >= 0
        safe_indices = indices.clamp_min(0).to(torch.long)
        byte_indices = safe_indices // 8
        bit_values = (
            torch.ones_like(safe_indices, dtype=torch.int32)
            << (safe_indices % 8).to(torch.int32)
        )
        bit_values = bit_values * valid.to(torch.int32)
        workspace = torch.zeros(
            (batch, query_end - query_start, route_byte_count),
            dtype=torch.int32,
            device=indices.device,
        )
        workspace.scatter_add_(2, byte_indices, bit_values)
        packed[:, query_start:query_end] = workspace.to(torch.uint8)
    return packed.contiguous()


def _check_unique_summary_block_indices(selected_block_indices):
    """Reject duplicate valid IDs because the packed route uses scatter-add."""
    if selected_block_indices.shape[-1] <= 1:
        return
    sorted_indices = selected_block_indices.sort(dim=-1).values
    duplicate = (
        (sorted_indices[..., 1:] == sorted_indices[..., :-1])
        & (sorted_indices[..., 1:] >= 0)
    )
    if bool(duplicate.any().item()):
        raise ValueError("valid selected block IDs must be unique within each query")


def _summary_block_sparse_tensors(
    selected_block_indices,
    *,
    text_length,
    block_size_tokens,
    local_window_tokens,
    local_alignment_tokens=None,
    query_tile_size=128,
    forward_kv_tile_size=64,
    backward_kv_tile_size=64,
    packed_route=None,
):
    """Build conservative CuTe schedules from compact block IDs.

    Forward uses a narrower 64-token KV execution tile.  This retains the
    cross-query reuse of a 128-token Q tile while reducing over-computation
    when only some of the eight 8-token raw blocks in a 64-token KV tile are
    selected.  The model-local FlashAttention wheel supports a 64-token SM90
    backward KV tile, so the same schedule can be transposed directly instead
    of conservatively merging adjacent forward tiles.  Passing 128 remains
    available for numerical and performance A/B tests.

    Remote tiles are always marked partial because every query row may have a
    different route.  The exact per-query mask is still applied inside both
    kernels; these schedules only decide which execution tiles are launched.
    """
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    if query_tile_size != 128:
        raise ValueError("SM90 S-KSA schedules currently require 128-token Q tiles")
    if forward_kv_tile_size != 64:
        raise ValueError("SM90 S-KSA forward currently requires 64-token KV tiles")
    if backward_kv_tile_size not in (64, 128):
        raise ValueError("SM90 S-KSA backward requires a 64- or 128-token KV tile")
    if backward_kv_tile_size % forward_kv_tile_size != 0:
        raise ValueError("backward KV tile must be a multiple of forward KV tile")
    if local_alignment_tokens is None:
        local_alignment_tokens = block_size_tokens

    batch, route_length, _ = selected_block_indices.shape
    if route_length != text_length:
        raise ValueError("one selected-block row is required per text query")
    q_tile_count = math.ceil(text_length / query_tile_size)
    forward_kv_tile_count = math.ceil(text_length / forward_kv_tile_size)
    backward_kv_tile_count = math.ceil(text_length / backward_kv_tile_size)
    padded_query_count = q_tile_count * query_tile_size

    if packed_route is not None:
        # Reuse the already-built bitmap instead of traversing
        # [B,Q,TopK] again.  An 8-token route uses one byte per 64-token KV
        # tile; a token route uses eight bytes per tile.
        if forward_kv_tile_size % (block_size_tokens * 8):
            raise ValueError(
                "a forward KV tile must contain a whole number of route bytes"
            )
        route_bytes_per_kv_tile = (
            forward_kv_tile_size // (block_size_tokens * 8)
        )
        required_route_bytes = (
            forward_kv_tile_count * route_bytes_per_kv_tile
        )
        route_byte_count = math.ceil(math.ceil(text_length / block_size_tokens) / 8)
        if (
            packed_route.ndim != 3
            or packed_route.shape[0] != batch
            or packed_route.shape[1] != text_length
            or packed_route.shape[2] < route_byte_count
            or packed_route.dtype != torch.uint8
            or packed_route.device != selected_block_indices.device
        ):
            raise ValueError(
                "packed S-KSA route must be uint8 [batch, query, route_byte]"
            )
        if padded_query_count != text_length:
            packed_padding = packed_route.new_zeros(
                (
                    batch,
                    padded_query_count - text_length,
                    packed_route.shape[2],
                )
            )
            packed_padded = torch.cat((packed_route, packed_padding), dim=1)
        else:
            packed_padded = packed_route
        packed_padded = packed_padded[..., :required_route_bytes]
        if packed_padded.shape[-1] < required_route_bytes:
            packed_padded = torch.nn.functional.pad(
                packed_padded, (0, required_route_bytes - packed_padded.shape[-1])
            )
        allowed = packed_padded[..., :required_route_bytes].view(
            batch,
            q_tile_count,
            query_tile_size,
            forward_kv_tile_count,
            route_bytes_per_kv_tile,
        ).any(dim=(2, 4)).to(torch.bool).contiguous()
    else:
        # Compatibility/oracle path for callers that only have selected IDs.
        if padded_query_count != text_length:
            padding = selected_block_indices.new_full(
                (
                    batch,
                    padded_query_count - text_length,
                    selected_block_indices.shape[-1],
                ),
                -1,
            )
            selected_padded = torch.cat(
                (selected_block_indices, padding), dim=1
            )
        else:
            selected_padded = selected_block_indices
        allowed_with_sentinel = torch.zeros(
            (batch, q_tile_count, forward_kv_tile_count + 1),
            dtype=torch.bool,
            device=selected_block_indices.device,
        )
        # Chunk the int64 scatter indices so 128K x large-topk routes do not
        # create another sequence-sized temporary tensor.
        tile_chunk = 8
        for q_tile_start in range(0, q_tile_count, tile_chunk):
            q_tile_end = min(q_tile_start + tile_chunk, q_tile_count)
            query_start = q_tile_start * query_tile_size
            query_end = q_tile_end * query_tile_size
            logical = selected_padded[:, query_start:query_end]
            valid = logical >= 0
            physical = torch.where(
                valid,
                (logical * block_size_tokens) // forward_kv_tile_size,
                logical.new_full((), forward_kv_tile_count),
            )
            physical = physical.reshape(
                batch, q_tile_end - q_tile_start, -1
            )
            allowed_with_sentinel[:, q_tile_start:q_tile_end].scatter_(
                2,
                physical.to(torch.long),
                True,
            )
        allowed = allowed_with_sentinel[
            ..., :forward_kv_tile_count
        ].contiguous()

    q_tiles = torch.arange(q_tile_count, device=allowed.device)
    kv_tiles = torch.arange(forward_kv_tile_count, device=allowed.device)
    q_first_token = q_tiles * query_tile_size
    q_last_token = ((q_tiles + 1) * query_tile_size - 1).clamp_max(
        text_length - 1
    )
    kv_first_token = kv_tiles * forward_kv_tile_size
    kv_last_token = ((kv_tiles + 1) * forward_kv_tile_size - 1).clamp_max(
        text_length - 1
    )

    # Schedule every tile with at least one causal-local (q, k) pair.  Each
    # query's local start is clamped at zero and aligned down to a raw summary
    # block boundary.  The token mask still clips the two boundary tiles
    # exactly.
    q_first_local_start = (
        (q_first_token - local_window_tokens + 1).clamp_min(0)
        // local_alignment_tokens
    ) * local_alignment_tokens
    local_overlap = (
        (kv_first_token[None, :] <= q_last_token[:, None])
        & (kv_last_token[None, :] >= q_first_local_start[:, None])
    )
    causal_overlap = kv_first_token[None, :] <= q_last_token[:, None]
    allowed |= local_overlap.unsqueeze(0)
    allowed &= causal_overlap.unsqueeze(0)

    # A tile can skip mask_mod only when every actual q/k pair is causal and
    # inside the local window.  Selected remote tiles always stay partial.
    q_last_local_start = (
        (q_last_token - local_window_tokens + 1).clamp_min(0)
        // local_alignment_tokens
    ) * local_alignment_tokens
    local_full = (
        (kv_last_token[None, :] <= q_first_token[:, None])
        & (kv_first_token[None, :] >= q_last_local_start[:, None])
    )
    full_allowed = allowed & local_full.unsqueeze(0)
    partial_allowed = allowed & ~full_allowed

    def compact(partial, full, *, block_size):
        counts = partial.sum(dim=-1).to(torch.int32).unsqueeze(1)
        indices = torch.argsort(
            partial.to(torch.int8), dim=-1, descending=True, stable=True
        ).to(torch.int32).unsqueeze(1)
        full_counts = full.sum(dim=-1).to(torch.int32).unsqueeze(1)
        full_indices = torch.argsort(
            full.to(torch.int8), dim=-1, descending=True, stable=True
        ).to(torch.int32).unsqueeze(1)
        return BlockSparseTensorsTorch(
            mask_block_cnt=counts,
            mask_block_idx=indices,
            full_block_cnt=full_counts,
            full_block_idx=full_indices,
            block_size=block_size,
        )

    forward_sparse = compact(
        partial_allowed,
        full_allowed,
        block_size=(query_tile_size, forward_kv_tile_size),
    )

    # Backward consumes the KV-major transpose of the forward Q-major schedule.
    # With the default 64-token tile this is a direct transpose.  The generic
    # merge below also keeps the old 128-token schedule available for A/B:
    # a merged tile is full only if both halves are full.
    return forward_sparse, None


class SummaryBlockSparseAttentionFunc(torch.autograd.Function):
    """Exact per-query Full attention over selected raw-token route units."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        packed_route,
        selected_block_indices,
        block_size_tokens,
        local_window_tokens,
        local_alignment_tokens,
        softmax_scale=None,
        deterministic=False,
        cached_output=None,
        cached_lse=None,
    ):
        from flash_attn.cute.interface import _flash_attn_fwd
        from .mask import summary_block_sparse_attn_mask_func

        fwd_sparse, bwd_sparse = _summary_block_sparse_tensors(
            selected_block_indices,
            text_length=q.shape[1],
            block_size_tokens=block_size_tokens,
            local_window_tokens=local_window_tokens,
            local_alignment_tokens=local_alignment_tokens,
            packed_route=packed_route,
        )
        mask_mod = partial(
            summary_block_sparse_attn_mask_func,
            block_size_tokens=block_size_tokens,
            local_window_tokens=local_window_tokens,
            local_alignment_tokens=local_alignment_tokens,
        )
        if (cached_output is None) != (cached_lse is None):
            raise ValueError("cached sparse output and LSE must be provided together")
        if cached_output is None:
            out, lse = _flash_attn_fwd(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=False,
                pack_gqa=None,
                mask_mod=mask_mod,
                block_sparse_tensors=fwd_sparse,
                aux_tensors=[packed_route],
                m_block_size=128,
                n_block_size=64,
                return_lse=any(t.requires_grad for t in (q, k, v)),
            )
        else:
            expected_lse_shape = (q.shape[0], q.shape[2], q.shape[1])
            if (
                cached_output.shape != q.shape
                or cached_output.dtype != q.dtype
                or cached_output.device != q.device
                or cached_lse.shape != expected_lse_shape
                or cached_lse.dtype != torch.float32
                or cached_lse.device != q.device
            ):
                raise ValueError("cached sparse attention tensor metadata mismatch")
            # The enclosing non-reentrant layer checkpoint only needs to
            # rebuild tensors saved for this Function's backward. Reuse the
            # detached original-forward values instead of executing the same
            # sparse forward kernel a second time.
            out, lse = cached_output, cached_lse
        if lse is not None:
            ctx.mark_non_differentiable(lse)
        return out, lse





# ---------------------------------------------------------------------------
# FlexAttention path (Ampere / any GPU with PyTorch >= 2.5)
# ---------------------------------------------------------------------------

_FLEX_BLOCK_MASK_CACHE: Dict[Tuple, object] = {}


def _create_block_mask_no_materialize(mask_mod, seq_len_q, seq_len_k, device, BLOCK_SIZE=128):
    """Create BlockMask without materializing the full (seq_len_q, seq_len_k) dense mask.

    Processes one q_block row at a time, reducing peak memory from
    O(seq_q * seq_k) to O(BLOCK_SIZE * seq_k).
    """
    from torch.nn.attention.flex_attention import BlockMask

    num_q_blocks = (seq_len_q + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_kv_blocks = (seq_len_k + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Block-level sparsity: (num_q_blocks, num_kv_blocks)
    any_mask = torch.zeros(num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)
    full_mask = torch.zeros(num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)

    kv_arange = torch.arange(seq_len_k, device=device)
    pad_k = (-seq_len_k) % BLOCK_SIZE
    last_kv_len = seq_len_k - (num_kv_blocks - 1) * BLOCK_SIZE if seq_len_k % BLOCK_SIZE else BLOCK_SIZE

    for q_blk in range(num_q_blocks):
        q_start = q_blk * BLOCK_SIZE
        q_end = min(q_start + BLOCK_SIZE, seq_len_q)
        q_len = q_end - q_start
        q_idx = torch.arange(q_start, q_end, device=device)

        # (q_len, seq_len_k) bool
        mask = mask_mod(0, 0, q_idx[:, None], kv_arange[None, :])

        if pad_k > 0:
            mask = torch.nn.functional.pad(mask, (0, pad_k), value=False)

        # (q_len, num_kv_blocks, BLOCK_SIZE) -> sum per block
        block_sums = mask.reshape(q_len, num_kv_blocks, BLOCK_SIZE).sum(dim=(0, 2))

        any_mask[q_blk] = block_sums > 0
        full_mask[q_blk] = block_sums == (q_len * BLOCK_SIZE)
        # Last kv block may be shorter
        if pad_k > 0:
            full_mask[q_blk, -1] = block_sums[-1] == (q_len * last_kv_len)

    def _bool_mask_to_compact(mask_2d):
        """Convert (rows, cols) bool mask to (num_per_row, indices) format."""
        num_per_row = mask_2d.sum(dim=1).to(torch.int32)
        max_num = max(num_per_row.max().item(), 1)
        sorted_idx = mask_2d.to(torch.int32).argsort(dim=1, descending=True, stable=True)
        indices = sorted_idx[:, :max_num].to(torch.int32)
        return num_per_row, indices

    kv_num_blocks, kv_indices = _bool_mask_to_compact(any_mask)
    full_kv_num_blocks, full_kv_indices = _bool_mask_to_compact(full_mask)
    q_num_blocks, q_indices = _bool_mask_to_compact(any_mask.t().contiguous())
    full_q_num_blocks, full_q_indices = _bool_mask_to_compact(full_mask.t().contiguous())

    return BlockMask(
        kv_num_blocks=kv_num_blocks[None, None],
        kv_indices=kv_indices[None, None],
        full_kv_num_blocks=full_kv_num_blocks[None, None],
        full_kv_indices=full_kv_indices[None, None],
        q_num_blocks=q_num_blocks[None, None],
        q_indices=q_indices[None, None],
        full_q_num_blocks=full_q_num_blocks[None, None],
        full_q_indices=full_q_indices[None, None],
        BLOCK_SIZE=(BLOCK_SIZE, BLOCK_SIZE),
        mask_mod=mask_mod,
        seq_lengths=(seq_len_q, seq_len_k),
    )


def _get_flex_block_mask(seq_len_q, seq_len_k, chunk_size, summary_num, sliding_chunk_num,
                          skip_old_summary, device):
    cache_key = (seq_len_q, seq_len_k, chunk_size, summary_num, sliding_chunk_num, skip_old_summary)
    if cache_key not in _FLEX_BLOCK_MASK_CACHE:
        block = chunk_size + summary_num
        prefix_summary_length = seq_len_k - seq_len_q

        if prefix_summary_length > 0:
            def mask_mod(b, h, q_idx, kv_idx):
                q_chunk_id = q_idx // block
                kv_chunk_id = (kv_idx - prefix_summary_length) // block
                is_sliding = (q_chunk_id - kv_chunk_id) <= sliding_chunk_num
                is_q_summary = (q_idx % block) >= chunk_size
                is_kv_summary = ((kv_idx - prefix_summary_length) % block) >= chunk_size
                is_causal = q_idx >= (kv_idx - prefix_summary_length)
                valid_kv_q_summary = is_q_summary & (kv_chunk_id == q_chunk_id) & is_causal
                valid_sliding_q_text = is_sliding & (~is_kv_summary) & (kv_idx >= prefix_summary_length)
                if skip_old_summary:
                    valid_kv_q_text = (~is_q_summary) & is_causal & valid_sliding_q_text
                else:
                    valid_summary_q_text = ((kv_idx // summary_num) + sliding_chunk_num) < q_chunk_id
                    valid_kv_q_text = (~is_q_summary) & ((is_causal & valid_sliding_q_text) | valid_summary_q_text)
                return valid_kv_q_summary | valid_kv_q_text
        else:
            def mask_mod(b, h, q_idx, kv_idx):
                q_chunk_id = q_idx // block
                kv_chunk_id = kv_idx // block
                is_sliding = (q_chunk_id - kv_chunk_id) <= sliding_chunk_num
                is_q_summary = (q_idx % block) >= chunk_size
                is_kv_summary = (kv_idx % block) >= chunk_size
                is_causal = q_idx >= kv_idx
                valid_kv_q_summary = is_q_summary & (kv_chunk_id == q_chunk_id) & is_causal
                valid_sliding_q_text = is_sliding & (~is_kv_summary)
                if skip_old_summary:
                    valid_kv_q_text = (~is_q_summary) & is_causal & valid_sliding_q_text
                else:
                    valid_summary_q_text = is_kv_summary & ((q_chunk_id - kv_chunk_id) > sliding_chunk_num)
                    valid_kv_q_text = (~is_q_summary) & is_causal & (valid_sliding_q_text | valid_summary_q_text)
                return valid_kv_q_summary | valid_kv_q_text

        block_mask = _create_block_mask_no_materialize(
            mask_mod, seq_len_q, seq_len_k, device,
        )
        _FLEX_BLOCK_MASK_CACHE[cache_key] = block_mask
    return _FLEX_BLOCK_MASK_CACHE[cache_key]


_compiled_flex_attention = None


def _get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention
        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention


def _summary_attn_flex(q, k, v, chunk_size, summary_num, sliding_chunk_num, skip_old_summary):
    """FlexAttention-based implementation. Auto-compiles to efficient Triton block-sparse kernel.

    Works on any GPU with PyTorch >= 2.5 (Ampere, Hopper, etc).
    No KV duplication, no Python loops — single fused kernel call.
    """
    b, seq_len_q, h_q, d = q.shape
    _, seq_len_k, h_kv, _ = k.shape
    enable_gqa = (h_q != h_kv)

    block_mask = _get_flex_block_mask(seq_len_q, seq_len_k, chunk_size, summary_num,
                                       sliding_chunk_num, skip_old_summary, q.device)

    # flex_attention expects [B, H, S, D]
    q_flex = q.transpose(1, 2)  # [B, H, S, D]
    k_flex = k.transpose(1, 2)
    v_flex = v.transpose(1, 2)

    compiled_fn = _get_compiled_flex_attention()
    out = compiled_fn(q_flex, k_flex, v_flex, block_mask=block_mask, enable_gqa=enable_gqa)
    out = out.transpose(1, 2)  # [B, S, H, D]

    return out, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summary_attn_func(
    q, k, v, chunk_size, summary_num, sliding_chunk_num,
    summary_pos: Optional[torch.Tensor] = None,
    skip_old_summary: bool = False,
    force_return_lse: bool = False,
):
    if summary_pos is not None:
        k = torch.cat((k[:, summary_pos, :], k), dim=1)
        v = torch.cat((v[:, summary_pos, :], v), dim=1)

    if _check_cute_available():
        out = SummaryAttnFunc.apply(
            q, k, v, chunk_size, summary_num, sliding_chunk_num,
            None, False, None, False, skip_old_summary, force_return_lse,
        )
    else:
        out = _summary_attn_flex(
            q, k, v, chunk_size, summary_num, sliding_chunk_num, skip_old_summary,
        )
    return out


def hierarchical_summary_attn_func(
    q, k, v, meta_k, meta_v,
    chunk_size, sliding_chunk_num, group_size,
    softmax_scale=None, mass_correction=True, force_return_lse=False,
):
    """Compressed-KSA attention; tensors use [batch, sequence, heads, dim]."""
    tensors = (q, k, v, meta_k, meta_v)
    if not all(t.is_cuda for t in tensors):
        raise RuntimeError("hierarchical_summary_attn_func requires CUDA tensors")
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("q/k/v/meta_k/meta_v must be rank-4 tensors")
    if k.shape != v.shape or meta_k.shape != meta_v.shape:
        raise ValueError("key/value tensor shapes must match")
    if q.shape[0] != k.shape[0] or q.shape[0] != meta_k.shape[0]:
        raise ValueError("batch sizes must match")
    if q.shape[1] != k.shape[1] or k.shape[2:] != meta_k.shape[2:]:
        raise ValueError("base and meta K/V layouts are incompatible")
    if q.shape[2] % k.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    if group_size <= 1 or chunk_size <= 0 or sliding_chunk_num <= 0:
        raise ValueError("group_size must exceed one and chunk/window sizes must be positive")
    with torch.cuda.device(q.device):
        if not _check_cute_available():
            raise RuntimeError("hierarchical_summary_attn_func requires CuTe on sm90+")
        return HierarchicalSummaryAttnFunc.apply(
            q, k, v, meta_k, meta_v,
            chunk_size, sliding_chunk_num, group_size,
            softmax_scale, False, mass_correction, force_return_lse,
        )








def _pad_summary_block_sparse_cute_inputs(
    q,
    k,
    v,
    selected_block_indices,
    *,
    query_tile_size=128,
):
    """Pad CuTe's physical Q tile without changing logical attention.

    The block-sparse CuTe consumer may evaluate ``mask_mod`` for every
    physical row in its final Q tile before applying the sequence-length
    predicate to later partial KV blocks.  Supplying real all-invalid route
    rows for those physical lanes keeps the auxiliary lookup in bounds.
    Future zero K/V rows remain invisible to every real query through the
    kernel's causal guard, and the padded query outputs are discarded by the
    public wrapper.
    """
    text_length = q.shape[1]
    padding = (-text_length) % query_tile_size
    if not padding:
        return q, k, v, selected_block_indices, text_length

    def pad_sequence(tensor):
        return torch.cat(
            (
                tensor,
                tensor.new_zeros(
                    tensor.shape[0], padding, *tensor.shape[2:]
                ),
            ),
            dim=1,
        )

    padded_route = torch.cat(
        (
            selected_block_indices,
            selected_block_indices.new_full(
                (
                    selected_block_indices.shape[0],
                    padding,
                    selected_block_indices.shape[2],
                ),
                -1,
            ),
        ),
        dim=1,
    )
    return (
        pad_sequence(q),
        pad_sequence(k),
        pad_sequence(v),
        padded_route,
        text_length,
    )


def summary_block_sparse_attn_func(
    q,
    k,
    v,
    selected_block_indices,
    block_size_tokens,
    local_window_tokens,
    softmax_scale=None,
    deterministic=False,
):
    """Run exact per-query Full attention over selected raw-token blocks.

    ``selected_block_indices[b, q]`` contains the summary/raw-block IDs chosen
    by query ``q``; ``-1`` pads rows with fewer candidates.  A selected ID
    exposes the corresponding raw K/V block, never the summary K/V itself.
    Local causal tokens are added independently.  The caller is responsible
    for reserving block 0 inside the TopK budget when an attention sink is
    desired; this kernel always applies a final causal guard.

    The selected IDs must be unique within each row.  This is naturally true
    for ``torch.topk``.  Set ``SKSA_VALIDATE_BLOCK_INDICES=1`` to perform the
    synchronising bounds/duplicate checks during diagnostics.
    """
    _validate_summary_block_sparse_inputs(
        q,
        k,
        v,
        selected_block_indices,
        block_size_tokens,
        local_window_tokens,
        block_size_tokens,
    )
    if os.environ.get("SKSA_VALIDATE_BLOCK_INDICES", "0") == "1":
        _check_unique_summary_block_indices(selected_block_indices)
    route_indices = selected_block_indices.to(torch.int32).contiguous()
    with torch.cuda.device(q.device):
        if not _check_cute_available():
            raise RuntimeError(
                "summary_block_sparse_attn_func requires CuTe on sm90+"
            )
        # Qwen's ordinary multi-head raw K/V is much wider than DeepSeek MLA's
        # shared latent KV.  Direct per-query gather therefore loses crucial
        # cross-query KV reuse and is bandwidth-bound.  Keep the exact CuTe
        # tiled implementation as production default; ``direct`` is an
        # explicit diagnostic prototype only.
        q, k, v, route_indices, text_length = (
            _pad_summary_block_sparse_cute_inputs(
                q,
                k,
                v,
                route_indices,
            )
        )
        raw_block_count = math.ceil(k.shape[1] / block_size_tokens)
        packed_route = _pack_summary_block_indices(
            route_indices,
            raw_block_count=raw_block_count,
        )
        output, _ = SummaryBlockSparseAttentionFunc.apply(
            q,
            k,
            v,
            packed_route,
            route_indices,
            block_size_tokens,
            local_window_tokens,
            block_size_tokens,
            softmax_scale,
            deterministic,
            None,
            None,
        )
        return (
            output
            if output.shape[1] == text_length
            else output[:, :text_length].contiguous()
        )
