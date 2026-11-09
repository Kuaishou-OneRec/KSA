import cutlass
from cutlass import cute
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.utils import scalar_to_ssa, ssa_to_scalar


@cute.jit
def summary_block_sparse_attn_mask_func(
    batch_idx_ssa: cute.TensorSSA,
    head_idx_ssa: cute.TensorSSA,
    q_idx_ssa: cute.TensorSSA,
    kv_idx_ssa: cute.TensorSSA,
    seqlen_info: SeqlenInfoQK,
    aux_tensors,
    block_size_tokens: cutlass.Constexpr[int],
    local_window_tokens: cutlass.Constexpr[int],
    local_alignment_tokens: cutlass.Constexpr[int] = 0,
):
    """Per-query raw-token mask selected through summary-block indices.

    ``aux_tensors[0]`` is a packed uint8 tensor with layout
    ``[batch, query, ceil(raw_block_count / 8)]``.  Bit ``j`` is set exactly
    when the query selected summary/raw block ``j``.  Packing the dynamic
    ``[B,Q,K]`` indices before launching attention avoids materialising the
    quadratic ``[B,Q,raw_block_count]`` boolean route.

    The CuTe scheduler still works in 128-token tiles.  A scheduled tile is
    only a conservative union; this function is evaluated for every score and
    preserves exact per-query, per-``block_size_tokens`` semantics.
    """
    del head_idx_ssa, seqlen_info
    raw_block_idx_ssa = kv_idx_ssa // block_size_tokens
    route_byte_idx_ssa = raw_block_idx_ssa // 8
    route_bit_idx_ssa = raw_block_idx_ssa % 8
    packed_route = aux_tensors[0]
    route_byte = packed_route[
        ssa_to_scalar(batch_idx_ssa),
        ssa_to_scalar(q_idx_ssa),
        ssa_to_scalar(route_byte_idx_ssa),
    ]
    route_bit = 1 << ssa_to_scalar(route_bit_idx_ssa)
    selected = (route_byte & route_bit) != 0
    selected_ssa = scalar_to_ssa(selected, cutlass.Boolean)
    causal = kv_idx_ssa <= q_idx_ssa
    raw_local_start = q_idx_ssa - local_window_tokens + 1
    clamped_local_start = cute.where(
        raw_local_start > 0,
        raw_local_start,
        cute.full_like(raw_local_start, 0),
    )
    # Preserve the original summary-block API: callers that do not provide a
    # separate local alignment retain ``block_size_tokens`` alignment.  Raw
    # token routing explicitly supplies eight here while its route span is one.
    aligned_local_start = (
        clamped_local_start // block_size_tokens
    ) * block_size_tokens
    if local_alignment_tokens > 0:
        aligned_local_start = (
            clamped_local_start // local_alignment_tokens
        ) * local_alignment_tokens
    local = kv_idx_ssa >= aligned_local_start
    return causal & (local | selected_ssa)


@cute.jit
def hierarchical_summary_attn_score_mod(
    score_ssa: cute.TensorSSA,
    batch_idx_ssa: cute.TensorSSA,
    head_idx_ssa: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info: SeqlenInfoQK,
    aux_tensors,
    meta_length: cutlass.Constexpr[int],
    meta_logit_bias: cutlass.Constexpr[float],
):
    """Add log(G) to each physical meta entry before softmax."""
    return cute.where(
        kv_idx < meta_length,
        score_ssa + meta_logit_bias,
        score_ssa,
    )


@cute.jit
def hierarchical_summary_attn_score_mod_bwd(
    grad_ssa: cute.TensorSSA,
    score_ssa: cute.TensorSSA,
    batch_idx_ssa: cute.TensorSSA,
    head_idx_ssa: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info: SeqlenInfoQK,
    aux_tensors,
    meta_length: cutlass.Constexpr[int],
    meta_logit_bias: cutlass.Constexpr[float],
):
    """The derivative of an additive constant score bias is one."""
    return grad_ssa


@cute.jit
def hierarchical_summary_attn_mask_func(
    batch_idx_ssa: cute.TensorSSA,
    head_idx_ssa: cute.TensorSSA,
    q_idx_ssa: cute.TensorSSA,
    kv_idx_ssa: cute.TensorSSA,
    seqlen_info: SeqlenInfoQK,
    aux_tensors,
    chunk_size: cutlass.Constexpr[int],
    sliding_chunk_num: cutlass.Constexpr[int],
    group_size: cutlass.Constexpr[int],
    meta_length: cutlass.Constexpr[int],
    swap_indices: cutlass.Constexpr[bool],
):
    """Compressed-KSA mask for K/V laid out as ``[meta, base]``."""
    if swap_indices:
        q_idx_ssa, kv_idx_ssa = kv_idx_ssa, q_idx_ssa
    block_size = chunk_size + 1
    q_chunk_id = q_idx_ssa // block_size
    q_offset = q_idx_ssa % block_size
    is_q_text = q_offset < chunk_size
    raw_window_start = q_chunk_id - sliding_chunk_num
    window_start = cute.where(
        raw_window_start > 0,
        raw_window_start,
        cute.full_like(raw_window_start, 0),
    )
    complete_group_count = window_start // group_size
    complete_end = complete_group_count * group_size
    is_meta = kv_idx_ssa < meta_length
    base_idx = kv_idx_ssa - meta_length
    kv_chunk_id = base_idx // block_size
    kv_offset = base_idx % block_size
    is_kv_text = kv_offset < chunk_size
    is_kv_summary = kv_offset == chunk_size
    meta_group_id = kv_idx_ssa
    valid_meta = is_q_text & is_meta & (meta_group_id < complete_group_count)
    valid_text = (
        is_q_text & (is_meta ^ 1) & is_kv_text
        & (((kv_chunk_id >= window_start) & (kv_chunk_id < q_chunk_id))
           | ((kv_chunk_id == q_chunk_id) & (base_idx <= q_idx_ssa)))
    )
    valid_residual = (
        is_q_text & (is_meta ^ 1) & is_kv_summary
        & (kv_chunk_id >= complete_end) & (kv_chunk_id < window_start)
    )
    valid_summary = (
        (is_q_text ^ 1) & (is_meta ^ 1) & (kv_chunk_id == q_chunk_id)
        & (is_kv_text | (is_kv_summary & (base_idx <= q_idx_ssa)))
    )
    return valid_meta | valid_text | valid_residual | valid_summary

@cute.jit
def summary_attn_mask_func(
    batch_idx_ssa: cute.TensorSSA,
    head_idx_ssa: cute.TensorSSA,
    q_idx_ssa: cute.TensorSSA,
    kv_idx_ssa: cute.TensorSSA,
    seqlen_info: SeqlenInfoQK,
    aux_tensors,
    chunk_size: cutlass.Constexpr[int],
    summary_num: cutlass.Constexpr[int],
    sliding_chunk_num: cutlass.Constexpr[int],
    prefix_summary_length: cutlass.Constexpr[int],
    swap_indices: cutlass.Constexpr[bool],
):
    if swap_indices:
        q_idx_ssa, kv_idx_ssa = kv_idx_ssa, q_idx_ssa
    q_chunk_id = q_idx_ssa // (chunk_size + summary_num)
    kv_chunk_id = kv_idx_ssa // (chunk_size + summary_num)
    is_sliding_chunk = (q_chunk_id - kv_chunk_id) <= sliding_chunk_num
    is_q_summary = (q_idx_ssa % (chunk_size + summary_num)) >= chunk_size
    is_kv_summary = (kv_idx_ssa % (chunk_size + summary_num)) >= chunk_size
    is_causal = q_idx_ssa >= kv_idx_ssa
    valid_kv_q_summary = is_q_summary & (kv_chunk_id == q_chunk_id) & is_causal
    valid_sliding_q_text = is_sliding_chunk & (is_kv_summary ^ 1)
    valid_summary_q_text = is_kv_summary & ((q_chunk_id - kv_chunk_id) > sliding_chunk_num)
    valid_kv_q_text = (is_q_summary ^ 1) & is_causal & (valid_sliding_q_text | valid_summary_q_text)
    return valid_kv_q_summary | valid_kv_q_text

@cute.jit
def summary_attn_contiguous_mask_func(
    batch_idx_ssa: cute.TensorSSA,
    head_idx_ssa: cute.TensorSSA,
    q_idx_ssa: cute.TensorSSA,
    kv_idx_ssa: cute.TensorSSA,
    seqlen_info: SeqlenInfoQK,
    aux_tensors,
    chunk_size: cutlass.Constexpr[int],
    summary_num: cutlass.Constexpr[int],
    sliding_chunk_num: cutlass.Constexpr[int],
    prefix_summary_length: cutlass.Constexpr[int],
    swap_indices: cutlass.Constexpr[bool],
):
    if swap_indices:
        q_idx_ssa, kv_idx_ssa = kv_idx_ssa, q_idx_ssa
    q_chunk_id = q_idx_ssa // (chunk_size + summary_num)
    kv_chunk_id = (kv_idx_ssa - prefix_summary_length) // (chunk_size + summary_num)
    is_sliding_chunk = (q_chunk_id - kv_chunk_id) <= sliding_chunk_num
    is_q_summary = (q_idx_ssa % (chunk_size + summary_num)) >= chunk_size
    is_kv_summary = ((kv_idx_ssa - prefix_summary_length) % (chunk_size + summary_num)) >= chunk_size
    is_causal = q_idx_ssa >= (kv_idx_ssa - prefix_summary_length)
    valid_kv_q_summary = is_q_summary & (kv_chunk_id == q_chunk_id) & is_causal
    valid_sliding_q_text = is_sliding_chunk & (is_kv_summary ^ 1) & (kv_idx_ssa >= prefix_summary_length)
    valid_summary_q_text = ((kv_idx_ssa // summary_num) + sliding_chunk_num) < q_chunk_id
    valid_kv_q_text = (is_q_summary ^ 1) & ((is_causal & valid_sliding_q_text) | valid_summary_q_text)
    return valid_kv_q_summary | valid_kv_q_text

# ---------------------------------------------------------------------------
# Sliding-chunk-only variants: text tokens never attend to old summary tokens
# ---------------------------------------------------------------------------

@cute.jit
def sliding_chunk_mask_func(
    batch_idx_ssa: cute.TensorSSA,
    head_idx_ssa: cute.TensorSSA,
    q_idx_ssa: cute.TensorSSA,
    kv_idx_ssa: cute.TensorSSA,
    seqlen_info: SeqlenInfoQK,
    aux_tensors,
    chunk_size: cutlass.Constexpr[int],
    summary_num: cutlass.Constexpr[int],
    sliding_chunk_num: cutlass.Constexpr[int],
    prefix_summary_length: cutlass.Constexpr[int],
    swap_indices: cutlass.Constexpr[bool],
):
    if swap_indices:
        q_idx_ssa, kv_idx_ssa = kv_idx_ssa, q_idx_ssa
    q_chunk_id = q_idx_ssa // (chunk_size + summary_num)
    kv_chunk_id = kv_idx_ssa // (chunk_size + summary_num)
    is_sliding_chunk = (q_chunk_id - kv_chunk_id) <= sliding_chunk_num
    is_q_summary = (q_idx_ssa % (chunk_size + summary_num)) >= chunk_size
    is_kv_summary = (kv_idx_ssa % (chunk_size + summary_num)) >= chunk_size
    is_causal = q_idx_ssa >= kv_idx_ssa
    valid_kv_q_summary = is_q_summary & (kv_chunk_id == q_chunk_id) & is_causal
    valid_sliding_q_text = is_sliding_chunk & (is_kv_summary ^ 1)
    valid_kv_q_text = (is_q_summary ^ 1) & is_causal & valid_sliding_q_text
    return valid_kv_q_summary | valid_kv_q_text

@cute.jit
def sliding_chunk_contiguous_mask_func(
    batch_idx_ssa: cute.TensorSSA,
    head_idx_ssa: cute.TensorSSA,
    q_idx_ssa: cute.TensorSSA,
    kv_idx_ssa: cute.TensorSSA,
    seqlen_info: SeqlenInfoQK,
    aux_tensors,
    chunk_size: cutlass.Constexpr[int],
    summary_num: cutlass.Constexpr[int],
    sliding_chunk_num: cutlass.Constexpr[int],
    prefix_summary_length: cutlass.Constexpr[int],
    swap_indices: cutlass.Constexpr[bool],
):
    if swap_indices:
        q_idx_ssa, kv_idx_ssa = kv_idx_ssa, q_idx_ssa
    q_chunk_id = q_idx_ssa // (chunk_size + summary_num)
    kv_chunk_id = (kv_idx_ssa - prefix_summary_length) // (chunk_size + summary_num)
    is_sliding_chunk = (q_chunk_id - kv_chunk_id) <= sliding_chunk_num
    is_q_summary = (q_idx_ssa % (chunk_size + summary_num)) >= chunk_size
    is_kv_summary = ((kv_idx_ssa - prefix_summary_length) % (chunk_size + summary_num)) >= chunk_size
    is_causal = q_idx_ssa >= (kv_idx_ssa - prefix_summary_length)
    valid_kv_q_summary = is_q_summary & (kv_chunk_id == q_chunk_id) & is_causal
    valid_sliding_q_text = is_sliding_chunk & (is_kv_summary ^ 1) & (kv_idx_ssa >= prefix_summary_length)
    valid_kv_q_text = (is_q_summary ^ 1) & is_causal & valid_sliding_q_text
    return valid_kv_q_summary | valid_kv_q_text
