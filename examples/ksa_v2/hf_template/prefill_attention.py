"""Inference-only KSA-v2 masks/plans; preserve KV order and 128x64 tiles.

Uses the export's bundled CuTe API, including its matching SM90 producer.
Dynamic S-KSA routing is deliberately left to the original v2 implementation.
"""
from functools import lru_cache, partial
import math
import torch
import cutlass
from cutlass import cute
from ksa_attn import hierarchical_summary_attn_func as _legacy_hierarchical
from ksa_attn import summary_attn_func as _legacy_summary


@cute.jit
def hierarchical_mask(b, h, q, kv, seqlen_info, aux_tensors,
                      chunk_size: cutlass.Constexpr[int],
                      sliding_chunk_num: cutlass.Constexpr[int],
                      group_size: cutlass.Constexpr[int],
                      meta_length: cutlass.Constexpr[int]):
    period = chunk_size + 1
    q_chunk = q // period
    is_text = q % period < chunk_size
    raw_start = q_chunk - sliding_chunk_num
    window_start = cute.where(raw_start > 0, raw_start, cute.full_like(raw_start, 0))
    groups = window_start // group_size
    base = kv - meta_length
    is_base = kv >= meta_length
    kv_text = base % period < chunk_size
    causal = base <= q
    local = (base >= window_start * period) & causal & kv_text
    residual = (base >= groups * group_size * period) & (base < window_start * period) & (kv_text ^ 1)
    meta = (kv < meta_length) & (kv < groups)
    current = (base >= q_chunk * period) & causal
    return (is_text & (meta | (is_base & (local | residual)))) | ((is_text ^ 1) & is_base & current)


@cute.jit
def full_window_mask(b, h, q, kv, seqlen_info, aux_tensors,
                     chunk_size: cutlass.Constexpr[int],
                     prefix_length: cutlass.Constexpr[int]):
    period = chunk_size + 1
    remainder = q % period
    is_text = remainder < chunk_size
    base = kv - prefix_length
    causal = base <= q
    current = base >= q - remainder
    return (base >= 0) & causal & ((is_text & (base % period < chunk_size)) | ((is_text ^ 1) & current))


@lru_cache(maxsize=16)
def _hierarchical_plan(sq, sk, chunk, window, group, meta, device):
    from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
    from ksa_attn.mask import hierarchical_summary_attn_mask_func
    original = partial(hierarchical_summary_attn_mask_func, chunk_size=chunk,
                       sliding_chunk_num=window, group_size=group,
                       meta_length=meta, swap_indices=False)
    result = compute_block_sparsity(128, 64, 1, 1, sq, sk, original, None,
                                    torch.device('cuda', device))
    sparse = result if hasattr(result, 'mask_block_cnt') else result[-1]
    if not hasattr(sparse, 'mask_block_cnt'):
        raise TypeError('CuTe did not return block sparse tensors')
    if (sq - 1) // (chunk + 1) <= window:
        mask = partial(full_window_mask, chunk_size=chunk, prefix_length=meta)
    else:
        mask = partial(hierarchical_mask, chunk_size=chunk, sliding_chunk_num=window,
                       group_size=group, meta_length=meta)
    return sparse, mask


def _supported(q):
    return (not torch.is_grad_enabled() and q.is_cuda and q.dtype == torch.bfloat16
            and q.ndim == 4 and q.shape[0] == 1 and q.shape[2:] == (16, 128)
            and torch.cuda.get_device_capability(q.device)[0] == 9)


def hierarchical_summary_attn_prefill(q, k, v, meta_k, meta_v, chunk_size,
                                    sliding_chunk_num, group_size,
                                    softmax_scale=None, mass_correction=True,
                                    force_return_lse=False):
    if not _supported(q):
        return _legacy_hierarchical(q, k, v, meta_k, meta_v, chunk_size,
                                   sliding_chunk_num, group_size, softmax_scale,
                                   mass_correction, force_return_lse)
    from flash_attn.cute.interface import _flash_attn_fwd
    from ksa_attn.mask import hierarchical_summary_attn_score_mod
    full_k = torch.cat((meta_k, k), dim=1).contiguous()
    full_v = torch.cat((meta_v, v), dim=1).contiguous()
    sparse, mask = _hierarchical_plan(q.shape[1], full_k.shape[1], chunk_size,
                                     sliding_chunk_num, group_size, meta_k.shape[1], q.device.index)
    score = partial(hierarchical_summary_attn_score_mod, meta_length=meta_k.shape[1],
                    meta_logit_bias=math.log(group_size)) if mass_correction else None
    return _flash_attn_fwd(q, full_k, full_v, softmax_scale=softmax_scale,
                           causal=False, pack_gqa=None, score_mod=score, mask_mod=mask,
                           block_sparse_tensors=sparse, m_block_size=128,
                           n_block_size=64, return_lse=force_return_lse)


def summary_attn_prefill(q, k, v, chunk_size, summary_num, sliding_chunk_num,
                         summary_pos=None, skip_old_summary=False, force_return_lse=False):
    # The exported v2 configuration uses S-KSA on all non-hierarchical layers.
    # Retain the bundled summary path for compatible config variants.
    return _legacy_summary(q, k, v, chunk_size, summary_num, sliding_chunk_num,
                           summary_pos, skip_old_summary, force_return_lse)
