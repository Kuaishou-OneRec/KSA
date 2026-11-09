"""Request-owned, raw-bit-preserving summary-indexer decode materialization."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(K, V, Blocks, OK, OV, Mask, Candidates, Safe,
            L, H: tl.constexpr, D: tl.constexpr,
            N: tl.constexpr, W: tl.constexpr, R: tl.constexpr,
            START, KS0: tl.constexpr, KS1: tl.constexpr,
            KS2: tl.constexpr, KS3: tl.constexpr,
            VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
            VS3: tl.constexpr, BS0: tl.constexpr, BS2: tl.constexpr,
            TILE: tl.constexpr, DIM: tl.constexpr):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    t = tl.program_id(1) * TILE + tl.arange(0, TILE)
    d = tl.arange(0, DIM)
    remote = t < N * R
    block = tl.load(Blocks + b * BS0 + (t // R) * BS2,
                    mask=remote & (t < W), other=-1).to(tl.int64)
    index = tl.where(remote, block * R + t % R, START + t - N * R)
    valid = tl.where(remote, (block >= 0) & (index < L),
                     (index >= 0) & (index < L))
    candidate = tl.where(valid, index, L)
    safe = tl.minimum(candidate, L - 1)
    k = tl.load(K + b * KS0 + h * KS1 + safe[:, None] * KS2 + d[None, :] * KS3,
                mask=(t[:, None] < W) & (d[None, :] < D), other=0)
    v = tl.load(V + b * VS0 + h * VS1 + safe[:, None] * VS2 + d[None, :] * VS3,
                mask=(t[:, None] < W) & (d[None, :] < D), other=0)
    out = (bh * W + t[:, None]) * D + d[None, :]
    tl.store(OK + out, k, mask=(t[:, None] < W) & (d[None, :] < D))
    tl.store(OV + out, v, mask=(t[:, None] < W) & (d[None, :] < D))
    if h == 0:
        tl.store(Mask + b * W + t, valid, mask=t < W)
        tl.store(Candidates + b * W + t, candidate, mask=t < W)
        tl.store(Safe + b * W + t, safe, mask=t < W)


def fused_summary_decode_gather(key, value, selected, *, raw_block_size,
                                local_window_tokens, local_alignment_tokens,
                                workspace):
    """Write every output slot, including masked slots, on the input device.

    Masked K/V equal the reference's last-live-token copy. Exact compact shapes
    and contiguous strides preserve SDPA dispatch. Storage is cache-local;
    shape/dtype/device changes replace it. Returned views live until next call.
    """
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError('decode K/V must share [B,H,L,D]')
    b, h, length, dim = key.shape
    if selected.ndim != 3 or selected.shape[:2] != (b, 1):
        raise ValueError('decode selected blocks must have [B,1,K] layout')
    alignment = raw_block_size if local_alignment_tokens is None else local_alignment_tokens
    if raw_block_size <= 0 or local_window_tokens <= 0 or alignment <= 0 or alignment % raw_block_size:
        raise ValueError('invalid block/window/alignment')
    if length == 0:
        raise ValueError('decode requires live K/V')
    n = selected.shape[-1]
    width = n * raw_block_size + local_window_tokens + alignment - 1
    signature = (b, h, width, dim, key.dtype, key.device)
    if workspace.get('signature') != signature:
        workspace.clear()
        workspace.update(signature=signature,
                         key=torch.empty((b,h,width,dim),dtype=key.dtype,device=key.device),
                         value=torch.empty((b,h,width,dim),dtype=value.dtype,device=value.device),
                         mask=torch.empty((b,1,1,width),dtype=torch.bool,device=key.device),
                         candidates=torch.empty((b,width),dtype=torch.int64,device=key.device),
                         safe=torch.empty((b,width),dtype=torch.int64,device=key.device))
    start = max(length-local_window_tokens,0) // alignment * alignment
    with torch.cuda.device(key.device):
        _gather[(b*h,triton.cdiv(width,16))](
            key,value,selected,workspace['key'],workspace['value'],workspace['mask'],
            workspace['candidates'],workspace['safe'],length,h,dim,n,width,raw_block_size,start,
            *key.stride(),*value.stride(),selected.stride(0),selected.stride(2),
            16,triton.next_power_of_2(dim),num_warps=4)
    return workspace['key'],workspace['value'],workspace['mask']
