"""
Context Parallelism and Sequence Parallel Utilities.

This module implements context parallelism for handling long sequences in transformer
models. Context parallelism splits sequences across multiple devices, enabling training
on sequences longer than a single device's memory can hold.

Key features:
- Context parallel process group management
- Sequence sharding and gathering operations
- All-to-all communication for QKV tensors
- Automatic gradient handling for distributed operations
- Data parallel group support

The module provides both low-level communication primitives (all_to_all_4D, all_gather)
and high-level utilities (get_local_sequence, gather_by_group) for implementing
sequence parallel transformers.

Process Groups:
    - Context Parallel Group: GPUs working on different sequence chunks
    - Data Parallel Group: GPUs with identical model replicas

Functions:
    initialize_model_parallel: Set up context and data parallel groups
    get_context_parallel_group: Get context parallel process group
    get_context_parallel_world_size: Get CP world size
    get_context_parallel_rank: Get CP rank
    get_local_sequence_boundary: Compute sequence slice for current rank
    get_local_sequence: Extract local sequence chunk
    get_data_parallel_group: Get data parallel process group
    all_to_all_4D: All-to-all for 4D tensors (QKV)
    all_gather: All-gather for sequence dimensions
    gather_by_group: Gather data across process group

Classes:
    SeqAllToAll4D: Autograd function for sequence all-to-all
    AllGather: Autograd function for all-gather

Example:
    >>> import torch.distributed as dist
    >>> from muse.training.parallel import initialize_model_parallel
    >>> 
    >>> # Initialize with context parallelism
    >>> dist.init_process_group(backend="nccl")
    >>> initialize_model_parallel(context_parallel_size=4)
    >>> 
    >>> # Split sequence across 4 GPUs
    >>> full_seq = torch.randn(batch, 8192, hidden)
    >>> local_seq = get_local_sequence(full_seq, seq_idx=1)
    >>> # local_seq shape: (batch, 2048, hidden) on each GPU
    >>> 
    >>> # Gather back to full sequence
    >>> gathered = all_gather(local_seq, gather_idx=1)
    >>> # gathered shape: (batch, 8192, hidden)
"""
from typing import Any, Tuple, List, Iterable
import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist

from muse.utils.common import print_rank_0, Timer
import datetime

process_group_timeout = datetime.timedelta(minutes=60 * 24)

_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GROUP_GLOO = None
_DATA_PARALLEL_GROUP = None


def initialize_model_parallel(context_parallel_size: int = 1):
    world_size = dist.get_world_size()
    assert world_size % context_parallel_size == 0, \
        f"world_size ({world_size}) must be divisible by cp_size ({context_parallel_size})"
    num_context_parallel_groups: int = world_size // context_parallel_size
    num_data_parallel_groups: int = context_parallel_size
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GROUP
    for i in range(num_context_parallel_groups):
        ranks = range(i * context_parallel_size, (i + 1) * context_parallel_size)
        print_rank_0(f"Context Parallel Group: {i}, Ranks: {ranks}")
        group = torch.distributed.new_group(ranks)
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        rank = dist.get_rank()
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CONTEXT_PARALLEL_GROUP_GLOO = group_gloo
    for i in range(num_data_parallel_groups):
        ranks = [r for r in range(world_size) if r % context_parallel_size == i]
        print_rank_0(f"Data Parallel Group: {i}, Ranks: {ranks}")
        group = torch.distributed.new_group(ranks)
        rank = dist.get_rank()
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

def worker_init_fn(worker_id):
    if os.getenv("WORKER_MASTER_PORT", None) is None:
        os.environ["WORKER_MASTER_PORT"] = \
            str(int(os.environ["MASTER_PORT"]) + worker_id + 1)
    os.environ["MASTER_PORT"] = os.environ["WORKER_MASTER_PORT"]
    dist.init_process_group(
        "gloo", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
        timeout=process_group_timeout
    )

def get_context_parallel_group(backend="nccl"):
    """Get the context parallel group the caller rank belongs to."""
    if backend == "nccl":
        return _CONTEXT_PARALLEL_GROUP
    elif backend == "gloo":
        return _CONTEXT_PARALLEL_GROUP_GLOO
    else:
        raise NotImplementedError(f"Unsupport context parallel backend: {backend}")

def get_context_parallel_world_size():
    """Get the context parallel world size."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return 1
    return dist.get_world_size(group=get_context_parallel_group())

def get_context_parallel_rank():
    """Get the context parallel rank."""
    return dist.get_rank(group=get_context_parallel_group())

def get_local_sequence_boundary(seq_len: int) -> Tuple[int, int]:
    cp_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()
    local_seqlen = seq_len // cp_size
    start, end = cp_rank * local_seqlen, (cp_rank + 1) * local_seqlen
    return start, end

def get_local_sequence(sequence: torch.Tensor, seq_idx: int = 1) -> torch.Tensor:
    if get_context_parallel_world_size() > 1:
        seq_len = sequence.shape[seq_idx]
        start, end = get_local_sequence_boundary(seq_len)
        # Create a slice object for the specified dimension
        slices = [slice(None)] * sequence.dim()
        slices[seq_idx] = slice(start, end)
        # Use the slice object to index the tensor
        local_sequence = sequence[tuple(slices)]
        return local_sequence
    return sequence


def broadcast_batch_tensors(batch: dict, group: dist.ProcessGroup) -> dict:
    """Broadcast all CUDA tensors in batch from CP group rank 0."""
    if dist.get_world_size(group) <= 1:
        return batch
    src_rank = dist.get_process_group_ranks(group)[0]
    for _, value in batch.items():
        if isinstance(value, torch.Tensor):
            dist.broadcast(value, src=src_rank, group=group)
    return batch


def split_batch_for_context_parallel(batch: dict, cp_rank: int, cp_size: int) -> dict:
    """Split batch along seq dim + build pre-shifted labels.

    For all cp_size values (including 1), constructs pre-shifted labels so that
    the loss function can use shift_labels=False uniformly.

    - input_ids, position_ids: sliced to [start, end) when cp_size > 1
    - labels: sliced to [start+1, end+1), already shifted, includes boundary token
    - loss_mask: aligned with labels
    - cu_seqlens: kept global (attention uses it after all-to-all)
    """
    full_ids = batch["input_ids"]
    full_mask = batch["loss_mask"]
    L = full_ids.shape[1]

    if cp_size > 1:
        assert L % cp_size == 0, f"seq_len ({L}) must be divisible by cp_size ({cp_size})"
        assert "position_ids" in batch and batch["position_ids"] is not None, \
            "position_ids is required when cp_size > 1 (RoPE needs correct per-chunk positions)"
        chunk = L // cp_size
        start = cp_rank * chunk
        end = start + chunk
        batch["input_ids"] = full_ids[:, start:end].contiguous()
        if "position_ids" in batch and batch["position_ids"] is not None:
            batch["position_ids"] = batch["position_ids"][:, start:end].contiguous()
    else:
        start, end = 0, L

    # pre-shifted labels: [start+1, min(end+1, L))
    label_end = min(end + 1, L)
    shifted_ids = full_ids[:, start + 1 : label_end]
    shifted_mask = full_mask[:, start + 1 : label_end]

    # last rank (or cp_size=1): end+1 > L, pad with ignored token
    if shifted_ids.shape[1] < (end - start):
        shifted_ids = F.pad(shifted_ids, (0, 1), value=0)
        shifted_mask = F.pad(shifted_mask, (0, 1), value=0)

    batch["labels"] = shifted_ids * shifted_mask + (-100) * (1 - shifted_mask)
    batch["loss_mask"] = shifted_mask.contiguous()
    return batch


def get_data_parallel_group() -> dist.ProcessGroup:
    return _DATA_PARALLEL_GROUP

def get_data_parallel_rank() -> int:
    return dist.get_rank(group=get_data_parallel_group())

def get_data_parallel_world_size() -> int:
    return dist.get_world_size(group=get_data_parallel_group())

def all_to_all_4D(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group: dist.ProcessGroup = None,
    use_sync: bool = False
) -> torch.Tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.Tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.Tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.Tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs)
        # output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the context parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.Tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        # output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the context parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")

class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> torch.Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
            ),
            None,
            None,
            None,
        )

def all_gather(
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup = None,
    gather_idx: int = 0,
    use_sync: bool = False) -> torch.Tensor:
    """
    all-gather for Context

    Args:
        inputs (torch.Tensor): a tensor to gather, with shape (bs, seqlen/P, h)
        group : torch process group
        use_sync (bool): whether to synchronize after all-gather

    Returns:
        torch.Tensor: gathered tensor (bs, seqlen, h)
    """

    seq_world_size = dist.get_world_size(group)

    if seq_world_size > 1:
        output = [torch.empty_like(input_tensor) for _ in range(seq_world_size)]
        dist.all_gather(
            tensor_list=output, tensor=input_tensor.contiguous(), group=group)
        if use_sync:
            torch.cuda.synchronize()

        return torch.cat(output, dim=gather_idx)
    else:
        return input_tensor

def shard(input_tensor, group, shard_idx):
    world_size = dist.get_world_size(group)
    if world_size > 1:
        rank = dist.get_rank(group)
        local_tensor = torch.chunk(
            input_tensor, world_size, dim=shard_idx)[rank]
        return local_tensor
    return input_tensor

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                inputs: torch.Tensor,
                group: dist.ProcessGroup,
                gather_idx: int = 0,
                use_sync: bool = False) -> torch.Tensor:
        ctx.group = group
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_gather(
            inputs, group=group, gather_idx=gather_idx,
            use_sync=use_sync)

    @staticmethod
    def backward(ctx: Any,
                 *grad_output: torch.Tensor
        ) -> Tuple[None, torch.Tensor, None, None]:
        return (
            shard(
                *grad_output,
                ctx.group,
                ctx.gather_idx
            ),
            None,
            None,
            None,
        )

def gather_batches(buffer: List[Any], group: dist.ProcessGroup):
    """
    Gather batches from all ranks in the group.
    """
    world_size = dist.get_world_size(group)
    if world_size > 1:
      with Timer("Gather batches"):
        gathered_batches = [None for _ in range(world_size)]
        start = time.time()
        dist.all_gather_object(
            object_list=gathered_batches, obj=buffer,
            group=group
        )
      gathered_batches = sum(gathered_batches, [])
    else:
      gathered_batches = buffer
    #print_rank_0(f"Num batches: {len(gathered_batches)}")
    return gathered_batches

def gather_by_group(dataloader: Iterable[Any],
                    group: dist.ProcessGroup, buffer_size: int = 1) -> List[Any]:
    """
    Gather batches from all ranks in the group.
    """
    buffer = []
    for batch in dataloader:
        buffer.append(batch)
        if len(buffer) >= buffer_size:
            yield from gather_batches(buffer, group)
            buffer = []
    if len(buffer) > 0:
        yield from gather_batches(buffer, group)
