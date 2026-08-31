"""
Summary Attention Data Preparation.

Main function: maybe_build_summary_batch
"""

from typing import Optional, Tuple, Dict, Any

import torch

from muse.layers.summary_context import (
    SummaryBatchContext,
    SummaryChunkMeta,
    SummarySampleContext,
)


def maybe_build_summary_batch(
    batch: Dict[str, Any],
    config,
) -> Tuple[Dict[str, Any], Optional[SummaryBatchContext]]:
    """Insert summary tokens into the batch and build the runtime context.

    Operates on the batch dict in-place (replaces tensor values).
    Summary tokens are inserted globally every chunk_size tokens, regardless
    of document boundaries (cu_seqlens).

    Args:
        batch: dict with keys 'input_ids', 'loss_mask', 'position_ids', 'cu_seqlens'.
            - input_ids: [b, seq_len]
            - loss_mask: [b, seq_len]
            - position_ids: [b, seq_len] (optional, may be None)
            - cu_seqlens: [num_docs + 1] (optional, for packed sequences)
        config: Qwen3SummaryAttentionConfig instance.

    Returns:
        (batch, summary_ctx): batch with updated tensors, and SummaryBatchContext.
        Returns (batch, None) if summary attention is disabled.
    """
    chunk_size = config.summary_chunk_size
    summary_num = config.summary_token_num
    summary_token_begin = config.summary_token_begin
    summary_sliding_chunk_num = config.summary_sliding_chunk_num

    if chunk_size <= 0 or summary_num <= 0:
        return batch, None

    if summary_token_begin is None:
        raise ValueError(
            'summary_token_begin must be provided when enabling summary attention.'
        )

    tokens = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    position_ids = batch.get("position_ids", None)
    cu_seqlens = batch.get("cu_seqlens", None)

    batch_size, seq_length = tokens.shape
    full_chunk_count, remainder = divmod(seq_length, chunk_size)
    chunk_count = full_chunk_count + int(remainder > 0)
    new_seq_len = seq_length + chunk_count * summary_num
    device = tokens.device

    # Summary token IDs
    if config.summary_independent_parameters:
        # Placeholder IDs; actual embedding is replaced in model forward
        summary_ids = torch.zeros(summary_num, device=device, dtype=tokens.dtype)
    else:
        summary_ids = (
            torch.arange(summary_num, device=device, dtype=tokens.dtype)
            + summary_token_begin
        )

    # Allocate expanded tensors (NOTE: loss_mask is NOT expanded, kept at original length)
    new_tokens = torch.empty(
        (batch_size, new_seq_len), dtype=tokens.dtype, device=device
    )
    new_position_ids = torch.full(
        (batch_size, new_seq_len),
        fill_value=-1,
        dtype=position_ids.dtype if position_ids is not None else torch.long,
        device=device,
    )
    summary_mask = torch.zeros(
        (batch_size, new_seq_len), dtype=torch.bool, device=device
    )

    block_size = chunk_size + summary_num
    full_text_len = full_chunk_count * chunk_size
    full_expanded_len = full_chunk_count * block_size

    # Fill every complete chunk for the whole batch at once.  The former
    # batch/chunk loops launched many tiny CUDA operations (16,384 chunks at
    # 128K with chunk_size=8).
    if full_chunk_count > 0:
        token_blocks = new_tokens[:, :full_expanded_len].view(
            batch_size, full_chunk_count, block_size
        )
        token_blocks[:, :, :chunk_size] = tokens[:, :full_text_len].reshape(
            batch_size, full_chunk_count, chunk_size
        )
        token_blocks[:, :, chunk_size:] = summary_ids.view(1, 1, summary_num)

        position_blocks = new_position_ids[:, :full_expanded_len].view(
            batch_size, full_chunk_count, block_size
        )
        if config.summary_chunk_position_ids_type == 'inner_chunk':
            text_position_values = torch.arange(
                chunk_size, device=device, dtype=new_position_ids.dtype
            ).view(1, 1, chunk_size)
        elif config.summary_chunk_position_ids_type == 'origin':
            text_position_values = torch.arange(
                full_text_len, device=device, dtype=new_position_ids.dtype
            ).view(1, full_chunk_count, chunk_size)
        else:
            raise ValueError(
                f'Unknown summary_chunk_position_ids_type: '
                f'{config.summary_chunk_position_ids_type}'
            )
        position_blocks[:, :, :chunk_size] = text_position_values

        summary_mask_blocks = summary_mask[:, :full_expanded_len].view(
            batch_size, full_chunk_count, block_size
        )
        summary_mask_blocks[:, :, chunk_size:] = True
    elif chunk_count > 0 and config.summary_chunk_position_ids_type not in (
        'inner_chunk', 'origin'
    ):
        raise ValueError(
            f'Unknown summary_chunk_position_ids_type: '
            f'{config.summary_chunk_position_ids_type}'
        )

    # Build all summary-token RoPE positions in one operation.  The tail
    # formula intentionally preserves the legacy implementation's behavior
    # for a non-divisible sequence length.
    if chunk_count > 0:
        token_position_type = config.summary_token_position_ids_type
        if token_position_type == 'zeros':
            summary_position_id_values = torch.zeros(
                (chunk_count, summary_num),
                device=device,
                dtype=new_position_ids.dtype,
            )
        elif token_position_type in (
            'last_chunk_slice_right', 'last_chunk_slice_left'
        ):
            text_starts = (
                torch.arange(chunk_count, device=device, dtype=torch.long)
                * chunk_size
            )
            text_lengths = torch.full(
                (chunk_count,), chunk_size, device=device, dtype=torch.long
            )
            if remainder > 0:
                legacy_tail_start = full_chunk_count * remainder
                text_starts[-1] = legacy_tail_start
                text_lengths[-1] = seq_length - legacy_tail_start

            index_start = (
                1 if token_position_type == 'last_chunk_slice_right' else 0
            )
            summary_indices = torch.arange(
                index_start,
                index_start + summary_num,
                device=device,
                dtype=torch.long,
            )
            summary_position_id_values = (
                text_starts[:, None]
                + summary_indices[None, :] * text_lengths[:, None] // summary_num
                - 1
            )
            summary_position_id_values = torch.maximum(
                summary_position_id_values, text_starts[:, None]
            ).to(dtype=new_position_ids.dtype)
        else:
            raise ValueError(
                f'Unknown summary_token_position_ids_type: '
                f'{token_position_type}'
            )

        if full_chunk_count > 0:
            position_blocks[:, :, chunk_size:] = (
                summary_position_id_values[:full_chunk_count].unsqueeze(0)
            )

    # A sequence has at most one partial chunk.  Handle it once without
    # reintroducing a per-chunk CUDA loop.
    if remainder > 0:
        tail_start = full_expanded_len
        tail_text_end = tail_start + remainder
        tail_summary_end = tail_text_end + summary_num

        new_tokens[:, tail_start:tail_text_end] = tokens[:, full_text_len:]
        new_tokens[:, tail_text_end:tail_summary_end] = summary_ids

        if config.summary_chunk_position_ids_type == 'inner_chunk':
            tail_text_positions = torch.arange(
                remainder, device=device, dtype=new_position_ids.dtype
            )
        else:  # validated above: origin
            tail_text_positions = torch.arange(
                full_text_len,
                seq_length,
                device=device,
                dtype=new_position_ids.dtype,
            )
        new_position_ids[:, tail_start:tail_text_end] = tail_text_positions
        new_position_ids[:, tail_text_end:tail_summary_end] = (
            summary_position_id_values[-1]
        )
        summary_mask[:, tail_text_end:tail_summary_end] = True

    # Keep the metadata API, but make every historical summary prefix a view
    # into one flat tensor.  This removes the repeated torch.cat() copies whose
    # aggregate storage and work grew quadratically with the number of chunks.
    expanded_positions = torch.arange(
        new_seq_len, dtype=torch.long, device=device
    )
    if chunk_count > 0:
        full_summary_starts = (
            torch.arange(full_chunk_count, device=device, dtype=torch.long)
            * block_size
            + chunk_size
        )
        if remainder > 0:
            tail_summary_start = torch.tensor(
                [full_expanded_len + remainder],
                device=device,
                dtype=torch.long,
            )
            summary_starts = torch.cat(
                (full_summary_starts, tail_summary_start), dim=0
            )
        else:
            summary_starts = full_summary_starts
        summary_offsets = torch.arange(
            summary_num, device=device, dtype=torch.long
        )
        all_summary_positions = (
            summary_starts[:, None] + summary_offsets[None, :]
        ).reshape(-1)
    else:
        all_summary_positions = expanded_positions[:0]

    chunk_templates = []
    for chunk_idx in range(chunk_count):
        if chunk_idx < full_chunk_count:
            expanded_text_start = chunk_idx * block_size
            chunk_text_len = chunk_size
        else:
            expanded_text_start = full_expanded_len
            chunk_text_len = remainder

        summary_offset = chunk_idx * summary_num
        chunk_templates.append(
            SummaryChunkMeta(
                text_positions=expanded_positions[
                    expanded_text_start : expanded_text_start + chunk_text_len
                ],
                summary_positions=all_summary_positions[
                    summary_offset : summary_offset + summary_num
                ],
                prefix_summary_positions=all_summary_positions[:summary_offset],
            )
        )

    sample_contexts = [
        SummarySampleContext(chunks=list(chunk_templates))
        for _ in range(batch_size)
    ]

    # Update cu_seqlens: map old document boundaries to new positions
    new_cu_seqlens = None
    if cu_seqlens is not None:
        new_cu_seqlens = (
            cu_seqlens
            + torch.div(cu_seqlens, chunk_size, rounding_mode='floor')
            * summary_num
        )

    # Build context
    summary_ctx = SummaryBatchContext(
        samples=sample_contexts,
        position_ids=new_position_ids,
        summary_mask=summary_mask,
    )

    # Update batch (NOTE: loss_mask is NOT updated — kept at original length)
    batch["input_ids"] = new_tokens
    batch["position_ids"] = new_position_ids
    if new_cu_seqlens is not None:
        batch["cu_seqlens"] = new_cu_seqlens

    return batch, summary_ctx
