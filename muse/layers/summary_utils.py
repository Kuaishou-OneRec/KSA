"""
Summary Attention Data Preparation.

Main function: maybe_build_summary_batch
"""

import math
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
    chunk_count = math.ceil(seq_length / chunk_size)
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

    sample_contexts = []

    for batch_idx in range(batch_size):
        sample_chunks = []
        text_cursor = 0
        write_cursor = 0
        accumulated_summary = []

        for i in range(chunk_count):
            chunk_text_len = min(chunk_size, seq_length - text_cursor)
            if chunk_text_len <= 0:
                break

            text_slice = slice(write_cursor, write_cursor + chunk_text_len)
            src_slice = slice(text_cursor, text_cursor + chunk_text_len)

            # Copy text tokens
            new_tokens[batch_idx, text_slice] = tokens[batch_idx, src_slice]

            # Set text position IDs
            if config.summary_chunk_position_ids_type == 'inner_chunk':
                new_position_ids[batch_idx, text_slice] = torch.arange(
                    chunk_text_len, device=device,
                    dtype=new_position_ids.dtype,
                )
            elif config.summary_chunk_position_ids_type == 'origin':
                new_position_ids[batch_idx, text_slice] = torch.arange(
                    i * chunk_size,
                    min((i + 1) * chunk_size, seq_length),
                    device=device,
                    dtype=new_position_ids.dtype,
                )
            else:
                raise ValueError(
                    f'Unknown summary_chunk_position_ids_type: '
                    f'{config.summary_chunk_position_ids_type}'
                )

            # Record text positions in expanded sequence
            text_positions = torch.arange(
                text_slice.start, text_slice.stop,
                dtype=torch.long, device=device,
            )

            # Insert summary tokens
            summary_slice = slice(text_slice.stop, text_slice.stop + summary_num)
            new_tokens[batch_idx, summary_slice] = summary_ids
            # summary loss_mask stays 0 (initialized above)
            summary_mask[batch_idx, summary_slice] = True

            # Record summary positions in expanded sequence
            summary_positions = torch.arange(
                summary_slice.start, summary_slice.stop,
                dtype=torch.long, device=device,
            )

            # Set summary position IDs
            if config.summary_token_position_ids_type == 'zeros':
                new_position_ids[batch_idx, summary_slice] = 0
            elif config.summary_token_position_ids_type == 'last_chunk_slice_right':
                prev_text_end = i * chunk_text_len
                cur_text_end = min((i + 1) * chunk_size, seq_length)
                chunk_len = cur_text_end - prev_text_end

                idx = torch.arange(
                    1, summary_num + 1, device=device, dtype=torch.long,
                )
                slice_ends = prev_text_end + (idx * chunk_len) // summary_num - 1
                slice_ends = slice_ends.clamp(min=prev_text_end)
                new_position_ids[batch_idx, summary_slice] = slice_ends.to(
                    dtype=new_position_ids.dtype
                )
            elif config.summary_token_position_ids_type == 'last_chunk_slice_left':
                prev_text_end = i * chunk_text_len
                cur_text_end = min((i + 1) * chunk_size, seq_length)
                chunk_len = cur_text_end - prev_text_end

                idx = torch.arange(
                    0, summary_num, device=device, dtype=torch.long,
                )
                slice_ends = prev_text_end + (idx * chunk_len) // summary_num - 1
                slice_ends = slice_ends.clamp(min=prev_text_end)
                new_position_ids[batch_idx, summary_slice] = slice_ends.to(
                    dtype=new_position_ids.dtype
                )
            else:
                raise ValueError(
                    f'Unknown summary_token_position_ids_type: '
                    f'{config.summary_token_position_ids_type}'
                )

            # Build chunk metadata
            prefix_summary_positions = (
                torch.cat(accumulated_summary, dim=0)
                if accumulated_summary
                else torch.empty(0, dtype=torch.long, device=device)
            )

            chunk_meta = SummaryChunkMeta(
                text_positions=text_positions,
                summary_positions=summary_positions,
                prefix_summary_positions=prefix_summary_positions,
            )
            sample_chunks.append(chunk_meta)
            accumulated_summary.append(summary_positions)

            text_cursor += chunk_text_len
            write_cursor = summary_slice.stop

        sample_contexts.append(SummarySampleContext(chunks=sample_chunks))

    # Update cu_seqlens: map old document boundaries to new positions
    new_cu_seqlens = None
    if cu_seqlens is not None:
        new_cu_seqlens = torch.zeros_like(cu_seqlens)
        for i, boundary in enumerate(cu_seqlens):
            b = boundary.item()
            summaries_before = (b // chunk_size) * summary_num
            new_cu_seqlens[i] = b + summaries_before

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
