from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

__all__ = [
    "SummaryChunkMeta",
    "SummarySampleContext",
    "SummaryBatchContext",
    "build_summary_context",
    "build_summary_sliding_context",
]


@dataclass
class SummaryChunkMeta:
    text_positions: torch.Tensor
    summary_positions: torch.Tensor
    prefix_summary_positions: torch.Tensor

    @property
    def window_positions(self) -> torch.Tensor:
        if self.prefix_summary_positions.numel() == 0:
            if self.summary_positions.numel() == 0:
                return self.text_positions
            return torch.cat((self.text_positions, self.summary_positions), dim=0)
        if self.summary_positions.numel() == 0:
            return torch.cat((self.prefix_summary_positions, self.text_positions), dim=0)
        return torch.cat(
            (self.prefix_summary_positions, self.text_positions, self.summary_positions),
            dim=0,
        )


@dataclass
class SummarySampleContext:
    chunks: List[SummaryChunkMeta]


@dataclass
class SummaryBatchContext:
    samples: List[SummarySampleContext]
    position_ids: torch.Tensor
    summary_mask: torch.Tensor

    @property
    def enabled(self) -> bool:
        return self.summary_mask.numel() > 0


def build_summary_context(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    summary_chunk_size: int,
    summary_token_num: int,
    summary_token_begin: int,
) -> SummaryBatchContext:
    """
    Build SummaryBatchContext from already-expanded sequences: each chunk should
    be text tokens (<= chunk_size) followed by summary_token_num summary tokens.
    """
    batch_size, seq_len = input_ids.shape
    block_size = summary_chunk_size + summary_token_num

    summary_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    samples: List[SummarySampleContext] = []

    for b in range(batch_size):
        chunks: List[SummaryChunkMeta] = []
        prefix_summary_positions: List[torch.Tensor] = []
        cursor = 0
        while cursor < seq_len:
            text_len = min(summary_chunk_size, seq_len - cursor)
            if text_len <= 0:
                break

            text_positions = torch.arange(cursor, cursor + text_len, device=input_ids.device)
            summary_start = cursor + text_len
            summary_end = min(cursor + block_size, seq_len)

            # Keep only true summary tokens (in case of ragged last block).
            summary_positions = torch.arange(summary_start, summary_end, device=input_ids.device)
            if summary_positions.numel() > 0:
                summary_tokens = input_ids[b, summary_positions]
                valid = (summary_tokens >= summary_token_begin) & (
                    summary_tokens < summary_token_begin + summary_token_num
                )
                summary_positions = summary_positions[valid]
                if summary_positions.numel() > 0:
                    summary_mask[b, summary_positions] = True

            prefix_tensor = (
                torch.cat(prefix_summary_positions, dim=0)
                if prefix_summary_positions
                else torch.empty(0, device=input_ids.device, dtype=torch.long)
            )

            chunk_meta = SummaryChunkMeta(
                text_positions=text_positions,
                summary_positions=summary_positions,
                prefix_summary_positions=prefix_tensor,
            )
            chunks.append(chunk_meta)
            if summary_positions.numel() > 0:
                prefix_summary_positions.append(summary_positions)

            cursor += block_size

        samples.append(SummarySampleContext(chunks=chunks))

    return SummaryBatchContext(
        samples=samples,
        position_ids=position_ids,
        summary_mask=summary_mask,
    )


def build_summary_sliding_context(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    summary_token_num: int,
    summary_token_begin: int,
) -> SummaryBatchContext:
    summary_mask = (input_ids >= summary_token_begin) & (
        input_ids < summary_token_begin + summary_token_num
    )
    return SummaryBatchContext(
        samples=[],
        position_ids=position_ids,
        summary_mask=summary_mask,
    )
