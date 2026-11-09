from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "SummaryBatchContext",
    "build_summary_sliding_context",
]


@dataclass
class SummaryBatchContext:
    position_ids: torch.Tensor
    summary_mask: torch.Tensor
    text_positions: torch.Tensor | None = None
    summary_positions: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self.summary_mask.numel() > 0


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
        position_ids=position_ids,
        summary_mask=summary_mask,
    )
