"""
Layer Normalization with FP32 Precision.

This module provides a LayerNorm wrapper that performs normalization in FP32
for numerical stability during mixed-precision training.

When training with bfloat16 or float16, normalization operations can suffer
from numerical instability. This implementation temporarily upcasts to FP32,
performs the normalization, then casts back to the original dtype.

Classes:
    Fp32LayerNorm: LayerNorm with FP32 computation

Example:
    >>> import torch
    >>> from muse.layers.layer_norm import Fp32LayerNorm
    >>> 
    >>> # Create FP32 LayerNorm
    >>> norm = Fp32LayerNorm(normalized_shape=768, eps=1e-5)
    >>> 
    >>> # Input in bfloat16
    >>> x = torch.randn(16, 128, 768, dtype=torch.bfloat16)
    >>> 
    >>> # Normalization computed in FP32, output in bfloat16
    >>> normalized = norm(x)  # (16, 128, 768) in bfloat16
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

import torch
from torch import nn


class Fp32LayerNorm(nn.LayerNorm):
    """
    Wrapper around :class:`~torch.nn.LayerNorm` to support mixed-precision training.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The normalized output tensor having the same shape as ``x``.
        """
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)