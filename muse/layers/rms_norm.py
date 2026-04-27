"""
Root Mean Square Layer Normalization (RMSNorm).

This module implements RMSNorm, a simpler and more efficient alternative to
LayerNorm that normalizes using only the root mean square (without mean centering).

RMSNorm was introduced in "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
and is used in modern architectures like LLaMA, GPT-NeoX, and T5.

Formula:
    RMSNorm(x) = x / RMS(x) * scale
    where RMS(x) = sqrt(mean(x^2) + eps)

Advantages over LayerNorm:
- Simpler computation (no mean subtraction)
- Slightly faster
- Often achieves similar or better performance
- Computation is done in FP32 for numerical stability

Classes:
    RMSNorm: RMSNorm layer with learnable scale parameter

Functions:
    rms_norm: Functional version without learnable parameters

Example:
    >>> import torch
    >>> from muse.layers.rms_norm import RMSNorm
    >>> 
    >>> # Create RMSNorm layer
    >>> norm = RMSNorm(dim=768, eps=1e-6)
    >>> 
    >>> x = torch.randn(16, 128, 768)  # (batch, seq_len, dim)
    >>> normalized = norm(x)  # (16, 128, 768)
    >>> 
    >>> # Functional version
    >>> from muse.layers.rms_norm import rms_norm
    >>> normalized = rms_norm(x, eps=1e-6)
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization in fp32.

    See: https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale


def rms_norm(x: torch.Tensor, eps: float = 1e-6):
    """
    This is just a functional RMSNorm without the trainable scale parameter.

    Args:
        x (torch.Tensor): input tensor to normalize
        eps (float): small value to avoid division by zero. Default: 1e-6

    Returns:
        torch.Tensor: The normalized tensor having the same shape as ``x``.

    """
    x_fp32 = x.float()
    x_normed = (
        x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    ).type_as(x)
    return x_normed