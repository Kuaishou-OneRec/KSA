"""
Rotary Positional Embeddings (RoPE) Implementation.

This module provides implementations of Rotary Positional Embeddings (RoPE),
a relative positional encoding method that applies rotations to query and key
vectors in attention mechanisms.

RoPE was introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
(Su et al., 2021) and is used in modern LLMs like LLaMA, GPT-NeoX, and PaLM.

Key advantages of RoPE:
- Encodes relative positions rather than absolute positions
- Naturally extends to sequences longer than those seen during training
- Works well with modern attention mechanisms like Flash Attention
- No need for learned positional embeddings

Classes:
    LlamaRotaryPositionalEmbeddings: LLaMA-style RoPE with cached frequencies
    RotaryPositionalEmbeddings: Generic RoPE implementation  
    Qwen2RotaryEmbedding: Qwen2-style RoPE with NTK-aware interpolation

Example:
    >>> import torch
    >>> from muse.layers.position_embeddings import LlamaRotaryPositionalEmbeddings
    >>> 
    >>> # Create RoPE layer
    >>> rope = LlamaRotaryPositionalEmbeddings(
    ...     dim=128,  # head_dim
    ...     max_seq_len=4096,
    ...     base=10000
    ... )
    >>> 
    >>> # Apply to queries/keys
    >>> q = torch.randn(batch, seq_len, num_heads, head_dim)
    >>> q_rotated = rope(q)  # Same shape, with positional info encoded
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch
from torch import nn

class LlamaRotaryPositionalEmbeddings(nn.Module):
    """
    Llama-style Rotary Positional Embeddings (RoPE).

    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L132

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    This is the Llama-style implementation where cos and sin are interleaved
    in the dimension.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).

    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    This implementation uses separated cos and sin (upper half for cos, lower half for sin),
    which is different from the Llama-style interleaved implementation.

    Reference implementation can be found in HuggingFace transformers:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        """Build RoPE cache for HuggingFace-style (separated) implementation."""
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # For HF style: concatenate freqs to get [max_seq_len, dim]
        # This matches HF's implementation: emb = torch.cat((freqs, freqs), dim=-1)
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)  # [max_seq_len, dim]

        # cache includes both the cos and sin components
        # Compute cos and sin in float32, then store as float32
        # output shape is [max_seq_len, dim, 2] where [..., 0] is cos and [..., 1] is sin
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cache = torch.stack([cos, sin], dim=-1)  # [max_seq_len, dim, 2]
        # Store cache as float32 for precision, will convert to target dtype in forward
        self.register_buffer("cache", cache, persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input.
        
        Args:
            x: Input tensor with shape [..., h_d]
            
        Returns:
            Rotated tensor with shape [..., h_d]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        b, seq_len, n_h, h_d = x.shape

        # extract the values based on whether input_pos is set or not
        # cache shape: [max_seq_len, h_d, 2]
        if input_pos is None:
            # Use sequential positions [0, 1, ..., seq_len-1]
            rope_cache = self.cache[:seq_len]  # [s, h_d, 2]
            # Expand to [1, s, 1, h_d, 2] for broadcasting
            rope_cache = rope_cache.unsqueeze(0).unsqueeze(2)  # [1, s, 1, h_d, 2]
        else:
            # input_pos shape: [b, s] - each element is a position index
            # Index cache for each position: [b, s, h_d, 2]
            rope_cache = self.cache[input_pos]  # [b, s, h_d, 2]
            # Expand to [b, s, 1, h_d, 2] for broadcasting
            rope_cache = rope_cache.unsqueeze(2)  # [b, s, 1, h_d, 2]

        # Separate cos and sin
        # rope_cache shape: [b, s, 1, h_d, 2] or [1, s, 1, h_d, 2]
        cos = rope_cache[..., 0].to(dtype=x.dtype)  # [b, s, 1, h_d] or [1, s, 1, h_d]
        sin = rope_cache[..., 1].to(dtype=x.dtype)  # [b, s, 1, h_d] or [1, s, 1, h_d]

        # Apply RoPE: x_embed = (x * cos) + (rotate_half(x) * sin)
        x_rotated = self.rotate_half(x)
        x_out = (x * cos) + (x_rotated * sin)

        return x_out
