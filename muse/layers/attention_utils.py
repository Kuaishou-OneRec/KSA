"""
Attention Function Utilities and Implementations.

This module provides a unified interface and multiple implementations for attention
mechanisms, including standard eager attention and Flash Attention 2.

The module defines an AttentionFunction protocol that all attention implementations
must follow, ensuring consistent APIs across different attention variants. This
allows easy swapping between implementations without changing model code.

Key features:
- Protocol-based design for pluggable attention functions
- EagerAttention: Standard PyTorch attention implementation
- FlashAttention2: Memory-efficient attention using Flash Attention
- Support for causal masking, custom masks, and dropout
- Variable-length sequence support (with cu_seqlens)

Classes:
    AttentionFunction: Protocol defining attention interface
    EagerAttention: Standard attention implementation
    FlashAttention2: Flash Attention 2 implementation
    FlashAttentionVarlen: Variable-length Flash Attention

Functions:
    get_attention_function: Factory for creating attention functions by name

Example:
    >>> from muse.layers.attention_utils import get_attention_function
    >>> 
    >>> # Get Flash Attention 2
    >>> attn_fn = get_attention_function("flash_attention_2")
    >>> 
    >>> # Use in forward pass
    >>> q = torch.randn(batch, seq_len, num_heads, head_dim)
    >>> k = torch.randn(batch, seq_len, num_heads, head_dim)
    >>> v = torch.randn(batch, seq_len, num_heads, head_dim)
    >>> output = attn_fn(q, k, v, is_causal=True)
"""

from typing import Protocol, Optional, Any
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func


class AttentionFunction(Protocol):
    """Unified attention function interface protocol
    
    All functions implementing attention mechanisms should follow this interface signature.
    """
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor, 
                v: torch.Tensor,
                is_causal: bool = False,
                attn_dropout: float = 0.0,
                **kwargs: Any) -> torch.Tensor:
        """Forward propagation function
        
        Args:
            q: Query tensor, shape: (b, s_q, n_h, h_d)
            k: Key tensor, shape: (b, s_kv, n_h, h_d)
            v: Value tensor, shape: (b, s_kv, n_h, h_d)
            is_causal: Whether to use causal mask (only see information before current position)
            attn_dropout: Dropout probability for attention weights
            **kwargs: Other optional parameters, such as attention mask, positional encoding, training mode, etc.
            
        Returns:
            torch.Tensor: Attention output, same shape as input
        """
        ...


# Example: Concrete implementation conforming to AttentionFunction protocol
class EagerAttention:
    """Standard eager attention implementation, conforming to AttentionFunction protocol"""
    
    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 is_causal: bool = False,
                 attn_dropout: float = 0.0,
                 **kwargs: Any) -> torch.Tensor:
        """Make the class callable, delegates to forward method"""
        return self.forward(q, k, v, is_causal, attn_dropout, **kwargs)
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                is_causal: bool = False,
                attn_dropout: float = 0.0,
                **kwargs: Any) -> torch.Tensor:
        """Implements standard eager attention
        Args:
            q: Query tensor, shape: (b, s_q, n_h, h_d)
            k: Key tensor, shape: (b, s_k, n_h, h_d)
            v: Value tensor, shape: (b, s_v, n_h, h_d)
            is_causal: Whether to use causal mask (only see information before current position)
            attn_dropout: Dropout probability for attention weights
            **kwargs: Other optional parameters, such as attention mask, positional encoding, cu_seqlens, etc.
        """
        # Calculate attention scores
        h_d = q.size(-1)
        window_size = kwargs.get("window_size", -1)
        
        # Handle custom mask if provided
        mask = kwargs.get('mask', None)
        
        # Use einsum for efficient batch matrix multiplication: Q @ K^T
        # q: (b, s_q, n_h, h_d), k: (b, s_k, n_h, h_d) -> scores: (b, n_h, s_q, s_k)
        # Contract over h_d dimension, match n_h, compute s_q @ s_k
        scores = torch.einsum('bqnd, bknd -> bnqk', q, k) * \
            kwargs.get("softmax_scale", (h_d ** -0.5))
        
        # Apply custom mask (if provided)
        if mask is not None:
            # mask shape: [b, s_q, s_k] or [b, n_h, s_q, s_k]
            # scores shape: [b, n_h, s_q, s_k]
            # If mask doesn't have the head dimension, unsqueeze it
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [b, 1, s_q, s_k]
            scores = scores + mask
        
        # Apply causal mask
        if is_causal:
            assert mask is None, "Causal mask and custom mask are not supported together"
            # Flash Attention 2.1 style: right-bottom aligned causal mask
            # This supports incremental decoding where seqlen_q != seqlen_k
            s_q, s_k = q.size(1), k.size(1)
            # Show causal_mask:
            # q=2, k=2 (square, most common case):
            #   1 1
            #   1 1
            # q=2, k=4 (with common prefix):
            #   1 1 1 0
            #   1 1 1 1
            # q=4, k=2 (uncommon case, just keep same as flash attention 2.1 & sdpa):
            #   0 0
            #   0 0
            #   1 0
            #   1 1
            q_idx = torch.arange(s_q, device=q.device).unsqueeze(1)
            k_idx = torch.arange(s_k, device=q.device).unsqueeze(0)
            right_aligned_offset = s_k - s_q
            causal_mask = k_idx <= (q_idx + right_aligned_offset)
            if window_size is not None and window_size > 0:
                causal_mask &= k_idx > (q_idx + right_aligned_offset - window_size)
            # Mask out positions where causal_mask is False (invert for masked_fill)
            scores.masked_fill_(~causal_mask, -float('inf'))
            
        # Calculate attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if attn_dropout > 0.0:
            # Get training mode from kwargs, default to False (eval mode) for safety
            # This matches FlashAttention's behavior where dropout is automatically handled
            training = kwargs.get('training', False)
            attn_weights = F.dropout(attn_weights, p=attn_dropout, training=training)
        
        # Calculate output
        output = torch.einsum('bnqk, bnkd -> bnqd', attn_weights, v.transpose(1, 2))
        # transpose back to (b, s_q, n_h, h_d)
        return output.transpose(1, 2)

class FlashAttention2:
    """FlashAttention2 implementation, also conforming to AttentionFunction protocol"""
    
    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 is_causal: bool = False,
                 attn_dropout: float = 0.0,
                 **kwargs: Any) -> torch.Tensor:
        """Make the class callable, delegates to forward method"""
        return self.forward(q, k, v, is_causal, attn_dropout, **kwargs)
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                is_causal: bool = False,
                attn_dropout: float = 0.0,
                **kwargs: Any) -> torch.Tensor:
        """Flash Attention implementation (pseudo-code example)"""
        # Here you can call the actual Flash Attention implementation
        # Example: from flash_attn import flash_attn_func
        # return flash_attn_func(q, k, v, dropout_p=attn_dropout, causal=is_causal)

        cu_seqlens = kwargs.get("cu_seqlens")
        cu_seqlens_q, cu_seqlens_k = kwargs.get("cu_seqlens_q"), kwargs.get("cu_seqlens_k")

        if cu_seqlens is not None and cu_seqlens_q is None:
            cu_seqlens_q = cu_seqlens
        if cu_seqlens is not None and cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens
        
        max_seqlen_q = None
        max_seqlen_k = None
        if cu_seqlens_q is not None:
            max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        if cu_seqlens_k is not None:
            max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
            cu_seqlens_k = cu_seqlens_k.to(torch.int32)

        window_size = kwargs.get("window_size", -1)
        
        # FlashAttention doesn't automatically handle training mode, so we need to
        # set dropout_p to 0.0 during evaluation, matching the behavior of Qwen3Attention
        # Get training mode from kwargs, default to False (eval mode) for safety
        training = kwargs.get('training', False)
        dropout_p = attn_dropout if training else 0.0
        head_dim = q.size(-1)
        softmax_scale = kwargs.get("softmax_scale", head_dim ** -0.5)
        
        # Flash attention expects [batch, seq_len, num_heads, head_dim]
        if cu_seqlens_q is None or cu_seqlens_k is None:
            attn_output = flash_attn_func(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                window_size=(window_size, window_size),
                causal=is_causal
            )
        else:
            # if q,k,v is packing, the batch dimension is 1, so we need to squeeze it
            attn_output = flash_attn_varlen_func(
                q=q.squeeze(0),
                k=k.squeeze(0),
                v=v.squeeze(0),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                window_size=(window_size, window_size),
                causal=is_causal
            )
            attn_output = attn_output.unsqueeze(0)
        # TODO: support return_attn_weights
        return attn_output

# Usage example: Factory function
def get_attention_function(attention_type: str) -> AttentionFunction:
    """Factory function that returns the corresponding attention function based on type"""
    if attention_type == "eager":
        return EagerAttention()
    elif attention_type == "flash_attention_2":
        return FlashAttention2()
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
