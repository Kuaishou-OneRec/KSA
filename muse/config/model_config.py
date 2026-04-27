"""
Model Configuration Classes.

This module defines configuration classes for model architectures, using Pydantic
for validation and type checking. All model configurations inherit from BaseConfig
and follow a structured, type-safe approach to defining model hyperparameters.

The module provides:
- Base ModelConfig class for common model properties
- Qwen3Config for Qwen3 architecture
- Automatic validation of configuration values
- Serialization to/from JSON

Classes:
    ModelConfig: Base configuration for all models
    Qwen3Config: Configuration for Qwen3 transformer models

Example:
    >>> from muse.config.model_config import Qwen3Config
    >>> 
    >>> # Create configuration
    >>> config = Qwen3Config(
    ...     vocab_size=151936,
    ...     embed_dim=4096,
    ...     num_layers=32,
    ...     num_heads=32,
    ...     num_kv_heads=32,
    ...     intermediate_dim=11008
    ... )
    >>> 
    >>> # Save to file
    >>> config.save("model_config.json")
    >>> 
    >>> # Load from file
    >>> loaded_config = Qwen3Config.from_json_file("model_config.json")
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model configuration classes."""

import re
from typing import Optional, Literal, List, Tuple, Union
from pydantic import Field, field_validator, model_validator

from muse.config.base import BaseConfig


def _eval_pattern(pattern: str) -> List[int]:
    """Evaluate a string containing a Python list expression.

    Only allows safe characters: digits, commas, brackets, parens, +, *.
    Examples:
        "([4096]*1+[128]*3)*9" → 36 elements
        "([1]*3+[0]*1)*9" → 36 elements
    """
    if not isinstance(pattern, str):
        raise ValueError(f"Expected str, got {type(pattern)}")
    if bool(re.compile(r'[^,\d\[\]\(\)\+\*\s]').search(pattern)):
        raise ValueError(f"Invalid pattern: {pattern}")
    return eval(pattern)


class ModelConfig(BaseConfig):
    """Base model configuration.
    
    This serves as the base class for all model-specific configurations.
    """
    
    # Model identification
    model_class: str = Field(
        description="Model class name (e.g., 'Qwen3Model')"
    )


class Qwen3Config(ModelConfig):
    """Configuration for Qwen3 model architecture.
    
    This configuration is specific to the Qwen3 model family.
    """
    
    # Architecture dimensions
    vocab_size: int = Field(
        default=151936,
        description="Vocabulary size"
    )
    embed_dim: int = Field(
        default=4096,
        description="Hidden dimension size"
    )
    num_layers: int = Field(
        default=32,
        description="Number of transformer layers"
    )
    tie_word_embeddings: bool = Field(
        default=True,
        description="Whether to tie the word embeddings"
    )
    hidden_act: str = Field(
        default="silu",
        description="Activation function for MLP (e.g., silu, gelu, gelu_pytorch_tanh)"
    )
    # Attention configuration
    num_heads: int = Field(
        default=32,
        description="Number of attention heads"
    )
    num_kv_heads: int = Field(
        default=32,
        description="Number of key-value heads for GQA/MQA"
    )
    head_dim: int = Field(
        default=128,
        description="Dimension of each attention head"
    )
    attn_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Attention dropout probability"
    )
    attention_function: Literal["eager", "flash_attention_2"] = Field(
        default="eager",
        description="Attention implementation to use"
    )
    q_proj_bias: bool = Field(
        default=False,
        description="Whether to use bias in the q_proj layer"
    )
    k_proj_bias: bool = Field(
        default=False,
        description="Whether to use bias in the k_proj layer"
    )
    v_proj_bias: bool = Field(
        default=False,
        description="Whether to use bias in the v_proj layer"
    )
    attention_bias: bool = Field(
        default=False,
        description="Whether to use bias terms in attention projections"
    )
    
    # Feed-forward configuration
    intermediate_dim: int = Field(
        default=11008,
        description="Intermediate size in FFN"
    )
    
    # Position embeddings
    max_seq_len: int = Field(
        default=32768,
        description="Maximum sequence length"
    )
    rope_base: float = Field(
        default=10000.0,
        description="RoPE theta parameter"
    )
    rope_impl: Literal["llama", "hf"] = Field(
        default="llama",
        description="RoPE implementation style: 'llama' (interleaved cos/sin) or 'hf' (separated cos/sin)"
    )
    rope_theta: float = Field(
        default=10000.0,
        description="RoPE theta parameter (alias for rope_base)"
    )
    rope_scaling: Optional[dict] = Field(
        default=None,
        description="RoPE scaling config (e.g., {'rope_type': 'default'})"
    )
    use_sliding_window: bool = Field(
        default=False,
        description="Enable sliding-window attention"
    )
    sliding_window: Optional[int] = Field(
        default=None,
        description="Global sliding window size when sliding-window attention is enabled"
    )
    layer_sliding_window: Optional[Union[int, List[int]]] = Field(
        default=None,
        description="Per-layer sliding window override. int = same window on every layer. "
                    "List[int] = per-layer values. Values <= 0 mean full attention on that layer. "
                    "Accepts pattern strings in JSON, e.g. '([0]*4+[4096]*8+[0]*4)*2'. "
                    "When set, overrides sliding_window."
    )
    
    # Normalization
    norm_eps: float = Field(
        default=1e-6,
        description="RMS normalization epsilon"
    )
    rms_norm_eps: float = Field(
        default=1e-6,
        description="Alias for RMS normalization epsilon"
    )
    q_norm: bool = Field(
        default=True,
        description="Whether to use normalization in the q_proj layer"
    )
    k_norm: bool = Field(
        default=True,
        description="Whether to use normalization in the k_proj layer"
    )
    eos_token_id: Optional[int] = Field(
        default=151645,
        description="End-of-sequence token ID"
    )

    @field_validator("num_heads")
    @classmethod
    def validate_num_heads(cls, v, info):
        """Validate that num_heads is divisible by num_kv_heads."""
        if "num_kv_heads" in info.data:
            num_kv_heads = info.data["num_kv_heads"]
            if v % num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({v}) must be divisible by "
                    f"num_kv_heads ({num_kv_heads})"
                )
        return v
    
    @field_validator("head_dim")
    @classmethod
    def validate_head_dim(cls, v, info):
        """Validate that embed_dim equals num_heads * head_dim."""
        return v

    @field_validator("layer_sliding_window", mode="before")
    @classmethod
    def parse_layer_sliding_window(cls, v):
        """Parse pattern strings like '([0]*4+[4096]*8)*2' into List[int]."""
        if isinstance(v, str) and '[' in v:
            return _eval_pattern(v)
        return v

    @model_validator(mode="after")
    def validate_head_relationships(cls, values: "Qwen3Config") -> "Qwen3Config":
        """Ensure head-related fields stay consistent after initialization."""
        if values.num_heads % values.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({values.num_heads}) must be divisible by "
                f"num_kv_heads ({values.num_kv_heads})"
            )
        return values

    @property
    def sliding_window_enabled(self) -> bool:
        """Whether any sliding-window attention configuration is active."""
        if self.layer_sliding_window is not None:
            return True
        return self.use_sliding_window and self.sliding_window is not None and self.sliding_window > 0

    def get_layer_sliding_window(self, layer_idx: int) -> int:
        """Return the effective sliding window size for a specific layer.

        Returns:
            Positive int for sliding-window attention, or -1 for full attention.
        """
        if not self.sliding_window_enabled:
            return -1

        val = self.layer_sliding_window
        if isinstance(val, list):
            assert len(val) == self.num_layers, (
                f"layer_sliding_window list length ({len(val)}) != num_layers ({self.num_layers})"
            )
            window = val[layer_idx]
        elif isinstance(val, int):
            window = val
        else:
            window = self.sliding_window

        if window is None or window <= 0:
            return -1
        return int(window)


class Qwen3SummaryAttentionConfig(Qwen3Config):
    """Qwen3Config with Summary Attention support.

    Extends Qwen3Config with fields for the summary attention mechanism:
    divides sequences into fixed-size chunks, inserts summary tokens after
    each chunk to compress context for subsequent chunks.

    Field names are aligned with the megatron-trained HF config where possible
    (e.g. summary_token_num, summary_token_begin).
    """

    # Summary Attention
    use_summary_attention: bool = Field(
        default=False,
        description="Explicit switch to enable summary attention."
    )
    summary_chunk_size: int = Field(
        default=0,
        description="Number of text tokens per chunk. 0 = summary attention disabled."
    )
    summary_token_num: int = Field(
        default=0,
        description="Number of summary tokens inserted after each chunk."
    )
    summary_sliding_chunk_num: Union[int, List[int]] = Field(
        default=0,
        description="Number of previous chunks whose text tokens are visible to current text tokens. "
                    "int = same for all layers. List[int] = per-layer values. "
                    "Accepts pattern strings in JSON, e.g. '([4096]*1+[128]*3)*9'."
    )
    summary_token_begin: Optional[int] = Field(
        default=None,
        description="Starting vocab ID for summary tokens. Required when summary is enabled."
    )
    summary_layer_freq: Union[int, List[int]] = Field(
        default=0,
        description="Which layers use summary attention. 0 = all layers. "
                    "int N > 0 = every Nth layer uses standard causal attention (rest use SA). "
                    "List[int] = per-layer pattern (1=summary, 0=causal). "
                    "Accepts pattern strings in JSON, e.g. '([1]*3+[0]*1)*9'."
    )
    summary_independent_parameters: bool = Field(
        default=False,
        description="Use independent embedding for summary tokens (placeholder ID=0, "
                    "replaced by summary_embedding in forward). Controls summary_embedding creation."
    )
    summary_independent_qkv: Optional[bool] = Field(
        default=None,
        description="Use independent QKV projections for summary tokens in attention layers. "
                    "Defaults to summary_independent_parameters if not set. "
                    "Set to false after mixoff to skip independent QKV while keeping summary_embedding."
    )
    summary_independent_attention_layernorm: bool = Field(
        default=False,
        description="Use independent layer norm (RMSNorm) for summary tokens before QKV projection. "
                    "Only effective when summary_independent_parameters=True. "
                    "When False, reuses the main QKV layer's norm weights on summary tokens."
    )
    summary_independent_qk_norm: bool = Field(
        default=False,
        description="Use independent QK norms (q_norm_summary/k_norm_summary) for summary tokens. "
                    "Only effective when summary_independent_parameters=True and q_norm is enabled. "
                    "When False (default), summary Q/K reuse the shared q_norm/k_norm (aligns with Megatron). "
                    "When True, creates separate RMSNorm modules initialized to ones."
    )
    summary_mix_qkv: bool = Field(
        default=False,
        description="Enable linear annealing from summary QKV to base QKV during continue training. "
                    "Requires summary_independent_parameters=True."
    )
    summary_mix_start_iter: int = Field(
        default=0,
        description="Iteration at which mix coefficient starts decaying from 1.0."
    )
    summary_mix_end_iter: int = Field(
        default=0,
        description="Iteration at which mix coefficient reaches 0.0 (pure base QKV)."
    )
    summary_chunk_position_ids_type: str = Field(
        default="inner_chunk",
        description="Position ID strategy for text tokens: "
                    "'inner_chunk' = restart from 0 per chunk, "
                    "'origin' = preserve global position."
    )
    summary_token_position_ids_type: str = Field(
        default="zeros",
        description="Position ID strategy for summary tokens: "
                    "'zeros' = always 0 (RoPE has no rotation), "
                    "'last_chunk_slice_right' = map to chunk text range endpoints, "
                    "'last_chunk_slice_left' = map to chunk text range start points."
    )
    summary_attention_mode: str = Field(
        default="kernel",
        description="Attention computation mode: "
                    "'kernel' = CUDA summary_attn_func (requires summary_attn package), "
                    "'mask' = native PyTorch full-mask mode (F.scaled_dot_product_attention, O(n²) memory), "
                    "'block' = native PyTorch block mode (block-wise compact masks with caching, memory efficient)."
    )
    enable_summary_distill_attention: bool = Field(
        default=False,
        description="Collect per-layer attention outputs for distillation."
    )
    summary_distill_attention_norm: bool = Field(
        default=False,
        description="Apply RMS normalization to distillation attention outputs."
    )

    @property
    def summary_enabled(self) -> bool:
        """Whether summary attention is active."""
        return self.use_summary_attention and self.summary_chunk_size > 0 and self.summary_token_num > 0

    @property
    def use_independent_qkv(self) -> bool:
        """Whether to use independent QKV projections for summary tokens.
        Defaults to summary_independent_parameters if summary_independent_qkv is None."""
        if self.summary_independent_qkv is not None:
            return self.summary_independent_qkv
        return self.summary_independent_parameters

    @field_validator("summary_layer_freq", "summary_sliding_chunk_num", mode="before")
    @classmethod
    def parse_pattern_string(cls, v):
        """Parse pattern strings like '([1]*3+[0]*1)*9' into List[int]."""
        if isinstance(v, str) and '[' in v:
            return _eval_pattern(v)
        return v

    def get_layer_is_summary(self) -> List[bool]:
        """Return per-layer bool indicating whether the layer uses summary attention.

        - freq=0 → all layers are summary
        - int N>0 → every Nth layer (0-indexed: i%N==0) is summary, rest are causal
        - List[int] → bitmask (1=summary, 0=causal)
        """
        freq = self.summary_layer_freq
        if isinstance(freq, list):
            assert len(freq) == self.num_layers, (
                f"summary_layer_freq list length ({len(freq)}) != num_layers ({self.num_layers})"
            )
            return [bool(x) for x in freq]
        if freq == 0:
            return [True] * self.num_layers
        return [(i % freq == 0) for i in range(self.num_layers)]

    def get_layer_sliding_chunk_num(self, layer_idx: int) -> int:
        """Return sliding_chunk_num for a specific layer.

        - int → same for all layers
        - List[int] → per-layer value by index
        """
        val = self.summary_sliding_chunk_num
        if isinstance(val, list):
            assert len(val) == self.num_layers, (
                f"summary_sliding_chunk_num list length ({len(val)}) != num_layers ({self.num_layers})"
            )
            return val[layer_idx]
        return val


