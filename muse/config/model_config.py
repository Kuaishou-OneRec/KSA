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
    summary_partial_chunk_policy: Literal["legacy", "summarize", "defer"] = Field(
        default="legacy",
        description="How to handle a final chunk shorter than summary_chunk_size. "
                    "'legacy' preserves V1 tail positions; 'summarize' uses the "
                    "actual tail boundaries and inserts summaries; 'defer' keeps "
                    "the partial text chunk open until more decode tokens complete it."
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
    summary_block_size: int = Field(
        default=4096,
        gt=0,
        description="Query block size used by native block-mode Summary Attention."
    )
    summary_meta_pool_mode: Literal[
        "none", "mean", "scalar", "dimension_wise"
    ] = Field(
        default="none",
        description="Second-level pooling for summaries outside the local text window. "
                    "'none' disables pooling, 'mean' uses uniform weights, and "
                    "'scalar' uses one learned score per summary token. "
                    "'dimension_wise' learns an independent score for every "
                    "KV coordinate of every summary token."
    )
    summary_meta_pool_group_size: Union[int, List[int]] = Field(
        default=0,
        description="Number of old summary entries pooled into one meta KV entry. "
                    "0 or 1 disables pooling for a layer; int applies to all layers; "
                    "List[int] configures each layer independently."
    )
    summary_meta_pool_position: Literal["group_last"] = Field(
        default="group_last",
        description="Position assigned to pooled meta keys after pre-RoPE pooling."
    )
    summary_meta_pool_mass_correction: Literal["none", "log_group_size"] = Field(
        default="none",
        description="Optional additive attention-logit correction for a meta KV. "
                    "'log_group_size' preserves the softmax mass of the entries "
                    "represented by one pooled key when their logits are equal."
    )
    summary_meta_pool_anneal: bool = Field(
        default=False,
        description="Training-only transition from the original summary-memory "
                    "attention output to the pooled meta-memory attention output."
    )
    summary_meta_pool_anneal_start_iter: int = Field(
        default=0,
        ge=0,
        description="Global iteration at which the summary-memory coefficient "
                    "starts decaying from 1.0."
    )
    summary_meta_pool_anneal_end_iter: int = Field(
        default=0,
        ge=0,
        description="Global iteration at which the summary-memory coefficient "
                    "reaches 0.0 and only meta memory is used."
    )
    summary_meta_pool_distill_attention: bool = Field(
        default=False,
        description="Training-only layer-local distillation from the original "
                    "KSA attention output to the meta-pooled attention output. "
                    "The pooled output remains the only path consumed by the model."
    )
    # Sparse Compressed Summary Attention (S-KSA).  These fields are deliberately
    # inert unless ``sksa_enabled`` is true so existing KSA/C-KSA configs keep
    # the exact same module graph and state dict.
    sksa_enabled: bool = Field(
        default=False,
        description="Enable S-KSA routing on explicitly selected dense/full KSA layers.",
    )
    sksa_sparse_full_layers: List[int] = Field(
        default_factory=list,
        description="Zero-based full-layer indices whose text attention is sparsified.",
    )
    sksa_routing_mode: Literal["summary_indexer"] = Field(
        default="summary_indexer",
        description="Layer-local summary-query block routing (S-KSA).",
    )
    sksa_indexer_token_budget: int = Field(
        default=2048,
        gt=0,
        description="Number of remote original-text tokens retained for each "
                    "Full-layer query, converted to a whole number of blocks.",
    )
    sksa_indexer_query_rank: int = Field(default=256, gt=0)
    sksa_indexer_heads: int = Field(default=4, gt=0)
    sksa_indexer_head_dim: int = Field(default=128, gt=0)
    sksa_indexer_rope_dim: int = Field(
        default=64,
        ge=0,
        description="Trailing Lightning-Indexer dimensions receiving RoPE.",
    )
    sksa_indexer_query_block_size: int = Field(
        default=512,
        gt=0,
        description="Execution tile for indexer scoring; routes remain per query.",
    )
    sksa_indexer_teacher_query_block_size: int = Field(default=32, gt=0)
    sksa_indexer_teacher_pooling: Literal["max", "sum"] = Field(
        default="max",
        description="Pool dense raw-token attention into one teacher score per "
                    "summary block. Max is the QSA-style default.",
    )
    sksa_indexer_summary_key_source: Literal[
        "input_summary", "chunk_attention"
    ] = Field(
        default="input_summary",
        description="Input used to build each summary-block index key. "
                    "input_summary projects summary hidden states directly; "
                    "chunk_attention lets the current Full "
                    "layer's summary query aggregate its raw block through "
                    "the configured auxiliary Q/K/V projections. This "
                    "auxiliary aggregation is detached from the backbone.",
    )
    sksa_indexer_chunk_attention_qk_source: Literal[
        "backbone", "independent"
    ] = Field(
        default="backbone",
        description="Q/K projections used by the auxiliary chunk-attention "
                    "summary-key path. 'backbone' preserves the existing "
                    "frozen Full-layer Q/K; 'independent' creates trainable "
                    "indexer-local Q/K and Q/K RMSNorm.",
    )
    sksa_indexer_chunk_attention_v_source: Literal[
        "backbone", "independent"
    ] = Field(
        default="backbone",
        description="Values aggregated by auxiliary chunk attention: backbone V "
                    "or an independent trainable value projection.",
    )
    sksa_indexer_chunk_attention_include_summary: bool = Field(
        default=True,
        description="Include the Summary's own K/V in whole-chunk index-key "
                    "aggregation. False uses only raw-text K/V at every block "
                    "size and requires independent chunk Q/K. The Summary "
                    "still supplies the query.",
    )
    sksa_indexer_scorer_backend: Literal["torch", "fused", "auto"] = Field(
        default="torch",
        description="Backend for Lightning-Indexer candidate scoring. Torch "
                    "preserves the reference implementation, fused selects the "
                    "dedicated kernel, and auto may choose either at runtime.",
    )
    sksa_indexer_block_size: Literal[2, 4, 8] = Field(
        default=8,
        description="Raw tokens per indexer key/selected block; independent of "
                    "summary insertion interval.",
    )
    sksa_indexer_force_sink: bool = Field(
        default=True,
        description="Force the leading routing candidates into every non-empty "
                    "remote budget.",
    )
    sksa_indexer_summary_sink_tokens: int = Field(
        default=8,
        gt=0,
        description="Leading raw tokens forced into summary-indexer routes; "
                    "independent of summary insertion interval and included in the remote budget.",
    )
    sksa_apply_sparse_attention: bool = Field(
        default=True,
        description="Apply the selected raw-block sparse Full attention. Dense "
                    "indexer warm-up overrides this to false.",
    )
    sksa_indexer_distill_attention: bool = Field(
        default=False,
        description="Return the layer-local Lightning-Indexer KL objective.",
    )
    sksa_local_window_tokens: int = Field(
        default=1024,
        gt=0,
        description="Nominal always-visible causal local window in original-text "
                    "tokens. Summary-indexer routing aligns its left edge down "
                    "to a raw summary-block boundary, so its effective span may "
                    "be up to summary_chunk_size-1 tokens longer.",
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

    @field_validator(
        "summary_layer_freq",
        "summary_sliding_chunk_num",
        "summary_meta_pool_group_size",
        mode="before",
    )
    @classmethod
    def parse_pattern_string(cls, v):
        """Parse pattern strings like '([1]*3+[0]*1)*9' into List[int]."""
        if isinstance(v, str) and '[' in v:
            return _eval_pattern(v)
        return v

    @model_validator(mode="after")
    def validate_meta_pooling(self) -> "Qwen3SummaryAttentionConfig":
        sizes = self.summary_meta_pool_group_size
        if isinstance(sizes, list):
            if len(sizes) != self.num_layers:
                raise ValueError(
                    f"summary_meta_pool_group_size list length ({len(sizes)}) != "
                    f"num_layers ({self.num_layers})"
                )
            active_sizes = sizes
        else:
            active_sizes = [sizes]

        if any(size < 0 for size in active_sizes):
            raise ValueError("summary_meta_pool_group_size values must be non-negative")

        pooling_active = (
            self.summary_meta_pool_mode != "none"
            and any(size > 1 for size in active_sizes)
        )
        if pooling_active:
            if self.summary_token_num != 1:
                raise ValueError(
                    "Hierarchical summary pooling currently requires summary_token_num=1"
                )
            if self.summary_attention_mode not in {"block", "kernel"}:
                raise ValueError(
                    "Hierarchical summary pooling currently requires "
                    "summary_attention_mode='block' or 'kernel'"
                )
        if self.summary_meta_pool_anneal:
            if not pooling_active:
                raise ValueError(
                    "summary_meta_pool_anneal requires active hierarchical "
                    "summary pooling"
                )
            if (
                self.summary_meta_pool_anneal_end_iter
                <= self.summary_meta_pool_anneal_start_iter
            ):
                raise ValueError(
                    "summary_meta_pool_anneal_end_iter must be greater than "
                    "summary_meta_pool_anneal_start_iter"
                )
        if self.summary_meta_pool_distill_attention:
            if not pooling_active:
                raise ValueError(
                    "summary_meta_pool_distill_attention requires active "
                    "hierarchical summary pooling"
                )
            if self.summary_meta_pool_anneal:
                raise ValueError(
                    "summary_meta_pool_distill_attention cannot be combined "
                    "with summary_meta_pool_anneal"
                )

        if self.sksa_enabled:
            if not self.sksa_sparse_full_layers:
                raise ValueError(
                    "sksa_enabled requires at least one sksa_sparse_full_layers entry"
                )
            if len(set(self.sksa_sparse_full_layers)) != len(
                self.sksa_sparse_full_layers
            ):
                raise ValueError("sksa_sparse_full_layers must not contain duplicates")
            if self.summary_chunk_position_ids_type != "origin":
                raise ValueError(
                    "S-KSA currently requires summary_chunk_position_ids_type='origin'"
                )
            if self.summary_token_num != 1:
                raise ValueError("S-KSA currently requires summary_token_num=1")
            if (
                self.sksa_indexer_chunk_attention_qk_source == "independent"
                and (
                    (self.sksa_indexer_summary_key_source != 'chunk_attention')
                )
            ):
                raise ValueError(
                    "independent chunk Q/K requires "
                    "summary-indexer routing with "
                    "sksa_indexer_summary_key_source='chunk_attention'"
                )
            if (
                self.sksa_indexer_chunk_attention_v_source in {"independent"}
                and self.sksa_indexer_chunk_attention_qk_source != "independent"
            ):
                raise ValueError("independent chunk V requires independent chunk Q/K")
            if (
                (not self.sksa_indexer_chunk_attention_include_summary and self.sksa_indexer_chunk_attention_qk_source != 'independent')
            ):
                raise ValueError("raw-only chunk attention requires independent chunk Q/K")
            if self.summary_meta_pool_anneal:
                raise ValueError("S-KSA cannot be combined with meta-pool annealing")
            if self.summary_meta_pool_distill_attention:
                raise ValueError("S-KSA cannot be combined with meta-pool distillation")

            allowed_chunks = ((4, 8))
            if self.summary_chunk_size not in allowed_chunks:
                raise ValueError(
                    "S-KSA requires chunk4/8"
                )
            if self.summary_chunk_size % self.sksa_indexer_block_size:
                raise ValueError("indexer block size must divide the summary interval")
            if self.sksa_indexer_summary_sink_tokens % self.sksa_indexer_block_size:
                raise ValueError("summary sink tokens must cover whole indexer blocks")
            if self.sksa_indexer_block_size < self.summary_chunk_size:
                if self.sksa_indexer_chunk_attention_qk_source != "independent":
                    raise ValueError("subchunk indexer blocks require independent chunk Q/K")
                if self.sksa_local_window_tokens < self.summary_chunk_size:
                    raise ValueError("subchunk indexer requires a local window >= summary_chunk_size")
            if self.sksa_indexer_token_budget % self.sksa_indexer_block_size:
                raise ValueError(
                    "sksa_indexer_token_budget must be divisible by "
                    "sksa_indexer_block_size"
                )
            if (
                (self.sksa_indexer_force_sink and self.sksa_indexer_token_budget < self.sksa_indexer_summary_sink_tokens)
            ):
                raise ValueError("indexer token budget must cover the summary sink tokens")
            if self.sksa_indexer_rope_dim > self.sksa_indexer_head_dim:
                raise ValueError(
                    "sksa_indexer_rope_dim cannot exceed indexer_head_dim"
                )
            if self.sksa_indexer_rope_dim % 2:
                raise ValueError("sksa_indexer_rope_dim must be even")

            layer_is_summary = self.get_layer_is_summary()
            for target in self.sksa_sparse_full_layers:
                if not 0 <= target < self.num_layers:
                    raise ValueError(f"S-KSA full layer {target} is out of range")
                if not layer_is_summary[target]:
                    raise ValueError(
                        "Sparse routing targets KSA full layers; "
                        f"layer {target} is configured as a summary-bypass layer"
                    )
                if self.get_layer_meta_pool_group_size(target) > 1:
                    raise ValueError(
                        f"S-KSA target layer {target} is compressed, not a full layer"
                    )
                covered_tokens = (
                    self.get_layer_sliding_chunk_num(target)
                    * self.summary_chunk_size
                )
                if covered_tokens < self.max_seq_len:
                    raise ValueError(
                        f"S-KSA target layer {target} is not globally dense: its "
                        f"configured window covers {covered_tokens} < {self.max_seq_len} tokens"
                    )

        return self

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
    def get_layer_meta_pool_group_size(self, layer_idx: int) -> int:
        """Return the second-level summary pooling group size for one layer."""
        val = self.summary_meta_pool_group_size
        if isinstance(val, list):
            assert len(val) == self.num_layers, (
                f"summary_meta_pool_group_size list length ({len(val)}) != "
                f"num_layers ({self.num_layers})"
            )
            return val[layer_idx]
        return val


    def get_sksa_indexer_topk_blocks(self) -> int:
        """Convert the configured raw-token budget to summary-block top-k."""
        return (
            self.sksa_indexer_token_budget + self.sksa_indexer_block_size - 1
        ) // self.sksa_indexer_block_size

    def get_sksa_indexer_topk_candidates(self) -> int:
        """Return the route-slot count for the configured Indexer mode."""
        return self.get_sksa_indexer_topk_blocks()
