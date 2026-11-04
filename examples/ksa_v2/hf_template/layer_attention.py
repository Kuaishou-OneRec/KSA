# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Any, Callable, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from .hierarchical_summary_attention import (
    hierarchical_summary_attention_prefill,
    summary_attention_prefill,
)
from .sksa_attention import (
    SummaryBlockLightningIndexer,
    shared_indexer_step,
    shared_indexer_positions,
    aligned_summary_block_local_start,
    gather_summary_block_decode_kv,
    score_indexer_candidates,
    summary_chunk_attention_indexer_key_input,
    summary_block_sparse_prefill_attention,
)
from .summary_context import SummaryBatchContext, build_summary_sliding_context
try:
    from .prefill_attention import (
        hierarchical_summary_attn_prefill as hierarchical_summary_attn_func,
        summary_attn_prefill as summary_attn_func,
    )
except ImportError:
    hierarchical_summary_attn_func = None
    summary_attn_func = None
from .runtime_config import _get_shared_batch_mask, _parse_config_pattern
from .layer_norm_rope import Qwen3RMSNorm, apply_rotary_pos_emb, rotate_half
from .cache_ring_buffer import Qwen3RingBufferCache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    # Keep the original non-GQA Flash computation while avoiding repeat_kv's
    # sequence-sized copies. With batch=1 and one query, the GQA group axis can
    # be represented as a batch axis; each group broadcasts the same KV heads.
    # Native enable_gqa packs these queries differently and changes BF16 outputs.
    groups = module.num_key_value_groups
    if (attention_mask is None and not module.training and not torch.is_grad_enabled() and query.is_cuda
            and query.shape[0] == key.shape[0] == value.shape[0] == 1
            and query.shape[2] == 1 and groups > 1
            and query.shape[1] == key.shape[1] * groups
            and torch.backends.cuda.flash_sdp_enabled()):
        kv_heads, head_dim = key.shape[1], query.shape[-1]
        grouped_query = query.reshape(1, kv_heads, groups, 1, head_dim).permute(
            2, 1, 0, 3, 4).reshape(groups, kv_heads, 1, head_dim)
        grouped_key = key.expand(groups, -1, -1, -1)
        grouped_value = value.expand(groups, -1, -1, -1)
        params = torch.backends.cuda.SDPAParams(
            grouped_query, grouped_key, grouped_value, None, dropout, False, False)
        if torch.backends.cuda.can_use_flash_attention(params):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                grouped_output = F.scaled_dot_product_attention(
                    grouped_query, grouped_key, grouped_value,
                    attn_mask=None, dropout_p=dropout, is_causal=False)
            attn_output = grouped_output.permute(1, 0, 2, 3).reshape(
                1, query.shape[1], 1, head_dim)
            return attn_output.transpose(1, 2).contiguous(), None

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)


        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = _sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3SummaryAttention(Qwen3Attention):
    """
    Summary-aware variant of Qwen3Attention: uses a sliding summary mask.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.summary_chunk_size = getattr(self.config, "summary_chunk_size", 0)
        self.summary_token_num = getattr(self.config, "summary_token_num", 0)

        # Cache sliding_chunk_num to avoid eval() on every forward call
        val = getattr(config, "summary_sliding_chunk_num", 0) or 0
        val = _parse_config_pattern(val)
        if isinstance(val, list):
            self._sliding_chunk_num = val[layer_idx]
        else:
            self._sliding_chunk_num = int(val)

        pool_mode = getattr(config, "summary_meta_pool_mode", "none") or "none"
        group_val = _parse_config_pattern(
            getattr(config, "summary_meta_pool_group_size", 0) or 0
        )
        self._meta_pool_group_size = (
            int(group_val[layer_idx]) if isinstance(group_val, list) else int(group_val)
        )
        self._meta_pool_mode = pool_mode
        self._meta_pool_enabled = (
            pool_mode in ("mean", "scalar", "dimension_wise")
            and self._meta_pool_group_size > 1
        )
        self._meta_pool_mass_correction = (
            getattr(config, "summary_meta_pool_mass_correction", "none")
            == "log_group_size"
        )
        self._meta_pool_block_size = int(
            getattr(config, "summary_meta_pool_block_size", 512)
        )
        self._sksa_enabled = bool(getattr(config, "sksa_enabled", False))
        self._sksa_indexer_summary_key_source = getattr(
            config, "sksa_indexer_summary_key_source", "input_summary"
        )
        self._sksa_indexer_chunk_attention_qk_source = getattr(
            config, "sksa_indexer_chunk_attention_qk_source", "backbone"
        )
        self._sksa_sparse_full_layers = [
            int(layer)
            for layer in getattr(config, "sksa_sparse_full_layers", [])
        ]
        self._sksa_target_enabled = (
            self._sksa_enabled and layer_idx in self._sksa_sparse_full_layers
        )
        self._sksa_apply_sparse_attention = bool(
            getattr(config, "sksa_apply_sparse_attention", True)
        )
        routing_source_layers = set()
        all_group_sizes = (
            [int(value) for value in group_val]
            if isinstance(group_val, list)
            else [int(group_val)] * config.num_hidden_layers
        )
        if self._sksa_target_enabled:
            self._sksa_local_window_tokens = int(
                getattr(config, "sksa_local_window_tokens", 1024)
            )
            self._sksa_indexer_raw_block_size = (
                int(getattr(config, "sksa_indexer_block_size", 8))
            )
            self._sksa_indexer_local_alignment = (
                self._sksa_indexer_raw_block_size
            )
            raw_block_size = self._sksa_indexer_raw_block_size
            if self.summary_chunk_size != 8 or raw_block_size not in ((2, 4, 8)):
                raise ValueError("indexer requires summary interval 8 and raw block size 2/4/8")
            if True and raw_block_size < 8 and (
                self._sksa_indexer_chunk_attention_qk_source != "independent"
                or self._sksa_local_window_tokens < 8
            ):
                raise ValueError("subchunk indexer requires independent Q/K and local window >= 8")
            token_budget = int(config.sksa_indexer_token_budget)
            if token_budget % raw_block_size:
                raise ValueError("indexer token budget must be divisible by raw block size")
            self._sksa_indexer_selected_blocks = (
                token_budget + self._sksa_indexer_raw_block_size - 1
            ) // self._sksa_indexer_raw_block_size
            self._sksa_indexer_force_sink = bool(
                getattr(config, "sksa_indexer_force_sink", True)
            )
            if True and self._sksa_indexer_force_sink and token_budget < 8:
                raise ValueError("indexer token budget must cover 8 sink tokens")
            self.sksa_indexer = SummaryBlockLightningIndexer(
                config.hidden_size,
                query_rank=int(config.sksa_indexer_query_rank),
                num_heads=int(config.sksa_indexer_heads),
                head_dim=int(config.sksa_indexer_head_dim),
                rope_dim=int(config.sksa_indexer_rope_dim),
                rope_theta=float(
                    getattr(config, "rope_theta", None)
                    or config.rope_parameters["rope_theta"]
                ),
                max_seq_len=int(config.max_position_embeddings),
                query_block_size=int(config.sksa_indexer_query_block_size),
                scorer_backend=str(
                    getattr(
                        config,
                        "sksa_indexer_scorer_backend",
                        "torch",
                    )
                ),
                chunk_attention_qk_source=(
                    self._sksa_indexer_chunk_attention_qk_source
                ),
                chunk_num_heads=int(config.num_attention_heads),
                chunk_num_kv_heads=int(config.num_key_value_heads),
                chunk_head_dim=int(self.head_dim),
                chunk_norm_eps=float(config.rms_norm_eps),
                chunk_q_bias=bool(config.attention_bias),
                chunk_k_bias=bool(config.attention_bias),
                chunk_v_bias=bool(config.attention_bias),
                chunk_attention_v_source=getattr(config, "sksa_indexer_chunk_attention_v_source", "backbone"),
                chunk_attention_include_summary=getattr(
                    config, "sksa_indexer_chunk_attention_include_summary", True
                ),
            )
        if self._meta_pool_enabled:
            if self.summary_token_num != 1:
                raise ValueError("Hierarchical pooling currently requires summary_token_num=1")
            if getattr(config, "summary_meta_pool_position", "group_last") != "group_last":
                raise ValueError("Only group_last meta-summary positions are supported")
            if pool_mode in ("scalar", "dimension_wise"):
                scorer_width = (
                    1
                    if pool_mode == "scalar"
                    else config.num_key_value_heads * self.head_dim
                )
                self.meta_pool_scorer = nn.Linear(
                    config.hidden_size, scorer_width, bias=False
                )
                position_bias_shape = (
                    (self._meta_pool_group_size,)
                    if pool_mode == "scalar"
                    else (self._meta_pool_group_size, scorer_width)
                )
                self.meta_pool_position_bias = nn.Parameter(
                    torch.zeros(position_bias_shape)
                )


    def _get_sliding_chunk_num(self):
        return self._sliding_chunk_num

    def _build_meta_kv(
        self,
        hidden_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        summary_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        position_ids: Optional[torch.Tensor] = None,
        return_cache_state: bool = False,
        summary_positions: Optional[torch.Tensor] = None,
    ):
        """Pool summary K/V before RoPE and rotate at each group's last position."""
        shared_mask = _get_shared_batch_mask(summary_mask, "summary_mask")
        summary_pos = (summary_positions if summary_positions is not None else
                       torch.nonzero(shared_mask, as_tuple=False).flatten())
        summary_count = summary_pos.numel()
        group_size = self._meta_pool_group_size
        group_count = summary_count // group_size
        complete_end = group_count * group_size

        batch, kv_heads = key_states.shape[:2]
        head_dim = key_states.shape[-1]
        k_summary = key_states[:, :, summary_pos, :]
        v_summary = value_states[:, :, summary_pos, :]
        if self._meta_pool_mode in ("scalar", "dimension_wise"):
            x_summary_all = hidden_states[:, summary_pos, :]
            summary_scores = self.meta_pool_scorer(x_summary_all)
        else:
            summary_scores = key_states.new_zeros((batch, summary_count, 1))
        if group_count == 0:
            empty = key_states.new_empty((batch, kv_heads, 0, head_dim))
            if return_cache_state:
                if position_ids is None:
                    raise ValueError("hierarchical cache state requires position IDs")
                all_summary_position_ids = position_ids[:, summary_pos]
                return (
                    empty,
                    empty,
                    summary_pos,
                    summary_scores,
                    all_summary_position_ids,
                )
            return empty, empty, summary_pos, summary_scores

        k_grouped = k_summary[:, :, :complete_end, :].reshape(
            batch, kv_heads, group_count, group_size, head_dim
        )
        v_grouped = v_summary[:, :, :complete_end, :].reshape(
            batch, kv_heads, group_count, group_size, head_dim
        )
        if self._meta_pool_mode == "mean":
            weights = key_states.new_full(
                (batch, group_count, group_size), 1.0 / group_size
            )
            meta_k = (k_grouped * weights[:, None, :, :, None]).sum(dim=3)
            meta_v = (v_grouped * weights[:, None, :, :, None]).sum(dim=3)
        elif self._meta_pool_mode == "scalar":
            logits = summary_scores[:, :complete_end].reshape(
                batch, group_count, group_size
            )
            logits = logits + self.meta_pool_position_bias.view(1, 1, group_size)
            weights = torch.softmax(logits.float(), dim=-1).to(key_states.dtype)
            meta_k = (k_grouped * weights[:, None, :, :, None]).sum(dim=3)
            meta_v = (v_grouped * weights[:, None, :, :, None]).sum(dim=3)
        elif self._meta_pool_mode == "dimension_wise":
            kv_width = kv_heads * head_dim
            logits = summary_scores[:, :complete_end].reshape(
                batch, group_count, group_size, kv_width
            )
            logits = logits + self.meta_pool_position_bias.view(
                1, 1, group_size, kv_width
            )
            weights = torch.softmax(logits.float(), dim=2).to(key_states.dtype)
            weights = weights.view(
                batch, group_count, group_size, kv_heads, head_dim
            ).permute(0, 3, 1, 2, 4)
            meta_k = (k_grouped * weights).sum(dim=3)
            meta_v = (v_grouped * weights).sum(dim=3)
        else:
            raise RuntimeError(
                f"Unsupported meta-pool mode: {self._meta_pool_mode}"
            )
        last_summary_pos = summary_pos[group_size - 1 : complete_end : group_size]
        meta_cos = cos[:, last_summary_pos, :].unsqueeze(1)
        meta_sin = sin[:, last_summary_pos, :].unsqueeze(1)
        meta_k = meta_k * meta_cos + rotate_half(meta_k) * meta_sin
        if return_cache_state:
            if position_ids is None:
                raise ValueError("hierarchical cache state requires position IDs")
            all_summary_position_ids = position_ids[:, summary_pos]
            return (
                meta_k,
                meta_v,
                summary_pos,
                summary_scores,
                all_summary_position_ids,
            )
        return meta_k, meta_v, summary_pos, summary_scores

    def get_query_key_value_tensors(self, hidden_states):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        return query_states, key_states, value_states



    @torch.no_grad()
    def _build_summary_indexer_prefill_route(
        self,
        hidden_states: torch.Tensor,
        summary_mask: torch.Tensor,
        position_ids: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        sksa_aux_key_states: Optional[torch.Tensor] = None,
        sksa_aux_value_states: Optional[torch.Tensor] = None,
        text_positions: Optional[torch.Tensor] = None,
        summary_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project this Full layer's own text queries and routing keys."""
        shared_mask = _get_shared_batch_mask(summary_mask, "summary_mask")
        text_pos = (text_positions if text_positions is not None else
                    torch.nonzero(~shared_mask, as_tuple=False).flatten())
        summary_pos = (summary_positions if summary_positions is not None else
                       torch.nonzero(shared_mask, as_tuple=False).flatten())
        text_hidden = hidden_states.index_select(1, text_pos)
        if self._sksa_indexer_summary_key_source == "chunk_attention":
            if summary_pos.numel() % self.summary_token_num:
                raise ValueError(
                    "summary count is not divisible by summary_token_num"
                )
            complete_prefix = (
                summary_pos.numel()
                // self.summary_token_num
                * (self.summary_chunk_size + self.summary_token_num)
            )
            if getattr(
                self, "_sksa_indexer_chunk_attention_qk_source", "backbone"
            ) == "independent":
                if cos is None or sin is None:
                    raise ValueError(
                        "independent chunk Q/K prefill routing requires RoPE tensors"
                    )
                hidden_chunks = hidden_states[:, :complete_prefix].view(
                    hidden_states.shape[0],
                    -1,
                    self.summary_chunk_size + self.summary_token_num,
                    hidden_states.shape[-1],
                )
                value_chunks = repeat_kv(
                    value_states[:, :, :complete_prefix],
                    self.num_key_value_groups,
                ).transpose(1, 2).contiguous().view(
                    hidden_states.shape[0],
                    -1,
                    self.summary_chunk_size + self.summary_token_num,
                    self.config.num_attention_heads,
                    self.head_dim,
                )
                if self.sksa_indexer.chunk_v_proj is not None:
                    auxiliary_values = (
                        self.sksa_indexer.project_chunk_values(
                            hidden_states[:, :complete_prefix]
                        )
                        if sksa_aux_value_states is None
                        else sksa_aux_value_states[:, :, :complete_prefix].transpose(1, 2)
                    )
                    value_chunks = self.sksa_indexer.expand_chunk_values(
                        auxiliary_values
                    ).reshape_as(value_chunks)
                cos_chunks = cos[:, :complete_prefix].view(
                    hidden_states.shape[0],
                    -1,
                    self.summary_chunk_size + self.summary_token_num,
                    cos.shape[-1],
                )
                sin_chunks = sin[:, :complete_prefix].view(
                    hidden_states.shape[0],
                    -1,
                    self.summary_chunk_size + self.summary_token_num,
                    sin.shape[-1],
                )
                if sksa_aux_key_states is None:
                    summary_hidden = self.sksa_indexer.build_chunk_attention_summary(
                        hidden_chunks,
                        value_chunks,
                        summary_chunk_size=self.summary_chunk_size,
                        summary_token_num=self.summary_token_num,
                        chunk_cos=cos_chunks,
                        chunk_sin=sin_chunks,
                        indexer_block_size=self._sksa_indexer_raw_block_size,
                    )
                else:
                    expected_key_shape = (
                        hidden_states.shape[0],
                        self.config.num_key_value_heads,
                        hidden_states.shape[1],
                        self.head_dim,
                    )
                    if tuple(sksa_aux_key_states.shape) != expected_key_shape:
                        raise ValueError(
                            "independent chunk K projection has unexpected prefill shape: "
                            f"got {tuple(sksa_aux_key_states.shape)}, "
                            f"expected {expected_key_shape}"
                        )
                    flat_summary_hidden = hidden_chunks[
                        :, :, self.summary_chunk_size :, :
                    ].reshape(
                        hidden_states.shape[0],
                        -1,
                        hidden_states.shape[-1],
                    )
                    flat_summary_cos = cos_chunks[
                        :, :, self.summary_chunk_size :, :
                    ].reshape(
                        hidden_states.shape[0],
                        -1,
                        cos_chunks.shape[-1],
                    )
                    flat_summary_sin = sin_chunks[
                        :, :, self.summary_chunk_size :, :
                    ].reshape(
                        hidden_states.shape[0],
                        -1,
                        sin_chunks.shape[-1],
                    )
                    summary_query = self.sksa_indexer.project_chunk_queries(
                        flat_summary_hidden,
                        cos=flat_summary_cos,
                        sin=flat_summary_sin,
                    ).view(
                        hidden_states.shape[0],
                        -1,
                        self.summary_token_num,
                        self.config.num_attention_heads,
                        self.head_dim,
                    )
                    summary_key = sksa_aux_key_states[:, :, :complete_prefix]
                    summary_key = summary_key.transpose(1, 2).contiguous().view(
                        hidden_states.shape[0],
                        -1,
                        self.summary_chunk_size + self.summary_token_num,
                        self.config.num_key_value_heads,
                        self.head_dim,
                    )
                    summary_hidden = (
                        self.sksa_indexer.build_chunk_attention_summary_from_projected(
                            summary_query,
                            summary_key,
                            value_chunks,
                            summary_chunk_size=self.summary_chunk_size,
                            summary_token_num=self.summary_token_num,
                            indexer_block_size=self._sksa_indexer_raw_block_size,
                        )
                    )
            else:
                summary_hidden = summary_chunk_attention_indexer_key_input(
                    query_states[:, :, :complete_prefix],
                    repeat_kv(
                        key_states[:, :, :complete_prefix],
                        self.num_key_value_groups,
                    ),
                    repeat_kv(
                        value_states[:, :, :complete_prefix],
                        self.num_key_value_groups,
                    ),
                    summary_chunk_size=self.summary_chunk_size,
                    summary_token_num=self.summary_token_num,
                )
        else:
            summary_hidden = hidden_states.index_select(1, summary_pos)
        def build_index_positions():
            if self._sksa_indexer_raw_block_size < self.summary_chunk_size:
                block_size = self._sksa_indexer_raw_block_size
                return position_ids[:, :complete_prefix].reshape(
                    hidden_states.shape[0], -1, self.summary_chunk_size + self.summary_token_num
                )[:, :, block_size - 1 : self.summary_chunk_size : block_size].reshape(hidden_states.shape[0], -1)
            return position_ids.index_select(1, summary_pos)
        index_position_ids = shared_indexer_positions(
            position_ids, "prefill_keys", build_index_positions,
            self.summary_chunk_size, self.summary_token_num, self._sksa_indexer_raw_block_size,
        )
        index_keys = self.sksa_indexer.project_keys(summary_hidden, index_position_ids)
        selected = self.sksa_indexer.select_blocks(
            text_hidden,
            index_keys,
            shared_indexer_positions(position_ids, "prefill_queries",
                                     lambda: position_ids.index_select(1, text_pos)),
            raw_block_size=self._sksa_indexer_raw_block_size,
            local_window_tokens=self._sksa_local_window_tokens,
            selected_block_capacity=self._sksa_indexer_selected_blocks,
            force_sink=self._sksa_indexer_force_sink,
            sink_block_count=(
                self.summary_chunk_size // self._sksa_indexer_raw_block_size
            ),
            local_alignment_tokens=self._sksa_indexer_local_alignment,
        )
        return selected, index_keys

    @torch.no_grad()
    def _build_summary_indexer_decode_route(
        self,
        text_hidden: torch.Tensor,
        index_keys: torch.Tensor,
        text_position_ids: torch.Tensor,
        text_length: int,
    ) -> torch.Tensor:
        """Route one decode token using the cached summary/raw index keys."""
        # ``select_blocks`` derives raw query indices from its local tensor.
        # For decode, emulate the absolute index by padding only the inexpensive
        # hidden/RoPE rows and retain the final selected row.  Avoid doing this
        # at long contexts by directly evaluating the one-row score below.
        batch = text_hidden.shape[0]
        capacity = self._sksa_indexer_selected_blocks
        selected = torch.full(
            (batch, 1, capacity), -1, dtype=torch.int32,
            device=text_hidden.device,
        )
        local_alignment = getattr(
            self,
            "_sksa_indexer_local_alignment",
            self._sksa_indexer_raw_block_size,
        )
        local_start = aligned_summary_block_local_start(
            text_length - 1,
            self._sksa_local_window_tokens,
            local_alignment,
        )
        eligible_count = min(
            index_keys.shape[1],
            local_start // self._sksa_indexer_raw_block_size,
        )
        take = min(eligible_count, capacity)
        if take == 0:
            return selected
        query = self.sksa_indexer.project_queries(
            text_hidden,
            text_position_ids,
        )
        head_weight = self.sksa_indexer.head_weight_proj(text_hidden).float()
        head_weight.mul_(self.sksa_indexer.num_heads**-0.5)
        score = score_indexer_candidates(
            query,
            index_keys[:, :eligible_count],
            head_weight,
            self.sksa_indexer.head_dim**-0.5,
            scorer_backend=self.sksa_indexer.scorer_backend,
        )
        if self._sksa_indexer_force_sink:
            score[..., :min(8 // self._sksa_indexer_raw_block_size, take)] = float("inf")
        indices = torch.topk(score, take, dim=-1, sorted=False).indices
        indices = indices.to(torch.int32)
        indices = torch.sort(indices, dim=-1).values
        selected[..., :take] = indices
        return selected

    def _run_summary_indexer_decode_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        selected_blocks: torch.Tensor,
        cache: Qwen3RingBufferCache,
    ) -> torch.Tensor:
        workspace = None
        if not self.training and key_states.is_cuda and key_states.dtype == torch.bfloat16 and key_states.shape[0] == 1:
            workspace = cache._sparse_gather_storage.setdefault(self.layer_idx, {})
        compact_key, compact_value, compact_mask = gather_summary_block_decode_kv(
            key_states,
            value_states,
            selected_blocks,
            raw_block_size=self._sksa_indexer_raw_block_size,
            local_window_tokens=self._sksa_local_window_tokens,
            local_alignment_tokens=self._sksa_indexer_local_alignment,
            workspace=workspace,
        )
        output, _ = _sdpa_attention_forward(
            self,
            query_states,
            compact_key,
            compact_value,
            compact_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        return output



    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        summary_ctx: Optional[SummaryBatchContext] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        # Batch is supported only when all samples share the same summary mask
        # (time-aligned generation).

        # Compute q/k/v for the full sequence once.
        if self.config.summary_independent_parameters:
            if summary_ctx is None:
                raise ValueError("summary_ctx is required when using summary_independent_parameters.")
            summary_mask = summary_ctx.summary_mask

            query_states, key_states, value_states = self.get_query_key_value_tensors(hidden_states)
        else:
            query_states, key_states, value_states = self.get_query_key_value_tensors(hidden_states)

        cos, sin = position_embeddings
        query_len = hidden_states.shape[1]
        is_prefill = past_key_values is None or not past_key_values._reorganized[self.layer_idx]
        independent_chunk_qk = (
            self._sksa_target_enabled
            and True
            and self._sksa_indexer_chunk_attention_qk_source == "independent"
        )
        sksa_aux_query_states = None
        sksa_aux_key_states = None
        sksa_aux_value_states = None
        if independent_chunk_qk:
            if self.sksa_indexer.chunk_v_proj is not None:
                sksa_aux_value_states = self.sksa_indexer.project_chunk_values(
                    hidden_states
                ).transpose(1, 2).contiguous()
                if not is_prefill:
                    past_key_values.append_sksa_chunk_v(
                        self.layer_idx, sksa_aux_value_states[:, :, :1, :]
                    )

            sksa_aux_key_states = self.sksa_indexer.project_chunk_keys(
                hidden_states, cos=cos, sin=sin
            ).transpose(1, 2).contiguous()
            if not is_prefill and query_len > 1:
                sksa_aux_query_states = self.sksa_indexer.project_chunk_queries(
                    hidden_states[:, 1:],
                    cos=cos[:, 1:],
                    sin=sin[:, 1:],
                )
        pre_rope_key_states = key_states
        meta_key_states = None
        meta_value_states = None
        meta_summary_pos = None
        summary_scores = None
        all_summary_position_ids = None
        if self._meta_pool_enabled and (
            is_prefill or query_len != 1 or summary_ctx.position_ids is None
            or summary_ctx.position_ids.ndim < 2 or summary_ctx.summary_positions is None
            or summary_ctx.summary_positions.numel() != 0
        ):
            # Ordinary decode has no consumer of empty meta tensors/scores.
            # Prefill initialization and every summary boundary stay unchanged.
            (
                meta_key_states,
                meta_value_states,
                meta_summary_pos,
                summary_scores,
                all_summary_position_ids,
            ) = self._build_meta_kv(
                hidden_states,
                key_states,
                value_states,
                summary_ctx.summary_mask,
                cos,
                sin,
                position_ids=summary_ctx.position_ids,
                return_cache_state=True,
                summary_positions=summary_ctx.summary_positions,
            )
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        sksa_routing = None
        summary_index_keys = None
        if is_prefill and self._sksa_target_enabled and True:
            sksa_routing, summary_index_keys = self._build_summary_indexer_prefill_route(
                hidden_states,
                summary_ctx.summary_mask,
                summary_ctx.position_ids,
                query_states,
                key_states,
                value_states,
                cos,
                sin,
                sksa_aux_key_states,
                sksa_aux_value_states,
                text_positions=summary_ctx.text_positions,
                summary_positions=summary_ctx.summary_positions,
            )

        if is_prefill:
            # Prefill: feed projected K/V directly to the kernel — no full-length cache buffer.
            if (
                self._sksa_target_enabled
                and True
                and self._sksa_apply_sparse_attention
            ):
                attn_output = summary_block_sparse_prefill_attention(
                    query_states,
                    key_states,
                    value_states,
                    summary_ctx.summary_mask,
                    sksa_routing,
                    raw_block_size=self._sksa_indexer_raw_block_size,
                    summary_chunk_size=self.summary_chunk_size,
                    summary_token_num=self.summary_token_num,
                    local_window_tokens=self._sksa_local_window_tokens,
                    text_positions=summary_ctx.text_positions,
                    summary_positions=summary_ctx.summary_positions,
                ).transpose(1, 2).contiguous()
                attn_weights = None
            elif self._meta_pool_enabled:
                compressed_lse = None
                if (
                    hierarchical_summary_attn_func is not None
                    and
                    query_states.is_cuda
                    and not torch.is_grad_enabled()
                ):
                    attn_output, compressed_lse = hierarchical_summary_attn_func(
                        query_states.transpose(1, 2).contiguous(),
                        key_states.transpose(1, 2).contiguous(),
                        value_states.transpose(1, 2).contiguous(),
                        meta_key_states.transpose(1, 2).contiguous(),
                        meta_value_states.transpose(1, 2).contiguous(),
                        self.summary_chunk_size,
                        self._get_sliding_chunk_num(),
                        self._meta_pool_group_size,
                        mass_correction=self._meta_pool_mass_correction,
                        force_return_lse=False,
                    )
                else:
                    # Retain the reference implementation for unsupported dispatch.
                    compressed_result = hierarchical_summary_attention_prefill(
                        query_states,
                        key_states,
                        value_states,
                        meta_key_states,
                        meta_value_states,
                        chunk_size=self.summary_chunk_size,
                        sliding_chunk_num=self._get_sliding_chunk_num(),
                        group_size=self._meta_pool_group_size,
                        block_size=self._meta_pool_block_size,
                        mass_correction=self._meta_pool_mass_correction,
                        return_lse=False,
                    )
                    attn_output = compressed_result
                attn_weights = None
            else:
                if (
                    summary_attn_func is not None
                    and query_states.is_cuda
                ):
                    with torch.cuda.device(query_states.device):
                        attn_output, attn_weights = summary_attn_func(
                            query_states.transpose(1,2).contiguous(),
                            key_states.transpose(1,2).contiguous(),
                            value_states.transpose(1,2).contiguous(),
                            self.summary_chunk_size,
                            self.summary_token_num,
                            self._get_sliding_chunk_num(),
                            summary_pos=_get_shared_batch_mask(
                                summary_ctx.summary_mask, "summary_mask"
                            ),
                        )
                else:
                    if self.summary_token_num != 1:
                        raise RuntimeError(
                            "PyTorch KSA prefill fallback currently requires summary_token_num=1"
                        )
                    attn_output = summary_attention_prefill(
                        query_states,
                        key_states,
                        value_states,
                        summary_ctx.summary_mask,
                        chunk_size=self.summary_chunk_size,
                        sliding_chunk_num=self._get_sliding_chunk_num(),
                        block_size=self._meta_pool_block_size,
                    )
                    attn_weights = None

            # After attention is done with K/V, materialize this layer's cache directly
            # into the post-reorganize ring buffer layout (no intermediate prefill buffer).
            if past_key_values is not None:
                past_key_values.init_layer_from_prefill(
                    key_states,
                    value_states,
                    summary_ctx.summary_mask,
                    self.layer_idx,
                    pre_rope_key_states=(pre_rope_key_states if self._meta_pool_enabled else None),
                    summary_scores=summary_scores,
                    summary_cos=(cos[:, meta_summary_pos, :] if self._meta_pool_enabled else None),
                    summary_sin=(sin[:, meta_summary_pos, :] if self._meta_pool_enabled else None),
                    summary_position_ids=all_summary_position_ids,
                    meta_key_states=meta_key_states,
                    meta_value_states=meta_value_states,
                    sksa_chunk_key_states=sksa_aux_key_states,
                )
                past_key_values.set_sksa_chunk_v_from_prefill(
                    self.layer_idx, sksa_aux_value_states, summary_ctx.summary_mask
                )
                if summary_index_keys is not None:
                    past_key_values.set_sksa_index_keys(
                        self.layer_idx, summary_index_keys
                    )
        elif query_len == 1:
            # Single text token decode: write to cache, attend to full buffer
            past_key_values.update_text(
                key_states,
                value_states,
                self.layer_idx,
                sksa_chunk_key_states=(
                    sksa_aux_key_states if independent_chunk_qk else None
                ),
            )
            k_full, v_full, meta_len = past_key_values.get_attention_kv(self.layer_idx)
            if self._sksa_target_enabled and True:
                sksa_routing = self._build_summary_indexer_decode_route(
                    hidden_states,
                    past_key_values.get_sksa_index_keys(self.layer_idx),
                    shared_indexer_positions(summary_ctx.position_ids, "decode_query",
                                             lambda: summary_ctx.position_ids),
                    k_full.shape[2],
                )
            if (
                self._sksa_target_enabled
                and True
                and self._sksa_apply_sparse_attention
            ):
                attn_output = self._run_summary_indexer_decode_attention(
                    query_states, k_full, v_full, sksa_routing, past_key_values
                )
                attn_weights = None
            else:
                decode_bias = None
                if meta_len and self._meta_pool_mass_correction:
                    decode_bias = query_states.new_zeros((1, k_full.shape[2]))
                    decode_bias[:, -meta_len:] = math.log(self._meta_pool_group_size)
                attn_output, attn_weights = _sdpa_attention_forward(
                    self,
                    query_states,
                    k_full,
                    v_full,
                    decode_bias,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )
        else:
            # Chunk boundary: query = [text_token, summary_token(s)]
            # Split into text (first token) and summary (remaining tokens)
            q_text = query_states[:, :, :1, :]
            q_summary = query_states[:, :, 1:, :]
            k_text = key_states[:, :, :1, :]
            v_text = value_states[:, :, :1, :]
            k_summary = key_states[:, :, 1:, :]
            v_summary = value_states[:, :, 1:, :]

            # 1. Write text token to cache, get full KV, run text attention
            past_key_values.update_text(
                k_text,
                v_text,
                self.layer_idx,
                sksa_chunk_key_states=(
                    sksa_aux_key_states[:, :, :1, :]
                    if independent_chunk_qk
                    else None
                ),
            )
            k_full, v_full, meta_len = past_key_values.get_attention_kv(self.layer_idx)
            if self._sksa_target_enabled and True:
                sksa_routing = self._build_summary_indexer_decode_route(
                    hidden_states[:, :1],
                    past_key_values.get_sksa_index_keys(self.layer_idx),
                    shared_indexer_positions(summary_ctx.position_ids, "decode_query",
                                             lambda: summary_ctx.position_ids[:, :1]),
                    k_full.shape[2],
                )
            if (
                self._sksa_target_enabled
                and True
                and self._sksa_apply_sparse_attention
            ):
                text_out = self._run_summary_indexer_decode_attention(
                    q_text, k_full, v_full, sksa_routing, past_key_values
                )
            else:
                decode_bias = None
                if meta_len and self._meta_pool_mass_correction:
                    decode_bias = query_states.new_zeros((1, k_full.shape[2]))
                    decode_bias[:, -meta_len:] = math.log(self._meta_pool_group_size)
                text_out, _ = _sdpa_attention_forward(
                    self,
                    q_text,
                    k_full,
                    v_full,
                    decode_bias,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )

            # 2. Summary attention: attend to current chunk's C text tokens + own KV (self-attention)
            #    The original model includes the summary token's own KV in its attention
            #    (causal within summary positions). With S=1, this is just self-attention.
            k_chunk, v_chunk = past_key_values.get_current_chunk_kv(self.layer_idx)
            k_chunk_with_self = torch.cat((k_chunk, k_summary), dim=2)
            v_chunk_with_self = torch.cat((v_chunk, v_summary), dim=2)
            summary_out, _ = _sdpa_attention_forward(
                self,
                q_summary,
                k_chunk_with_self,
                v_chunk_with_self,
                None,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            # 3. Write summary KV to cache
            past_key_values.update_summary(
                k_summary,
                v_summary,
                self.layer_idx,
                pre_rope_key_states=(
                    pre_rope_key_states[:, :, 1:, :] if self._meta_pool_enabled else None
                ),
                summary_scores=summary_scores,
                summary_cos=(
                    cos[:, meta_summary_pos, :] if self._meta_pool_enabled else None
                ),
                summary_sin=(
                    sin[:, meta_summary_pos, :] if self._meta_pool_enabled else None
                ),
                summary_position_ids=all_summary_position_ids,
                position_bias=getattr(self, "meta_pool_position_bias", None),
            )
            if self._sksa_target_enabled and True:
                if (
                    self._sksa_indexer_summary_key_source == "chunk_attention"
                    and independent_chunk_qk
                ):
                    cached_aux_k = past_key_values.get_current_sksa_chunk_k(
                        self.layer_idx
                    )
                    if cached_aux_k is None:
                        raise RuntimeError(
                            "independent chunk Q/K cache is missing current raw keys"
                        )
                    summary_aux_q = sksa_aux_query_states
                    summary_aux_k = sksa_aux_key_states[:, :, 1:, :]
                    raw_aux_k = torch.cat((cached_aux_k, summary_aux_k), dim=2)
                    chunk_width = raw_aux_k.shape[2]
                    q_chunks = summary_aux_q.unsqueeze(1)
                    k_chunks = raw_aux_k.transpose(1, 2).unsqueeze(1)
                    auxiliary_values = v_chunk_with_self
                    if sksa_aux_value_states is not None:
                        cached_aux_v = past_key_values.get_current_sksa_chunk_v(self.layer_idx)
                        if cached_aux_v is None:
                            raise RuntimeError("independent chunk V cache is missing current raw values")
                        auxiliary_values = torch.cat((cached_aux_v, sksa_aux_value_states[:, :, 1:, :]), dim=2)
                    v_chunks = repeat_kv(
                        auxiliary_values, self.num_key_value_groups
                    ).transpose(1, 2).unsqueeze(1)
                    new_index_key_hidden = (
                        self.sksa_indexer.build_chunk_attention_summary_from_projected(
                            q_chunks,
                            k_chunks,
                            v_chunks,
                            summary_chunk_size=chunk_width - self.summary_token_num,
                            summary_token_num=self.summary_token_num,
                            indexer_block_size=self._sksa_indexer_raw_block_size,
                        )
                    )
                elif self._sksa_indexer_summary_key_source == "chunk_attention":
                    # ``summary_out`` is [B,S,H,D] and is exactly the same
                    # pre-output-projection chunk-local context used for
                    # prefill/training index keys.
                    new_index_key_hidden = summary_out.reshape(
                        summary_out.shape[0], summary_out.shape[1], -1
                    )
                else:
                    new_index_key_hidden = hidden_states[:, 1:]
                def build_index_positions():
                    if self._sksa_indexer_raw_block_size < self.summary_chunk_size:
                        block_size = self._sksa_indexer_raw_block_size
                        # Same exact completed-chunk subkey positions as baseline.
                        offsets = torch.arange(
                            block_size - self.summary_chunk_size, 1, block_size,
                            device=hidden_states.device,
                        )
                        return summary_ctx.position_ids[:, :1] + offsets[None]
                    return summary_ctx.position_ids[:, 1:]
                index_position_ids = shared_indexer_positions(
                    summary_ctx.position_ids, "decode_keys", build_index_positions,
                    self.summary_chunk_size, self.summary_token_num, self._sksa_indexer_raw_block_size,
                )
                new_index_key = self.sksa_indexer.project_keys(new_index_key_hidden, index_position_ids)
                past_key_values.append_sksa_index_keys(
                    self.layer_idx, new_index_key
                )
            if independent_chunk_qk:
                past_key_values.clear_sksa_chunk_k(self.layer_idx)
                past_key_values.clear_sksa_chunk_v(self.layer_idx)

            attn_output = torch.cat([text_out, summary_out], dim=2)
            attn_weights = None


        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
