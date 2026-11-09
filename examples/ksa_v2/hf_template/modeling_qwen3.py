# Explicit dependency closure for Transformers dynamic-module copying.
from .cache_ring_buffer import __name__ as _transport_cache_ring_buffer
from .elementwise_fusion import __name__ as _transport_elementwise_fusion
from .hierarchical_summary_attention import __name__ as _transport_hierarchical_summary_attention
from .indexer_step import __name__ as _transport_indexer_step
from .layer_attention import __name__ as _transport_layer_attention
from .layer_decoder import __name__ as _transport_layer_decoder
from .layer_norm_rope import __name__ as _transport_layer_norm_rope
from .prefill_attention import __name__ as _transport_prefill_attention
from .runtime_config import __name__ as _transport_runtime_config
from .sksa_attention import __name__ as _transport_sksa_attention
from .sparse_decode_gather import __name__ as _transport_sparse_decode_gather
from .summary_context import __name__ as _transport_summary_context

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


from copy import copy as _copy_summary_context
from typing import Any, Callable, Optional, Union
import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
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
from .runtime_config import _get_shared_batch_mask, _parse_config_pattern, _validate_runtime_config
from .layer_norm_rope import Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from .cache_ring_buffer import Qwen3RingBufferCache
from .layer_attention import Qwen3Attention, Qwen3SummaryAttention, _sdpa_attention_forward, repeat_kv
from .layer_decoder import Qwen3DecoderLayer, Qwen3MLP


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        """The summary model initializes and owns its ring/meta cache."""
        return False


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        partial_chunk_policy = getattr(
            config, "summary_partial_chunk_policy", "defer"
        )
        if partial_chunk_policy != "defer":
            raise ValueError(
                "C-KSA runtime requires summary_partial_chunk_policy='defer' "
                "so prefill and cache decode share one chunk-boundary policy"
            )
        if not getattr(config, "summary_layer_freq", False):
            if config.use_summary_attention:
                config.summary_layer_freq = [1]*config.num_hidden_layers
            else:
                config.summary_layer_freq = [0]*config.num_hidden_layers
            Warning(f'Please set config.summary_layer_freq, temp set summary_layer_freq = {config.num_hidden_layers}')
        else:
            config.summary_layer_freq = _parse_config_pattern(config.summary_layer_freq)

        _validate_runtime_config(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Cache per-layer sliding_chunk_nums for KV cache eviction
        _sv = _parse_config_pattern(getattr(config, "summary_sliding_chunk_num", 0) or 0)
        if isinstance(_sv, list):
            self._sliding_chunk_nums = [int(v) for v in _sv]
        else:
            self._sliding_chunk_nums = [int(_sv)] * config.num_hidden_layers

        # Initialize weights and apply final processing
        self.post_init()

    def _expand_input_with_summary_tokens(self, input_ids):
        """Expand input_ids with summary tokens for prefill phase (vectorized).

        Returns:
            Tuple of (expanded_input_ids, position_ids, text_only_mask)
        """
        summary_chunk = self.config.summary_chunk_size
        summary_num = self.config.summary_token_num
        summary_begin = self.config.summary_token_begin

        if summary_chunk == 0 or summary_num == 0:
            return input_ids, None, None

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype
        block = summary_chunk + summary_num

        # Number of full chunks and remainder
        n_full_chunks = seq_len // summary_chunk
        remainder = seq_len % summary_chunk
        has_remainder = remainder > 0

        # Total expanded length: full_chunks * block + remainder
        expanded_len = n_full_chunks * block + (remainder if has_remainder else 0)

        # --- Build expanded_input_ids ---
        expanded_ids = torch.empty((batch_size, expanded_len), dtype=dtype, device=device)
        text_only_mask = torch.zeros((batch_size, expanded_len), dtype=torch.bool, device=device)

        # Compute text positions: for chunk i, text goes to [i*block, i*block+summary_chunk)
        # Summary positions: [i*block+summary_chunk, (i+1)*block)
        if n_full_chunks > 0:
            chunk_indices = torch.arange(n_full_chunks, device=device)
            # Text source positions in original input_ids
            text_src_offsets = (chunk_indices * summary_chunk).unsqueeze(1) + torch.arange(summary_chunk, device=device).unsqueeze(0)  # [n_full_chunks, summary_chunk]
            # Text dest positions in expanded
            text_dst_offsets = (chunk_indices * block).unsqueeze(1) + torch.arange(summary_chunk, device=device).unsqueeze(0)  # [n_full_chunks, summary_chunk]
            # Summary dest positions
            summary_dst_offsets = (chunk_indices * block + summary_chunk).unsqueeze(1) + torch.arange(summary_num, device=device).unsqueeze(0)  # [n_full_chunks, summary_num]

            text_src_flat = text_src_offsets.reshape(-1)
            text_dst_flat = text_dst_offsets.reshape(-1)
            summary_dst_flat = summary_dst_offsets.reshape(-1)

            # Copy text tokens
            expanded_ids[:, text_dst_flat] = input_ids[:, text_src_flat]
            text_only_mask[:, text_dst_flat] = True

            # Fill summary tokens
            summary_ids_val = torch.arange(summary_num, device=device, dtype=dtype) + summary_begin
            expanded_ids[:, summary_dst_flat] = summary_ids_val.repeat(n_full_chunks).unsqueeze(0).expand(batch_size, -1)

        # Handle remainder (last partial chunk, no summary tokens)
        if has_remainder:
            rem_start_src = n_full_chunks * summary_chunk
            rem_start_dst = n_full_chunks * block
            rem_offsets = torch.arange(remainder, device=device)
            expanded_ids[:, rem_start_dst + rem_offsets] = input_ids[:, rem_start_src + rem_offsets]
            text_only_mask[:, rem_start_dst + rem_offsets] = True

        # --- Build position_ids ---
        position_ids = torch.empty((batch_size, expanded_len), dtype=torch.long, device=device)

        if n_full_chunks > 0:
            # Text position IDs
            if self.config.summary_chunk_position_ids_type == 'origin':
                text_pos = text_src_flat.unsqueeze(0).expand(batch_size, -1)
            elif self.config.summary_chunk_position_ids_type == 'inner_chunk':
                inner_pos = torch.arange(summary_chunk, device=device).repeat(n_full_chunks)
                text_pos = inner_pos.unsqueeze(0).expand(batch_size, -1)
            else:
                raise ValueError(f'Check config.summary_chunk_position_ids_type: {self.config.summary_chunk_position_ids_type}')
            position_ids[:, text_dst_flat] = text_pos

            # Summary position IDs
            if self.config.summary_token_position_ids_type == 'zeros':
                position_ids[:, summary_dst_flat] = 0
            elif self.config.summary_token_position_ids_type in ('last_chunk_slice_left', 'last_chunk_slice_right'):
                # Vectorized slice_ends computation for all chunks at once
                if self.config.summary_token_position_ids_type == 'last_chunk_slice_left':
                    idx = torch.arange(0, summary_num, device=device, dtype=torch.long)
                else:
                    idx = torch.arange(1, summary_num + 1, device=device, dtype=torch.long)
                # For each chunk i: prev_text_end = i * summary_chunk
                prev_ends = (chunk_indices * summary_chunk).unsqueeze(1)  # [n_full_chunks, 1]
                slice_ends = prev_ends + (idx.unsqueeze(0) * summary_chunk) // summary_num - 1  # [n_full_chunks, summary_num]
                slice_ends = slice_ends.clamp(min=0)
                # Clamp per-chunk: min is prev_text_end for that chunk
                slice_ends = torch.max(slice_ends, prev_ends)
                position_ids[:, summary_dst_flat] = slice_ends.reshape(-1).unsqueeze(0).expand(batch_size, -1)
            else:
                raise ValueError(f'Unknown summary_token_position_ids_type: {self.config.summary_token_position_ids_type}')

        # Remainder position IDs
        if has_remainder:
            if self.config.summary_chunk_position_ids_type == 'origin':
                rem_pos = (rem_start_src + rem_offsets).unsqueeze(0).expand(batch_size, -1)
            elif self.config.summary_chunk_position_ids_type == 'inner_chunk':
                rem_pos = rem_offsets.unsqueeze(0).expand(batch_size, -1)
            else:
                raise ValueError(f'Check config.summary_chunk_position_ids_type: {self.config.summary_chunk_position_ids_type}')
            position_ids[:, rem_start_dst + rem_offsets] = rem_pos

        return expanded_ids, position_ids, text_only_mask

    def _build_summary_context(self, input_ids, position_ids, is_prefill, use_cache):
        """Build summary context for attention layers."""
        summary_chunk = self.config.summary_chunk_size
        summary_num = self.config.summary_token_num
        summary_begin = self.config.summary_token_begin

        if summary_chunk > 0 and summary_num > 0:
            return build_summary_sliding_context(
                input_ids=input_ids,
                position_ids=position_ids,
                summary_token_num=summary_num,
                summary_token_begin=summary_begin,
            )
        return None

    def _filter_summary_tokens(self, hidden_states, text_only_mask, use_summary, is_decode):
        """Filter out summary tokens from output hidden states."""
        if text_only_mask is not None:
            if text_only_mask.ndim == 2 and text_only_mask.shape[0] > 1:
                shared = text_only_mask[0]
                if not torch.equal(text_only_mask, shared.unsqueeze(0).expand_as(text_only_mask)):
                    return hidden_states[text_only_mask.to(hidden_states.device)].unsqueeze(0)
            # Prefill synchronized batch: all rows share the same expanded layout.
            text_pos = _get_shared_batch_mask(text_only_mask, "text_only_mask").to(hidden_states.device)
            return hidden_states[:, text_pos, :]
        elif use_summary and is_decode and hidden_states.size(1) > 1:
            # Decode: if we have multiple tokens, only return the first (text token)
            return hidden_states[:, :1, :]
        return hidden_states

    @check_model_inputs()
    @auto_docstring
    @shared_indexer_step
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        summary_ctx: Optional[SummaryBatchContext] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        summary_ctx (`SummaryBatchContext`, *optional*):
            Precomputed summary-attention metadata. When omitted, the model builds it from the expanded input during
            summary-attention prefill.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Inference-only export: call eval() under torch.no_grad() or torch.inference_mode()")
        if inputs_embeds is not None or input_ids is None:
            raise ValueError("This inference runtime requires input_ids, not inputs_embeds")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("This inference runtime requires one nonempty request")
        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.ndim != 2 or not bool(torch.all(attention_mask == 1)):
                raise ValueError("Padded inputs and custom attention masks are not supported")
        use_summary = getattr(self.config, "use_summary_attention", False)
        is_prefill = past_key_values is None or past_key_values.get_seq_length() == 0

        # Prefill phase with summary attention: expand input_ids with summary tokens
        text_only_mask = None
        if use_summary and input_ids is not None and inputs_embeds is None and is_prefill:
            input_ids, position_ids, text_only_mask = self._expand_input_with_summary_tokens(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize cache
        if use_cache and past_key_values is None:
            if use_summary:
                past_key_values = Qwen3RingBufferCache(
                    config=self.config, sliding_chunk_nums=self._sliding_chunk_nums)
            else:
                past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Caller-owned decode contexts may be updated and reused across steps.
        # Preserve their lookup fields exactly; step-derived lookups belong to
        # a local shallow context, while the original mask normalization stays.
        provided_summary_ctx = summary_ctx
        # Build summary context if needed
        if use_summary and summary_ctx is None and input_ids is not None:
            summary_ctx = self._build_summary_context(input_ids, position_ids, is_prefill, use_cache)
        if use_summary and summary_ctx is not None:
            summary_ctx.summary_mask = _get_shared_batch_mask(summary_ctx.summary_mask, "summary_mask")
            if not is_prefill and summary_ctx is provided_summary_ctx:
                summary_ctx = _copy_summary_context(summary_ctx)
            if not torch.is_grad_enabled():
                # Prefill historically rebuilds lookup positions. Decode may
                # receive explicit custom positions: preserve those inputs and
                # their original validation/consumers rather than repair them.
                if is_prefill or summary_ctx.text_positions is None:
                    summary_ctx.text_positions = torch.nonzero(~summary_ctx.summary_mask, as_tuple=False).flatten()
                if is_prefill or summary_ctx.summary_positions is None:
                    summary_ctx.summary_positions = torch.nonzero(summary_ctx.summary_mask, as_tuple=False).flatten()
                if is_prefill and isinstance(past_key_values, Qwen3RingBufferCache):
                    past_key_values.prepare_prefill(summary_ctx.text_positions, summary_ctx.summary_positions)

        causal_mask_mapping = attention_mask
        if not isinstance(causal_mask_mapping, (dict, list)):
            if summary_ctx and summary_ctx.enabled:
                seq_len = inputs_embeds.shape[1]
                # During prefill, Qwen3SummaryAttention uses summary_attn_func
                # which does not need a dense mask. Skip expensive mask construction.
                # During decode, prepare_inputs_for_generation already computed
                # per-layer keep_indices and passed them as attention_mask (list).
                # If we reach here with a non-list, it means no mask is needed.
                causal_mask_mapping = None
            else:
                # Prepare mask arguments
                mask_kwargs = {
                    "config": self.config,
                    "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }
                # Create the masks - disable causal mask when summary context is enabled
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                # The sliding window alternating layers are not always activated depending on the config
                if self.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if causal_mask_mapping is None:
                layer_mask = None
            elif isinstance(causal_mask_mapping, list):
                layer_mask = causal_mask_mapping[layer_idx]
            else:
                layer_mask = causal_mask_mapping[decoder_layer.attention_type]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                summary_ctx=summary_ctx,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # After prefill: reorganize cache to ring buffer layout
        if use_cache and use_summary and past_key_values is not None and is_prefill:
            if hasattr(past_key_values, 'reorganize_after_prefill') and summary_ctx is not None:
                past_key_values.reorganize_after_prefill(summary_ctx.summary_mask)

        # After chunk boundary decode: reset chunk counters
        if use_cache and use_summary and past_key_values is not None and not is_prefill:
            if hasattr(past_key_values, 'reset_chunk_counter'):
                past_key_values.reset_chunk_counter()

        # Filter out summary tokens from output
        hidden_states = self._filter_summary_tokens(hidden_states, text_only_mask, use_summary,
                                                      past_key_values is not None and past_key_values.get_seq_length() > 0)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        summary_ctx: Optional[SummaryBatchContext] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        summary_ctx (`SummaryBatchContext`, *optional*):
            Precomputed summary positions and masks for this inference forward.
        labels (`torch.LongTensor`, *optional*):
            Unsupported by this inference export; passing labels raises an error.
        """
        if self.training or torch.is_grad_enabled() or labels is not None:
            raise RuntimeError("Inference-only export: call eval() under torch.no_grad() or torch.inference_mode(), without labels")
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            summary_ctx=summary_ctx,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Keep BF16 logits and the established last-token default.
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            # Inference: only need last token's logits to avoid OOM from [seq_len, vocab_size]
            logits = self.lm_head(hidden_states[:, -1:, :])
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        truncate_n = getattr(self.config, "truncate_predict_nums", None)
        if truncate_n is None:
            truncate_n = self.config.vocab_size - getattr(
                self.config, "summary_token_num", 0
            )
        if truncate_n > 0:
            logits = logits[..., :truncate_n]

        loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _build_summary_attention_mask_for_generation(
        self,
        *,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Ring buffer cache handles attention internally — no mask needed for decode."""
        if isinstance(past_key_values, Qwen3RingBufferCache):
            return None
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            raise ValueError("This inference runtime requires input_ids, not inputs_embeds")
        use_summary = getattr(self.config, "use_summary_attention", False)

        # If not using summary attention, use standard behavior
        if not use_summary:
            return super().prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                **kwargs,
            )

        # For summary attention: handle cache-based input slicing
        summary_chunk_size = getattr(self.config, "summary_chunk_size", 0)
        summary_token_num = getattr(self.config, "summary_token_num", 0)
        summary_token_begin = getattr(self.config, "summary_token_begin", 0)

        # Prefill phase: pass full sequence, forward() will handle summary token insertion
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            if cache_position is None:
                cache_position = torch.arange(0, input_ids.shape[1], device=input_ids.device)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "cache_position": cache_position,
                "use_cache": kwargs.get("use_cache"),
            }

        # Decode phase: only pass new tokens not in cache
        # Get current chunk size (number of text tokens in current chunk)
        cur_chunk = past_key_values.get_cur_chunk_size() if hasattr(past_key_values, "get_cur_chunk_size") else 0
        true_token_num = past_key_values.get_true_token_num()

        # Only take the new tokens that haven't been processed
        if input_ids.shape[1] > 1:
            # Slice to get only new tokens
            new_token_count = input_ids.shape[1] - true_token_num
            assert new_token_count > 0, f'new_token_count={new_token_count} should be greater than 0'
            input_ids = input_ids[:, -new_token_count:]
        if input_ids.shape[1] != 1:
            raise ValueError("Cached generation supports one new text token at a time")
        device = input_ids.device
        # Check if we need to insert summary tokens
        # If cur_chunk >= summary_chunk_size, we need to generate summary tokens
        if cur_chunk == summary_chunk_size - 1:
            # Insert summary tokens
            batch_size = input_ids.shape[0]
            summary_ids = (
                torch.arange(summary_token_num, device=device, dtype=input_ids.dtype)
                + summary_token_begin
            ).unsqueeze(0).repeat(batch_size, 1)

            # Concatenate: [text_token, summary_tokens]
            input_ids = torch.cat([input_ids, summary_ids], dim=1)

            # Position IDs: text token uses cur_chunk, summary tokens use 0
            if self.config.summary_chunk_position_ids_type == 'origin':
                text_pos = torch.full((batch_size, 1), past_key_values.get_true_token_num(), device=device, dtype=torch.long)
            elif self.config.summary_chunk_position_ids_type == 'inner_chunk':
                text_pos = torch.full((batch_size, 1), cur_chunk, device=device, dtype=torch.long)
            else:
                raise ValueError(f'Check config.summary_chunk_position_ids_type: {self.config.summary_chunk_position_ids_type}')

            if self.config.summary_token_position_ids_type == 'zeros':
                summary_pos = torch.zeros((batch_size, summary_token_num), device=device, dtype=torch.long)
            elif self.config.summary_token_position_ids_type == 'last_chunk_slice_left':
                # 等分成 summary_num 份，每个 summary token 取对应 slice 的末尾
                prev_text_end = past_key_values.get_true_token_num()+1-summary_chunk_size
                cur_text_end = past_key_values.get_true_token_num()+1
                chunk_len = cur_text_end - prev_text_end

                idx = torch.arange(0, summary_token_num, device=device, dtype=torch.long,)

                # 每一份的末尾（全局 position）
                slice_ends = prev_text_end + (idx * chunk_len) // summary_token_num - 1
                slice_ends = slice_ends.clamp(min=prev_text_end)

                summary_pos = slice_ends.to(dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            elif self.config.summary_token_position_ids_type == 'last_chunk_slice_right':
                # 等分成 summary_num 份，每个 summary token 取对应 slice 的末尾
                prev_text_end = past_key_values.get_true_token_num()+1-summary_chunk_size
                cur_text_end = past_key_values.get_true_token_num()+1
                chunk_len = cur_text_end - prev_text_end

                idx = torch.arange(1, summary_token_num + 1, device=device, dtype=torch.long,)

                # 每一份的末尾（全局 position）
                slice_ends = prev_text_end + (idx * chunk_len) // summary_token_num - 1
                slice_ends = slice_ends.clamp(min=prev_text_end)

                summary_pos = slice_ends.to(dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

            else:
                raise ValueError('')

            position_ids = torch.cat([text_pos, summary_pos], dim=1)
        else:
            # Normal decode: just the new text token with position = cur_chunk
            if position_ids is None:
                batch_size = input_ids.shape[0]
                if self.config.summary_chunk_position_ids_type == 'origin':
                    position_ids = torch.full((batch_size, input_ids.shape[1]), past_key_values.get_true_token_num(), device=input_ids.device, dtype=torch.long)
                elif self.config.summary_chunk_position_ids_type == 'inner_chunk':
                    position_ids = torch.full((batch_size, input_ids.shape[1]), cur_chunk, device=input_ids.device, dtype=torch.long)
                else:
                    raise ValueError(f'Check config.summary_chunk_position_ids_type: {self.config.summary_chunk_position_ids_type}')
        return {
            "input_ids": input_ids,
            "attention_mask": self._build_summary_attention_mask_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            ),
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "use_cache": kwargs.get("use_cache"),
        }


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3RingBufferCache",
    "Qwen3SummaryAttention",
    "SummaryBatchContext",
    "build_summary_sliding_context",
]
