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


from typing import Any, Callable, Optional, Union
import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from .summary_context import SummaryBatchContext, build_summary_sliding_context
from .runtime_config import _get_shared_batch_mask
from .layer_norm_rope import Qwen3RMSNorm
from .layer_attention import Qwen3Attention, Qwen3SummaryAttention


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Use SummaryAttention if enabled in config
        if getattr(config, "use_summary_attention", False) is True and config.summary_layer_freq[layer_idx] == 1:
            self.self_attn = Qwen3SummaryAttention(config=config, layer_idx=layer_idx)
        elif getattr(config, "use_summary_attention", False) is False and config.summary_layer_freq[layer_idx] == 0:
            self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f'Check config.summary_layer_freq {config.summary_layer_freq} and config.use_summary_attention {config.use_summary_attention}')

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if getattr(config, "summary_independent_attention_layernorm", False):
            self.input_layernorm_summary = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        summary_ctx: Optional[SummaryBatchContext] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        if getattr(self.config, "summary_independent_attention_layernorm", False):
            summary_mask = summary_ctx.summary_mask
            summary_pos = _get_shared_batch_mask(summary_mask, "summary_mask").to(residual.device)
            hidden_states = self.input_layernorm(hidden_states)
            if summary_pos.any():
                hidden_summary = residual[:, summary_pos, :]
                hidden_summary = self.input_layernorm_summary(hidden_summary)
                hidden_states[:, summary_pos, :] = hidden_summary
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention - pass summary_ctx if using summary attention
        attn_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            **kwargs,
        }
        if isinstance(self.self_attn, Qwen3SummaryAttention):
            attn_kwargs["summary_ctx"] = summary_ctx

        hidden_states, _ = self.self_attn(**attn_kwargs)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
