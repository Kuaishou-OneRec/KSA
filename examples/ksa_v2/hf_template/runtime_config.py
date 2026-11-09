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


import ast
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


def _parse_config_pattern(val):
    """Read integer/list layer patterns without executing configuration text."""
    if not isinstance(val, str):
        return val

    def parse(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.List):
            return [parse(item) for item in node.elts]
        if isinstance(node, ast.BinOp):
            left, right = parse(node.left), parse(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Mult):
                return left * right
        raise ValueError("Layer patterns support only integers, lists, + and *")

    return parse(ast.parse(val, mode="eval").body)


def _get_shared_batch_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    """Return a 1D mask shared by all batch rows.

    Summary attention currently supports batched prefill only when every sample
    has the same expanded layout. Packed/ragged batches need document-local
    boundaries and are intentionally rejected here.
    """
    if mask.ndim == 1:
        return mask
    if mask.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {tuple(mask.shape)}.")

    shared = mask[0]
    if mask.shape[0] > 1 and not torch.equal(mask, shared.unsqueeze(0).expand_as(mask)):
        raise ValueError(
            f"{name} must be identical across the batch for synchronized batched prefill; "
            "packed or ragged summary layouts are not supported yet."
        )
    return shared


def _validate_runtime_config(config: Qwen3Config) -> None:
    """KSA / C-KSA + Full and C-KSA + S-KSA share the same inference runtime."""
    if getattr(config, "mix_coeff", None) != 0:
        raise ValueError("This inference runtime requires shared Summary QKV (mix_coeff=0)")
    if getattr(config, "summary_token_num", 0) != 1:
        raise ValueError("The runtime requires one summary token per chunk")
    if int(config.summary_chunk_size) != 8:
        raise ValueError("The runtime requires 8-token chunks")
    if getattr(config, "summary_chunk_position_ids_type", None) != "origin":
        raise ValueError("The runtime requires original text position IDs")

    groups = _parse_config_pattern(config.summary_meta_pool_group_size)
    sliding = _parse_config_pattern(config.summary_sliding_chunk_num)
    layers = _parse_config_pattern(config.summary_layer_freq)
    if not isinstance(groups, list):
        groups = [int(groups)] * config.num_hidden_layers
    if not isinstance(sliding, list):
        sliding = [int(sliding)] * config.num_hidden_layers
    if not isinstance(layers, list):
        layers = [int(layers)] * config.num_hidden_layers
    if not all(len(values) == config.num_hidden_layers for values in (groups, sliding, layers)):
        raise ValueError("Runtime layer patterns must match num_hidden_layers")

    # Compression warmup / Full CPT have no Indexer parameters or sparse routes.
    if not getattr(config, "sksa_enabled", False):
        return
    if getattr(config, "sksa_routing_mode", None) != "summary_indexer":
        raise ValueError("S-KSA requires summary_indexer routing")
    targets = [int(value) for value in config.sksa_sparse_full_layers]
    if not targets:
        raise ValueError("S-KSA requires at least one sparse Full layer")

    token_budget = getattr(config, "sksa_indexer_token_budget", 0)
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget <= 0:
        raise ValueError("indexer token budget must be a positive integer")
    for name in (
        "sksa_indexer_query_rank",
        "sksa_indexer_heads",
        "sksa_indexer_head_dim",
        "sksa_indexer_query_block_size",
    ):
        value = getattr(config, name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    rope_dim = getattr(config, "sksa_indexer_rope_dim", 0)
    if (
        isinstance(rope_dim, bool)
        or not isinstance(rope_dim, int)
        or rope_dim < 0
        or rope_dim > int(config.sksa_indexer_head_dim)
        or rope_dim % 2
    ):
        raise ValueError("sksa_indexer_rope_dim must be even and within head_dim")
    scorer_backend = getattr(
        config, "sksa_indexer_scorer_backend", "torch"
    )
    if scorer_backend not in ("torch", "fused", "auto"):
        raise ValueError(
            "sksa_indexer_scorer_backend must be 'torch', 'fused', or 'auto'"
        )
    key_source = getattr(
        config, "sksa_indexer_summary_key_source", "input_summary"
    )
    if key_source not in ("input_summary", "chunk_attention"):
        raise ValueError(
            "sksa_indexer_summary_key_source must be 'input_summary' or "
            "'chunk_attention'"
        )
    chunk_qk_source = getattr(
        config, "sksa_indexer_chunk_attention_qk_source", "backbone"
    )
    if chunk_qk_source not in ("backbone", "independent"):
        raise ValueError(
            "sksa_indexer_chunk_attention_qk_source must be "
            "'backbone' or 'independent'"
        )
    chunk_v_source = getattr(config, "sksa_indexer_chunk_attention_v_source", "backbone")
    if chunk_v_source not in ("backbone", "independent"):
        raise ValueError("invalid sksa_indexer_chunk_attention_v_source")
    if chunk_v_source == "independent" and chunk_qk_source != "independent":
        raise ValueError("independent chunk V requires independent chunk Q/K")
    if chunk_qk_source == "independent" and key_source != "chunk_attention":
        raise ValueError(
            "independent chunk Q/K requires "
            "sksa_indexer_summary_key_source='chunk_attention'"
        )
    raw_block_size = (getattr(config, "sksa_indexer_block_size", 8))
    if (
        isinstance(raw_block_size, bool)
        or not isinstance(raw_block_size, int)
        or raw_block_size not in ((2, 4, 8))
    ):
        raise ValueError("sksa_indexer_block_size must be 2, 4, or 8")
    if raw_block_size < 8 and (
        chunk_qk_source != "independent"
        or int(getattr(config, "sksa_local_window_tokens", 0)) < 8
    ):
        raise ValueError("subchunk indexer requires independent Q/K and local window >= 8")
    if token_budget % raw_block_size:
        raise ValueError(
            "summary indexer token budget must be divisible by the raw block size"
        )
    if int(getattr(config, "sksa_local_window_tokens", 0)) <= 0:
        raise ValueError("summary indexer requires a positive local window")
    if getattr(config, "sksa_indexer_force_sink", True) and token_budget < 8:
        raise ValueError("indexer token budget must cover 8 sink tokens")
    for target in targets:
        if not 0 <= target < config.num_hidden_layers:
            raise ValueError(f"S-KSA target layer {target} is out of range")
        if not layers[target] or int(groups[target]) > 1:
            raise ValueError(f"S-KSA target layer {target} is not a full KSA layer")
        if int(sliding[target]) * config.summary_chunk_size < config.max_position_embeddings:
            raise ValueError(f"S-KSA target layer {target} is not globally dense")
