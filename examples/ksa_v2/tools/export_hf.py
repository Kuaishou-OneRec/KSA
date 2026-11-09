#!/usr/bin/env python3
"""Export a KSA/C-KSA Muse DCP checkpoint as a self-contained HF model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
KERNEL_ROOT = REPO_ROOT / "summary_attention_kernel/ksa_v2"
TOKENIZER_DIR = REPO_ROOT / "assets/tokenizer"

ASSET_FILES = (
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def runtime_files_for(runtime_source: Path) -> tuple[str, ...]:
    """Include all runtime modules, including transitive relative imports."""
    files = tuple(sorted(path.name for path in runtime_source.glob("*.py")))
    if "modeling_qwen3.py" not in files:
        raise FileNotFoundError(f"missing modeling_qwen3.py in {runtime_source}")
    for name in files:
        tree = ast.parse((runtime_source / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or not node.module or node.module + ".py" not in files:
                    raise ValueError(f"unbundled relative import in {name}: {node.module}")
    return files


def validate_runtime_config(training: dict) -> None:
    """Check the shared-QKV chunk8 profile supported by the infra runtime."""
    required = {
        "use_summary_attention": True,
        "summary_chunk_size": 8,
        "summary_token_num": 1,
        "summary_partial_chunk_policy": "defer",
        "summary_chunk_position_ids_type": "origin",
        "summary_token_position_ids_type": "last_chunk_slice_right",
        "summary_independent_parameters": True,
        "summary_independent_qkv": False,
        "q_norm": True,
        "k_norm": True,
    }
    if training["summary_meta_pool_mode"] == "dimension_wise":
        required.update(
            summary_meta_pool_position="group_last",
            summary_meta_pool_mass_correction="log_group_size",
        )
    if training.get("sksa_enabled", False):
        required.update(
            sksa_routing_mode="summary_indexer",
            sksa_indexer_block_size=4,
            sksa_indexer_summary_key_source="chunk_attention",
            sksa_indexer_chunk_attention_qk_source="independent",
            sksa_indexer_chunk_attention_v_source="independent",
            sksa_indexer_chunk_attention_include_summary=False,
            sksa_indexer_summary_sink_tokens=8,
            sksa_indexer_force_sink=True,
            sksa_apply_sparse_attention=True,
        )
    mismatch = [name for name, value in required.items() if training.get(name) != value]
    if training["summary_meta_pool_mode"] not in {"none", "dimension_wise"}:
        mismatch.append("summary_meta_pool_mode")
    for name in (
        "q_proj_bias", "k_proj_bias", "v_proj_bias", "sksa_indexer_qk_norm",
        "summary_independent_qk_norm", "summary_independent_attention_layernorm",
    ):
        if training.get(name, False):
            mismatch.append(name)
    if training["num_heads"] != training["num_kv_heads"]:
        mismatch.append("num_kv_heads")
    if training["embed_dim"] != training["num_kv_heads"] * training["head_dim"]:
        mismatch.append("embed_dim")
    layers = training["num_layers"]
    groups = parse_pattern(training["summary_meta_pool_group_size"], layers)
    windows = parse_pattern(training["summary_sliding_chunk_num"], layers)
    full_layers = {i for i in range(layers) if windows[i] * 8 >= training["max_seq_len"]}
    if any(groups[i] != 0 for i in full_layers):
        mismatch.append("Full layer patterns")
    if training["summary_meta_pool_mode"] == "dimension_wise":
        if any(groups[i] <= 1 for i in range(layers) if i not in full_layers):
            mismatch.append("C-KSA layer patterns")
    elif any(groups):
        mismatch.append("summary_meta_pool_group_size")
    if training.get("sksa_enabled", False):
        targets = training.get("sksa_sparse_full_layers", [])
        if not targets or set(targets) != full_layers:
            mismatch.append("sksa_sparse_full_layers")
    if parse_pattern(training["summary_layer_freq"], layers) != [1] * layers:
        mismatch.append("summary_layer_freq")
    if mismatch:
        raise ValueError("HF runtime is incompatible with: " + ", ".join(mismatch))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_source_files(directory: Path) -> list[Path]:
    ignored_parts = {"build", ".build", ".deps", "__pycache__", ".pytest_cache", "inference"}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not ignored_parts.intersection(path.relative_to(directory).parts)
        and not any(part.endswith(".egg-info") for part in path.relative_to(directory).parts)
        and path.suffix != ".pyc"
    )


def _pattern_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.List):
        return [_pattern_node(item) for item in node.elts]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _pattern_node(node.left) + _pattern_node(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _pattern_node(node.left)
        right = _pattern_node(node.right)
        if isinstance(left, list) and isinstance(right, int):
            return left * right
        if isinstance(left, int) and isinstance(right, list):
            return right * left
    raise ValueError("unsupported model-config pattern expression")


def parse_pattern(value, num_layers: int) -> list[int]:
    if isinstance(value, str):
        value = _pattern_node(ast.parse(value, mode="eval").body)
    if isinstance(value, int):
        value = [value] * num_layers
    if not isinstance(value, list) or len(value) != num_layers:
        raise ValueError(f"pattern must contain exactly {num_layers} layer values")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("pattern values must be integers")
    return value


def canonical_checkpoint_keys(values: dict) -> dict:
    """Accept legacy names only at the export boundary; reject mixed aliases."""
    canonical = {}
    for name, value in values.items():
        target = name.replace("scsa_", "sksa_")
        if target in canonical:
            raise ValueError(f"conflicting checkpoint aliases for {target}")
        canonical[target] = value
    return canonical


def load_training_config(model_config_path: Path) -> dict:
    with model_config_path.open(encoding="utf-8") as handle:
        config = canonical_checkpoint_keys(json.load(handle))
    required = {
        "vocab_size",
        "embed_dim",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "intermediate_dim",
        "max_seq_len",
        "rope_base",
        "norm_eps",
        "tie_word_embeddings",
        "summary_chunk_size",
        "summary_token_num",
        "summary_partial_chunk_policy",
        "summary_token_begin",
        "summary_layer_freq",
        "summary_sliding_chunk_num",
        "summary_chunk_position_ids_type",
        "summary_token_position_ids_type",
        "summary_independent_parameters",
        "summary_independent_qkv",
        "summary_meta_pool_mode",
        "summary_meta_pool_group_size",
        "summary_meta_pool_position",
        "summary_meta_pool_mass_correction",
        "summary_block_size",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"training model config is missing fields: {missing}")
    if config["summary_meta_pool_mode"] not in {
        "none", "scalar", "dimension_wise"
    }:
        raise ValueError(
            "export supports KSA, scalar C-KSA, or dimension-wise C-KSA"
        )
    if config["summary_token_num"] <= 0:
        raise ValueError("summary_token_num must be positive")
    if config["summary_token_begin"] != config["vocab_size"]:
        raise ValueError(
            "export requires summary_token_begin == vocab_size because summary "
            "embeddings are appended to the HF token embedding"
        )
    if config.get("sksa_enabled", False):
        routing_mode = config.get("sksa_routing_mode", "summary_indexer")
        if routing_mode != "summary_indexer":
            raise ValueError("export supports summary_indexer (S-KSA) routing only")
        sksa_required = {"sksa_sparse_full_layers", "sksa_local_window_tokens"}
        key_source = config.setdefault(
            "sksa_indexer_summary_key_source", "input_summary"
        )
        if key_source not in ("input_summary", "chunk_attention"):
            raise ValueError(
                "sksa_indexer_summary_key_source must be one of: "
                "input_summary, chunk_attention"
            )
        chunk_qk_source = config.setdefault(
            "sksa_indexer_chunk_attention_qk_source", "backbone"
        )
        if chunk_qk_source not in ("backbone", "independent"):
            raise ValueError(
                "sksa_indexer_chunk_attention_qk_source must be one of: "
                "backbone, independent"
            )
        if chunk_qk_source == "independent" and key_source != "chunk_attention":
            raise ValueError(
                "independent chunk Q/K requires "
                "sksa_indexer_summary_key_source=chunk_attention"
            )
        chunk_v_source = config.setdefault("sksa_indexer_chunk_attention_v_source", "backbone")
        if chunk_v_source not in ("backbone", "independent"):
            raise ValueError("invalid sksa_indexer_chunk_attention_v_source")
        if chunk_v_source in ("independent",) and chunk_qk_source != "independent":
            raise ValueError("independent chunk V requires independent chunk Q/K")
        include_summary = config.setdefault("sksa_indexer_chunk_attention_include_summary", True)
        if not isinstance(include_summary, bool):
            raise ValueError("sksa_indexer_chunk_attention_include_summary must be boolean")
        if not include_summary and chunk_qk_source != "independent":
            raise ValueError("raw-only chunk attention requires independent chunk Q/K")
        raw_block_size = config.setdefault("sksa_indexer_block_size", 8)
        if (
            not isinstance(raw_block_size, int)
            or isinstance(raw_block_size, bool)
            or raw_block_size not in (2, 4, 8)
            or config["summary_chunk_size"] not in (4, 8)
            or config["summary_chunk_size"] % raw_block_size
        ):
            raise ValueError("indexer block2/4/8 must divide summary interval4/8")
        if raw_block_size < config["summary_chunk_size"] and (
            chunk_qk_source != "independent"
            or config["sksa_local_window_tokens"] < config["summary_chunk_size"]
        ):
            raise ValueError("subchunk indexer requires independent Q/K and local window >= summary interval")
        if config["sksa_indexer_token_budget"] % raw_block_size:
            raise ValueError("indexer token budget must be divisible by sksa_indexer_block_size")
        sink_tokens = config.setdefault("sksa_indexer_summary_sink_tokens", 8)
        if isinstance(sink_tokens, bool) or not isinstance(sink_tokens, int) or sink_tokens <= 0 or sink_tokens % raw_block_size:
            raise ValueError("summary sink tokens must be a positive whole number of blocks")
        if (
            config.get("sksa_indexer_force_sink", True)
            and config["sksa_indexer_token_budget"] < sink_tokens
        ):
            raise ValueError("indexer token budget must cover the summary sink tokens")
        sksa_required |= {
            "sksa_indexer_token_budget",
            "sksa_indexer_query_rank",
            "sksa_indexer_heads",
            "sksa_indexer_head_dim",
            "sksa_indexer_rope_dim",
            "sksa_indexer_query_block_size",
            "sksa_indexer_teacher_query_block_size",
            "sksa_indexer_teacher_pooling",
            "sksa_indexer_distill_attention",
            "sksa_apply_sparse_attention",
            "sksa_indexer_force_sink",
        }
        scorer_backend = config.setdefault(
            "sksa_indexer_scorer_backend", "torch"
        )
        if scorer_backend not in ("torch", "fused", "auto"):
            raise ValueError(
                "sksa_indexer_scorer_backend must be one of: "
                "torch, fused, auto"
            )
        sksa_missing = sorted(sksa_required - config.keys())
        if sksa_missing:
            raise ValueError(f"S-KSA model config is missing fields: {sksa_missing}")
    return config


def build_export_config(runtime_source: Path, model_config_path: Path) -> dict:
    with (runtime_source / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    # A shared runtime template may describe a later sparse stage. The saved
    # training config, not template defaults, decides which modules to build.
    config = {name: value for name, value in config.items() if not name.startswith("sksa_")}
    training = load_training_config(model_config_path)
    summary_vocab = training["summary_token_num"]
    config.update(
        {
            "vocab_size": training["vocab_size"] + summary_vocab,
            "hidden_size": training["embed_dim"],
            "num_hidden_layers": training["num_layers"],
            "max_window_layers": training["num_layers"],
            "num_attention_heads": training["num_heads"],
            "num_key_value_heads": training["num_kv_heads"],
            "head_dim": training["head_dim"],
            "intermediate_size": training["intermediate_dim"],
            "max_position_embeddings": training["max_seq_len"],
            "rope_theta": training["rope_base"],
            "rms_norm_eps": training["norm_eps"],
            "tie_word_embeddings": training["tie_word_embeddings"],
            "attention_bias": training.get("q_proj_bias", False),
            "attention_dropout": training.get("attn_dropout", 0.0),
            "use_summary_attention": training.get("use_summary_attention", True),
            "summary_chunk_size": training["summary_chunk_size"],
            "summary_token_num": training["summary_token_num"],
            "summary_partial_chunk_policy": training[
                "summary_partial_chunk_policy"
            ],
            "summary_token_begin": training["summary_token_begin"],
            "summary_layer_freq": training["summary_layer_freq"],
            "summary_sliding_chunk_num": training["summary_sliding_chunk_num"],
            "summary_chunk_position_ids_type": training[
                "summary_chunk_position_ids_type"
            ],
            "summary_token_position_ids_type": training[
                "summary_token_position_ids_type"
            ],
            "summary_independent_parameters": training[
                "summary_independent_parameters"
            ],
            "summary_independent_qk_norm": training.get("summary_independent_qk_norm", False),
            "summary_independent_attention_layernorm": training.get("summary_independent_attention_layernorm", False),
            "mix_coeff": 1.0 if training["summary_independent_qkv"] else 0.0,
            "summary_meta_pool_mode": training["summary_meta_pool_mode"],
            "summary_meta_pool_group_size": training[
                "summary_meta_pool_group_size"
            ],
            "summary_meta_pool_position": training["summary_meta_pool_position"],
            "summary_meta_pool_mass_correction": training[
                "summary_meta_pool_mass_correction"
            ],
            "summary_meta_pool_block_size": training["summary_block_size"],
            "truncate_predict_nums": training["vocab_size"],
            "sksa_enabled": training.get("sksa_enabled", False),
        }
    )
    if training.get("sksa_enabled", False):
        common_names = [
            "sksa_enabled",
            "sksa_sparse_full_layers",
            "sksa_local_window_tokens",
        ]
        routing_mode = training.get("sksa_routing_mode", "summary_indexer")
        config["sksa_routing_mode"] = routing_mode
        common_names.extend((
            "sksa_indexer_token_budget",
            "sksa_indexer_query_rank",
            "sksa_indexer_heads",
            "sksa_indexer_head_dim",
            "sksa_indexer_rope_dim",
            "sksa_indexer_query_block_size",
            "sksa_indexer_teacher_query_block_size",
            "sksa_indexer_teacher_pooling",
            "sksa_indexer_scorer_backend",
            "sksa_indexer_distill_attention",
            "sksa_apply_sparse_attention",
            "sksa_indexer_force_sink",
        ))
        common_names.extend((
            "sksa_indexer_summary_key_source",
            "sksa_indexer_chunk_attention_qk_source",
            "sksa_indexer_chunk_attention_v_source",
            "sksa_indexer_chunk_attention_include_summary",
            "sksa_indexer_block_size",
            "sksa_indexer_summary_sink_tokens",
        ))
        for name in common_names:
            config[name] = training[name]
    auto_map = config.get("auto_map", {})
    if auto_map.get("AutoModelForCausalLM") != "modeling_qwen3.Qwen3ForCausalLM":
        raise ValueError("runtime config does not define the trust_remote_code contract")
    return config


def muse_to_hf_name(name: str) -> str:
    global_map = {
        "model.norm.scale": "model.norm.weight",
        "model.output.weight": "lm_head.weight",
    }
    if name in global_map:
        return global_map[name]
    if not name.startswith("model.layers."):
        raise KeyError(f"unmapped Muse tensor: {name}")
    parts = name.split(".")
    layer_idx = parts[2]
    suffix = ".".join(parts[3:])
    suffix_map = {
        "sa_norm.scale": "input_layernorm.weight",
        "sa_norm_summary.scale": "input_layernorm_summary.weight",
        "mlp_norm.scale": "post_attention_layernorm.weight",
        "attn.q_proj.weight": "self_attn.q_proj.weight",
        "attn.k_proj.weight": "self_attn.k_proj.weight",
        "attn.v_proj.weight": "self_attn.v_proj.weight",
        "attn.output_proj.weight": "self_attn.o_proj.weight",
        "attn.q_norm.scale": "self_attn.q_norm.weight",
        "attn.k_norm.scale": "self_attn.k_norm.weight",
        "attn.q_proj_summary.weight": "self_attn.q_proj_summary.weight",
        "attn.k_proj_summary.weight": "self_attn.k_proj_summary.weight",
        "attn.v_proj_summary.weight": "self_attn.v_proj_summary.weight",
        "attn.q_norm_summary.scale": "self_attn.q_norm_summary.weight",
        "attn.k_norm_summary.scale": "self_attn.k_norm_summary.weight",
        "attn.meta_pool_scorer.weight": "self_attn.meta_pool_scorer.weight",
        "attn.meta_pool_position_bias": "self_attn.meta_pool_position_bias",
        "attn.sksa_indexer.query_down_proj.weight": "self_attn.sksa_indexer.query_down_proj.weight",
        "attn.sksa_indexer.query_up_proj.weight": "self_attn.sksa_indexer.query_up_proj.weight",
        "attn.sksa_indexer.key_proj.weight": "self_attn.sksa_indexer.key_proj.weight",
        "attn.sksa_indexer.head_weight_proj.weight": "self_attn.sksa_indexer.head_weight_proj.weight",
        "attn.sksa_indexer.chunk_q_proj.weight": "self_attn.sksa_indexer.chunk_q_proj.weight",
        "attn.sksa_indexer.chunk_v_proj.weight": "self_attn.sksa_indexer.chunk_v_proj.weight",
        "attn.sksa_indexer.chunk_v_proj.bias": "self_attn.sksa_indexer.chunk_v_proj.bias",
        "attn.sksa_indexer.chunk_k_proj.weight": "self_attn.sksa_indexer.chunk_k_proj.weight",
        "attn.sksa_indexer.chunk_q_proj.bias": "self_attn.sksa_indexer.chunk_q_proj.bias",
        "attn.sksa_indexer.chunk_k_proj.bias": "self_attn.sksa_indexer.chunk_k_proj.bias",
        "attn.sksa_indexer.chunk_q_norm.scale": "self_attn.sksa_indexer.chunk_q_norm.weight",
        "attn.sksa_indexer.chunk_k_norm.scale": "self_attn.sksa_indexer.chunk_k_norm.weight",
        "mlp.w1.weight": "mlp.gate_proj.weight",
        "mlp.w2.weight": "mlp.down_proj.weight",
        "mlp.w3.weight": "mlp.up_proj.weight",
    }
    if suffix not in suffix_map:
        raise KeyError(f"unmapped Muse layer tensor: {name}")
    return f"model.layers.{layer_idx}.{suffix_map[suffix]}"


def expected_hf_keys(training: dict) -> set[str]:
    layers = training["num_layers"]
    layer_is_summary = parse_pattern(training["summary_layer_freq"], layers)
    pool_groups = parse_pattern(training["summary_meta_pool_group_size"], layers)
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    if not training["tie_word_embeddings"]:
        keys.add("lm_head.weight")
    base_suffixes = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
        "mlp.up_proj.weight",
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        keys.update(prefix + suffix for suffix in base_suffixes)
        if layer_is_summary[layer] and training["summary_independent_qkv"]:
            keys.update(
                prefix + f"self_attn.{projection}_proj_summary.weight"
                for projection in ("q", "k", "v")
            )
        if layer_is_summary[layer] and training.get(
            "summary_independent_qk_norm", False
        ):
            keys.update(
                prefix + f"self_attn.{projection}_norm_summary.weight"
                for projection in ("q", "k")
            )
        if layer_is_summary[layer] and training.get(
            "summary_independent_attention_layernorm", False
        ):
            keys.add(prefix + "input_layernorm_summary.weight")
        if (
            layer_is_summary[layer]
            and training["summary_meta_pool_mode"] in {
                "scalar", "dimension_wise"
            }
            and pool_groups[layer] > 1
        ):
            keys.add(prefix + "self_attn.meta_pool_scorer.weight")
            keys.add(prefix + "self_attn.meta_pool_position_bias")
        if (
            training.get("sksa_enabled", False)
            and training.get("sksa_routing_mode", "summary_indexer")
            in {"summary_indexer"}
            and layer in {int(value) for value in training["sksa_sparse_full_layers"]}
        ):
            keys.update(
                prefix + f"self_attn.sksa_indexer.{name}.weight"
                for name in (
                    "query_down_proj",
                    "query_up_proj",
                    "key_proj",
                    "head_weight_proj",
                )
            )
            if training.get(
                "sksa_indexer_chunk_attention_qk_source", "backbone"
            ) == "independent":
                keys.update(
                    prefix + f"self_attn.sksa_indexer.{name}"
                    for name in (
                        "chunk_q_proj.weight",
                        "chunk_k_proj.weight",
                        "chunk_q_norm.weight",
                        "chunk_k_norm.weight",
                    )
                )
                if training.get("q_proj_bias", False):
                    keys.add(
                        prefix + "self_attn.sksa_indexer.chunk_q_proj.bias"
                    )
                if training.get("k_proj_bias", False):
                    keys.add(
                        prefix + "self_attn.sksa_indexer.chunk_k_proj.bias"
                    )
            if training.get("sksa_indexer_chunk_attention_v_source", "backbone") == "independent":
                keys.add(prefix + "self_attn.sksa_indexer.chunk_v_proj.weight")
                if training.get("v_proj_bias", False):
                    keys.add(prefix + "self_attn.sksa_indexer.chunk_v_proj.bias")
    return keys


def validate_meta_pool_shapes(
    state: Dict[str, torch.Tensor], training: dict
) -> None:
    mode = training["summary_meta_pool_mode"]
    if mode not in {"scalar", "dimension_wise"}:
        return
    layers = training["num_layers"]
    layer_is_summary = parse_pattern(training["summary_layer_freq"], layers)
    pool_groups = parse_pattern(training["summary_meta_pool_group_size"], layers)
    kv_width = training["num_kv_heads"] * training["head_dim"]
    scorer_shape = (
        (1, training["embed_dim"])
        if mode == "scalar"
        else (kv_width, training["embed_dim"])
    )
    for layer, (is_summary, group_size) in enumerate(
        zip(layer_is_summary, pool_groups)
    ):
        if not is_summary or group_size <= 1:
            continue
        prefix = f"model.layers.{layer}.self_attn."
        expected = {
            prefix + "meta_pool_scorer.weight": scorer_shape,
            prefix + "meta_pool_position_bias": (
                (group_size,)
                if mode == "scalar"
                else (group_size, kv_width)
            ),
        }
        for name, shape in expected.items():
            actual = tuple(state[name].shape)
            if actual != shape:
                raise RuntimeError(
                    f"invalid {mode} tensor shape for {name}: "
                    f"expected {shape}, got {actual}"
                )


def validate_sksa_indexer_shapes(
    state: Dict[str, torch.Tensor], training: dict
) -> None:
    if (
        not training.get("sksa_enabled", False)
        or training.get("sksa_routing_mode", "summary_indexer")
        not in {"summary_indexer"}
    ):
        return
    hidden = training["embed_dim"]
    rank = training["sksa_indexer_query_rank"]
    heads = training["sksa_indexer_heads"]
    head_dim = training["sksa_indexer_head_dim"]
    expected_suffixes = {
        "query_down_proj.weight": (rank, hidden),
        "query_up_proj.weight": (heads * head_dim, rank),
        "key_proj.weight": (head_dim, hidden),
        "head_weight_proj.weight": (heads, hidden),
    }
    if training.get(
        "sksa_indexer_chunk_attention_qk_source", "backbone"
    ) == "independent":
        chunk_heads = training["num_heads"]
        chunk_kv_heads = training["num_kv_heads"]
        chunk_head_dim = training["head_dim"]
        expected_suffixes.update(
            {
                "chunk_q_proj.weight": (chunk_heads * chunk_head_dim, hidden),
                "chunk_k_proj.weight": (
                    chunk_kv_heads * chunk_head_dim,
                    hidden,
                ),
                "chunk_q_norm.weight": (chunk_head_dim,),
                "chunk_k_norm.weight": (chunk_head_dim,),
            }
        )
        if training.get("q_proj_bias", False):
            expected_suffixes["chunk_q_proj.bias"] = (
                chunk_heads * chunk_head_dim,
            )
        if training.get("k_proj_bias", False):
            expected_suffixes["chunk_k_proj.bias"] = (
                chunk_kv_heads * chunk_head_dim,
            )
    if training.get("sksa_indexer_chunk_attention_v_source", "backbone") == "independent":
        expected_suffixes["chunk_v_proj.weight"] = (
            training["num_kv_heads"] * training["head_dim"], hidden,
        )
        if training.get("v_proj_bias", False):
            expected_suffixes["chunk_v_proj.bias"] = (training["num_kv_heads"] * training["head_dim"],)
    for layer in training["sksa_sparse_full_layers"]:
        prefix = f"model.layers.{int(layer)}.self_attn.sksa_indexer."
        for suffix, shape in expected_suffixes.items():
            name = prefix + suffix
            actual = tuple(state[name].shape)
            if actual != shape:
                raise RuntimeError(
                    f"invalid S-KSA indexer tensor shape for {name}: "
                    f"expected {shape}, got {actual}"
                )


def load_and_map_dcp(dcp_dir: Path, training: dict) -> Dict[str, torch.Tensor]:
    state = {}
    _load_state_dict(
        state,
        storage_reader=FileSystemReader(dcp_dir),
        # Optimizer/training state is much larger and is irrelevant to HF export.
        planner=_EmptyStateDictLoadPlanner(keys={"app.model"}),
        no_dist=True,
    )
    model_state = canonical_checkpoint_keys(state["app"]["model"])
    token_embedding = model_state.pop("model.tok_embeddings.weight")
    summary_embedding = model_state.pop("summary_embedding.weight")
    mapped = {
        "model.embed_tokens.weight": torch.cat(
            (token_embedding, summary_embedding), dim=0
        ).contiguous()
    }
    for name, tensor in model_state.items():
        mapped[muse_to_hf_name(name)] = tensor.contiguous()

    if "lm_head.weight" in mapped:
        lm_head = mapped["lm_head.weight"]
        if lm_head.shape[0] == training["vocab_size"]:
            summary_rows = lm_head.new_zeros(
                (training["summary_token_num"], lm_head.shape[1])
            )
            mapped["lm_head.weight"] = torch.cat((lm_head, summary_rows), dim=0)

    expected = expected_hf_keys(training)
    missing = sorted(expected - mapped.keys())
    unexpected = sorted(mapped.keys() - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"mapped checkpoint key mismatch; missing={missing}, unexpected={unexpected}"
        )
    validate_meta_pool_shapes(mapped, training)
    validate_sksa_indexer_shapes(mapped, training)
    return mapped


def save_sharded(
    state: Dict[str, torch.Tensor], output_dir: Path, max_shard_bytes: int
) -> None:
    shards, current, current_bytes = [], {}, 0
    for name in sorted(state):
        tensor = state[name]
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + tensor_bytes > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[name] = tensor
        current_bytes += tensor_bytes
    if current:
        shards.append(current)

    weight_map = {}
    total_size = sum(t.numel() * t.element_size() for t in state.values())
    for shard_idx, shard in enumerate(shards, start=1):
        filename = f"model-{shard_idx:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, output_dir / filename, metadata={"format": "pt"})
        for name in shard:
            weight_map[name] = filename
    with (output_dir / "model.safetensors.index.json").open("x") as handle:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def export(
    dcp_dir: Path,
    model_config_path: Path,
    runtime_source: Path,
    kernel_source: Path,
    generate_script: Path,
    output_dir: Path,
    shard_gib: float,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    if not (dcp_dir / ".metadata").is_file():
        raise FileNotFoundError(f"DCP checkpoint is incomplete: {dcp_dir}")
    if not kernel_source.is_dir():
        raise FileNotFoundError(f"kernel directory does not exist: {kernel_source}")
    if not generate_script.is_file():
        raise FileNotFoundError(f"generate script does not exist: {generate_script}")
    kernel_files = {
        str(path.relative_to(kernel_source)).removeprefix("v0.1/"): path
        for path in list_source_files(kernel_source)
    }
    if not kernel_files:
        raise FileNotFoundError(f"kernel directory contains no source files: {kernel_source}")

    training = load_training_config(model_config_path)
    validate_runtime_config(training)
    source_files = runtime_files_for(runtime_source)
    asset_paths = {
        name: runtime_source / name if (runtime_source / name).is_file() else TOKENIZER_DIR / name
        for name in ASSET_FILES
    }
    missing = [name for name, path in asset_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing tokenizer/config assets under {runtime_source}: {missing}")
    config = build_export_config(runtime_source, model_config_path)
    if (
        config.get("sksa_enabled", False)
        and config.get("summary_partial_chunk_policy") != "defer"
    ):
        raise ValueError("sparse export requires summary_partial_chunk_policy=defer")
    print(f"Loading and validating DCP: {dcp_dir}", flush=True)
    mapped = load_and_map_dcp(dcp_dir, training)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_sharded(mapped, output_dir, int(shard_gib * 1024**3))
    for name in source_files:
        shutil.copy2(runtime_source / name, output_dir / name)
    for name, path in asset_paths.items():
        shutil.copy2(path, output_dir / name)
    shutil.copy2(REPO_ROOT / "LICENSE", output_dir / "LICENSE")
    shutil.copytree(REPO_ROOT / "licenses", output_dir / "licenses")
    shutil.copy2(ROOT / "requirements-inference.txt", output_dir / "requirements.txt")
    shutil.copy2(generate_script, output_dir / "generate.py")
    for name, path in kernel_files.items():
        target = output_dir / "kernel" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    with (output_dir / "config.json").open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    pool_groups = parse_pattern(
        training["summary_meta_pool_group_size"], training["num_layers"]
    )
    meta_tensor_count = sum(
        2
        for group in pool_groups
        if training["summary_meta_pool_mode"] in {"scalar", "dimension_wise"}
        and group > 1
    )
    manifest = {
        "schema_version": 3,
        "format": "muse-dcp-to-huggingface",
        "runtime": "infra",
        "model_family": (
            ("C-KSA + S-KSA" if training["summary_meta_pool_mode"] != "none" else "KSA + S-KSA")
            if training.get("sksa_enabled", False)
            else {
                "none": "KSA",
                "scalar": "C-KSA",
                "dimension_wise": "C-KSA",
            }[training["summary_meta_pool_mode"]]
        ),
        "source_metadata_sha256": sha256(dcp_dir / ".metadata"),
        "training_model_config_sha256": sha256(model_config_path),
        "runtime_files_sha256": {
            name: sha256(runtime_source / name) for name in source_files
        },
        "asset_files_sha256": {
            name: sha256(path) for name, path in asset_paths.items()
        },
        "generate_script_sha256": sha256(generate_script),
        "kernel_files_sha256": {
            name: sha256(path) for name, path in kernel_files.items()
        },
        "tensor_count": len(mapped),
        "meta_tensor_count": meta_tensor_count,
        "tensor_bytes": sum(t.numel() * t.element_size() for t in mapped.values()),
        "meta_pool_mode": config["summary_meta_pool_mode"],
        "training_summary_partial_chunk_policy": training[
            "summary_partial_chunk_policy"
        ],
        "runtime_summary_partial_chunk_policy": config[
            "summary_partial_chunk_policy"
        ],
        "sksa_enabled": config.get("sksa_enabled", False),
    }
    if config.get("sksa_routing_mode") == "summary_indexer":
        manifest.update({
            "sksa_routing_mode": config["sksa_routing_mode"],
            "sksa_indexer_token_budget": config["sksa_indexer_token_budget"],
            "sksa_indexer_block_size": (
                (config["sksa_indexer_block_size"])
            ),
            "sksa_indexer_force_sink": config["sksa_indexer_force_sink"],
            "sksa_indexer_scorer_backend": config[
                "sksa_indexer_scorer_backend"
            ],
        })
        manifest.update({
            name: config[name] for name in (
                "sksa_indexer_summary_key_source",
                "sksa_indexer_chunk_attention_v_source",
                "sksa_indexer_chunk_attention_qk_source",
                "sksa_indexer_summary_sink_tokens",
            )
        })
    with (output_dir / "export_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Exported {len(mapped)} tensors to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", "--dcp-dir", dest="dcp_dir", type=Path, required=True,
        help="Muse DCP checkpoint directory (for example global_step5100)",
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--runtime-dir", type=Path, default=ROOT / "hf_template",
        help="HF runtime template directory",
    )
    parser.add_argument(
        "--kernel-dir", type=Path, default=KERNEL_ROOT / "inference",
        help="Inference kernel directory",
    )
    parser.add_argument(
        "--generate-script",
        type=Path,
        default=Path(__file__).with_name("generate.py"),
        help="Generation helper copied to <output>/generate.py",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="HF output directory (default: <checkpoint>/convert)",
    )
    parser.add_argument("--shard-gib", type=float, default=4.0)
    args = parser.parse_args()
    if args.shard_gib <= 0:
        parser.error("--shard-gib must be positive")
    output_dir = args.output_dir or args.dcp_dir / "convert"
    export(
        args.dcp_dir,
        args.model_config,
        args.runtime_dir,
        args.kernel_dir,
        args.generate_script,
        output_dir,
        args.shard_gib,
    )


if __name__ == "__main__":
    main()
