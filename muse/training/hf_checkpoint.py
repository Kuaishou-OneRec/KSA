"""Initialize a training model from a released Hybrid KSA HF checkpoint."""

import json
from pathlib import Path

import torch

from muse.training.checkpoint import load_hf_checkpoint


def validate_hf_parent(hf, config):
    """Check semantics which tensor shapes alone cannot establish."""
    fields = {
        "hidden_size": config.embed_dim,
        "intermediate_size": config.intermediate_dim,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_heads,
        "num_key_value_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "rms_norm_eps": config.norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
        "summary_chunk_size": config.summary_chunk_size,
        "summary_token_num": config.summary_token_num,
        "summary_token_begin": config.vocab_size,
        "vocab_size": config.vocab_size + config.summary_token_num,
        "summary_chunk_position_ids_type": config.summary_chunk_position_ids_type,
        "summary_token_position_ids_type": config.summary_token_position_ids_type,
    }
    mismatch = [name for name, value in fields.items() if hf.get(name) != value]
    rope = hf.get("rope_parameters", hf.get("rope_scaling")) or {}
    if rope.get("rope_theta", hf.get("rope_theta", 10000)) != config.rope_base:
        mismatch.append("rope_theta")
    if rope.get("rope_type", "default") != "default":
        mismatch.append("rope_type")
    if not hf.get("use_summary_attention") or not config.use_summary_attention:
        mismatch.append("use_summary_attention")
    if not hf.get("summary_independent_parameters") or not config.summary_independent_parameters:
        mismatch.append("summary_independent_parameters")
    if hf.get("mix_coeff", 1) != int(config.use_independent_qkv):
        mismatch.append("Summary QKV sharing")
    for name in ("summary_independent_attention_layernorm", "summary_independent_qk_norm"):
        if hf.get(name, False) != getattr(config, name):
            mismatch.append(name)
    if config.summary_token_begin != config.vocab_size:
        mismatch.append("summary_token_begin")
    if hf.get("summary_meta_pool_mode", "none") != "none" or hf.get("sksa_enabled", False):
        mismatch.append("parent must be Hybrid KSA (without C-KSA / S-KSA)")
    for name, default in (("summary_layer_freq", 1), ("summary_sliding_chunk_num", None)):
        source = config.parse_pattern_string(hf.get(name, default))
        target = getattr(config, name)
        source = [source] * config.num_layers if isinstance(source, int) else source
        target = [target] * config.num_layers if isinstance(target, int) else target
        if source != target:
            mismatch.append(name)
    if hf.get("attention_bias", False) != config.q_proj_bias or any(
        getattr(config, name) != config.q_proj_bias for name in ("k_proj_bias", "v_proj_bias")
    ):
        mismatch.append("attention_bias")
    if mismatch:
        raise ValueError("HF parent is incompatible with training config: " + ", ".join(mismatch))


def hf_to_muse(state, config):
    """Invert the public HF export, including the appended Summary embedding."""
    embedding = state["model.embed_tokens.weight"]
    expected_shape = (config.vocab_size + config.summary_token_num, config.embed_dim)
    if tuple(embedding.shape) != expected_shape:
        raise ValueError(f"HF embedding shape {tuple(embedding.shape)} != {expected_shape}")
    out = {
        "model.tok_embeddings.weight": embedding[:config.vocab_size],
        "summary_embedding.weight": embedding[config.vocab_size:],
    }
    for name, tensor in state.items():
        if name == "model.embed_tokens.weight":
            continue
        if name == "lm_head.weight":
            if tuple(tensor.shape) != expected_shape:
                raise ValueError("HF LM head must include the appended Summary rows")
            if config.tie_word_embeddings:
                if not torch.equal(tensor[:config.vocab_size], embedding[:config.vocab_size]):
                    raise ValueError("HF tied LM head and token embeddings disagree")
            else:
                out["model.output.weight"] = tensor[:config.vocab_size]
            continue
        target = name.replace("model.norm.weight", "model.norm.scale")
        if name.startswith("model.layers."):
            target = target.replace(".input_layernorm_summary.weight", ".sa_norm_summary.scale")
            target = target.replace(".input_layernorm.weight", ".sa_norm.scale")
            target = target.replace(".post_attention_layernorm.weight", ".mlp_norm.scale")
            target = target.replace(".self_attn.", ".attn.").replace(".o_proj.", ".output_proj.")
            for norm in ("q_norm", "k_norm", "q_norm_summary", "k_norm_summary"):
                target = target.replace(f".{norm}.weight", f".{norm}.scale")
            for hf_name, muse_name in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
                target = target.replace(f".mlp.{hf_name}.", f".mlp.{muse_name}.")
        if target in out:
            raise ValueError(f"Duplicate HF mapping: {target}")
        out[target] = tensor
    return out


def load_ksa_hf_checkpoint(model_dir, model, allow_random_init_params=None):
    """Load weights only; optimizer, step counter and data cursor start fresh."""
    with (Path(model_dir) / "config.json").open() as handle:
        hf = json.load(handle)
    validate_hf_parent(hf, model.config)
    state = hf_to_muse(load_hf_checkpoint(model_dir), model.config)
    expected = model.state_dict()
    allowed = [name.strip() for name in (allow_random_init_params or "").split(",") if name.strip()]
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    disallowed = [name for name in missing if not any(
        name == suffix or name.endswith("." + suffix) for suffix in allowed
    )]
    mismatched = [name for name in set(expected) & set(state) if expected[name].shape != state[name].shape]
    if unexpected or disallowed or mismatched:
        raise ValueError(
            f"HF weights do not match training model: missing={sorted(disallowed)}, "
            f"unexpected={sorted(unexpected)}, shape_mismatch={sorted(mismatched)}"
        )
    for name in expected:
        if name in missing:
            value = torch.empty(expected[name].shape, dtype=expected[name].dtype, device="cpu")
            model.get_initializer(name)(value)
            state[name] = value
    return state
