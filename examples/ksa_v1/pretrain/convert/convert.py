import json
import os
import os.path as osp
import argparse
import shutil
from pathlib import Path

from checkpoint import dcp_to_torch_save

REPO_ROOT = Path(__file__).resolve().parents[4]


def build_export_config(training, template):
    """Use the run's architecture, not the template's example dimensions."""
    if training.get("sksa_enabled") or training.get("scsa_enabled") or training.get("summary_meta_pool_mode", "none") != "none":
        raise ValueError("Use the V2 exporter for C-KSA or S-KSA checkpoints")
    if training.get("summary_mix_qkv") or training.get("summary_independent_qk_norm"):
        raise ValueError("V1 export requires fixed summary QKV and shared Q/K normalization")
    if training.get("summary_independent_qkv") is False:
        raise ValueError("Use the V2 exporter for shared-QKV checkpoints")
    config = dict(template)
    fields = {
        "embed_dim": "hidden_size", "num_layers": "num_hidden_layers",
        "num_heads": "num_attention_heads", "num_kv_heads": "num_key_value_heads",
        "head_dim": "head_dim", "intermediate_dim": "intermediate_size",
        "max_seq_len": "max_position_embeddings", "rope_base": "rope_theta",
        "norm_eps": "rms_norm_eps", "tie_word_embeddings": "tie_word_embeddings",
    }
    for source, target in fields.items():
        config[target] = training[source]
    for name in (
        "use_summary_attention", "summary_chunk_size", "summary_token_num",
        "summary_token_begin", "summary_sliding_chunk_num",
        "summary_chunk_position_ids_type", "summary_token_position_ids_type",
        "summary_independent_parameters",
    ):
        config[name] = training[name]
    config.update(
        attention_bias=training.get("q_proj_bias", False),
        attention_dropout=training.get("attn_dropout", 0.0),
        max_window_layers=training["num_layers"],
        vocab_size=training["vocab_size"] + training["summary_token_num"],
        truncate_predict_nums=training["vocab_size"],
        summary_layer_freq=training.get("summary_layer_freq", 0),
        summary_independent_attention_layernorm=training.get("summary_independent_attention_layernorm", False),
        mix_coeff=1.0,
    )
    return config


def _sync_vocab_size_from_checkpoint(converted_path):
    """Read actual embed_tokens / lm_head shapes from the converted safetensors
    and update vocab_size in config.json to match.

    Works for any model — no assumptions about summary tokens or specific architectures.
    Only updates when there is a mismatch; skips silently if files are missing.
    """
    config_path = osp.join(converted_path, "config.json")
    index_path = osp.join(converted_path, "model.safetensors.index.json")
    if not osp.exists(config_path) or not osp.exists(index_path):
        return

    with open(index_path) as f:
        weight_map = json.load(f).get("weight_map", {})

    target_key = None
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        if key in weight_map:
            target_key = key
            break
    if target_key is None:
        return

    shard_file = osp.join(converted_path, weight_map[target_key])
    if not osp.exists(shard_file):
        return

    from safetensors import safe_open
    with safe_open(shard_file, framework="pt") as f:
        actual_vocab = f.get_tensor(target_key).shape[0]

    with open(config_path) as f:
        cfg = json.load(f)

    old_vocab = cfg.get("vocab_size")
    if old_vocab != actual_vocab:
        cfg["vocab_size"] = actual_vocab
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"Updated config.json vocab_size: {old_vocab} → {actual_vocab} "
              f"(detected from {target_key})")
    else:
        print(f"config.json vocab_size={old_vocab} matches checkpoint, no update needed")


def convert_dcp_to_hf(checkpoint_dir, model_path, base_model_path,
                     dtype="bf16", output_subdir="hf", remap="none", model_config=None):
    """Convert a Muse DCP checkpoint to HF-compatible safetensors.

    Args:
        checkpoint_dir: training output dir containing global_step* subdirs
        model_path: subdir name inside checkpoint_dir, e.g. "global_step5000"
        base_model_path: HF model dir whose tokenizer/config/modeling files
            are copied alongside the converted weights
        dtype: bf16 / fp16 / fp32
        output_subdir: subfolder under <checkpoint_dir>/<model_path> to write into
        remap: key remapping strategy.
            "none"     — no remapping (raw DCP keys)
            "muse2hf"  — Muse keys → HuggingFace keys (Qwen3 + Summary Attention)
    """
    real_model_path = osp.join(checkpoint_dir, model_path)
    converted_path = osp.join(real_model_path, output_subdir)
    if Path(converted_path).exists() and any(Path(converted_path).iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {converted_path}")
    config_path = Path(model_config) if model_config else Path(checkpoint_dir) / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Pass --model-config with the actual training config; " + str(config_path))
    config = build_export_config(
        json.loads(config_path.read_text()),
        json.loads((Path(base_model_path) / "config.json").read_text()),
    )
    dcp_to_torch_save(real_model_path, converted_path, model_only=True,
                      use_safetensor=True, max_gb_per_shard=4, dtype=dtype,
                      remap=remap)

    print(f"converted_path={converted_path}")

    for file in os.listdir(base_model_path):
        if file in {"model.safetensors.index.json", "config.json"}:
            continue
        if Path(file).suffix in {'.json', '.py', '.txt', '.jinja'}:
            shutil.copy(osp.join(base_model_path, file), osp.join(converted_path, file))

    for source in (REPO_ROOT / "assets/tokenizer").iterdir():
        if source.suffix in {".json", ".txt"} and not (Path(converted_path) / source.name).exists():
            shutil.copy2(source, Path(converted_path) / source.name)
    (Path(converted_path) / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    _sync_vocab_size_from_checkpoint(converted_path)

    return converted_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcp_path', type=str, required=True,
                        help="Training output dir containing global_step* subdirs")
    parser.add_argument('--step', type=str, required=True,
                        help="e.g. global_step5000")
    parser.add_argument('--base_model_path', type=str, required=True,
                        help="HF model dir whose tokenizer/config/modeling files "
                             "are copied alongside the converted weights")
    parser.add_argument('--dtype', type=str, default="bf16")
    parser.add_argument('--model-config', type=Path,
                        help="Saved training config; defaults to <dcp_path>/model_config.json")
    parser.add_argument('--remap', type=str, default="none", choices=["none", "muse2hf"],
                        help="Key remapping: none (raw DCP keys) or muse2hf (Muse→HF)")
    args = parser.parse_args()

    converted_path = convert_dcp_to_hf(
        args.dcp_path,
        args.step,
        args.base_model_path,
        dtype=args.dtype,
        remap=args.remap,
        model_config=args.model_config,
    )

    print(converted_path)
