import json
import os
import os.path as osp
import argparse
import shutil

from checkpoint import dcp_to_torch_save


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
                     dtype="bf16", output_subdir="hf", remap="none"):
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
    dcp_to_torch_save(real_model_path, converted_path, model_only=True,
                      use_safetensor=True, max_gb_per_shard=4, dtype=dtype,
                      remap=remap)

    print(f"converted_path={converted_path}")

    for file in os.listdir(base_model_path):
        if file == "model.safetensors.index.json":
            continue
        if file.endswith('.json') or file.endswith('.py') or file.endswith('.txt'):
            shutil.copy(osp.join(base_model_path, file), osp.join(converted_path, file))

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
    parser.add_argument('--remap', type=str, default="none", choices=["none", "muse2hf"],
                        help="Key remapping: none (raw DCP keys) or muse2hf (Muse→HF)")
    args = parser.parse_args()

    converted_path = convert_dcp_to_hf(
        args.dcp_path,
        args.step,
        args.base_model_path,
        dtype=args.dtype,
        remap=args.remap,
    )

    print(converted_path)
