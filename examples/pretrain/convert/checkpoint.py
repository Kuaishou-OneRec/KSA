from typing import Union, Dict

import os
import json
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
import tqdm
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner
)
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import re


def remap_muse_to_hf(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap Muse state_dict keys to HuggingFace format.

    Covers standard Qwen3 and Summary Attention extensions.
    Pure string replacement — no tensor transform, no muse dependency.

    summary_embedding.weight is appended to model.embed_tokens.weight
    so that HF embed_tokens covers summary token IDs natively.
    """
    out: Dict[str, torch.Tensor] = {}
    summary_emb = None

    for k, v in state_dict.items():
        # ── top-level ──
        if k == "model.tok_embeddings.weight":
            out["model.embed_tokens.weight"] = v
            continue
        if k == "model.norm.scale":
            out["model.norm.weight"] = v
            continue
        if k == "model.output.weight":
            out["lm_head.weight"] = v
            continue
        if k == "summary_embedding.weight":
            summary_emb = v
            continue

        # ── per-layer ──
        if k.startswith("model.layers."):
            parts = k.split(".", 3)  # ['model', 'layers', '{i}', rest]
            if len(parts) < 4:
                out[k] = v
                continue
            prefix = f"model.layers.{parts[2]}"
            rest = parts[3]

            # layer norms (must match sa_norm_summary before sa_norm)
            if rest == "sa_norm_summary.scale":
                out[f"{prefix}.input_layernorm_summary.weight"] = v
                continue
            if rest == "sa_norm.scale":
                out[f"{prefix}.input_layernorm.weight"] = v
                continue
            if rest == "mlp_norm.scale":
                out[f"{prefix}.post_attention_layernorm.weight"] = v
                continue

            # attention
            if rest.startswith("attn."):
                hf_rest = rest.replace("attn.", "self_attn.", 1)
                hf_rest = hf_rest.replace("output_proj", "o_proj")
                hf_rest = hf_rest.replace("q_norm_summary.scale", "q_norm_summary.weight")
                hf_rest = hf_rest.replace("k_norm_summary.scale", "k_norm_summary.weight")
                hf_rest = hf_rest.replace("q_norm.scale", "q_norm.weight")
                hf_rest = hf_rest.replace("k_norm.scale", "k_norm.weight")
                out[f"{prefix}.{hf_rest}"] = v
                continue

            # mlp
            if rest == "mlp.w1.weight":
                out[f"{prefix}.mlp.gate_proj.weight"] = v
                continue
            if rest == "mlp.w3.weight":
                out[f"{prefix}.mlp.up_proj.weight"] = v
                continue
            if rest == "mlp.w2.weight":
                out[f"{prefix}.mlp.down_proj.weight"] = v
                continue

        # fallthrough — keep as-is
        out[k] = v

    # Append summary_embedding to embed_tokens and pad lm_head to match
    if summary_emb is not None and "model.embed_tokens.weight" in out:
        embed = out["model.embed_tokens.weight"]
        out["model.embed_tokens.weight"] = torch.cat([embed, summary_emb], dim=0)
        new_vocab = out["model.embed_tokens.weight"].shape[0]
        print(f"Merged summary_embedding ({summary_emb.shape[0]} tokens) into embed_tokens: "
              f"{embed.shape[0]} → {new_vocab}")

        # Pad lm_head with zeros so its vocab dim matches embed_tokens
        if "lm_head.weight" in out:
            lm = out["lm_head.weight"]
            if lm.shape[0] < new_vocab:
                pad = torch.zeros(new_vocab - lm.shape[0], lm.shape[1],
                                  dtype=lm.dtype, device=lm.device)
                out["lm_head.weight"] = torch.cat([lm, pad], dim=0)
                print(f"Padded lm_head.weight: {lm.shape[0]} → {new_vocab}")

    return out


SHARD_FNAME = "model-{cpt_idx}-of-{num_shards}"

def dcp_to_torch_save(dcp_checkpoint_dir: Union[str, os.PathLike],
                      output_dir: Union[str, os.PathLike],
                      model_only: bool=True,
                      use_safetensor: bool=True,
                      max_gb_per_shard: int = 4,
                      model_type:str="Intern",
                      dtype="bf16",
                      remap="none"
                      ):
  """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch save file.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        torch_save_path: Filename to store the converted Torch save file, e.g., 
            /path/to/model/pytorch_model.bin
        model_only: Save model weights only

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
  """
  if dtype == "fp32":
    dtype_ = torch.float32
  elif dtype == "fp16":
    dtype_ = torch.float16
  elif dtype == "bf16":
    dtype_ = torch.bfloat16
  else:
    raise ValueError(f"Unsupported dtype {dtype}")

  print(f"using dtype={dtype}")


  sd: STATE_DICT_TYPE = {}
  import os
  print(f"dcp_to_torch_save({os.getpid()}): _load_state_dict ...")
  _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
  )

  print(f"dcp_to_torch_save: _load_state_dict done.")
  # if listinstr(["qwen"], model_type.lower()):
  #   sd = Qwen2VLCheckpointConverter().tp_to_original(sd)
  if model_only:
    sd = sd["app"]["model"]

  if remap == "muse2hf":
    print(f"Remapping {len(sd)} keys: Muse → HuggingFace")
    sd = remap_muse_to_hf(sd)

  # Split into shards + cast dtype in a single pass
  split_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
  cpt_idx = 0
  total_size = 0
  current_size = 0
  for key, weight in tqdm.tqdm(sd.items(), desc="Shard+Cast"):
    weight = weight.to(dtype_)
    if cpt_idx not in split_state_dicts:
      split_state_dicts[cpt_idx] = {}
    split_state_dicts[cpt_idx][key] = weight
    wsize = weight.numel() * weight.element_size()
    current_size += wsize
    total_size += wsize
    if current_size >= max_gb_per_shard * 1024 * 1024 * 1024:
      cpt_idx += 1
      current_size = 0
  del sd

  # write the partitioned state dicts to the right checkpoint file
  # e.g. model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc
  num_shards = len(split_state_dicts)
  weight_map = {}
  for cpt_idx, model_state_dict in tqdm.tqdm(split_state_dicts.items()):
    # TODO: We should probably use the original shard name and just add a prefix
    # however, having the SHARD_FNAME standardizes our checkpoints
    shard_name = SHARD_FNAME.format(
      cpt_idx=f"{cpt_idx}".zfill(5), num_shards=f"{num_shards}".zfill(5)
    )
    output_path = Path(output_dir) / shard_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not use_safetensor:
      output_path = output_path.with_suffix(".bin")
      torch.save(model_state_dict, output_path)
    else:
      output_path = output_path.with_suffix(".safetensors")
      save_file(model_state_dict, output_path, metadata={"format": "pt"})
    for key, weight in model_state_dict.items():
      weight_map[key] = str(output_path.parts[-1])

    print(
      "Model checkpoint of size "
      f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
      f"saved to {output_path}"
    )
    
  if use_safetensor:
    weight_map_path = Path(output_dir) / "model.safetensors.index.json"
  else:
    weight_map_path = Path(output_dir) / "model.bin.index.json"
  with open(weight_map_path, "w") as f:
    f.write(json.dumps({
      "metadata": {
        "total_size": total_size
      },
      "weight_map": weight_map,
    }, indent=2))

def get_argument_parser():
  parser = argparse.ArgumentParser()

  ############ Checkpoint args ############
  parser.add_argument("--checkpoint_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  parser.add_argument("--output_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  return parser

def main():
  arg_parser = get_argument_parser()
  args = arg_parser.parse_args()
  dcp_to_torch_save(
    dcp_checkpoint_dir=args.checkpoint_dir,
    output_dir=args.output_dir,
    model_only=True,
    use_safetensor=True,
    max_gb_per_shard=5
  )

if __name__ == "__main__":
  main()

