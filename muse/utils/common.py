# TODO: clean utils
from typing import Tuple, Any
#from rich import print
import time
import torch
import random
import numpy as np
from pathlib import Path
from transformers import set_seed as set_transformers_seed
import torch.distributed as dist
import pickle
import traceback
import subprocess
import os
import inspect


def print_rank_n(*msg, rank=0):
  """
  Print message only on specified rank.
  If torch.distributed is not initialized, behaves like regular print (all ranks are 0).
  """
  if not dist.is_initialized():
    # If dist is not initialized, treat as rank 0 and always print
    print(*msg)
  elif dist.get_rank() == rank:
    print(*msg)

def print_rank_0(*msg):
  print_rank_n(*msg, rank=0)

def to_device(batch, device, non_blocking=True):
  for key in list(batch.keys()):
    if isinstance(batch[key], torch.Tensor):
      batch[key] = batch[key].to(device=device, non_blocking=non_blocking)
  return batch

def to_cuda(batch, non_blocking=True):
  to_device(batch, device=torch.cuda.current_device(), non_blocking=non_blocking)

def set_random_seed(seed):
  if seed is not None:
    set_transformers_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def increment_version(version):
  major, minor, patch = map(int, version.split('.'))
  patch += 1
  return f"{major}.{minor}.{patch}"

def dist_reduce_dict(local_dict, group=None):
  gather_list = [None for _ in range(dist.get_world_size(group=group))]

  dist.all_gather_object(
    object_list=gather_list, obj=local_dict, group=group)

  def reduce_dicts(dicts):
    def _reduce(d1, d2):
      for key, value in d2.items():
        if isinstance(value, dict):
          if key not in d1:
            d1[key] = {}
          _reduce(d1[key], value)
        else:
          if key in d1:
            d1[key] += value
          else:
            d1[key] = value
      return d1

    result = {}
    for d in dicts:
        result = _reduce(result, d)
    return result

  return reduce_dicts(gather_list)

class Timer:
  def __init__(self, desc: str = ""):
    self.desc = desc

  def __enter__(self):
    print_rank_0(f"Start... {self.desc}")
    self.start = time.time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.end = time.time()
    self.elapsed = self.end - self.start
    print_rank_0(f"End... {self.desc} elapsed: {self.elapsed:.3f} ")

def get_root_dir():
  current_file_path = Path(__file__).resolve()
  # recovlm/recovlm/utils/common.py
  root_dir = current_file_path.parent.parent.parent
  return root_dir

def load_env():
  env_path = get_root_dir() / ".env"
  env = {}
  with open(env_path, encoding="utf-8") as f:
    for line in f:
      key, value = line.strip().split("=")
      env[key] = str(value)
  return env

def format_dict_or_list(obj, indent_level=0, indent_size=2):
    """Format dict/list for printing, used to replace json.dumps"""
    def format_value(value, indent_level=0, indent_size=2):
        if isinstance(value, (dict, list)):
            return format_dict_or_list(value, indent_level, indent_size)
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)

    if isinstance(obj, dict):
        items = [f": {format_value(v, indent_level + 1)}" for k, v in obj.items()]
        keys = [f'"{k}"' for k in obj.keys()]
        formatted_items = ',\n'.join(f'{(" " * indent_size * (indent_level + 1))}{k}{v}' for k, v in zip(keys, items))
        return '{\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + '}'
    elif isinstance(obj, list):
        items = [format_value(item, indent_level + 1) for item in obj]
        formatted_items = ',\n'.join(' ' * indent_size * (indent_level + 1) + item for item in items)
        return '[\n' + formatted_items + '\n' + (' ' * indent_size * indent_level) + ']'
    else:
        return obj

def inspect_file(py_obj: Any):
  # 获取源代码文件路径
  try:
    source_file = inspect.getfile(py_obj.__class__)
    print(f"source file: {source_file}")
  except:
    print("cannot get source file path")

  # 获取源代码行号
  try:
    source_lines = inspect.getsourcelines(py_obj.__class__)
    print(f"start line number: {source_lines[1]}")
  except:
    print("cannot get source code line number")

def parse_config_overrides(overrides: list) -> dict:
    """Parse config override strings into a dictionary.
    
    Args:
        overrides: List of strings in format "key=value"
        
    Returns:
        Dictionary of parsed overrides with appropriate types
        
    Example:
        >>> parse_config_overrides(["use_pe=true", "pe_interpolation=1.0"])
        {"use_pe": True, "pe_interpolation": 1.0}
    """
    result = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")
        
        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # Parse value to appropriate type
        if value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        elif value.lower() == "none":
            result[key] = None
        else:
            # Try to parse as number
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                # Keep as string
                result[key] = value
    
    return result
