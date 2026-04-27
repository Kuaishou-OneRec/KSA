"""
Distributed Training Utilities and FSDP Support.

This module provides comprehensive utilities for distributed training with PyTorch,
with a focus on Fully Sharded Data Parallel (FSDP) training. Key features:

- Process group management and initialization
- FSDP model sharding with configurable strategies
- Distributed state dict loading and saving
- Mixed precision training support
- CPU offloading capabilities
- Checkpoint loading from full model state dicts

The module supports both single-machine and multi-node distributed training,
with automatic detection of distributed environments. It provides helpers for
FSDP-specific operations like sharding conditions, parameter prefetching, and
distributed checkpoint management.

Functions:
    get_world_size_and_rank: Get distributed world size and rank
    is_distributed: Check if running in distributed mode
    get_distributed_backend: Get appropriate backend for device type
    validate_no_params_on_meta_device: Validate no meta device parameters
    get_shard_conditions: Determine which modules to shard
    shard_model: Apply FSDP sharding to a model
    load_from_full_model_state_dict: Load full state dict into FSDP model
    initialize_model_params: Initialize model parameters for training from scratch
    load_from_full_model_state_dict_local: Single-machine version of above

Constants:
    _DISTRIBUTED_STATE_DICT_API_IS_AVAILABLE: Flag for DSD API availability
    process_group_timeout: Timeout for process group operations

Example:
    >>> import torch.distributed as dist
    >>> from muse.training.distributed import shard_model, is_distributed
    >>> 
    >>> # Initialize distributed training
    >>> if is_distributed():
    ...     dist.init_process_group(backend="nccl")
    >>> 
    >>> # Shard model with FSDP
    >>> model = MyLargeModel()
    >>> shard_model(
    ...     model,
    ...     cpu_offload=False,
    ...     reshard_after_forward=True,
    ...     param_dtype=torch.bfloat16
    ... )
    >>> 
    >>> # Load from checkpoint
    >>> full_state_dict = load_hf_checkpoint("./checkpoint")
    >>> load_from_full_model_state_dict(model, full_state_dict)
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# 
import logging
import os
from itertools import chain
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
import datetime

process_group_timeout = datetime.timedelta(minutes=60*24)

from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer

from muse.utils.common import format_dict_or_list, print_rank_n, print_rank_0


torch_version = torch.__version__
# TODO: Fix issues with DSD before uncommenting. See #2313 and #2277.
# _DISTRIBUTED_STATE_DICT_API_IS_AVAILABLE = (
#     "dev" not in torch_version and torch_version_ge("2.6.0")
# ) or ("dev" in torch_version and torch_version.split("dev")[1] >= "20241220")
_DISTRIBUTED_STATE_DICT_API_IS_AVAILABLE = False


def get_world_size_and_rank() -> Tuple[int, int]:
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current process in the default process group.

    Returns:
        Tuple[int, int]: world size, rank
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    else:
        return 1, 0
    
def is_distributed() -> bool:
    """Check if all environment variables required to initialize torch.distributed are set
    and distributed is properly installed. This indicates a distributed run.
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization

    Checks the following conditions:

    * torch.distributed is available
    * master port and master address environment variables are set
    * world size is >1
    * rank environment variable is set

    Returns:
        bool: True if all of the above conditions hold, False otherwise.
    """
    port = os.environ.get("MASTER_PORT", "")
    addr = os.environ.get("MASTER_ADDR", "")
    size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    avlb = dist.is_available()
    return bool(port and addr and size >= 1 and rank >= 0 and avlb)


def get_distributed_backend(device_type: str, offload_ops_to_cpu: bool = False) -> str:
    """Gets the PyTorch Distributed backend based on device type.

    Args:
        device_type (str): Device type to get backend for.
        offload_ops_to_cpu (bool, optional): Flag to check if any operations should be offloaded to CPU.
            Examples of these kinds of operations are CPU offload for FSDP and asynchronous save for distributed
            checkpointing. Defaults to False.

    Example:
        >>> get_distributed_backend("cuda")
        'nccl'
        >>> get_distributed_backend("cpu")
        'gloo'
        >>> get_distributed_backend("cuda", offload_ops_to_cpu=True)
        'cuda:nccl,cpu:gloo'

    Returns:
        str: Distributed backend for use in ``torch.distributed.init_process_group``.
    """
    default_device_backend_map = dist.Backend.default_device_backend_map
    backend = "nccl"
    if device_type in default_device_backend_map:
        backend = default_device_backend_map[device_type]
    if offload_ops_to_cpu:
        backend = f"{device_type}:{backend},cpu:gloo"
    return backend


def validate_no_params_on_meta_device(model: nn.Module) -> None:
    """
    Utility to validate that model has no params or buffers on meta device.
    If a meta param or buffer is found, an error indicating the param name will
    be raised.

    Args:
        model (nn.Module): model to check for meta params

    Raises:
        RuntimeError: If meta params or buffers exist in model
    """
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")

def get_shard_conditions(
    name: str,
    module: nn.Module,
    names_to_match: Optional[List[str]] = None,
    model_class=None,
    *args,
    **kwargs,
) -> bool:
    """
    Returs True for layers named {}.layers.i or layers that exactly match names_to_match, otherwise,
    returns False. This is a helper function for sharding a model with FSDP.
    In :func:`~torchtune.training.shard_model`, we iterate over the model's named modules
    and apply fully_shard using this condition.

    As part of our sharding strategy, we want each layer to be sharded separately, as this is
    generally efficient. We may also want to shard certain modules that are not layers, such as
    the embedding module.

    #TODO: a more robust way would be to shard on the module type, not the name.

    Args:
        name (str): Name of the module.
        module (nn.Module): Module to be sharded.
        names_to_match (Optional[List[str]]): List of names to match, if any.
        *args: Variable length argument list to be passed to the Embedding module.
        **kwargs: Arbitrary keyword arguments to be passed to the Embedding module.

    Returns:
        bool: True if the module name matches the condition, False otherwise.

    Examples:
        >>> names_to_match = ["embedding"]
        >>> layer_names = ["layers.0", "decoder.layers.1", "encoder.layers.2.attention",
            "my_wrapper.layer.1.something", "embedding"]
        >>> matches = []
        >>> for name in layer_names:
        >>>     if shard_condition_is_layer_or_match(name, None): matches.append(name)
        >>> print(matches)
        >>> ["layers.0", "decoder.layers.1", "embedding"]
    """

    if names_to_match and name in names_to_match:
        return True

    name_list = name.split(".")
    if len(name_list) >= 2:
        res = name_list[-2] == "layers" and str.isdigit(name_list[-1])
        return res

    return False


def shard_model(
    model: "Model", # noqa: F821, muse.models.base.Model
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    prefetch_params_in_forward: bool = True,
    dp_mesh: Optional[DeviceMesh] = None,
    fp32_weight=True,
    fp32_reduce=True,
    param_dtype=torch.bfloat16) -> None:
    """
    Utility to shard a model with FSDP using the PyTorch Distributed fully_shard API.

    Args:
        model (Model): Model to shard with FSDP.
        cpu_offload (bool): If set to True, FSDP will offload parameters, gradients, and optimizer
            states to CPU.
        reshard_after_forward (bool): Whether to reshard parameters and buffers after
            the forward pass. Setting this to True corresponds to the FULL_SHARD sharding strategy
            from FSDP1, while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.
        dp_mesh (Optional[DeviceMesh]): Device mesh to use for FSDP sharding under mutliple parallelism.
            Default to None.

    Raises:
        ValueError: If no layer modules were sharded.
    """
    fsdp_kwargs = {"reshard_after_forward": reshard_after_forward, "mesh": dp_mesh}
    fp32_reduce = True
    if fp32_weight: fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=torch.float32 if fp32_reduce else torch.bfloat16)
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # Store compute dtype on model for use in forward pass
    model._compute_dtype = param_dtype if fp32_weight else None

    # Shard the model with FSDP, iterating in reverse to start with
    # lowest-level modules first
    num_layers_sharded = 0

    layers = model.get_layers_to_shard()
    for m in layers:
        fully_shard(m, **fsdp_kwargs)
        num_layers_sharded += 1

    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)

    if prefetch_params_in_forward:
        prev = None
        for layer in reversed(layers):
            if prev is not None:
                layer.set_modules_to_forward_prefetch([prev])
            prev = layer

        model.set_modules_to_forward_prefetch([prev])


def load_from_full_model_state_dict(model: "FSDPModule",
                                    full_sd: Dict[str, Any],
                                    allow_random_init_params: Optional[Union[str, List[str]]] = None,
                                    skip_load_params: Optional[Union[str, List[str]]] = None):
    if isinstance(allow_random_init_params, str):
      allow_random_init_params = allow_random_init_params.split(',')
    
    # Parse skip_load_params (prefix matching)
    if isinstance(skip_load_params, str):
        skip_load_params = [p.strip() for p in skip_load_params.split(',') if p.strip()]
    
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    if dist.get_rank() == 0:
        extra_meta_sharded_sd = set(meta_sharded_sd.keys()) - set((full_sd.keys()))
        extra_full_ds = set(full_sd.keys()) - set((meta_sharded_sd.keys()))
        extra_meta_sharded_sd = {
            k:(v.shape, v.device, v.dtype) for k, v in meta_sharded_sd.items() if k in extra_meta_sharded_sd
        }
        extra_full_ds = {
            k:(v.shape, v.device, v.dtype) for k, v in full_sd.items() if k in extra_full_ds
        }

        # Detect shape mismatches for keys that exist in both
        shape_mismatches = {}
        for k in set(meta_sharded_sd.keys()) & set(full_sd.keys()):
            model_shape = meta_sharded_sd[k].shape
            sd_shape = full_sd[k].shape
            if model_shape != sd_shape:
                shape_mismatches[k] = {
                    'model_shape': model_shape,
                    'state_dict_shape': sd_shape
                }

        full_sd_info = {
            k: (v.shape, v.device, v.dtype)
            for k, v in full_sd.items()
        }
        print_rank_0(f"full_sd={format_dict_or_list(full_sd_info)}")
        
        meta_sharded_sd_info = {
            k: (v.shape, v.device, v.dtype)
            for k, v in meta_sharded_sd.items()
        }
        print_rank_0(f"meta_sharded_sd={format_dict_or_list(meta_sharded_sd_info)}")

        device0 = full_sd[list(full_sd)[0]]
        
        # Force random init for skip_load_params (even if they exist in checkpoint)
        if skip_load_params:
            for k in list(full_sd.keys()):
                should_skip = any(prefix in k for prefix in skip_load_params)
                if should_skip and k in meta_sharded_sd:
                    full_sd[k] = torch.empty(meta_sharded_sd[k].shape)
                    model.get_initializer(k)(full_sd[k])
                    full_sd[k] = full_sd[k].to(device0)
                    print_rank_0(f"Skip load, random init: {k}, shape={full_sd[k].shape}")
        
        for k in extra_meta_sharded_sd:
            if allow_random_init_params is not None and k in allow_random_init_params:
                full_sd[k] = torch.empty(extra_meta_sharded_sd[k][0])
                model.get_initializer(k)(full_sd[k])
                full_sd[k] = full_sd[k].to(device0)
                print_rank_0(
                    f"random init k={k}, {extra_meta_sharded_sd[k]}\n, "
                    f"meta_sharded_sd={meta_sharded_sd[k]} \nfull={full_sd[k]}")

        # Handle shape mismatches
        for k, mismatch_info in shape_mismatches.items():
            # Check if handled by allow_random_init_params or skip_load_params
            in_allow_random = allow_random_init_params is not None and k in allow_random_init_params
            in_skip_load = skip_load_params and any(pattern in k for pattern in skip_load_params)
            
            if in_allow_random or in_skip_load:
                model_shape = mismatch_info['model_shape']
                sd_shape = mismatch_info['state_dict_shape']
                
                # Initialize with model's expected shape
                full_sd[k] = torch.empty(model_shape)
                model.get_initializer(k)(full_sd[k])
                full_sd[k] = full_sd[k].to(device0)
                
                reason = "skip_load_params" if in_skip_load else "allow_random_init_params"
                print_rank_0(
                    f"Shape mismatch, random init ({reason}): k={k}\n"
                    f"  model_shape: {model_shape}\n"
                    f"  state_dict_shape: {sd_shape}\n"
                    f"  new tensor shape: {full_sd[k].shape}"
                )
            else:
                # Log warning for unhandled mismatches
                print_rank_0(
                    f"WARNING: Shape mismatch (not handled): k={k}\n"
                    f"  model_shape: {mismatch_info['model_shape']}\n"
                    f"  state_dict_shape: {mismatch_info['state_dict_shape']}"
                )

        # Check for unhandled shape mismatches (shape mismatches will cause errors)
        # Exclude params that are in allow_random_init_params or match skip_load_params patterns
        def is_handled(k):
            if allow_random_init_params and k in allow_random_init_params:
                return True
            if skip_load_params and any(pattern in k for pattern in skip_load_params):
                return True
            return False
        
        unhandled_shape_mismatches = {
            k: v for k, v in shape_mismatches.items()
            if not is_handled(k)
        }
        assert len(unhandled_shape_mismatches) == 0, \
            f"Unhandled shape mismatches found. Add these params to allow_random_init_params " \
            f"or skip_load_params to randomly initialize them:\n{format_dict_or_list(unhandled_shape_mismatches)}"

        assert len(meta_sharded_sd) == len(full_sd), \
            f"Sharded State Dict doesn't equal to Full State Dict, " \
            f"{len(meta_sharded_sd) } v.s {len(full_sd)}\n" + \
            f"extra_meta_sharded_sd={format_dict_or_list(extra_meta_sharded_sd)}, " \
            f"extra_full_ds={format_dict_or_list(extra_full_ds)}"

        assert sorted(list(meta_sharded_sd.keys())) == sorted(list(full_sd.keys())), \
            "Keys of Sharded State Dict doesn't equal to Full State Dict"

    for param_name, sharded_meta_param in meta_sharded_sd.items():
        print_rank_0(f"param_name={param_name}\nsharded_meta_param={sharded_meta_param.shape}")
        if dist.get_rank() == 0:
            try:
                full_tensor = full_sd[param_name].detach().cuda().type(sharded_meta_param.dtype)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print_rank_n(f"bad param_name={param_name}\nsharded_meta_param={sharded_meta_param}")
                raise e
        else:
            full_tensor = torch.empty(
                sharded_meta_param.size(),
                device="cuda",
                dtype=sharded_meta_param.dtype,
            )
        
        # Check if it's a DTensor (sharded parameter) or regular Tensor (e.g., buffer)
        if isinstance(sharded_meta_param, DTensor):
            mesh = sharded_meta_param.device_mesh
            dist.broadcast(full_tensor, src=0, group=mesh.get_group(0))
            dist.barrier()
            sharded_tensor = distribute_tensor(
                full_tensor, mesh, sharded_meta_param.placements
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)  # default: requires_grad=True
        else:
            # Regular tensor (e.g., buffers) - just broadcast and store
            dist.broadcast(full_tensor, src=0)
            dist.barrier()
            sharded_sd[param_name] = full_tensor

    model.load_state_dict(sharded_sd, assign=True)


def initialize_model_params(model: "FSDPModule"):
    """
    Initialize model parameters randomly for training from scratch.
    
    This function initializes all parameters in a FSDP-sharded model from meta device
    to CUDA device with proper random initialization. It ensures consistent initialization
    across all ranks by broadcasting from rank 0.
    
    Args:
        model: FSDP-wrapped model with parameters on meta device
        
    Example:
        >>> model = ModelClass(config)  # Model on meta device
        >>> shard_model(model, ...)
        >>> initialize_model_params(model)
        >>> # Now all parameters are initialized on CUDA
    """
    from torch.distributed._tensor import distribute_tensor
    
    print_rank_0("Initializing model parameters from scratch...")
    
    # Get the model's state dict (currently on meta device)
    meta_sd = model.state_dict()
    initialized_sd = {}
    
    # Initialize each parameter
    for param_name, meta_param in meta_sd.items():
        if dist.get_rank() == 0:
            # Initialize on rank 0
            if hasattr(model, 'get_initializer'):
                # Use model-specific initializer if available
                init_fn = model.get_initializer(param_name)
                param_tensor = torch.empty(
                    meta_param.size(), 
                    dtype=meta_param.dtype, 
                    device='cuda'
                )
                with torch.no_grad():
                    init_fn(param_tensor)
            else:
                # Use default initialization
                param_tensor = torch.empty(
                    meta_param.size(), 
                    dtype=meta_param.dtype, 
                    device='cuda'
                )
                with torch.no_grad():
                    if param_tensor.ndim >= 2:
                        # Kaiming initialization for weight matrices
                        nn.init.kaiming_normal_(
                            param_tensor, 
                            a=0, 
                            mode='fan_in', 
                            nonlinearity='relu'
                        )
                    else:
                        # Zero initialization for biases
                        nn.init.zeros_(param_tensor)
        else:
            # Other ranks create empty tensors for broadcast
            param_tensor = torch.empty(
                meta_param.size(), 
                dtype=meta_param.dtype, 
                device='cuda'
            )
        
        # Check if it's a DTensor (sharded parameter) or regular Tensor (e.g., buffer)
        if isinstance(meta_param, DTensor):
            mesh = meta_param.device_mesh
            dist.broadcast(param_tensor, src=0, group=mesh.get_group(0))
            dist.barrier()
            # Distribute tensor according to FSDP sharding
            sharded_tensor = distribute_tensor(
                param_tensor, mesh, meta_param.placements
            )
            initialized_sd[param_name] = nn.Parameter(sharded_tensor)
        else:
            # Regular tensor (e.g., buffers) - just broadcast and store
            dist.broadcast(param_tensor, src=0)
            dist.barrier()
            initialized_sd[param_name] = param_tensor
    
    # Load initialized parameters into model
    model.load_state_dict(initialized_sd, assign=True)
    print_rank_0(f"Model parameters initialized successfully ({len(initialized_sd)} parameters)")


# 这个是单机版本的load_from_full_model_state_dict
def load_from_full_model_state_dict_local(model, full_sd: Dict[str, Any], allow_random_init_params="mlp_AR.pre_norm.weight,mlp_AR.pre_norm.bias,mlp_AR.linear_1.weight,mlp_AR.linear_1.bias,mlp_AR.linear_2.weight,mlp_AR.linear_2.bias"):
    if isinstance(allow_random_init_params, str): allow_random_init_params = allow_random_init_params.split(',')
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    # 
    extra_meta_sharded_sd = set(meta_sharded_sd.keys()) - set((full_sd.keys()))
    extra_full_ds = set(full_sd.keys()) - set((meta_sharded_sd.keys()))
    extra_meta_sharded_sd = {
        k:(v.shape, v.device, v.dtype) for k, v in meta_sharded_sd.items() if k in extra_meta_sharded_sd
    }
    extra_full_ds = {
        k:(v.shape, v.device, v.dtype) for k, v in full_sd.items() if k in extra_full_ds
    }
    print(f"full_sd=\n{format_dict_or_list({k:(v.shape, v.device, v.dtype) for k, v in full_sd.items()})}")
    print(f"meta_sharded_sd=\n{format_dict_or_list({k:(v.shape, v.device, v.dtype) for k, v in meta_sharded_sd.items()})}")
    # 
    device0 = full_sd[list(full_sd)[0]]
    for k in extra_meta_sharded_sd:
        if allow_random_init_params is not None and k in allow_random_init_params:
            # full_sd[k] = meta_sharded_sd[k].clone()
            full_sd[k] = torch.rand(extra_meta_sharded_sd[k][0]) * 0.1 # ) .to(device0)
            if full_sd[k].ndim >= 2:
                nn.init.kaiming_normal_(full_sd[k], a=0, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.zeros_(full_sd[k])  # 最常见
            full_sd[k] = full_sd[k].to(device0)
            # full_sd[k] = meta_sharded_sd[k].clone().to(device0)
            print(f"random init k={k}, {extra_meta_sharded_sd[k]}\n, meta_sharded_sd={meta_sharded_sd[k]} \nfull={full_sd[k]}")
    # 
    assert len(meta_sharded_sd) == len(full_sd), \
        f"Sharded State Dict doesn't equal to Full State Dict, {len(meta_sharded_sd) } v.s {len(full_sd)}" + "\n" + \
        f"extra_meta_sharded_sd={format_dict_or_list(extra_meta_sharded_sd)}, extra_full_ds={format_dict_or_list(extra_full_ds)}"
    assert sorted(list(meta_sharded_sd.keys())) == sorted(list(full_sd.keys())), \
        "Keys of Sharded State Dict doesn't equal to Full State Dict"
    # 
    for param_name, sharded_meta_param in meta_sharded_sd.items():
        full_tensor = full_sd[param_name].detach().cuda().type(sharded_meta_param.dtype)
        sharded_sd[param_name] = nn.Parameter(full_tensor)
    model.load_state_dict(sharded_sd, assign=True)

