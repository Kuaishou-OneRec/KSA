"""
Activation Checkpointing Utilities.

This module provides utilities for applying activation checkpointing (also known
as gradient checkpointing) to PyTorch models. Activation checkpointing trades
compute for memory by recomputing activations during the backward pass instead
of storing them.

This is particularly useful for training large models where memory is a constraint.

Functions:
    set_activation_checkpointing: Apply activation checkpointing to a model

Example:
    >>> from torch.nn import TransformerEncoderLayer
    >>> from muse.training.activations import set_activation_checkpointing
    >>> 
    >>> model = MyTransformerModel()
    >>> # Checkpoint all TransformerEncoderLayer modules
    >>> set_activation_checkpointing(model, auto_wrap_policy={TransformerEncoderLayer})
"""
import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

def set_activation_checkpointing(
    model: nn.Module, auto_wrap_policy, **kwargs
) -> None:
    """
    Apply activation checkpointing to specified modules in a model.
    
    Activation checkpointing reduces memory usage by not storing intermediate
    activations during the forward pass. Instead, they are recomputed during
    the backward pass. This trades increased compute for reduced memory.
    
    The policy determines which modules get checkpointed. Common strategies:
    - Checkpoint transformer layers individually
    - Checkpoint specific layer types (e.g., TransformerBlock, ResNetBlock)
    - Use custom policies for fine-grained control
    
    Args:
        model (nn.Module): Model to apply activation checkpointing to
        auto_wrap_policy (set or callable): Policy for selecting modules to checkpoint.
            - If set: Set of nn.Module types to wrap (e.g., {TransformerBlock})
            - If callable: Function that takes (module, recurse, unwrapped_params)
              and returns whether to wrap that module
        **kwargs: Additional arguments passed to torch.distributed's
            apply_activation_checkpointing (e.g., checkpoint_wrapper_fn)
            
    Note:
        - Works with both regular and FSDP-wrapped models
        - Typical memory savings: 30-50% for transformer models
        - Training slowdown: typically 10-30% depending on model and hardware
        
    Example:
        >>> from torch.nn import TransformerEncoderLayer
        >>> 
        >>> # Checkpoint specific layer types
        >>> model = MyModel()
        >>> set_activation_checkpointing(
        ...     model,
        ...     auto_wrap_policy={TransformerEncoderLayer}
        ... )
        >>> 
        >>> # Custom policy
        >>> def my_policy(module, recurse, unwrapped_params):
        ...     if isinstance(module, MyLargeLayer):
        ...         return True
        ...     return recurse
        >>> 
        >>> set_activation_checkpointing(model, auto_wrap_policy=my_policy)
    """
    if isinstance(auto_wrap_policy, set):
        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy)
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)
