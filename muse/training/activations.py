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
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


_ACTIVATION_CHECKPOINT_RECOMPUTING = ContextVar(
    "activation_checkpoint_recomputing", default=False
)
_ACTIVATION_CHECKPOINT_ACTIVE = ContextVar(
    "activation_checkpoint_active", default=False
)


@contextmanager
def _activation_checkpoint_phase(recomputing: bool):
    active_token = _ACTIVATION_CHECKPOINT_ACTIVE.set(True)
    recomputing_token = _ACTIVATION_CHECKPOINT_RECOMPUTING.set(recomputing)
    try:
        yield
    finally:
        _ACTIVATION_CHECKPOINT_RECOMPUTING.reset(recomputing_token)
        _ACTIVATION_CHECKPOINT_ACTIVE.reset(active_token)


def _activation_checkpoint_context_fn():
    """Mark normal forward and non-reentrant checkpoint recomputation."""
    return (
        _activation_checkpoint_phase(False),
        _activation_checkpoint_phase(True),
    )


def is_activation_checkpoint_recomputing() -> bool:
    """Whether execution is inside a checkpoint backward recomputation."""
    return _ACTIVATION_CHECKPOINT_RECOMPUTING.get()


def is_activation_checkpoint_active() -> bool:
    """Whether execution is inside either phase of a checkpointed module.

    Unlike :func:`is_activation_checkpoint_recomputing`, this is also true in
    the original forward.  Modules can use the distinction to retain small,
    detached auxiliary results that are cheaper to reuse than to rebuild in
    the backward recomputation.  Ordinary, non-checkpointed forwards remain
    false so such ephemeral caches cannot leak between training iterations.
    """
    return _ACTIVATION_CHECKPOINT_ACTIVE.get()


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
    # The marker is inert for ordinary modules. S-KSA uses it to reuse compact,
    # detached route/teacher results from the original forward during backward
    # recomputation; rebuilding those no-grad decisions cannot affect gradients.
    kwargs.setdefault(
        "checkpoint_wrapper_fn",
        partial(
            checkpoint_wrapper,
            context_fn=_activation_checkpoint_context_fn,
        ),
    )
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)
