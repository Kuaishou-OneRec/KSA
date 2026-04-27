"""
Learning Rate Schedulers for Training.

This module provides learning rate scheduling utilities for PyTorch training,
with a focus on cosine annealing with linear warmup. The schedulers support:

- Linear warmup phase
- Cosine annealing decay
- Configurable minimum learning rate
- Optional stop steps for delayed start
- Multiple decay cycles

The implementation is based on HuggingFace Transformers and adapted for
flexible training workflows.

Functions:
    get_cosine_scheduler: Create cosine annealing with warmup scheduler
    get_cosine_scheduler_v2: Create cosine annealing with warmup scheduler (version 2)
    get_scheduler: Factory function for creating schedulers by name

Example:
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> scheduler = get_cosine_scheduler(
    ...     optimizer,
    ...     num_warmup_steps=1000,
    ...     num_training_steps=10000,
    ...     min_lr_rate=0.1
    ... )
    >>> 
    >>> for epoch in range(num_epochs):
    ...     for batch in dataloader:
    ...         loss = model(batch)
    ...         loss.backward()
    ...         optimizer.step()
    ...         scheduler.step()
"""
# Copy from torchtune
import math
from typing import Optional
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float,
        num_stop_steps: int = 0,
        min_lr_rate: float = 0.0) -> float:
    """
    Compute learning rate multiplier for cosine schedule with warmup.
    
    This is an internal helper function that implements the learning rate
    schedule logic. It returns a multiplier (0.0 to 1.0) that will be applied
    to the base learning rate.
    
    Schedule phases:
    1. Stop phase (0 to num_stop_steps): LR = 0
    2. Warmup phase (num_stop_steps to num_warmup_steps): Linear increase
    3. Cosine phase (num_warmup_steps to num_training_steps): Cosine decay
    4. Post-training (after num_training_steps): min_lr_rate
    
    Args:
        current_step (int): Current training step
        num_warmup_steps (int): Number of warmup steps
        num_training_steps (int): Total number of training steps
        num_cycles (float): Number of cosine cycles (0.5 = half cosine)
        num_stop_steps (int): Number of steps to keep LR at 0. Defaults to 0.
        min_lr_rate (float): Minimum LR as fraction of base LR. Defaults to 0.0.
        
    Returns:
        float: Learning rate multiplier in range [min_lr_rate, 1.0]
    """
        
    if num_stop_steps > 0 and current_step < num_stop_steps:
        return 0.0
    else:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return min_lr_rate
        progress = float(current_step - num_warmup_steps) /\
            float(max(1, num_training_steps - num_warmup_steps))
        factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = factor * (1 - min_lr_rate) + min_lr_rate
        return max(0, factor)

def _get_cosine_schedule_with_warmup_lr_lambda_v2(
        current_step: int, *, num_warmup_steps: int,
        num_decay_steps: int,
        num_training_steps: int,
        num_cycles: float,
        num_stop_steps: int = 0,
        min_lr_rate: float = 0.0) -> float:
    """
    Compute learning rate multiplier for cosine schedule with warmup (version 2).
    
    This is an internal helper function that implements the learning rate
    schedule logic. It returns a multiplier (0.0 to 1.0) that will be applied
    to the base learning rate.
    
    Schedule phases:
    1. Stop phase (0 to num_stop_steps): LR = 0
    2. Warmup phase (num_stop_steps to num_warmup_steps): Linear increase
    3. Cosine phase (num_warmup_steps to num_decay_steps): Cosine decay
    4. Hold phase (num_decay_steps to num_training_steps): Hold at min_lr_rate
    5. Post-training (after num_training_steps): min_lr_rate
    
    Args:
        current_step (int): Current training step
        num_warmup_steps (int): Number of warmup steps
        num_decay_steps (int): Number of steps for the decay phase
        num_training_steps (int): Total number of training steps
        num_cycles (float): Number of cosine cycles (0.5 = half cosine)
        num_stop_steps (int): Number of steps to keep LR at 0. Defaults to 0.
        min_lr_rate (float): Minimum LR as fraction of base LR. Defaults to 0.0.
        
    Returns:
        float: Learning rate multiplier in range [min_lr_rate, 1.0]
    """
        
    if num_stop_steps > 0 and current_step < num_stop_steps:
        return 0.0
    else:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return min_lr_rate
        if current_step > num_decay_steps:
            return min_lr_rate
        progress = float(current_step - num_warmup_steps) /\
            float(max(1, num_decay_steps - num_warmup_steps))
        factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = factor * (1 - min_lr_rate) + min_lr_rate
        return max(0, factor)

def _get_constant_schedule_with_warmup_lr_lambda(
        current_step: int, *,
        num_warmup_steps: int) -> float:
    """
    Compute learning rate multiplier for constant schedule with warmup.
    
    This is an internal helper function that implements the learning rate
    schedule logic. It returns a multiplier (0.0 to 1.0) that will be applied
    to the base learning rate.
    
    Schedule phases:
    1. Warmup phase (0 to num_warmup_steps): Linear increase from 0 to 1
    2. Constant phase (after num_warmup_steps): LR = 1.0
    
    Args:
        current_step (int): Current training step
        num_warmup_steps (int): Number of warmup steps
        
    Returns:
        float: Learning rate multiplier in range [0.0, 1.0]
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return 1.0

def get_cosine_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    num_stop_steps: int = 0,
    last_epoch: int = -1,
    min_lr: float = None,
    min_lr_rate: float = None,
    **kwargs) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over ``num_warmup_steps``, then decreases to 0.0 on a cosine schedule over
    the remaining ``num_training_steps-num_warmup_steps`` (assuming ``num_cycles`` = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError(
            "One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`"
        )

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
        num_stop_steps=num_stop_steps,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_scheduler_v2(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_decay_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    num_stop_steps: int = 0,
    last_epoch: int = -1,
    min_lr: float = None,
    min_lr_rate: float = None,
    **kwargs) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over ``num_warmup_steps``, then decreases to min_lr on a cosine schedule over
    ``num_decay_steps-num_warmup_steps`` steps, and holds at min_lr until ``num_training_steps``
    (assuming ``num_cycles`` = 0.5).

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_decay_steps (int): The number of steps for the decay phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError(
            "One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`"
        )

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda_v2,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=num_decay_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
        num_stop_steps=num_stop_steps,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_wsd_schedule_lr_lambda(
        current_step: int, *, num_warmup_steps: int,
        num_decay_steps: int,
        num_training_steps: int,
        min_lr_rate: float = 0.0) -> float:
    """
    Compute learning rate multiplier for Warmup-Stable-Decay (WSD) schedule.

    Schedule phases:
    1. Warmup (0 to num_warmup_steps): Linear increase from 0 to 1
    2. Stable (num_warmup_steps to num_training_steps - num_decay_steps): Hold at 1.0
    3. Decay (last num_decay_steps): Linear decrease from 1.0 to min_lr_rate

    Args:
        current_step (int): Current training step
        num_warmup_steps (int): Number of warmup steps
        num_decay_steps (int): Duration of decay phase (last N steps)
        num_training_steps (int): Total number of training steps
        min_lr_rate (float): Minimum LR as fraction of base LR. Defaults to 0.0.

    Returns:
        float: Learning rate multiplier in range [min_lr_rate, 1.0]
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step >= num_training_steps:
        return min_lr_rate
    decay_start = num_training_steps - num_decay_steps
    if current_step < decay_start:
        return 1.0
    progress = float(current_step - decay_start) / float(max(1, num_decay_steps))
    return 1.0 - progress * (1.0 - min_lr_rate)


def get_wsd_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_decay_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    min_lr: float = None,
    min_lr_rate: float = None,
    **kwargs) -> LambdaLR:
    """
    Create a Warmup-Stable-Decay (WSD) learning rate schedule.

    Linearly increases LR from 0 to peak over ``num_warmup_steps``, holds at peak,
    then linearly decays to ``min_lr`` over the last ``num_decay_steps`` steps.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_decay_steps: Duration of decay phase (last N steps).
        num_training_steps: Total number of training steps.
        last_epoch: Index of last epoch when resuming. Defaults to -1.
        min_lr: Minimum learning rate (absolute).
        min_lr_rate: Minimum LR as fraction of base LR.

    Returns:
        LambdaLR with WSD schedule.
    """
    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError(
            "One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`"
        )

    lr_lambda = partial(
        _get_wsd_schedule_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=num_decay_steps,
        num_training_steps=num_training_steps,
        min_lr_rate=min_lr_rate,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int = 0,
    last_epoch: int = -1,
    **kwargs) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over ``num_warmup_steps``, then keeps it constant.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase. Defaults to 0.
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """
    lr_lambda = partial(
        _get_constant_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs) -> LambdaLR:
    """
    Factory function to create learning rate schedulers by name.
    
    Currently supports:
    - "cosine": Cosine annealing with warmup
    - "cosine_v2": Cosine annealing with warmup (version 2)
    - "wsd": Warmup-Stable-Decay
    - "constant": Constant learning rate with optional warmup
    
    Args:
        name (str): Name of the scheduler ("cosine", "cosine_v2", "constant")
        optimizer (Optimizer): PyTorch optimizer to schedule
        num_warmup_steps (int, optional): Number of warmup steps
        num_training_steps (int, optional): Total training steps
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        LambdaLR: Configured learning rate scheduler
        
    Raises:
        NotImplementedError: If scheduler name is not supported
        
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = get_scheduler(
        ...     name="cosine",
        ...     optimizer=optimizer,
        ...     num_warmup_steps=500,
        ...     num_training_steps=10000,
        ...     min_lr_rate=0.1
        ... )
    """
    if name == "cosine":
        return get_cosine_scheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    elif name == "cosine_v2":
        return get_cosine_scheduler_v2(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    elif name == "wsd":
        return get_wsd_scheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    elif name == "constant":
        return get_constant_scheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps or 0,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported LR scheduler `{name}`")