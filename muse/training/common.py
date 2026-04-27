"""
Training Utilities and Helper Functions.

This module provides essential utilities for PyTorch training workflows, including:

- Default dtype management with context managers
- Gradient clipping and norm computation
- FSDP Zero-2 gradient norm calculation
- Gradient norm logging for debugging
- Step scheduling for training loops

The utilities support both single-machine and distributed training environments,
with special handling for FSDP (Fully Sharded Data Parallel) models.

Functions:
    set_default_dtype: Context manager for temporary dtype changes
    clip_grad_by_value: Clip gradients by value
    get_total_norm: Compute total norm of tensors
    get_global_grad_norm: Compute global gradient norm for a model
    compute_fsdp_zero2_grad_norm: FSDP Zero-2 specific gradient norm
    define_metrics: Create and configure metrics for training

Classes:
    GradNormLogger: Logger for tracking gradient norms during training
    StepScheduler: Scheduler for managing training loop timing and checkpointing

Example:
    >>> with set_default_dtype(torch.bfloat16):
    ...     model = MyModel()  # Created with bfloat16 weights
    >>> 
    >>> grad_norm = get_global_grad_norm(model)
    >>> print(f"Gradient norm: {grad_norm:.4f}")
    >>> 
    >>> # Use StepScheduler for training loop
    >>> scheduler = StepScheduler(args)
    >>> for batch in dataloader:
    ...     loss = model(batch)
    ...     loss.backward()
    ...     scheduler.step()
    ...     if scheduler.should_logging():
    ...         log_metrics()
"""
from re import M
from typing import Dict, Any, Union, Optional, Generator, \
    Iterable, Tuple, List

import torch
from torch import Tensor
import math
import os
import time
import shutil
from pathlib import Path
import torch.distributed as dist

import contextlib

from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from muse.utils.metrics import Metrics, Logger

@contextlib.contextmanager
def set_default_dtype(dtype: Union[str, torch.dtype]) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (Union[str, torch.dtype]): The desired default dtype inside the context manager.
            Can be either a string ("bfloat16", "float16", "float32") or a torch.dtype.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16
        
        >>> with set_default_dtype("bfloat16"):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16

    """
    # Convert string to torch.dtype if needed
    if isinstance(dtype, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(
                f"Invalid dtype string: {dtype}. "
                f"Supported values: {list(dtype_map.keys())}"
            )
        dtype = dtype_map[dtype]
    
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Get torch.dtype from string.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]

def clip_grad_by_value(model: torch.nn.Module, clip_range: Optional[float] = None):
    """
    Clip gradients by value for all model parameters.
    
    Wraps torch.nn.utils.clip_grad_value_ for convenience. Each gradient tensor
    will be clipped to the range [-clip_range, +clip_range].
    
    Args:
        model (torch.nn.Module): Model whose gradients will be clipped
        clip_range (float, optional): Maximum absolute value for gradients.
            If None, no clipping is performed. Defaults to None.
            
    Example:
        >>> model = MyModel()
        >>> loss.backward()
        >>> clip_grad_by_value(model, clip_range=1.0)  # Clip to [-1, 1]
        >>> optimizer.step()
    """
    if clip_range is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_range)

def get_total_norm(
    tensors: Union[torch.Tensor, Iterable[torch.Tensor]],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: List[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm

def get_global_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    """
    Compute the global L2 norm of all gradients in a model.
    
    Collects gradients from all parameters that have them and computes
    the total L2 norm using get_total_norm().
    
    Args:
        model (torch.nn.Module): Model to compute gradient norm for
        
    Returns:
        torch.Tensor: Scalar tensor containing the L2 norm of all gradients
        
    Example:
        >>> model = MyModel()
        >>> loss.backward()
        >>> grad_norm = get_global_grad_norm(model)
        >>> print(f"Gradient norm: {grad_norm.item():.4f}")
    """
    grads = [
        param.grad.data for param in model.parameters() \
            if param.grad is not None]
    return get_total_norm(grads, norm_type=2.0)


def freeze_params_by_pattern(model: torch.nn.Module, patterns: List[str]) -> int:
    """Freeze parameters based on pattern matching (contains).
    
    Supports two modes:
    1. Normal mode: freeze params containing any pattern
       E.g., ['y_embedder', 'cross_attn'] freezes params containing these substrings
    2. Inverse mode (^ prefix): freeze all EXCEPT params matching patterns
       E.g., ['^y_embedder', '^cross_attn'] freezes everything except params containing these
    
    Args:
        model: The model to freeze parameters in
        patterns: List of patterns to match in parameter names (use ^ for inverse selection)
        
    Returns:
        Number of parameters frozen
    """
    # Check if inverse mode (all patterns start with ^)
    inverse_mode = all(p.startswith('^') for p in patterns)
    if inverse_mode:
        # Remove ^ prefix for matching
        patterns = [p[1:] for p in patterns]
    
    frozen_count = 0
    for name, param in model.named_parameters():
        matches_pattern = any(pattern in name for pattern in patterns)

        # In inverse mode: freeze if NOT matching; normal mode: freeze if matching
        should_freeze = not matches_pattern if inverse_mode else matches_pattern
        
        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
    
    return frozen_count


def compute_fsdp_zero2_grad_norm(model: torch.nn.Module, 
                                 ignore_unused_parameters: bool = True) -> float:
    """
    Compute gradient L2 norm for FSDP Zero-2 mode.
    
    In FSDP Zero-2, gradients are stored as DTensor (distributed tensors). This
    function correctly handles the sharded gradients by:
    1. Computing local norm for each rank's gradient shard
    2. Aggregating across all ranks using all_reduce
    3. Computing the global L2 norm
    
    Args:
        model (torch.nn.Module): FSDP-wrapped model
        ignore_unused_parameters (bool): Whether to skip parameters without gradients.
            If False, raises ValueError if any parameter lacks gradients. Defaults to True.
            
    Returns:
        float: Global L2 norm of all gradients
        
    Raises:
        ValueError: If ignore_unused_parameters=False and a parameter has no gradient
        
    Example:
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> model = FSDP(MyModel())
        >>> loss.backward()
        >>> grad_norm = compute_fsdp_zero2_grad_norm(model)
        >>> print(f"FSDP gradient norm: {grad_norm:.4f}")
    """
    # 初始化全局平方和
    total_sq = torch.tensor(0.0, device=next(model.parameters()).device)
    
    # 遍历所有参数，累加本地梯度分片的平方和
    for param in model.parameters():
        if param.grad is None:
            if not ignore_unused_parameters:
                raise ValueError(f"参数 {param} 没有梯度，请检查是否被正确使用")
            continue
        
        # 从DTensor中获取本地分片（FSDP的梯度是DTensor类型）
        # 注意：DTensor的.local_tensor()方法返回当前进程持有的分片
        local_grad = param.grad.to_local()
        
        # 计算本地分片的平方和，并累加到total_sq
        total_sq += torch.sum(local_grad **2)
    
    # 跨进程聚合所有分片的平方和（使用all_reduce求和）
    # 注意：需要使用FSDP的通信组，或默认的全局通信组
    dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    
    # 计算全局L2范数（平方和的平方根）
    grad_norm = torch.sqrt(total_sq).item()
    
    return grad_norm

class GradNormLogger:
    """
    Logger for tracking gradient norms during training.
    
    Records per-parameter gradient norms to files for debugging gradient flow
    and identifying vanishing/exploding gradients. Each rank logs to its own
    file to avoid conflicts in distributed training.
    
    The logger creates a directory and writes CSV-formatted logs with columns:
    - Step: Training step number
    - Parameter Name: Full parameter name (e.g., "model.layer1.weight")
    - Gradient Norm: L2 norm of the gradient
    - shape: Parameter shape
    - local_shape: Local shard shape (for FSDP)
    
    Args:
        log_dir (str): Directory to store log files. Will be cleared if it exists.
            Defaults to "grad_norm_logs".
            
    Attributes:
        step (int): Current training step counter
        log_dir (str): Path to log directory
        rank (int): Current process rank
        
    Example:
        >>> logger = GradNormLogger(log_dir="./gradient_logs")
        >>> for step, batch in enumerate(dataloader):
        ...     loss = model(batch)
        ...     loss.backward()
        ...     logger(model, step=step)
        ...     optimizer.step()
    """
    
    def __init__(self, log_dir: str = "grad_norm_logs"):
        """
        Initialize gradient norm logger.
        
        Warning: This will clear the log directory if it already exists on rank 0.
        
        Args:
            log_dir (str): Directory path for storing log files. Defaults to "grad_norm_logs".
        """
        self.step = 0  # 记录当前是第几个step
        self.log_dir = log_dir
        self.rank = self._get_rank()  # 获取当前进程的rank
        
        # 若文件夹已存在则删除并重建（清空效果），否则直接创建
        if self._get_rank() == 0:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)  # 递归删除文件夹及内部所有内容
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)  # 重新创建文件夹
        
    def _get_rank(self) -> int:
        """
        Get current process rank, handling non-distributed environments.
        
        Safely retrieves the rank in distributed mode, or returns 0 if:
        - torch.distributed is not available
        - distributed is not initialized
        - any errors occur during rank retrieval
        
        Returns:
            int: Process rank (0-indexed), or 0 if not in distributed mode
        """
        try:
            if not hasattr(torch, 'distributed'):
                return 0
                
            if torch.distributed.is_initialized():
                return torch.distributed.get_rank()
            else:
                return 0
        except (RuntimeError, ImportError):
            return 0
        
    def __call__(self, model: torch.nn.Module, step: Optional[int] = None):
        """
        Log gradient norms for all parameters at the current step.
        
        Computes and writes the L2 norm of each parameter's gradient to the log file.
        Parameters without gradients are recorded as NaN.
        
        Args:
            model (torch.nn.Module): Model whose gradients will be logged
            step (int, optional): Training step number. If None, uses internal
                counter and auto-increments. Defaults to None.
                
        Note:
            - Logs are appended to rank-specific files: grad_norm_rank_{rank}.txt
            - Header row is written on first call to each file
            - For FSDP models, logs both global and local shapes
        """
        if step is None:
            self.step += 1
            step = self.step
        
        # 为当前rank创建独立的日志文件
        log_file = os.path.join(self.log_dir, f"grad_norm_rank_{self.rank}.txt")
        
        # 第一次写入时添加表头
        write_header = not os.path.exists(log_file)
        
        with open(log_file, 'a') as f:
            if write_header:
                f.write("Step,Parameter Name,Gradient Norm,shape,local_shape\n")
                
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = math.sqrt(torch.sum(param.grad**2).item())
                    shape = param.data.shape
                    try:
                        local_shape = param.grad.to_local().shape
                    except:
                        local_shape = None
                    f.write(f"{step},{name},{grad_norm:.6f}. {shape}/{local_shape}\n")
                    if 'head' in name:
                        f.write('\n' + str(param) + '\n')
                else:
                    f.write(f"{step},{name},NaN,None,None\n")


def initialize_metrics(acc_steps: int, logging_per_step: int, loggers: List[Logger], dp_size: int = 1):
    """
    Initialize metrics for training, define some basic metrics, you can add more metrics later.

    Args:
        acc_steps: Number of gradient accumulation steps
        logging_per_step: Frequency of logging (every N global steps)
        loggers: List of loggers to use
        dp_size: Data parallel world size (excludes CP). Used for throughput calculation.
    """
    metrics = Metrics()

    # Micro-step metrics
    metrics.new("loss", dtype="float", reduce="mean")
    metrics.new("grad_norm", dtype="float", reduce="mean")
    metrics.new("learning_rate", dtype="float")
    metrics.new("step_time", dtype="timestamp", initial_value=lambda: time.time())
    metrics.new("tokens", dtype="int", initial_value=0)
    metrics.new("samples", dtype="int", initial_value=0)

    # 增加哨兵节点，初始化部分序列
    metrics.initialize()

    # rank-0 local cumsum, scale by dp_size for global unique counts
    total_tokens = metrics.tokens.cumsum() * dp_size
    total_samples = metrics.samples.cumsum() * dp_size

    # Global-step metrics, skip the first step
    avg_loss = metrics.loss.avg(window=acc_steps)[::acc_steps][1:]

    avg_grad_norm = metrics.grad_norm[::acc_steps][1:]

    learning_rate = metrics.learning_rate[::acc_steps][1:]

    # Fixed: Skip the first None from diff(), same pattern as other metrics
    seconds_per_step = metrics.step_time[::acc_steps].diff()[1:]

    # Per-GPU throughput: rank-0 local tokens/time (no reduce, no division)
    tokens_per_sec_per_gpu = (metrics.tokens.cumsum().diff() / metrics.step_time.diff())[::acc_steps][1:]
    samples_per_sec_per_gpu = (metrics.samples.cumsum().diff() / metrics.step_time.diff())[::acc_steps][1:]

    # System throughput = per-GPU × dp_size (unique data only, excludes CP duplicates)
    tokens_per_day = tokens_per_sec_per_gpu * 86400 * dp_size

    for logger in loggers:
        metrics.add_logger(logger)

    # Logging metrics, avg over the last logging_per_step steps
    metrics.logger.track(
        avg_loss.avg(window=logging_per_step)[::logging_per_step], 
        name="loss", group="training")
    metrics.logger.track(
        avg_grad_norm.avg(window=logging_per_step)[::logging_per_step], 
        name="grad_norm", group="training")
    metrics.logger.track(
        learning_rate.avg(window=logging_per_step)[::logging_per_step], 
        name="learning_rate", group="training")
    # Skip first None value from seconds_per_step before final logging slice
    metrics.logger.track(
        seconds_per_step.avg(window=logging_per_step)[::logging_per_step], 
        name="seconds_per_step", group="perf")
    metrics.logger.track(
        total_tokens[::acc_steps][::logging_per_step], 
        name="total_tokens", group="perf")
    metrics.logger.track(
        total_samples[::acc_steps][::logging_per_step], 
        name="total_samples", group="perf")
    metrics.logger.track(
        tokens_per_sec_per_gpu.avg(window=logging_per_step)[::logging_per_step], 
        name="tokens_per_sec_per_gpu", group="perf")
    metrics.logger.track(
        samples_per_sec_per_gpu.avg(window=logging_per_step)[::logging_per_step], 
        name="samples_per_sec_per_gpu", group="perf")
    metrics.logger.track(
        tokens_per_day.avg(window=logging_per_step)[::logging_per_step], 
        name="tokens_per_day", group="perf")

    return metrics


class StepScheduler:
    """
    Step scheduler for managing training loop scheduling logic.
    
    This class manages micro-step and global-step counters to determine when to:
    - Log training metrics
    - Save model checkpoints  
    - Update optimizer (at gradient accumulation boundaries)
    
    The scheduler maintains internal state and automatically tracks both micro-steps
    (number of forward passes) and global-steps (number of optimizer updates).
    
    Attributes:
        gradient_accumulation_steps (int): Number of micro-steps to accumulate gradients
        logging_per_step (int): Frequency of logging (every N global steps)
        save_checkpoint_per_step (int): Frequency of checkpoint saving (every N global steps)
        
    Example:
        >>> # Initialize from args
        >>> scheduler = StepScheduler(args)
        >>> 
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     
        ...     # Advance the scheduler
        ...     scheduler.step()
        ...     
        ...     # Update optimizer at gradient accumulation boundaries
        ...     if scheduler.is_gradient_accumulation_boundary():
        ...         optimizer.step()
        ...         optimizer.zero_grad()
        ...     
        ...     # Log metrics when appropriate
        ...     if scheduler.should_logging():
        ...         print(f"Step {scheduler.global_step}: loss={loss.item()}")
        ...     
        ...     # Save checkpoints when appropriate
        ...     if scheduler.should_save_checkpoint():
        ...         save_checkpoint(model, f"step_{scheduler.global_step}")
    """
    
    def __init__(self, args):
        """
        Initialize the step scheduler from command-line arguments.
        
        Args:
            args: Argument namespace containing:
                - gradient_accumulation_steps (int): Number of micro-steps per optimizer update
                - logging_per_step (int): Log metrics every N global steps
                - save_checkpoint_per_step (int): Save checkpoint every N global steps
                
        Example:
            >>> parser = argparse.ArgumentParser()
            >>> parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
            >>> parser.add_argument("--logging-per-step", type=int, default=100)
            >>> parser.add_argument("--save-checkpoint-per-step", type=int, default=1000)
            >>> args = parser.parse_args()
            >>> scheduler = StepScheduler(args)
        """
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.logging_per_step = args.logging_per_step
        self.save_checkpoint_per_step = args.save_checkpoint_per_step
        
        # Internal state tracking
        self._micro_step = 0
        self._global_step = getattr(args, 'global_step', 0)
    
    def step(self):
        """
        Advance the scheduler by one micro-step.
        
        This method should be called after each forward/backward pass. It increments
        the micro-step counter and automatically increments the global-step counter
        when reaching a gradient accumulation boundary.
        
        Example:
            >>> scheduler = StepScheduler(args)
            >>> for batch in dataloader:
            ...     loss = model(batch)
            ...     loss.backward()
            ...     scheduler.step()  # Call after each forward/backward
        """
        self._micro_step += 1
        if self.is_gradient_accumulation_boundary():
            self._global_step += 1
    
    @property
    def micro_step(self) -> int:
        """
        Get the current micro-step count.
        
        Micro-steps represent the total number of forward passes executed,
        including those within gradient accumulation cycles.
        
        Returns:
            int: Current micro-step count (starts at 0)
            
        Example:
            >>> scheduler = StepScheduler(args)
            >>> print(scheduler.micro_step)  # 0
            >>> scheduler.step()
            >>> print(scheduler.micro_step)  # 1
        """
        return self._micro_step
    
    @property
    def global_step(self) -> int:
        """
        Get the current global-step count.
        
        Global-steps represent the number of optimizer updates performed.
        This equals micro_steps // gradient_accumulation_steps.
        
        Returns:
            int: Current global-step count (starts at 0)
            
        Example:
            >>> # With gradient_accumulation_steps=4
            >>> scheduler = StepScheduler(args)
            >>> print(scheduler.global_step)  # 0
            >>> for i in range(4):
            ...     scheduler.step()
            >>> print(scheduler.global_step)  # 1 (updated at boundary)
        """
        return self._global_step

    def state_dict(self) -> Dict[str, int]:
        """Serialize scheduler state needed for checkpoint resume."""
        return {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_per_step": self.logging_per_step,
            "save_checkpoint_per_step": self.save_checkpoint_per_step,
            "micro_step": self._micro_step,
            "global_step": self._global_step,
        }

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """Restore scheduler state from a checkpoint payload."""
        # Only gradient_accumulation_steps must match (affects training correctness).
        # logging_per_step and save_checkpoint_per_step are runtime config and may change.
        loaded_ga = state_dict.get(
            "gradient_accumulation_steps", self.gradient_accumulation_steps)
        if loaded_ga != self.gradient_accumulation_steps:
            raise ValueError(
                f"StepScheduler config mismatch for gradient_accumulation_steps: "
                f"expected {self.gradient_accumulation_steps}, got {loaded_ga}"
            )

        self._micro_step = int(state_dict.get("micro_step", 0))
        self._global_step = int(state_dict.get("global_step", 0))
    
    def is_gradient_accumulation_boundary(self) -> bool:
        """
        Check if current step is at a gradient accumulation boundary.
        
        Returns True when the number of micro-steps is divisible by 
        gradient_accumulation_steps, indicating it's time to update the optimizer.
        
        Returns:
            bool: True if at gradient accumulation boundary, False otherwise
            
        Example:
            >>> # With gradient_accumulation_steps=4
            >>> scheduler = StepScheduler(args)
            >>> for i in range(5):
            ...     scheduler.step()
            ...     if scheduler.is_gradient_accumulation_boundary():
            ...         print(f"Boundary at micro_step {scheduler.micro_step}")
            # Output:
            # Boundary at micro_step 4
        """
        return (
            self._micro_step > 0 and
            self._micro_step % self.gradient_accumulation_steps == 0
        )
    
    def should_logging(self) -> bool:
        """
        Check if metrics should be logged at the current step.
        
        Returns True only when:
        1. At a gradient accumulation boundary (optimizer update step)
        2. Global step is divisible by logging_per_step
        
        This ensures logging happens only after optimizer updates and at
        the specified frequency.
        
        Returns:
            bool: True if should log metrics, False otherwise
            
        Example:
            >>> # With gradient_accumulation_steps=4, logging_per_step=100
            >>> scheduler = StepScheduler(args)
            >>> for i in range(401):
            ...     scheduler.step()
            ...     if scheduler.should_logging():
            ...         print(f"Log at global_step {scheduler.global_step}")
            # Output:
            # Log at global_step 100
        """
        return (
            self._global_step > 0 and
            self.is_gradient_accumulation_boundary() and
            self._global_step % self.logging_per_step == 0
        )
    
    def should_save_checkpoint(self) -> bool:
        """
        Check if checkpoint should be saved at the current step.
        
        Returns True only when:
        1. At a gradient accumulation boundary (optimizer update step)
        2. Global step is divisible by save_checkpoint_per_step
        
        This ensures checkpoints are saved only after optimizer updates and
        at the specified frequency.
        
        Returns:
            bool: True if should save checkpoint, False otherwise
            
        Example:
            >>> # With gradient_accumulation_steps=4, save_checkpoint_per_step=1000
            >>> scheduler = StepScheduler(args)
            >>> for i in range(4001):
            ...     scheduler.step()
            ...     if scheduler.should_save_checkpoint():
            ...         print(f"Save at global_step {scheduler.global_step}")
            # Output:
            # Save at global_step 1000
        """
        return (
            self._global_step > 0 and
            self.is_gradient_accumulation_boundary() and
            self._global_step % self.save_checkpoint_per_step == 0
        )
