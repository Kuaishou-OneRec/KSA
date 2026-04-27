"""
Training Configuration Classes.

This module defines configuration classes for training hyperparameters, including
optimizer settings, learning rate scheduling, checkpointing, and logging. All
configurations use Pydantic for validation and type safety.

The module provides hierarchical configuration:
- OptimizerConfig: AdamW optimizer parameters
- SchedulerConfig: Learning rate scheduling
- CheckpointConfig: Checkpoint saving and loading
- LoggingConfig: Logging and monitoring
- TrainingConfig: Complete training configuration

Classes:
    OptimizerConfig: Optimizer hyperparameters
    SchedulerConfig: Learning rate scheduler settings
    CheckpointConfig: Checkpoint management configuration
    LoggingConfig: Logging and monitoring configuration
    TrainingConfig: Complete training configuration

Example:
    >>> from muse.config.training_config import TrainingConfig
    >>> 
    >>> # Create training configuration
    >>> config = TrainingConfig(
    ...     optimizer=OptimizerConfig(
    ...         learning_rate=2e-4,
    ...         weight_decay=0.1,
    ...         beta1=0.9,
    ...         beta2=0.95
    ...     ),
    ...     scheduler=SchedulerConfig(
    ...         lr_scheduler_type="cosine_with_min_lr",
    ...         num_warmup_steps=1000,
    ...         num_training_steps=10000,
    ...         min_lr=1e-6
    ...     ),
    ...     checkpoint=CheckpointConfig(
    ...         output_dir="./checkpoints",
    ...         save_checkpoint_per_step=1000
    ...     )
    ... )
    >>> 
    >>> # Save configuration
    >>> config.save("training_config.json")
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training configuration classes."""

from typing import Optional, Literal
from pydantic import Field, field_validator

from muse.config.base import BaseConfig


class OptimizerConfig(BaseConfig):
    """Optimizer configuration."""
    
    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        description="Peak learning rate for optimizer"
    )
    weight_decay: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight decay for AdamW optimizer"
    )
    beta1: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Beta1 for AdamW optimizer"
    )
    beta2: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Beta2 for AdamW optimizer"
    )


class SchedulerConfig(BaseConfig):
    """Learning rate scheduler configuration."""
    
    lr_scheduler_type: Literal[
        "cosine_with_min_lr",
        "cosine",
        "linear",
        "constant",
        "constant_with_warmup"
    ] = Field(
        default="cosine_with_min_lr",
        description="Type of learning rate scheduler"
    )
    num_warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of warmup steps"
    )
    num_decay_steps: int = Field(
        default=1000,
        ge=0,
        description="Number of decay steps"
    )
    num_training_steps: int = Field(
        default=1000,
        ge=1,
        description="Total number of training steps"
    )
    num_epochs: int = Field(
        default=1,
        ge=1,
        description="Number of training epochs"
    )
    min_lr: float = Field(
        default=1e-6,
        ge=0.0,
        description="Minimum learning rate after decay"
    )


class CheckpointConfig(BaseConfig):
    """Checkpoint configuration."""
    
    output_dir: str = Field(
        description="Directory to save checkpoints and logs"
    )
    save_checkpoint_per_step: int = Field(
        default=1000,
        ge=1,
        description="Save checkpoint every N steps"
    )
    save_checkpoint_every_epoch: bool = Field(
        default=False,
        description="Save checkpoint at the end of every epoch"
    )
    resume_from: Optional[str] = Field(
        default=None,
        description="Checkpoint directory to resume from"
    )
    resume_from_tag: Optional[str] = Field(
        default=None,
        description="Checkpoint tag to resume from"
    )
    resume_dataloader: bool = Field(
        default=False,
        description="Whether to resume dataloader state"
    )
    load_weights_only: bool = Field(
        default=False,
        description="Only load model weights, skip optimizer and scheduler"
    )
    auto_resume_local_latest: bool = Field(
        default=False,
        description="Auto resume from latest checkpoint in output_dir"
    )
    merge_checkpoint: bool = Field(
        default=False,
        description="Merge checkpoint files into a single file"
    )
    merge_checkpoint_dtype: Literal["fp32", "fp16", "bf16"] = Field(
        default="fp16",
        description="Data type for merged checkpoint"
    )
    merge_checkpoint_output_file: str = Field(
        default="pytorch_model.bin",
        description="Output filename for merged checkpoint"
    )


class LoggingConfig(BaseConfig):
    """Logging configuration."""
    
    logging_per_step: int = Field(
        default=100,
        ge=1,
        description="Log metrics every N steps"
    )
    monitor_datasource_loss: bool = Field(
        default=False,
        description="Monitor loss per data source"
    )
    monitor_datasource_cnt: bool = Field(
        default=False,
        description="Monitor sample count per data source"
    )
    monitor_image_tokens: bool = Field(
        default=False,
        description="Monitor image token statistics"
    )
    comment: Optional[str] = Field(
        default=None,
        description="Comment for this training run"
    )
    commit_id: Optional[str] = Field(
        default=None,
        description="Git commit ID for reproducibility"
    )
    heartbeat_monitor: bool = Field(
        default=False,
        description="Enable heartbeat monitoring"
    )
    enable_profile: bool = Field(
        default=False,
        description="Enable PyTorch profiler"
    )


class TrainingConfig(BaseConfig):
    """Complete training configuration.
    
    This aggregates all training-related configurations including
    optimizer, scheduler, checkpointing, and logging settings.
    """
    
    # Sub-configurations
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer configuration"
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler configuration"
    )
    checkpoint: CheckpointConfig = Field(
        description="Checkpoint configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    
    # Training dynamics
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Number of gradient accumulation steps"
    )
    clip_range: float = Field(
        default=1.0,
        gt=0.0,
        description="Gradient clipping range"
    )
    
    # Model freezing options
    freeze_llm: bool = Field(
        default=False,
        description="Freeze all LLM parameters"
    )

    # Parallelism
    context_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Context parallelism size"
    )
    
    # Reproducibility
    seed: int = Field(
        default=123,
        description="Random seed for reproducibility"
    )
    
    @field_validator("scheduler")
    @classmethod
    def validate_scheduler_steps(cls, v, info):
        """Validate scheduler step counts."""
        if v.num_warmup_steps >= v.num_training_steps:
            raise ValueError(
                f"num_warmup_steps ({v.num_warmup_steps}) must be less than "
                f"num_training_steps ({v.num_training_steps})"
            )
        return v

