# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration module for Muse framework.

This module provides Pydantic-based configuration classes for models,
datasets, and training. Supports loading from JSON/YAML files and
command-line arguments with full type validation.

Example usage:
    >>> from muse.config import Qwen3Config, DatasetConfig, TrainingConfig
    >>> from muse.config import create_config_from_args_and_file
    >>>
    >>> # Create configs from args and optional config file
    >>> model_config, dataset_config, training_config = create_config_from_args_and_file(
    ...     args=args,
    ...     config_file="configs/train.json"
    ... )
"""

from muse.config.base import BaseConfig, get_config, load_config
from muse.config.model_config import (
    ModelConfig,
    Qwen3Config,
    Qwen3SummaryAttentionConfig,
)

from muse.config.dataset_config import DatasetConfig
from muse.config.training_config import (
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    CheckpointConfig,
    LoggingConfig,
)
from muse.config.utils import (
    load_config_from_file,
    load_config_from_args,
    merge_configs,
    create_config_from_args_and_file,
    save_config_to_checkpoint,
    save_all_configs_to_checkpoint,
    load_configs_from_checkpoint,
)

__all__ = [
    # Base config
    "BaseConfig",

    # Model configs
    "ModelConfig",
    "Qwen3Config",
    "Qwen3SummaryAttentionConfig",

    # Dataset config
    "DatasetConfig",

    # Training configs
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "CheckpointConfig",
    "LoggingConfig",

    # Utility functions
    "load_config_from_file",
    "load_config_from_args",
    "merge_configs",
    "create_config_from_args_and_file",
    "save_config_to_checkpoint",
    "save_all_configs_to_checkpoint",
    "load_configs_from_checkpoint",
    "get_config",
    "load_config",
]
