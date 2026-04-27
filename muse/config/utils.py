"""
Configuration Utility Functions.

This module provides utilities for loading, parsing, and managing configurations
from various sources including JSON files, YAML files, and command-line arguments.

The utilities help bridge between different configuration sources and the structured
Pydantic configuration classes, enabling flexible configuration management in
training scripts.

Functions:
    load_config_from_file: Load configuration from JSON/YAML file
    load_config_from_args: Convert argparse Namespace to config dict
    get_config: Instantiate appropriate config class from dict
    parse_args_and_config: Parse command-line args and load configs

Example:
    >>> from muse.config.utils import load_config_from_file, get_config
    >>> 
    >>> # Load from file
    >>> config_dict = load_config_from_file("config.json")
    >>> 
    >>> # Get typed config object
    >>> model_config = get_config(config_dict["model"])
    >>> # Returns appropriate ModelConfig subclass (e.g., Qwen3Config)
    >>> 
    >>> # From command line
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> # ... add arguments ...
    >>> args, config = parse_args_and_config(parser)
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration utility functions."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import yaml

from muse.config.base import BaseConfig
from muse.config.model_config import ModelConfig, Qwen3Config
from muse.config.dataset_config import DatasetConfig
from muse.config.training_config import (
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    CheckpointConfig,
    LoggingConfig,
)


def load_config_from_file(path: str) -> Dict[str, Any]:
    """Load configuration from a JSON or YAML file.
    
    Args:
        path: Path to configuration file (.json or .yaml/.yml)
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        ValueError: If file format is not supported
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.suffix == '.json':
            return json.load(f)
        elif file_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                "Supported formats: .json, .yaml, .yml"
            )


def load_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse Namespace to configuration dictionary.
    
    This function maps command-line arguments to the structured
    configuration format expected by the config classes.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary containing configuration values
    """
    config = {
        "model": {},
        "dataset": {},
        "training": {
            "optimizer": {},
            "scheduler": {},
            "checkpoint": {},
            "logging": {},
        }
    }
    
    # Map model arguments
    model_attrs = [
        'model_class', 'model_dir', 'use_flash_attention_2',
        'enable_gradient_checkpointing', 'compile', 'fp32_weight',
        'fp32_reduce', 'reshard_after_forward', 'prefetch_parameters',
        'model_processor', 'allow_random_init_params'
    ]
    for attr in model_attrs:
        if hasattr(args, attr):
            config["model"][attr] = getattr(args, attr)
    
    # Map dataset arguments
    dataset_attrs = [
        'dataset_config', 'dataset', 'data_format',
        'max_length',
    ]
    for attr in dataset_attrs:
        if hasattr(args, attr):
            config["dataset"][attr] = getattr(args, attr)
    
    # Map optimizer arguments
    optimizer_attrs = [
        'learning_rate',
        'weight_decay', 'beta1', 'beta2'
    ]
    for attr in optimizer_attrs:
        if hasattr(args, attr):
            config["training"]["optimizer"][attr] = getattr(args, attr)
    
    # Map scheduler arguments
    scheduler_attrs = [
        'lr_scheduler_type', 'num_warmup_steps', 'num_decay_steps',
        'num_training_steps', 'num_epochs', 'min_lr'
    ]
    for attr in scheduler_attrs:
        if hasattr(args, attr):
            config["training"]["scheduler"][attr] = getattr(args, attr)
    
    # Map checkpoint arguments
    checkpoint_attrs = [
        'output_dir', 'save_checkpoint_per_step', 'save_checkpoint_every_epoch',
        'resume_from', 'resume_from_tag', 'resume_dataloader',
        'load_weights_only', 'auto_resume_local_latest',
        'merge_checkpoint', 'merge_checkpoint_dtype',
        'merge_checkpoint_output_file'
    ]
    for attr in checkpoint_attrs:
        if hasattr(args, attr):
            config["training"]["checkpoint"][attr] = getattr(args, attr)
    
    # Map logging arguments
    logging_attrs = [
        'logging_per_step', 'monitor_datasource_loss',
        'monitor_datasource_cnt', 'monitor_image_tokens',
        'comment', 'commit_id', 'heartbeat_monitor', 'enable_profile'
    ]
    for attr in logging_attrs:
        if hasattr(args, attr):
            config["training"]["logging"][attr] = getattr(args, attr)
    
    # Map training arguments
    training_attrs = [
        'gradient_accumulation_steps', 'clip_range',
        'freeze_llm',
        'context_parallel_size', 'seed',
    ]
    for attr in training_attrs:
        if hasattr(args, attr):
            config["training"][attr] = getattr(args, attr)
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries recursively.
    
    Values in override_config will take precedence over base_config.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def create_config_from_args_and_file(
    args: argparse.Namespace,
    config_file: Optional[str] = None,
    model_config_class: type = ModelConfig,
) -> Tuple[Union[ModelConfig, Qwen3Config], DatasetConfig, TrainingConfig]:
    """Create configuration objects from both file and command-line args.
    
    Priority: command-line args > config file
    
    Args:
        args: Parsed command-line arguments
        config_file: Optional path to configuration file
        model_config_class: Model config class to use (default: ModelConfig)
        
    Returns:
        Tuple of (model_config, dataset_config, training_config)
    """
    # Start with empty config
    config = {
        "model": {},
        "dataset": {},
        "training": {
            "optimizer": {},
            "scheduler": {},
            "checkpoint": {},
            "logging": {},
        }
    }
    
    # Load from file if provided
    if config_file:
        file_config = load_config_from_file(config_file)
        config = merge_configs(config, file_config)
    
    # Override with command-line arguments
    args_config = load_config_from_args(args)
    config = merge_configs(config, args_config)
    
    # Create config objects
    model_config = model_config_class(**config["model"])
    dataset_config = DatasetConfig(**config["dataset"])
    training_config = TrainingConfig(**config["training"])
    
    return model_config, dataset_config, training_config


def save_config_to_checkpoint(
    config: BaseConfig,
    checkpoint_dir: str,
    tag: str = "config",
) -> None:
    """Save configuration to checkpoint directory.
    
    Args:
        config: Configuration object to save
        checkpoint_dir: Checkpoint directory path
        tag: Tag/name for the config file (without extension)
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    config_file = checkpoint_path / f"{tag}.json"
    config.save(str(config_file))


def save_all_configs_to_checkpoint(
    model_config: Union[ModelConfig, Qwen3Config],
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    checkpoint_dir: str,
) -> None:
    """Save all configuration objects to checkpoint directory.
    
    Args:
        model_config: Model configuration
        dataset_config: Dataset configuration
        training_config: Training configuration
        checkpoint_dir: Checkpoint directory path
    """
    save_config_to_checkpoint(model_config, checkpoint_dir, "model_config")
    save_config_to_checkpoint(dataset_config, checkpoint_dir, "dataset_config")
    save_config_to_checkpoint(training_config, checkpoint_dir, "training_config")


def load_configs_from_checkpoint(
    checkpoint_dir: str,
    model_config_class: type = ModelConfig,
) -> Tuple[Union[ModelConfig, Qwen3Config], DatasetConfig, TrainingConfig]:
    """Load all configuration objects from checkpoint directory.
    
    Args:
        checkpoint_dir: Checkpoint directory path
        model_config_class: Model config class to use (default: ModelConfig)
        
    Returns:
        Tuple of (model_config, dataset_config, training_config)
    """
    checkpoint_path = Path(checkpoint_dir)
    
    model_config = model_config_class.load(str(checkpoint_path / "model_config.json"))
    dataset_config = DatasetConfig.load(str(checkpoint_path / "dataset_config.json"))
    training_config = TrainingConfig.load(str(checkpoint_path / "training_config.json"))
    
    return model_config, dataset_config, training_config

