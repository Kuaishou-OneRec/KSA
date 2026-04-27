# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base configuration class for all configs."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration class with common functionality.
    
    All configuration classes should inherit from this base class.
    Provides serialization, validation, and merging capabilities.
    """
    
    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields
        validate_assignment=True,  # Validate on assignment
        use_enum_values=True,  # Use enum values
        arbitrary_types_allowed=True,  # Allow arbitrary types
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        result = self.model_dump()
        # Add __class__ field to indicate the config class name
        result["__class__"] = self.__class__.__name__
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize config to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, path: str) -> None:
        """Save config to a JSON file.
        
        Args:
            path: File path to save the config
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    def merge(self, other: Dict[str, Any]) -> "BaseConfig":
        """Merge this config with another config dict.
        
        Values in 'other' will override values in this config.
        
        Args:
            other: Dictionary to merge with
            
        Returns:
            New config instance with merged values
        """
        current_dict = self.to_dict()
        current_dict.update(other)
        return self.__class__(**current_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary containing config values
            Note: __class__ field will be ignored if present
            
        Returns:
            Config instance
        """
        # Remove __class__ field if present (it's metadata, not a config field)
        config_dict = {k: v for k, v in config_dict.items() if k != "__class__"}
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> "BaseConfig":
        """Create config from JSON string.
        
        Args:
            json_str: JSON string containing config values
            
        Returns:
            Config instance
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def load(cls, path: str) -> "BaseConfig":
        """Load config from a JSON file.
        
        Args:
            path: File path to load the config from
            
        Returns:
            Config instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())


def get_config(config_dict: Dict[str, Any]) -> BaseConfig:
    """Load config from dictionary based on __class__ field.
    
    This function dynamically loads the appropriate config class based on the
    __class__ field in the config dictionary.
    
    Args:
        config_dict: Dictionary containing config values, must include __class__ field
        
    Returns:
        Config instance of the appropriate type
        
    Raises:
        ValueError: If __class__ field is missing or config class not found
        KeyError: If config class cannot be imported
        
    Example:
        >>> config_dict = {"__class__": "Qwen3Config", "model_class": "Qwen3Model", ...}
        >>> config = get_config(config_dict)
        >>> isinstance(config, Qwen3Config)
        True
    """
    config_class_name = config_dict.get("__class__")
    if not config_class_name:
        raise ValueError(
            "Config dictionary must contain '__class__' field to specify the config class"
        )
    
    # Remove __class__ from dict before creating config instance
    config_dict = {k: v for k, v in config_dict.items() if k != "__class__"}
    
    # Try to import config class from muse.config module
    try:
        import muse.config as config_module
        
        # Try direct import from config module first (configs are exported in __init__.py)
        if hasattr(config_module, config_class_name):
            config_class = getattr(config_module, config_class_name)
            return config_class.from_dict(config_dict)
        
        # Try model_config submodule
        if hasattr(config_module, 'model_config'):
            model_config_module = getattr(config_module, 'model_config')
            if hasattr(model_config_module, config_class_name):
                config_class = getattr(model_config_module, config_class_name)
                return config_class.from_dict(config_dict)
        
        # Try other config submodules
        for attr_name in ['dataset_config', 'training_config']:
            if hasattr(config_module, attr_name):
                submodule = getattr(config_module, attr_name)
                if hasattr(submodule, config_class_name):
                    config_class = getattr(submodule, config_class_name)
                    return config_class.from_dict(config_dict)
        
        # List available config classes for better error message
        available_classes = []
        if hasattr(config_module, 'model_config'):
            available_classes.extend([
                name for name in dir(config_module.model_config) 
                if name.endswith('Config') and not name.startswith('_')
            ])
        
        raise ValueError(
            f"Config class '{config_class_name}' not found in muse.config module. "
            f"Available config classes: {available_classes}"
        )
    except ImportError as e:
        raise ValueError(
            f"Failed to import config class '{config_class_name}': {e}"
        ) from e


def load_config(config_path: str) -> BaseConfig:
    """Load config directly from a configuration file path.
    
    This function reads a JSON or YAML configuration file and automatically
    determines the config class type based on the __class__ field in the file.
    
    Args:
        config_path: Path to configuration file (.json, .yaml, or .yml)
        
    Returns:
        Config instance of the appropriate type
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If file format is not supported, __class__ field is missing,
                    or config class not found
        
    Example:
        >>> # Load a Qwen3Config from file
        >>> config = load_config("configs/qwen3.json")
        >>> isinstance(config, Qwen3Config)
        True
        
        >>> # Load a TrainingConfig from file
        >>> config = load_config("configs/training.yaml")
        >>> isinstance(config, TrainingConfig)
        True
    """
    file_path = Path(config_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load config dictionary from file
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.suffix == '.json':
            config_dict = json.load(f)
        elif file_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                "Supported formats: .json, .yaml, .yml"
            )
    
    # Use get_config to create the appropriate config instance
    return get_config(config_dict)

