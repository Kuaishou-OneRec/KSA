"""
Dataset Configuration Classes.

This module defines configuration classes for dataset loading, preprocessing,
and batching. It uses Pydantic for validation and provides a flexible way to
configure data pipelines for training and evaluation.

The configuration handles:
- Dataset source specification
- Data format (chatml, completion, etc.)
- Sequence length limits
- Visual token handling for multimodal models
- DataLoader settings (batch size, workers)
- Dataset-specific extra configuration via JSON

Classes:
    DatasetConfig: Complete dataset configuration

Example:
    >>> from muse.config.dataset_config import DatasetConfig
    >>> 
    >>> # Create dataset configuration
    >>> config = DatasetConfig(
    ...     dataset="path/to/dataset",
    ...     data_format="chatml",
    ...     max_length=2048,
    ...     batch_size=32,
    ...     num_workers=4
    ... )
    >>> 
    >>> # Load additional config from JSON
    >>> config.load_extra_from_file("dataset_specific.json")
    >>> 
    >>> # Save configuration
    >>> config.save("dataset_config.json")
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset configuration classes."""

from typing import Optional, Literal, Dict, Any
from pydantic import Field

from muse.config.base import BaseConfig


class DatasetConfig(BaseConfig):
    """Dataset configuration for training and evaluation.
    
    This configuration handles data loading, preprocessing, and batching.
    """
    
    # Dataset source
    dataset_config: Optional[str] = Field(
        default=None,
        description="Path to dataset configuration JSON file"
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Dataset name or path"
    )
    
    # Data format
    data_format: Literal["chatml", "completion"] = Field(
        default="chatml",
        description="Data format for training"
    )

    # Sequence length
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum sequence length (tokens per sample)"
    )
    
    # DataLoader settings
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Batch size for training"
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of data loading workers"
    )
    
    # Advanced settings
    use_flops_balance: bool = Field(
        default=False,
        description="Use FLOPS-balanced data loading"
    )
    
    # Additional config that can be loaded from JSON
    extra_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional dataset-specific configuration"
    )
    
    def load_extra_from_file(self, path: str) -> None:
        """Load additional configuration from a JSON file.
        
        This allows loading dataset-specific parameters that are not
        defined in the base DatasetConfig schema.
        
        Args:
            path: Path to JSON configuration file
        """
        import json
        with open(path, 'r', encoding='utf-8') as f:
            extra = json.load(f)
            self.extra_config.update(extra)

