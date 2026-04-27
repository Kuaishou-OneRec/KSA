"""
Model registry for muse models.

This module provides a centralized registry for model classes that can be 
accessed by name in training scripts.
"""

from typing import Dict, Type, Optional, List
from muse.models.base import Model

# Global model registry
MODEL_REGISTRY: Dict[str, Type[Model]] = {}


def register_model(name: Optional[str] = None):
    """
    Decorator to register a model class in the model registry.
    
    Args:
        name: Name to register the model under. If None, uses the class name.
        
    Example:
        @register_model("Qwen3Model")
        class Qwen3Model(Model):
            ...
    """
    def decorator(cls: Type[Model]):
        model_name = name if name is not None else cls.__name__
        
        if model_name in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_name}' is already registered. "
                f"Existing: {MODEL_REGISTRY[model_name]}, New: {cls}"
            )
        
        MODEL_REGISTRY[model_name] = cls
        return cls
    
    return decorator


def get_model_class(name: str) -> Type[Model]:
    """
    Retrieve a registered model class by name.
    
    Args:
        name: Name of the model to retrieve.
        
    Returns:
        The model class.
        
    Raises:
        KeyError: If the model name is not registered.
        
    Example:
        model_cls = get_model_class("Qwen3Model")
        model = model_cls(config)
    """
    if name not in MODEL_REGISTRY:
        available_models = list_models()
        raise KeyError(
            f"Model '{name}' is not registered. "
            f"Available models: {available_models}"
        )
    
    return MODEL_REGISTRY[name]


def list_models() -> List[str]:
    """
    List all registered model names.
    
    Returns:
        List of registered model names.
    """
    return sorted(MODEL_REGISTRY.keys())


# Import model modules to trigger registration
from muse.models import qwen3  # noqa: E402, F401
from muse.models import qwen3_sa  # noqa: E402, F401


# Export public API
__all__ = [
    "Model",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_class",
    "list_models",
]


