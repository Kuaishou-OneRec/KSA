"""
Qwen3 model implementation.
"""

from muse.models.qwen3.modeling import Qwen3Model

# Register the model (import here to avoid circular imports)
# The registration decorator is applied when this module is imported by muse.models
from muse.models.base import Model

# Apply decorator manually to avoid circular import issues
def _register_qwen3():
    """Register Qwen3Model in the model registry."""
    try:
        from muse.models import register_model
        # Register the model
        register_model("Qwen3Model")(Qwen3Model)
    except ImportError:
        # Registry not yet available during initial import
        pass

_register_qwen3()

__all__ = ["Qwen3Model"]

