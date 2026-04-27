"""
Qwen3 Summary Attention model implementation.
"""

from muse.models.qwen3_sa.modeling import Qwen3SummaryModel


def _register_qwen3_sa():
    """Register Qwen3SummaryModel in the model registry."""
    try:
        from muse.models import register_model
        register_model("Qwen3SummaryModel")(Qwen3SummaryModel)
    except ImportError:
        pass

_register_qwen3_sa()

__all__ = ["Qwen3SummaryModel"]
