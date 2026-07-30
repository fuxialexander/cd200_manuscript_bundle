"""Built-in model adapters."""

from .base import SequenceModelAdapter
from .cd200 import CD200MotifCNNAdapter
from .registry import ModelAdapterRegistry

__all__ = [
    "SequenceModelAdapter",
    "CD200MotifCNNAdapter",
    "ModelAdapterRegistry",
]
