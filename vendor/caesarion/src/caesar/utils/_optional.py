"""Utilities for checking optional dependencies.

This module provides utilities for checking if optional dependencies like PyTorch
are available, allowing graceful degradation when they are not installed.
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Cache for import results
_import_cache: dict[str, bool] = {}


def is_torch_available() -> bool:
    """Check if PyTorch is available.

    Returns
    -------
    bool
        True if torch is installed and importable, False otherwise.
    """
    if "torch" not in _import_cache:
        try:
            import torch  # noqa: F401

            _import_cache["torch"] = True
        except ImportError:
            _import_cache["torch"] = False
    return _import_cache["torch"]


def is_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch.

    Returns
    -------
    bool
        True if torch is installed and CUDA is available, False otherwise.
    """
    if not is_torch_available():
        return False
    import torch

    return torch.cuda.is_available()


def get_device(preferred: str = "auto") -> str:
    """Get the appropriate compute device.

    Parameters
    ----------
    preferred : str
        Preferred device: 'auto', 'cuda', or 'cpu'.
        If 'auto', uses CUDA if available, otherwise CPU.
        If 'cuda' is requested but not available, falls back to CPU.

    Returns
    -------
    str
        The device string: 'cuda' or 'cpu'.
    """
    if preferred == "auto":
        return "cuda" if is_cuda_available() else "cpu"
    elif preferred == "cuda":
        if is_cuda_available():
            return "cuda"
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
    else:
        return "cpu"


def requires_torch(func):
    """Decorator that raises ImportError if torch is not available.

    Use this decorator on functions that require PyTorch to provide
    a clear error message when torch is not installed.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_torch_available():
            raise ImportError(
                f"Function '{func.__name__}' requires PyTorch. "
                "Install it with: pip install caesar[torch]"
            )
        return func(*args, **kwargs)

    return wrapper


class TorchNotAvailableError(ImportError):
    """Raised when a torch-dependent feature is used but torch is not installed."""

    def __init__(self, feature_name: str = "This feature"):
        super().__init__(
            f"{feature_name} requires PyTorch. "
            "Install it with: pip install caesar[torch]"
        )
