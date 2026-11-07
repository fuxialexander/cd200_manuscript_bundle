"""Caesar utility modules."""

from __future__ import annotations

from caesar.utils._optional import (
    TorchNotAvailableError,
    get_device,
    is_cuda_available,
    is_torch_available,
    requires_torch,
)

__all__ = [
    "is_torch_available",
    "is_cuda_available",
    "get_device",
    "requires_torch",
    "TorchNotAvailableError",
]
