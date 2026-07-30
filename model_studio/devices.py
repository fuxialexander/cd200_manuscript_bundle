"""PyTorch device discovery and validation."""

from __future__ import annotations

from typing import Any

import torch


class DeviceManager:
    """Resolve user-facing device choices to available PyTorch devices."""

    @staticmethod
    def capabilities() -> dict[str, Any]:
        mps_built = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        )
        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        cuda_available = torch.cuda.is_available()
        cuda_devices = [
            {
                "id": f"cuda:{index}",
                "name": torch.cuda.get_device_name(index),
            }
            for index in range(torch.cuda.device_count())
        ]
        return {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_devices": cuda_devices,
            "mps_built": mps_built,
            "mps_available": mps_available,
            "cpu_available": True,
            "recommended": DeviceManager.resolve("auto"),
        }

    @staticmethod
    def resolve(requested: str | None) -> str:
        value = (requested or "auto").strip().lower()
        if value == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if value == "cuda":
            value = "cuda:0"
        if value.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if value == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available")
        if value != "cpu" and value != "mps" and not value.startswith("cuda"):
            raise ValueError(f"Unsupported device: {requested}")
        return value
