"""Architecture-neutral inference contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..domain import ArchitectureDescriptor, ModelMetadata


class SequenceModelAdapter(ABC):
    """Convert one model family to canonical DNA profile inference.

    The studio uses channels-last one-hot arrays shaped ``(batch, length, 4)``.
    Adapters own checkpoint reconstruction, tensor-layout conversion, device
    transfer, and normalization of outputs to ``(batch, output_length)``.
    """

    descriptor: ArchitectureDescriptor

    def __init__(
        self,
        checkpoint: str | Path,
        device: str,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = device
        self.options = options or {}
        self._metadata: ModelMetadata | None = None

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError("Model adapter has not been loaded")
        return self._metadata

    @abstractmethod
    def load(self) -> ModelMetadata:
        """Load and validate the checkpoint."""

    @abstractmethod
    def predict(
        self,
        sequences: np.ndarray,
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        """Return named outputs shaped ``(batch, output_length)``."""

    def close(self) -> None:
        """Release architecture-specific resources."""
