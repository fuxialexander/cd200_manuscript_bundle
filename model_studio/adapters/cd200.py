"""Adapter for the bundled CD200 motif CNN checkpoints."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from ..domain import ArchitectureDescriptor, ModelMetadata
from .base import SequenceModelAdapter


class CD200MotifCNNAdapter(SequenceModelAdapter):
    descriptor = ArchitectureDescriptor(
        id="cd200_motif_cnn",
        label="CD200 motif CNN",
        description=(
            "Bundled DNA-to-ATAC motif CNN with regression/count heads and "
            "variable-length sequence input."
        ),
    )

    def __init__(self, checkpoint: str, device: str, options: dict[str, Any] | None = None):
        super().__init__(checkpoint, device, options)
        self.model: torch.nn.Module | None = None
        self.output_heads: tuple[str, ...] = ()

    def load(self) -> ModelMetadata:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        # Caesarion imports optional numba-compiled analysis helpers while the
        # mutation package initializes. They are not used for model inference,
        # and disabling their JIT avoids cache-locator failures in relocatable
        # local bundles.
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        from mutation_analysis.model import load_model

        load_options = dict(self.options)
        model, output_heads = load_model(
            str(self.checkpoint),
            device=self.device,
            **load_options,
        )
        self.model = model.to(self.device).eval()
        self.output_heads = tuple(output_heads)
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self._metadata = ModelMetadata(
            architecture_id=self.descriptor.id,
            architecture_label=self.descriptor.label,
            checkpoint=str(self.checkpoint),
            device=self.device,
            output_heads=self.output_heads,
            parameter_count=parameter_count,
            details={
                "checkpoint_size_bytes": self.checkpoint.stat().st_size,
                "cell_types": list(
                    self.options.get("cell_types", getattr(self.model, "celltype_ids", []))
                ),
            },
        )
        return self._metadata

    @staticmethod
    def _normalize_output(value: torch.Tensor) -> np.ndarray:
        array = value.detach().float().cpu().numpy()
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0, :]
        elif array.ndim == 1:
            array = array[:, None]
        elif array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        return np.asarray(array, dtype=np.float32)

    @torch.inference_mode()
    def predict(self, sequences: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Load the model before inference")
        if sequences.ndim != 3 or sequences.shape[-1] != 4:
            raise ValueError(f"Expected (N, L, 4), got {sequences.shape}")
        collected: dict[str, list[np.ndarray]] = {
            head: [] for head in self.output_heads
        }
        for offset in range(0, len(sequences), batch_size):
            chunk = np.ascontiguousarray(sequences[offset : offset + batch_size])
            tensor = torch.from_numpy(chunk).permute(0, 2, 1).to(self.device)
            outputs = self.model(tensor)
            if not isinstance(outputs, dict):
                outputs = {"regression": outputs}
            for head in self.output_heads:
                if head not in outputs:
                    raise KeyError(f"Model did not return declared output head {head!r}")
                collected[head].append(self._normalize_output(outputs[head]))
        return {
            head: np.concatenate(chunks, axis=0)
            for head, chunks in collected.items()
        }

    def close(self) -> None:
        self.model = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
