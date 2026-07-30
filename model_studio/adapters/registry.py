"""Model-adapter registry used by the API and service layer."""

from __future__ import annotations

from collections.abc import Iterable

from .base import SequenceModelAdapter
from .cd200 import CD200MotifCNNAdapter


class ModelAdapterRegistry:
    def __init__(
        self,
        adapters: Iterable[type[SequenceModelAdapter]] | None = None,
    ) -> None:
        # GET v2 pool16 transformer adapter is intentionally not published yet.
        classes = adapters or (CD200MotifCNNAdapter,)
        self._adapters = {adapter.descriptor.id: adapter for adapter in classes}

    def descriptors(self) -> list[dict]:
        return [
            adapter.descriptor.to_dict()
            for adapter in self._adapters.values()
        ]

    def create(
        self,
        architecture_id: str,
        checkpoint: str,
        device: str,
        options: dict | None = None,
    ) -> SequenceModelAdapter:
        adapter_class = self._adapters.get(architecture_id)
        if adapter_class is None:
            known = ", ".join(sorted(self._adapters))
            raise ValueError(
                f"Unknown architecture {architecture_id!r}; available: {known}"
            )
        return adapter_class(checkpoint, device, options)
