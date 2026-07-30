"""Domain objects shared by the model studio services and HTTP layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArchitectureDescriptor:
    id: str
    label: str
    description: str
    option_schema: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["option_schema"] = list(self.option_schema)
        return payload


@dataclass(frozen=True)
class ModelMetadata:
    architecture_id: str
    architecture_label: str
    checkpoint: str
    device: str
    output_heads: tuple[str, ...]
    parameter_count: int | None = None
    input_layout: str = "B,L,4"
    fixed_input_length: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_heads"] = list(self.output_heads)
        return payload


@dataclass(frozen=True)
class SequenceRecord:
    sequence: str
    label: str
    chromosome: str | None = None
    start: int | None = None
    end: int | None = None

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def gc_fraction(self) -> float:
        called = [base for base in self.sequence if base in "ACGT"]
        if not called:
            return 0.0
        return sum(base in "GC" for base in called) / len(called)

    def position_for_offset(self, offset: int) -> int:
        return (self.start or 0) + offset

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "length": self.length,
            "gc_fraction": self.gc_fraction,
            "chromosome": self.chromosome,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class DeletionSettings:
    deletion_size: int = 15
    stride: int = 1
    batch_size: int = 8
    smoothing_window: int = 1
    perturbation: str = "real"
    output_head: str = "regression"
    scan_scope: str = "context"

    def validate(self, sequence_length: int) -> None:
        if self.deletion_size < 1 or self.deletion_size >= sequence_length:
            raise ValueError(
                f"deletion_size must be between 1 and {sequence_length - 1}"
            )
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1")
        if self.perturbation not in {"real", "mask"}:
            raise ValueError("perturbation must be 'real' or 'mask'")
        if self.scan_scope not in {"context", "display"}:
            raise ValueError("scan_scope must be 'context' or 'display'")


@dataclass
class RunResult:
    sequence: dict[str, Any]
    regions: dict[str, Any]
    model: dict[str, Any]
    profile_model: dict[str, Any] | None
    settings: dict[str, Any]
    baseline: dict[str, list[float]]
    profile: dict[str, Any] | None
    window_positions: list[int]
    window_effects: list[float]
    nucleotide_positions: list[int]
    nucleotide_effects: list[float]
    plot_effects: list[float]
    plot_effect: dict[str, Any]
    provenance: dict[str, Any]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
