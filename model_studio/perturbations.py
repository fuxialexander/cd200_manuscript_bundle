"""Architecture-neutral batched deletion analysis."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from .adapters.base import SequenceModelAdapter
from .domain import DeletionSettings, RunResult, SequenceRecord
from .sequences import one_hot_encode


ProgressCallback = Callable[[int, int, str], None]


def manuscript_plot_effects(
    raw_effects: np.ndarray,
    baseline_prediction: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the manuscript deletion-effect plotting transformation.

    This matches ``analysis/plot_scan_results.py::prepare_plot_effects`` with
    ``use_absolute_effect=True`` and
    ``normalize_to_max_predicted_atac=True``.
    """
    raw = np.asarray(raw_effects, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    absolute = np.abs(raw)
    finite_baseline = np.abs(baseline[np.isfinite(baseline)])
    denominator = (
        float(np.max(finite_baseline))
        if finite_baseline.size
        else 0.0
    )
    applied = denominator > 0
    transformed = (
        (absolute / denominator) * 100.0
        if applied
        else absolute
    )
    metadata: dict[str, object] = {
        "id": "absolute_percent_of_max_predicted_output",
        "source": "analysis/plot_scan_results.py::prepare_plot_effects",
        "formula": "100 * abs(raw deletion effect) / max(abs(baseline prediction))",
        "denominator": denominator,
        "applied": applied,
        "unit": "%" if applied else "absolute raw effect",
        "label": (
            "Absolute deletion effect (% max predicted accessibility)"
            if applied
            else "Absolute deletion effect"
        ),
    }
    return transformed.astype(np.float32), metadata


def _profile_effect(baseline: np.ndarray, altered: np.ndarray) -> np.ndarray:
    return baseline.max(axis=1) - altered.max(axis=1)


def _count_effect(baseline: np.ndarray, altered: np.ndarray) -> np.ndarray:
    reference = baseline.reshape(baseline.shape[0], -1).sum(axis=1)
    alternative = altered.reshape(altered.shape[0], -1).sum(axis=1)
    return np.log2((alternative + 1.0) / (reference + 1.0))


def _generic_effect(baseline: np.ndarray, altered: np.ndarray) -> np.ndarray:
    return (
        baseline.reshape(baseline.shape[0], -1).mean(axis=1)
        - altered.reshape(altered.shape[0], -1).mean(axis=1)
    )


def _differential_effect(
    baseline: np.ndarray,
    altered: np.ndarray,
) -> np.ndarray:
    """Match mutation_analysis.metrics.calculate_fc_effects: alt - ref."""
    reference = baseline.reshape(baseline.shape[0], -1).mean(axis=1)
    alternative = altered.reshape(altered.shape[0], -1).mean(axis=1)
    return alternative - reference


class DeletionAnalyzer:
    """Run baseline inference and a batched sliding-window deletion scan."""

    def __init__(self, adapter: SequenceModelAdapter) -> None:
        self.adapter = adapter

    @staticmethod
    def _build_chunk(
        sequence: np.ndarray,
        starts: np.ndarray,
        deletion_size: int,
        perturbation: str,
    ) -> np.ndarray:
        if perturbation == "mask":
            batch = np.repeat(sequence[None, :, :], len(starts), axis=0)
            for row, start in enumerate(starts):
                batch[row, start : start + deletion_size, :] = 0.0
            return batch
        output_length = len(sequence) - deletion_size
        batch = np.empty((len(starts), output_length, 4), dtype=np.float32)
        for row, start in enumerate(starts):
            batch[row] = np.concatenate(
                (
                    sequence[:start],
                    sequence[start + deletion_size :],
                ),
                axis=0,
            )
        return batch

    @staticmethod
    def _score(
        head: str,
        baseline: np.ndarray,
        altered: np.ndarray,
        core_slice: tuple[int, int] | None = None,
    ) -> np.ndarray:
        repeated = np.repeat(baseline, len(altered), axis=0)
        if head == "regression":
            if core_slice is not None:
                start, end = core_slice
                repeated = repeated[:, start:end]
                altered = altered[:, start:end]
            return _profile_effect(repeated, altered)
        if head == "count":
            return _count_effect(repeated, altered)
        if head in {"differential", "fc"}:
            return _differential_effect(repeated, altered)
        return _generic_effect(repeated, altered)

    @staticmethod
    def _distribute(
        starts: np.ndarray,
        effects: np.ndarray,
        sequence_length: int,
        deletion_size: int,
    ) -> np.ndarray:
        scores = np.zeros(sequence_length, dtype=np.float64)
        for start, effect in zip(starts, effects, strict=True):
            scores[start : start + deletion_size] += effect / deletion_size
        return scores.astype(np.float32)

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values
        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(values, kernel, mode="same").astype(np.float32)

    def run(
        self,
        record: SequenceRecord,
        settings: DeletionSettings,
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        *,
        core_region: tuple[int, int] | None = None,
        display_region: tuple[int, int] | None = None,
        profile_adapter: SequenceModelAdapter | None = None,
        observed_atac: np.ndarray | None = None,
        reference_profile: np.ndarray | None = None,
        reference_profile_source: str | None = None,
        preset_id: str = "custom",
    ) -> RunResult:
        settings.validate(record.length)
        if settings.output_head not in self.adapter.metadata.output_heads:
            raise ValueError(
                f"Output head {settings.output_head!r} is unavailable; "
                f"choose from {self.adapter.metadata.output_heads}"
            )
        started = time.perf_counter()
        encoded = one_hot_encode(record.sequence)
        core_slice: tuple[int, int] | None = None
        if core_region is not None:
            if record.start is None or record.end is None:
                raise ValueError("Core scoring requires a genomic-region input")
            core_slice = (
                int(core_region[0]) - record.start,
                int(core_region[1]) - record.start,
            )
            if core_slice[0] < 0 or core_slice[1] > record.length:
                raise ValueError(
                    f"Core region {core_region} is outside {record.label}"
                )
        if display_region is not None:
            if record.start is None or record.end is None:
                raise ValueError("Display region requires a genomic-region input")
            if (
                display_region[0] < record.start
                or display_region[1] > record.end
            ):
                raise ValueError(
                    f"Display region {display_region} is outside {record.label}"
                )
        baseline_outputs = self.adapter.predict(
            encoded[None, :, :],
            batch_size=1,
        )
        profile: dict[str, object] | None = None
        if profile_adapter is not None:
            profile_outputs = profile_adapter.predict(
                encoded[None, :, :],
                batch_size=1,
            )
            if "regression" not in profile_outputs:
                raise ValueError(
                    "The profile model must provide a regression output head"
                )
            predicted = np.asarray(
                profile_outputs["regression"][0],
                dtype=np.float32,
            ).reshape(-1)
            if record.start is None:
                profile_positions = np.arange(len(predicted), dtype=np.int64)
            elif len(predicted) <= 1:
                profile_positions = np.asarray([record.start], dtype=np.int64)
            else:
                profile_positions = np.rint(
                    np.linspace(
                        record.start,
                        record.end - 1,
                        len(predicted),
                    )
                ).astype(np.int64)
            profile = {
                "positions": profile_positions.tolist(),
                "predicted": predicted.tolist(),
                "observed": (
                    np.asarray(observed_atac, dtype=np.float32)
                    .reshape(-1)
                    .tolist()
                    if observed_atac is not None
                    else None
                ),
                "reference": (
                    np.asarray(reference_profile, dtype=np.float32)
                    .reshape(-1)
                    .tolist()
                    if reference_profile is not None
                    else None
                ),
                "reference_source": reference_profile_source,
            }
        all_starts = np.arange(
            0,
            record.length - settings.deletion_size + 1,
            settings.stride,
            dtype=np.int64,
        )
        output_start = 0
        output_end = record.length
        starts = all_starts
        if settings.scan_scope == "display":
            if display_region is None or record.start is None:
                raise ValueError(
                    "Display-only scanning requires a genomic display region"
                )
            output_start = int(display_region[0]) - record.start
            output_end = int(display_region[1]) - record.start
            smoothing_margin = (
                settings.smoothing_window
                if settings.smoothing_window > 1
                else 0
            )
            needed_start = max(0, output_start - smoothing_margin)
            needed_end = min(record.length, output_end + smoothing_margin)
            relevant_indices = np.flatnonzero(
                (all_starts < needed_end)
                & (all_starts + settings.deletion_size > needed_start)
            )
            if relevant_indices.size:
                aligned_start = (
                    int(relevant_indices[0]) // settings.batch_size
                ) * settings.batch_size
                aligned_end = min(
                    len(all_starts),
                    (
                        (
                            int(relevant_indices[-1])
                            + settings.batch_size
                        )
                        // settings.batch_size
                    )
                    * settings.batch_size,
                )
                starts = all_starts[aligned_start:aligned_end]
            else:
                starts = all_starts[:0]
        effects: list[np.ndarray] = []
        total = len(starts)
        for offset in range(0, total, settings.batch_size):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Run cancelled")
            chunk_starts = starts[offset : offset + settings.batch_size]
            altered_sequences = self._build_chunk(
                encoded,
                chunk_starts,
                settings.deletion_size,
                settings.perturbation,
            )
            altered_outputs = self.adapter.predict(
                altered_sequences,
                batch_size=settings.batch_size,
            )
            effects.append(
                self._score(
                    settings.output_head,
                    baseline_outputs[settings.output_head],
                    altered_outputs[settings.output_head],
                    core_slice,
                )
            )
            if progress is not None:
                progress(
                    min(offset + len(chunk_starts), total),
                    total,
                    "Deletion scan",
                )
        window_effects = np.concatenate(effects).astype(np.float32)
        nucleotide_effects = self._distribute(
            starts,
            window_effects,
            record.length,
            settings.deletion_size,
        )
        nucleotide_effects_full = self._smooth(
            nucleotide_effects,
            settings.smoothing_window,
        )
        nucleotide_effects = nucleotide_effects_full[output_start:output_end]
        head_baseline = np.asarray(
            baseline_outputs[settings.output_head][0],
            dtype=np.float32,
        ).reshape(-1)
        plot_effects, plot_effect = manuscript_plot_effects(
            nucleotide_effects,
            head_baseline,
        )
        position_offset = record.start or 0
        baseline = {
            head: np.asarray(values[0], dtype=np.float32).reshape(-1).tolist()
            for head, values in baseline_outputs.items()
        }
        context_payload = {
            "chromosome": record.chromosome,
            "start": record.start,
            "end": record.end,
            "label": record.label,
        }
        core_payload = (
            {
                "chromosome": record.chromosome,
                "start": int(core_region[0]),
                "end": int(core_region[1]),
                "label": (
                    f"{record.chromosome}:{core_region[0]}-{core_region[1]}"
                ),
            }
            if core_region is not None
            else context_payload.copy()
        )
        display_payload = (
            {
                "chromosome": record.chromosome,
                "start": int(display_region[0]),
                "end": int(display_region[1]),
                "label": (
                    f"{record.chromosome}:{display_region[0]}-{display_region[1]}"
                ),
            }
            if display_region is not None
            else core_payload.copy()
        )
        scan_payload = (
            display_payload.copy()
            if settings.scan_scope == "display"
            else context_payload.copy()
        )
        if settings.output_head == "regression":
            effect_definition = (
                "reference max minus altered max within the scoring core"
                if core_slice is not None
                else "reference max minus altered max"
            )
        elif settings.output_head in {"differential", "fc"}:
            effect_definition = "altered mean minus reference mean"
        elif settings.output_head == "count":
            effect_definition = "log2((altered + 1) / (reference + 1))"
        else:
            effect_definition = "reference mean minus altered mean"
        return RunResult(
            sequence=record.summary(),
            regions={
                "context": context_payload,
                "core": core_payload,
                "display": display_payload,
                "scan": scan_payload,
            },
            model=self.adapter.metadata.to_dict(),
            profile_model=(
                profile_adapter.metadata.to_dict()
                if profile_adapter is not None
                else None
            ),
            settings={
                "deletion_size": settings.deletion_size,
                "stride": settings.stride,
                "batch_size": settings.batch_size,
                "smoothing_window": settings.smoothing_window,
                "perturbation": settings.perturbation,
                "output_head": settings.output_head,
                "scan_scope": settings.scan_scope,
                "scanned_window_count": len(starts),
                "full_context_window_count": len(all_starts),
                "effect_definition": effect_definition,
            },
            baseline=baseline,
            profile=profile,
            window_positions=(
                starts + settings.deletion_size // 2 + position_offset
            ).tolist(),
            window_effects=window_effects.tolist(),
            nucleotide_positions=(
                np.arange(output_start, output_end, dtype=np.int64)
                + position_offset
            ).tolist(),
            nucleotide_effects=nucleotide_effects.tolist(),
            plot_effects=plot_effects.tolist(),
            plot_effect=plot_effect,
            provenance={
                "preset_id": preset_id,
                "analysis_role": "deletion effect",
                "profile_role": (
                    "predicted accessibility overlay"
                    if profile_adapter is not None
                    else None
                ),
            },
            elapsed_seconds=time.perf_counter() - started,
        )
