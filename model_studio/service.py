"""Application service coordinating adapters, inputs, and jobs."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .adapters.base import SequenceModelAdapter
from .adapters.registry import ModelAdapterRegistry
from .annotations import MotifAnnotationService
from .devices import DeviceManager
from .domain import DeletionSettings, ModelMetadata, SequenceRecord
from .jobs import JobManager
from .perturbations import DeletionAnalyzer
from .sequences import (
    GenomeZarrSequenceSource,
    TextSequenceSource,
    _read_local_zarr_v2_1d_slice,
)


class ModelStudioService:
    """Stateful local service with one loaded model and many background runs."""

    def __init__(self, app_root: Path) -> None:
        self.app_root = app_root
        self.registry = ModelAdapterRegistry()
        self.jobs = JobManager(app_root / "runs")
        self.annotations = MotifAnnotationService()
        self._adapter: SequenceModelAdapter | None = None
        self._profile_adapter: SequenceModelAdapter | None = None
        self._model_lock = threading.RLock()

    def _bundle_path(self, relative: str) -> str:
        return str((self.app_root.parent / relative).resolve())

    def _presets(self) -> list[dict[str, Any]]:
        regression_checkpoint = self._bundle_path(
            "analysis/checkpoints/"
            "shapescale_loss_scca_pos_with_count_head_fixed_"
            "motif_curriculum_from_scratch-epoch=105-val_loss=6.76.ckpt"
        )
        regression_options = {
            "config_file": self._bundle_path("configs/train_atac_phased.yaml"),
            "simple_motifs_path": self._bundle_path(
                "artifacts/motifs_with_rc_aligned.pt"
            ),
        }
        phase3_options = {
            **regression_options,
            "tn5_motifs_path": self._bundle_path("artifacts/bias.meme"),
            "phase_config_file": self._bundle_path(
                "configs/train_atac_phased.yaml"
            ),
        }
        observed_track = {
            "path": self._bundle_path(
                "data/zarr/cd200_scc_dense_new.zarr"
            ),
            "cell_type": "SCCApos.cd200_scc",
            "normalization_factor": 1e8,
            "convolution_bp": 20,
            "cached_csv": self._bundle_path(
                "analysis/results/cd200_promoter_analysis/"
                "regression_profile_overlay/"
                "chr16_45408000_45409426_deletion_regression_nucleotide.csv"
            ),
        }
        shared = {
            "source": {
                "kind": "region",
                "region": "chr16:45408000-45409426",
                "genome_zarr": self._bundle_path("data/zarr/mm10.zarr"),
            },
            "context_bp": 4096,
            "display_bp": 2000,
            "deletion": {
                "deletion_size": 15,
                "stride": 1,
                "batch_size": 32,
                "smoothing_window": 1,
                "perturbation": "real",
                "scan_scope": "display",
            },
            "annotations": {
                "rna_path": self._bundle_path(
                    "analysis/rnaseq_deg_analysis/1F.deg.gene.csv"
                ),
                "rna_path_2": self._bundle_path(
                    "analysis/rnaseq_deg_analysis/AF.deg.gene.csv"
                ),
                "gene_list_path": self._bundle_path(
                    "analysis/rnaseq_deg_analysis/"
                    "shared_DEGs_1F_AF_tf_relaxed.csv"
                ),
            },
            "observed_track": observed_track,
        }
        return [
            {
                "id": "custom",
                "label": "Custom checkpoint",
                "description": (
                    "Choose one checkpoint, input, scoring head, and annotation "
                    "filter explicitly."
                ),
            },
            {
                **shared,
                "id": "manuscript_differential_promoter",
                "label": "Manuscript promoter · differential",
                "description": (
                    "Reproduces the promoter panel: phase-3 differential "
                    "deletion scores, a separate regression-profile overlay, "
                    "4,096-bp model context, and the relaxed manuscript TF list."
                ),
                "analysis_model": {
                    "architecture_id": "cd200_motif_cnn",
                    "checkpoint": self._bundle_path(
                        "analysis/checkpoints/"
                        "ATAC-Sequential-20250919-022148_phase3_final_model.ckpt"
                    ),
                    "options": phase3_options,
                    "output_head": "differential",
                    "role": "Differential deletion model",
                },
                "profile_model": {
                    "architecture_id": "cd200_motif_cnn",
                    "checkpoint": regression_checkpoint,
                    "options": regression_options,
                    "role": "Regression accessibility profile",
                },
                "profile_reference": {
                    "path": self._bundle_path(
                        "analysis/results/cd200_promoter_analysis/"
                        "regression_profile_overlay/"
                        "chr16_45408000_45409426_deletion_regression_nucleotide.csv"
                    ),
                    "column": "observed_prediction",
                    "label": "Saved manuscript profile",
                },
                "annotations": {
                    **shared["annotations"],
                    "motif_path": self._bundle_path(
                        "model_studio/data/"
                        "cd200_promoter_all_motifs_p1e-4.tsv"
                    ),
                    "default_mode": "gene_list",
                    "min_effect_pct": 10,
                    "group_style": "figure",
                },
                "provenance": {
                    "figure": (
                        "analysis/results/cd200_promoter_analysis/"
                        "combined_plot_highlighted_differential_promoter_relaxed_"
                        "chr16_45408000_45409426.png"
                    ),
                    "analysis": "analysis/scripts/run_cd200_promoter_mutation.sh",
                    "regression_checkpoint": (
                        "analysis/checkpoints/"
                        "shapescale_loss_scca_pos_with_count_head_fixed_"
                        "motif_curriculum_from_scratch-epoch=105-val_loss=6.76.ckpt"
                    ),
                    "differential_checkpoint": (
                        "analysis/checkpoints/"
                        "ATAC-Sequential-20250919-022148_phase3_final_model.ckpt"
                    ),
                },
            },
            {
                **shared,
                "id": "regression_profile_promoter",
                "label": "Promoter · regression model",
                "description": (
                    "Uses the regression/profile checkpoint for deletion "
                    "scoring. Gene selection remains an explicit independent "
                    "choice."
                ),
                "analysis_model": {
                    "architecture_id": "cd200_motif_cnn",
                    "checkpoint": regression_checkpoint,
                    "options": regression_options,
                    "output_head": "regression",
                    "role": "Regression deletion and profile model",
                },
                "annotations": {
                    **shared["annotations"],
                    "motif_path": self._bundle_path(
                        "model_studio/data/"
                        "cd200_promoter_all_motifs_p1e-4.tsv"
                    ),
                    "default_mode": "expression",
                    "min_effect_pct": 10,
                    "group_style": "tf",
                },
            },
        ]

    def _preset(self, preset_id: str) -> dict[str, Any] | None:
        return next(
            (preset for preset in self._presets() if preset["id"] == preset_id),
            None,
        )

    def capabilities(self) -> dict[str, Any]:
        presets = self._presets()
        regression = presets[2]
        return {
            "devices": DeviceManager.capabilities(),
            "architectures": self.registry.descriptors(),
            "analysis_presets": presets,
            "defaults": {
                "preset_id": "manuscript_differential_promoter",
                "genome_zarr": regression["source"]["genome_zarr"],
                "region": regression["source"]["region"],
                "checkpoint": regression["analysis_model"]["checkpoint"],
                "motif_annotations": regression["annotations"]["motif_path"],
                "rna_csv": regression["annotations"]["rna_path"],
            },
        }

    @staticmethod
    def _chromosome_sort_key(name: str) -> tuple[int, int | str]:
        suffix = name.removeprefix("chr")
        if suffix.isdigit():
            return (0, int(suffix))
        order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
        return (1, order.get(suffix, suffix))

    def chromosome_sizes(self, preset_id: str) -> list[tuple[str, int]]:
        preset = self._preset(preset_id)
        if preset is None or not preset.get("source"):
            raise ValueError(f"Preset {preset_id!r} has no reference genome")
        genome_zarr = Path(preset["source"]["genome_zarr"])
        chromosome_root = genome_zarr / "chrs"
        if not chromosome_root.exists():
            raise FileNotFoundError(f"Genome zarr not found: {genome_zarr}")
        sizes: list[tuple[str, int]] = []
        for chromosome_path in chromosome_root.iterdir():
            metadata_path = chromosome_path / ".zarray"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text())
            sizes.append((chromosome_path.name, int(metadata["shape"][0])))
        if not sizes:
            raise ValueError(f"No chromosomes found in {genome_zarr}")
        return sorted(sizes, key=lambda item: self._chromosome_sort_key(item[0]))

    @staticmethod
    def _smooth_nearest(values: np.ndarray, window: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if window <= 1 or len(values) == 0:
            return values
        left = (window - 1) // 2
        right = window - 1 - left
        padded = np.pad(values, (left, right), mode="edge")
        kernel = np.full(window, 1.0 / window, dtype=np.float32)
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)

    @classmethod
    def _observed_values(
        cls,
        track_config: dict[str, Any],
        chromosome: str,
        start: int,
        end: int,
    ) -> np.ndarray:
        path = Path(str(track_config.get("path", "")).strip())
        cell_type = str(track_config.get("cell_type", "")).strip()
        if not path or not cell_type:
            raise ValueError("Observed ATAC path and cell type are required")
        chromosome_path = path / cell_type / "chrs" / chromosome
        metadata_path = chromosome_path / ".zarray"
        if not metadata_path.exists():
            raise ValueError(
                f"Observed ATAC has no signal for chromosome {chromosome!r}"
            )
        metadata = json.loads(metadata_path.read_text())
        chromosome_length = int(metadata["shape"][0])
        if start < 0 or end <= start or end > chromosome_length:
            raise ValueError(
                f"{chromosome}:{start}-{end} is outside chromosome length "
                f"{chromosome_length}"
            )
        window = max(1, int(track_config.get("convolution_bp", 20)))
        left = (window - 1) // 2
        right = window - 1 - left
        read_start = max(0, start - left)
        read_end = min(chromosome_length, end + right)
        raw, _ = _read_local_zarr_v2_1d_slice(
            chromosome_path,
            read_start,
            read_end,
        )
        normalization_factor = float(
            track_config.get("normalization_factor", 1e8)
        )
        if normalization_factor <= 0:
            raise ValueError("Observed ATAC normalization factor must be positive")
        normalized = raw.astype(np.float32) / normalization_factor * 1e8
        smoothed = cls._smooth_nearest(normalized, window)
        offset = start - read_start
        return smoothed[offset : offset + (end - start)]

    def observed_atac_bedgraph(
        self,
        preset_id: str,
        chromosome: str,
        start: int,
        end: int,
        *,
        max_points: int = 50_000,
    ) -> tuple[str, dict[str, int | str]]:
        preset = self._preset(preset_id)
        track_config = (preset or {}).get("observed_track")
        if not track_config:
            raise ValueError(
                f"Preset {preset_id!r} does not define an observed ATAC track"
            )
        chromosome_sizes = dict(self.chromosome_sizes(preset_id))
        if chromosome not in chromosome_sizes:
            raise ValueError(f"Unknown chromosome {chromosome!r}")
        chromosome_length = chromosome_sizes[chromosome]
        start = max(0, int(start))
        end = min(chromosome_length, int(end))
        if end <= start:
            raise ValueError("Track end must be greater than start")
        if end - start > 2_000_000:
            raise ValueError("Observed-track requests are limited to 2,000,000 bp")
        max_points = min(50_000, max(500, int(max_points)))
        values = self._observed_values(
            track_config,
            chromosome,
            start,
            end,
        )
        bin_bp = max(1, int(np.ceil(len(values) / max_points)))
        lines: list[str] = []
        for offset in range(0, len(values), bin_bp):
            bin_end = min(len(values), offset + bin_bp)
            value = float(np.mean(values[offset:bin_end]))
            lines.append(
                f"{chromosome}\t{start + offset}\t{start + bin_end}\t"
                f"{value:.6g}"
            )
        metadata: dict[str, int | str] = {
            "chromosome": chromosome,
            "start": start,
            "end": end,
            "bin_bp": bin_bp,
            "source": "local_zarr",
        }
        return "\n".join(lines) + "\n", metadata

    def model_metadata(self) -> dict[str, Any] | None:
        with self._model_lock:
            return self._adapter.metadata.to_dict() if self._adapter else None

    def profile_model_metadata(self) -> dict[str, Any] | None:
        with self._model_lock:
            return (
                self._profile_adapter.metadata.to_dict()
                if self._profile_adapter
                else None
            )

    def load_model(
        self,
        architecture_id: str,
        checkpoint: str,
        requested_device: str = "auto",
        options: dict[str, Any] | None = None,
        *,
        preset_id: str = "custom",
        profile_architecture_id: str | None = None,
        profile_checkpoint: str | None = None,
        profile_options: dict[str, Any] | None = None,
    ) -> ModelMetadata:
        if any(job["status"] in {"queued", "running"} for job in self.jobs.list()):
            raise RuntimeError(
                "Wait for the active run to finish or cancel it before replacing the model"
            )
        preset = self._preset(preset_id)
        if preset_id != "custom":
            if preset is None:
                raise ValueError(f"Unknown analysis preset: {preset_id}")
            configured = preset.get("analysis_model") or {}
            architecture_id = str(configured.get("architecture_id", ""))
            checkpoint = str(configured.get("checkpoint", ""))
            options = dict(configured.get("options") or {})
            profile = preset.get("profile_model") or {}
            profile_architecture_id = profile.get("architecture_id")
            profile_checkpoint = profile.get("checkpoint")
            profile_options = dict(profile.get("options") or {})
        device = DeviceManager.resolve(requested_device)
        candidate = self.registry.create(
            architecture_id, checkpoint, device, options
        )
        profile_candidate: SequenceModelAdapter | None = None
        try:
            metadata = candidate.load()
            if profile_checkpoint:
                profile_candidate = self.registry.create(
                    profile_architecture_id or architecture_id,
                    profile_checkpoint,
                    device,
                    profile_options,
                )
                profile_candidate.load()
        except Exception:
            candidate.close()
            if profile_candidate is not None:
                profile_candidate.close()
            raise
        with self._model_lock:
            previous = self._adapter
            previous_profile = self._profile_adapter
            self._adapter = candidate
            self._profile_adapter = profile_candidate
        if previous is not None:
            previous.close()
        if previous_profile is not None:
            previous_profile.close()
        return metadata

    @staticmethod
    def _centered_interval(
        start: int,
        end: int,
        length: int,
        *,
        inclusive_center: bool = False,
    ) -> tuple[int, int]:
        if length <= 0:
            raise ValueError("Interval length must be positive")
        center = (
            (start + end - 1) / 2
            if inclusive_center
            else (start + end) // 2
        )
        interval_start = int(center - length / 2)
        return interval_start, interval_start + length

    @classmethod
    def _sequence_geometry(
        cls,
        payload: dict[str, Any],
    ) -> tuple[SequenceRecord, tuple[int, int] | None, tuple[int, int] | None]:
        source = payload.get("source", {})
        kind = source.get("kind", "region")
        if kind == "region":
            chromosome, core_start, core_end = GenomeZarrSequenceSource.parse_region(
                source.get("region", "")
            )
            context_bp = int(payload.get("context_bp") or core_end - core_start)
            if context_bp < core_end - core_start:
                raise ValueError(
                    "Selected region is wider than the model context "
                    f"({core_end - core_start:,} bp versus {context_bp:,} bp)"
                )
            context_start, context_end = cls._centered_interval(
                core_start,
                core_end,
                context_bp,
            )
            display_bp = int(payload.get("display_bp") or context_bp)
            display_bp = min(display_bp, context_bp)
            if display_bp == core_end - core_start:
                display_start, display_end = core_start, core_end
            else:
                display_start, display_end = cls._centered_interval(
                    context_start,
                    context_end,
                    display_bp,
                    inclusive_center=True,
                )
            record = GenomeZarrSequenceSource(
                f"{chromosome}:{context_start}-{context_end}",
                source.get("genome_zarr", ""),
            ).load()
            return (
                record,
                (core_start, core_end),
                (display_start, display_end),
            )
        if kind in {"sequence", "fasta"}:
            return (
                TextSequenceSource(
                    source.get("sequence", ""),
                    source.get("label", "Pasted sequence"),
                ).load(),
                None,
                None,
            )
        raise ValueError(f"Unsupported sequence source: {kind}")

    @classmethod
    def _sequence_from_payload(cls, payload: dict[str, Any]) -> SequenceRecord:
        return cls._sequence_geometry(payload)[0]

    def inspect_sequence(self, payload: dict[str, Any]) -> dict[str, Any]:
        record, core, display = self._sequence_geometry(payload)
        summary = record.summary()
        summary["regions"] = {
            "context": record.label,
            "core": (
                f"{record.chromosome}:{core[0]}-{core[1]}" if core else None
            ),
            "display": (
                f"{record.chromosome}:{display[0]}-{display[1]}"
                if display
                else None
            ),
        }
        return summary

    def load_annotations(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload.get("job_id", "")).strip()
        if job_id:
            result = self.result(job_id)
            display = (result.get("regions") or {}).get("display")
            sequence = display or result["sequence"]
            chromosome = sequence.get("chromosome")
            start = sequence.get("start")
            end = sequence.get("end")
            if chromosome is None or start is None or end is None:
                raise ValueError(
                    "Motif annotations require a genomic-region model run"
                )
        else:
            chromosome, start, end = GenomeZarrSequenceSource.parse_region(
                str(payload.get("region", ""))
            )
        return self.annotations.load(
            str(payload.get("motif_path", "")),
            chromosome=str(chromosome),
            start=int(start),
            end=int(end),
            rna_path=str(payload.get("rna_path", "")).strip() or None,
            rna_path_2=str(payload.get("rna_path_2", "")).strip() or None,
            gene_list_path=(
                str(payload.get("gene_list_path", "")).strip() or None
            ),
            rna_columns=payload.get("rna_columns") or None,
            merge_overlaps=bool(payload.get("merge_overlaps", True)),
        )

    def filter_annotations(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            raise ValueError("Annotation filtering requires a completed job")
        result = self.result(job_id)
        return self.annotations.filter(
            result,
            mode=str(payload.get("mode", "rna")),
            min_abs_logfc=float(payload.get("min_abs_logfc", 2.0)),
            min_expression=float(payload.get("min_expression", 10.0)),
            max_padj=float(payload.get("max_padj", 0.05)),
            selected_tf=str(payload.get("selected_tf", "")).strip() or None,
            min_effect_pct=float(payload.get("min_effect_pct", 10.0)),
        )

    @staticmethod
    def _cached_column(
        csv_path: str,
        column: str,
        record: SequenceRecord,
    ) -> np.ndarray | None:
        if (
            not csv_path
            or not Path(csv_path).is_file()
            or record.start is None
            or record.end is None
        ):
            return None
        import pandas as pd

        frame = pd.read_csv(csv_path, usecols=["position", column])
        selected = frame[
            frame["position"].ge(int(record.start))
            & frame["position"].lt(int(record.end))
        ].sort_values("position")
        expected_positions = np.arange(
            int(record.start),
            int(record.end),
            dtype=np.int64,
        )
        observed_positions = selected["position"].to_numpy(dtype=np.int64)
        if np.array_equal(observed_positions, expected_positions):
            return selected[column].to_numpy(dtype=np.float32)
        return None

    @classmethod
    def _observed_atac(
        cls,
        track_config: dict[str, Any] | None,
        record: SequenceRecord,
    ) -> np.ndarray | None:
        if not track_config or record.chromosome is None:
            return None
        path = str(track_config.get("path", "")).strip()
        cell_type = str(track_config.get("cell_type", "")).strip()
        cached_csv = str(track_config.get("cached_csv", "")).strip()
        cached = cls._cached_column(cached_csv, "observed_atac", record)
        if cached is not None:
            return cached
        if not path or not cell_type:
            return None
        values = cls._observed_values(
            track_config,
            record.chromosome,
            int(record.start),
            int(record.end),
        )
        if len(values) != record.length:
            raise ValueError(
                "Observed ATAC length does not match the model context: "
                f"{len(values)} versus {record.length}"
            )
        return values

    @staticmethod
    def _apply_preset_sequence(
        payload: dict[str, Any],
        preset: dict[str, Any],
    ) -> dict[str, Any]:
        preset_source = dict(preset["source"])
        requested_source = payload.get("source") or {}
        requested_region = str(requested_source.get("region", "")).strip()
        if requested_source.get("kind") == "region" and requested_region:
            preset_source["region"] = requested_region
        context_bp = int(preset["context_bp"])
        requested_display = int(payload.get("display_bp") or preset["display_bp"])
        display_bp = min(context_bp, max(32, requested_display))
        return {
            **payload,
            "source": preset_source,
            "context_bp": context_bp,
            "display_bp": display_bp,
        }

    def submit_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._model_lock:
            adapter = self._adapter
            profile_adapter = self._profile_adapter
        if adapter is None:
            raise RuntimeError("Load a model before starting inference")
        preset_id = str(payload.get("preset_id", "custom"))
        preset = self._preset(preset_id)
        if preset_id != "custom":
            if preset is None:
                raise ValueError(f"Unknown analysis preset: {preset_id}")
            payload = self._apply_preset_sequence(payload, preset)
        record, core_region, display_region = self._sequence_geometry(payload)
        raw_settings = payload.get("deletion", {})
        if preset_id != "custom":
            preset_settings = dict(preset.get("deletion") or {})
            if "batch_size" not in raw_settings:
                model_device = adapter.metadata.device
                preset_settings["batch_size"] = (
                    32
                    if model_device.startswith("cuda")
                    else 8
                    if model_device.startswith("mps")
                    else 4
                )
            raw_settings = {
                **preset_settings,
                **raw_settings,
                "output_head": preset["analysis_model"]["output_head"],
            }
        settings = DeletionSettings(
            deletion_size=int(raw_settings.get("deletion_size", 15)),
            stride=int(raw_settings.get("stride", 1)),
            batch_size=int(raw_settings.get("batch_size", 8)),
            smoothing_window=int(raw_settings.get("smoothing_window", 1)),
            perturbation=str(raw_settings.get("perturbation", "real")),
            output_head=str(raw_settings.get("output_head", "regression")),
            scan_scope=str(raw_settings.get("scan_scope", "context")),
        )
        settings.validate(record.length)
        observed = self._observed_atac(
            (preset or {}).get("observed_track")
            if preset_id != "custom"
            else payload.get("observed_track"),
            record,
        )
        reference_config = (
            (preset or {}).get("profile_reference")
            if preset_id != "custom"
            else payload.get("profile_reference")
        ) or {}
        reference_profile = self._cached_column(
            str(reference_config.get("path", "")),
            str(reference_config.get("column", "observed_prediction")),
            record,
        )

        def task(progress, cancel_event):
            return DeletionAnalyzer(adapter).run(
                record,
                settings,
                progress=progress,
                cancel_event=cancel_event,
                core_region=core_region,
                display_region=display_region,
                profile_adapter=profile_adapter,
                observed_atac=observed,
                reference_profile=reference_profile,
                reference_profile_source=(
                    str(reference_config.get("path"))
                    if reference_profile is not None
                    else None
                ),
                preset_id=preset_id,
            )

        return self.jobs.submit(task).public()

    def result(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job.status != "complete" or not job.result_path:
            raise RuntimeError(f"Run {job_id} is not complete")
        return json.loads(Path(job.result_path).read_text())
