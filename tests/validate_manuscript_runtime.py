"""Run the real dual-checkpoint promoter preset and compare saved figure inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_studio.annotations import MotifAnnotationCatalog
from model_studio.domain import DeletionSettings
from model_studio.perturbations import DeletionAnalyzer
from model_studio.service import ModelStudioService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--scan-scope",
        choices=["display", "context"],
        default="display",
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    service = ModelStudioService(bundle / "model_studio")
    preset = service._preset("manuscript_differential_promoter")
    assert preset is not None
    metadata = service.load_model(
        "",
        "",
        args.device,
        preset_id=preset["id"],
    )
    record, core, display = service._sequence_geometry(
        {
            "source": preset["source"],
            "context_bp": preset["context_bp"],
            "display_bp": preset["display_bp"],
        }
    )
    observed = service._observed_atac(preset["observed_track"], record)
    reference_config = preset["profile_reference"]
    reference = service._cached_column(
        reference_config["path"],
        reference_config["column"],
        record,
    )
    result = DeletionAnalyzer(service._adapter).run(
        record,
        DeletionSettings(
            deletion_size=15,
            stride=1,
            batch_size=args.batch_size,
            smoothing_window=1,
            perturbation="real",
            output_head="differential",
            scan_scope=args.scan_scope,
        ),
        core_region=core,
        display_region=display,
        profile_adapter=service._profile_adapter,
        observed_atac=observed,
        reference_profile=reference,
        reference_profile_source=reference_config["path"],
        preset_id=preset["id"],
    )

    analysis_root = bundle / "analysis/results/cd200_promoter_analysis"
    saved_diff = pd.read_csv(
        analysis_root
        / "chr16_45408000_45409426_deletion_differential_nucleotide.csv"
    )
    expected_effect_full = (
        saved_diff["effect"].abs().to_numpy()
        / saved_diff["baseline_prediction"].abs().max()
        * 100
    )
    expected_by_position = dict(
        zip(
            saved_diff["position"].astype(int),
            expected_effect_full,
            strict=True,
        )
    )
    expected_effect = np.asarray(
        [expected_by_position[position] for position in result.nucleotide_positions]
    )
    actual_effect = np.asarray(result.plot_effects)
    effect_r = float(np.corrcoef(actual_effect, expected_effect)[0, 1])
    effect_mae = float(np.mean(np.abs(actual_effect - expected_effect)))

    saved_profile = pd.read_csv(
        analysis_root
        / "regression_profile_overlay/"
        "chr16_45408000_45409426_deletion_regression_nucleotide.csv"
    )
    actual_profile = np.asarray(result.profile["predicted"])
    expected_profile = saved_profile["observed_prediction"].to_numpy()
    profile_r = float(np.corrcoef(actual_profile, expected_profile)[0, 1])
    profile_mae = float(np.mean(np.abs(actual_profile - expected_profile)))
    reference_profile = np.asarray(result.profile["reference"])
    reference_profile_max_delta = float(
        np.max(np.abs(reference_profile - expected_profile))
    )

    catalog = MotifAnnotationCatalog(
        preset["annotations"]["motif_path"],
        chromosome="chr16",
        start=display[0],
        end=display[1],
        rna_path=preset["annotations"]["rna_path"],
        rna_path_2=preset["annotations"]["rna_path_2"],
        gene_list_path=preset["annotations"]["gene_list_path"],
    )
    filtered = catalog.filter(
        {
            "nucleotide_positions": result.nucleotide_positions,
            "plot_effects": result.plot_effects,
        },
        mode="gene_list",
        min_effect_pct=10,
    )
    group_labels = [group["label"] for group in filtered["groups"]]

    summary = {
        "device": metadata.device,
        "context": result.regions["context"]["label"],
        "core": result.regions["core"]["label"],
        "display": result.regions["display"]["label"],
        "effect_max_pct": float(actual_effect.max()),
        "saved_effect_max_pct": float(expected_effect.max()),
        "effect_correlation": effect_r,
        "effect_mean_absolute_delta_pp": effect_mae,
        "profile_correlation": profile_r,
        "profile_mean_absolute_delta": profile_mae,
        "saved_reference_max_delta": reference_profile_max_delta,
        "motif_rows": group_labels,
        "motif_full_labels": [
            group["full_label"] for group in filtered["groups"]
        ],
        "elapsed_seconds": result.elapsed_seconds,
        "batch_size": args.batch_size,
        "scan_scope": args.scan_scope,
        "scanned_windows": result.settings["scanned_window_count"],
        "full_context_windows": result.settings["full_context_window_count"],
    }
    print(json.dumps(summary, indent=2))
    if effect_r < 0.999:
        raise SystemExit("Differential curve correlation is below 0.999")
    if effect_mae > 0.1:
        raise SystemExit("Differential curve mean delta exceeds 0.1 percentage point")
    if reference_profile_max_delta > 1e-6:
        raise SystemExit("Saved profile reference does not match the figure input")
    if group_labels != ["SNAI", "CEBPB"]:
        raise SystemExit(f"Unexpected manuscript motif rows: {group_labels}")


if __name__ == "__main__":
    main()
