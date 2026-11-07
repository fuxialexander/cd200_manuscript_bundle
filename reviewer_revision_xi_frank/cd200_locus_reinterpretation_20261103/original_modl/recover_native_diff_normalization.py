#!/usr/bin/env python3
"""Recover the original Diff_scca transform algebraically from retained columns.

This recovers constants encoded in each row; it does not fit a calibration or
use newly predicted/test outcomes. Current signal and historical row sums remain
separate observations even after they share the recovered transformation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(frame, state):
    sums = frame[f"SumValue_scca_{state}"].to_numpy(float)
    logs = frame[f"logSumValue_scca_{state}"].to_numpy(float)
    mask = (sums > 0) & (logs > 0)
    inverses = sums[mask] * 1e5 / np.expm1(logs[mask] * np.log(10))
    denominator = int(round(float(np.median(inverses))))
    max_error = float(np.max(abs(inverses - denominator)))
    if max_error > 1e-3:
        raise ValueError("Retained rows do not encode one constant normalization denominator")
    rebuilt_logs = np.log10(1 + sums / denominator * 1e5)
    return denominator, {
        "rows_used": int(mask.sum()), "denominator": denominator,
        "inverse_denominator_min": float(inverses.min()), "inverse_denominator_max": float(inverses.max()),
        "maximum_inverse_denominator_absolute_residual": max_error,
        "maximum_log_column_reconstruction_absolute_residual": float(np.max(abs(rebuilt_logs - logs))),
        "current_final_BED_SumValue_sum": float(sums.sum()),
    }


def transform(pos, neg, lp, ln):
    return np.log10(1 + np.asarray(pos) / lp * 1e5) - np.log10(1 + np.asarray(neg) / ln * 1e5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-exports", type=Path, default=HERE / "results_core_reproduction")
    args = p.parse_args()
    output = args.original_exports
    audit_path = output / "native_diff_normalization_audit.json"
    table_path = output / "native_diff_recovered_contrasts.tsv"
    if audit_path.exists() or table_path.exists():
        p.error("Recovery artifacts already exist; preserve them")
    bed_path = REPO / "training/data/scc_all_peaks.deduplicated.bed"
    frame = pd.read_csv(bed_path, sep="\t")
    lp, pa = recover(frame, "pos")
    ln, na = recover(frame, "neg")
    log_identity = frame.logSumValue_scca_pos - frame.logSumValue_scca_neg
    reconstructed = transform(frame.SumValue_scca_pos, frame.SumValue_scca_neg, lp, ln)
    source_path = output / "s5b_original_and_redefined_contrasts.tsv"
    rows = pd.read_csv(source_path, sep="\t")
    rows["recovered_native_library_pos"] = lp
    rows["recovered_native_library_neg"] = ln
    rows["BED_row_reconstructed_native_Diff_scca"] = transform(rows.BED_row_SumValue_pos, rows.BED_row_SumValue_neg, lp, ln)
    rows["current_signal_old_peak_native_transform"] = transform(rows.old_peak_current_stored_pos_sum, rows.old_peak_current_stored_neg_sum, lp, ln)
    rows["current_signal_center2048_native_transform"] = transform(rows.new_center2048_stored_pos_sum, rows.new_center2048_stored_neg_sum, lp, ln)
    rows["archived_chrombpnet_old_peak_recovered_native_transform"] = transform(rows.archived_chrombpnet_pos_sum, rows.archived_chrombpnet_neg_sum, lp, ln)
    rows["historical_observed_uses_native_denominator"] = rows.archived_observed_source == "peak_Diff_scca"
    rows.to_csv(table_path, sep="\t", index=False)
    original_source = Path("/home/xf2217/Repos/motif_chip_model/atac_prediction/cd200_plot.py")
    audit = {
        "status": "original_native_Diff_scca_transform_recovered",
        "method": "For every row with positive SumValue/logSumValue, L=SumValue*1e5/expm1(logSumValue*ln(10)); all rows independently encode the same integerL within numerical precision.",
        "no_fitted_calibration": True, "normalization_formula": "log10(1+pos_sum/Lpos*1e5)-log10(1+neg_sum/Lneg*1e5)",
        "positive_state": pa, "negative_state": na, "all_rows": len(frame),
        "log_pos_minus_log_neg_vs_Diff_scca_maximum_absolute_residual": float(np.max(abs(log_identity - frame.Diff_scca))),
        "recovered_formula_vs_Diff_scca_maximum_absolute_residual": float(np.max(abs(reconstructed - frame.Diff_scca))),
        "source_trace": {"path": str(original_source), "log_normalization_lines": [60, 62], "Diff_scca_line": 69, "later_signal_filter_line": 89,
                         "interpretation": "Log columns and Diff are computed before later filtering/deduplication; summing the final BED is not the original normalization denominator."},
        "historical_S5B_mixed_normalizers": "P1/promoter used stored native BED labels; P2 had no overlapping BED row and original S5B used a fallback with finalBED summed denominators. Recover native units before constructing a uniform revised comparison.",
        "remaining_observation_difference": "Current Zarr sums over the final representative BED intervals differ from retained historical SumValue. Original physical windows/curation of those retained sums remain unverified; recovered units do not make those observations identical.",
        "new_comparison_contract": "New CBP pos/neg predicted sums and current observed sums over the SAME central2048 coordinates may use recovered historical constants. Show retained native training labels separately where available. Original unscaled MoDL differential remains its learned native target; no extra scale is fitted.",
        "legacy_scale": "3.307237044473427 retained only as historical display variant, not needed to recover native normalization",
        "sources_sha256": {str(p): digest(p) for p in [bed_path, source_path, original_source, Path(__file__)]},
        "contrast_table_sha256": digest(table_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"Lpos": lp, "Lneg": ln, "maximum_Diff_residual": audit["recovered_formula_vs_Diff_scca_maximum_absolute_residual"], "table": str(table_path)}, indent=2))


if __name__ == "__main__":
    main()
