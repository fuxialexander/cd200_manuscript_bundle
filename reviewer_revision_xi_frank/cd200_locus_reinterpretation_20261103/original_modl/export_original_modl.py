#!/usr/bin/env python3
"""CPU export of original MoDL checkpoints and selected deletion checks.

Preserves native4096 regression geometry and original S5B8192 geometry.
The additional2114 input is explicitly an inference-context adaptation.
No new calibration, fitting, shared-unit count loss or predictive benchmark.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import convolve1d

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORIGINAL = Path("/home/xf2217/Repos/motif_chip_model")
sys.path.insert(0, str(ORIGINAL))
from atac_prediction.dna_atac_lightning_module import DNAATACMotifCNN
from atac_diff_prediction.bias_model import BiasModel, DifferentialHead
from atac_diff_prediction.models import Phase3Model

cache_path = REPO / "reviewer_revision_xi_frank/mouse_chr16_benchmark/prepare_mouse_cache.py"
spec = importlib.util.spec_from_file_location("mouse_cache_source", cache_path)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)
DenseZarrReader = cache_module.DenseZarrReader
LOCI = {"P1": (45383863, 45384762), "P2": (45356164, 45357191), "promoter": (45408000, 45409426)}
PEAKS_S5B = {"P1": (45383001, 45385049), "P2": LOCI["P2"], "promoter": (45407695, 45409743)}
REG_CKPT = REPO / "analysis/checkpoints/shapescale_loss_scca_pos_with_count_head_fixed_motif_curriculum_from_scratch-epoch=105-val_loss=6.76.ckpt"
DIFF_CKPT = REPO / "analysis/checkpoints/ATAC-Sequential-20250919-022148_phase3_final_model.ckpt"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def table(path, rows):
    with Path(path).open("x") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def clean(state):
    return {k.replace("_orig_mod.", "").removeprefix("model."): v for k, v in state.items()}


def bare_model(state):
    state = clean(state)
    motif = state["motif_conv.weight"].permute(0, 2, 1).numpy()
    model = DNAATACMotifCNN(motif, num_atac_samples=1)
    model.load_state_dict(state, strict=True)
    return model.eval()


def models():
    reg_ck = torch.load(REG_CKPT, map_location="cpu", weights_only=True)
    reg = bare_model(reg_ck["state_dict"])
    ck = torch.load(DIFF_CKPT, map_location="cpu", weights_only=True)
    main = bare_model(ck["main_model_state_dict"])
    bs = clean(ck["bias_model_state_dict"])
    bias = BiasModel(num_atac_samples=1, num_filters=bs["tn5_conv.weight"].shape[0],
                     filter_size=bs["tn5_conv.weight"].shape[2], use_learnable_kernels=True)
    bias.load_state_dict(bs, strict=True)
    ds = clean(ck["differential_head_state_dict"])
    diff = DifferentialHead(input_channels=ds["diff_blocks.0.weight"].shape[1], num_conditions=ds["diff_blocks.6.weight"].shape[0])
    diff.load_state_dict(ds, strict=True)
    phase3 = Phase3Model(main, bias, diff).eval()
    full = clean(ck["full_model_state_dict"])
    phase3.load_state_dict(full, strict=True)
    if not all(torch.equal(v, full[k]) for k, v in phase3.state_dict().items()):
        raise ValueError("Assembled full phase3 state differs from saved full state")
    return reg, phase3


def sequence(genome, center, length):
    left = center - length // 2
    return torch.from_numpy(genome.read("chr16", left, left + length).T.copy()).float()


def forward(model, x):
    with torch.no_grad():
        out = model(x.unsqueeze(0))
    if not all(bool(torch.isfinite(v).all()) for v in out.values()):
        raise ValueError("Nonfinite original checkpoint output")
    return {k: v.detach().cpu() for k, v in out.items()}


def probability(profile):
    p = np.asarray(profile, dtype=np.float64)
    if np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("Invalid shape profile")
    return p / p.sum()


def archived_paths(locus):
    a, b = LOCI[locus]
    folder = REPO / "analysis/results/cd200_analysis_new"
    if locus == "promoter":
        folder = REPO / "analysis/results/cd200_promoter_analysis/regression_profile_overlay"
    stem = f"chr16_{a}_{b}_deletion"
    return {"regression": folder / f"{stem}_regression_nucleotide.csv", "count": folder / f"{stem}_count_nucleotide.csv"}


def contrast(pos, neg, lp, ln):
    return float(np.log10(1 + pos / lp * 1e5) - np.log10(1 + neg / ln * 1e5))


def deletion_position_check(model, baseline, x, window_left, positions, archive, heads, core_region):
    # Each nucleotide receives1/15 of every15bp deletion covering that base.
    # Real deletion shortens input4096 to4081; no masking or repadding.
    starts = sorted({s for p in positions for s in range(max(0, p - window_left - 14), min(p - window_left, 4081) + 1)})
    effects = {head: {} for head in heads}
    core_start, core_end = core_region
    ref_max = float(baseline["regression"][..., core_start:core_end].max())
    ref_count = float(baseline["count"].item())
    ref_diff = float(baseline["differential"].item()) if "differential" in baseline else None
    for first in range(0, len(starts), 4):
        group = starts[first:first + 4]
        alt = torch.stack([torch.cat([x[:, :s], x[:, s + 15:]], dim=-1) for s in group])
        with torch.no_grad():
            out = model(alt)
        for i, s in enumerate(group):
            if "regression" in heads:
                effects["regression"][s] = ref_max - float(out["regression"][i, ..., core_start:core_end].max())
            if "count" in heads:
                effects["count"][s] = float(np.log2((float(out["count"][i].item()) + 1) / (ref_count + 1)))
            if "differential" in heads:
                effects["differential"][s] = float(out["differential"][i].item()) - ref_diff
    result = []
    for head in heads:
        saved = archive[head].set_index("position")["effect"]
        for p in positions:
            i = p - window_left
            value = sum(effects[head][s] / 15 for s in range(max(0, i - 14), min(i, 4081) + 1))
            old = float(saved.loc[p])
            result.append({"head": head, "position": p, "archived_distributed_effect": old,
                           "fresh_distributed_effect": value, "absolute_difference": abs(old - value),
                           "covering_deletions": min(i, 4081) - max(0, i - 14) + 1})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE / "results")
    p.add_argument("--mouse-cache", type=Path, default=Path("/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/cache"))
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; preserve prior results and choose a fresh directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    reg, phase3 = models()
    genome = DenseZarrReader(REPO / "data/zarr/mm10.zarr/chrs", "u1", columns=4)
    signal = REPO / "data/zarr/cd200_scc_dense_new.zarr"
    pos = DenseZarrReader(signal / "SCCApos.cd200_scc/chrs", "<u2")
    neg = DenseZarrReader(signal / "SCCAneg.cd200_scc/chrs", "<u2")
    args.output.mkdir(parents=True)
    archive_folder = args.output / "archived_deletion_tracks"
    archive_folder.mkdir()
    inputs = {str(REG_CKPT): digest(REG_CKPT), str(DIFF_CKPT): digest(DIFF_CKPT), str(Path(__file__)): digest(__file__), str(cache_path): digest(cache_path)}
    for path in [ORIGINAL / "atac_prediction/dna_atac_lightning_module.py", ORIGINAL / "atac_diff_prediction/models.py",
                 ORIGINAL / "atac_diff_prediction/bias_model.py", REPO / "src/mutation_analysis/dataset.py",
                 REPO / "src/mutation_analysis/analyzer.py", REPO / "src/mutation_analysis/metrics.py",
                 REPO / "analysis/mutation_analysis_cli.py",
                 REPO / "analysis/plot_cd200_profile_corr_and_diff_logfc.py",
                 Path("/home/xf2217/Repos/Caesarion/src/caesar/io/normalizer.py"),
                 Path("/home/xf2217/Repos/Caesarion/src/caesar/io/genomic_io.py")]:
        inputs[str(path)] = digest(path)
    native, audited, checks, copied = {}, [], [], []
    for locus, (a, b) in LOCI.items():
        center = (a + b) // 2
        left = center - 2048
        x = sequence(genome, center, 4096)
        out = forward(reg, x)
        raw = pos.read("chr16", left, left + 4096)
        observed = convolve1d(raw.astype(np.float32), np.ones(20) / 20, mode="nearest")
        profile = out["regression"][0, 0].numpy()
        native[locus] = {"positions": np.arange(left, left + 4096), "prediction_profile_native": profile,
                         "observed_overlay_conv20": observed, "observed_stored_counts": raw,
                         "count_head_native_scalar": np.array([out["count"].item()])}
        np.savez_compressed(args.output / f"native4096_{locus}.npz", **native[locus])
        archive = {}
        for head, path in archived_paths(locus).items():
            archive[head] = pd.read_csv(path)
            dest = archive_folder / f"{locus}_{head}_epoch105.csv"
            shutil.copyfile(path, dest)
            inputs[str(path)] = digest(path)
            copied.append({"locus": locus, "head": head, "model": "epoch105_regression", "source": str(path), "copy": str(dest), "sha256": digest(dest)})
        old = archive["regression"]
        if not np.array_equal(old.position, native[locus]["positions"]):
            raise ValueError(f"Archived/fresh coordinate mismatch: {locus}")
        old_pred, old_obs = old.observed_prediction.to_numpy(), old.observed_atac.to_numpy()
        audited.append({"locus": locus, "positions": len(profile),
                        "fresh_vs_archived_profile_mean_absolute_difference": float(np.mean(abs(profile - old_pred))),
                        "fresh_vs_archived_profile_max_absolute_difference": float(np.max(abs(profile - old_pred))),
                        "fresh_vs_archived_profile_pearson": float(np.corrcoef(profile, old_pred)[0, 1]),
                        "reconstructed_overlay_vs_archived_max_absolute_difference": float(np.max(abs(observed - old_obs))),
                        "native_count_head_scalar": float(out["count"].item())})
        # Three selected checks per locus: center, named-family motif and the
        # largest archived absolute effect inside the annotated interval.
        motif = {"P1": 45384070, "P2": 45356660, "promoter": 45408947}[locus]
        in_locus = old[(old.position >= a) & (old.position < b)]
        peak = int(in_locus.loc[in_locus.effect.abs().idxmax(), "position"])
        selected = sorted({center, motif, peak})
        for row in deletion_position_check(reg, out, x, left, selected, archive, ["regression", "count"], (a - left, b - left)):
            checks.append({"locus": locus, "checkpoint": "epoch105", **row})
        if locus == "promoter":
            path = REPO / f"analysis/results/cd200_promoter_analysis/chr16_{a}_{b}_deletion_differential_nucleotide.csv"
            archive_diff = pd.read_csv(path)
            shutil.copyfile(path, archive_folder / "promoter_differential_phase3.csv")
            inputs[str(path)] = digest(path)
            copied.append({"locus": locus, "head": "differential", "model": "phase3", "source": str(path),
                           "copy": str(archive_folder / "promoter_differential_phase3.csv"), "sha256": digest(path)})
            baseline = forward(phase3, x)
            for row in deletion_position_check(phase3, baseline, x, left, selected, {"differential": archive_diff}, ["differential"], (a - left, b - left)):
                checks.append({"locus": locus, "checkpoint": "phase3", **row})
        print(f"Exported native profile and selected deletion checks: {locus}", flush=True)
    table(args.output / "profile_reproduction_audit.tsv", audited)
    table(args.output / "deletion_validation.tsv", checks)
    table(args.output / "archived_deletion_tracks.tsv", copied)

    # Both context choices predict the exact existing seven1000bp windows.
    windows_file = args.mouse_cache / "locus_windows.tsv"
    windows = list(csv.DictReader(windows_file.open(), delimiter="\t"))
    inputs[str(windows_file)] = digest(windows_file)
    arrays = {key: [] for key in ["native4096_profile", "adapted2114_profile", "native4096_shape_probability",
                                 "adapted2114_shape_probability", "observed_stored_counts", "observed_overlay_conv20",
                                 "native4096_count_head_scalar", "adapted2114_count_head_scalar"]}
    for row in windows:
        a, b = int(row["output_start"]), int(row["output_end"])
        center = (a + b) // 2
        for length, label in [(4096, "native4096"), (2114, "adapted2114")]:
            out = forward(reg, sequence(genome, center, length))
            start = length // 2 - 500
            profile = out["regression"][0, 0, start:start + 1000].numpy()
            arrays[f"{label}_profile"].append(profile)
            arrays[f"{label}_shape_probability"].append(probability(profile))
            arrays[f"{label}_count_head_scalar"].append(float(out["count"].item()))
        arrays["observed_stored_counts"].append(pos.read("chr16", a, b))
        # Smooth within the native4096 input, then crop to common output.
        raw = pos.read("chr16", center - 2048, center + 2048).astype(np.float32)
        arrays["observed_overlay_conv20"].append(convolve1d(raw, np.ones(20) / 20, mode="nearest")[1548:2548])
    np.savez_compressed(args.output / "matched1000_profiles.npz", **{key: np.asarray(value) for key, value in arrays.items()})
    table(args.output / "matched1000_windows.tsv", windows)

    bed_file = REPO / "training/data/scc_all_peaks.deduplicated.bed"
    bed = pd.read_csv(bed_file, sep="\t")
    lp, ln = float(bed.SumValue_scca_pos.sum()), float(bed.SumValue_scca_neg.sum())
    shrink_file = REPO / "analysis/results/cd200_deletion_vs_chrombpnet_original/diff_head_shrinkage_factor.csv"
    scale = float(pd.read_csv(shrink_file).inverse_shrinkage_mean_ratio.iloc[0])
    archived_s5b = REPO / "analysis/results/cd200_deletion_vs_chrombpnet_original/diff_scca_logfc_summary.csv"
    archived = pd.read_csv(archived_s5b)
    for path in [bed_file, shrink_file, archived_s5b]:
        inputs[str(path)] = digest(path)
    differentials = []
    for locus, (a, b) in LOCI.items():
        center = (a + b) // 2
        out = forward(phase3, sequence(genome, center, 8192))
        value = float(out["differential"].item())
        peak_a, peak_b = PEAKS_S5B[locus]
        old = archived[archived.region == {"P1": "Enhancer 1", "P2": "Enhancer 2", "promoter": "Promoter"}[locus]].iloc[0]
        hits = bed[(bed.Chromosome == "chr16") & (bed.Start <= peak_b) & (bed.End >= peak_a)]
        table_row = hits.iloc[0] if len(hits) else None
        sp, sn = float(pos.read("chr16", peak_a, peak_b).sum()), float(neg.read("chr16", peak_a, peak_b).sum())
        cp, cn = float(pos.read("chr16", center - 1024, center + 1024).sum()), float(neg.read("chr16", center - 1024, center + 1024).sum())
        differentials.append({"locus": locus, "center": center, "phase3_input_bp": 8192,
            "phase3_native_differential_unscaled": value, "phase3_archived_differential_unscaled": float(old.our_differential_head),
            "fresh_vs_archived_absolute_difference": abs(value - float(old.our_differential_head)),
            "legacy_scaled_differential_for_historical_display_only": value * scale, "legacy_scale": scale,
            "archived_observed_Diff_scca": float(old.observed_Diff_scca), "archived_observed_source": old.observed_source,
            "old_peak_start": peak_a, "old_peak_end": peak_b, "old_peak_current_stored_pos_sum": sp, "old_peak_current_stored_neg_sum": sn,
            "old_peak_current_signal_recomputed_contrast": contrast(sp, sn, lp, ln),
            "BED_row_SumValue_pos": float(table_row.SumValue_scca_pos) if table_row is not None else None,
            "BED_row_SumValue_neg": float(table_row.SumValue_scca_neg) if table_row is not None else None,
            "BED_row_sums_recomputed_contrast": contrast(float(table_row.SumValue_scca_pos), float(table_row.SumValue_scca_neg), lp, ln) if table_row is not None else None,
            "new_center2048_start": center - 1024, "new_center2048_end": center + 1024,
            "new_center2048_stored_pos_sum": cp, "new_center2048_stored_neg_sum": cn,
            "new_center2048_redefined_contrast": contrast(cp, cn, lp, ln), "library_denominator_pos_BED_sum": lp,
            "library_denominator_neg_BED_sum": ln, "archived_chrombpnet_contrast": float(old.chrombpnet_diff),
            "archived_chrombpnet_pos_sum": float(old.chrombpnet_sum_pos), "archived_chrombpnet_neg_sum": float(old.chrombpnet_sum_neg)})
        np.savez_compressed(args.output / f"phase3_8192_{locus}.npz", regression_native=out["regression"][0, 0].numpy(),
                            count_native=np.array([out["count"].item()]), differential_native=np.array([value]))
    table(args.output / "s5b_original_and_redefined_contrasts.tsv", differentials)
    manifest = {"status": "complete", "device": "cpu", "torch_version": str(torch.__version__), "cpu_threads": args.threads,
        "original_module": sys.modules[DNAATACMotifCNN.__module__].__file__, "model_weights_loaded": "strict=True for regression, phase3 components and assembled full state",
        "regression_native_input_bp": 4096, "phase3_original_S5B_input_bp": 8192, "matched_output_bp": 1000,
        "matched_input_choices": [4096, 2114], "input2114_status": "inference-context adaptation; no retraining",
        "original_overlay_target": "20bp mean convolution of stored ATAC counts, nearest-edge padding; normalize_factor1e8 is identity in FactorNormalizer",
        "native_scalar_count_units": "original positive scalar count-head output; do not treat as raw counts or common1000bp ln1p counts",
        "normalization_of_exported_shapes": "divide positive profile by its sum over the exported1000bp window",
        "differential_caveat": "Native phase3 learns supplied Diff_scca, not an independently identifiable pair of pos/neg profile heads. Current rawZarr or BED-row sums with finalBED library sums do not reproduce saved Diff_scca. Redefined center2048 contrast is separate and is not a verified native model target.",
        "no_shared_unit_count_loss_computed": True, "no_new_fitting_or_calibration": True,
        "old_scale_status": "legacy2123-positive-peak inverse shrinkage factor retained only for historical display",
        "deletion_definition": {"length": 15, "mode": "real shortened sequence, no padding", "profile_effect": "max(ref profile within annotated locus core)-max(deleted profile within same sequence-index core); matches region CLI default",
             "count_effect": "log2((alt_native_count+1)/(ref_native_count+1))", "differential_effect": "alt_native_diff-ref_native_diff",
             "per_base_distribution": "sum of1/15 of all deletion-window effects covering each nucleotide", "sampled_positions_only": True},
        "sources_sha256": inputs, "zarr_accessed_sources": {"genome": genome.sources, "SCCApos": pos.sources, "SCCAneg": neg.sources}}
    manifest["outputs_sha256"] = {str(f.relative_to(args.output)): digest(f) for f in args.output.rglob("*") if f.is_file()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
