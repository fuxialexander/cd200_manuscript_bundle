#!/usr/bin/env python3
"""
Compare our model vs ChromBPNet on CD200 regions:

1) Profile scatter panels: predicted vs observed ATAC at ChromBPNet-covered bases
   (ChromBPNet uses a 20 bp rolling sum; axes are independently scaled).
2) Diff_scca-style logFC barplot: observed vs our differential head vs ChromBPNet
   profile-sum contrast (SCCApos vs SCCAneg).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import torch
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr, spearmanr

from caesar.io.zarr_io import CelltypeDenseZarrIO, SequenceDenseZarrIO
from mutation_analysis.dataset import MutationDataset
from mutation_analysis.model import load_model


_BUNDLE = Path(__file__).resolve().parents[1]
_ANALYSIS_NEW = _BUNDLE / "analysis/results/cd200_analysis_new"
_PROMOTER = _BUNDLE / "analysis/results/cd200_promoter_analysis"

REGIONS = [
    {
        "name": "Enhancer 2",
        "label": "chr16:45356164-45357191",
        "start": 45356164,
        "end": 45357191,
        "csv": _ANALYSIS_NEW
        / "chr16_45356164_45357191_deletion_regression_nucleotide.csv",
        # no overlapping training peak; use annotated bounds
        "peak_start": 45356164,
        "peak_end": 45357191,
    },
    {
        "name": "Enhancer 1",
        "label": "chr16:45383863-45384762",
        "start": 45383863,
        "end": 45384762,
        "csv": _ANALYSIS_NEW
        / "chr16_45383863_45384762_deletion_regression_nucleotide.csv",
        "peak_start": 45383001,
        "peak_end": 45385049,
    },
    {
        "name": "Promoter",
        "label": "chr16:45408000-45409426",
        "start": 45408000,
        "end": 45409426,
        "csv": _PROMOTER
        / "regression_profile_overlay"
        / "chr16_45408000_45409426_deletion_regression_nucleotide.csv",
        "peak_start": 45407695,
        "peak_end": 45409743,
    },
]


def _bw_values(bw: pyBigWig.pyBigWig, chrom: str, start: int, end_inclusive: int) -> np.ndarray:
    raw = bw.values(chrom, start, end_inclusive + 1)
    if raw is None:
        return np.zeros(end_inclusive - start + 1, dtype=float)
    return np.array(
        [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in raw],
        dtype=float,
    )


def _corrs(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])


def _diffscca(sum_pos: float, sum_neg: float, lib_pos: float, lib_neg: float) -> float:
    """Match training Diff_scca: log10(CPM+1) pos − log10(CPM+1) neg."""
    return float(
        np.log10(sum_pos / lib_pos * 1e5 + 1.0) - np.log10(sum_neg / lib_neg * 1e5 + 1.0)
    )


def _log10_sum_diff(sum_pos: float, sum_neg: float) -> float:
    return float(np.log10(sum_pos + 1.0) - np.log10(sum_neg + 1.0))


def _scale_sum_to_2048(raw_sum: float, window_len: int, target_len: int = 2048) -> float:
    """Convert a window sum to an equivalent 2048-bp peak sum via mean density."""
    if window_len <= 0:
        return 0.0
    return float(raw_sum) / float(window_len) * float(target_len)


def load_bed_diff_and_libs(bed_path: Path) -> tuple[pd.DataFrame, float, float]:
    bed = pd.read_csv(bed_path, sep="\t")
    lib_pos = float(bed["SumValue_scca_pos"].sum())
    lib_neg = float(bed["SumValue_scca_neg"].sum())
    return bed, lib_pos, lib_neg


def observed_diff_for_region(
    bed: pd.DataFrame,
    *,
    peak_start: int,
    peak_end: int,
    lib_pos: float,
    lib_neg: float,
    atac_pos: np.ndarray,
    atac_neg: np.ndarray,
) -> tuple[float, str]:
    hit = bed[
        (bed["Chromosome"] == "chr16")
        & (bed["Start"] <= peak_end)
        & (bed["End"] >= peak_start)
    ]
    if not hit.empty:
        row = hit.iloc[0]
        return float(row["Diff_scca"]), "peak_Diff_scca"
    return (
        _diffscca(float(np.nansum(atac_pos)), float(np.nansum(atac_neg)), lib_pos, lib_neg),
        "zarr_Diff_scca",
    )


def predict_differential(
    model: torch.nn.Module,
    dataset: MutationDataset,
    *,
    start: int,
    end: int,
    device: str,
) -> float:
    center = (start + end) // 2
    seq = dataset.get_sequence("chr16", center - 4096, center + 4096).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(seq)
    return float(out["differential"].detach().cpu().reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    bundle = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output-dir",
        default=str(bundle / "analysis/results/cd200_deletion_vs_chrombpnet_original"),
    )
    parser.add_argument("--smooth-bp", type=int, default=20)
    parser.add_argument(
        "--phase3-checkpoint",
        default=str(
            bundle
            / "analysis/checkpoints/ATAC-Sequential-20250919-022148_phase3_final_model.ckpt"
        ),
    )
    parser.add_argument(
        "--bed",
        default=str(bundle / "training/data/scc_all_peaks.deduplicated.bed"),
    )
    parser.add_argument(
        "--chrombpnet-pos-bw",
        default=os.environ.get(
            "CHROMBPNET_POS_BW",
            "/mnt/storage/cd200/chrombpnet_moa/SCCApos_chrombpnet.bw",
        ),
    )
    parser.add_argument(
        "--chrombpnet-neg-bw",
        default=os.environ.get(
            "CHROMBPNET_NEG_BW",
            "/mnt/storage/cd200/chrombpnet_moa/SCCAneg_chrombpnet.bw",
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--diff-inverse-shrinkage",
        type=float,
        default=None,
        help=(
            "Multiply differential-head predictions by this factor. "
            "Default: load from diff_head_shrinkage_factor.csv or 3.307 "
            "(inverse of mean pred/obs on Diff_scca>=0.05 peaks)."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shrink_csv = out_dir / "diff_head_shrinkage_factor.csv"
    if args.diff_inverse_shrinkage is not None:
        inv_shrink = float(args.diff_inverse_shrinkage)
    elif shrink_csv.is_file():
        inv_shrink = float(
            pd.read_csv(shrink_csv).iloc[0]["inverse_shrinkage_mean_ratio"]
        )
    else:
        inv_shrink = 3.307237044473427
    print(f"Differential-head inverse shrinkage factor: {inv_shrink:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _heads = load_model(args.phase3_checkpoint, device=device)
    model.eval()
    genome = SequenceDenseZarrIO(str(bundle / "data/zarr/mm10.zarr"), mode="r")
    dataset = MutationDataset(genome_io=genome)
    atac = CelltypeDenseZarrIO(str(bundle / "data/zarr/cd200_scc_dense_new.zarr"), mode="r")
    atac_pos_io = atac.subset(["SCCApos.cd200_scc"])
    atac_neg_io = atac.subset(["SCCAneg.cd200_scc"])
    bed, lib_pos, lib_neg = load_bed_diff_and_libs(Path(args.bed))

    bw_pos = pyBigWig.open(args.chrombpnet_pos_bw)
    bw_neg = pyBigWig.open(args.chrombpnet_neg_bw)

    scatter_rows = []
    logfc_rows = []

    plt.rcParams["font.family"] = "Arial"
    win = int(args.smooth_bp)

    for region in REGIONS:
        df = pd.read_csv(region["csv"])
        pred_col = (
            "observed_prediction"
            if "observed_prediction" in df.columns
            else "baseline_prediction"
        )
        pos = df["position"].to_numpy(dtype=int)
        p0, p1 = int(pos.min()), int(pos.max())
        our = df[pred_col].to_numpy(dtype=float)
        obs = df["observed_atac"].to_numpy(dtype=float)

        cb = _bw_values(bw_pos, "chr16", p0, p1)
        if len(cb) != len(pos):
            cb = np.array(
                [_bw_values(bw_pos, "chr16", int(p), int(p))[0] for p in pos], dtype=float
            )

        cov = np.isfinite(cb) & (cb != 0.0) & np.isfinite(our) & np.isfinite(obs)
        # ChromBPNet: 20 bp rolling SUM (not mean). Observed ATAC uses the same
        # window sum so the correlation is on matched aggregates.
        cb_s = uniform_filter1d(cb.astype(float), size=win, mode="nearest") * win
        obs_s = uniform_filter1d(obs.astype(float), size=win, mode="nearest") * win

        our_r, our_rho = _corrs(our[cov], obs[cov])
        cb_r, cb_rho = _corrs(cb_s[cov], obs_s[cov])

        # logFC / Diff_scca
        peak_s, peak_e = int(region["peak_start"]), int(region["peak_end"])
        atac_pos = np.asarray(
            atac_pos_io.get_track("chr16", peak_s, peak_e).tracks["default_sum"], dtype=float
        )
        atac_neg = np.asarray(
            atac_neg_io.get_track("chr16", peak_s, peak_e).tracks["default_sum"], dtype=float
        )

        obs_diff, obs_src = observed_diff_for_region(
            bed,
            peak_start=peak_s,
            peak_end=peak_e,
            lib_pos=lib_pos,
            lib_neg=lib_neg,
            atac_pos=atac_pos,
            atac_neg=atac_neg,
        )
        our_diff = predict_differential(
            model,
            dataset,
            start=region["start"],
            end=region["end"],
            device=device,
        )
        our_diff_scaled = our_diff * inv_shrink

        cb_pos = _bw_values(bw_pos, "chr16", peak_s, peak_e - 1)
        cb_neg = _bw_values(bw_neg, "chr16", peak_s, peak_e - 1)
        window_len = int(peak_e - peak_s)
        # Include zeros across the full window so density matches peak SumValue definition.
        cb_sum_pos = float(np.sum(cb_pos))
        cb_sum_neg = float(np.sum(cb_neg))
        cb_sum_pos_2048 = _scale_sum_to_2048(cb_sum_pos, window_len, 2048)
        cb_sum_neg_2048 = _scale_sum_to_2048(cb_sum_neg, window_len, 2048)

        # Same Diff_scca formula as training: peak-sum CPM with BED peak libraries.
        if cb_sum_pos_2048 > 0 and cb_sum_neg_2048 > 0:
            cb_diff = _diffscca(cb_sum_pos_2048, cb_sum_neg_2048, lib_pos, lib_neg)
            cb_method = "profile_sum_2048_peak_sum_cpm"
        else:
            cb_diff = float("nan")
            cb_method = "undefined"

        cb_diff_cpm = cb_diff
        cb_diff_raw = _log10_sum_diff(cb_sum_pos, cb_sum_neg)

        scatter_rows.append(
            {
                "region": region["name"],
                "n": int(cov.sum()),
                "our_pearson": our_r,
                "our_spearman": our_rho,
                "chrombpnet_pearson_20bp": cb_r,
                "chrombpnet_spearman_20bp": cb_rho,
            }
        )
        logfc_rows.append(
            {
                "region": region["name"],
                "label": region["label"],
                "observed_Diff_scca": obs_diff,
                "observed_source": obs_src,
                "our_differential_head": our_diff,
                "our_differential_head_scaled": our_diff_scaled,
                "inverse_shrinkage": inv_shrink,
                "chrombpnet_diff": cb_diff,
                "chrombpnet_method": cb_method,
                "chrombpnet_sum_pos": cb_sum_pos,
                "chrombpnet_sum_neg": cb_sum_neg,
                "chrombpnet_sum_pos_2048": cb_sum_pos_2048,
                "chrombpnet_sum_neg_2048": cb_sum_neg_2048,
                "chrombpnet_window_len": window_len,
                "peak_sum_lib_pos": lib_pos,
                "peak_sum_lib_neg": lib_neg,
                "chrombpnet_diff_cpm_unscaled": cb_diff_cpm,
                "chrombpnet_diff_log10sum": cb_diff_raw,
            }
        )

    fig, (ax_pearson, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(5, 5),
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.45},
    )
    names = [r["name"] for r in REGIONS]
    x = np.arange(len(names))
    width = 0.32
    gap = 0.0  # no horizontal padding between bars within a group

    our_r_vals = [row["our_pearson"] for row in scatter_rows]
    cb_r_vals = [row["chrombpnet_pearson_20bp"] for row in scatter_rows]
    bars_our = ax_pearson.bar(
        x - width / 2,
        our_r_vals,
        width,
        label="MoDL",
        color="#7DAEE0",
    )
    bars_cb = ax_pearson.bar(
        x + width / 2,
        cb_r_vals,
        width,
        label="ChromBPNet",
        color="#EA8379",
    )
    ax_pearson.set_ylim(0, 1.12)
    ax_pearson.set_xlim(-0.6, len(names) - 0.4)
    ax_pearson.set_xticks(x)
    ax_pearson.set_xticklabels(names, fontsize=9)
    ax_pearson.set_ylabel("Pearson r", fontsize=9)
    ax_pearson.legend(
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.5,
        handlelength=1.2,
    )
    ax_pearson.tick_params(labelsize=8)
    ax_pearson.spines["top"].set_visible(False)
    ax_pearson.spines["right"].set_visible(False)
    for bars in (bars_our, bars_cb):
        for bar in bars:
            h = bar.get_height()
            ax_pearson.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    width = 0.25
    gap = 0.0
    obs_vals = [row["observed_Diff_scca"] for row in logfc_rows]
    our_vals = [row["our_differential_head_scaled"] for row in logfc_rows]
    cb_vals = [row["chrombpnet_diff"] for row in logfc_rows]
    offsets = np.array([-width, 0.0, width])

    ax_bar.bar(x + offsets[0], obs_vals, width, label="Observed", color="#B395BD")
    ax_bar.bar(x + offsets[1], our_vals, width, label="MoDL", color="#7DAEE0")
    ax_bar.bar(x + offsets[2], cb_vals, width, label="ChromBPNet", color="#EA8379")
    ax_bar.axhline(0, color="0.7", lw=0.8)
    ax_bar.set_xlim(-0.7, len(names) - 0.3)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, fontsize=9)
    ax_bar.set_ylabel("Accessibility contrast", fontsize=9)
    y_min, y_max = ax_bar.get_ylim()
    pad = 0.08 * (y_max - y_min + 1e-8)
    ax_bar.set_ylim(y_min - pad, y_max + pad * 1.8)
    ax_bar.legend(
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=1.5,
        handlelength=1.2,
    )
    ax_bar.tick_params(labelsize=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.subplots_adjust(top=0.92, hspace=0.5)
    out_png = out_dir / "profile_scatter_and_diff_logfc.png"
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    scatter_csv = out_dir / "profile_correlation_summary.csv"
    pd.DataFrame(scatter_rows).to_csv(scatter_csv, index=False)
    logfc_csv = out_dir / "diff_scca_logfc_summary.csv"
    pd.DataFrame(logfc_rows).to_csv(logfc_csv, index=False)
    print(f"Saved {scatter_csv}")
    print(f"Saved {logfc_csv}")
    print(pd.DataFrame(logfc_rows).to_string(index=False))

    bw_pos.close()
    bw_neg.close()


if __name__ == "__main__":
    main()
