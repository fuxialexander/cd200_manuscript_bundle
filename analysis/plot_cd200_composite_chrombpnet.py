#!/usr/bin/env python3
"""
Four-row composite for three CD200 regions:

  1. Deletion effect (same style as plot_cd200_composite_three.py)
  2. Motifs overlapping high deletion-effect positions (expression/DEG-filtered TSV)
  3. ChromBPNet counts_scores at native 1 bp resolution (no smoothing/binning)
  4. Motifs overlapping high |ChromBPNet count| positions (same expression filter)

Column order matches the composite figure:
  1. Promoter chr16:45408000-45409426 (differential deletion scores)
  2. Enhancer chr16:45383863-45384762 (regression)
  3. Enhancer chr16:45356164-45357191 (regression)

Missing ChromBPNet values are padded with 0.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import pyranges
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MultipleLocator

from plot_scan_results import (
    build_overlap_plot_rows,
    merge_motifs,
    prepare_plot_effects,
)


def _chromosome_from_scores_path(scores_path: str) -> str:
    return os.path.basename(scores_path).split("_")[0]


def _window_from_scores(scores_df: pd.DataFrame, center_window_bp: int) -> tuple[int, int]:
    plot_start = int(scores_df["position"].min())
    plot_end = int(scores_df["position"].max())
    center_pos = (plot_start + plot_end) / 2
    half = center_window_bp / 2
    return int(center_pos - half), int(center_pos + half)


def filter_motifs_by_plot_effect(
    motifs_df: pd.DataFrame,
    *,
    chromosome: str,
    scores_df: pd.DataFrame,
    threshold_pct: float | None,
) -> pd.DataFrame:
    """Keep motifs overlapping positions with plot_effect >= threshold_pct.

    Matches plot_cd200_composite_three.py / previous composite motif filtering.
    """
    if motifs_df.empty or threshold_pct is None:
        return motifs_df.copy()

    significant_scores = scores_df[scores_df["plot_effect"] >= float(threshold_pct)]
    if significant_scores.empty:
        return motifs_df.iloc[0:0].copy()

    significant_pr = pyranges.PyRanges(
        chromosomes=[chromosome] * len(significant_scores),
        starts=significant_scores["position"],
        ends=significant_scores["position"] + 1,
    )
    motifs_pr = pyranges.PyRanges(
        motifs_df.rename(columns={"chromosome": "Chromosome", "start": "Start", "end": "End"})
    )
    overlapping = motifs_pr.overlap(significant_pr)
    if overlapping.empty:
        return motifs_df.iloc[0:0].copy()
    return overlapping.df.rename(
        columns={"Chromosome": "chromosome", "Start": "start", "End": "end"}
    )


def load_base_motifs(
    motif_path: str,
    *,
    window_start: int,
    window_end: int,
    merge: bool,
) -> pd.DataFrame:
    motifs_df = pd.read_csv(motif_path, sep="\t")
    if not motifs_df.empty:
        motifs_df = motifs_df[
            (motifs_df["end"] > window_start) & (motifs_df["start"] < window_end)
        ].copy()
    if merge and not motifs_df.empty:
        motifs_df = merge_motifs(motifs_df)
    return motifs_df.dropna(subset=["gene_name"])


def extract_chrombpnet_1bp(
    bw: pyBigWig.pyBigWig,
    chromosome: str,
    window_start: int,
    window_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 1 bp positions and signed scores for [window_start, window_end]."""
    positions = np.arange(window_start, window_end + 1, dtype=int)
    raw = bw.values(chromosome, window_start, window_end + 1)
    if raw is None:
        return positions, np.zeros(len(positions), dtype=float)

    values = np.zeros(len(positions), dtype=float)
    for i, val in enumerate(raw):
        if val is None:
            continue
        val_f = float(val)
        if not np.isnan(val_f):
            values[i] = val_f
    return positions, values


def draw_motifs(ax, motifs_df: pd.DataFrame, scores_df: pd.DataFrame) -> None:
    if motifs_df.empty:
        ax.set_yticks([])
        return
    plot_rows = build_overlap_plot_rows(motifs_df, scores_df)
    row_to_y = {row["cluster_id"]: i for i, row in enumerate(plot_rows)}
    ax.set_yticks(range(len(plot_rows)))

    def _dedup_label(label: str) -> str:
        seen: list[str] = []
        for name in label.split("/"):
            stem = re.sub(r"\d+$", "", name)
            if stem not in seen:
                seen.append(stem)
        return "/".join(seen)

    ax.set_yticklabels([_dedup_label(row["label"]) for row in plot_rows], fontsize=8)
    ax.set_ylim(len(plot_rows) - 0.5, -0.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    for plot_row in plot_rows:
        y_pos = row_to_y[plot_row["cluster_id"]]
        for _, row in plot_row["motifs"].iterrows():
            w = row["end"] - row["start"]
            ax.add_patch(
                patches.Rectangle(
                    (row["start"], y_pos - 0.4),
                    w,
                    0.8,
                    facecolor="grey",
                    edgecolor="grey",
                    linewidth=0.5,
                )
            )


def _symmetric_ylim(values: np.ndarray, *, pad_frac: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return -1.0, 1.0
    max_mag = float(np.max(np.abs(values)))
    if max_mag <= 0:
        return -1.0, 1.0
    pad = max_mag * pad_frac
    limit = max_mag + pad
    return -limit, limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    bundle_root = Path(__file__).resolve().parents[1]
    default_base = bundle_root / "analysis/results/cd200_analysis_new"
    default_prom = bundle_root / "analysis/results/cd200_promoter_analysis"
    default_bw = Path(
        os.environ.get(
            "CHROMBPNET_COUNTS_BW",
            "/mnt/storage/cd200/chrombpnet_moa/SCCA_pos.counts_scores.bw",
        )
    )

    parser.add_argument(
        "--output-file",
        default=str(
            bundle_root
            / "analysis/results/cd200_deletion_vs_chrombpnet_original/"
            "cd200_deletion_vs_chrombpnet_count_scores_composite.png"
        ),
        help="Path for the output PNG.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-scale", type=float, default=1.0)
    parser.add_argument("--center-window-bp", type=int, default=2000)
    parser.add_argument("--motif-effect-threshold-pct", type=float, default=10.0)
    parser.add_argument(
        "--chrombpnet-motif-threshold-pct",
        type=float,
        default=20.0,
        help="Motif filter on ChromBPNet |count| as %% of max (default: 20).",
    )
    parser.add_argument("--no-merge-motifs", action="store_true")
    parser.add_argument(
        "--chrombpnet-bw",
        default=str(default_bw),
        help="ChromBPNet counts_scores bigWig (default: SCCA_pos).",
    )
    parser.add_argument(
        "--row1-motif-tsv",
        default=str(default_base / "motif_scan_deg_chr16_45383863_45384762.tsv"),
    )
    parser.add_argument(
        "--row1-scores-csv",
        default=str(default_base / "chr16_45383863_45384762_deletion_regression_nucleotide.csv"),
    )
    parser.add_argument(
        "--row1-title",
        default="chr16:45383863-45384762 (regression)",
    )
    parser.add_argument(
        "--row2-motif-tsv",
        default=str(default_base / "motif_scan_deg_chr16_45356164_45357191.tsv"),
    )
    parser.add_argument(
        "--row2-scores-csv",
        default=str(default_base / "chr16_45356164_45357191_deletion_regression_nucleotide.csv"),
    )
    parser.add_argument(
        "--row2-title",
        default="chr16:45356164-45357191 (regression)",
    )
    parser.add_argument(
        "--row3-motif-tsv",
        default=str(default_prom / "motif_scan_promoter_relaxed_chr16_45408000_45409426.tsv"),
    )
    parser.add_argument(
        "--row3-scores-csv",
        default=str(default_prom / "chr16_45408000_45409426_deletion_differential_nucleotide.csv"),
    )
    parser.add_argument(
        "--row3-title",
        default="Promoter chr16:45408000-45409426 (differential)",
    )
    args = parser.parse_args()

    rows = [
        (args.row3_title, args.row3_motif_tsv, args.row3_scores_csv),
        (args.row1_title, args.row1_motif_tsv, args.row1_scores_csv),
        (args.row2_title, args.row2_motif_tsv, args.row2_scores_csv),
    ]

    merge = not args.no_merge_motifs
    for _title, motif_path, scores_path in rows:
        if not os.path.isfile(motif_path):
            print(f"Error: motif file not found: {motif_path}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(scores_path):
            print(f"Error: scores file not found: {scores_path}", file=sys.stderr)
            sys.exit(1)
    if not os.path.isfile(args.chrombpnet_bw):
        print(f"Error: ChromBPNet bigWig not found: {args.chrombpnet_bw}", file=sys.stderr)
        sys.exit(1)

    prepared: list[
        tuple[
            str,
            str,
            int,
            int,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            np.ndarray,
            np.ndarray,
            pd.DataFrame,
        ]
    ] = []
    with pyBigWig.open(args.chrombpnet_bw) as bw:
        for title, motif_path, scores_path in rows:
            scores_df = pd.read_csv(scores_path)
            scores_df, _ = prepare_plot_effects(
                scores_df,
                use_absolute_effect=True,
                normalize_to_max_predicted_atac=True,
            )
            chrom = _chromosome_from_scores_path(scores_path)
            p0, p1 = _window_from_scores(scores_df, args.center_window_bp)
            sdf = scores_df[
                (scores_df["position"] >= p0) & (scores_df["position"] <= p1)
            ].copy()

            base_motifs = load_base_motifs(
                motif_path,
                window_start=p0,
                window_end=p1,
                merge=merge,
            )
            deletion_motifs = filter_motifs_by_plot_effect(
                base_motifs,
                chromosome=chrom,
                scores_df=sdf,
                threshold_pct=args.motif_effect_threshold_pct,
            )

            cb_positions, chrombpnet = extract_chrombpnet_1bp(bw, chrom, p0, p1)
            # Same motif filter as deletion scores: abs score as % of max, then
            # keep motifs overlapping plot_effect >= threshold_pct.
            chrombpnet_abs_pct = np.zeros_like(chrombpnet)
            max_abs = float(np.max(np.abs(chrombpnet))) if len(chrombpnet) else 0.0
            if max_abs > 0:
                chrombpnet_abs_pct = (np.abs(chrombpnet) / max_abs) * 100.0
            chrombpnet_scores_df = pd.DataFrame(
                {
                    "position": cb_positions,
                    "plot_effect": chrombpnet_abs_pct,
                }
            )
            chrombpnet_motifs = filter_motifs_by_plot_effect(
                base_motifs,
                chromosome=chrom,
                scores_df=chrombpnet_scores_df,
                threshold_pct=args.chrombpnet_motif_threshold_pct,
            )

            prepared.append(
                (
                    title,
                    chrom,
                    p0,
                    p1,
                    sdf,
                    deletion_motifs,
                    chrombpnet_motifs,
                    cb_positions,
                    chrombpnet,
                    chrombpnet_scores_df,
                )
            )

    plt.rcParams["font.family"] = "Arial"
    deletion_color = "#2b6cb0"
    chrombpnet_color = "#c0392b"

    def _render(show_labels: bool, output_path: str) -> None:
        text_color = "black"
        fig_w = 8.0 * float(args.fig_scale)
        fig_h = 4.0 * float(args.fig_scale)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(
            4,
            3,
            figure=fig,
            height_ratios=[1.0, 1.0, 1.0, 1.0],
            hspace=0.35,
            wspace=0.35,
        )

        deletion_axes = []
        deletion_motif_axes = []
        chrombpnet_axes = []
        chrombpnet_motif_axes = []
        for i in range(3):
            da = fig.add_subplot(gs[0, i])
            dma = fig.add_subplot(gs[1, i], sharex=da)
            if i == 0:
                ca = fig.add_subplot(gs[2, i], sharex=da)
            else:
                ca = fig.add_subplot(gs[2, i], sharex=da, sharey=chrombpnet_axes[0])
            cma = fig.add_subplot(gs[3, i], sharex=da)
            deletion_axes.append(da)
            deletion_motif_axes.append(dma)
            chrombpnet_axes.append(ca)
            chrombpnet_motif_axes.append(cma)

        for i, packed in enumerate(prepared):
            (
                title,
                _chrom,
                p0,
                p1,
                sdf,
                deletion_motifs,
                chrombpnet_motifs,
                cb_positions,
                chrombpnet_raw,
                chrombpnet_scores_df,
            ) = packed
            da = deletion_axes[i]
            dma = deletion_motif_axes[i]
            ca = chrombpnet_axes[i]
            cma = chrombpnet_motif_axes[i]
            del_positions = sdf["position"].to_numpy()

            da.fill_between(
                del_positions,
                0,
                sdf["plot_effect"],
                color=deletion_color,
                alpha=0.8,
                linewidth=0,
            )
            da.plot(del_positions, sdf["plot_effect"], color=deletion_color, linewidth=0.8)
            da.set_ylim(0, 100)
            da.set_xlim(p0, p1)
            da.grid(False)

            dma.set_xlim(p0, p1)
            dma.set_ylabel("")
            dma.set_xlabel("")
            draw_motifs(dma, deletion_motifs, sdf)

            ca.axhline(0, color="0.85", linewidth=0.6, zorder=0)
            ca.plot(
                cb_positions,
                chrombpnet_raw,
                color=chrombpnet_color,
                linewidth=0.5,
                drawstyle="steps-post",
            )
            ca.set_xlim(p0, p1)
            ca.grid(False)

            cma.set_xlim(p0, p1)
            cma.set_ylabel("")
            cma.set_xlabel("")
            draw_motifs(cma, chrombpnet_motifs, chrombpnet_scores_df)

            if not show_labels:
                dma.set_yticklabels([])
                cma.set_yticklabels([])

            if show_labels:
                da.set_title(title, loc="center", fontsize=8, color=text_color)

            da.tick_params(axis="both", labelsize=8, labelleft=show_labels, labelbottom=False)
            dma.tick_params(axis="both", labelsize=8, labelleft=show_labels, labelbottom=False)
            ca.tick_params(axis="both", labelsize=8, labelleft=show_labels, labelbottom=False)
            cma.tick_params(axis="both", labelsize=8, labelleft=show_labels, labelbottom=show_labels)

            def format_bp(x, _p):
                return f"{int(x):,} bp"

            for ax in (da, dma, ca, cma):
                ax.xaxis.set_major_formatter(FuncFormatter(format_bp))
                ax.xaxis.set_major_locator(MultipleLocator(1000))

        chrombpnet_values_all = np.concatenate([item[8] for item in prepared])
        chrombpnet_ylim = _symmetric_ylim(chrombpnet_values_all)
        for ca in chrombpnet_axes:
            ca.set_ylim(chrombpnet_ylim)

        if show_labels:
            deletion_axes[0].set_ylabel("Deletion effect (%)", fontsize=8)
            chrombpnet_axes[0].set_ylabel("ChromBPNet count score", fontsize=8)

        fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10)
        plt.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {output_path}")

    base_out, ext = os.path.splitext(args.output_file)
    _render(True, f"{base_out}_black{ext}")
    _render(False, f"{base_out}_nolabels{ext}")


if __name__ == "__main__":
    main()
