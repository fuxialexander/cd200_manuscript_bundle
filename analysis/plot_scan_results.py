#!/usr/bin/env python3
"""
Plot motif scan results alongside deletion scores.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import os
from scipy.stats import zscore
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from gcell.rna.gencode import Gencode
except ImportError:
    print("Gencode not found, please install gcell.", file=sys.stderr)
    sys.exit(1)

try:
    from pyliftover import LiftOver
except ImportError:
    print("pyliftover not found, installing...", file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyliftover"])
    from pyliftover import LiftOver

try:
    import pyranges
except ImportError:
    print("pyranges not found, installing...", file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyranges"])
    import pyranges

# From the Nature paper and related analysis (hg19 coordinates)
HIGHLIGHT_REGIONS_HG19 = [
    ('chr2', 136875450, 136875469), ('chr2', 136875317, 136875336),
    ('chr2', 136875296, 136875315), ('chr2', 136875276, 136875295),
    ('chr2', 136875128, 136875137), ('chr2', 136875075, 136875094),
    ('chr2', 136875027, 136875046), ('chr2', 136874962, 136874981),
    ('chr2', 136874596, 136874615),
]

def gene_track_has_content(gencode, chrom: str, start: int, end: int, genes_to_highlight=None) -> bool:
    """
    Returns True if the gene-track would have any features to draw.

    We use this to avoid allocating an empty gene-body panel (common when the
    assembly/coordinates don't match the provided gene label).
    """
    if gencode is None:
        return False
    gtf = gencode.original_gtf
    region_gtf = gtf[
        (gtf["Chromosome"] == chrom)
        & (gtf["End"] >= start)
        & (gtf["Start"] <= end)
    ]
    if genes_to_highlight:
        region_gtf = region_gtf[region_gtf["gene_name"].isin(genes_to_highlight)]
    return not region_gtf.empty

def plot_gene_track(ax, gencode, chrom, start, end, genes_to_highlight, isoforms=False):
    """
    Plots gene bodies on a given axes.
    """
    gtf = gencode.original_gtf
    region_gtf = gtf[
        (gtf["Chromosome"] == chrom)
        & (gtf["End"] >= start)
        & (gtf["Start"] <= end)
    ].copy()

    if genes_to_highlight:
        region_gtf = region_gtf[region_gtf["gene_name"].isin(genes_to_highlight)]

    if region_gtf.empty:
        ax.set_yticks([])
        ax.set_frame_on(False)
        return

    feature_info = []
    gene_rows = region_gtf[region_gtf["Feature"] == "gene"]

    if isoforms:
        for _, gene_row in gene_rows.iterrows():
            gene_name = gene_row["gene_name"]
            gene_data = region_gtf[region_gtf["gene_id"] == gene_row["gene_id"]]
            transcripts = gene_data[gene_data["Feature"] == "transcript"]
            for _, transcript in transcripts.iterrows():
                feature_info.append({
                    "name": transcript.get("transcript_name", transcript["transcript_id"]),
                    "start": max(start, transcript["Start"]),
                    "end": min(end, transcript["End"]),
                    "strand": transcript["Strand"],
                    "data": gene_data[gene_data["transcript_id"] == transcript["transcript_id"]]
                })
    else:
        for _, gene_row in gene_rows.iterrows():
            gene_name = gene_row["gene_name"]
            gene_data = region_gtf[region_gtf["gene_id"] == gene_row["gene_id"]]
            feature_info.append({
                "name": gene_name,
                "start": max(start, gene_row["Start"]),
                "end": min(end, gene_row["End"]),
                "strand": gene_row["Strand"],
                "data": gene_data,
            })

    # Simple layout: one row per feature
    y_pos = 0
    track_height = 0.2
    exon_height = 0.1
    
    ax.set_ylim(-0.5, len(feature_info) - 0.5)
    ax.set_yticks(range(len(feature_info)))
    ax.set_yticklabels([f['name'] for f in feature_info], fontsize=9)


    for feature in feature_info:
        # Plot gene body line
        ax.plot([feature['start'], feature['end']], [y_pos, y_pos], 'k-', linewidth=1)
        
        # Plot exons
        exons = feature['data'][feature['data']['Feature'] == 'exon']
        for _, exon in exons.iterrows():
            exon_start = max(start, exon["Start"])
            exon_end = min(end, exon["End"])
            if exon_start < exon_end:
                rect = patches.Rectangle(
                    (exon_start, y_pos - exon_height / 2),
                    exon_end - exon_start,
                    exon_height,
                    facecolor="grey",
                    edgecolor="grey"
                )
                ax.add_patch(rect)
        
        # Add strand arrows
        num_arrows = int((feature['end'] - feature['start']) / ((end - start) / 20))
        arrow_positions = np.linspace(feature['start'], feature['end'], num_arrows + 2)[1:-1]
        for pos in arrow_positions:
            if feature['strand'] == '+':
                ax.plot(pos, y_pos, marker=r'>', color='k', markersize=4)
            else:
                ax.plot(pos, y_pos, marker=r'<', color='k', markersize=4)

        y_pos += 1

    ax.set_ylabel("Transcripts")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def liftover_regions(regions_hg19, target_assembly='hg38'):
    """Liftover a list of regions from hg19 to target assembly."""
    if target_assembly == 'hg19':
        # No liftover needed
        return [{'hg19': r, target_assembly: r} for r in regions_hg19]
    
    lo = LiftOver('hg19', target_assembly)
    lifted_regions = []
    for chrom, start, end in regions_hg19:
        start_result = lo.convert_coordinate(chrom, start)
        end_result = lo.convert_coordinate(chrom, end)
        if start_result and end_result:
            new_chrom = start_result[0][0]
            new_start = start_result[0][1]
            new_end = end_result[0][1]
            lifted_regions.append({
                'hg19': (chrom, start, end),
                target_assembly: (new_chrom, new_start, new_end)
            })
    return lifted_regions

def merge_motifs(motifs_df):
    """
    Merge overlapping motifs for the same TF using pyranges.
    The most significant p-value and score are retained for each merged region.
    """
    if motifs_df.empty:
        return motifs_df

    # Prepare DataFrame for pyranges
    pr_df = motifs_df.rename(columns={
        'chromosome': 'Chromosome',
        'start': 'Start',
        'end': 'End'
    })
    pr = pyranges.PyRanges(pr_df)

    # Cluster by gene name to find overlapping/adjacent motifs of the same TF
    clustered = pr.cluster(by='gene_name', strand=False)
    
    merged_motifs = []
    # Group by the original gene name and the new cluster ID
    for (gene, cluster), group in clustered.df.groupby(['gene_name', 'Cluster']):
        if len(group) == 1:
            # No merging needed for single motifs
            merged_motifs.append(group.iloc[0])
            continue
            
        # For a cluster of motifs, find the union of their genomic regions
        merged_start = group.Start.min()
        merged_end = group.End.max()
        
        # Find the most significant hit within the cluster (lowest p-value)
        best_hit = group.loc[group.p_value.idxmin()]
        
        # Create a new record for the merged motif
        merged_record = best_hit.copy()
        merged_record['Start'] = merged_start
        merged_record['End'] = merged_end
        merged_motifs.append(merged_record)
        
    merged_df = pd.DataFrame(merged_motifs).rename(columns={
        'Chromosome': 'chromosome',
        'Start': 'start',
        'End': 'end'
    })
    
    return merged_df.sort_values(by='start').reset_index(drop=True)


def rank_tfs_by_max_abs_effect(motifs_df: pd.DataFrame, scores_df: pd.DataFrame) -> list[str]:
    """Rank TF rows by the strongest absolute overlapping deletion effect."""
    if motifs_df.empty:
        return []

    score_positions = scores_df['position'].to_numpy()
    effect_values = scores_df['plot_effect'].to_numpy()
    tf_scores = {}

    for gene_name, group in motifs_df.groupby('gene_name'):
        max_abs_effect = 0.0
        for row in group.itertuples(index=False):
            overlap_mask = (score_positions >= row.start) & (score_positions < row.end)
            if np.any(overlap_mask):
                max_abs_effect = max(max_abs_effect, float(np.max(np.abs(effect_values[overlap_mask]))))
        tf_scores[gene_name] = max_abs_effect

    return sorted(tf_scores, key=lambda tf: (-tf_scores[tf], tf))


def _max_abs_effect_for_interval(start: int, end: int, score_positions: np.ndarray, effect_values: np.ndarray) -> float:
    """Return the max absolute effect overlapping a genomic interval."""
    overlap_mask = (score_positions >= start) & (score_positions < end)
    if not np.any(overlap_mask):
        return 0.0
    return float(np.max(np.abs(effect_values[overlap_mask])))


def prepare_plot_effects(
    scores_df: pd.DataFrame,
    *,
    use_absolute_effect: bool = False,
    normalize_to_max_predicted_atac: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Create plot-only effect values without modifying the saved raw scores."""
    plot_scores_df = scores_df.copy()
    plot_effect = plot_scores_df['effect'].to_numpy(dtype=float)

    if use_absolute_effect:
        plot_effect = np.abs(plot_effect)

    effect_label = 'Effect'
    if use_absolute_effect:
        effect_label = 'Abs Effect'
    if normalize_to_max_predicted_atac:
        normalization_source = None
        if 'baseline_prediction' in plot_scores_df.columns and plot_scores_df['baseline_prediction'].notna().any():
            normalization_source = plot_scores_df['baseline_prediction'].to_numpy(dtype=float)
        elif 'observed_prediction' in plot_scores_df.columns and plot_scores_df['observed_prediction'].notna().any():
            normalization_source = plot_scores_df['observed_prediction'].to_numpy(dtype=float)

        if normalization_source is None:
            print(
                "Warning: no baseline prediction column is available; "
                "cannot normalize deletion effects to the predicted output scale.",
                file=sys.stderr,
            )
        else:
            max_predicted_output = float(np.nanmax(np.abs(normalization_source)))
            if max_predicted_output > 0:
                plot_effect = (plot_effect / max_predicted_output) * 100.0
                effect_label = 'Effect (%)'
            else:
                print(
                    "Warning: max predicted output is non-positive; "
                    "skipping deletion-effect normalization.",
                    file=sys.stderr,
                )

    plot_scores_df['plot_effect'] = plot_effect
    return plot_scores_df, effect_label


def build_overlap_plot_rows(motifs_df: pd.DataFrame, scores_df: pd.DataFrame) -> list[dict]:
    """Build plot rows by clustering motifs that overlap in genomic coordinates."""
    if motifs_df.empty:
        return []

    motifs_df = motifs_df.drop(columns=['Cluster'], errors='ignore')
    score_positions = scores_df['position'].to_numpy()
    effect_values = scores_df['plot_effect'].to_numpy()

    pr_df = motifs_df.rename(columns={
        'chromosome': 'Chromosome',
        'start': 'Start',
        'end': 'End'
    })
    clustered = pyranges.PyRanges(pr_df).cluster(strand=False).df

    plot_rows = []
    for cluster_id, group in clustered.groupby('Cluster'):
        cluster_start = int(group['Start'].min())
        cluster_end = int(group['End'].max())

        gene_scores = {}
        for gene_name, gene_group in group.groupby('gene_name'):
            gene_max_score = 0.0
            for row in gene_group.itertuples(index=False):
                gene_max_score = max(
                    gene_max_score,
                    _max_abs_effect_for_interval(int(row.Start), int(row.End), score_positions, effect_values)
                )
            gene_scores[gene_name] = gene_max_score

        ordered_genes = sorted(gene_scores, key=lambda gene: (-gene_scores[gene], gene))
        plot_rows.append({
            'cluster_id': cluster_id,
            'label': '/'.join(ordered_genes),
            'score': _max_abs_effect_for_interval(cluster_start, cluster_end, score_positions, effect_values),
            'start': cluster_start,
            'end': cluster_end,
            'motifs': group.rename(columns={
                'Chromosome': 'chromosome',
                'Start': 'start',
                'End': 'end'
            }).copy()
        })

    return sorted(plot_rows, key=lambda row: (-row['score'], row['start'], row['label']))

def plot_results(
    motif_file,
    scores_file,
    output_file,
    highlight,
    merge,
    tss_highlights,
    heatmap_file=None,
    core_region=None,
    gencode=None,
    isoforms=False,
    motif_sigma=3.0,
    motif_effect_threshold_pct=None,
    use_absolute_effect=False,
    normalize_to_max_predicted_atac=False,
    assembly='hg38',
    center_window_bp=None,
    dpi: int = 300,
    profile_scores_file=None,
    fig_scale: float = 1.0,
    show_legend: bool = False,
):
    """
    Generates a multi-panel plot showing deletion scores, motif hits,
    and optionally a deletion difference heatmap and gene tracks.
    
    Args:
        center_window_bp: If specified, restrict the plot to show only the central N bp of the data.
    """
    # Set font to Liberation Sans for all text
    plt.rcParams['font.family'] = 'Liberation Sans'
    
    # Load data
    motifs_df = pd.read_csv(motif_file, sep='	')
    scores_df = pd.read_csv(scores_file)
    profile_scores_df = pd.read_csv(profile_scores_file) if profile_scores_file else scores_df
    scores_df, effect_label = prepare_plot_effects(
        scores_df,
        use_absolute_effect=use_absolute_effect,
        normalize_to_max_predicted_atac=normalize_to_max_predicted_atac,
    )
    
    # Calculate center window if requested
    plot_start = scores_df['position'].min()
    plot_end = scores_df['position'].max()
    if center_window_bp is not None and center_window_bp > 0:
        center_pos = (plot_start + plot_end) / 2
        half_window = center_window_bp / 2
        plot_start = int(center_pos - half_window)
        plot_end = int(center_pos + half_window)
        print(f"Restricting plot to central {center_window_bp} bp: {plot_start:,} - {plot_end:,}")
        
        # Filter motifs to only those overlapping with the central window
        if not motifs_df.empty:
            count_before = len(motifs_df)
            # Keep motifs that overlap with the central window
            # A motif overlaps if: motif_start < plot_end AND motif_end > plot_start
            motifs_df = motifs_df[
                (motifs_df['end'] > plot_start) & (motifs_df['start'] < plot_end)
            ].copy()
            count_after = len(motifs_df)
            print(f"   Filtered motifs from {count_before} to {count_after} that overlap with central window")

    # --- Check for new data columns ---
    has_obs_pred = 'observed_prediction' in profile_scores_df.columns and profile_scores_df['observed_prediction'].notna().any()
    has_obs_atac = 'observed_atac' in profile_scores_df.columns and profile_scores_df['observed_atac'].notna().any()
    show_obs_plot = has_obs_pred and has_obs_atac

    genes_to_highlight = [label for _, label in tss_highlights] if tss_highlights else None
    chromosome = os.path.basename(scores_file).split('_')[0]

    include_gene_track = False
    if genes_to_highlight and gencode:
        include_gene_track = gene_track_has_content(
            gencode=gencode,
            chrom=chromosome,
            start=plot_start,
            end=plot_end,
            genes_to_highlight=genes_to_highlight,
        )

    # Merge motifs if requested
    if merge:
        print(f"Original motif count: {len(motifs_df)}")
        motifs_df = merge_motifs(motifs_df)
        print(f"Merged motif count: {len(motifs_df)}")

    # --- Filter motifs based on plot-scaled effect or z-score ---
    if not motifs_df.empty:
        significant_scores = None
        if motif_effect_threshold_pct is not None:
            print(f"Filtering motifs to positions with deletion effect >= {motif_effect_threshold_pct}%...")
            significant_scores = scores_df[scores_df['plot_effect'] >= motif_effect_threshold_pct]
        elif motif_sigma is not None and motif_sigma > 0:
            print(f"Filtering motifs to positions with |z-score(effect)| > {motif_sigma}...")

            effect_values = scores_df['plot_effect'].values
            p10 = np.percentile(effect_values, 10)
            p90 = np.percentile(effect_values, 90)
            filtered_effect = effect_values[(effect_values >= p10) & (effect_values <= p90)]

            mean_effect = np.mean(filtered_effect)
            std_effect = np.std(filtered_effect, ddof=0)
            scores_df['zscore'] = (effect_values - mean_effect) / std_effect if std_effect > 0 else np.zeros_like(effect_values)
            significant_scores = scores_df[np.abs(scores_df['zscore']) > motif_sigma]

        if significant_scores is not None:
            if significant_scores.empty:
                print("   - No significant positions found. No motifs will be plotted.")
                motifs_df = motifs_df.iloc[0:0]
            else:
                print(f"   ✓ Found {len(significant_scores)} positions exceeding threshold.")

                chromosome = os.path.basename(scores_file).split('_')[0]
                significant_pr = pyranges.PyRanges(
                    chromosomes=[chromosome] * len(significant_scores),
                    starts=significant_scores['position'],
                    ends=significant_scores['position'] + 1
                )

                motifs_pr = pyranges.PyRanges(motifs_df.rename(columns={
                    'chromosome': 'Chromosome',
                    'start': 'Start',
                    'end': 'End'
                }))
                overlapping_motifs_pr = motifs_pr.overlap(significant_pr)

                count_before = len(motifs_df)
                if overlapping_motifs_pr.empty:
                    print("   - No motifs overlap with significant positions.")
                    motifs_df = motifs_df.iloc[0:0]
                else:
                    motifs_df = overlapping_motifs_pr.df.rename(columns={
                        'Chromosome': 'chromosome',
                        'Start': 'start',
                        'End': 'end'
                    })
                    print(f"   ✓ Filtered {count_before} motifs down to {len(motifs_df)} that overlap with significant positions.")

    # Drop motifs with no gene name for this plot style
    motifs_df = motifs_df.dropna(subset=['gene_name'])

    # --- Plot Setup ---
    num_panels = 2 + (1 if heatmap_file else 0) + (1 if show_obs_plot else 0)
    if include_gene_track:
        num_panels += 1

    height_ratios = []
    if show_obs_plot: height_ratios.append(2)
    if heatmap_file: height_ratios.append(3)
    height_ratios.append(3) # Deletion scores
    height_ratios.append(3) # Motifs
    if include_gene_track:
        height_ratios.append(2) # Gene track

    fig_w = 7.0 * float(fig_scale)
    figsize_height = 2 * num_panels * float(fig_scale)

    if num_panels == 1: # Handle case with a single plot
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, figsize_height))
        axes = [ax]
    else:
        fig, axes = plt.subplots(num_panels, 1, figsize=(fig_w, figsize_height), sharex=True,
                                              gridspec_kw={'height_ratios': height_ratios})
    
    # --- Assign axes ---
    ax_idx = 0
    obs_ax = None
    if show_obs_plot:
        obs_ax = axes[ax_idx]
        ax_idx += 1

    heatmap_ax = None
    if heatmap_file:
        heatmap_ax = axes[ax_idx]
        ax_idx += 1
    
    scores_ax = axes[ax_idx]
    ax_idx += 1
    motifs_ax = axes[ax_idx]
    ax_idx += 1

    gene_ax = None
    if include_gene_track:
        if ax_idx < len(axes):
            gene_ax = axes[ax_idx]
        else: # This case can happen if axes is not properly indexed
            print("Warning: Could not create gene track axis.", file=sys.stderr)


    # fig.suptitle('Combined Deletion Scan Analysis', fontsize=18)

    # --- Panel 1: Observed vs. Predicted ATAC (Conditional) ---
    if obs_ax:
        obs_ax.plot(profile_scores_df['position'], profile_scores_df['observed_prediction'], label='Predicted ATAC', color='r', alpha=0.8)
        
        # Smooth the observed ATAC for better visualization
        smoothed_atac = profile_scores_df['observed_atac']
        obs_ax.plot(profile_scores_df['position'], smoothed_atac, label='Observed ATAC', color='g', alpha=0.8)
        
        # obs_ax.set_title('Predicted vs. Observed ATAC Signal')
        obs_ax.set_ylabel('ATAC Signal')
        if show_legend:
            obs_ax.legend(frameon=False)
        obs_ax.grid(False)

        if core_region:
            _, core_start, core_end = core_region
            obs_ax.axvspan(core_start, core_end, color='lightgrey', alpha=0.3, zorder=0, label='Core Region')
            if show_legend:
                obs_ax.legend(frameon=False)

    # --- Panel 2: Deletion Difference Heatmap (Conditional) ---
    if heatmap_ax:
        heatmap_data = np.load(heatmap_file)
        zscored_heatmap = zscore(heatmap_data, axis=1)
        
        # Create binary mask: 1 where significant (|zscore| > motif_sigma), 0 otherwise
        # Use the same threshold as for motif filtering
        binary_heatmap = (np.abs(zscored_heatmap) > motif_sigma).astype(float)
        
        genomic_coords = scores_df['position'].values
        num_windows = zscored_heatmap.shape[1]
        start_coord = genomic_coords[0]
        end_coord = genomic_coords[num_windows - 1]
        
        # Create a binary colormap: white for 0, grey for 1
        binary_cmap = ListedColormap(['white', 'grey'])
        
        # Display as grey box: grey where significant, white where not
        # Use interpolation='nearest' to ensure crisp binary display
        im = heatmap_ax.imshow(binary_heatmap, cmap=binary_cmap, vmin=0, vmax=1, 
                              aspect='auto', extent=[start_coord, end_coord, zscored_heatmap.shape[0], 0],
                              interpolation='nearest', origin='upper')
        # heatmap_ax.set_title('Deletion Difference Heatmap')
        heatmap_ax.set_ylabel('Sequence Position')
        
        # Set shared x-axis limit based on heatmap which is narrowest
        # But respect center_window_bp if specified
        if center_window_bp is not None:
            heatmap_ax.set_xlim(plot_start, plot_end)
            motifs_ax.set_xlim(plot_start, plot_end)
        else:
            heatmap_ax.set_xlim(start_coord, end_coord)
            motifs_ax.set_xlim(start_coord, end_coord)

    # --- Panel 3: Deletion Scores ---
    scores_ax.plot(scores_df['position'], scores_df['plot_effect'], color='b')
    if not use_absolute_effect:
        scores_ax.axhline(0, color='grey', linestyle='--', linewidth=0.7)
    if normalize_to_max_predicted_atac:
        scores_ax.set_ylim(0, 100)
    scores_ax.set_ylabel(effect_label)
    # scores_ax.set_title('Deletion Effect on Core Region Accessibility (Grey area above)')
    scores_ax.grid(False)

    # --- Panel 4: Motif Hits ---
    # motifs_ax.set_title('Significant Motif Hits', pad=2)
    motifs_ax.set_ylabel('Motifs')
    motifs_ax.set_xlabel(f"Genomic Position ({chromosome})")

    # --- Panel 5: Gene Track (Conditional) ---
    if gene_ax and gencode and genes_to_highlight:
        # gene_ax.set_title('Gene Annotations')
        # Use plot_start and plot_end which may be restricted by center_window_bp
        plot_gene_track(gene_ax, gencode, chromosome, plot_start, plot_end, genes_to_highlight, isoforms=isoforms)
        gene_ax.set_xlabel(f"Genomic Position ({chromosome})")
    # --- Apply Highlights to all relevant axes ---
    all_axes = [ax for ax in [heatmap_ax, obs_ax, scores_ax, motifs_ax, gene_ax] if ax is not None]

    # Highlight TSS
    if tss_highlights:
        for i, (pos_str, label) in enumerate(tss_highlights):
            pos = int(pos_str)
            for ax in all_axes:
                ax.axvline(x=pos, color='darkgreen', linestyle=':', linewidth=1.5, 
                           label=f'{label} TSS' if ax == scores_ax else "", zorder=3)
        if show_legend:
            scores_ax.legend(frameon=False)
    
    # Set x-axis limits (use center window if specified, otherwise full range)
    if not heatmap_ax: # Only set xlim here if not set by heatmap
        scores_ax.set_xlim(plot_start, plot_end)
    else:
        # If heatmap exists, update its limits too
        if center_window_bp is not None:
            heatmap_ax.set_xlim(plot_start, plot_end)

    # --- Highlight Regions ---
    if highlight:
        lifted_regions = liftover_regions(HIGHLIGHT_REGIONS_HG19, target_assembly=assembly)
        data_start, data_end = scores_df['position'].min(), scores_df['position'].max()
        
        overlapping_regions = [
            r for r in lifted_regions 
            if r[assembly][0] == os.path.basename(scores_file).split('_')[0] 
            and r[assembly][1] <= data_end and r[assembly][2] >= data_start
        ]
        
        colors = plt.colormaps['Set2'].resampled(len(overlapping_regions))
        legend_handles = []
        
        for i, region_pair in enumerate(overlapping_regions):
            chrom, start, end = region_pair[assembly]
            for ax in all_axes:
                ax.axvspan(start, end, color=colors(i), alpha=0.2, zorder=0)

            hg19_chrom, hg19_start, hg19_end = region_pair['hg19']
            label = (f"R{i+1}: {hg19_chrom}:{hg19_start:,}-{hg19_end:,} (hg19) -> "
                     f"{chrom}:{start:,}-{end:,} ({assembly})")
            legend_handles.append(patches.Patch(color=colors(i), alpha=0.4, label=label))

    # --- Motif Plotting Logic (on motifs_ax) ---
    if not motifs_df.empty:
        plot_rows = build_overlap_plot_rows(motifs_df, scores_df)
        row_to_y = {row['cluster_id']: i for i, row in enumerate(plot_rows)}
        
        motifs_ax.set_yticks(range(len(plot_rows)))
        motifs_ax.set_yticklabels([row['label'] for row in plot_rows], fontsize=9)
        motifs_ax.set_ylim(len(plot_rows) - 0.5, -0.5)
        motifs_ax.grid(axis='y', linestyle='--', linewidth=0.5)

        # Use grey color for all motifs instead of color gradient
        for plot_row in plot_rows:
            y_pos = row_to_y[plot_row['cluster_id']]
            for _, row in plot_row['motifs'].iterrows():
                width = row['end'] - row['start']

                rect = patches.Rectangle(
                    (row['start'], y_pos - 0.4), width, 0.8,
                    facecolor='grey',
                    edgecolor='grey',
                    linewidth=0.5
                )
                motifs_ax.add_patch(rect)
    else:
        motifs_ax.set_yticks([])

    # Add region legend at the bottom if highlights are enabled
    if show_legend and highlight and legend_handles:
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            fontsize='x-small',
            title="Highlighted Regions",
            frameon=False
        )

    # Format x-axis to show full numbers with "bp" suffix (no scientific notation)
    # Apply to the bottom-most axis (which shares x-axis with all others)
    bottom_ax = gene_ax if gene_ax is not None else (motifs_ax if motifs_ax is not None else scores_ax)
    if bottom_ax is not None:
        def format_bp(x, p):
            return f'{int(x):,} bp'
        for ax in all_axes:
            ax.xaxis.set_major_formatter(FuncFormatter(format_bp))
            ax.xaxis.set_major_locator(MultipleLocator(1000))
        # FuncFormatter already prevents scientific notation, no need for ticklabel_format
    
    # Final adjustments
    bottom_pad = 0.1 if (show_legend and highlight and legend_handles) else 0.04
    fig.tight_layout(rect=[0, bottom_pad, 1, 0.96])
    
    # Save the plot
    plt.savefig(output_file, dpi=int(dpi), bbox_inches='tight')
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--motif-scan-file', 
        required=True, 
        help='Path to the motif scan results TSV file.'
    )
    parser.add_argument(
        '--deletion-scores-file', 
        required=True, 
        help='Path to the deletion scores CSV file.'
    )
    parser.add_argument(
        '--output-file', 
        required=True, 
        help='Path to save the output plot PNG file.'
    )
    parser.add_argument(
        '--heatmap-file',
        required=False,
        default=None,
        help='Path to the deletion difference heatmap .npy file. If provided, a three-panel plot is generated.'
    )
    parser.add_argument(
        '--highlight-regions',
        action='store_true',
        help='If specified, highlight the 8 key genomic regions on the plot.'
    )
    parser.add_argument(
        '--merge-motifs',
        action='store_true',
        help='If specified, merge overlapping motifs of the same TF.'
    )
    parser.add_argument(
        '--tss-highlight',
        nargs=2,
        metavar=('POSITION', 'LABEL'),
        action='append',
        help='Genomic coordinate and label for a TSS to highlight (e.g., 136118149 CXCR4). Can be specified multiple times.'
    )
    parser.add_argument(
        '--core-region',
        type=str,
        help='Core region to highlight in format chr:start-end (e.g., chr2:136117300-136117800).'
    )
    parser.add_argument(
        '--isoforms',
        action='store_true',
        help='If specified, plot gene isoforms separately.'
    )
    parser.add_argument(
        '--motif-sigma',
        type=float,
        default=3.0,
        help='Z-score threshold for mutation scores to filter motifs. Only motifs overlapping with scores outside this many standard deviations from the mean are shown.'
    )
    parser.add_argument(
        '--motif-effect-threshold-pct',
        type=float,
        default=None,
        help='If set, filter plotted motifs to positions with plot-scaled deletion effect at or above this percentage threshold.'
    )
    parser.add_argument(
        '--plot-absolute-effect',
        action='store_true',
        help='If specified, plot the absolute deletion effect instead of the signed value.'
    )
    parser.add_argument(
        '--normalize-effect-to-max-predicted-atac',
        action='store_true',
        help='If specified, scale plotted deletion effects to percent of the maximum predicted ATAC signal in the region.'
    )
    parser.add_argument(
        '--assembly',
        type=str,
        default='hg38',
        choices=['hg19', 'hg38', 'mm9', 'mm10'],
        help='Genome assembly to use for gene annotations (default: hg38).'
    )
    parser.add_argument(
        '--center-window-bp',
        type=int,
        default=None,
        help='If specified, restrict the plot to show only the central N bp of the data (e.g., 2000 or 3000).'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output DPI (PPI) for saved PNG (default: 300).'
    )
    parser.add_argument(
        '--fig-scale',
        type=float,
        default=1.0,
        help='Scale factor for figure width and height (default: 1.0). Use 0.5 for half-size figures.'
    )
    parser.add_argument(
        '--show-legend',
        action='store_true',
        help='If set, show legends (ATAC panel, TSS, highlighted regions). Default is no legends.'
    )
    parser.add_argument(
        '--profile-scores-file',
        default=None,
        help='Optional scores CSV to use for the top predicted/observed ATAC panel and normalization source.'
    )
    args = parser.parse_args()
    
    # Map assembly to Gencode version
    assembly_to_version = {
        'hg38': '40',
        'hg19': '40',
        'mm10': 'M10',  # M38 for mm10
        'mm9': 'M10',
    }
    gencode_version = assembly_to_version.get(args.assembly, '40')
    gencode = Gencode(assembly=args.assembly, version=gencode_version)

    core_region_parsed = None
    if args.core_region:
        try:
            chrom, coords = args.core_region.split(':')
            start, end = map(int, coords.split('-'))
            core_region_parsed = (chrom, start, end)
        except ValueError:
            print(f"Error: Invalid format for --core-region '{args.core_region}'. "
                  "Expected format: chr:start-end", file=sys.stderr)
            sys.exit(1)

    plot_results(
        args.motif_scan_file, 
        args.deletion_scores_file, 
        args.output_file, 
        args.highlight_regions,
        args.merge_motifs,
        args.tss_highlight,
        args.heatmap_file,
        core_region_parsed,
        gencode=gencode,
        isoforms=args.isoforms,
        motif_sigma=args.motif_sigma,
        motif_effect_threshold_pct=args.motif_effect_threshold_pct,
        use_absolute_effect=args.plot_absolute_effect,
        normalize_to_max_predicted_atac=args.normalize_effect_to_max_predicted_atac,
        assembly=args.assembly,
        center_window_bp=args.center_window_bp,
        dpi=args.dpi,
        profile_scores_file=args.profile_scores_file,
        fig_scale=args.fig_scale,
        show_legend=args.show_legend,
    )

if __name__ == "__main__":
    main()
