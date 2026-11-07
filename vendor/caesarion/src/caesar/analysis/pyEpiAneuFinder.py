# Required libraries: pandas, numpy, scipy, statsmodels, joblib
# Install them: pip install pandas numpy scipy statsmodels joblib
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # Suppress common warnings from stats/lowess


def filter_low_coverage_cells(counts_df, threshold_cells_nbins=0.05):
    """Filters cells with too many zero-count bins."""
    n_bins = counts_df.shape[0]
    zeros_per_cell = (counts_df == 0).sum(axis=0)
    keep_cells = zeros_per_cell < ((1 - threshold_cells_nbins) * n_bins)
    n_removed = counts_df.shape[1] - keep_cells.sum()
    print(
        f"Filtering cells: Removed {n_removed} cells with > "
        f"{100 * (1 - threshold_cells_nbins):.1f}% zero bins."
    )
    return counts_df.loc[:, keep_cells]


def filter_low_coverage_bins(counts_df, bin_info_df, threshold_blacklist_bins=0.85):
    """Filters bins with zero counts in too many cells."""
    n_cells = counts_df.shape[1]
    zeros_per_bin = (counts_df == 0).sum(axis=1)
    keep_bins = zeros_per_bin < (threshold_blacklist_bins * n_cells)
    n_removed = counts_df.shape[0] - keep_bins.sum()
    print(
        f"Filtering bins: Removed {n_removed} bins with zero counts in > "
        f"{100 * threshold_blacklist_bins:.1f}% of cells."
    )

    # Ensure boolean indexing works even if index names differ slightly
    keep_indices = counts_df.index[keep_bins]

    return counts_df.loc[keep_indices, :], bin_info_df.loc[keep_indices, :]


def gc_correct_cell(cell_counts, gc_content):
    """Applies LOESS GC correction to a single cell's counts."""
    # Filter out bins with zero count for fitting, but keep them for correction
    non_zero_mask = cell_counts > 0
    if non_zero_mask.sum() < 2:  # Need at least 2 points for LOESS
        return cell_counts  # Return original if not enough data

    counts_fit = cell_counts[non_zero_mask]
    gc_fit = gc_content[non_zero_mask]

    try:
        # frac controls the smoothness, lowess in R uses 0.75 default
        # Ensure inputs are float64 for lowess
        lowess_fit = sm.nonparametric.lowess(
            counts_fit.astype(np.float64),
            gc_fit.astype(np.float64),
            frac=0.75,
            it=3,  # Number of robustness iterations
        )

        # Create a mapping from GC value to fitted value
        # Handle potential duplicate GC values by averaging fitted values
        fitted_map = (
            pd.DataFrame(lowess_fit, columns=["gc", "fitted"]).groupby("gc").mean()
        )

        # Map fitted values back to the original gc_content series
        # Use interpolation or nearest neighbor for GC values not in the fit
        fitted_values_all = pd.Series(index=gc_content.index, dtype=np.float64)

        # Simple mapping for GC values present in the fit
        common_gc = gc_content.index[gc_content.isin(fitted_map.index)]
        fitted_values_all[common_gc] = gc_content[common_gc].map(fitted_map["fitted"])

        # For GC values not in the fit, use interpolation (or ffill/bfill)
        # This part can be complex; using mean as a fallback for simplicity here
        # A better approach might involve interpolating fitted_map
        missing_gc_mask = fitted_values_all.isna()
        if missing_gc_mask.any():
            # Very basic fallback: use mean fitted value for missing ones
            mean_fitted = fitted_map["fitted"].mean()
            fitted_values_all[missing_gc_mask] = mean_fitted

        # Avoid division by zero or near-zero
        fitted_values_all = np.maximum(fitted_values_all, 1e-6)  # Floor fitted value

        # Calculate correction factor
        mean_counts = cell_counts.mean()  # Use mean of original counts
        correction_factor = mean_counts / fitted_values_all

        # Apply correction
        corrected_counts = (cell_counts * correction_factor).round().astype(int)
        # Ensure non-negative counts
        corrected_counts = np.maximum(corrected_counts, 0)

    except Exception as e:
        print(
            f"  Warning: LOESS failed for cell {cell_counts.name}. Returning original counts. Error: {e}"
        )
        corrected_counts = cell_counts

    return corrected_counts


def parallel_gc_correct(counts_df, gc_content, n_cores=4):
    """Performs GC correction in parallel across cells."""
    print(f"Performing GC correction using {n_cores} cores...")
    results = Parallel(n_jobs=n_cores)(
        delayed(gc_correct_cell)(counts_df[cell], gc_content)
        for cell in counts_df.columns
    )
    corrected_df = pd.concat(results, axis=1)
    corrected_df.columns = counts_df.columns
    return corrected_df


# --- Simplified CNV Calling ---
# NOTE: This is a conceptual placeholder. Replicating the exact recursive
# binary segmentation from distanceStatistic.R requires more involved code.


def find_best_breakpoint(segment_data, test_type="KS"):
    """Finds the single best breakpoint in a segment based on KS or AD test."""
    best_stat = -1
    best_idx = -1
    min_segment_len = 5  # Minimum bins per segment to perform test

    # Iterate through possible split points
    for i in range(min_segment_len, len(segment_data) - min_segment_len):
        segment1 = segment_data[:i]
        segment2 = segment_data[i:]

        if test_type == "KS":
            stat, p_val = stats.ks_2samp(segment1, segment2)
        elif test_type == "AD":
            # Anderson-Darling k-sample test. Returns statistic and critical values.
            # We use the statistic directly. Higher means more different.
            try:
                stat, crit_vals, p_approx = stats.anderson_ksamp([segment1, segment2])
                # Anderson-Darling statistic can be negative if identical dists; use abs or max(0, stat)
                stat = max(0, stat)
            except ValueError:  # Can happen with insufficient data
                stat = 0
        else:
            raise ValueError("test_type must be 'KS' or 'AD'")

        if stat > best_stat:
            best_stat = stat
            best_idx = i

    return best_idx  # Returns index relative to the start of the segment_data


def get_breakpoints_recursive(
    segment_data, current_k, max_k, test_type="KS", min_segment_len=10
):
    """Recursively finds breakpoints up to 2^k segments."""
    if current_k >= max_k or len(segment_data) < 2 * min_segment_len:
        return []

    split_idx_rel = find_best_breakpoint(segment_data, test_type)

    if split_idx_rel == -1:  # No valid split point found
        return []

    # Convert relative index to absolute index within the original data frame slice
    split_idx_abs = segment_data.index[split_idx_rel]

    # Recursively find breakpoints in the left and right segments
    left_segment = segment_data.iloc[:split_idx_rel]
    right_segment = segment_data.iloc[split_idx_rel:]

    breakpoints = [split_idx_abs]
    breakpoints.extend(
        get_breakpoints_recursive(
            left_segment, current_k + 1, max_k, test_type, min_segment_len
        )
    )
    breakpoints.extend(
        get_breakpoints_recursive(
            right_segment, current_k + 1, max_k, test_type, min_segment_len
        )
    )

    return breakpoints


def assign_somy(segment_counts):
    """Assigns a basic copy number state based on median count."""
    # This is highly simplified. Real assignment might involve comparison
    # to a cell-wide baseline or a more sophisticated model.
    # Example: Assume overall median count represents diploid (CN=2)

    # Calculate a baseline - e.g., median of non-zero counts for the whole cell (needs to be passed)
    # For this placeholder, let's just use the segment median directly
    # and assign states relative to some arbitrary expected value (e.g., 50)

    # A robust implementation would require passing cell-specific baseline
    segment_median = np.median(segment_counts)

    # Placeholder thresholds (adjust based on expected ploidy and data scale)
    baseline_diploid_median = 50  # Example arbitrary baseline
    if segment_median < baseline_diploid_median * 0.75:
        return "Loss"  # Potential Monosomy/Deletion
    elif segment_median > baseline_diploid_median * 1.25:
        return "Gain"  # Potential Trisomy/Amplification
    else:
        return "Neutral"  # Potential Disomy


def process_cell_cnv(
    cell_id, cell_counts, bin_info_df, k=4, test_type="KS", minsizeCNV=3
):
    """Finds breakpoints and assigns somy for a single cell."""
    print(f"  Processing CNVs for cell: {cell_id}")
    cell_results = []

    # Calculate a baseline for somy assignment (e.g., median of non-zero counts)
    cell_baseline_median = np.median(cell_counts[cell_counts > 0])
    if np.isnan(cell_baseline_median) or cell_baseline_median == 0:
        cell_baseline_median = 1  # Avoid division by zero if cell is mostly empty

    for chrom, chrom_bins in bin_info_df.groupby("seqnames"):
        chrom_counts = cell_counts.loc[chrom_bins.index]

        # Get breakpoints for this chromosome
        # The recursive function needs data as numpy array potentially
        breakpoints_abs_indices = get_breakpoints_recursive(
            chrom_counts,
            current_k=0,
            max_k=k,
            test_type=test_type,
            min_segment_len=max(5, minsizeCNV),  # Ensure segments are large enough
        )

        # Add chromosome start and end as implicit breakpoints
        segment_indices = sorted(
            list(
                set(
                    [chrom_bins.index[0]]
                    + breakpoints_abs_indices
                    + [chrom_bins.index[-1]]
                )
            )
        )

        # Define segments and assign somy
        for i in range(len(segment_indices) - 1):
            start_idx = segment_indices[i]
            # Need to handle the end index carefully for slicing
            # Find the position of the next breakpoint index
            end_loc = chrom_counts.index.get_loc(segment_indices[i + 1])
            segment_data = chrom_counts.loc[start_idx : segment_indices[i + 1]]

            # Correct segment slicing if start_idx == end_idx
            if start_idx == segment_indices[i + 1]:
                segment_data = chrom_counts.loc[[start_idx]]  # Single bin segment

            elif end_loc > chrom_counts.index.get_loc(start_idx):
                # Slice up to *and including* the end index for full segment coverage
                segment_data = chrom_counts.iloc[
                    chrom_counts.index.get_loc(start_idx) : end_loc + 1
                ]
            else:  # Should not happen with sorted unique indices but handle just in case
                segment_data = chrom_counts.loc[[start_idx]]

            # Check minimum segment size
            if len(segment_data) < minsizeCNV:
                state = "Neutral (short)"  # Or merge with neighbor?
            else:
                # Simple assignment (pass cell baseline if needed)
                segment_median = np.median(segment_data)
                ratio = (
                    segment_median / cell_baseline_median
                    if cell_baseline_median > 0
                    else 0
                )

                if ratio < 0.75:
                    state = "Loss"
                elif ratio > 1.25:
                    state = "Gain"
                else:
                    state = "Neutral"

            start_pos = bin_info_df.loc[start_idx, "start"]
            end_pos = bin_info_df.loc[
                segment_indices[i + 1], "end"
            ]  # End pos of the last bin in segment

            cell_results.append(
                {
                    "cell_id": cell_id,
                    "seqnames": chrom,
                    "start": start_pos,
                    "end": end_pos,
                    "state": state,
                    "median_count": np.median(segment_data),  # Add median for info
                }
            )

    return pd.DataFrame(cell_results)


def run_pipeline(
    count_matrix_file,
    bin_info_file,
    output_prefix,
    threshold_cells_nbins=0.05,
    threshold_blacklist_bins=0.85,
    n_cores=4,
    k=4,
    test="AD",
    minsizeCNV=3,
):
    """Runs the full pipeline from count matrix to CNV calls."""

    print("Loading data...")
    # Load counts, index by bin_id
    counts_df = pd.read_csv(count_matrix_file, sep="\t", index_col=0)
    # Load bin info, index by bin_id
    bin_info_df = pd.read_csv(bin_info_file, sep="\t", index_col="bin_id")

    # Ensure indices match
    common_bins = counts_df.index.intersection(bin_info_df.index)
    counts_df = counts_df.loc[common_bins]
    bin_info_df = bin_info_df.loc[common_bins]
    print(f"Loaded {counts_df.shape[0]} bins and {counts_df.shape[1]} cells.")

    # --- Filtering ---
    counts_filt_cells = filter_low_coverage_cells(counts_df, threshold_cells_nbins)
    if counts_filt_cells.empty:
        print("Error: No cells remaining after filtering.")
        return

    counts_filtered, bin_info_filtered = filter_low_coverage_bins(
        counts_filt_cells, bin_info_df, threshold_blacklist_bins
    )
    if counts_filtered.empty:
        print("Error: No bins remaining after filtering.")
        return

    # Save filtered matrix (optional)
    # counts_filtered.to_csv(f"{output_prefix}_counts_filtered.tsv", sep='\t')

    # --- GC Correction ---
    gc_content = bin_info_filtered["GC"]
    counts_gc_corrected = parallel_gc_correct(counts_filtered, gc_content, n_cores)

    # Save corrected matrix (optional)
    counts_gc_corrected.to_csv(f"{output_prefix}_counts_gc_corrected.tsv", sep="\t")
    print("GC correction complete.")

    # --- CNV Calling ---
    print(f"Performing CNV calling (k={k}, test={test}, minsizeCNV={minsizeCNV})...")
    # Use joblib to parallelize over cells
    cnv_results_list = Parallel(n_jobs=n_cores)(
        delayed(process_cell_cnv)(
            cell_id,
            counts_gc_corrected[cell_id],
            bin_info_filtered,
            k=k,
            test_type=test,
            minsizeCNV=minsizeCNV,
        )
        for cell_id in counts_gc_corrected.columns
    )

    # Combine results from all cells
    cnv_results_df = pd.concat(cnv_results_list, ignore_index=True)

    # Save CNV results
    cnv_results_df.to_csv(f"{output_prefix}_cnv_calls.tsv", sep="\t", index=False)
    print(f"CNV calling complete. Results saved to {output_prefix}_cnv_calls.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="epiAneufinder-like CNV analysis pipeline (Python version)"
    )
    parser.add_argument(
        "count_matrix_file",
        help="Path to the input count matrix (TSV, bins x cells, first col=bin_id)",
    )
    parser.add_argument(
        "bin_info_file",
        help="Path to the bin information file (TSV, with bin_id, seqnames, start, end, GC columns)",
    )
    parser.add_argument(
        "output_prefix", help="Prefix for output files (e.g., 'results/sample1')"
    )

    parser.add_argument(
        "--threshold_cells_nbins",
        type=float,
        default=0.05,
        help="Filter cells with more than this fraction of zero bins (default: 0.05)",
    )
    parser.add_argument(
        "--threshold_blacklist_bins",
        type=float,
        default=0.85,
        help="Filter bins with zero counts in more than this fraction of cells (default: 0.85)",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=4,
        help="Number of cores for parallel processing (default: 4)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Segmentation level (2^k segments per chromosome) (default: 4)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="AD",
        choices=["AD", "KS"],
        help="Statistical test for breakpoints (AD or KS) (default: AD)",
    )
    parser.add_argument(
        "--minsizeCNV",
        type=int,
        default=3,
        help="Minimum number of consecutive bins to call a CNV (default: 3)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_pipeline(
        args.count_matrix_file,
        args.bin_info_file,
        args.output_prefix,
        args.threshold_cells_nbins,
        args.threshold_blacklist_bins,
        args.n_cores,
        args.k,
        args.test,
        args.minsizeCNV,
    )

    print("Pipeline finished.")


# python pyAneufinder.py counts.tsv bins.tsv path/to/output/prefix --n_cores 8 --k 5 --test AD --minsizeCNV 3
