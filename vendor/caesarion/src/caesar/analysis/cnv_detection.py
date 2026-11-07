# Fast CNV detection from ATAC-seq tile count matrices
# Designed for 500bp tile matrices from Caesar
# Uses log2 ratio + HMM/rolling segmentation for speed

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=RuntimeWarning)


def gc_correct_fast(
    counts: np.ndarray,
    gc_content: np.ndarray,
    method: Literal["linear", "median_bin"] = "median_bin",
    n_gc_bins: int = 20,
) -> np.ndarray:
    """
    Fast GC correction using binned median approach.

    Parameters
    ----------
    counts : np.ndarray
        Count matrix (bins x cells)
    gc_content : np.ndarray
        GC content per bin (0-1)
    method : str
        'linear' for linear regression, 'median_bin' for binned median (faster)
    n_gc_bins : int
        Number of GC content bins for median method

    Returns
    -------
    np.ndarray
        GC-corrected count matrix
    """
    if issparse(counts):
        counts = np.asarray(counts.toarray())  # type: ignore[union-attr]

    counts = np.asarray(counts, dtype=np.float64)
    gc_content = np.asarray(gc_content)
    corrected = np.zeros_like(counts)

    if method == "median_bin":
        # Bin GC content and compute median count per bin
        gc_bins = np.digitize(gc_content, np.linspace(0, 1, n_gc_bins + 1))

        for cell_idx in range(counts.shape[1]):
            cell_counts = counts[:, cell_idx]
            global_median = np.median(cell_counts[cell_counts > 0])

            if np.isnan(global_median) or global_median == 0:
                corrected[:, cell_idx] = cell_counts
                continue

            # Compute expected counts per GC bin
            expected = np.zeros(len(cell_counts))
            for gc_bin in range(1, n_gc_bins + 2):
                mask = gc_bins == gc_bin
                if mask.sum() > 0:
                    bin_median = np.median(cell_counts[mask])
                    if bin_median > 0:
                        expected[mask] = bin_median
                    else:
                        expected[mask] = global_median

            # Correct counts
            correction_factor = global_median / np.maximum(expected, 1e-6)
            corrected[:, cell_idx] = cell_counts * correction_factor

    elif method == "linear":
        # Fast linear regression based correction
        for cell_idx in range(counts.shape[1]):
            cell_counts = counts[:, cell_idx]
            mask = cell_counts > 0

            if mask.sum() < 10:
                corrected[:, cell_idx] = cell_counts
                continue

            # Log-linear regression
            log_counts = np.log1p(cell_counts[mask])
            gc_masked = gc_content[mask]

            # Simple linear fit
            slope, intercept = np.polyfit(gc_masked, log_counts, 1)
            expected = np.exp(slope * gc_content + intercept) - 1

            # Correct
            global_median = np.median(cell_counts[mask])
            correction_factor = global_median / np.maximum(expected, 1e-6)
            corrected[:, cell_idx] = cell_counts * correction_factor

    return np.maximum(corrected, 0)


def normalize_library_size(counts: np.ndarray, target_sum: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize counts by library size (CPM-like).

    Returns
    -------
    tuple
        (normalized_counts, library_sizes)
    """
    lib_sizes = counts.sum(axis=0)
    if target_sum is None:
        target_sum = np.median(lib_sizes[lib_sizes > 0])

    normalized = counts / np.maximum(lib_sizes, 1) * target_sum
    return normalized, lib_sizes


def compute_log2_ratio(
    counts: np.ndarray,
    reference: np.ndarray | None = None,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """
    Compute log2 ratio vs reference.

    Parameters
    ----------
    counts : np.ndarray
        Normalized count matrix (bins x cells)
    reference : np.ndarray, optional
        Reference profile. If None, use median across cells.
    pseudocount : float
        Pseudocount to add before log transform

    Returns
    -------
    np.ndarray
        Log2 ratio matrix
    """
    if reference is None:
        reference = np.median(counts, axis=1)

    # Handle zero reference - ensure reference is always an ndarray
    reference = np.asarray(reference)
    reference = np.maximum(reference, pseudocount)

    log2_ratio = np.log2((counts + pseudocount) / reference[:, np.newaxis])
    return log2_ratio


def smooth_signal(
    signal: np.ndarray,
    method: Literal["rolling", "gaussian", "wavelet"] = "rolling",
    window_size: int = 21,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Smooth signal to reduce noise.

    Parameters
    ----------
    signal : np.ndarray
        Signal matrix (bins x cells)
    method : str
        'rolling' for rolling median, 'gaussian' for Gaussian filter
    window_size : int
        Window size for rolling median
    sigma : float
        Sigma for Gaussian filter

    Returns
    -------
    np.ndarray
        Smoothed signal
    """
    smoothed = np.zeros_like(signal)

    if method == "rolling":
        # Use pandas for efficient rolling operations
        for cell_idx in range(signal.shape[1]):
            smoothed[:, cell_idx] = (
                pd.Series(signal[:, cell_idx])
                .rolling(window_size, center=True, min_periods=1)
                .median()
                .values
            )

    elif method == "gaussian":
        for cell_idx in range(signal.shape[1]):
            smoothed[:, cell_idx] = ndimage.gaussian_filter1d(
                signal[:, cell_idx], sigma=sigma, mode="nearest"
            )

    return smoothed


def segment_chromosome_cbs(
    signal: np.ndarray,
    min_segment_size: int = 10,
    significance_threshold: float = 0.01,
    max_segments: int = 20,
) -> list[tuple[int, int]]:
    """
    Circular Binary Segmentation (simplified, fast version).

    Uses t-test to find breakpoints instead of permutation test.

    Parameters
    ----------
    signal : np.ndarray
        1D signal array for one chromosome
    min_segment_size : int
        Minimum segment size in bins
    significance_threshold : float
        P-value threshold for segmentation
    max_segments : int
        Maximum number of segments per chromosome

    Returns
    -------
    list of tuples
        List of (start, end) indices for segments
    """
    n = len(signal)

    if n < 2 * min_segment_size:
        return [(0, n)]

    # Remove NaN
    nan_fill = float(np.nanmedian(signal))
    signal_clean = np.nan_to_num(signal, nan=nan_fill)

    breakpoints = [0]

    def find_best_split(start: int, end: int) -> tuple[int, float]:
        """Find best split point using t-test statistic."""
        best_stat = 0
        best_pos = -1

        segment = signal_clean[start:end]
        seg_len = len(segment)

        if seg_len < 2 * min_segment_size:
            return -1, 0

        for i in range(min_segment_size, seg_len - min_segment_size):
            left = segment[:i]
            right = segment[i:]

            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(left, right, equal_var=False)

            if abs(t_stat) > best_stat:
                best_stat = abs(t_stat)
                best_pos = start + i

        return best_pos, best_stat

    # Iteratively find breakpoints
    segments_to_split = [(0, n)]

    while len(breakpoints) < max_segments and segments_to_split:
        # Find segment with best potential split
        best_overall_stat = 0
        best_segment_idx = -1
        best_split_pos = -1

        for idx, (start, end) in enumerate(segments_to_split):
            pos, stat = find_best_split(start, end)
            if stat > best_overall_stat:
                best_overall_stat = stat
                best_segment_idx = idx
                best_split_pos = pos

        if best_split_pos == -1:
            break

        # Check significance (approximate using t-distribution)
        df = n - 2
        p_val = 2 * (1 - stats.t.cdf(best_overall_stat, df))

        if p_val > significance_threshold:
            break

        # Add breakpoint
        breakpoints.append(best_split_pos)
        old_start, old_end = segments_to_split.pop(best_segment_idx)

        # Add new segments to check
        if best_split_pos - old_start >= 2 * min_segment_size:
            segments_to_split.append((old_start, best_split_pos))
        if old_end - best_split_pos >= 2 * min_segment_size:
            segments_to_split.append((best_split_pos, old_end))

    breakpoints.append(n)
    breakpoints = sorted(set(breakpoints))

    # Convert to segment list
    segments = [(breakpoints[i], breakpoints[i + 1]) for i in range(len(breakpoints) - 1)]
    return segments


def assign_cnv_states(
    log2_ratio: np.ndarray,
    segments: list[tuple[int, int]],
    loss_threshold: float = -0.3,
    gain_threshold: float = 0.3,
) -> np.ndarray:
    """
    Assign CNV states to segments based on log2 ratio.

    Parameters
    ----------
    log2_ratio : np.ndarray
        Log2 ratio values (1D, single cell/chromosome)
    segments : list of tuples
        Segment boundaries
    loss_threshold : float
        Log2 ratio threshold for loss (typically -0.3 to -0.5)
    gain_threshold : float
        Log2 ratio threshold for gain (typically 0.3 to 0.5)

    Returns
    -------
    np.ndarray
        CNV states: -1 (loss), 0 (neutral), 1 (gain)
    """
    states = np.zeros(len(log2_ratio), dtype=np.int8)

    for start, end in segments:
        seg_median = np.nanmedian(log2_ratio[start:end])

        if seg_median < loss_threshold:
            states[start:end] = -1
        elif seg_median > gain_threshold:
            states[start:end] = 1
        # else remains 0 (neutral)

    return states


def joint_segmentation(
    log2_ratio_matrix: np.ndarray,
    chrom_indices: dict[str, tuple[int, int]],
    min_segment_size: int = 10,
    use_median_profile: bool = True,
) -> dict[str, list[tuple[int, int]]]:
    """
    Joint segmentation across cells - find shared breakpoints.

    This is much faster than per-cell segmentation and leverages
    the fact that CNV breakpoints are often shared across cells.

    Parameters
    ----------
    log2_ratio_matrix : np.ndarray
        Log2 ratio matrix (bins x cells)
    chrom_indices : dict
        Chromosome -> (start_idx, end_idx) mapping
    min_segment_size : int
        Minimum segment size
    use_median_profile : bool
        If True, segment the median profile. If False, segment variance profile.

    Returns
    -------
    dict
        Chromosome -> list of segments
    """
    all_segments = {}

    for chrom, (start, end) in chrom_indices.items():
        chrom_data = log2_ratio_matrix[start:end, :]

        if use_median_profile:
            # Segment based on median across cells
            profile = np.nanmedian(chrom_data, axis=1)
        else:
            # Segment based on variance (high variance = potential CNV boundary)
            profile = np.nanvar(chrom_data, axis=1)

        segments = segment_chromosome_cbs(
            profile,
            min_segment_size=min_segment_size,
        )

        # Adjust indices to global coordinates
        all_segments[chrom] = [(s + start, e + start) for s, e in segments]

    return all_segments


def detect_cnv(
    counts_matrix: np.ndarray | pd.DataFrame,
    bin_info: pd.DataFrame,
    reference_cells: np.ndarray | None = None,
    gc_correction: bool = True,
    gc_method: Literal["linear", "median_bin"] = "median_bin",
    smooth_method: Literal["rolling", "gaussian"] = "rolling",
    window_size: int = 21,
    min_segment_size: int = 10,
    loss_threshold: float = -0.3,
    gain_threshold: float = 0.3,
    joint_segment: bool = True,
    n_jobs: int = 1,
) -> dict:
    """
    Main CNV detection function.

    Parameters
    ----------
    counts_matrix : np.ndarray or pd.DataFrame
        Count matrix (bins x cells)
    bin_info : pd.DataFrame
        Bin information with columns: seqnames, start, end, GC
    reference_cells : np.ndarray, optional
        Indices of reference (diploid) cells. If None, use median of all cells.
    gc_correction : bool
        Whether to perform GC correction
    gc_method : str
        GC correction method
    smooth_method : str
        Signal smoothing method
    window_size : int
        Window size for smoothing
    min_segment_size : int
        Minimum segment size in bins
    loss_threshold : float
        Log2 ratio threshold for calling loss
    gain_threshold : float
        Log2 ratio threshold for calling gain
    joint_segment : bool
        If True, use joint segmentation across cells (faster)
    n_jobs : int
        Number of parallel jobs (for per-cell segmentation)

    Returns
    -------
    dict with keys:
        'cnv_states': CNV state matrix (bins x cells)
        'log2_ratio': Log2 ratio matrix
        'smoothed': Smoothed log2 ratio
        'segments': Segment boundaries per chromosome
        'summary': Summary statistics
    """
    # Convert to numpy if needed
    if isinstance(counts_matrix, pd.DataFrame):
        cell_ids = counts_matrix.columns.tolist()
        bin_ids = counts_matrix.index.tolist()
        counts = counts_matrix.values.astype(np.float64)
    else:
        counts = counts_matrix.astype(np.float64)
        cell_ids = list(range(counts.shape[1]))
        bin_ids = list(range(counts.shape[0]))

    n_bins, n_cells = counts.shape
    print(f"Processing {n_cells} cells x {n_bins} bins")

    # 1. GC correction
    if gc_correction and "GC" in bin_info.columns:
        print("Performing GC correction...")
        gc_content = bin_info["GC"].values
        counts = gc_correct_fast(counts, gc_content, method=gc_method)

    # 2. Library size normalization
    print("Normalizing library sizes...")
    normalized, lib_sizes = normalize_library_size(counts)

    # 3. Compute reference
    if reference_cells is not None:
        reference = np.median(normalized[:, reference_cells], axis=1)
        print(f"Using {len(reference_cells)} reference cells")
    else:
        reference = np.median(normalized, axis=1)
        print("Using median of all cells as reference")

    # 4. Log2 ratio
    print("Computing log2 ratios...")
    log2_ratio = compute_log2_ratio(normalized, reference)

    # 5. Smoothing
    print(f"Smoothing with {smooth_method} (window={window_size})...")
    smoothed = smooth_signal(log2_ratio, method=smooth_method, window_size=window_size)

    # 6. Build chromosome index
    chrom_indices = {}
    for chrom in bin_info["seqnames"].unique():
        mask = bin_info["seqnames"] == chrom
        indices = np.where(mask)[0]
        if len(indices) > 0:
            chrom_indices[chrom] = (indices[0], indices[-1] + 1)

    # 7. Segmentation
    if joint_segment:
        print("Performing joint segmentation...")
        segments = joint_segmentation(
            smoothed, chrom_indices, min_segment_size=min_segment_size
        )
    else:
        # Per-cell segmentation (slower but may be more accurate for heterogeneous samples)
        print("Performing per-cell segmentation...")
        segments = {}
        for chrom, (start, end) in chrom_indices.items():
            # Use first cell's segments as template (simplification)
            segments[chrom] = segment_chromosome_cbs(
                smoothed[start:end, 0], min_segment_size=min_segment_size
            )
            segments[chrom] = [(s + start, e + start) for s, e in segments[chrom]]

    # 8. Assign CNV states
    print("Assigning CNV states...")
    cnv_states = np.zeros((n_bins, n_cells), dtype=np.int8)

    for chrom, seg_list in segments.items():
        for cell_idx in range(n_cells):
            cell_states = assign_cnv_states(
                smoothed[:, cell_idx],
                seg_list,
                loss_threshold=loss_threshold,
                gain_threshold=gain_threshold,
            )
            for start, end in seg_list:
                cnv_states[start:end, cell_idx] = cell_states[start:end]

    # 9. Summary statistics
    n_gains = (cnv_states == 1).sum(axis=0)
    n_losses = (cnv_states == -1).sum(axis=0)

    summary = pd.DataFrame({
        "cell_id": cell_ids,
        "library_size": lib_sizes,
        "n_gain_bins": n_gains,
        "n_loss_bins": n_losses,
        "pct_altered": (n_gains + n_losses) / n_bins * 100,
    })

    print(f"Done. Mean {summary['pct_altered'].mean():.1f}% bins altered per cell.")

    return {
        "cnv_states": pd.DataFrame(cnv_states, index=bin_ids, columns=cell_ids),
        "log2_ratio": pd.DataFrame(log2_ratio, index=bin_ids, columns=cell_ids),
        "smoothed": pd.DataFrame(smoothed, index=bin_ids, columns=cell_ids),
        "segments": segments,
        "summary": summary,
        "bin_info": bin_info,
    }


def identify_diploid_cells(
    cnv_result: dict,
    max_altered_pct: float = 5.0,
) -> np.ndarray:
    """
    Identify likely diploid cells for use as reference.

    Parameters
    ----------
    cnv_result : dict
        Result from detect_cnv (first pass)
    max_altered_pct : float
        Maximum percentage of altered bins to consider diploid

    Returns
    -------
    np.ndarray
        Indices of likely diploid cells
    """
    summary = cnv_result["summary"]
    diploid_mask = summary["pct_altered"] < max_altered_pct

    diploid_indices = np.where(diploid_mask)[0]
    print(f"Identified {len(diploid_indices)} likely diploid cells")

    return diploid_indices


def detect_cnv_iterative(
    counts_matrix: np.ndarray | pd.DataFrame,
    bin_info: pd.DataFrame,
    max_iterations: int = 2,
    diploid_threshold: float = 5.0,
    **kwargs,
) -> dict:
    """
    Iterative CNV detection with automatic diploid reference identification.

    First pass uses all cells as reference, identifies likely diploid cells,
    then re-runs with diploid cells as reference.

    Parameters
    ----------
    counts_matrix : np.ndarray or pd.DataFrame
        Count matrix
    bin_info : pd.DataFrame
        Bin information
    max_iterations : int
        Maximum iterations
    diploid_threshold : float
        Altered percentage threshold for diploid classification
    **kwargs
        Additional arguments passed to detect_cnv

    Returns
    -------
    dict
        Final CNV detection result
    """
    result = None
    reference_cells = None

    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")

        result = detect_cnv(
            counts_matrix,
            bin_info,
            reference_cells=reference_cells,
            **kwargs,
        )

        # Identify diploid cells for next iteration
        diploid_cells = identify_diploid_cells(result, max_altered_pct=diploid_threshold)

        if len(diploid_cells) == 0:
            print("Warning: No diploid cells identified, using all cells as reference")
            break

        if reference_cells is not None and len(diploid_cells) == len(reference_cells):
            if np.array_equal(sorted(diploid_cells), sorted(reference_cells)):
                print("Converged - same diploid cells as previous iteration")
                break

        reference_cells = diploid_cells

    return result


def create_cnv_profile_matrix(
    cnv_result: dict,
    resolution: Literal["bin", "segment", "arm", "chromosome"] = "segment",
) -> pd.DataFrame:
    """
    Create a CNV profile matrix at different resolutions.

    Parameters
    ----------
    cnv_result : dict
        Result from detect_cnv
    resolution : str
        Resolution level: 'bin', 'segment', 'arm', 'chromosome'

    Returns
    -------
    pd.DataFrame
        CNV profile matrix
    """
    cnv_states = cnv_result["cnv_states"]
    bin_info = cnv_result["bin_info"]

    if resolution == "bin":
        return cnv_states

    elif resolution == "segment":
        segments = cnv_result["segments"]
        seg_profiles = []
        seg_names = []

        for chrom, seg_list in segments.items():
            for i, (start, end) in enumerate(seg_list):
                seg_data = cnv_states.iloc[start:end, :].mean(axis=0)
                seg_profiles.append(seg_data)
                start_pos = bin_info.iloc[start]["start"]
                end_pos = bin_info.iloc[end - 1]["end"]
                seg_names.append(f"{chrom}:{start_pos}-{end_pos}")

        return pd.DataFrame(seg_profiles, index=seg_names).T

    elif resolution == "chromosome":
        chrom_profiles = []
        for chrom in bin_info["seqnames"].unique():
            mask = bin_info["seqnames"] == chrom
            chrom_data = cnv_states.loc[mask, :].mean(axis=0)
            chrom_profiles.append(chrom_data)

        return pd.DataFrame(
            chrom_profiles, index=bin_info["seqnames"].unique()
        ).T

    else:
        raise ValueError(f"Unknown resolution: {resolution}")


# Convenience function for integration with Caesar
def detect_cnv_from_zarr(
    zarr_path: str,
    output_path: str | None = None,
    **kwargs,
) -> dict:
    """
    Detect CNVs from a Caesar zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to zarr store with tile counts
    output_path : str, optional
        Path to save results
    **kwargs
        Arguments passed to detect_cnv

    Returns
    -------
    dict
        CNV detection results
    """
    # Import here to avoid circular imports
    from caesar.io import load_io

    io = load_io(zarr_path)

    # Get count matrix and bin info
    # This assumes a tile matrix format
    counts = io.get_data()
    bin_info = io.get_peaks()  # or equivalent bin information

    result = detect_cnv(counts, bin_info, **kwargs)

    if output_path:
        result["cnv_states"].to_csv(f"{output_path}_cnv_states.tsv", sep="\t")
        result["summary"].to_csv(f"{output_path}_summary.tsv", sep="\t", index=False)
        print(f"Results saved to {output_path}_*")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast CNV detection from ATAC-seq tile count matrices"
    )
    parser.add_argument(
        "counts_file",
        help="Path to count matrix (TSV, bins x cells)",
    )
    parser.add_argument(
        "bin_info_file",
        help="Path to bin info (TSV with seqnames, start, end, GC)",
    )
    parser.add_argument(
        "output_prefix",
        help="Output prefix",
    )
    parser.add_argument(
        "--gc-correction",
        action="store_true",
        default=True,
        help="Perform GC correction",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=21,
        help="Smoothing window size in bins",
    )
    parser.add_argument(
        "--min-segment-size",
        type=int,
        default=10,
        help="Minimum segment size in bins",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=-0.3,
        help="Log2 ratio threshold for loss",
    )
    parser.add_argument(
        "--gain-threshold",
        type=float,
        default=0.3,
        help="Log2 ratio threshold for gain",
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Use iterative detection with diploid reference",
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    counts_df = pd.read_csv(args.counts_file, sep="\t", index_col=0)
    bin_info_df = pd.read_csv(args.bin_info_file, sep="\t", index_col=0)

    # Ensure index alignment
    common_bins = counts_df.index.intersection(bin_info_df.index)
    counts_df = counts_df.loc[common_bins]
    bin_info_df = bin_info_df.loc[common_bins]

    # Run detection
    if args.iterative:
        result = detect_cnv_iterative(
            counts_df,
            bin_info_df,
            gc_correction=args.gc_correction,
            window_size=args.window_size,
            min_segment_size=args.min_segment_size,
            loss_threshold=args.loss_threshold,
            gain_threshold=args.gain_threshold,
        )
    else:
        result = detect_cnv(
            counts_df,
            bin_info_df,
            gc_correction=args.gc_correction,
            window_size=args.window_size,
            min_segment_size=args.min_segment_size,
            loss_threshold=args.loss_threshold,
            gain_threshold=args.gain_threshold,
        )

    # Save results
    result["cnv_states"].to_csv(f"{args.output_prefix}_cnv_states.tsv", sep="\t")
    result["log2_ratio"].to_csv(f"{args.output_prefix}_log2_ratio.tsv", sep="\t")
    result["summary"].to_csv(f"{args.output_prefix}_summary.tsv", sep="\t", index=False)

    print(f"\nResults saved to {args.output_prefix}_*")
