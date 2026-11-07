"""Shared utilities for parallel processing of genomic chunks.

This module provides backend-agnostic logic for:
- Processing chromosomes sequentially
- Processing chunks within each chromosome in parallel
- Concatenating sparse matrices from chunks
- Reordering to match original peak order

Backends can use this shared logic and add their own post-processing
(e.g., writing to storage, concatenating chromosomes).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm

# Lazy import - reorder_matrix_columns imported inside function to avoid circular dependency

logger = logging.getLogger(__name__)


def process_chromosome_with_chunks(
    chromosome: str,
    peaks_chr: pd.DataFrame,
    process_chunk_fn: Callable,
    chunk_size: int,
    chunk_starts: list[int] | None = None,
    n_workers: int = 1,
    executor_class: type[ProcessPoolExecutor | ThreadPoolExecutor] = ThreadPoolExecutor,
    **chunk_kwargs: Any,
) -> tuple[csr_matrix, pd.DataFrame]:
    """
    Process a chromosome by splitting into chunks, processing in parallel, and reordering.

    This is the core shared logic for both Zarr and BPCells backends:
    1. Split chromosome into genomic chunks
    2. Process chunks in parallel
    3. Concatenate sparse matrices from chunks
    4. Reorder to match original peak order

    Args:
        chromosome: Chromosome name
        peaks_chr: DataFrame of peaks for this chromosome (must have 'Start' and 'End' columns)
        process_chunk_fn: Function to process a single chunk.
            Signature: (chromosome, chunk_start, chunk_end, peaks_chunk, **chunk_kwargs)
            Returns: (sparse_matrix, original_indices_array) or (dense_array, original_indices_array)
        chunk_size: Size of genomic chunks in base pairs
        chunk_starts: Optional list of chunk start positions. If None, auto-generate from peaks.
        n_workers: Number of parallel workers (default: 1 for sequential)
        executor_class: Executor class to use (ThreadPoolExecutor or ProcessPoolExecutor)
        **chunk_kwargs: Additional arguments to pass to process_chunk_fn

    Returns:
        Tuple of (sparse_matrix, peaks_dataframe):
        - sparse_matrix: Concatenated and reordered sparse matrix for this chromosome
        - peaks_dataframe: Peaks DataFrame matching the matrix columns

    Examples:
        >>> def my_chunk_processor(chrom, start, end, peaks, **kwargs):
        ...     # Process chunk and return (matrix, original_indices)
        ...     return csr_matrix(...), np.array([...])
        ...
        >>> matrix, peaks = process_chromosome_with_chunks(
        ...     "chr1", peaks_df, my_chunk_processor, chunk_size=2000000
        ... )
    """
    if peaks_chr.empty:
        logger.warning(f"No peaks for {chromosome}, returning empty matrix")
        return csr_matrix((0, 0)), peaks_chr.copy()

    # Reset index to ensure continuous indices
    peaks_chr = peaks_chr.reset_index(drop=True)

    # Generate chunk starts if not provided
    if chunk_starts is None:
        chrom_start = int(peaks_chr["Start"].min())
        chrom_end = int(peaks_chr["End"].max())
        chunk_starts = list(range(chrom_start, chrom_end, chunk_size))

    # Prepare chunk tasks
    chunk_tasks = []
    for chunk_start in chunk_starts:
        chunk_end_pos = chunk_start + chunk_size
        # Get peaks overlapping this chunk (preserve original index)
        chunk_peaks = peaks_chr[
            (peaks_chr["Start"] < chunk_end_pos) & (peaks_chr["End"] > chunk_start)
        ].copy()

        if len(chunk_peaks) > 0:
            # Store original indices (global position in peaks_chr) before resetting index
            # This ensures we can track the original order after any filtering
            chunk_peaks = chunk_peaks.copy()
            chunk_peaks["original_indices"] = chunk_peaks.index.values
            chunk_tasks.append((chunk_start, chunk_end_pos, chunk_peaks))

    if not chunk_tasks:
        logger.warning(f"No chunks with peaks for {chromosome}")
        return csr_matrix((0, 0)), peaks_chr.copy()

    # Process chunks in parallel or sequentially
    chunk_results = []

    if n_workers > 1:
        # Parallel processing
        with executor_class(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    process_chunk_fn,
                    chromosome,
                    chunk_start,
                    chunk_end_pos,
                    chunk_peaks,
                    **chunk_kwargs,
                ): (chunk_start, chunk_end_pos, chunk_peaks)
                for chunk_start, chunk_end_pos, chunk_peaks in chunk_tasks
            }

            # Collect results as they complete
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {chromosome} chunks",
                leave=False,
            ):
                chunk_start, chunk_end_pos, chunk_peaks = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        chunk_matrix, original_indices = result
                        # Ensure original_indices are from chunk_peaks
                        if original_indices is None:
                            original_indices = chunk_peaks["original_indices"].values
                        chunk_results.append((chunk_matrix, original_indices))
                except Exception as e:
                    logger.error(
                        f"Error processing {chromosome} chunk {chunk_start}-{chunk_end_pos}: {e}"
                    )
                    # Create empty matrix for this chunk
                    n_cells = chunk_kwargs.get("n_cells", 0)
                    if n_cells == 0:
                        # Try to infer from first successful result
                        if chunk_results:
                            n_cells = chunk_results[0][0].shape[0]
                        else:
                            n_cells = 1
                    empty_matrix = csr_matrix((n_cells, len(chunk_peaks)))
                    original_indices = chunk_peaks["original_indices"].values
                    chunk_results.append((empty_matrix, original_indices))
    else:
        # Sequential processing (for debugging or single-threaded)
        for chunk_start, chunk_end_pos, chunk_peaks in tqdm(
            chunk_tasks, desc=f"Processing {chromosome} chunks", leave=False
        ):
            try:
                result = process_chunk_fn(
                    chromosome, chunk_start, chunk_end_pos, chunk_peaks, **chunk_kwargs
                )
                if result is not None:
                    chunk_matrix, original_indices = result
                    if original_indices is None:
                        original_indices = chunk_peaks["original_indices"].values
                    chunk_results.append((chunk_matrix, original_indices))
            except Exception as e:
                logger.error(
                    f"Error processing {chromosome} chunk {chunk_start}-{chunk_end_pos}: {e}"
                )
                n_cells = chunk_kwargs.get("n_cells", 1)
                empty_matrix = csr_matrix((n_cells, len(chunk_peaks)))
                original_indices = chunk_peaks["original_indices"].values
                chunk_results.append((empty_matrix, original_indices))

    if not chunk_results:
        logger.warning(f"No results for {chromosome}")
        return csr_matrix((0, 0)), peaks_chr.copy()

    # Concatenate all chunk results (sparse)
    concat_start = time.time()
    matrices = []
    for chunk_matrix, _ in chunk_results:
        # Convert to sparse if needed (some backends return dense arrays)
        if isinstance(chunk_matrix, np.ndarray):
            chunk_matrix = csr_matrix(chunk_matrix)
        matrices.append(chunk_matrix)

    result = hstack(matrices)
    concat_time = time.time() - concat_start
    logger.debug(f"Concatenated {len(chunk_results)} chunks in {concat_time:.4f}s")

    # Collect all original indices in concatenated order
    all_original_indices = np.concatenate([result[1] for result in chunk_results])

    # Reorder to match original peak order
    reorder_start = time.time()
    sort_order = np.argsort(all_original_indices)
    # Lazy import to avoid circular dependency
    from caesar.io import reorder_matrix_columns
    result = reorder_matrix_columns(result, sort_order)
    reorder_time = time.time() - reorder_start
    logger.debug(f"Reordered matrix in {reorder_time:.4f}s")

    # Get peaks in the correct order (matching matrix columns)
    # Use sort_order to reorder peaks_chr to match the matrix
    reordered_peaks = peaks_chr.iloc[sort_order].reset_index(drop=True)

    return result, reordered_peaks


def process_all_chromosomes(
    peaks: pd.DataFrame,
    process_chromosome_fn: Callable,
    chromosomes: list[str] | None = None,
    **chromosome_kwargs: Any,
) -> dict[str, tuple[csr_matrix, pd.DataFrame]]:
    """
    Process all chromosomes sequentially and return per-chromosome results.

    This is a convenience function that processes chromosomes one by one,
    allowing backends to handle final assembly (e.g., concatenation, writing).

    Args:
        peaks: DataFrame of all peaks with 'Chromosome' column
        process_chromosome_fn: Function to process a single chromosome.
            Signature: (chromosome, peaks_chr, **chromosome_kwargs)
            Returns: (sparse_matrix, peaks_dataframe)
        chromosomes: Optional list of chromosomes to process. If None, uses all unique chromosomes.
        **chromosome_kwargs: Additional arguments to pass to process_chromosome_fn

    Returns:
        Dictionary mapping chromosome names to (sparse_matrix, peaks_dataframe) tuples

    Examples:
        >>> def my_chr_processor(chrom, peaks_chr, **kwargs):
        ...     # Process chromosome and return (matrix, peaks)
        ...     return csr_matrix(...), peaks_chr
        ...
        >>> results = process_all_chromosomes(peaks_df, my_chr_processor)
        >>> chr1_matrix, chr1_peaks = results["chr1"]
    """
    if chromosomes is None:
        chromosomes = peaks["Chromosome"].unique().tolist()

    results = {}
    for chrom in tqdm(chromosomes, desc="Processing chromosomes"):
        chrom_peaks = peaks[peaks["Chromosome"] == chrom].copy()
        if len(chrom_peaks) == 0:
            continue

        try:
            matrix, peaks_df = process_chromosome_fn(chrom, chrom_peaks, **chromosome_kwargs)
            results[chrom] = (matrix, peaks_df)
        except Exception as e:
            logger.error(f"Error processing {chrom}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results

