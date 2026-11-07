"""Shared utilities for filtering genome tiles by chromosomes and blacklist.

This module provides unified logic for filtering tiles, ensuring consistency
between Zarr and BPCells backends. However, the backends use different approaches:

**Zarr Backend - Two-Stage Filtering:**
1. **Preprocessing (Fragment Processor)**: Chromosome filtering only
   - `create_genome_tiles()` in `fragment_utils.py` applies chromosome filtering
   - Removes patches, chrM, chrY, etc. for efficiency
   - Stores chromosome-filtered tiles in Zarr under "tiles/regions/"

2. **Analysis (Clustering/AnnData Generation)**: Blacklist filtering on stored tiles
   - `ClusteringMixin.create_genome_tiles()` reads stored tiles from Zarr
   - Applies blacklist filtering on top of chromosome-filtered tiles
   - Computes matrix with blacklist-filtered tiles
   - Matrix columns directly match filtered tiles (no column selection needed)

**BPCells Backend - Post-Matrix Filtering:**
- Creates matrix with UNFILTERED chrom_sizes (all chromosomes, including patches, chrM, chrY)
- Filters tiles AFTER matrix creation
- Uses `create_column_indices()` to select correct columns from the full matrix
- Used in: `BPCellsProcessor._process_anndata()` (creates matrix, then filters tiles and selects columns)

**Key Difference:**
- **Zarr**: Filters tiles before matrix computation (two-stage: chromosome in preprocessing, blacklist in analysis)
- **BPCells**: Filters tiles after matrix creation (single-stage: both filters applied together after matrix)

Both backends use the same filtering logic (chromosome + blacklist) but apply it at different stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize column names to standard format (Chromosome, Start, End).

    Args:
        df: DataFrame with chromosome/start/end columns (case-insensitive)

    Returns:
        Tuple of (normalized_df, reverse_mapping) where reverse_mapping can be used
        to restore original column names
    """
    df_copy = df.copy()
    col_mapping = {}

    for col in df_copy.columns:
        col_lower = col.lower()
        if col_lower in ["chrom", "chromosome"]:
            col_mapping[col] = "Chromosome"
        elif col_lower == "start":
            col_mapping[col] = "Start"
        elif col_lower == "end":
            col_mapping[col] = "End"

    if col_mapping:
        df_copy = df_copy.rename(columns=col_mapping)
        reverse_mapping = {v: k for k, v in col_mapping.items()}
    else:
        reverse_mapping = {}

    return df_copy, reverse_mapping


def _load_blacklist(
    blacklist: str | pd.DataFrame | Path | None,
    assembly: str = "hg38",
) -> pd.DataFrame:
    """Load blacklist from various sources.

    Args:
        blacklist: Path to blacklist file, DataFrame, or None for assembly-specific
        assembly: Genome assembly (default: 'hg38')

    Returns:
        DataFrame with 'Chromosome', 'Start', 'End' columns
    """
    # Normalize empty strings to None
    if isinstance(blacklist, str) and not blacklist.strip():
        blacklist = None
    
    if blacklist is None:
        from gcell.dna.genome import read_blacklist

        blacklist_df = read_blacklist(assembly)
        logger.info(f"Loaded assembly-specific blacklist for {assembly}")
    elif isinstance(blacklist, (str, Path)):
        blacklist_path = Path(blacklist)
        if not blacklist_path.exists():
            logger.warning(f"Blacklist file not found: {blacklist_path}, using assembly-specific blacklist")
            from gcell.dna.genome import read_blacklist
            blacklist_df = read_blacklist(assembly)
        else:
            blacklist_df = pd.read_csv(blacklist, sep="\t", header=None)
        if len(blacklist_df.columns) < 3:
            raise ValueError(
                f"Blacklist file must have at least 3 columns (Chromosome, Start, End)"
            )
        blacklist_df.columns = ["Chromosome", "Start", "End"] + [
            f"col_{i}" for i in range(3, len(blacklist_df.columns))
        ]
        logger.info(f"Loaded blacklist from {blacklist}")
    else:
        blacklist_df = blacklist.copy()

    return blacklist_df


def filter_tiles_by_chromosomes(
    tiles: pd.DataFrame,
    chr_filter_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Filter tiles to keep only standard chromosomes.

    Removes patches, chrM, chrY, and other non-standard chromosomes using
    the shared ChromosomeFilter logic.

    **Used by: BOTH Zarr and BPCells backends**
    - **Zarr Preprocessing**: Called in `fragment_utils.create_genome_tiles()` during
      fragment processing to store chromosome-filtered tiles in Zarr
    - **Zarr Analysis**: Called in `ClusteringMixin.create_genome_tiles()` as fallback
      if stored tiles not found
    - **BPCells**: Called after matrix creation (in `filter_tiles()`)

    Args:
        tiles: DataFrame with 'Chromosome' column
        chr_filter_config: Optional chromosome filter configuration dict.
            If None, uses default filtering (max_length=5, excludes chrM, chrY, etc.)

    Returns:
        Filtered DataFrame with only standard chromosomes

    Examples:
        >>> tiles = pd.DataFrame({
        ...     'Chromosome': ['chr1', 'chr2', 'chrM', 'chrY', 'chrUn_GL000218v1'],
        ...     'Start': [0, 0, 0, 0, 0],
        ...     'End': [100, 100, 100, 100, 100]
        ... })
        >>> filtered = filter_tiles_by_chromosomes(tiles)
        >>> set(filtered['Chromosome'].unique())
        {'chr1', 'chr2'}
    """
    from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

    # Use shared ChromosomeFilter
    config = chr_filter_config if chr_filter_config is not None else {}
    chr_filter = ChromosomeFilter(config)

    # Filter tiles using the shared logic
    valid_chroms = set(
        chr_filter.filter_chromosomes(tiles["Chromosome"].unique().tolist())
    )
    filtered = tiles[tiles["Chromosome"].isin(valid_chroms)]

    if len(filtered) < len(tiles):
        logger.info(
            f"Filtered tiles: {len(tiles)} -> {len(filtered)} "
            f"({len(tiles) - len(filtered)} removed by chromosome filtering)"
        )

    # Reset index to ensure continuous indices [0, 1, 2, ...]
    # This is important for peak ordering to work correctly
    filtered = filtered.reset_index(drop=True)

    return filtered


def filter_tiles_by_blacklist(
    tiles: pd.DataFrame,
    blacklist: str | pd.DataFrame | None = None,
    assembly: str = "hg38",
) -> pd.DataFrame:
    """Filter tiles to remove those overlapping with blacklist regions.

    **Used by: BOTH Zarr and BPCells backends**
    - **Zarr Analysis**: Called in `ClusteringMixin.create_genome_tiles()` AFTER reading
      stored chromosome-filtered tiles from Zarr. Applies blacklist filtering on top of
      stored tiles, then computes matrix with blacklist-filtered tiles.
    - **BPCells**: Called AFTER matrix creation (in `filter_tiles()`) to create filtered
      tiles DataFrame, then `create_column_indices()` selects matching columns

    **Note:** This function does NOT handle `blacklist=False`. Use `filter_tiles()`
    with `blacklist=False` to skip blacklist filtering, or check before calling this function.

    Args:
        tiles: DataFrame with 'Chromosome', 'Start', 'End' columns
        blacklist: Path to blacklist file or DataFrame. If None, uses assembly-specific blacklist.
        assembly: Genome assembly (default: 'hg38')

    Returns:
        Filtered DataFrame with blacklisted tiles removed

    Examples:
        >>> tiles = pd.DataFrame({
        ...     'Chromosome': ['chr1', 'chr1', 'chr2'],
        ...     'Start': [1000, 5000, 1000],
        ...     'End': [1500, 5500, 1500]
        ... })
        >>> # blacklist overlaps with first tile
        >>> blacklist = pd.DataFrame({
        ...     'Chromosome': ['chr1'],
        ...     'Start': [1200],
        ...     'End': [1300]
        ... })
        >>> filtered = filter_tiles_by_blacklist(tiles, blacklist=blacklist)
        >>> len(filtered)
        2
    """
    # Load blacklist
    blacklist_df = _load_blacklist(blacklist, assembly)

    # Early return if blacklist is empty
    if len(blacklist_df) == 0:
        logger.info("Blacklist is empty, no tiles filtered")
        return tiles

    # Filter blacklist to standard chromosomes using shared ChromosomeFilter
    from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

    chr_filter = ChromosomeFilter({})
    valid_chroms = set(
        chr_filter.filter_chromosomes(blacklist_df["Chromosome"].unique().tolist())
    )
    blacklist_df = blacklist_df[blacklist_df["Chromosome"].isin(valid_chroms)]

    if len(blacklist_df) == 0:
        logger.info("Blacklist filtered to empty (no standard chromosomes), no tiles filtered")
        return tiles

    # Use pyranges for efficient overlap detection
    try:
        import pyranges as pr
    except ImportError:
        raise ImportError(
            "pyranges is required for blacklist filtering. Install with: pip install pyranges"
        )

    # Normalize column names for both tiles and blacklist
    tiles_df, col_mapping = _normalize_column_names(tiles)
    blacklist_df, _ = _normalize_column_names(blacklist_df)

    # Create PyRanges objects
    pr_tiles = pr.PyRanges(tiles_df)
    pr_blacklist = pr.PyRanges(blacklist_df)

    # Find tiles that do NOT overlap with blacklist
    non_blacklist_tiles = pr_tiles.overlap(pr_blacklist, invert=True)

    if len(non_blacklist_tiles) == 0:
        logger.warning("All tiles overlap with blacklist regions, returning empty DataFrame")
        return pd.DataFrame(columns=tiles.columns)

    filtered_tiles = non_blacklist_tiles.df

    # Restore original column names if they were changed
    if col_mapping:
        reverse_mapping = {v: k for k, v in col_mapping.items()}
        filtered_tiles = filtered_tiles.rename(columns=reverse_mapping)

    n_removed = len(tiles) - len(filtered_tiles)
    logger.info(
        f"Filtered tiles by blacklist: {len(tiles)} -> {len(filtered_tiles)} "
        f"({n_removed} blacklisted tiles removed)"
    )

    # Reset index to ensure continuous indices [0, 1, 2, ...]
    # This is important for peak ordering to work correctly
    filtered_tiles = filtered_tiles.reset_index(drop=True)

    return filtered_tiles


def filter_tiles(
    tiles: pd.DataFrame,
    blacklist: str | pd.DataFrame | None = None,
    assembly: str = "hg38",
    chr_filter_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply both chromosome and blacklist filtering to tiles.

    **Used by: BOTH Zarr and BPCells backends**

    This is a convenience function that applies both filters in the correct order:
    1. Chromosome filtering (removes patches, chrM, chrY, etc.) using ChromosomeFilter
    2. Blacklist filtering (removes tiles overlapping blacklist regions)

    **Zarr usage (Two-Stage Filtering):**
    - **Preprocessing**: Chromosome filtering only (in `fragment_utils.create_genome_tiles()`)
    - **Analysis**: Blacklist filtering on stored tiles (in `ClusteringMixin.create_genome_tiles()`)
    - Example:
      ```python
      # Preprocessing (fragment processor)
      tiles = filter_tiles_by_chromosomes(tiles)  # Store in Zarr

      # Analysis (clustering)
      stored_tiles = get_peaks("tiles")  # Read from Zarr
      tiles = filter_tiles_by_blacklist(stored_tiles, blacklist=blacklist)
      matrix = get_count_matrix(peaks=tiles)
      ```

    **BPCells usage (Post-Matrix Filtering):**
    - Called AFTER matrix creation in `BPCellsProcessor._process_anndata()`
    - Matrix is created with unfiltered chrom_sizes, then tiles are filtered
    - Column selection uses `create_column_indices()` to match filtered tiles
    - Example:
      ```python
      matrix = create_matrix(unfiltered_chrom_sizes)
      tiles = filter_tiles(all_tiles, blacklist=blacklist)
      indices = create_column_indices(all_tiles, tiles)
      matrix = matrix[:, indices]
      ```

    **Blacklist filtering is incorporated:**
    - If `blacklist=None`: Uses assembly-specific blacklist (default behavior)
    - If `blacklist=False`: Explicitly skips blacklist filtering
    - If `blacklist` is a path or DataFrame: Uses the provided blacklist

    Args:
        tiles: DataFrame with 'Chromosome', 'Start', 'End' columns
        blacklist: Path to blacklist file or DataFrame. If None, uses assembly-specific blacklist.
            Set to False to skip blacklist filtering.
        assembly: Genome assembly (default: 'hg38')
        chr_filter_config: Optional chromosome filter configuration dict.
            If None, uses default filtering (max_length=5, excludes chrM, chrY, etc.)

    Returns:
        Filtered DataFrame with both chromosome and blacklist filtering applied

    Examples:
        >>> tiles = pd.DataFrame({
        ...     'Chromosome': ['chr1', 'chrM', 'chrY'],
        ...     'Start': [0, 0, 0],
        ...     'End': [100, 100, 100]
        ... })
        >>> filtered = filter_tiles(tiles)
        >>> len(filtered)
        1
        >>> filtered['Chromosome'].iloc[0]
        'chr1'
    """
    # First filter by chromosomes using shared ChromosomeFilter
    filtered = filter_tiles_by_chromosomes(tiles, chr_filter_config=chr_filter_config)

    # Then filter by blacklist
    if blacklist is not False:  # Allow explicit False to skip blacklist filtering
        filtered = filter_tiles_by_blacklist(filtered, blacklist=blacklist, assembly=assembly)

    return filtered


def create_column_indices(
    all_tiles: pd.DataFrame,
    filtered_tiles: pd.DataFrame,
) -> np.ndarray:
    """Create column indices for matrix selection based on filtered tiles.

    **Used by: BPCells backend ONLY**

    **Why BPCells needs this:**
    - BPCells creates the count matrix with UNFILTERED chrom_sizes (all chromosomes)
    - The matrix has columns for ALL tiles (including patches, chrM, chrY, blacklisted regions)
    - After matrix creation, tiles are filtered (chromosome + blacklist filtering)
    - This function maps filtered tiles back to their original column indices in the full matrix
    - The indices are used to select: `matrix[:, indices]` to get only the filtered columns

    **Why Zarr doesn't need this:**
    - Zarr filters tiles BEFORE computing the matrix
    - Matrix is computed with already-filtered tiles
    - Matrix columns directly match filtered tiles (no selection needed)

    The indices are created by:
    1. Preserving the original index of all_tiles before filtering
    2. Using pandas merge operations (vectorized, no loops) to identify which tiles remain after filtering
    3. Returning the indices directly (for use with `matrix[:, indices]`) rather than a boolean mask

    Args:
        all_tiles: DataFrame with all tiles (before filtering), must have continuous index [0, 1, 2, ...]
            This should match the matrix columns (created with unfiltered chrom_sizes)
        filtered_tiles: DataFrame with filtered tiles (after filtering), must be a subset of all_tiles
            This is the result of `filter_tiles()` which applies chromosome + blacklist filtering

    Returns:
        Integer numpy array of indices to select from the matrix columns
        Usage: `filtered_matrix = full_matrix[:, column_indices]`

    Examples:
        >>> all_tiles = pd.DataFrame({
        ...     'Chromosome': ['chr1', 'chr1', 'chr2', 'chr2'],
        ...     'Start': [0, 500, 0, 500],
        ...     'End': [500, 1000, 500, 1000]
        ... })
        >>> filtered_tiles = pd.DataFrame({
        ...     'Chromosome': ['chr1', 'chr2'],
        ...     'Start': [0, 0],
        ...     'End': [500, 500]
        ... })
        >>> indices = create_column_indices(all_tiles, filtered_tiles)
        >>> indices
        array([0, 2])
    """
    # Ensure all_tiles has continuous index [0, 1, 2, ...]
    if not all_tiles.index.equals(pd.RangeIndex(len(all_tiles))):
        all_tiles = all_tiles.reset_index(drop=True)

    # Create a key for matching tiles (chrom, start, end)
    # This is more robust than using index alone
    all_tiles_with_key = all_tiles.copy()
    all_tiles_with_key['_tile_key'] = (
        all_tiles_with_key['Chromosome'].astype(str) + '_' +
        all_tiles_with_key['Start'].astype(str) + '_' +
        all_tiles_with_key['End'].astype(str)
    )

    filtered_tiles_with_key = filtered_tiles.copy()
    filtered_tiles_with_key['_tile_key'] = (
        filtered_tiles_with_key['Chromosome'].astype(str) + '_' +
        filtered_tiles_with_key['Start'].astype(str) + '_' +
        filtered_tiles_with_key['End'].astype(str)
    )

    # Use pandas merge to find matching indices (vectorized, no loop)
    # This preserves the order of filtered_tiles
    merged = filtered_tiles_with_key.merge(
        all_tiles_with_key.reset_index().rename(columns={'index': '_original_idx'}),
        on='_tile_key',
        how='left',
        suffixes=('', '_all')
    )

    # Extract the original indices (these are the column indices to select)
    indices = merged['_original_idx'].values

    # Validate that we found all filtered tiles
    if len(indices) != len(filtered_tiles):
        raise ValueError(
            f"Found {len(indices)} matching indices but expected {len(filtered_tiles)}. "
            f"This indicates a mismatch in tile filtering."
        )

    # Check for any missing indices (shouldn't happen if filtered_tiles is a subset)
    if pd.isna(indices).any():
        raise ValueError(
            "Some filtered tiles were not found in all_tiles. "
            "This indicates filtered_tiles is not a subset of all_tiles."
        )

    return indices.astype(np.int64)

