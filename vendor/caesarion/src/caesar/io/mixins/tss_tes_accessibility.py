"""TSS and TES accessibility extraction mixin for genomic IO classes.

This module provides functionality for extracting and aggregating accessibility
signals around Transcription Start Sites (TSS) and Transcription End Sites (TES).

Two modes are supported:
1. **Tile-based (fast)**: When AnnData with tile count matrix is available,
   uses PyRanges overlap to find tiles within TSS/TES windows. Much faster
   for large datasets.
2. **High-resolution**: When only dense IO is available, uses get_gene_tss/tes_track
   for base-pair resolution data.

Output format:
- AnnData h5ad with obs=cells/celltypes, var=TSS/TES sites (with annotation)
- Annotation includes: site_id, gene_name, transcript_id, chromosome, position, strand
- Supports multiple TSS/TES per gene (from different transcripts)
- Optional aggregation by cell type via --groupby

Performance optimizations:
- Uses numpy-based vectorized interval searching for regularly-spaced tiles
- Caches chromosome-sorted tile indices for efficient lookup
- Avoids PyRanges join overhead when tiles are uniform (fixed tile_size)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    import scanpy as sc

    from gcell.rna.gencode import Gencode


# Module-level cache for gencode dictionaries to avoid expensive re-computation
# The gcell library's gene_to_tss and gene_to_tes properties re-compute on each access
_GENCODE_CACHE: dict[str, dict] = {}


def _get_gencode_site_dicts(
    gencode: Gencode, site_type: str
) -> tuple[dict, dict, dict]:
    """Get cached gencode dictionaries for TSS/TES extraction.

    The gcell library's gene_to_tss and gene_to_tes properties are expensive
    (~3s each) because they re-parse the GTF on every access. This function
    caches them for reuse.

    Parameters:
        gencode: Gencode object
        site_type: "tss" or "tes"

    Returns:
        Tuple of (site_dict, chrom_dict, strand_dict)
    """
    # Create cache key from assembly and version
    cache_key = f"{gencode.assembly}_{gencode.version}"

    if cache_key not in _GENCODE_CACHE:
        _GENCODE_CACHE[cache_key] = {
            "chrom": dict(gencode.gene_to_chrom),
            "strand": dict(gencode.gene_to_strand),
        }

    cache = _GENCODE_CACHE[cache_key]

    # Cache TSS/TES separately since they're expensive
    site_key = f"{site_type}_dict"
    if site_key not in cache:
        logging.info(f"Caching gene_to_{site_type} dictionary (one-time, ~3s)...")
        if site_type == "tss":
            cache[site_key] = dict(gencode.gene_to_tss)
        else:
            cache[site_key] = dict(gencode.gene_to_tes)

    return cache[site_key], cache["chrom"], cache["strand"]


def _extract_site_position(site_obj, fallback_pos: int) -> int:
    """Extract integer position from a TSS/TES object or list.

    The gcell library's gene.tss and gene.tes properties may return:
    - A list of TSS/TES objects with .start attribute
    - A single TSS/TES object with .start attribute
    - An integer position directly

    Parameters:
        site_obj: The TSS/TES object (list, object, or int)
        fallback_pos: Fallback position if extraction fails

    Returns:
        Integer position
    """
    if isinstance(site_obj, (int, float)):
        return int(site_obj)
    if isinstance(site_obj, list) and len(site_obj) > 0:
        first = site_obj[0]
        if hasattr(first, "start"):
            return int(first.start)
        if isinstance(first, (int, float)):
            return int(first)
    if hasattr(site_obj, "start"):
        return int(site_obj.start)
    return fallback_pos


def _build_tile_index(tile_df: pd.DataFrame, tile_size: int) -> dict[str, Any]:
    """Build a fast lookup index for tiles organized by chromosome.

    This creates a numpy-friendly index structure that allows O(1) lookup
    of tiles overlapping a given genomic interval, avoiding PyRanges overhead
    for regularly-spaced tiles.

    Parameters:
        tile_df: DataFrame with Chromosome, Start, End, tile_idx columns
        tile_size: Size of each tile in base pairs (for uniform tile detection)

    Returns:
        Dictionary with chromosome-level index information:
        - 'chroms': dict mapping chrom -> {start_array, end_array, idx_array, min_start}
        - 'uniform': bool indicating if tiles are uniformly spaced
        - 'tile_size': detected or provided tile size
    """
    index = {
        "chroms": {},
        "uniform": True,
        "tile_size": tile_size,
    }

    # Check if tiles are uniformly spaced
    for chrom in tile_df["Chromosome"].unique():
        chrom_tiles = tile_df[tile_df["Chromosome"] == chrom].sort_values("Start")
        starts = chrom_tiles["Start"].values
        ends = chrom_tiles["End"].values
        idxs = chrom_tiles["tile_idx"].values

        # Check uniformity
        if len(starts) > 1:
            diffs = np.diff(starts)
            if not np.allclose(diffs, tile_size, rtol=0.01):
                index["uniform"] = False

        # Check tile size consistency
        tile_sizes = ends - starts
        if not np.allclose(tile_sizes, tile_size, rtol=0.01):
            index["uniform"] = False

        index["chroms"][chrom] = {
            "starts": starts,
            "ends": ends,
            "idxs": idxs,
            "min_start": starts[0] if len(starts) > 0 else 0,
        }

    return index


def _find_overlapping_tiles_vectorized(
    tile_index: dict[str, Any],
    site_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find tiles overlapping with site regions using vectorized numpy operations.

    For uniformly-spaced tiles, uses direct index calculation instead of
    interval tree or PyRanges join operations.

    Parameters:
        tile_index: Index built by _build_tile_index
        site_df: DataFrame with Chromosome, Start, End, site_id, gene_name, etc.

    Returns:
        DataFrame with overlap information (site columns + tile_idx, Start_b, End_b)
    """
    tile_size = tile_index["tile_size"]
    uniform = tile_index["uniform"]

    results = []

    for chrom in site_df["Chromosome"].unique():
        if chrom not in tile_index["chroms"]:
            continue

        chrom_info = tile_index["chroms"][chrom]
        chrom_sites = site_df[site_df["Chromosome"] == chrom]

        if len(chrom_sites) == 0:
            continue

        tile_starts = chrom_info["starts"]
        tile_ends = chrom_info["ends"]
        tile_idxs = chrom_info["idxs"]
        min_start = chrom_info["min_start"]

        if uniform and len(tile_starts) > 0:
            # Fast path: uniform tiles - use direct index calculation
            # For each site, calculate which tile indices overlap
            site_starts = chrom_sites["Start"].values
            site_ends = chrom_sites["End"].values

            # Calculate tile index range for each site
            # tile_idx = (position - min_start) // tile_size
            tile_idx_first = (site_starts - min_start) // tile_size
            first_tile_idx = np.maximum(0, tile_idx_first).astype(int)
            tile_idx_last = (site_ends - min_start) // tile_size
            last_tile_idx = np.minimum(len(tile_starts) - 1, tile_idx_last).astype(int)

            # Build result rows
            for i, (site_row_idx, site_row) in enumerate(chrom_sites.iterrows()):
                start_idx = first_tile_idx[i]
                end_idx = last_tile_idx[i]

                # Get all tiles in range
                for t_idx in range(start_idx, end_idx + 1):
                    if t_idx < 0 or t_idx >= len(tile_starts):
                        continue

                    # Verify actual overlap (edge cases)
                    t_start = tile_starts[t_idx]
                    t_end = tile_ends[t_idx]
                    if t_end <= site_row["Start"] or t_start >= site_row["End"]:
                        continue

                    result_row = site_row.to_dict()
                    result_row["tile_idx"] = tile_idxs[t_idx]
                    result_row["Start_b"] = t_start
                    result_row["End_b"] = t_end
                    results.append(result_row)
        else:
            # Slow path: non-uniform tiles - use binary search
            for site_row_idx, site_row in chrom_sites.iterrows():
                site_start = site_row["Start"]
                site_end = site_row["End"]

                # Binary search for first tile that might overlap
                start_idx = np.searchsorted(tile_ends, site_start, side="right")

                # Scan forward for overlapping tiles
                for t_idx in range(start_idx, len(tile_starts)):
                    t_start = tile_starts[t_idx]
                    t_end = tile_ends[t_idx]

                    if t_start >= site_end:
                        break  # No more overlaps possible

                    if t_end > site_start:  # Overlap found
                        result_row = site_row.to_dict()
                        result_row["tile_idx"] = tile_idxs[t_idx]
                        result_row["Start_b"] = t_start
                        result_row["End_b"] = t_end
                        results.append(result_row)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def _is_bpcells_backend(io_obj) -> bool:
    """Check if the IO object is using BPCells backend."""
    # Check class name for BPCells types
    class_name = io_obj.__class__.__name__
    return "BPCells" in class_name


def _save_bpcells_accessibility(
    io_obj,
    dataset_name: str,
    data: np.ndarray,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> None:
    """Save accessibility data for BPCells backend.

    Saves data to a subdirectory within the BPCells store:
    - {dataset_name}/data.npy: The accessibility profiles
    - {dataset_name}/metadata.json: Associated metadata
    """
    output_dir = Path(io_obj.path) / dataset_name

    if output_dir.exists() and not overwrite:
        raise ValueError(f"Dataset '{dataset_name}' already exists")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save array data
    np.save(output_dir / "data.npy", data)

    # Save metadata
    with (output_dir / "metadata.json").open("w") as f:
        # Convert numpy types to Python types for JSON serialization
        json_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (np.integer, np.floating)):
                json_metadata[k] = v.item()
            elif isinstance(v, np.ndarray):
                json_metadata[k] = v.tolist()
            elif isinstance(v, tuple):
                json_metadata[k] = list(v)
            else:
                json_metadata[k] = v
        json.dump(json_metadata, f, indent=2)

    logging.info(f"Saved BPCells accessibility to {output_dir}")


def _bpcells_dataset_exists(io_obj, dataset_name: str) -> bool:
    """Check if a dataset already exists in BPCells store."""
    dataset_path = Path(io_obj.path) / dataset_name
    return dataset_path.exists() and (dataset_path / "data.npy").exists()


def _get_all_transcript_sites(
    gencode: Gencode,
    gene_list: list[str],
    site_type: str,
    upstream: int,
    downstream: int,
) -> pd.DataFrame:
    """
    Get all unique TSS or TES positions from all transcripts for given genes.

    This extracts sites from ALL transcripts, not just the canonical one,
    allowing analysis of alternative TSS/TES usage.

    Parameters:
        gencode: Gencode reference containing gene data
        gene_list: List of gene names
        site_type: "tss" or "tes"
        upstream: Base pairs upstream
        downstream: Base pairs downstream

    Returns:
        DataFrame with columns: site_id, gene_name, transcript_id, Chromosome,
        Start, End, strand, site_position
    """
    sites = []
    seen_positions = set()  # Track unique positions per gene to avoid duplicates

    for gene_name in gene_list:
        try:
            gene = gencode.get_gene(gene_name)
            chrom, g_start, g_end, strand = gene.genomic_range

            # Try to get all transcripts for this gene
            transcripts = []
            if hasattr(gene, "transcripts") and gene.transcripts:
                transcripts = gene.transcripts
            elif hasattr(gencode, "get_transcripts"):
                import contextlib

                with contextlib.suppress(Exception):
                    transcripts = gencode.get_transcripts(gene_name)

            if transcripts:
                # Process each transcript
                for tx in transcripts:
                    try:
                        tx_id = (
                            tx.transcript_id
                            if hasattr(tx, "transcript_id")
                            else str(tx)
                        )
                        tx_start = tx.start if hasattr(tx, "start") else g_start
                        tx_end = tx.end if hasattr(tx, "end") else g_end

                        if site_type == "tss":
                            site = tx_start if strand == "+" else tx_end
                        else:  # tes
                            site = tx_end if strand == "+" else tx_start

                        # Create unique key for this position
                        pos_key = (gene_name, chrom, site, strand)
                        if pos_key in seen_positions:
                            continue
                        seen_positions.add(pos_key)

                        # Calculate window
                        if strand == "+":
                            win_start = site - upstream
                            win_end = site + downstream
                        else:
                            win_start = site - downstream
                            win_end = site + upstream

                        win_start = max(0, win_start)

                        site_id = f"{gene_name}_{chrom}_{site}_{strand}_{site_type}"
                        sites.append(
                            {
                                "site_id": site_id,
                                "gene_name": gene_name,
                                "transcript_id": tx_id,
                                "Chromosome": chrom,
                                "Start": win_start,
                                "End": win_end,
                                "strand": strand,
                                "site_position": site,
                            }
                        )
                    except Exception as e:
                        logging.debug(
                            f"Error processing transcript {tx} for {gene_name}: {e}"
                        )
                        continue
            else:
                # Fallback to gene-level TSS/TES (canonical)
                fallback_pos = g_start if strand == "+" else g_end
                if site_type == "tss":
                    try:
                        site = _extract_site_position(gene.tss, fallback_pos)
                    except (ValueError, AttributeError):
                        site = fallback_pos
                else:
                    fallback_pos = g_end if strand == "+" else g_start
                    try:
                        site = _extract_site_position(gene.tes, fallback_pos)
                    except (ValueError, AttributeError):
                        site = fallback_pos

                pos_key = (gene_name, chrom, site, strand)
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)

                    if strand == "+":
                        win_start = site - upstream
                        win_end = site + downstream
                    else:
                        win_start = site - downstream
                        win_end = site + upstream

                    win_start = max(0, win_start)

                    site_id = f"{gene_name}_{chrom}_{site}_{strand}_{site_type}"
                    sites.append(
                        {
                            "site_id": site_id,
                            "gene_name": gene_name,
                            "transcript_id": "canonical",
                            "Chromosome": chrom,
                            "Start": win_start,
                            "End": win_end,
                            "strand": strand,
                            "site_position": site,
                        }
                    )

        except Exception as e:
            logging.debug(
                f"Could not get {site_type.upper()} for gene {gene_name}: {e}"
            )
            continue

    return pd.DataFrame(sites)


def _get_site_regions_pyranges(
    gencode: Gencode,
    gene_list: list[str],
    site_type: str,
    upstream: int,
    downstream: int,
    include_all_transcripts: bool = False,
) -> pd.DataFrame:
    """
    Create PyRanges-compatible DataFrame for TSS or TES regions.

    Parameters:
        gencode: Gencode reference containing gene data
        gene_list: List of gene names
        site_type: "tss" or "tes"
        upstream: Base pairs upstream
        downstream: Base pairs downstream
        include_all_transcripts: If True, include TSS/TES from all transcripts

    Returns:
        DataFrame with Chromosome, Start, End, gene_name columns
    """
    if include_all_transcripts:
        return _get_all_transcript_sites(
            gencode, gene_list, site_type, upstream, downstream
        )

    # OPTIMIZED: Use gencode dictionary attributes directly instead of
    # calling get_gene() for each gene (which is slow due to GTF parsing).
    # gencode has gene_to_tss, gene_to_tes, gene_to_chrom, gene_to_strand dicts
    # that map gene names to their positions/metadata directly.
    #
    # IMPORTANT: The gcell library's gene_to_tss and gene_to_tes properties
    # re-compute on each access (~3s each). Use our caching wrapper.

    # Get cached dictionaries (first call is slow, subsequent calls are fast)
    site_dict, chrom_dict, strand_dict = _get_gencode_site_dicts(gencode, site_type)

    regions = []
    for gene_name in gene_list:
        try:
            # Fast dictionary lookups instead of get_gene()
            if gene_name not in site_dict:
                logging.debug(f"Gene {gene_name} not found in gene_to_{site_type}")
                continue

            site = int(site_dict[gene_name])
            chrom = chrom_dict[gene_name]
            strand = strand_dict[gene_name]

            # Calculate window based on strand
            if strand == "+":
                start = site - upstream
                end = site + downstream
            else:
                start = site - downstream
                end = site + upstream

            start = max(0, start)

            site_id = f"{gene_name}_{chrom}_{site}_{strand}_{site_type}"
            regions.append(
                {
                    "site_id": site_id,
                    "Chromosome": chrom,
                    "Start": start,
                    "End": end,
                    "gene_name": gene_name,
                    "transcript_id": "canonical",
                    "strand": strand,
                    "site_position": site,
                }
            )
        except Exception as e:
            logging.debug(
                f"Could not get {site_type.upper()} for gene {gene_name}: {e}"
            )
            continue

    return pd.DataFrame(regions)


def compute_accessibility_from_tiles(
    adata: sc.AnnData,
    gencode: Gencode,
    gene_list: list[str],
    site_type: str,
    upstream: int = 2000,
    downstream: int = 2000,
    tile_size: int = 500,
    groupby: str | None = None,
    include_all_transcripts: bool = False,
    use_pyranges: bool = False,
) -> sc.AnnData:
    """
    Compute TSS or TES accessibility from tile count matrix.

    This is much faster than querying dense data per gene because it:
    1. Creates regions for all TSS/TES at once
    2. Uses vectorized numpy operations for tile overlap detection
    3. Uses sparse matrix multiplication for aggregation

    Performance optimization (default):
    - For uniformly-spaced tiles (common case), uses direct index calculation
      instead of PyRanges join, which is significantly faster for large datasets.
    - Falls back to PyRanges for non-uniform tiles or when use_pyranges=True.

    Parameters:
        adata: AnnData with tile count matrix (cells x tiles)
               adata.var must have Chromosome, Start, End columns
        gencode: Gencode reference containing gene data
        gene_list: List of gene names to process
        site_type: "tss" or "tes"
        upstream: Base pairs upstream of site
        downstream: Base pairs downstream of site
        tile_size: Size of tiles in base pairs (for relative position calculation)
        groupby: Column in adata.obs to aggregate by (e.g., 'celltype', 'cluster')
                 If None, returns per-cell accessibility
        include_all_transcripts: If True, include TSS/TES from all transcripts
        use_pyranges: If True, force use of PyRanges for overlap detection
                      (slower but handles complex interval relationships)

    Returns:
        AnnData with:
        - obs: cells or celltypes (if groupby)
        - var: TSS/TES sites with annotation (site_id, gene_name, transcript_id,
               chromosome, position, strand)
        - X: accessibility counts (obs x var) - summed across window bins
        - layers['binned']: per-bin profiles (obs x var x n_bins) if space permits
        - uns: metadata (upstream, downstream, tile_size, method, etc.)
    """
    import scanpy as sc
    from scipy import sparse

    # Get site regions
    site_regions_df = _get_site_regions_pyranges(
        gencode,
        gene_list,
        site_type,
        upstream,
        downstream,
        include_all_transcripts=include_all_transcripts,
    )

    if site_regions_df.empty:
        logging.warning(f"No valid {site_type.upper()} regions found")
        return None

    # Get tile regions from adata.var
    tile_df = adata.var.copy()
    if "Chromosome" not in tile_df.columns:
        # Try alternate column names
        if "chrom" in tile_df.columns:
            tile_df["Chromosome"] = tile_df["chrom"]
        elif "chr" in tile_df.columns:
            tile_df["Chromosome"] = tile_df["chr"]
        else:
            raise ValueError("adata.var must have Chromosome/chrom/chr column")

    if "Start" not in tile_df.columns:
        if "start" in tile_df.columns:
            tile_df["Start"] = tile_df["start"]
        else:
            raise ValueError("adata.var must have Start/start column")

    if "End" not in tile_df.columns:
        if "end" in tile_df.columns:
            tile_df["End"] = tile_df["end"]
        else:
            raise ValueError("adata.var must have End/end column")

    # Add tile index for later retrieval
    tile_df["tile_idx"] = np.arange(len(tile_df))

    # Find overlaps between site regions and tiles
    # Use optimized vectorized approach for uniform tiles (default)
    # Fall back to PyRanges for non-uniform tiles or when explicitly requested
    if use_pyranges:
        import pyranges as pr

        logging.info("Using PyRanges for tile overlap detection (requested)")
        site_regions = pr.PyRanges(site_regions_df)
        tile_regions = pr.PyRanges(tile_df[["Chromosome", "Start", "End", "tile_idx"]])
        overlaps = site_regions.join(tile_regions, how="left")
        overlaps_df = overlaps.df
    else:
        # Use optimized vectorized overlap detection
        logging.info("Using vectorized tile overlap detection (optimized)")
        tile_index = _build_tile_index(
            tile_df[["Chromosome", "Start", "End", "tile_idx"]], tile_size
        )

        if tile_index["uniform"]:
            logging.info(f"Tiles are uniform ({tile_size}bp) - using fast path")
        else:
            logging.info("Non-uniform tiles - using binary search fallback")

        overlaps_df = _find_overlapping_tiles_vectorized(tile_index, site_regions_df)

    if overlaps_df.empty:
        logging.warning(f"No tiles overlap with {site_type.upper()} regions")
        return None

    # Calculate relative position of each tile to the site
    window_size = upstream + downstream
    n_bins = window_size // tile_size

    # Handle groupby aggregation BEFORE accessing data
    # This reduces memory by aggregating early
    if groupby is not None:
        if groupby not in adata.obs.columns:
            raise ValueError(f"groupby column '{groupby}' not found in adata.obs")

        groups = list(adata.obs[groupby].unique())
        n_obs = len(groups)
        obs_names = groups
        obs_df = pd.DataFrame(index=obs_names)
        obs_df[groupby] = obs_names

        # Create group index mapping
        group_indices = {g: np.where(adata.obs[groupby] == g)[0] for g in groups}
    else:
        n_obs = adata.n_obs
        obs_names = list(adata.obs_names)
        obs_df = adata.obs.copy()
        group_indices = None

    # Keep data as sparse if possible, only convert needed columns
    X_is_sparse = sparse.issparse(adata.X)

    # Remove rows with missing tile indices
    overlaps_df = overlaps_df.dropna(subset=["tile_idx"])
    if overlaps_df.empty:
        logging.warning(f"No valid tile overlaps for {site_type.upper()} regions")
        return None

    # Calculate bin indices for all overlaps at once (vectorized)
    tile_centers = (overlaps_df["Start_b"].values + overlaps_df["End_b"].values) // 2
    site_positions = overlaps_df["site_position"].values
    strands = overlaps_df["strand"].values

    # Vectorized relative position calculation
    rel_pos = np.where(
        strands == "+",
        tile_centers - site_positions + upstream,
        site_positions - tile_centers + upstream,
    )
    bin_indices = (rel_pos // tile_size).astype(np.int32)

    # Filter to valid bins
    valid_bin_mask = (bin_indices >= 0) & (bin_indices < n_bins)
    overlaps_df = overlaps_df[valid_bin_mask].copy()
    bin_indices = bin_indices[valid_bin_mask]
    overlaps_df["bin_idx"] = bin_indices

    if overlaps_df.empty:
        logging.warning(f"No overlaps with valid bin indices for {site_type.upper()}")
        return None

    # Get unique sites and create site index mapping
    site_id_col = "site_id" if "site_id" in overlaps_df.columns else "gene_name"
    unique_sites = overlaps_df[site_id_col].unique()
    site_to_idx = {site: i for i, site in enumerate(unique_sites)}
    n_sites = len(unique_sites)

    logging.info(
        f"Processing {n_sites} {site_type.upper()} sites with vectorized operations..."
    )

    # Create site indices for all overlaps
    site_indices = overlaps_df[site_id_col].map(site_to_idx).values

    # Get tile indices
    tile_indices = overlaps_df["tile_idx"].values.astype(np.int32)

    # OPTIMIZED: Fully vectorized aggregation using sparse matrix multiplication
    # Instead of iterating over (site, bin) pairs, we construct a sparse mapping matrix
    # and perform a single matrix multiplication
    logging.info("Aggregating tile counts using vectorized sparse matrix operations...")

    # Create site indices for all overlaps
    overlaps_df["site_idx"] = site_indices

    # Get unique tiles that we need to access
    unique_tile_indices = np.unique(tile_indices)
    n_unique_tiles = len(unique_tile_indices)

    # Create a mapping from original tile index to compact index
    tile_idx_to_compact = {t: i for i, t in enumerate(unique_tile_indices)}
    compact_tile_indices = np.array([tile_idx_to_compact[t] for t in tile_indices])

    # Extract only the needed columns from X (much smaller than full matrix)
    logging.info(f"Extracting {n_unique_tiles} unique tiles from matrix...")
    if X_is_sparse:
        X_subset = adata.X[:, unique_tile_indices]  # (n_cells, n_unique_tiles)
    else:
        X_subset = adata.X[:, unique_tile_indices]

    # Handle groupby aggregation by pre-aggregating the data matrix
    if group_indices is not None:
        logging.info(f"Pre-aggregating by {len(group_indices)} groups...")
        if X_is_sparse:
            # For sparse: sum cells by group using a sparse aggregation matrix
            n_cells_orig = adata.n_obs
            agg_rows = []
            agg_cols = []
            agg_data = []
            for g_idx, (group, cell_indices) in enumerate(group_indices.items()):
                for cell_idx in cell_indices:
                    agg_rows.append(g_idx)
                    agg_cols.append(cell_idx)
                    agg_data.append(1.0)
            agg_matrix = sparse.csr_matrix(
                (agg_data, (agg_rows, agg_cols)), shape=(n_obs, n_cells_orig)
            )
            X_aggregated = agg_matrix @ X_subset  # (n_groups, n_unique_tiles)
        else:
            X_aggregated = np.zeros((n_obs, n_unique_tiles), dtype=np.float32)
            for g_idx, (group, cell_indices) in enumerate(group_indices.items()):
                X_aggregated[g_idx] = X_subset[cell_indices].sum(axis=0)
    else:
        X_aggregated = X_subset  # (n_cells, n_unique_tiles)

    # Convert to dense array for final operations (only the subset we need)
    if sparse.issparse(X_aggregated):
        X_aggregated = np.asarray(X_aggregated.todense())

    # Create sparse matrix: (site_idx * n_bins + bin_idx) -> tile contributions
    # Rows = (site, bin) pairs, Cols = unique tiles, Values = 1s (sum)
    n_site_bins = n_sites * n_bins

    # Compute the flat index for (site_idx, bin_idx) pairs
    flat_indices = site_indices * n_bins + bin_indices

    # Build the mapping sparse matrix
    # Rows: flat (site, bin) indices
    # Cols: compact tile indices
    # Data: all 1s (we sum contributions)
    mapping_data = np.ones(len(flat_indices), dtype=np.float32)
    mapping_matrix = sparse.csr_matrix(
        (mapping_data, (flat_indices, compact_tile_indices)),
        shape=(n_site_bins, n_unique_tiles),
    )

    # Count tiles per (site, bin) for normalization
    bin_counts_flat = np.asarray(mapping_matrix.sum(axis=1)).flatten()

    # Matrix multiply: mapping_matrix @ X_aggregated.T gives (n_site_bins, n_obs)
    # But we want X_aggregated to be (n_unique_tiles, n_obs), so transpose first
    logging.info("Computing accessibility via sparse matrix multiplication...")
    site_bin_accessibility = mapping_matrix @ X_aggregated.T  # (n_site_bins, n_obs)

    if sparse.issparse(site_bin_accessibility):
        site_bin_accessibility = np.asarray(site_bin_accessibility.todense())

    # Normalize by bin counts where > 1 (average when multiple tiles map to same bin)
    # This is vectorized normalization
    normalization_mask = bin_counts_flat > 1
    site_bin_accessibility[normalization_mask] /= bin_counts_flat[
        normalization_mask, np.newaxis
    ]

    # Reshape to (n_sites, n_bins, n_obs)
    site_profiles = site_bin_accessibility.reshape(n_sites, n_bins, n_obs).astype(
        np.float32
    )

    # Build site metadata from the first occurrence of each site
    logging.info("Building site metadata...")
    first_occurrence = overlaps_df.drop_duplicates(subset=[site_id_col])
    site_metadata_dict = first_occurrence.set_index(site_id_col).to_dict("index")

    valid_sites = []
    for site_id in unique_sites:
        site_meta = site_metadata_dict[site_id]
        valid_sites.append(
            {
                "site_id": site_id,
                "gene_name": site_meta.get("gene_name", site_id),
                "transcript_id": site_meta.get("transcript_id", "canonical"),
                "chromosome": site_meta["Chromosome"],
                "position": int(site_meta["site_position"]),
                "strand": site_meta["strand"],
                "window_start": int(site_meta["Start"]),
                "window_end": int(site_meta["End"]),
            }
        )

    # Build output AnnData
    var_df = pd.DataFrame(valid_sites)
    var_df = var_df.set_index("site_id")

    # X matrix: (n_obs, n_sites) - total accessibility per site (sum across bins)
    # site_profiles is (n_sites, n_bins, n_obs)
    # Sum across bins: (n_sites, n_obs), then transpose to (n_obs, n_sites)
    X_out = site_profiles.sum(axis=1).T.astype(np.float32)

    result = sc.AnnData(
        X=X_out,
        obs=obs_df,
        var=var_df,
    )

    # Store mean profile per site in varm
    # site_profiles is (n_sites, n_bins, n_obs)
    # Mean across observations: (n_sites, n_bins)
    result.varm["mean_profile"] = site_profiles.mean(axis=2)

    # Store metadata
    result.uns[f"{site_type}_accessibility"] = {
        "upstream": upstream,
        "downstream": downstream,
        "tile_size": tile_size,
        "n_bins": n_bins,
        "window_size": window_size,
        "method": "tile_based",
        "groupby": groupby,
        "include_all_transcripts": include_all_transcripts,
        "n_sites": len(valid_sites),
        "n_genes": len(set(var_df["gene_name"])),
    }

    # Store gencode/assembly info for traceback
    if hasattr(gencode, "assembly"):
        result.uns["assembly"] = gencode.assembly
    if hasattr(gencode, "version"):
        result.uns["gencode_version"] = gencode.version

    logging.info(
        f"Created {site_type.upper()} accessibility AnnData: "
        f"{result.n_obs} {'groups' if groupby else 'cells'} x {result.n_vars} sites "
        f"({len(set(var_df['gene_name']))} genes)"
    )

    return result


def compute_accessibility_from_tiles_legacy(
    adata: sc.AnnData,
    gencode: Gencode,
    gene_list: list[str],
    site_type: str,
    upstream: int = 2000,
    downstream: int = 2000,
    tile_size: int = 500,
) -> tuple[np.ndarray, list[str]]:
    """
    Legacy function returning raw arrays (for backward compatibility).

    Returns:
        Tuple of (accessibility_matrix, valid_genes) where:
        - accessibility_matrix: (n_genes, n_bins, n_cells) array with binned profiles
        - valid_genes: list of genes that were successfully processed
    """
    result_adata = compute_accessibility_from_tiles(
        adata,
        gencode,
        gene_list,
        site_type,
        upstream,
        downstream,
        tile_size,
        groupby=None,
        include_all_transcripts=False,
    )

    if result_adata is None:
        return np.array([]), []

    # Extract profiles from varm and reshape
    # varm['mean_profile'] is (n_sites, n_bins), but we need (n_genes, n_bins, n_cells)
    # This is a simplified version - full profiles would need to be reconstructed
    n_bins = result_adata.uns[f"{site_type}_accessibility"]["n_bins"]
    n_cells = result_adata.n_obs
    n_sites = result_adata.n_vars

    # For legacy compatibility, use X (total accessibility) reshaped
    # Note: This loses the binned profile information
    profiles = result_adata.X.T  # (n_sites, n_cells)
    profiles = profiles[:, np.newaxis, :]  # (n_sites, 1, n_cells)
    profiles = np.broadcast_to(profiles, (n_sites, n_bins, n_cells)).copy()

    valid_genes = list(result_adata.var["gene_name"])

    return profiles, valid_genes


class TSSAccessibilityMixin:
    """
    Mixin for computing and storing TSS accessibility profiles.

    Supports two modes:
    - Tile-based (fast): Uses AnnData with tile count matrix
    - High-resolution: Uses get_gene_tss_track for base-pair data
    """

    def compute_tss_accessibility_from_tiles(
        self,
        adata: sc.AnnData,
        gencode: Gencode,
        gene_list: list[str],
        upstream: int = 2000,
        downstream: int = 2000,
        tile_size: int = 500,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute TSS accessibility from tile count matrix (fast method).

        Parameters:
            adata: AnnData with tile count matrix
            gencode: Gencode reference
            gene_list: List of gene names
            upstream: Base pairs upstream of TSS
            downstream: Base pairs downstream of TSS
            tile_size: Size of tiles in base pairs

        Returns:
            Tuple of (profiles, valid_genes)
        """
        return compute_accessibility_from_tiles(
            adata, gencode, gene_list, "tss", upstream, downstream, tile_size
        )

    def compute_tss_accessibility(
        self,
        gene_name: str,
        gencode: Gencode,
        upstream: int = 2000,
        downstream: int = 2000,
        **track_kwargs,
    ) -> tuple[str, np.ndarray | None]:
        """
        Compute TSS accessibility profile for a single gene (high-resolution).

        Parameters:
            gene_name: Name of the gene
            gencode: Gencode reference containing gene data
            upstream: Base pairs upstream of TSS to include
            downstream: Base pairs downstream of TSS to include
            **track_kwargs: Additional arguments passed to get_gene_tss_track()

        Returns:
            Tuple (gene_name, accessibility_profile) where accessibility_profile
            is the computed signal or None if error
        """
        try:
            track_data = self.get_gene_tss_track(
                gene_name,
                upstream=upstream,
                downstream=downstream,
                output_format="raw_array",
                **track_kwargs,
            )
            return (gene_name, track_data)
        except Exception as e:
            logging.warning(f"Error processing TSS for gene {gene_name}: {e}")
            return (gene_name, None)

    def compute_tss_accessibility_batch(
        self,
        gene_list: list[str],
        gencode: Gencode,
        upstream: int = 2000,
        downstream: int = 2000,
        n_jobs: int = 1,
        **track_kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Compute TSS accessibility profiles for multiple genes (high-resolution).

        Parameters:
            gene_list: List of gene names to process
            gencode: Gencode reference containing gene data
            upstream: Base pairs upstream of TSS to include
            downstream: Base pairs downstream of TSS to include
            n_jobs: Number of parallel jobs (1 for serial processing)
            **track_kwargs: Additional arguments passed to get_gene_tss_track()

        Returns:
            Dictionary mapping gene names to accessibility profiles
        """
        results = {}

        if n_jobs == 1:
            for gene_name in tqdm(gene_list, desc="Computing TSS accessibility"):
                gene_name, track_data = self.compute_tss_accessibility(
                    gene_name, gencode, upstream, downstream, **track_kwargs
                )
                if track_data is not None:
                    results[gene_name] = track_data
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(
                        self.compute_tss_accessibility,
                        gene_name,
                        gencode,
                        upstream,
                        downstream,
                        **track_kwargs,
                    ): gene_name
                    for gene_name in gene_list
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Computing TSS accessibility",
                ):
                    gene_name, track_data = future.result()
                    if track_data is not None:
                        results[gene_name] = track_data

        return results

    def write_tss_accessibility(
        self,
        gene_list: list[str],
        gencode: Gencode,
        upstream: int = 2000,
        downstream: int = 2000,
        n_jobs: int = 1,
        overwrite: bool = False,
        dataset_name: str = "tss_accessibility",
        aggregate: bool = True,
        adata: sc.AnnData | None = None,
        tile_size: int = 500,
        **track_kwargs,
    ) -> dict[str, Any]:
        """
        Compute and write TSS accessibility profiles.

        If adata is provided, uses fast tile-based method.
        Otherwise, falls back to high-resolution method.

        Parameters:
            gene_list: List of gene names to process
            gencode: Gencode reference containing gene data
            upstream: Base pairs upstream of TSS to include
            downstream: Base pairs downstream of TSS to include
            n_jobs: Number of parallel jobs (for high-res method)
            overwrite: Whether to overwrite existing dataset
            dataset_name: Name of the dataset to create/update
            aggregate: If True, aggregate across genes; if False, store per-gene
            adata: Optional AnnData with tile count matrix (enables fast mode)
            tile_size: Size of tiles in base pairs (for tile-based method)
            **track_kwargs: Additional arguments for high-resolution method

        Returns:
            Dictionary with results summary
        """
        if self.mode != "a":
            raise ValueError("Cannot write in read-only mode")

        # Check backend type for appropriate storage method
        is_bpcells = _is_bpcells_backend(self)

        # Check if dataset exists (backend-specific)
        if is_bpcells:
            if _bpcells_dataset_exists(self, dataset_name) and not overwrite:
                logging.info(f"Dataset '{dataset_name}' already exists. Skipping.")
                return {"status": "skipped", "reason": "dataset_exists"}
        else:
            if dataset_name in self.dataset and not overwrite:
                logging.info(f"Dataset '{dataset_name}' already exists. Skipping.")
                return {"status": "skipped", "reason": "dataset_exists"}

        if adata is not None:
            # Fast tile-based method
            logging.info("Using fast tile-based TSS accessibility extraction")
            profiles, valid_genes = self.compute_tss_accessibility_from_tiles(
                adata, gencode, gene_list, upstream, downstream, tile_size
            )

            if len(valid_genes) == 0:
                return {"status": "error", "reason": "no_valid_profiles"}

            # profiles shape: (n_genes, n_bins, n_cells)
            # Transpose to (n_cells, n_bins) for each gene, then aggregate
            if aggregate:
                # Mean across genes: (n_bins, n_cells)
                aggregated = np.mean(profiles, axis=0)
                result_shape = aggregated.shape
                result_data = aggregated
                method_metadata = {
                    "n_genes": len(valid_genes),
                    "aggregation": "mean",
                    "method": "tile_based",
                    "tile_size": tile_size,
                }
            else:
                result_shape = profiles.shape
                result_data = profiles
                method_metadata = {
                    "gene_list": valid_genes,
                    "method": "tile_based",
                    "tile_size": tile_size,
                }
            gene_names = valid_genes
        else:
            # High-resolution method
            logging.info("Using high-resolution TSS accessibility extraction")
            results = self.compute_tss_accessibility_batch(
                gene_list, gencode, upstream, downstream, n_jobs, **track_kwargs
            )

            if not results:
                return {"status": "error", "reason": "no_valid_profiles"}

            gene_names = list(results.keys())
            profiles = np.stack([results[g] for g in gene_names], axis=0)

            if aggregate:
                aggregated = np.mean(profiles, axis=0)
                result_shape = aggregated.shape
                result_data = aggregated
                method_metadata = {
                    "n_genes": len(gene_names),
                    "aggregation": "mean",
                    "method": "high_resolution",
                }
            else:
                result_shape = profiles.shape
                result_data = profiles
                method_metadata = {
                    "gene_list": gene_names,
                    "method": "high_resolution",
                }

        # Add common metadata
        common_metadata = {
            "upstream": upstream,
            "downstream": downstream,
            "window_size": upstream + downstream,
        }
        all_metadata = {**method_metadata, **common_metadata}

        # Store results (backend-specific)
        if is_bpcells:
            _save_bpcells_accessibility(
                self, dataset_name, result_data, all_metadata, overwrite
            )
        else:
            self.dataset.create_array(
                dataset_name, data=result_data, overwrite=overwrite
            )
            for key, value in all_metadata.items():
                self.dataset[dataset_name].attrs[key] = value

        return {
            "status": "success",
            "n_genes": len(gene_names),
            "dataset_name": dataset_name,
            "shape": result_shape,
        }


class TESAccessibilityMixin:
    """
    Mixin for computing and storing TES accessibility profiles.

    Supports two modes:
    - Tile-based (fast): Uses AnnData with tile count matrix
    - High-resolution: Uses get_gene_tes_track for base-pair data
    """

    def compute_tes_accessibility_from_tiles(
        self,
        adata: sc.AnnData,
        gencode: Gencode,
        gene_list: list[str],
        upstream: int = 2000,
        downstream: int = 2000,
        tile_size: int = 500,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute TES accessibility from tile count matrix (fast method).

        Parameters:
            adata: AnnData with tile count matrix
            gencode: Gencode reference
            gene_list: List of gene names
            upstream: Base pairs upstream of TES
            downstream: Base pairs downstream of TES
            tile_size: Size of tiles in base pairs

        Returns:
            Tuple of (profiles, valid_genes)
        """
        return compute_accessibility_from_tiles(
            adata, gencode, gene_list, "tes", upstream, downstream, tile_size
        )

    def compute_tes_accessibility(
        self,
        gene_name: str,
        gencode: Gencode,
        upstream: int = 2000,
        downstream: int = 2000,
        **track_kwargs,
    ) -> tuple[str, np.ndarray | None]:
        """
        Compute TES accessibility profile for a single gene (high-resolution).

        Parameters:
            gene_name: Name of the gene
            gencode: Gencode reference containing gene data
            upstream: Base pairs upstream of TES to include
            downstream: Base pairs downstream of TES to include
            **track_kwargs: Additional arguments passed to get_gene_tes_track()

        Returns:
            Tuple (gene_name, accessibility_profile) where accessibility_profile
            is the computed signal or None if error
        """
        try:
            track_data = self.get_gene_tes_track(
                gene_name,
                upstream=upstream,
                downstream=downstream,
                output_format="raw_array",
                **track_kwargs,
            )
            return (gene_name, track_data)
        except Exception as e:
            logging.warning(f"Error processing TES for gene {gene_name}: {e}")
            return (gene_name, None)

    def compute_tes_accessibility_batch(
        self,
        gene_list: list[str],
        gencode: Gencode,
        upstream: int = 2000,
        downstream: int = 2000,
        n_jobs: int = 1,
        **track_kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Compute TES accessibility profiles for multiple genes (high-resolution).

        Parameters:
            gene_list: List of gene names to process
            gencode: Gencode reference containing gene data
            upstream: Base pairs upstream of TES to include
            downstream: Base pairs downstream of TES to include
            n_jobs: Number of parallel jobs (1 for serial processing)
            **track_kwargs: Additional arguments passed to get_gene_tes_track()

        Returns:
            Dictionary mapping gene names to accessibility profiles
        """
        results = {}

        if n_jobs == 1:
            for gene_name in tqdm(gene_list, desc="Computing TES accessibility"):
                gene_name, track_data = self.compute_tes_accessibility(
                    gene_name, gencode, upstream, downstream, **track_kwargs
                )
                if track_data is not None:
                    results[gene_name] = track_data
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(
                        self.compute_tes_accessibility,
                        gene_name,
                        gencode,
                        upstream,
                        downstream,
                        **track_kwargs,
                    ): gene_name
                    for gene_name in gene_list
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Computing TES accessibility",
                ):
                    gene_name, track_data = future.result()
                    if track_data is not None:
                        results[gene_name] = track_data

        return results

    def write_tes_accessibility(
        self,
        gene_list: list[str],
        gencode: Gencode,
        upstream: int = 2000,
        downstream: int = 2000,
        n_jobs: int = 1,
        overwrite: bool = False,
        dataset_name: str = "tes_accessibility",
        aggregate: bool = True,
        adata: sc.AnnData | None = None,
        tile_size: int = 500,
        **track_kwargs,
    ) -> dict[str, Any]:
        """
        Compute and write TES accessibility profiles.

        If adata is provided, uses fast tile-based method.
        Otherwise, falls back to high-resolution method.

        Parameters:
            gene_list: List of gene names to process
            gencode: Gencode reference containing gene data
            upstream: Base pairs upstream of TES to include
            downstream: Base pairs downstream of TES to include
            n_jobs: Number of parallel jobs (for high-res method)
            overwrite: Whether to overwrite existing dataset
            dataset_name: Name of the dataset to create/update
            aggregate: If True, aggregate across genes; if False, store per-gene
            adata: Optional AnnData with tile count matrix (enables fast mode)
            tile_size: Size of tiles in base pairs (for tile-based method)
            **track_kwargs: Additional arguments for high-resolution method

        Returns:
            Dictionary with results summary
        """
        if self.mode != "a":
            raise ValueError("Cannot write in read-only mode")

        # Check backend type for appropriate storage method
        is_bpcells = _is_bpcells_backend(self)

        # Check if dataset exists (backend-specific)
        if is_bpcells:
            if _bpcells_dataset_exists(self, dataset_name) and not overwrite:
                logging.info(f"Dataset '{dataset_name}' already exists. Skipping.")
                return {"status": "skipped", "reason": "dataset_exists"}
        else:
            if dataset_name in self.dataset and not overwrite:
                logging.info(f"Dataset '{dataset_name}' already exists. Skipping.")
                return {"status": "skipped", "reason": "dataset_exists"}

        if adata is not None:
            # Fast tile-based method
            logging.info("Using fast tile-based TES accessibility extraction")
            profiles, valid_genes = self.compute_tes_accessibility_from_tiles(
                adata, gencode, gene_list, upstream, downstream, tile_size
            )

            if len(valid_genes) == 0:
                return {"status": "error", "reason": "no_valid_profiles"}

            if aggregate:
                aggregated = np.mean(profiles, axis=0)
                result_shape = aggregated.shape
                result_data = aggregated
                method_metadata = {
                    "n_genes": len(valid_genes),
                    "aggregation": "mean",
                    "method": "tile_based",
                    "tile_size": tile_size,
                }
            else:
                result_shape = profiles.shape
                result_data = profiles
                method_metadata = {
                    "gene_list": valid_genes,
                    "method": "tile_based",
                    "tile_size": tile_size,
                }
            gene_names = valid_genes
        else:
            # High-resolution method
            logging.info("Using high-resolution TES accessibility extraction")
            results = self.compute_tes_accessibility_batch(
                gene_list, gencode, upstream, downstream, n_jobs, **track_kwargs
            )

            if not results:
                return {"status": "error", "reason": "no_valid_profiles"}

            gene_names = list(results.keys())
            profiles = np.stack([results[g] for g in gene_names], axis=0)

            if aggregate:
                aggregated = np.mean(profiles, axis=0)
                result_shape = aggregated.shape
                result_data = aggregated
                method_metadata = {
                    "n_genes": len(gene_names),
                    "aggregation": "mean",
                    "method": "high_resolution",
                }
            else:
                result_shape = profiles.shape
                result_data = profiles
                method_metadata = {
                    "gene_list": gene_names,
                    "method": "high_resolution",
                }

        # Add common metadata
        common_metadata = {
            "upstream": upstream,
            "downstream": downstream,
            "window_size": upstream + downstream,
        }
        all_metadata = {**method_metadata, **common_metadata}

        # Store results (backend-specific)
        if is_bpcells:
            _save_bpcells_accessibility(
                self, dataset_name, result_data, all_metadata, overwrite
            )
        else:
            self.dataset.create_array(
                dataset_name, data=result_data, overwrite=overwrite
            )
            for key, value in all_metadata.items():
                self.dataset[dataset_name].attrs[key] = value

        return {
            "status": "success",
            "n_genes": len(gene_names),
            "dataset_name": dataset_name,
            "shape": result_shape,
        }


class TSSTESAccessibilityMixin(TSSAccessibilityMixin, TESAccessibilityMixin):
    """
    Combined mixin for both TSS and TES accessibility extraction.

    This mixin combines TSSAccessibilityMixin and TESAccessibilityMixin to provide
    a unified interface for extracting accessibility signals around both transcription
    start and end sites.
    """

    def write_tss_tes_accessibility(
        self,
        gene_list: list[str],
        gencode: Gencode,
        upstream: int = 2000,
        downstream: int = 2000,
        n_jobs: int = 1,
        overwrite: bool = False,
        aggregate: bool = True,
        adata: sc.AnnData | None = None,
        tile_size: int = 500,
        **track_kwargs,
    ) -> dict[str, Any]:
        """
        Compute and write both TSS and TES accessibility profiles.

        If adata is provided, uses fast tile-based method for both.
        Otherwise, falls back to high-resolution method.

        Parameters:
            gene_list: List of gene names to process
            gencode: Gencode reference containing gene data
            upstream: Base pairs upstream to include
            downstream: Base pairs downstream to include
            n_jobs: Number of parallel jobs
            overwrite: Whether to overwrite existing datasets
            aggregate: If True, aggregate across genes; if False, store per-gene
            adata: Optional AnnData with tile count matrix (enables fast mode)
            tile_size: Size of tiles in base pairs
            **track_kwargs: Additional arguments passed to track methods

        Returns:
            Dictionary with results summary for both TSS and TES
        """
        tss_results = self.write_tss_accessibility(
            gene_list,
            gencode,
            upstream,
            downstream,
            n_jobs,
            overwrite,
            "tss_accessibility",
            aggregate,
            adata,
            tile_size,
            **track_kwargs,
        )

        tes_results = self.write_tes_accessibility(
            gene_list,
            gencode,
            upstream,
            downstream,
            n_jobs,
            overwrite,
            "tes_accessibility",
            aggregate,
            adata,
            tile_size,
            **track_kwargs,
        )

        return {
            "tss": tss_results,
            "tes": tes_results,
        }
