"""Pure clustering operations for AnnData objects.

This module provides backend-agnostic clustering utilities that work on
any AnnData object, regardless of the data source (Zarr, BPCells, etc.).

Extracted from ClusteringMixin to enable reusability and cleaner architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from caesar.utils import get_device, is_torch_available

if TYPE_CHECKING:
    import scanpy as sc


def cluster_adata(
    adata: sc.AnnData,
    n_components: int = 30,
    device: str = "auto",
    filter_quantile: float = 0.995,
    total_features: int = 100_000,
    scale_factor: float = 100000,
    binarize: bool = True,
    blacklist: str | None = None,
    assembly: str = "hg38",
    return_intermediate: bool = False,
) -> sc.AnnData:
    """Perform clustering on an AnnData object using TF-IDF and SVD.

    This function implements the standard scATAC-seq clustering workflow:
    1. Feature filtering (top features by count)
    2. Blacklist filtering (optional)
    3. Binarization (optional)
    4. TF-IDF transformation
    5. LSI (SVD) dimensionality reduction
    6. Leiden clustering
    7. UMAP visualization

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing raw counts and metadata.
        Expects cells x features matrix.
    n_components : int
        Number of SVD components to compute (default: 30)
    device : str
        Device for computations: 'cuda', 'cpu', or 'auto' (default: 'auto')
    filter_quantile : float
        Quantile for filtering features (default: 0.995)
    total_features : int
        Number of top features to keep (default: 100,000)
    scale_factor : float
        Scaling factor for TF-IDF (default: 100,000)
    binarize : bool
        Whether to binarize before TF-IDF (default: True)
    blacklist : str, optional
        Path to blacklist BED file. If None, uses assembly default.
    assembly : str
        Genome assembly for blacklist (default: 'hg38')
    return_intermediate : bool
        Whether to return intermediate results (default: False)

    Returns
    -------
    sc.AnnData
        Clustered AnnData with:
        - .obsm['X_svd']: LSI representation
        - .obs['leiden']: Cluster assignments
        - .obs['n_fragments']: Fragment counts per cell
        - .obsm['X_umap']: UMAP coordinates

    Notes
    -----
    This is a pure utility function with no backend dependencies.
    Extracted from ClusteringMixin.cluster_adata() for reusability.
    """
    import scanpy as sc
    from scipy import sparse

    logging.info("Starting clustering...")

    # Determine device - use utility function for consistent device handling
    device = get_device(device)
    logging.info(f"Using device: {device}")

    # Get raw count matrix - keep sparse
    tile_matrix = (
        adata.X if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    )
    logging.info(f"Tile matrix shape: {tile_matrix.shape}")

    # Filter low-count tiles while keeping sparsity
    tile_sums = np.array(tile_matrix.sum(axis=0)).flatten()
    tile_sums_top = tile_sums.argsort()[-total_features:]
    tile_matrix = tile_matrix[:, tile_sums_top]
    # Update adata to match filtered tiles
    adata = adata[:, tile_sums_top]

    # Binarize
    if binarize:
        logging.info("Binarizing tile matrix")
        tile_matrix = tile_matrix > 0

    # Filter out blacklisted tiles (only if blacklist is not False)
    # If blacklist=False, filtering was already done in create_genome_tiles()
    if blacklist is not False:
        if blacklist:
            blacklist_df = pd.read_csv(blacklist, sep="\t", header=None)
            blacklist_df.columns = ["Chromosome", "Start", "End", "name"]
        else:
            from gcell.dna.genome import read_blacklist

            blacklist_df = read_blacklist(assembly)

        blacklist_df = blacklist_df.query("Chromosome.str.len()<=5")

        if len(blacklist_df) > 0:
            import pyranges as pr

            # Create PyRanges objects for blacklist and tiles
            pr_blacklist = pr.PyRanges(blacklist_df)

            # Prepare tiles dataframe for PyRanges (needs capitalized columns)
            tiles_df = adata.var.reset_index(drop=True).reset_index()
            # Standardize column names for PyRanges (case-insensitive mapping)
            col_mapping = {}
            for col in tiles_df.columns:
                col_lower = col.lower()
                if col_lower == 'chrom' or col_lower == 'chromosome':
                    col_mapping[col] = 'Chromosome'
                elif col_lower == 'start':
                    col_mapping[col] = 'Start'
                elif col_lower == 'end':
                    col_mapping[col] = 'End'
            tiles_df = tiles_df.rename(columns=col_mapping)

            pr_tiles = pr.PyRanges(tiles_df)

            # Find tiles that do NOT overlap with blacklist
            non_blacklist_tiles = pr_tiles.overlap(pr_blacklist, invert=True)

            if len(non_blacklist_tiles) > 0:
                # Get indices of non-blacklisted tiles
                non_blacklist_indices = non_blacklist_tiles.index.values

                # Create boolean mask
                blacklist_mask = np.zeros(tile_matrix.shape[1], dtype=bool)
                blacklist_mask[non_blacklist_indices] = True

                # Filter tile matrix
                tile_matrix = tile_matrix[:, blacklist_mask]
                adata = adata[:, blacklist_mask]
                logging.info(
                    f"Filtered out {(~blacklist_mask).sum()} blacklisted tiles, "
                    f"{blacklist_mask.sum()} tiles remaining"
                )
            else:
                logging.warning(
                    "All tiles overlap with blacklist regions, proceeding without filtering"
                )
    else:
        logging.info("Skipping blacklist filtering (already applied in tile creation)")

    # Compute SVD
    if device == "cuda" and is_torch_available():
        logging.info("Computing TF-IDF")
        # Compute IDF (adapted from Stuart et al.)
        n_cells = tile_matrix.shape[0]
        tile_sums = np.array(tile_matrix.sum(axis=0)).flatten()
        idf = sparse.diags(n_cells / tile_sums)

        # Compute TF-IDF matrix while maintaining sparsity
        tfidf_matrix = tile_matrix @ idf
        # Log transform TF-IDF values
        tfidf_matrix.data = np.log1p(tfidf_matrix.data * scale_factor)

        logging.info("Converting to torch tensor and moving to device")
        import torch

        # Convert to torch tensor and move to device
        tfidf_tensor = torch.tensor(tfidf_matrix.todense(), dtype=torch.float32).to(
            device
        )
        with torch.no_grad():
            svd = torch.svd_lowrank(tfidf_tensor, q=n_components)
        svd_components = svd[0].cpu().numpy()
        singular_values = svd[1].cpu().numpy()
        # Add SVD as an additional layer in original AnnData
        adata.obsm["X_svd"] = svd_components / singular_values

    else:
        # Compute using scikit learn
        logging.info("Computing SVD using scikit learn")

        from caesar.analysis.lsi import LSI

        lsi = LSI(n_components=n_components, scale_factor=scale_factor)
        logging.info("Fitting LSI")
        lsi.fit_transform(tile_matrix, downsample=50000)
        logging.info("Transforming LSI")
        svd_components = lsi.transform(tile_matrix)
        adata.obsm["X_svd"] = svd_components

    logging.info("Computing neighbors")
    # Compute neighbors using SVD representation
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_svd")

    logging.info("Performing leiden clustering")
    # Leiden clustering
    sc.tl.leiden(adata)

    # Print number of clusters
    n_clusters = len(adata.obs["leiden"].unique())
    logging.info(f"Number of clusters: {n_clusters}")

    # Add fragment counts to metadata
    adata.obs["n_fragments"] = np.array(tile_matrix.sum(axis=1)).flatten()

    logging.info("Computing UMAP")
    # Compute UMAP
    sc.tl.umap(adata)

    logging.info("Clustering complete")

    if return_intermediate:
        return adata, tfidf_matrix if device == "cuda" else None, svd_components
    return adata


def save_clustering_results(
    adata: sc.AnnData,
    output_dir: Path,
    filename: str = "genome_tiles.h5ad",
    export_metadata: bool = True,
) -> dict[str, Path]:
    """Save clustered AnnData and export metadata.

    Saves:
    - AnnData object to .h5ad file
    - Metadata (obs table) to metadata.csv (optional)

    Parameters
    ----------
    adata : sc.AnnData
        Clustered AnnData object with .obs['labels'] or .obs['leiden']
    output_dir : Path
        Directory to save outputs
    filename : str
        Name for .h5ad file (default: 'genome_tiles.h5ad')
    export_metadata : bool
        Whether to export metadata.csv (default: True)

    Returns
    -------
    dict[str, Path]
        Dictionary with paths:
        - 'adata': Path to .h5ad file
        - 'metadata': Path to metadata.csv (if exported)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save AnnData
    adata_path = output_dir / filename
    logging.info(f"Saving AnnData to {adata_path}")
    adata.write(adata_path, compression="gzip")

    result = {"adata": adata_path}

    # Export metadata
    if export_metadata:
        metadata_path = output_dir / "metadata.csv"
        logging.info(f"Exporting metadata to {metadata_path}")
        adata.obs.to_csv(metadata_path, index=True)
        result["metadata"] = metadata_path

    return result


def load_and_cluster(
    adata_path: Path,
    output_dir: Path | None = None,
    **clustering_params,
) -> sc.AnnData:
    """Load AnnData, cluster, and save results.

    Convenience function for the complete workflow:
    Load → Cluster → Add labels → Save

    Parameters
    ----------
    adata_path : Path
        Path to input .h5ad file
    output_dir : Path, optional
        Directory to save results. If None, saves to same directory as input.
    **clustering_params
        Parameters passed to cluster_adata()

    Returns
    -------
    sc.AnnData
        Clustered AnnData object
    """
    import anndata as ad

    logging.info(f"Loading AnnData from {adata_path}")
    adata = ad.read_h5ad(adata_path)

    logging.info("Clustering AnnData")
    adata = cluster_adata(adata, **clustering_params)

    # Add labels from leiden clusters
    if "labels" not in adata.obs.columns:
        adata.obs["labels"] = "cluster_" + adata.obs["leiden"].astype(str)

    # Determine output directory
    if output_dir is None:
        output_dir = Path(adata_path).parent

    # Save results
    save_clustering_results(adata, output_dir, filename=Path(adata_path).name)

    return adata
