from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyranges as pr
from scipy.sparse import csr_matrix

from gcell.dna.genome import ChromSize, read_blacklist

# Import clustering utilities
from caesar.analysis.clustering_ops import (
    cluster_adata as cluster_adata_utility,
    save_clustering_results,
)
from caesar.utils import get_device, is_torch_available

# only import scanpy when type checking
if TYPE_CHECKING:
    import scanpy as sc



class ClusteringMixin:
    """
    A mixin class that provides clustering functionality for sparse genomic data.

    This mixin adds methods for:
    - Creating tile matrices
    - TF-IDF transformation
    - SVD dimensionality reduction
    - Hierarchical clustering
    - UMAP visualization
    """

    def get_count_matrix(self, peaks: str | pd.DataFrame | dict, name="peaks"):
        NotImplementedError("Subclass must implement this method")

    def create_and_cluster_tiles_adata(
        self,
        n_components: int = 30,
        device: str = "auto",
        filter_quantile: float = 0.995,
        total_features: int = 100_000,
        binarize: bool = True,
        overwrite: bool = False,
        return_intermediate: bool = False,
        blacklist: str = None,
    ) -> sc.AnnData:
        """
        Create genome tiles AnnData object and perform clustering.

        This is a convenience function that combines:
        1. create_genome_tiles_adata() - Create tiles from store
        2. cluster_adata() - Perform clustering
        3. save_clustering_results() - Save results

        Parameters
        ----------
        n_components : int
            Number of SVD components for clustering
        device : str
            Device to use for torch computations ('cuda' or 'cpu')
        filter_quantile : float
            Quantile for filtering features
        total_features : int
            Number of top features to keep
        binarize : bool
            Whether to binarize the data before TF-IDF
        overwrite : bool
            Whether to overwrite existing data
        return_intermediate : bool
            Whether to return intermediate results
        blacklist : str, optional
            Path to blacklist file. If None, uses assembly-specific blacklist.

        Returns
        -------
        sc.AnnData
            Processed AnnData object with clustering results
        """
        # Create or load genome tiles
        output_path = Path(self.path) / "genome_tiles.h5ad"

        # Try to load existing genome tiles if not overwriting
        if not overwrite and output_path.exists():
            try:
                import anndata as ad

                adata = ad.read_h5ad(output_path)
                logging.info("Loaded existing genome tiles from zarr")
                return adata
            except Exception as e:
                logging.warning(f"Failed to load existing genome tiles: {e}")
                logging.info("Creating new genome tiles...")

        logging.info("Creating genome tiles adata")
        # Create new genome tiles from store (blacklist filtering applied here)
        adata = self.create_genome_tiles_adata(tile_size=500, blacklist=blacklist)

        logging.info("Clustering genome tiles")
        # Perform clustering using utility function
        # Note: blacklist filtering is now done in create_genome_tiles_adata(),
        # so we pass blacklist=False to cluster_adata to skip duplicate filtering
        adata = self.cluster_adata(
            adata,
            n_components=n_components,
            device=device,
            filter_quantile=filter_quantile,
            total_features=total_features,
            binarize=binarize,
            return_intermediate=return_intermediate,
            blacklist=False,  # Already filtered in create_genome_tiles_adata
        )

        # Add labels
        adata.obs["labels"] = "cluster_" + adata.obs["leiden"].astype(str)

        logging.info("Saving genome tiles")
        # Save using utility function
        save_clustering_results(
            adata,
            Path(self.path),
            filename="genome_tiles.h5ad",
            export_metadata=True,
        )

        logging.info("Done")
        return adata

    def create_genome_tiles_adata(
        self, tile_size: int = 500, blacklist: str | pd.DataFrame | None = None
    ) -> sc.AnnData:
        """
        Create genome tiles AnnData object with blacklist filtering.

        **Filtering Strategy:**
        - Reads stored chromosome-filtered tiles from Zarr (from fragment processor)
        - Applies blacklist filtering on top of stored tiles
        - Creates AnnData with blacklist-filtered tiles and matrix

        See `create_genome_tiles()` for detailed filtering workflow.

        Parameters
        ----------
        tile_size : int
            Size of tiles in base pairs (default: 500)
        blacklist : str | pd.DataFrame | None
            Path to blacklist file or DataFrame. If None, uses assembly-specific blacklist.
            Set to False to skip blacklist filtering.

        Returns
        -------
        sc.AnnData
            AnnData object containing tile matrix and metadata (with blacklist filtering applied)
        """
        from scanpy import AnnData

        # Create new genome tiles (with filtering applied)
        tile_matrix, tiles = self.create_genome_tiles(
            tile_size=tile_size, blacklist=blacklist
        )

        # Create AnnData object
        adata = AnnData(X=tile_matrix)
        if isinstance(tiles, dict):
            tiles = list(tiles.values())[0]
        adata.var = tiles
        adata.obs = self.metadata

        return adata

    def create_genome_tiles(
        self,
        tile_size: int = 500,
        overwrite: bool = False,
        blacklist: str | pd.DataFrame | None = None,
    ) -> tuple:
        """
        Create genome-wide tiles with blacklist filtering applied on stored tiles.

        **Filtering Strategy (Zarr Backend):**
        1. **Preprocessing (Fragment Processor)**: Chromosome filtering only
           - Tiles are created with chromosome filtering (removes patches, chrM, chrY)
           - Stored in Zarr under "tiles/regions/"
        2. **Analysis (This method)**: Always uses stored tiles, optionally applies blacklist filtering
           - Always reads stored chromosome-filtered tiles from Zarr
           - If `blacklist=None` (default): Applies assembly-specific blacklist filtering
           - If `blacklist=<path/DataFrame>`: Applies specified blacklist filtering
           - If `blacklist=False`: Skips blacklist filtering (uses stored tiles as-is)
           - Computes matrix with (optionally) blacklist-filtered tiles
           - Ensures indices are continuous after filtering

        **Note:** This is different from BPCells backend, which creates matrix with
        unfiltered chrom_sizes and filters tiles after matrix creation.

        Parameters
        ----------
        tile_size : int
            Size of tiles in base pairs (default: 500)
        overwrite : bool
            Whether to overwrite existing tile group (default: False)
        blacklist : str | pd.DataFrame | None
            Path to blacklist file or DataFrame. If None, uses assembly-specific blacklist.
            Set to False to skip blacklist filtering.

        Returns
        -------
        tuple
            (tile_matrix, tiles) where tiles are blacklist-filtered and match matrix columns
        """
        from caesar.preprocess.core.tile_filtering import filter_tiles_by_blacklist
        from caesar.io import ensure_peaks_match_matrix

        # Always try to load stored tiles from Zarr (these have chromosome filtering from fragment processor)
        stored_tiles = self.get_peaks("tiles")

        if stored_tiles is None:
            # Fallback: create tiles from scratch if not stored
            # This should rarely happen if fragment processor ran correctly
            logging.warning(
                "No stored tiles found in Zarr. Creating tiles from scratch. "
                "This may indicate fragment processor did not complete successfully."
            )
            chrom_sizes = ChromSize(self.assembly)
            tiles = chrom_sizes.tiling_region(tile_size)
            # Apply chromosome filtering to match preprocessing behavior
            from caesar.preprocess.core.tile_filtering import filter_tiles_by_chromosomes
            tiles = filter_tiles_by_chromosomes(tiles)
            logging.info(
                f"Created {len(tiles)} tiles from scratch (chromosome-filtered)"
            )
        else:
            # Handle MultiSparseIO case where get_peaks returns a dict {data_key: peaks_df}
            if isinstance(stored_tiles, dict):
                # For MultiSparseIO, all samples should have the same tiles
                # Use the first one (they should all be identical)
                if not stored_tiles:
                    raise ValueError("No stored tiles found in any sample")
                tiles = list(stored_tiles.values())[0].copy()
                logging.info(
                    f"Loaded {len(tiles)} stored tiles from MultiSparseIO "
                    f"(chromosome-filtered from preprocessing, using tiles from first sample)"
                )
            else:
                # Use stored tiles (already chromosome-filtered from preprocessing)
                tiles = stored_tiles.copy()
                logging.info(
                    f"Loaded {len(tiles)} stored tiles (chromosome-filtered from preprocessing)"
                )

        # Apply blacklist filtering on top of stored tiles if requested
        # blacklist=None: Use assembly-specific blacklist (default)
        # blacklist=<path/DataFrame>: Use specified blacklist
        # blacklist=False: Skip blacklist filtering (use stored tiles as-is)
        if blacklist is not False:  # Apply blacklist filtering if not explicitly disabled
            tiles_before_blacklist = len(tiles)
            tiles = filter_tiles_by_blacklist(
                tiles,
                blacklist=blacklist,  # None will use assembly-specific blacklist
                assembly=self.assembly,
            )
            if len(tiles) < tiles_before_blacklist:
                logging.info(
                    f"Applied blacklist filtering: {tiles_before_blacklist} -> {len(tiles)} tiles "
                    f"({tiles_before_blacklist - len(tiles)} blacklisted tiles removed)"
                )
            else:
                logging.info("Blacklist filtering applied (no tiles removed)")
        else:
            logging.info("Skipping blacklist filtering (blacklist=False) - using stored tiles as-is")

        # Ensure indices are continuous after filtering
        tiles = tiles.reset_index(drop=True)

        # Get stored tiles count for comparison (handle dict case)
        if stored_tiles is None:
            stored_tiles_count = 0
            stored_tiles_for_matrix = None
        elif isinstance(stored_tiles, dict):
            stored_tiles_count = len(list(stored_tiles.values())[0]) if stored_tiles else 0
            stored_tiles_for_matrix = list(stored_tiles.values())[0].copy() if stored_tiles else None
        else:
            stored_tiles_count = len(stored_tiles)
            stored_tiles_for_matrix = stored_tiles.copy()

        # Check if blacklist filtering was applied and if matrix might already exist
        blacklist_filtered = blacklist is not False and stored_tiles_count > 0 and len(tiles) < stored_tiles_count

        # Compute matrix with blacklist-filtered tiles
        # If blacklist filtering was applied and matrix already exists, we need to:
        # 1. Get matrix with stored (unfiltered) tiles
        # 2. Use create_column_indices to select columns matching filtered tiles
        # 3. Select those columns from the matrix
        if blacklist_filtered and not overwrite and stored_tiles_for_matrix is not None:
            # Check if matrix already exists (by trying to get it with stored tiles)
            # If it exists, it was created with unfiltered tiles, so we need to filter columns
            try:
                # Prepare stored tiles for matrix (before blacklist filtering)
                stored_tiles_for_matrix = stored_tiles_for_matrix.reset_index(drop=True)

                # Check if matrix exists by checking if peak_name exists in dataset
                # This is backend-specific, so we'll try to get the matrix and check its shape
                matrix_with_stored = self.get_count_matrix(
                    peaks=stored_tiles_for_matrix, peak_name="tiles", overwrite=False
                )

                # If matrix has more columns than filtered tiles, it was created with unfiltered tiles
                if matrix_with_stored.shape[1] > len(tiles):
                    logging.info(
                        f"Matrix was created with {matrix_with_stored.shape[1]} columns (unfiltered tiles), "
                        f"but filtered tiles have {len(tiles)} tiles. Filtering matrix columns to match."
                    )
                    # Use create_column_indices to select correct columns (similar to BPCells approach)
                    from caesar.preprocess.core.tile_filtering import create_column_indices

                    # Ensure stored tiles have continuous index for create_column_indices
                    stored_tiles_for_indices = stored_tiles_for_matrix.reset_index(drop=True)
                    filtered_tiles_for_indices = tiles.reset_index(drop=True)

                    column_indices = create_column_indices(
                        all_tiles=stored_tiles_for_indices,
                        filtered_tiles=filtered_tiles_for_indices
                    )

                    # Select columns from matrix
                    tile_matrix = matrix_with_stored[:, column_indices]
                    logging.info(
                        f"Filtered matrix columns: {matrix_with_stored.shape[1]} -> {tile_matrix.shape[1]} "
                        f"(removed {matrix_with_stored.shape[1] - tile_matrix.shape[1]} blacklisted columns)"
                    )
                else:
                    # Matrix already matches filtered tiles (or was recomputed)
                    tile_matrix = matrix_with_stored
            except Exception as e:
                # If there's an error, fall back to computing with filtered tiles
                logging.warning(
                    f"Could not filter existing matrix columns: {e}. "
                    "Computing matrix with filtered tiles instead."
                )
                tile_matrix = self.get_count_matrix(
                    peaks=tiles, peak_name="tiles", overwrite=overwrite
                )
        else:
            # No blacklist filtering or overwrite=True: compute matrix normally
            tile_matrix = self.get_count_matrix(
                peaks=tiles, peak_name="tiles", overwrite=overwrite
            )

        # Ensure tiles match matrix columns (should already match, but verify)
        tiles = ensure_peaks_match_matrix(
            peaks=tiles,  # Blacklist-filtered tiles
            matrix=tile_matrix,
            stored_peaks=None,  # Don't use stored tiles (they don't have blacklist filtering)
        )

        return tile_matrix, tiles

    def create_tile_matrix(
        self, region: pr.PyRanges, tile_size: int = 50
    ) -> csr_matrix:
        """
        Create a tile matrix for a genomic region.

        Parameters
        ----------
        region : pr.PyRanges
            PyRanges object defining the region to tile
        tile_size : int
            Size of tiles in base pairs

        Returns
        -------
        scipy.sparse.csr_matrix
            Matrix of tile counts
        """
        # Create tiles
        tiles = region.tile(tile_size).df

        # Get count matrix using existing method
        tile_matrix = self.get_count_matrix(tiles, name="tile")

        return tile_matrix

    def cluster_adata(
        self,
        adata: sc.AnnData,
        filter_quantile: float = 0.995,
        total_features: int = 100_000,
        scale_factor: float = 100000,
        binarize: bool = True,
        n_components: int = 30,
        device: str = "auto",
        return_intermediate: bool = False,
        blacklist: str = None,
    ) -> sc.AnnData | tuple:
        """
        Perform clustering on an AnnData object using TF-IDF and SVD.

        This method delegates to the utility function cluster_adata_utility()
        for backend-agnostic clustering operations.

        Parameters
        ----------
        adata : sc.AnnData
            AnnData object containing raw counts and metadata
        filter_quantile : float
            Quantile for filtering features
        total_features : int
            Number of top features to keep
        binarize : bool
            Whether to binarize the data before TF-IDF
        n_components : int
            Number of SVD components to compute
        device : str
            Device to use for torch computations ('cuda' or 'cpu')
        return_intermediate : bool
            Whether to return intermediate results
        blacklist : str, optional
            Path to blacklist file. If None, uses assembly-specific blacklist.

        Returns
        -------
        sc.AnnData or tuple
            AnnData object with clustering results if return_intermediate=False
            (AnnData, tfidf_matrix, svd_components) if return_intermediate=True
        """
        # Delegate to utility function
        result = cluster_adata_utility(
            adata=adata,
            n_components=n_components,
            device=device,
            filter_quantile=filter_quantile,
            total_features=total_features,
            scale_factor=scale_factor,
            binarize=binarize,
            blacklist=blacklist,
            assembly=self.assembly,
            return_intermediate=return_intermediate,
        )

        return result

    def cluster_region(
        self,
        region: pr.PyRanges,
        tile_size: int = 50,
        min_tile_count: int = 10,
        n_components: int = 30,
        device: str = "auto",
        return_intermediate: bool = False,
    ) -> tuple:
        """
        Perform clustering on a genomic region using TF-IDF and SVD.

        Parameters
        ----------
        region : pr.PyRanges
            PyRanges object defining the region to analyze
        tile_size : int
            Size of tiles in base pairs
        min_tile_count : int
            Minimum count for a tile to be included
        n_components : int
            Number of SVD components to compute
        device : str
            Device to use for torch computations ('cuda' or 'cpu')
        return_intermediate : bool
            Whether to return intermediate results (TF-IDF matrix, SVD components)

        Returns
        -------
        tuple
            (clusters, metadata, AnnData) if return_intermediate=False
            (clusters, metadata, AnnData, tfidf_matrix, svd_components) if return_intermediate=True
        """
        import scanpy as sc
        from sklearn.feature_extraction.text import TfidfTransformer

        # Create tile matrix
        tile_matrix = self.create_tile_matrix(region, tile_size)
        device = get_device(device)
        # Convert to dense if it's a dict of matrices (MultiSparseZarrIO case)
        if isinstance(tile_matrix, dict):
            tile_matrix = np.vstack([m.todense() for m in tile_matrix.values()])
        else:
            tile_matrix = tile_matrix.todense()

        # Filter low-count tiles
        tile_sums = np.array(tile_matrix.sum(axis=0)).flatten()
        high_count_tiles = tile_sums > min_tile_count
        tile_matrix = tile_matrix[:, high_count_tiles]

        # TF-IDF transformation
        tfidf = TfidfTransformer(smooth_idf=False)
        tfidf_matrix = tfidf.fit_transform(tile_matrix)

        # Use torch for SVD if available and CUDA requested, otherwise use sklearn
        if device == "cuda" and is_torch_available():
            import torch

            # Convert to torch tensor and move to device
            tfidf_tensor = torch.tensor(tfidf_matrix.todense(), dtype=torch.float32)
            tfidf_tensor = tfidf_tensor.to(device)

            # Compute SVD
            svd = torch.svd_lowrank(tfidf_tensor, q=n_components)
            svd_components = svd[0].cpu().numpy()
        else:
            # Fallback to sklearn TruncatedSVD
            from sklearn.decomposition import TruncatedSVD

            svd_model = TruncatedSVD(n_components=n_components)
            svd_components = svd_model.fit_transform(tfidf_matrix)

        # Create AnnData object for visualization
        adata = sc.AnnData(svd_components)

        # Add metadata
        metadata = pd.DataFrame(
            {
                "n_fragments": tile_matrix.sum(axis=1).A1
                if isinstance(tile_matrix, csr_matrix)
                else tile_matrix.sum(axis=1)
            }
        )

        # Compute neighbors
        sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")

        # leiden clustering
        sc.tl.leiden(adata)

        metadata["clusters"] = adata.obs["leiden"]
        clusters = adata.obs["leiden"].values

        # Compute UMAP
        sc.tl.umap(adata)

        if return_intermediate:
            return clusters, metadata, adata, tfidf_matrix, svd_components
        return clusters, metadata, adata

    def get_cluster_queries(
        self, cluster_ids: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """
        Generate cell index queries for multiple clusters.

        Args:
            cluster_ids: List of cluster IDs to generate queries for.
                       If None, uses all available clusters.

        Returns:
            Dictionary mapping cluster IDs to {data_key: cell_idx}.
        """
        if cluster_ids is None:
            cluster_ids = self.metadata.labels.unique()
            # remove cluster_unassigned - convert to list and filter explicitly
            cluster_ids = [cid for cid in cluster_ids if cid != "cluster_unassigned"]

        queries = {}
        for cluster_id in cluster_ids:
            df_subset = self.metadata[self.metadata.labels == cluster_id]
            # get {data_key: cell_idx} mapping
            data_key_to_barcode = (
                df_subset.groupby("data_key")["cell_idx"].apply(list).to_dict()
            )
            queries[cluster_id] = data_key_to_barcode
        return queries
