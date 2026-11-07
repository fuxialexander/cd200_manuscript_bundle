"""Clustering processor for single-cell ATAC data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec
from zarr.core.dtype import VariableLengthUTF8

from caesar.io.zarr_io import MultiSparseZarrIO, SingleSparseZarrIO
from caesar.io.bpcell_io import MultiSparseBPCellsIO, SingleSparseBPCellsIO
from caesar.preprocess.core.base import AnnotationProcessor
from caesar.utils import get_device

if TYPE_CHECKING:
    import argparse
    from caesar.preprocess.core.config import CaesarConfig


class ClusteringProcessor(AnnotationProcessor):
    """Add clustering annotations to single-cell data.

    This processor performs dimensionality reduction and clustering
    on single-cell ATAC data, adding cluster labels as cell metadata.
    """

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add clustering-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Input/Output
        parser.add_argument(
            "-i", "--input", required=False,
            help="Input sparse Zarr store or AnnData file"
        )
        parser.add_argument(
            "-o", "--output", help="Output path (optional)"
        )

        # Clustering parameters
        parser.add_argument(
            "--n-components", type=int,
            help="Number of LSI components (default: 30)"
        )
        parser.add_argument(
            "--resolution", type=float,
            help="Leiden clustering resolution (default: 0.8)"
        )
        parser.add_argument(
            "--device", choices=["auto", "cpu", "cuda"],
            help="Device for computation (default: auto)"
        )
        parser.add_argument(
            "--no-blacklist", action="store_true",
            help="Disable blacklist region filtering"
        )
        parser.add_argument(
            "--overwrite", action="store_true",
            help="Overwrite existing clustering results"
        )

    def __init__(self, config: CaesarConfig):
        """Initialize clustering processor with unified configuration.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        # Direct access to clustering configuration
        self.clustering_config = config.clustering

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate clustering configuration."""
        # If clustering disabled, check if we can use existing metadata
        if not self.clustering_config.enabled:
            if not self._can_use_existing_metadata():
                raise ValueError(
                    "Clustering is disabled but required groupby column not found in metadata. "
                    f"Either enable clustering or provide metadata with the '{self.config.pseudobulk.groupby}' column."
                )
            return  # Skip clustering parameter validation

        # Validate clustering parameters when enabled
        if self.clustering_config.n_components < 2:
            raise ValueError("n_components must be at least 2")

        if self.clustering_config.resolution <= 0:
            raise ValueError("resolution must be positive")

        # Note: Input validation moved to process() method since
        # input_zarr is set by workflow engine after initialization

    def _can_use_existing_metadata(self) -> bool:
        """Check if required groupby column exists in metadata."""
        target_column = self.config.pseudobulk.groupby

        # Check global metadata file
        if self.config.data.metadata:
            try:
                df = pd.read_csv(self.config.data.metadata)
                if target_column in df.columns:
                    self.logger.info(
                        f"Found '{target_column}' column in global metadata: {self.config.data.metadata}"
                    )
                    return True
            except Exception as e:
                self.logger.warning(
                    f"Could not read global metadata {self.config.data.metadata}: {e}"
                )

        # Check per-fragment metadata files
        if self.config.data.fragments:
            for fragment in self.config.data.fragments:
                if fragment.metadata:
                    try:
                        df = pd.read_csv(fragment.metadata)
                        # Use fragment-specific metadata_column or fall back to global groupby
                        check_column = fragment.metadata_column or target_column
                        if check_column in df.columns:
                            self.logger.info(
                                f"Found '{check_column}' column in fragment metadata: {fragment.metadata}"
                            )
                            return True
                    except Exception as e:
                        self.logger.warning(
                            f"Could not read fragment metadata {fragment.metadata}: {e}"
                        )

        return False

    def _detect_input_type(self, input_path: Path) -> str:
        """Detect input type: AnnData file, BPCells fragments, or stores.

        Args:
            input_path: Path to the input data

        Returns:
            Input type: 'anndata_file', 'bpcells_fragments',
                       'single_sparse_zarr', 'multi_sparse_zarr',
                       'single_sparse_bpcells', or 'multi_sparse_bpcells'
        """
        # Check for .h5ad file
        if input_path.suffix == '.h5ad':
            self.logger.info(f"Detected AnnData file at {input_path}")
            return "anndata_file"

        # Check for BPCells fragments (has chr_ptr but no metadata.csv)
        # Raw fragments from BPCellsProcessor import have chr_ptr/chr_names but no processed metadata
        has_chr_ptr = (input_path / "chr_ptr").exists()
        has_chr_names = (input_path / "chr_names").exists()
        has_version = (input_path / "version").exists()
        has_metadata = (input_path / "metadata.csv").exists()

        # Check for Zarr signature (chrs directory)
        has_chrs_dir = (input_path / "chrs").exists()

        if has_chr_ptr and has_chr_names and has_version:
            if not has_metadata:
                # Raw BPCells fragments (not yet processed)
                self.logger.info(f"Detected BPCells fragments at {input_path}")
                return "bpcells_fragments"
            else:
                # Processed SingleSparseBPCells store
                self.logger.info(f"Detected SingleSparseBPCells store at {input_path}")
                return "single_sparse_bpcells"
        elif has_chrs_dir:
            # SingleSparseZarr store
            self.logger.info(f"Detected SingleSparseZarr store at {input_path}")
            return "single_sparse_zarr"
        else:
            # Check if it's a multi-store by looking at subdirectories
            subdirs = [d for d in input_path.iterdir() if d.is_dir()]

            if not subdirs:
                raise ValueError(
                    f"Cannot determine input type for {input_path}: "
                    "no recognized structure found"
                )

            # Check first subdirectory to determine type
            first_subdir = subdirs[0]
            subdir_has_chr_ptr = (first_subdir / "chr_ptr").exists()
            subdir_has_chr_names = (first_subdir / "chr_names").exists()
            subdir_has_chrs = (first_subdir / "chrs").exists()

            if subdir_has_chr_ptr and subdir_has_chr_names:
                self.logger.info(f"Detected MultiSparseBPCells store at {input_path}")
                return "multi_sparse_bpcells"
            elif subdir_has_chrs:
                self.logger.info(f"Detected MultiSparseZarr store at {input_path}")
                return "multi_sparse_zarr"
            else:
                raise ValueError(
                    f"Cannot determine input type for {input_path}: "
                    f"subdirectory {first_subdir} has unknown structure"
                )

    def _process_anndata_file(self, adata_path: Path) -> dict[str, Any]:
        """Process pre-generated AnnData file (BPCells fast path).

        Args:
            adata_path: Path to .h5ad file

        Returns:
            Processing results dictionary
        """
        import anndata as ad
        from caesar.analysis.clustering_ops import cluster_adata, save_clustering_results

        # Load AnnData
        self.logger.info(f"Loading AnnData from {adata_path}")
        adata = ad.read_h5ad(adata_path)

        self.logger.info(f"Loaded AnnData: {adata.n_obs} cells x {adata.n_vars} features")

        # Check if already clustered and overwrite is False
        if "leiden" in adata.obs.columns and not self.clustering_config.overwrite:
            self.logger.info("AnnData already clustered and overwrite=False, skipping")
            return {
                "output_path": str(adata_path),
                "status": "skipped",
                "message": "Already clustered",
            }

        # Perform clustering
        self.logger.info("Clustering AnnData...")
        adata = cluster_adata(
            adata=adata,
            n_components=self.clustering_config.n_components,
            device=self.clustering_config.device,
            total_features=self.clustering_config.total_features,
            blacklist=self.config.blacklist_file,
            assembly=self.config.assembly,
        )

        # Add labels
        adata.obs["labels"] = "cluster_" + adata.obs["leiden"].astype(str)
        n_clusters = len(adata.obs["leiden"].unique())

        self.logger.info(f"Clustering completed with {n_clusters} clusters")

        # Save back to same location
        self.logger.info(f"Saving clustered AnnData to {adata_path}")
        paths = save_clustering_results(
            adata,
            adata_path.parent,
            filename=adata_path.name,
            export_metadata=True,
        )

        runtime = self._end_processing()

        return {
            "output_path": str(paths["adata"]),
            "metadata_path": str(paths["metadata"]),
            "n_cells": adata.n_obs,
            "n_clusters": n_clusters,
            "method": self.clustering_config.method,
            "runtime": runtime,
        }

    def process(self) -> dict[str, Any]:
        """Perform clustering on single-cell data.

        Returns:
            Processing results dictionary
        """
        self._start_processing()

        # Get target path for in-place modification
        input_path = self.validate_input_path(self.get_target_path())

        # Check if clustering is disabled and we can use existing metadata
        if not self.clustering_config.enabled:
            if self._can_use_existing_metadata():
                self.logger.info(
                    f"Skipping clustering - using existing '{self.config.pseudobulk.groupby}' "
                    "column from metadata"
                )
                runtime = self._end_processing()
                return {
                    "output_path": str(input_path),
                    "status": "skipped",
                    "message": f"Using existing {self.config.pseudobulk.groupby} from metadata",
                    "runtime": runtime,
                }
            else:
                # This should not happen due to validate_config, but add safety check
                raise ValueError(
                    f"Clustering disabled but '{self.config.pseudobulk.groupby}' column not found in metadata"
                )

        # Auto-detect input type (AnnData file, BPCells fragments, or stores)
        input_type = self._detect_input_type(input_path)

        # Route to appropriate handler
        if input_type == "anndata_file":
            # Fast path for BPCells-generated AnnData
            return self._process_anndata_file(input_path)
        elif input_type == "bpcells_fragments":
            # Raw BPCells fragments - requires BPCellsProcessor first
            raise ValueError(
                f"Input is raw BPCells fragments directory: {input_path}\n\n"
                "Please run BPCellsProcessor with 'output_type: anndata' first to generate "
                "the AnnData file, then run ClusteringProcessor on the .h5ad file.\n\n"
                "Example workflow:\n"
                "  1. caesar bpcells --output-type anndata -i fragments.tsv.gz -o output/\n"
                "  2. caesar clustering -i output/anndata/anndata.h5ad"
            )

        # Open store with appropriate IO class
        if input_type == "single_sparse_zarr":
            io = SingleSparseZarrIO(path=str(input_path), mode="a")
        elif input_type == "multi_sparse_zarr":
            io = MultiSparseZarrIO(path=str(input_path), mode="a")
        elif input_type == "single_sparse_bpcells":
            io = SingleSparseBPCellsIO(path=str(input_path), mode="a")
        elif input_type == "multi_sparse_bpcells":
            io = MultiSparseBPCellsIO(path=str(input_path), mode="a")
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        # Check if already has clustering and overwrite is False
        if hasattr(io, "clustering") and not self.clustering_config.overwrite:
            self.logger.info("Clustering already exists and overwrite=False, skipping")
            return {
                "output_path": str(input_path),
                "status": "skipped",
                "message": "Clustering already exists",
            }

        self.logger.info(
            f"Starting clustering with method: {self.clustering_config.method}"
        )

        # Set device for clustering - use utility function for consistent handling
        device = get_device(self.clustering_config.device)

        # The output file will be saved by the method at Path(io.path) / "genome_tiles.h5ad"
        output_file = input_path / "genome_tiles.h5ad"

        # Perform clustering using Caesar's native method
        try:
            # The create_and_cluster_tiles_adata method handles:
            # 1. Creating genome-wide tiles
            # 2. Aggregating fragments into tiles
            # 3. TF-IDF transformation
            # 4. LSI (SVD) dimensionality reduction
            # 5. Leiden clustering
            # 6. UMAP visualization
            # Create and cluster tiles - ensure no cell filtering happens
            # Pass min_libsize=0 and max_libsize=None to disable library size filtering
            adata = io.create_and_cluster_tiles_adata(
                n_components=self.clustering_config.n_components,
                device=device,
                filter_quantile=0.995,  # Default value from method signature
                total_features=self.clustering_config.total_features,
                binarize=True,  # Default value from method signature
                overwrite=self.clustering_config.overwrite,
                return_intermediate=False,
                blacklist=self.config.blacklist_file,
            )
            
            # Ensure adata.obs is a DataFrame (safety check)
            if not isinstance(adata.obs, pd.DataFrame):
                raise ValueError(
                    f"Expected adata.obs to be a DataFrame, but got {type(adata.obs)}. "
                    "This may indicate an issue with the clustering method."
                )
            
            # Log adata shape to verify no cells were filtered
            self.logger.info(
                f"After clustering: adata.obs has {len(adata.obs)} rows, "
                f"expected ~{len(io.metadata)} cells from metadata"
            )

            # Get cluster information from the returned adata
            n_clusters = len(adata.obs["leiden"].unique())

            self.logger.info(f"Clustering completed with {n_clusters} clusters")

            # Update the store with the new cluster labels
            self.logger.info("Updating store with cluster labels")

            # Determine if BPCells or Zarr backend
            is_bpcells = isinstance(io, (SingleSparseBPCellsIO, MultiSparseBPCellsIO))
            is_multi = hasattr(io, "zarr_dict") or hasattr(io, "bpcells_dict")

            if not is_multi:
                # For single store (Zarr or BPCells)
                # Get all barcodes from the store in the correct order
                # Use index since barcodes are now the DataFrame index
                all_barcodes = io.metadata.index.values
                
                self.logger.info(
                    f"Total barcodes in metadata: {len(all_barcodes)}, "
                    f"Barcodes in adata after clustering: {len(adata.obs)}"
                )
                
                # Check if adata.obs has barcodes column or if barcodes are the index
                # Barcodes should be in the index (from metadata) or in a column
                if "barcodes" in adata.obs.columns:
                    adata_barcodes = adata.obs.barcodes.values
                elif adata.obs.index.name == "barcodes":
                    adata_barcodes = adata.obs.index.values
                else:
                    # Barcodes are likely the index (from metadata DataFrame)
                    adata_barcodes = adata.obs.index.values
                    self.logger.info(
                        f"Using index as barcodes (index name: {adata.obs.index.name})"
                    )
                
                # Ensure adata.obs["labels"] exists and has correct length
                if "labels" not in adata.obs.columns:
                    raise ValueError(
                        f"'labels' column not found in adata.obs. Available columns: {list(adata.obs.columns)}"
                    )
                
                adata_labels = adata.obs["labels"].values
                
                self.logger.info(
                    f"adata.obs shape: {adata.obs.shape}, "
                    f"adata_barcodes length: {len(adata_barcodes)}, "
                    f"adata_labels length: {len(adata_labels)}"
                )
                
                if len(adata_labels) != len(adata_barcodes):
                    raise ValueError(
                        f"Length mismatch: adata.obs['labels'] has {len(adata_labels)} values, "
                        f"but adata has {len(adata_barcodes)} barcodes"
                    )
                
                # Create a mapping from the adata barcodes to cluster labels
                barcode_to_label = dict(
                    zip(adata_barcodes, adata_labels)
                )
                
                self.logger.info(
                    f"Created barcode_to_label mapping with {len(barcode_to_label)} entries"
                )

                # Create labels array aligned with store barcodes
                # All barcodes in metadata should get a label (either from clustering or "cluster_unassigned")
                cluster_labels_list = [
                    barcode_to_label.get(bc, "cluster_unassigned")
                    for bc in all_barcodes
                ]
                cluster_labels = np.array(cluster_labels_list, dtype=object)
                
                self.logger.info(
                    f"Created cluster_labels with length: {len(cluster_labels)}, "
                    f"Unique labels: {len(np.unique(cluster_labels))}, "
                    f"all_barcodes length: {len(all_barcodes)}, "
                    f"barcode_to_label size: {len(barcode_to_label)}"
                )
                
                # Validate lengths match
                if len(cluster_labels) != len(all_barcodes):
                    raise ValueError(
                        f"cluster_labels length ({len(cluster_labels)}) does not match "
                        f"all_barcodes length ({len(all_barcodes)})"
                    )

                if is_bpcells:
                    # For BPCells: Update metadata.csv file with labels
                    metadata_df = io.metadata.copy()
                    
                    # Ensure cluster_labels matches metadata_df length
                    if len(cluster_labels) != len(metadata_df):
                        raise ValueError(
                            f"Length mismatch: cluster_labels has {len(cluster_labels)} elements, "
                            f"but metadata_df has {len(metadata_df)} rows. "
                            f"adata has {len(adata.obs)} barcodes, "
                            f"all_barcodes has {len(all_barcodes)} barcodes."
                        )
                    
                    metadata_df["labels"] = cluster_labels

                    # Save updated metadata
                    metadata_path = Path(io.path) / "metadata.csv"
                    metadata_df.to_csv(metadata_path, index=True)

                    # Also save attrs to attrs.json
                    import json
                    attrs_path = Path(io.path) / "attrs.json"
                    attrs_data = {}
                    if attrs_path.exists():
                        with open(attrs_path, 'r') as f:
                            attrs_data = json.load(f)

                    attrs_data["n_clusters"] = n_clusters
                    attrs_data["clustering_completed"] = True

                    with open(attrs_path, 'w') as f:
                        json.dump(attrs_data, f, indent=2)

                    self.logger.info(f"Updated BPCells metadata at {metadata_path}")
                else:
                    # For Zarr: Create labels dataset
                    # Ensure cluster_labels matches expected length
                    if len(cluster_labels) != len(all_barcodes):
                        raise ValueError(
                            f"Length mismatch for Zarr: cluster_labels has {len(cluster_labels)} elements, "
                            f"but all_barcodes has {len(all_barcodes)} elements. "
                            f"adata has {len(adata.obs)} barcodes."
                        )
                    
                    compressor = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")
                    labels_arr = io.dataset.create_array(
                        "labels",
                        shape=cluster_labels.shape,
                        dtype=VariableLengthUTF8(),
                        chunks=(1000,),
                        compressors=compressor,
                        overwrite=True,
                    )
                    labels_arr[:] = cluster_labels

                    # Update attributes
                    io.dataset.attrs["n_clusters"] = n_clusters
                    io.dataset.attrs["clustering_completed"] = True

            else:
                # For MultiSparse stores, handle each child store separately
                if is_bpcells:
                    # MultiSparseBPCellsIO
                    for data_key, bpcells_store in io.bpcells_dict.items():
                        all_barcodes = bpcells_store.metadata.index.values

                        # Filter adata for this data_key
                        data_key_mask = adata.obs["data_key"] == data_key
                        data_key_adata = adata.obs[data_key_mask]
                        # Ensure it's a DataFrame (not Series if single row)
                        if not isinstance(data_key_adata, pd.DataFrame):
                            data_key_adata = data_key_adata.to_frame().T
                        barcode_to_label = dict(
                            zip(data_key_adata["barcodes"].values, data_key_adata["labels"].values)
                        )

                        cluster_labels = np.array(
                            [
                                barcode_to_label.get(bc, "cluster_unassigned")
                                for bc in all_barcodes
                            ]
                        )

                        # Update metadata.csv for this sample
                        metadata_df = bpcells_store.metadata.copy()
                        metadata_df["labels"] = cluster_labels

                        metadata_path = Path(bpcells_store.path) / "metadata.csv"
                        metadata_df.to_csv(metadata_path, index=True)

                        # Update attrs.json
                        import json
                        attrs_path = Path(bpcells_store.path) / "attrs.json"
                        attrs_data = {}
                        if attrs_path.exists():
                            with open(attrs_path, 'r') as f:
                                attrs_data = json.load(f)

                        attrs_data["n_clusters"] = n_clusters
                        attrs_data["clustering_completed"] = True

                        with open(attrs_path, 'w') as f:
                            json.dump(attrs_data, f, indent=2)

                        self.logger.info(f"Updated BPCells metadata at {metadata_path}")
                else:
                    # MultiSparseZarrIO
                    for data_key, zarr_store in io.zarr_dict.items():
                        all_barcodes = zarr_store.metadata.barcodes.values

                        # Filter adata for this data_key
                        data_key_mask = adata.obs["data_key"] == data_key
                        data_key_adata = adata.obs[data_key_mask]
                        # Ensure it's a DataFrame (not Series if single row)
                        if not isinstance(data_key_adata, pd.DataFrame):
                            data_key_adata = data_key_adata.to_frame().T
                        barcode_to_label = dict(
                            zip(data_key_adata["barcodes"].values, data_key_adata["labels"].values)
                        )

                        cluster_labels = np.array(
                            [
                                barcode_to_label.get(bc, "cluster_unassigned")
                                for bc in all_barcodes
                            ]
                        )

                        compressor = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")
                        labels_arr = zarr_store.dataset.create_array(
                            "labels",
                            shape=cluster_labels.shape,
                            dtype=VariableLengthUTF8(),
                            chunks=(1000,),
                            compressors=compressor,
                            overwrite=True,
                        )
                        labels_arr[:] = cluster_labels

                        zarr_store.dataset.attrs["n_clusters"] = n_clusters
                        zarr_store.dataset.attrs["clustering_completed"] = True

            self.logger.info(
                f"Successfully updated store with {n_clusters} cluster labels"
            )

        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}")
            raise

        runtime = self._end_processing()

        return {
            "output_path": str(input_path),
            "clustering_file": str(output_file),
            "n_cells": adata.n_obs,
            "n_clusters": n_clusters,
            "method": self.clustering_config.method,
            "runtime": runtime,
        }
