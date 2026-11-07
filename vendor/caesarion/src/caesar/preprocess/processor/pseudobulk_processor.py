"""Pseudobulk processor with simplified configuration access.

Supports both Zarr sparse stores and BPCells fragments directories.
When BPCells fragments are detected, routes to BPCells-native pseudobulk
processing using build_cell_groups + precalculate_insertion_counts.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import zarr
from tqdm import tqdm

from caesar.io.aggregator import ByKeyAggregator
from caesar.io.query import MetadataQuery, MultiKeyQuery
from caesar.io.zarr_io import open_zarr
from caesar.preprocess.core.base import TransformationProcessor
from caesar.preprocess.core.exceptions import ProcessingError, ValidationError

if TYPE_CHECKING:
    import argparse

    import numpy as np

    from caesar.preprocess.core.config import CaesarConfig

# Conditional import for BPCells
try:
    import bpcells.cpp
    from bpcells.experimental.fragments import (
        build_cell_groups,
        precalculate_insertion_counts,
    )

    HAS_BPCELLS = True
except ImportError:
    HAS_BPCELLS = False


class PseudobulkProcessor(TransformationProcessor):
    """Pseudobulk processor with flexible metadata-based grouping.

    This processor handles aggregation of single-cell data into
    pseudobulk or other grouped formats. It supports multiple ways
    to group cells:

    1. **Default labels**: Use the 'labels' stored in the zarr file
    2. **Alternative zarr column**: Specify a different column via metadata_column
    3. **External metadata**: Load new groupings from an external CSV/TSV file
    4. **Custom grouping file**: Legacy support for grouping files

    Metadata Handling Priority:
    - external_metadata (if specified) takes precedence
    - metadata_column (if specified) selects alternative zarr column
    - Default to 'labels' column from zarr

    The processor now uses 'labels' consistently instead of 'clusters'
    for better clarity and alignment with the rest of the codebase.
    """

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add pseudobulk-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Pseudobulk-specific arguments
        parser.add_argument(
            "-i", "--input", required=True, help="Input sparse zarr path"
        )
        parser.add_argument(
            "-o",
            "--output",
            default="pseudobulked.zarr",
            help="Output dense zarr path (default: pseudobulked.zarr)",
        )
        parser.add_argument(
            "--groupby",
            default="labels",
            help="Grouping variable (default: labels)",
        )
        parser.add_argument(
            "--grouping-file",
            help="Custom grouping file (CSV with barcode and group columns)",
        )
        parser.add_argument(
            "--metadata-column",
            help="Column name from zarr metadata to group by (default: use stored labels)",
        )
        parser.add_argument(
            "--external-metadata",
            help="External metadata file for regrouping (CSV/TSV with barcodes)",
        )
        parser.add_argument(
            "--barcode-column",
            default="barcode",
            help="Column name for barcodes in external metadata (default: barcode)",
        )
        parser.add_argument(
            "--min-cells",
            type=int,
            default=100,
            help="Minimum cells per group (default: 100)",
        )
        parser.add_argument(
            "--min-libsize",
            type=int,
            default=3_000_000,
            help="Minimum library size per group (default: 3000000)",
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return (
            "Aggregate single-cell data into pseudobulk samples. "
            "Groups cells by metadata (clusters, cell types, conditions) to create dense coverage tracks."
        )

    def __init__(self, config: CaesarConfig):
        """Initialize PseudoBulkProcessor.

        Args:
            config: Structured CaesarConfig instance
        """
        super().__init__(config)

        # Store full config for direct access
        self.config = config

        # Direct access to configuration - clean and simple
        self.groupby = config.pseudobulk.groupby
        self.min_cells_per_group = config.pseudobulk.min_cells_per_group
        self.min_libsize = config.pseudobulk.min_libsize
        self.grouping_file = config.pseudobulk.grouping_file
        self.source = config.pseudobulk.source

        # Metadata handling
        self.metadata_column = getattr(config.pseudobulk, "metadata_column", None)
        self.external_metadata = getattr(config.pseudobulk, "external_metadata", None)
        self.barcode_column = getattr(config.pseudobulk, "barcode_column", "barcode")

        # New sampling strategy fields
        self.sampling_strategy = getattr(
            config.pseudobulk, "sampling_strategy", None
        )  # None, 'cell_count', 'insertion_count'
        self.n_cells_per_group = getattr(
            config.pseudobulk, "n_cells_per_group", None
        )  # For cell_count strategy
        self.target_insertions = getattr(
            config.pseudobulk, "target_insertions", None
        )  # For insertion_count strategy
        self.integrated = getattr(
            config.pseudobulk, "integrated", False
        )  # For multi-sample integration
        self.expression_data = getattr(
            config.pseudobulk, "expression_data", None
        )  # Path to expression data
        self.expression_groupby = getattr(
            config.pseudobulk, "expression_groupby", "labels"
        )  # Column for expression grouping

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate pseudobulk configuration."""
        if self.min_cells_per_group < 1:
            raise ValidationError("min_cells_per_group must be at least 1")

        if self.min_libsize < 0:
            raise ValidationError("min_libsize must be non-negative")

        # Validate grouping file if specified
        if self.grouping_file:
            grouping_path = Path(self.grouping_file)
            if not grouping_path.exists():
                raise ValidationError(f"Grouping file not found: {grouping_path}")

            if grouping_path.suffix not in [".csv", ".tsv", ".txt"]:
                raise ValidationError(
                    f"Grouping file must be CSV or TSV, got: {grouping_path.suffix}"
                )

        # Validate sampling strategy
        if self.sampling_strategy:
            valid_strategies = ["cell_count", "insertion_count"]
            if self.sampling_strategy not in valid_strategies:
                raise ValidationError(
                    f"Invalid sampling strategy: {self.sampling_strategy}. Must be one of {valid_strategies}"
                )

            if self.sampling_strategy == "cell_count" and not self.n_cells_per_group:
                raise ValidationError(
                    "n_cells_per_group must be specified for cell_count strategy"
                )

            if (
                self.sampling_strategy == "insertion_count"
                and not self.target_insertions
            ):
                raise ValidationError(
                    "target_insertions must be specified for insertion_count strategy"
                )

        # Validate external metadata if specified
        if self.external_metadata:
            metadata_path = Path(self.external_metadata)
            if not metadata_path.exists():
                raise ValidationError(
                    f"External metadata file not found: {metadata_path}"
                )

        # Validate metadata column consistency
        if self.metadata_column and self.external_metadata:
            raise ValidationError(
                "Cannot specify both metadata_column and external_metadata. "
                "Use metadata_column for zarr columns, external_metadata for external files."
            )

        # Validate expression data if specified
        if self.expression_data:
            expr_path = Path(self.expression_data)
            if not expr_path.exists():
                raise ValidationError(f"Expression data not found: {expr_path}")

            valid_extensions = [".h5ad", ".h5", ".mtx"]
            if not any(str(expr_path).endswith(ext) for ext in valid_extensions):
                raise ValidationError(
                    f"Expression data must be in h5ad, h5, or mtx format, got: {expr_path.suffix}"
                )

    def _detect_store_type(self, input_path: Path) -> str:
        """Detect the type of sparse store (Zarr or BPCells).

        Returns:
            Store type: 'zarr_sparse' (for zarr sparse stores),
                       'bpcells_fragments' (for BPCells fragments),
                       'bpcells_fragments_multi' (for multi-sample BPCells fragments)
        """
        # First try zarr detection
        from caesar.io.zarr_io.base import ZarrIO

        store_type = ZarrIO.detect_store_type(input_path)

        if store_type in ["multiome_sparse", "single_sparse", "multi_sparse"]:
            self.logger.info(f"Detected {store_type} zarr at {input_path}")
            return "zarr_sparse"

        # Try BPCells detection
        try:
            from caesar.io.bpcell_io import BPCellsIO

            bpcells_type = BPCellsIO.detect_store_type(input_path)
            if bpcells_type in ["bpcells_single_sparse", "bpcells_multi_sparse"]:
                self.logger.info(f"Detected {bpcells_type} at {input_path}")
                # For sparse BPCells, we can still process using zarr-style aggregation
                return "zarr_sparse"
        except Exception:
            pass

        # Check for BPCells fragments format (has chr_ptr, chr_names, version)
        if input_path.is_dir():
            frag_markers = ["cell_names", "version", "chr_names"]
            if all((input_path / marker).exists() for marker in frag_markers):
                # Check version to distinguish from sparse matrix
                version_file = input_path / "version"
                with version_file.open() as f:
                    version_str = f.read().strip()
                if "fragments" in version_str.lower():
                    self.logger.info(f"Detected BPCells fragments at {input_path}")
                    return "bpcells_fragments"

            # Check if parent is a multi-sample fragments directory
            if (input_path / "attrs.json").exists():
                subdirs = [d for d in input_path.iterdir() if d.is_dir()]
                if subdirs and all((d / "version").exists() for d in subdirs[:3]):
                    self.logger.info(
                        f"Detected multi-sample BPCells fragments at {input_path}"
                    )
                    return "bpcells_fragments_multi"

        raise ValueError(
            f"Cannot determine store type for {input_path}. "
            "PseudobulkProcessor requires sparse zarr or BPCells fragments."
        )

    def load_grouping(self) -> pd.DataFrame | None:
        """Load custom grouping from file if specified.

        Returns:
            DataFrame with barcode and group columns, or None
        """
        if not self.grouping_file:
            return None

        grouping_path = Path(self.grouping_file)
        if not grouping_path.exists():
            raise ValidationError(f"Grouping file not found: {grouping_path}")

        # Load grouping file (CSV or TSV)
        if grouping_path.suffix == ".csv":
            df = pd.read_csv(grouping_path)
        else:
            df = pd.read_csv(grouping_path, sep="\t")

        # Validate columns
        if "barcode" not in df.columns:
            raise ValidationError("Grouping file must have 'barcode' column")

        if self.groupby not in df.columns and self.groupby != "barcode":
            raise ValueError(f"Grouping column '{self.groupby}' not found in file")

        return df

    def load_external_metadata(self, zarr_barcodes: np.ndarray) -> pd.DataFrame | None:
        """Load external metadata file for regrouping.

        Args:
            zarr_barcodes: Array of barcodes from the zarr file

        Returns:
            DataFrame with barcode and metadata columns, or None
        """
        if not self.external_metadata:
            return None

        metadata_path = Path(self.external_metadata)
        if not metadata_path.exists():
            raise ValidationError(f"External metadata file not found: {metadata_path}")

        # Load metadata file (CSV or TSV)
        if metadata_path.suffix == ".csv":
            df = pd.read_csv(metadata_path)
        else:
            df = pd.read_csv(metadata_path, sep="\t")

        # Validate columns
        if self.barcode_column not in df.columns:
            raise ValidationError(
                f"Barcode column '{self.barcode_column}' not found in metadata file. "
                f"Available columns: {list(df.columns)}"
            )

        # Check if groupby column exists
        if self.groupby not in df.columns:
            raise ValidationError(
                f"Groupby column '{self.groupby}' not found in metadata file. "
                f"Available columns: {list(df.columns)}"
            )

        # Filter to only barcodes present in zarr
        zarr_barcode_set = set(zarr_barcodes)
        df_filtered = df[df[self.barcode_column].isin(zarr_barcode_set)].copy()

        # Log statistics
        self.logger.info(
            f"Loaded external metadata: {len(df_filtered)}/{len(zarr_barcodes)} barcodes matched"
        )

        # Rename barcode column to 'barcode' for consistency
        if self.barcode_column != "barcode":
            df_filtered = df_filtered.rename(columns={self.barcode_column: "barcode"})

        return df_filtered

    def process(self) -> dict[str, Any]:
        """Process pseudobulk aggregation.

        Supports both Zarr sparse stores and BPCells fragments directories.
        Auto-detects the backend and routes to the appropriate processing method.

        Returns:
            Processing results
        """
        self._start_processing()

        # Get and validate input path
        input_path = self.validate_input_path()

        self.logger.info(f"Processing pseudobulk aggregation on {input_path}")

        # Detect store type to route to appropriate backend
        store_type = self._detect_store_type(input_path)

        # Get output path - use configured output filename
        output_path = self.get_output_path(self.config.pseudobulk.output)

        # Route to BPCells processing if detected
        if store_type in ["bpcells_fragments", "bpcells_fragments_multi"]:
            if not HAS_BPCELLS:
                raise ProcessingError(
                    "BPCells fragments detected but bpcells library is not installed. "
                    "Install with: pip install bpcells"
                )
            return self._process_bpcells_fragments(input_path, store_type, output_path)

        # Zarr sparse processing (original code path)
        z = open_zarr(str(input_path), mode="r")

        # Initialize root group
        root = zarr.open_group(str(output_path), mode="a")

        # Process the pseudobulk aggregation
        try:
            libsizes = self._process_pseudobulk(z, root)

            # Store global metadata
            # Get the actual assembly from input zarr (not "auto")
            actual_assembly = z.assembly if hasattr(z, "assembly") else self.assembly
            if (
                actual_assembly == "auto"
                and hasattr(z, "_root")
                and "assembly" in z._root.attrs
            ):
                actual_assembly = z._root.attrs["assembly"]

            root.attrs.update(
                {
                    "n_groups": len(libsizes),
                    "group_ids": sorted(list(libsizes.keys())),
                    "library_size": sum(libsizes.values()),
                    "assembly": actual_assembly,
                    "groupby": self.groupby,
                    "completed": True,
                    "zarr_type": "celltype_dense",  # Add zarr type for clarity
                    "celltypes": sorted(list(libsizes.keys())),  # Add celltype list
                }
            )

            self.logger.info(
                f"Completed pseudobulk processing. Final pseudobulk count: {len(libsizes)}"
            )

        except Exception as e:
            self.logger.error(f"Error during pseudobulk processing: {e}")
            raise ProcessingError(f"Pseudobulk processing failed: {e}") from e

        runtime = self._end_processing()

        return {
            "output_path": str(output_path),
            "n_groups": len(libsizes),
            "groupby": self.groupby,
            "library_sizes": libsizes,
            "runtime": runtime,
        }

    def _subsample_cells(
        self, z, cluster_name: str, metadata: pd.DataFrame
    ) -> MetadataQuery | MultiKeyQuery:
        """Subsample cells based on the configured sampling strategy.

        Args:
            z: SingleSparseZarrIO or MultiSparseZarrIO instance
            cluster_name: Name of the cluster/group to subsample
            metadata: Metadata DataFrame for this cluster (already filtered for the group)

        Returns:
            Query object for the subsampled cells
        """
        if not self.sampling_strategy:
            # No sampling, use cell_idx from the provided metadata
            # This is essential when using external metadata (e.g., metacell assignments)
            # where cluster_name doesn't match the zarr's labels column
            if hasattr(z, "data_keys") and len(z.data_keys) > 1:
                # Multi-sample zarr
                cell_queries = {}
                for data_key in metadata["data_key"].unique():
                    dk_metadata = metadata[metadata["data_key"] == data_key]
                    cell_queries[data_key] = dk_metadata["cell_idx"].tolist()
                return MultiKeyQuery(cell_queries)
            else:
                # Single-sample zarr - use cell_idx for precise selection
                cell_indices = metadata["cell_idx"].tolist()
                return MetadataQuery(f"cell_idx in {cell_indices}")

        if self.sampling_strategy == "cell_count":
            # Sample fixed number of cells
            n_cells = min(self.n_cells_per_group, len(metadata))
            if n_cells < len(metadata):
                # Randomly sample cells
                sampled_metadata = metadata.sample(n=n_cells, random_state=42)
                if hasattr(z, "data_keys") and len(z.data_keys) > 1:
                    # Multi-sample zarr
                    cell_queries = {}
                    for data_key in sampled_metadata["data_key"].unique():
                        dk_metadata = sampled_metadata[
                            sampled_metadata["data_key"] == data_key
                        ]
                        cell_queries[data_key] = dk_metadata["cell_idx"].tolist()
                    return MultiKeyQuery(cell_queries)
                else:
                    # Single-sample zarr
                    cell_indices = sampled_metadata["cell_idx"].tolist()
                    return MetadataQuery(f"cell_idx in {cell_indices}")
            else:
                # Use all cells if we have fewer than requested
                # Use cell_idx for precise selection (essential for external metadata)
                if hasattr(z, "data_keys") and len(z.data_keys) > 1:
                    cell_queries = {}
                    for data_key in metadata["data_key"].unique():
                        dk_metadata = metadata[metadata["data_key"] == data_key]
                        cell_queries[data_key] = dk_metadata["cell_idx"].tolist()
                    return MultiKeyQuery(cell_queries)
                else:
                    cell_indices = metadata["cell_idx"].tolist()
                    return MetadataQuery(f"cell_idx in {cell_indices}")

        elif self.sampling_strategy == "insertion_count":
            # Sample cells to reach target insertion count
            # Sort by library size (insertions per cell)
            metadata_sorted = metadata.sort_values("per_cell_libsize", ascending=False)
            metadata_sorted["cumsum_libsize"] = metadata_sorted[
                "per_cell_libsize"
            ].cumsum()

            # Find cells that collectively have approximately target_insertions
            sampled_metadata = metadata_sorted[
                metadata_sorted["cumsum_libsize"] <= self.target_insertions
            ]

            if len(sampled_metadata) == 0:
                # If even the first cell exceeds target, include at least one cell
                sampled_metadata = metadata_sorted.iloc[:1]

            if hasattr(z, "data_keys") and len(z.data_keys) > 1:
                # Multi-sample zarr
                cell_queries = {}
                for data_key in sampled_metadata["data_key"].unique():
                    dk_metadata = sampled_metadata[
                        sampled_metadata["data_key"] == data_key
                    ]
                    cell_queries[data_key] = dk_metadata["cell_idx"].tolist()
                return MultiKeyQuery(cell_queries)
            else:
                # Single-sample zarr
                cell_indices = sampled_metadata["cell_idx"].tolist()
                return MetadataQuery(f"cell_idx in {cell_indices}")

    def _load_and_aggregate_expression(
        self, cluster_name: str, sampled_cells: pd.DataFrame | None = None
    ) -> dict[str, np.ndarray] | None:
        """Load and aggregate expression data for a cluster.

        Args:
            cluster_name: Name of the cluster/group
            sampled_cells: Optional DataFrame of sampled cells to use

        Returns:
            Dictionary mapping gene names to aggregated expression values
        """
        if not self.expression_data:
            return None

        try:
            import scanpy as sc

            # Load expression data
            expr_path = Path(self.expression_data)
            if expr_path.suffix == ".h5ad":
                adata = sc.read_h5ad(expr_path)
            elif expr_path.suffix == ".h5":
                adata = sc.read_10x_h5(expr_path)
            else:
                # Assume MTX format
                adata = sc.read_10x_mtx(expr_path.parent)

            # Filter to cells in this cluster
            if self.expression_groupby in adata.obs.columns:
                cluster_mask = adata.obs[self.expression_groupby] == cluster_name
                adata_cluster = adata[cluster_mask]

                # Further filter to sampled cells if provided
                if sampled_cells is not None and "barcode" in sampled_cells.columns:
                    sampled_barcodes = set(sampled_cells["barcode"].values)
                    cell_mask = adata_cluster.obs.index.isin(sampled_barcodes)
                    adata_cluster = adata_cluster[cell_mask]

                # Aggregate expression (sum across cells)
                if adata_cluster.n_obs > 0:
                    expr_matrix = adata_cluster.X
                    if hasattr(expr_matrix, "toarray"):
                        expr_matrix = expr_matrix.toarray()

                    # Sum expression across cells
                    aggregated_expr = expr_matrix.sum(axis=0)

                    # Create gene expression dictionary
                    gene_expr = {}
                    for i, gene in enumerate(adata_cluster.var_names):
                        gene_expr[gene] = aggregated_expr[i]

                    return gene_expr

            return None

        except Exception as e:
            self.logger.warning(
                f"Failed to load expression data for {cluster_name}: {e}"
            )
            return None

    def _auto_discover_metacell_assignments(self, z) -> str | None:
        """Auto-discover metacell assignments CSV when groupby='metacell_id'.

        Args:
            z: SingleSparseZarrIO, MultiSparseZarrIO, or MultiomeSparseZarrIO instance

        Returns:
            Path to metacell_assignments.csv if found, None otherwise
        """
        if self.groupby != "metacell_id":
            return None

        if self.external_metadata:
            # User explicitly provided external metadata, don't override
            return None

        # Get zarr path
        if hasattr(z, "path"):
            zarr_path = Path(z.path)
        elif hasattr(z, "_root") and hasattr(z._root, "store") and hasattr(z._root.store, "root"):
            zarr_path = Path(str(z._root.store.root))
        elif hasattr(z, "root") and hasattr(z.root, "store") and hasattr(z.root.store, "root"):
            zarr_path = Path(str(z.root.store.root))
        else:
            self.logger.warning("Cannot determine zarr path for metacell auto-discovery")
            return None

        # Check if zarr has metacells
        has_metacells = False
        if hasattr(z, "_root") and "has_metacells" in z._root.attrs:
            has_metacells = z._root.attrs.get("has_metacells", False)
        elif hasattr(z, "root") and "has_metacells" in z.root.attrs:
            has_metacells = z.root.attrs.get("has_metacells", False)

        if not has_metacells:
            self.logger.warning(
                "groupby='metacell_id' but zarr does not have has_metacells=True. "
                "Will try to find metacell_assignments.csv anyway."
            )

        # Look for metacell_assignments.csv
        metacell_csv = zarr_path / "metacell_assignments.csv"
        if metacell_csv.exists():
            self.logger.info(f"Auto-discovered metacell assignments: {metacell_csv}")
            return str(metacell_csv)

        # Also check parent directory (for cases where zarr is nested)
        parent_csv = zarr_path.parent / "metacell_assignments.csv"
        if parent_csv.exists():
            self.logger.info(f"Auto-discovered metacell assignments: {parent_csv}")
            return str(parent_csv)

        self.logger.warning(
            f"groupby='metacell_id' but metacell_assignments.csv not found at {metacell_csv}"
        )
        return None

    def _process_pseudobulk(self, z, root) -> dict[str, int]:
        """Process pseudobulk aggregation based on sc_accessibility_simple.py.

        Args:
            z: SingleSparseZarrIO or MultiSparseZarrIO instance
            root: Zarr root group for output

        Returns:
            Dictionary mapping cluster/group names to their library sizes
        """
        # Auto-discover metacell assignments if groupby='metacell_id'
        auto_metadata = self._auto_discover_metacell_assignments(z)
        if auto_metadata:
            self.external_metadata = auto_metadata
            self.barcode_column = "barcode"

        # Get metadata for grouping
        if self.external_metadata:
            # Load external metadata for regrouping
            zarr_barcodes = (
                z.barcodes if hasattr(z, "barcodes") else z.metadata.index.values
            )
            external_df = self.load_external_metadata(zarr_barcodes)
            if external_df is None:
                raise ProcessingError("Failed to load external metadata")

            # Create metadata DataFrame that matches z.metadata structure
            # Merge with existing metadata to preserve cell indices
            z_metadata = z.metadata.copy()
            z_metadata["barcode"] = z_metadata.index

            # Merge external metadata
            merged_metadata = z_metadata.merge(
                external_df[["barcode", self.groupby]],
                on="barcode",
                how="left",
                suffixes=("_orig", ""),
            )

            # Restore barcode as index (merge resets to integer index)
            merged_metadata = merged_metadata.set_index("barcode")

            # Use the external groupby column for grouping
            group_column = self.groupby
            merged_metadata["labels"] = merged_metadata[group_column]

        elif self.metadata_column and self.metadata_column != "labels":
            # Use a different column from zarr metadata
            if not hasattr(z.metadata, self.metadata_column):
                available_cols = (
                    list(z.metadata.columns) if hasattr(z.metadata, "columns") else []
                )
                raise ValidationError(
                    f"Metadata column '{self.metadata_column}' not found in zarr. "
                    f"Available columns: {available_cols}"
                )
            merged_metadata = z.metadata.copy()
            merged_metadata["labels"] = merged_metadata[self.metadata_column]
            group_column = "labels"
        else:
            # Use default labels from zarr
            merged_metadata = z.metadata
            group_column = "labels"

        # Get unique groups
        labels = merged_metadata[group_column].unique()
        base_queries = {
            label: merged_metadata[merged_metadata[group_column] == label]
            for label in labels
            if label != "unassigned" and pd.notna(label)
        }

        # Apply sampling if configured
        queries = {}
        sampled_metadata = {}  # Store sampled cells for expression aggregation

        for label, label_metadata in base_queries.items():
            # Check minimum cell count
            if len(label_metadata) < self.min_cells_per_group:
                self.logger.info(
                    f"Skipping group {label} with {len(label_metadata)} cells < {self.min_cells_per_group}"
                )
                continue

            # Apply sampling strategy
            query = self._subsample_cells(z, label, label_metadata)
            queries[label] = query

            # Store sampled cells metadata for expression aggregation
            if self.sampling_strategy and self.expression_data:
                if self.sampling_strategy == "cell_count":
                    n_cells = min(self.n_cells_per_group, len(label_metadata))
                    sampled_metadata[label] = label_metadata.sample(
                        n=n_cells, random_state=42
                    )
                elif self.sampling_strategy == "insertion_count":
                    label_metadata_sorted = label_metadata.sort_values(
                        "per_cell_libsize", ascending=False
                    )
                    label_metadata_sorted["cumsum_libsize"] = label_metadata_sorted[
                        "per_cell_libsize"
                    ].cumsum()
                    sampled_metadata[label] = label_metadata_sorted[
                        label_metadata_sorted["cumsum_libsize"]
                        <= self.target_insertions
                    ]

        cluster_queries = MultiKeyQuery(queries)
        self.logger.info(
            f"Found {len(cluster_queries.mapping)} groups after filtering: {list(cluster_queries.mapping.keys())}"
        )

        # Initialize arrays for each group and chromosome
        from zarr.codecs import BloscCodec

        compressor = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")

        for group_name in cluster_queries.mapping:
            if group_name not in root:
                root.create_group(group_name)
            if "chrs" not in root[group_name]:
                root[group_name].create_group("chrs")

            for chrom in z.chroms:
                if chrom not in root[group_name]["chrs"]:
                    root[group_name]["chrs"].create_array(
                        chrom,
                        shape=(z.chrom_sizes[chrom],),
                        chunks=(z.chunk_size,),
                        dtype="u4",
                        compressors=compressor,
                    )

        # Process chunks in parallel
        futures = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for chrom in z.chroms:
                chrom_length = z.chrom_sizes[chrom]
                for start in range(0, chrom_length, z.chunk_size):
                    end = min(start + z.chunk_size, chrom_length)
                    futures.append(
                        executor.submit(
                            self._process_pseudobulk_chunk,
                            z=z,
                            root=root,
                            cluster_queries=cluster_queries,
                            chrom=chrom,
                            start=start,
                            end=end,
                        )
                    )

            # Collect results
            failed_chunks = []
            libsizes = {group_name: 0 for group_name in cluster_queries.mapping}

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing chunks"
            ):
                try:
                    chunk_libsizes = future.result()
                    for group_name, libsize in chunk_libsizes.items():
                        libsizes[group_name] += libsize
                except Exception as e:
                    self.logger.error(f"Failed to process chunk: {e}")
                    failed_chunks.append(len(failed_chunks))

            if failed_chunks:
                raise RuntimeError(f"Failed to process chunks: {failed_chunks}")

        # Filter groups with library size less than min_libsize
        filtered_libsizes = {}
        filtered_out_groups = []
        for group_name, libsize in libsizes.items():
            if libsize >= self.min_libsize:
                filtered_libsizes[group_name] = libsize
            else:
                self.logger.info(
                    f"Skipping group {group_name} with library size {libsize} < {self.min_libsize}"
                )
                filtered_out_groups.append(group_name)
                # Remove the group from root if it was created
                # if group_name in root:
                #     del root[group_name]

        # Log filtering summary
        self.logger.info(
            f"Filtered out {len(filtered_out_groups)} groups due to library size < {self.min_libsize}"
        )
        self.logger.info(f"After filtering: {len(filtered_libsizes)} groups remaining")

        # Store metadata in root attrs for CelltypeDenseZarrIO compatibility
        root.attrs["assembly"] = self.assembly or z.assembly
        root.attrs["zarr_type"] = "celltype_dense"
        root.attrs["celltypes"] = list(filtered_libsizes.keys())

        # Add metadata handling info
        if self.external_metadata:
            root.attrs["external_metadata"] = self.external_metadata
            root.attrs["metadata_groupby"] = self.groupby
        elif self.metadata_column:
            root.attrs["metadata_column"] = self.metadata_column

        # Add sampling metadata if applicable
        if self.sampling_strategy:
            root.attrs["sampling_strategy"] = self.sampling_strategy
            if self.sampling_strategy == "cell_count":
                root.attrs["n_cells_per_group"] = self.n_cells_per_group
            elif self.sampling_strategy == "insertion_count":
                root.attrs["target_insertions"] = self.target_insertions

        if self.integrated:
            root.attrs["integrated"] = True

        # Store metadata only for groups that meet the library size threshold
        for group_name, libsize in filtered_libsizes.items():
            root[group_name].attrs["library_size"] = libsize
            root[group_name].attrs["celltype"] = group_name
            root[group_name].attrs["assembly"] = self.assembly or z.assembly
            root[group_name].attrs["completed"] = True

            # Add expression data if available
            if self.expression_data:
                expr_data = self._load_and_aggregate_expression(
                    group_name,
                    sampled_metadata.get(group_name)
                    if self.sampling_strategy
                    else None,
                )
                if expr_data:
                    # Store aggregated expression as attributes or in a separate group
                    if "expression" not in root[group_name]:
                        root[group_name].create_group("expression")

                    # Store top expressed genes as example
                    top_genes = sorted(
                        expr_data.items(), key=lambda x: x[1], reverse=True
                    )[:100]
                    root[group_name]["expression"].attrs["top_genes"] = [
                        g[0] for g in top_genes
                    ]
                    root[group_name]["expression"].attrs["top_gene_counts"] = [
                        float(g[1]) for g in top_genes
                    ]
                    root[group_name]["expression"].attrs["total_gene_count"] = len(
                        expr_data
                    )

                    self.logger.info(
                        f"Added expression data for {group_name} with {len(expr_data)} genes"
                    )

        # Log final summary
        self.logger.info(
            f"Created {len(filtered_libsizes)} pseudobulk datasets: {list(filtered_libsizes.keys())}"
        )

        return filtered_libsizes

    def _process_pseudobulk_chunk(
        self, z, root, cluster_queries, chrom, start, end
    ) -> dict[str, int]:
        """Process a single chunk for pseudobulk creation for all groups.

        Args:
            z: SingleSparseZarrIO or MultiSparseZarrIO instance
            root: Zarr root group for output
            cluster_queries: A MultiKeyQuery object with group names as keys
            chrom: Chromosome name
            start: Start position
            end: End position

        Returns:
            Dictionary mapping group names to their chunk library sizes
        """
        try:
            chunk_libsizes = {}

            # The new query system handles both single and multi zarr transparently
            tracks = z.get_track(
                chr_name=chrom,
                start=start,
                end=end,
                query=cluster_queries,
                aggregator=ByKeyAggregator(),
                output_format="raw_dict",
            )

            # Store tracks and calculate library sizes
            for group_name, signal in tracks.items():
                signal_path = f"{group_name}/chrs/{chrom}"
                if signal_path in root:
                    root[signal_path][start:end] = signal
                chunk_libsizes[group_name] = int(signal.sum())

            return chunk_libsizes

        except Exception as e:
            self.logger.error(
                f"Failed to process chunk for groups at {chrom}:{start}-{end}: {e}"
            )
            raise

    def _load_bpcells_grouping(
        self, input_path: Path, all_cell_names: list[str]
    ) -> dict[str, str] | None:
        """Load grouping metadata for BPCells fragments.

        Searches for grouping information in the following order:
        1. External metadata file (--external-metadata)
        2. metacell_assignments.csv in the input directory
        3. AnnData file (anndata/anndata.h5ad) with groupby column

        Args:
            input_path: Path to BPCells fragments directory
            all_cell_names: List of all cell barcodes from fragments

        Returns:
            Dictionary mapping cell barcode to group ID, or None if not found
        """
        group_assignments = {}
        all_cell_set = set(all_cell_names)

        # 1. Check external metadata
        if self.external_metadata:
            metadata_path = Path(self.external_metadata)
            if metadata_path.suffix == ".csv":
                df = pd.read_csv(metadata_path)
            else:
                df = pd.read_csv(metadata_path, sep="\t")

            if self.barcode_column not in df.columns:
                raise ValidationError(
                    f"Barcode column '{self.barcode_column}' not found in external metadata"
                )
            if self.groupby not in df.columns:
                raise ValidationError(
                    f"Groupby column '{self.groupby}' not found in external metadata"
                )

            for _, row in df.iterrows():
                barcode = row[self.barcode_column]
                if barcode in all_cell_set:
                    group_assignments[barcode] = str(row[self.groupby])

            self.logger.info(
                f"Loaded {len(group_assignments)} groupings from external metadata"
            )
            return group_assignments

        # 2. Check for metacell_assignments.csv
        metacell_paths = [
            input_path / "metacell_assignments.csv",
            input_path.parent / "metacell_assignments.csv",
        ]

        for metacell_path in metacell_paths:
            if metacell_path.exists():
                df = pd.read_csv(metacell_path)
                barcode_col = "barcode" if "barcode" in df.columns else df.columns[0]

                # Use groupby column if present, otherwise default to metacell_id
                if self.groupby in df.columns:
                    group_col = self.groupby
                elif "metacell_id" in df.columns:
                    group_col = "metacell_id"
                else:
                    group_col = df.columns[1]

                for _, row in df.iterrows():
                    barcode = row[barcode_col]
                    if barcode in all_cell_set:
                        group = row[group_col]
                        if pd.notna(group):
                            group_assignments[barcode] = str(group)

                self.logger.info(
                    f"Loaded {len(group_assignments)} groupings from {metacell_path}"
                )
                return group_assignments

        # 3. Check for AnnData file
        anndata_paths = [
            input_path.parent / "anndata" / "anndata.h5ad",
            input_path.parent.parent / "anndata" / "anndata.h5ad",
        ]

        for anndata_path in anndata_paths:
            if anndata_path.exists():
                try:
                    import anndata as ad

                    adata = ad.read_h5ad(anndata_path)

                    if self.groupby not in adata.obs.columns:
                        # Try 'labels' as fallback
                        if "labels" in adata.obs.columns:
                            group_col = "labels"
                        else:
                            self.logger.warning(
                                f"Groupby column '{self.groupby}' not found in AnnData. "
                                f"Available: {list(adata.obs.columns)}"
                            )
                            continue
                    else:
                        group_col = self.groupby

                    for barcode in adata.obs.index:
                        if barcode in all_cell_set:
                            group = adata.obs.loc[barcode, group_col]
                            if pd.notna(group):
                                group_assignments[barcode] = str(group)

                    self.logger.info(
                        f"Loaded {len(group_assignments)} groupings from {anndata_path}"
                    )
                    return group_assignments

                except Exception as e:
                    self.logger.warning(f"Failed to load AnnData from {anndata_path}: {e}")

        return group_assignments if group_assignments else None

    def _detect_assembly_from_chroms(self, chrom_names: list[str]) -> str:
        """Detect assembly from chromosome names.

        Args:
            chrom_names: List of chromosome names

        Returns:
            Detected assembly ('hg38' or 'mm10')
        """
        # Check for typical hg38 chromosomes
        chrom_set = set(chrom_names)
        if "chr1" in chrom_set or "1" in chrom_set:
            # Check chromosome sizes to distinguish hg38 from mm10
            # mm10 chr1 is ~195M, hg38 chr1 is ~248M
            # For simplicity, use assembly from config if available
            if hasattr(self, "assembly") and self.assembly not in [None, "auto"]:
                return self.assembly

            # Default to hg38 for human-like chromosome names
            if "chrY" in chrom_set or "Y" in chrom_set:
                return "hg38"

        return "hg38"  # Default

    def _process_bpcells_fragments(
        self, input_path: Path, store_type: str, output_path: Path
    ) -> dict[str, Any]:
        """Process BPCells fragments to generate pseudobulk (celltype_dense).

        This method uses build_cell_groups + precalculate_insertion_counts from
        bpcells.experimental.fragments to create celltype_dense pseudobulk data.

        Args:
            input_path: Path to BPCells fragments directory
            store_type: 'bpcells_fragments' or 'bpcells_fragments_multi'
            output_path: Path to output directory

        Returns:
            Processing results dictionary
        """
        self.logger.info("Processing BPCells fragments for pseudobulk")

        # Discover fragment directories
        if store_type == "bpcells_fragments_multi":
            frag_dirs = sorted(
                [d for d in input_path.iterdir() if d.is_dir() and (d / "version").exists()]
            )
        else:
            # Single fragment directory - might be wrapped in a parent dir
            if (input_path / "chr_names").exists():
                frag_dirs = [input_path]
            else:
                # Check subdirectories
                frag_dirs = sorted(
                    [d for d in input_path.iterdir() if d.is_dir() and (d / "chr_names").exists()]
                )

        if not frag_dirs:
            raise ProcessingError(
                f"No valid BPCells fragment directories found in {input_path}"
            )

        self.logger.info(f"Found {len(frag_dirs)} fragment directories")

        # Get chromosome names from first fragment
        chrom_names = list(bpcells.cpp.chr_names_fragments_dir(str(frag_dirs[0])))

        # Detect assembly from chromosome names
        assembly = self._detect_assembly_from_chroms(chrom_names)
        self.logger.info(f"Detected assembly: {assembly}")

        # Get chromosome sizes from gcell ChromSize
        try:
            from gcell import _settings

            annotation_dir = _settings.get_setting("annotation_dir")
        except (ImportError, AttributeError):
            annotation_dir = None

        if annotation_dir is None:
            raise ProcessingError(
                "Annotation directory not configured. Please set gcell annotation_dir setting."
            )

        from gcell.dna.genome import ChromSize

        chrom_size_obj = ChromSize(assembly=assembly, annotation_dir=annotation_dir)
        chrom_sizes = chrom_size_obj.get_dict(chr_included=chrom_names)
        # Filter for pseudobulk - remove patches, chrM, chrY, etc.
        standard_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
        chrom_sizes = {
            k: v
            for k, v in chrom_sizes.items()
            if v is not None and k in standard_chroms
        }
        self.logger.info(f"Got chromosome sizes: {len(chrom_sizes)} chromosomes")

        # Load cell names per fragment
        frag_cell_names = {}
        all_cell_names = []
        for frag_dir in frag_dirs:
            frag_path = str(frag_dir)
            cell_names = list(bpcells.cpp.cell_names_fragments_dir(frag_path))
            frag_cell_names[frag_path] = cell_names
            all_cell_names.extend(cell_names)

        self.logger.info(f"Total cells: {len(all_cell_names)}")

        # Load grouping metadata
        group_assignments = self._load_bpcells_grouping(input_path, all_cell_names)

        if group_assignments is None or len(group_assignments) == 0:
            raise ProcessingError(
                "No grouping assignments found for BPCells fragments. "
                "Ensure metacell_assignments.csv exists, provide --external-metadata, "
                "or ensure AnnData has the groupby column."
            )

        # Get unique groups and filter by minimum cells
        group_to_cells = {}
        for cell_name, group in group_assignments.items():
            if group not in group_to_cells:
                group_to_cells[group] = []
            group_to_cells[group].append(cell_name)

        # Filter groups by minimum cells
        group_to_cells = {
            g: cells
            for g, cells in group_to_cells.items()
            if len(cells) >= self.min_cells_per_group
        }

        if len(group_to_cells) == 0:
            raise ProcessingError("No groups remaining after minimum cells filter")

        unique_groups = sorted(group_to_cells.keys())
        self.logger.info(f"Found {len(unique_groups)} groups after filtering")

        # Build cell groups for BPCells
        cell_ids_per_frag = {}
        group_ids_per_frag = {}

        for frag_path, cell_names in frag_cell_names.items():
            cell_ids_for_frag = []
            group_ids_for_frag = []
            for cell_name in cell_names:
                if cell_name in group_assignments:
                    group = group_assignments[cell_name]
                    if group in group_to_cells:
                        cell_ids_for_frag.append(cell_name)
                        group_ids_for_frag.append(group)

            if cell_ids_for_frag:
                cell_ids_per_frag[frag_path] = cell_ids_for_frag
                group_ids_per_frag[frag_path] = group_ids_for_frag

        frag_paths = list(cell_ids_per_frag.keys())

        self.logger.info(f"Building cell groups for {len(frag_paths)} fragments...")
        cell_groups = build_cell_groups(
            fragments=frag_paths,
            cell_ids=cell_ids_per_frag,
            group_ids=group_ids_per_frag,
            group_order=unique_groups,
            min_library_size=self.min_libsize if hasattr(self, "min_libsize") else 0,
            max_library_size=float("inf"),
        )

        # Remove existing output directory if present
        # Note: Do NOT mkdir here - precalculate_insertion_counts will create it
        # The bpcells C++ code expects the output directory to NOT exist
        output_path = Path(output_path)
        if output_path.exists():
            import shutil

            shutil.rmtree(output_path)

        # Use precalculate_insertion_counts to create celltype_dense
        self.logger.info(f"Computing insertion counts for {len(unique_groups)} groups...")
        precalculate_insertion_counts(
            fragments=frag_paths,
            output_dir=str(output_path),
            cell_groups=cell_groups,
            chrom_sizes=chrom_sizes,
            threads=self.n_jobs if hasattr(self, "n_jobs") else 4,
            group_names=unique_groups,
            preserve_chrom_order=True,
        )

        # Save group metadata
        try:
            from bpcells.experimental import PrecalculatedInsertionMatrix

            matrix = PrecalculatedInsertionMatrix(str(output_path))
            libsizes = {g: int(ls) for g, ls in zip(unique_groups, matrix.library_size)}
        except Exception as e:
            self.logger.warning(f"Could not load library sizes: {e}")
            libsizes = {g: 0 for g in unique_groups}

        # Save group metadata CSV
        group_metadata = pd.DataFrame(
            {
                "group": unique_groups,
                "library_size": [libsizes.get(g, 0) for g in unique_groups],
                "n_cells": [len(group_to_cells.get(g, [])) for g in unique_groups],
            }
        )
        group_metadata_path = output_path / "group_metadata.csv"
        group_metadata.to_csv(group_metadata_path, index=False)
        self.logger.info(f"Saved group metadata to {group_metadata_path}")

        # Save attrs.json for BPCellsIO compatibility
        import json

        attrs = {
            "zarr_type": "celltype_dense",
            "assembly": assembly,
            "celltypes": unique_groups,
            "groupby": self.groupby,
            "n_groups": len(unique_groups),
            "library_sizes": libsizes,
            "completed": True,
        }
        attrs_path = output_path / "attrs.json"
        with attrs_path.open("w") as f:
            json.dump(attrs, f, indent=2)

        runtime = self._end_processing()

        self.logger.info(
            f"Completed BPCells pseudobulk processing. Created {len(unique_groups)} groups."
        )

        return {
            "output_path": str(output_path),
            "n_groups": len(unique_groups),
            "groupby": self.groupby,
            "library_sizes": libsizes,
            "runtime": runtime,
        }
