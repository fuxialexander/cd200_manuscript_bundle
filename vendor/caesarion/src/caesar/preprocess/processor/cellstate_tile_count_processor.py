"""Cell-state tile count processor for generating full tile count matrix.

This processor creates a complete tile count matrix aggregated at the cell-state level
(cluster, metacell, or any custom grouping), immediately after cell-state assignment
and BEFORE variable-tile shrinking.

The output is a compact h5ad where:
- .obs = cell-state metadata (cluster/metacell IDs, library sizes, etc.)
- .var = all genomic tiles (Chromosome, Start, End)
- .X = tile counts per cell-state (dense or sparse)

This is required for:
- GET 1.0 pipeline (peak count matrix extraction)
- Peak × motif matrix generation
- Downstream QC and coverage analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from tqdm import tqdm

from caesar.io.aggregator import ByKeyAggregator
from caesar.io.query import MultiKeyQuery
from caesar.preprocess.core.base import TransformationProcessor
from caesar.preprocess.core.exceptions import ProcessingError, ValidationError

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig, CellStateTileCountConfig


class CellStateTileCountProcessor(TransformationProcessor):
    """Processor for generating full tile count matrix at cell-state level.

    This processor aggregates single-cell tile counts by cell-state grouping
    (clusters, metacells, or custom groups) to create a complete tile count matrix.

    Key features:
    - Works with both clustering-based groupings (labels) and metacell assignments
    - Preserves ALL tiles (not just variable tiles)
    - Output is suitable for GET 1.0 pipeline and peak/motif analysis
    - Supports both sparse and dense output formats
    - Compatible with both Zarr and BPCells backends
    """

    @classmethod
    def get_command_name(cls) -> str:
        """Get the CLI command name for this processor."""
        return "cellstate-tile-count"

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return (
            "Generate full tile count matrix aggregated at cell-state level. "
            "Creates h5ad with all genomic tiles before variable-tile shrinking."
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add processor-specific CLI arguments."""
        super().add_cli_args(parser)

        parser.add_argument(
            "-i", "--input", required=True, help="Input sparse zarr path"
        )
        parser.add_argument(
            "-o",
            "--output",
            default="cellstate_tile_counts.h5ad",
            help="Output file path (default: cellstate_tile_counts.h5ad)",
        )
        parser.add_argument(
            "--groupby",
            default="labels",
            help="Grouping variable (default: labels)",
        )
        parser.add_argument(
            "--tile-size",
            type=int,
            default=500,
            help="Tile size in base pairs (default: 500)",
        )
        parser.add_argument(
            "--min-cells",
            type=int,
            default=1,
            help="Minimum cells per group (default: 1)",
        )
        parser.add_argument(
            "--min-libsize",
            type=int,
            default=0,
            help="Minimum library size per group (default: 0)",
        )
        parser.add_argument(
            "--external-metadata",
            help="External metadata file for custom groupings (CSV/TSV)",
        )
        parser.add_argument(
            "--barcode-column",
            default="barcode",
            help="Column name for barcodes in external metadata (default: barcode)",
        )
        parser.add_argument(
            "--no-blacklist",
            action="store_true",
            help="Skip blacklist filtering",
        )
        parser.add_argument(
            "--dense",
            action="store_true",
            help="Save as dense matrix instead of sparse",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="Apply CPM normalization",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing output file",
        )

    def __init__(self, config: CaesarConfig):
        """Initialize CellStateTileCountProcessor."""
        super().__init__(config)

        self.config = config
        self.cellstate_config: CellStateTileCountConfig = config.cellstate_tile_count

        # Direct access to configuration
        self.groupby = self.cellstate_config.groupby
        self.tile_size = self.cellstate_config.tile_size
        self.min_cells_per_group = self.cellstate_config.min_cells_per_group
        self.min_libsize = self.cellstate_config.min_libsize
        self.use_stored_tiles = self.cellstate_config.use_stored_tiles
        self.apply_blacklist = self.cellstate_config.apply_blacklist
        self.sparse_output = self.cellstate_config.sparse_output
        self.normalize = self.cellstate_config.normalize
        self.overwrite = self.cellstate_config.overwrite
        self.output_format = self.cellstate_config.output_format

        # Metadata handling
        self.external_metadata = self.cellstate_config.external_metadata
        self.barcode_column = self.cellstate_config.barcode_column

        self.validate_config()

    def validate_config(self) -> None:
        """Validate processor configuration."""
        if self.tile_size <= 0:
            raise ValidationError("tile_size must be greater than 0")
        if self.min_cells_per_group < 0:
            raise ValidationError("min_cells_per_group must be non-negative")
        if self.min_libsize < 0:
            raise ValidationError("min_libsize must be non-negative")

        # Validate external metadata if specified
        if self.external_metadata:
            metadata_path = Path(self.external_metadata)
            if not metadata_path.exists():
                raise ValidationError(
                    f"External metadata file not found: {metadata_path}"
                )

    def _detect_store_type(self, input_path: Path) -> str:
        """Detect the type of sparse store (Zarr or BPCells).

        Returns:
            Store type: 'single_sparse', 'multi_sparse', 'multiome_sparse',
                       'bpcells_single_sparse', 'bpcells_multi_sparse',
                       'bpcells_fragments', 'bpcells_fragments_multi'
        """
        # First try zarr detection
        from caesar.io.zarr_io.base import ZarrIO

        store_type = ZarrIO.detect_store_type(input_path)

        if store_type in ["multiome_sparse", "single_sparse", "multi_sparse"]:
            self.logger.info(f"Detected {store_type} zarr at {input_path}")
            return store_type

        # Try BPCells detection
        try:
            from caesar.io.bpcell_io import BPCellsIO

            bpcells_type = BPCellsIO.detect_store_type(input_path)
            if bpcells_type in ["bpcells_single_sparse", "bpcells_multi_sparse"]:
                self.logger.info(f"Detected {bpcells_type} at {input_path}")
                return bpcells_type
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
                    self.logger.info(f"Detected multi-sample BPCells fragments at {input_path}")
                    return "bpcells_fragments_multi"

        raise ValueError(
            f"Cannot determine store type for {input_path}. "
            "CellStateTileCountProcessor requires sparse zarr, BPCells stores, or BPCells fragments."
        )

    def _load_io(self, input_path: Path, store_type: str):
        """Load the appropriate IO class based on store type.

        Args:
            input_path: Path to the sparse store
            store_type: Type of store ('single_sparse', 'multi_sparse', 'multiome_sparse',
                       'bpcells_single_sparse', 'bpcells_multi_sparse')

        Returns:
            IO instance (SingleSparseZarrIO, MultiSparseZarrIO, MultiomeSparseZarrIO,
                        SingleSparseBPCellsIO, or MultiSparseBPCellsIO)
        """
        if store_type == "multiome_sparse":
            from caesar.io.zarr_io.multiome import MultiomeSparseZarrIO
            return MultiomeSparseZarrIO(str(input_path), mode="r")

        elif store_type in ["single_sparse", "multi_sparse"]:
            from caesar.io.zarr_io import open_zarr
            return open_zarr(str(input_path), mode="r")

        elif store_type == "bpcells_single_sparse":
            from caesar.io.bpcell_io import SingleSparseBPCellsIO
            return SingleSparseBPCellsIO(str(input_path), mode="r")

        elif store_type == "bpcells_multi_sparse":
            from caesar.io.bpcell_io import MultiSparseBPCellsIO
            return MultiSparseBPCellsIO(str(input_path), mode="r")

        else:
            raise ValueError(f"Unsupported store type: {store_type}")

    def _auto_discover_metacell_assignments(self, z) -> str | None:
        """Auto-discover metacell assignments CSV when groupby='metacell_id'.

        Args:
            z: SingleSparseZarrIO or MultiSparseZarrIO instance

        Returns:
            Path to metacell_assignments.csv if found, None otherwise
        """
        if self.groupby != "metacell_id":
            return None

        if self.external_metadata:
            return None

        # Get zarr path
        if hasattr(z, "path"):
            zarr_path = Path(z.path)
        else:
            self.logger.warning("Cannot determine zarr path for metacell auto-discovery")
            return None

        # Look for metacell_assignments.csv
        metacell_csv = zarr_path / "metacell_assignments.csv"
        if metacell_csv.exists():
            self.logger.info(f"Auto-discovered metacell assignments: {metacell_csv}")
            return str(metacell_csv)

        # Also check parent directory
        parent_csv = zarr_path.parent / "metacell_assignments.csv"
        if parent_csv.exists():
            self.logger.info(f"Auto-discovered metacell assignments: {parent_csv}")
            return str(parent_csv)

        self.logger.warning(
            f"groupby='metacell_id' but metacell_assignments.csv not found at {metacell_csv}"
        )
        return None

    def _load_external_metadata(self, zarr_barcodes: np.ndarray) -> pd.DataFrame | None:
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

        if self.groupby not in df.columns:
            raise ValidationError(
                f"Groupby column '{self.groupby}' not found in metadata file. "
                f"Available columns: {list(df.columns)}"
            )

        # Filter to only barcodes present in zarr
        zarr_barcode_set = set(zarr_barcodes)
        df_filtered = df[df[self.barcode_column].isin(zarr_barcode_set)].copy()

        self.logger.info(
            f"Loaded external metadata: {len(df_filtered)}/{len(zarr_barcodes)} barcodes matched"
        )

        # Rename barcode column to 'barcode' for consistency
        if self.barcode_column != "barcode":
            df_filtered = df_filtered.rename(columns={self.barcode_column: "barcode"})

        return df_filtered

    def _get_tiles(self, z) -> pd.DataFrame:
        """Get genomic tiles from store or create fresh ones.

        Args:
            z: Sparse IO instance (SingleSparseZarrIO, MultiSparseZarrIO,
               SingleSparseBPCellsIO, or MultiSparseBPCellsIO)

        Returns:
            DataFrame with Chromosome, Start, End columns
        """
        from caesar.preprocess.core.tile_filtering import filter_tiles_by_blacklist

        if self.use_stored_tiles:
            # Try to get stored tiles
            stored_tiles = z.get_peaks("tiles")
            if stored_tiles is not None:
                # Handle MultiSparseIO case
                if isinstance(stored_tiles, dict):
                    tiles = list(stored_tiles.values())[0].copy()
                else:
                    tiles = stored_tiles.copy()

                self.logger.info(f"Loaded {len(tiles)} stored tiles")
            else:
                # Create tiles from scratch
                from caesar.preprocess.core.tile_filtering import (
                    filter_tiles_by_chromosomes,
                )
                from gcell.dna.genome import ChromSize

                chrom_sizes = ChromSize(z.assembly)
                tiles = chrom_sizes.tiling_region(self.tile_size)
                tiles = filter_tiles_by_chromosomes(tiles)
                self.logger.info(f"Created {len(tiles)} tiles from scratch")
        else:
            # Create tiles from scratch
            from caesar.preprocess.core.tile_filtering import (
                filter_tiles_by_chromosomes,
            )
            from gcell.dna.genome import ChromSize

            chrom_sizes = ChromSize(z.assembly)
            tiles = chrom_sizes.tiling_region(self.tile_size)
            tiles = filter_tiles_by_chromosomes(tiles)
            self.logger.info(f"Created {len(tiles)} fresh tiles")

        # Apply blacklist filtering if requested
        # Use the global blacklist_file from config (or None for assembly-specific)
        # This ensures consistency with ClusteringMixin.create_genome_tiles()
        if self.apply_blacklist:
            tiles_before = len(tiles)
            # Use blacklist_file from global config for consistency
            blacklist = self.config.blacklist_file if self.config.blacklist_file else None
            tiles = filter_tiles_by_blacklist(tiles, blacklist=blacklist, assembly=z.assembly)
            if len(tiles) < tiles_before:
                self.logger.info(
                    f"Applied blacklist filtering: {tiles_before} -> {len(tiles)} tiles"
                )

        # Ensure indices are continuous
        tiles = tiles.reset_index(drop=True)
        return tiles

    def _prepare_grouping(self, z) -> tuple[pd.DataFrame, dict[str, list[int]]]:
        """Prepare cell grouping information.

        Args:
            z: Sparse IO instance (Zarr or BPCells)

        Returns:
            Tuple of (merged_metadata, group_to_cell_indices)
        """
        # Auto-discover metacell assignments if needed
        auto_metadata = self._auto_discover_metacell_assignments(z)
        if auto_metadata:
            self.external_metadata = auto_metadata
            self.barcode_column = "barcode"

        # Get base metadata
        z_metadata = z.metadata.copy()
        z_metadata["barcode"] = z_metadata.index

        if self.external_metadata:
            # Load and merge external metadata
            zarr_barcodes = (
                z.barcodes if hasattr(z, "barcodes") else z.metadata.index.values
            )
            external_df = self._load_external_metadata(zarr_barcodes)
            if external_df is None:
                raise ProcessingError("Failed to load external metadata")

            # Merge external metadata
            merged_metadata = z_metadata.merge(
                external_df[["barcode", self.groupby]],
                on="barcode",
                how="left",
                suffixes=("_orig", ""),
            )
            merged_metadata = merged_metadata.set_index("barcode")
            group_column = self.groupby

        else:
            # Use default labels from zarr
            merged_metadata = z_metadata
            merged_metadata = merged_metadata.set_index("barcode")
            group_column = "labels" if self.groupby == "labels" else self.groupby

            if group_column not in merged_metadata.columns:
                raise ValidationError(
                    f"Groupby column '{group_column}' not found in zarr metadata. "
                    f"Available columns: {list(merged_metadata.columns)}"
                )

        # Build group to cell indices mapping
        group_to_cells = {}
        for group_name in merged_metadata[group_column].unique():
            if pd.isna(group_name) or group_name == "unassigned":
                continue

            group_mask = merged_metadata[group_column] == group_name
            cell_indices = merged_metadata.loc[group_mask, "cell_idx"].tolist()

            # Apply minimum cells filter
            if len(cell_indices) < self.min_cells_per_group:
                self.logger.debug(
                    f"Skipping group {group_name}: {len(cell_indices)} cells < {self.min_cells_per_group}"
                )
                continue

            group_to_cells[group_name] = cell_indices

        self.logger.info(f"Found {len(group_to_cells)} valid groups")
        return merged_metadata, group_to_cells

    def _compute_tile_counts_for_group(
        self, z, group_name: str, cell_indices: list[int], tiles: pd.DataFrame
    ) -> tuple[str, np.ndarray, int, int]:
        """Compute tile counts for a single group.

        Args:
            z: Sparse IO instance (Zarr or BPCells)
            group_name: Name of the group
            cell_indices: List of cell indices in this group
            tiles: DataFrame with tile coordinates

        Returns:
            Tuple of (group_name, tile_counts, n_cells, total_counts)
        """
        # Get count matrix for all tiles using existing method
        # This efficiently computes counts for the specified tiles
        from caesar.io.query import MetadataQuery

        query = MetadataQuery(f"cell_idx in {cell_indices}")

        # Get tile counts by summing coverage within each tile region
        n_tiles = len(tiles)
        tile_counts = np.zeros(n_tiles, dtype=np.int32)

        # Group tiles by chromosome for efficient processing
        for chrom in tiles["Chromosome"].unique():
            chrom_tiles = tiles[tiles["Chromosome"] == chrom]
            chrom_indices = chrom_tiles.index.tolist()

            for idx, (_, row) in zip(chrom_indices, chrom_tiles.iterrows()):
                start = int(row["Start"])
                end = int(row["End"])

                # Get coverage for this region
                try:
                    coverage = z.get_track(
                        chr_name=chrom,
                        start=start,
                        end=end,
                        query=query,
                    )
                    tile_counts[idx] = int(coverage.sum())
                except Exception as e:
                    self.logger.debug(f"Error getting coverage for {chrom}:{start}-{end}: {e}")
                    tile_counts[idx] = 0

        total_counts = int(tile_counts.sum())
        return group_name, tile_counts, len(cell_indices), total_counts

    def _compute_tile_counts_batch(
        self, z, group_to_cells: dict[str, list[int]], tiles: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Compute tile counts for all groups using batch processing.

        This method processes tiles chromosome by chromosome, aggregating
        counts for all groups simultaneously for efficiency.

        Args:
            z: Sparse IO instance (Zarr or BPCells)
            group_to_cells: Mapping of group names to cell indices
            tiles: DataFrame with tile coordinates

        Returns:
            Tuple of (group_tile_counts, group_metadata)
        """
        n_tiles = len(tiles)
        n_groups = len(group_to_cells)
        group_names = list(group_to_cells.keys())

        # Initialize tile count matrix (groups x tiles)
        # Using lil_matrix for efficient construction, convert to csr later
        tile_matrix = lil_matrix((n_groups, n_tiles), dtype=np.int32)

        # Track metadata for each group
        group_metadata = {
            name: {"n_cells": len(cells), "total_counts": 0}
            for name, cells in group_to_cells.items()
        }

        # Create MultiKeyQuery for efficient batch processing
        queries = {name: cells for name, cells in group_to_cells.items()}
        multi_query = MultiKeyQuery(queries)

        # Process chromosome by chromosome
        chroms = tiles["Chromosome"].unique()
        self.logger.info(f"Processing {len(chroms)} chromosomes...")

        for chrom in tqdm(chroms, desc="Chromosomes"):
            chrom_tiles = tiles[tiles["Chromosome"] == chrom]
            if len(chrom_tiles) == 0:
                continue

            chrom_start = int(chrom_tiles["Start"].min())
            chrom_end = int(chrom_tiles["End"].max())

            try:
                # Get coverage for entire chromosome range for all groups
                tracks = z.get_track(
                    chr_name=chrom,
                    start=chrom_start,
                    end=chrom_end,
                    query=multi_query,
                    aggregator=ByKeyAggregator(),
                    output_format="raw_dict",
                )

                # Aggregate into tiles
                for group_idx, group_name in enumerate(group_names):
                    if group_name not in tracks:
                        continue

                    signal = tracks[group_name]

                    # Sum signal within each tile
                    for tile_idx, (_, row) in zip(
                        chrom_tiles.index.tolist(), chrom_tiles.iterrows()
                    ):
                        tile_start = int(row["Start"]) - chrom_start
                        tile_end = int(row["End"]) - chrom_start

                        # Bounds check
                        tile_start = max(0, tile_start)
                        tile_end = min(len(signal), tile_end)

                        if tile_end > tile_start:
                            count = int(signal[tile_start:tile_end].sum())
                            if count > 0:
                                tile_matrix[group_idx, tile_idx] = count
                                group_metadata[group_name]["total_counts"] += count

            except Exception as e:
                self.logger.warning(f"Error processing {chrom}: {e}")
                continue

        # Convert to dict of arrays for return
        tile_counts = {}
        for group_idx, group_name in enumerate(group_names):
            tile_counts[group_name] = np.array(tile_matrix[group_idx, :].toarray()).flatten()

        return tile_counts, group_metadata

    def _filter_groups_by_libsize(
        self,
        tile_counts: dict[str, np.ndarray],
        group_metadata: dict[str, dict],
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Filter groups by minimum library size.

        Args:
            tile_counts: Dict mapping group names to tile count arrays
            group_metadata: Dict mapping group names to metadata dicts

        Returns:
            Filtered (tile_counts, group_metadata)
        """
        if self.min_libsize <= 0:
            return tile_counts, group_metadata

        filtered_counts = {}
        filtered_metadata = {}

        for group_name, counts in tile_counts.items():
            total_counts = group_metadata[group_name]["total_counts"]
            if total_counts >= self.min_libsize:
                filtered_counts[group_name] = counts
                filtered_metadata[group_name] = group_metadata[group_name]
            else:
                self.logger.debug(
                    f"Filtering out group {group_name}: {total_counts} counts < {self.min_libsize}"
                )

        n_filtered = len(tile_counts) - len(filtered_counts)
        if n_filtered > 0:
            self.logger.info(f"Filtered out {n_filtered} groups due to min_libsize")

        return filtered_counts, filtered_metadata

    def _create_anndata(
        self,
        tile_counts: dict[str, np.ndarray],
        group_metadata: dict[str, dict],
        tiles: pd.DataFrame,
        assembly: str,
    ):
        """Create AnnData object from tile counts.

        Args:
            tile_counts: Dict mapping group names to tile count arrays
            group_metadata: Dict mapping group names to metadata dicts
            tiles: DataFrame with tile coordinates
            assembly: Genome assembly

        Returns:
            AnnData object
        """
        import anndata as ad
        from scipy.sparse import csr_matrix

        group_names = list(tile_counts.keys())
        n_groups = len(group_names)
        n_tiles = len(tiles)

        # Build count matrix
        if self.sparse_output:
            # Build as lil_matrix then convert to csr
            X = lil_matrix((n_groups, n_tiles), dtype=np.float32)
            for i, name in enumerate(group_names):
                counts = tile_counts[name]
                nonzero = counts > 0
                X[i, nonzero] = counts[nonzero]
            X = csr_matrix(X)
        else:
            X = np.zeros((n_groups, n_tiles), dtype=np.float32)
            for i, name in enumerate(group_names):
                X[i, :] = tile_counts[name]

        # Apply normalization if requested
        if self.normalize:
            row_sums = np.array(X.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            if self.sparse_output:
                # Normalize sparse matrix
                from scipy.sparse import diags
                norm_factors = 1e6 / row_sums
                X = diags(norm_factors) @ X
            else:
                X = X / row_sums[:, np.newaxis] * 1e6

        # Create obs DataFrame
        obs_data = {
            "group_id": group_names,
            "n_cells": [group_metadata[n]["n_cells"] for n in group_names],
            "total_counts": [group_metadata[n]["total_counts"] for n in group_names],
        }
        obs = pd.DataFrame(obs_data)
        obs.index = group_names
        obs.index.name = "group_id"

        # Create var DataFrame
        var = tiles.copy()
        var.index = [f"tile_{i}" for i in range(len(tiles))]
        var.index.name = "tile_id"

        # Create AnnData
        adata = ad.AnnData(X=X, obs=obs, var=var)

        # Add metadata
        adata.uns["assembly"] = assembly
        adata.uns["tile_size"] = self.tile_size
        adata.uns["groupby"] = self.groupby
        adata.uns["n_groups"] = n_groups
        adata.uns["n_tiles"] = n_tiles
        adata.uns["normalized"] = self.normalize
        adata.uns["blacklist_filtered"] = self.apply_blacklist

        return adata

    def _process_bpcells_fragments(
        self, input_path: Path, store_type: str, output_path: Path
    ) -> dict[str, Any]:
        """Process BPCells fragments to generate cell-state tile counts.

        This method uses the same mechanism as BPCellsProcessor._process_anndata(),
        utilizing precalculate_insertion_counts_binned from bpcells.experimental.fragments
        with proper cell group building and tile filtering.

        Args:
            input_path: Path to BPCells fragments directory
            store_type: 'bpcells_fragments' or 'bpcells_fragments_multi'
            output_path: Path to output file

        Returns:
            Processing results dictionary
        """
        import json
        import shutil
        import tempfile

        import bpcells.cpp
        from bpcells.experimental.fragments import (
            build_cell_groups,
            precalculate_insertion_counts_binned,
        )
        from bpcells.experimental.matrix import DirMatrix

        self.logger.info("Processing BPCells fragments for cell-state tile counts")

        # Discover fragment directories
        if store_type == "bpcells_fragments_multi":
            frag_dirs = sorted([
                d for d in input_path.iterdir()
                if d.is_dir() and (d / "version").exists()
            ])
        else:
            # Single fragment directory - might be wrapped in a parent dir
            if (input_path / "chr_names").exists():
                frag_dirs = [input_path]
            else:
                # Check subdirectories
                frag_dirs = sorted([
                    d for d in input_path.iterdir()
                    if d.is_dir() and (d / "chr_names").exists()
                ])

        if not frag_dirs:
            raise ProcessingError(f"No valid BPCells fragment directories found in {input_path}")

        self.logger.info(f"Found {len(frag_dirs)} fragment directories")

        # Get chromosome names from first fragment
        chrom_names = list(bpcells.cpp.chr_names_fragments_dir(str(frag_dirs[0])))

        # Detect assembly from chromosome names
        assembly = self._detect_assembly_from_chroms(chrom_names)
        self.logger.info(f"Detected assembly: {assembly}")

        # Get chromosome sizes from gcell ChromSize (same approach as BPCellsProcessor)
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
        chrom_sizes = {k: v for k, v in chrom_sizes.items() if v is not None}
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

        # Load grouping metadata (from metacell assignments or anndata)
        group_assignments = self._load_bpcells_grouping(input_path, all_cell_names)

        if group_assignments is None or len(group_assignments) == 0:
            raise ProcessingError(
                "No grouping assignments found for BPCells fragments. "
                "Ensure metacell_assignments.csv exists or provide --external-metadata"
            )

        # Get unique groups and filter by minimum cells
        group_to_cells = {}
        for cell_name, group in group_assignments.items():
            if group not in group_to_cells:
                group_to_cells[group] = []
            group_to_cells[group].append(cell_name)

        # Filter groups by minimum cells
        group_to_cells = {
            g: cells for g, cells in group_to_cells.items()
            if len(cells) >= self.min_cells_per_group
        }

        if len(group_to_cells) == 0:
            raise ProcessingError("No groups remaining after minimum cells filter")

        unique_groups = sorted(group_to_cells.keys())
        self.logger.info(f"Found {len(unique_groups)} groups after filtering")

        # Create temporary directory for binned matrices
        temp_dir = Path(tempfile.mkdtemp(prefix="bpcells_cellstate_"))

        try:
            # Build cell groups for BPCells - map cells to their group IDs
            # For each fragment, create mapping from cell_id to group_id
            cell_ids_per_frag = {}
            group_ids_per_frag = {}

            for frag_path, cell_names in frag_cell_names.items():
                cell_ids_for_frag = []
                group_ids_for_frag = []
                for cell_name in cell_names:
                    if cell_name in group_assignments:
                        group = group_assignments[cell_name]
                        if group in group_to_cells:  # Only include cells from valid groups
                            cell_ids_for_frag.append(cell_name)
                            group_ids_for_frag.append(group)

                if cell_ids_for_frag:
                    cell_ids_per_frag[frag_path] = cell_ids_for_frag
                    group_ids_per_frag[frag_path] = group_ids_for_frag

            # Build cell groups for all fragments together
            frag_paths = list(cell_ids_per_frag.keys())

            self.logger.info(f"Building cell groups for {len(frag_paths)} fragments...")
            cell_groups = build_cell_groups(
                fragments=frag_paths,
                cell_ids=cell_ids_per_frag,
                group_ids=group_ids_per_frag,
                group_order=unique_groups,
                min_library_size=0,  # Don't filter by library size here, do it later
                max_library_size=float("inf"),  # No upper bound
            )

            # Output path for binned matrix
            binned_output = temp_dir / "binned_groups"

            # Use precalculate_insertion_counts_binned with cell groups
            # This creates a matrix where rows are groups (not cells)
            self.logger.info(f"Computing binned insertion counts with tile_size={self.tile_size}bp...")
            precalculate_insertion_counts_binned(
                fragments=frag_paths,
                output_dir=str(binned_output),
                cell_groups=cell_groups,
                chrom_sizes=chrom_sizes,
                bin_size=self.tile_size,
                threads=1,
                group_names=None,  # Auto-extract from cell_groups
                preserve_chrom_order=True,
            )

            # Load the result matrix
            combined_mat = DirMatrix(str(binned_output))
            self.logger.info(f"Combined matrix shape: {combined_mat.shape}")
            n_groups_mat, n_bins = combined_mat.shape

            # Build tile annotations with proper filtering
            # Use filter_tiles for consistent blacklist handling
            from caesar.preprocess.core.tile_filtering import (
                create_column_indices,
                filter_tiles,
            )

            # Create tiles using ChromSize (same as BPCellsProcessor._process_anndata)
            # Note: assembly, annotation_dir, and chrom_size_obj were already set up earlier
            all_tiles_df = chrom_size_obj.tiling_region(
                tile_size=self.tile_size, tile_overlap=0
            )
            # Filter to only chromosomes we have in the matrix
            all_tiles_df = all_tiles_df[
                all_tiles_df["Chromosome"].isin(chrom_sizes.keys())
            ].copy()
            all_tiles_df = all_tiles_df.reset_index(drop=True)

            # Apply blacklist filtering if requested
            if self.apply_blacklist:
                blacklist = (
                    self.config.blacklist_file if self.config.blacklist_file else None
                )
                tiles_df = filter_tiles(
                    all_tiles_df.copy(),
                    blacklist=blacklist,
                    assembly=assembly,
                )
                tiles_df = tiles_df[
                    tiles_df["Chromosome"].isin(chrom_sizes.keys())
                ].copy()
                tiles_df = tiles_df.reset_index(drop=True)

                # Create column indices for filtering
                column_indices = create_column_indices(all_tiles_df, tiles_df)
                n_kept = len(column_indices)
                n_removed = len(all_tiles_df) - n_kept
                self.logger.info(
                    f"Tile filtering: {len(all_tiles_df)} -> {n_kept} tiles "
                    f"({n_removed} removed by blacklist/chromosome filter)"
                )
            else:
                tiles_df = all_tiles_df
                column_indices = None

            # Load group names from the binned output
            group_names_path = binned_output / "group_names.json"
            if group_names_path.exists():
                with group_names_path.open() as f:
                    group_names = json.load(f)
            else:
                group_names = unique_groups

            # Compute group metadata and tile counts
            tile_counts = {}
            group_metadata = {}

            # Load full matrix (groups x bins)
            self.logger.info("Loading group tile counts...")
            chunk_size = DirMatrix.calculate_safe_chunk_size(n_groups_mat)

            try:
                chunk_matrices = combined_mat.load_chunked_columns(
                    start_col=0,
                    end_col=n_bins,
                    chunk_size=chunk_size,
                    format="csr",
                )

                import scipy.sparse
                full_matrix = scipy.sparse.hstack(chunk_matrices, format="csr")
                self.logger.info(f"Loaded matrix: shape {full_matrix.shape}")

                # Apply column filtering if needed
                if column_indices is not None and len(column_indices) < n_bins:
                    full_matrix = full_matrix[:, column_indices]
                    self.logger.info(
                        f"Filtered matrix: shape {full_matrix.shape}"
                    )

            except Exception as e:
                self.logger.error(f"Failed to load matrix: {e}")
                raise ProcessingError(f"Failed to load binned matrix: {e}") from e

            # Extract counts per group
            for group_idx, group_name in enumerate(group_names):
                if group_idx >= full_matrix.shape[0]:
                    break

                group_counts = np.array(full_matrix[group_idx, :].toarray()).flatten()
                total_counts = int(group_counts.sum())

                # Get number of cells in this group
                n_cells = len(group_to_cells.get(group_name, []))

                # Apply library size filter
                if self.min_libsize > 0 and total_counts < self.min_libsize:
                    self.logger.debug(
                        f"Skipping group {group_name}: {total_counts} counts < {self.min_libsize}"
                    )
                    continue

                tile_counts[group_name] = group_counts
                group_metadata[group_name] = {
                    "n_cells": n_cells,
                    "total_counts": total_counts,
                }

            if len(tile_counts) == 0:
                raise ProcessingError("No groups remaining after library size filter")

            self.logger.info(f"Final groups: {len(tile_counts)}")

            # Create AnnData
            adata = self._create_anndata(tile_counts, group_metadata, tiles_df, assembly)

            # Save output
            self.logger.info(f"Saving to {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.output_format == "h5ad":
                adata.write_h5ad(output_path)
            else:
                adata.write_zarr(output_path)

            self.logger.info(
                f"Created cell-state tile count matrix: {adata.n_obs} groups x {adata.n_vars} tiles"
            )

        finally:
            # Cleanup temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        runtime = self._end_processing()

        return {
            "status": "success",
            "output_path": str(output_path),
            "n_groups": len(tile_counts),
            "n_tiles": len(tiles_df),
            "groupby": self.groupby,
            "normalized": self.normalize,
            "sparse": self.sparse_output,
            "runtime": runtime,
        }

    def _load_bpcells_grouping(
        self, input_path: Path, cell_names: list[str]
    ) -> dict[str, str] | None:
        """Load grouping assignments for BPCells fragments.

        Searches for metacell_assignments.csv or external metadata.

        Args:
            input_path: Path to BPCells fragments directory
            cell_names: List of cell names

        Returns:
            Dict mapping cell name to group, or None if not found
        """
        # Check for external metadata
        if self.external_metadata:
            metadata_path = Path(self.external_metadata)
            if metadata_path.exists():
                if metadata_path.suffix == ".csv":
                    df = pd.read_csv(metadata_path)
                else:
                    df = pd.read_csv(metadata_path, sep="\t")

                if self.barcode_column not in df.columns:
                    raise ValidationError(
                        f"Barcode column '{self.barcode_column}' not found in metadata"
                    )
                if self.groupby not in df.columns:
                    raise ValidationError(
                        f"Groupby column '{self.groupby}' not found in metadata"
                    )

                return dict(zip(df[self.barcode_column], df[self.groupby]))

        # Auto-discover metacell assignments
        candidates = [
            input_path / "metacell_assignments.csv",
            input_path.parent / "metacell_assignments.csv",
            input_path.parent / "anndata" / "metadata.csv",
        ]

        # For metacell_id grouping
        if self.groupby == "metacell_id":
            for cand in candidates:
                if cand.exists():
                    self.logger.info(f"Found metacell assignments at {cand}")
                    df = pd.read_csv(cand)
                    if "barcode" in df.columns and "metacell_id" in df.columns:
                        return dict(zip(df["barcode"], df["metacell_id"]))

        # For labels grouping, check anndata metadata
        if self.groupby == "labels":
            anndata_path = input_path.parent / "anndata" / "anndata.h5ad"
            if anndata_path.exists():
                import scanpy as sc
                adata = sc.read_h5ad(anndata_path)
                if "labels" in adata.obs.columns:
                    return dict(zip(adata.obs_names, adata.obs["labels"]))

            # Check metadata.csv
            metadata_path = input_path.parent / "anndata" / "metadata.csv"
            if metadata_path.exists():
                df = pd.read_csv(metadata_path, index_col=0)
                if "labels" in df.columns:
                    return dict(zip(df.index, df["labels"]))

        return None

    def _build_tile_annotations(
        self, chrom_sizes: dict[str, int], tile_size: int
    ) -> pd.DataFrame:
        """Build tile annotations DataFrame from chromosome sizes.

        Args:
            chrom_sizes: Dict of chromosome name to length
            tile_size: Size of each tile in bp

        Returns:
            DataFrame with Chromosome, Start, End columns
        """
        tiles = []
        for chrom, length in chrom_sizes.items():
            # Skip non-standard chromosomes if apply_blacklist
            if self.apply_blacklist and (chrom.startswith("chrUn") or "_" in chrom):
                continue

            for start in range(0, length, tile_size):
                end = min(start + tile_size, length)
                tiles.append({
                    "Chromosome": chrom,
                    "Start": start,
                    "End": end,
                })

        return pd.DataFrame(tiles)

    def _detect_assembly_from_chroms(self, chrom_names: list[str]) -> str:
        """Detect genome assembly from chromosome names.

        Args:
            chrom_names: List of chromosome names

        Returns:
            Assembly name (hg38, mm10, etc.)
        """
        # Simple heuristic based on chromosome naming
        has_ucsc_style = any("chr" in c for c in chrom_names)
        has_chrM = any(c == "chrM" for c in chrom_names)

        if has_ucsc_style and has_chrM:
            # Could be human or mouse
            # Check for human-specific or mouse-specific chromosomes
            if any(c in ["chr22", "chr21", "chr20"] for c in chrom_names):
                return "hg38"
            if any(c == "chr19" for c in chrom_names) and not any(c == "chr22" for c in chrom_names):
                return "mm10"
        return "hg38"  # Default

    def process(self) -> dict[str, Any]:
        """Process cell-state tile count generation.

        Returns:
            Processing results dictionary
        """
        self._start_processing()

        # Check if enabled
        if not self.cellstate_config.enabled:
            self.logger.info("Cell-state tile count processor is disabled")
            runtime = self._end_processing()
            return {
                "status": "skipped",
                "reason": "disabled",
                "runtime": runtime,
            }

        # Get and validate input path
        input_path = self.validate_input_path()
        self.logger.info(f"Processing cell-state tile counts from {input_path}")

        # Detect store type
        store_type = self._detect_store_type(input_path)

        # Get output path
        output_path = self.get_output_path(self.cellstate_config.output)

        # Check if output exists
        if output_path.exists() and not self.overwrite:
            self.logger.info(f"Output already exists: {output_path}")
            runtime = self._end_processing()
            return {
                "status": "skipped",
                "reason": "output_exists",
                "output_path": str(output_path),
                "runtime": runtime,
            }

        # Handle BPCells fragments separately (different data structure)
        if store_type in ["bpcells_fragments", "bpcells_fragments_multi"]:
            return self._process_bpcells_fragments(input_path, store_type, output_path)

        # Load appropriate IO for sparse stores
        z = self._load_io(input_path, store_type)

        try:
            # Get tiles
            tiles = self._get_tiles(z)
            self.logger.info(f"Using {len(tiles)} tiles")

            # Prepare grouping
            merged_metadata, group_to_cells = self._prepare_grouping(z)
            self.logger.info(f"Processing {len(group_to_cells)} groups")

            # Compute tile counts for all groups
            tile_counts, group_metadata = self._compute_tile_counts_batch(
                z, group_to_cells, tiles
            )

            # Filter by library size
            tile_counts, group_metadata = self._filter_groups_by_libsize(
                tile_counts, group_metadata
            )

            if len(tile_counts) == 0:
                raise ProcessingError("No groups remaining after filtering")

            # Get assembly from zarr
            assembly = z.assembly if hasattr(z, "assembly") else self.assembly
            if assembly == "auto":
                assembly = z._root.attrs.get("assembly", "hg38")

            # Create AnnData
            adata = self._create_anndata(tile_counts, group_metadata, tiles, assembly)

            # Save output
            self.logger.info(f"Saving to {output_path}")
            if self.output_format == "h5ad":
                adata.write_h5ad(output_path)
            else:
                # Zarr output
                adata.write_zarr(output_path)

            self.logger.info(
                f"Created cell-state tile count matrix: {adata.n_obs} groups x {adata.n_vars} tiles"
            )

        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            raise ProcessingError(f"Cell-state tile count processing failed: {e}") from e

        runtime = self._end_processing()

        return {
            "status": "success",
            "output_path": str(output_path),
            "n_groups": len(tile_counts),
            "n_tiles": len(tiles),
            "groupby": self.groupby,
            "normalized": self.normalize,
            "sparse": self.sparse_output,
            "runtime": runtime,
        }
