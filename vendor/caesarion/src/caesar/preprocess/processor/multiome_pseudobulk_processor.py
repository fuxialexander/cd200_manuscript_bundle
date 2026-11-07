"""Multiome pseudobulk processor for aggregating paired ATAC + RNA data.

This processor creates pseudobulked multiome data from single-cell multiome,
aggregating both ATAC and RNA by cell type/cluster while maintaining pairing.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec
from tqdm import tqdm

from caesar.io.aggregator import ByKeyAggregator
from caesar.io.query import MetadataQuery, MultiKeyQuery
from caesar.io.zarr_io import MultiomeSparseZarrIO, open_zarr
from caesar.preprocess.core.base import TransformationProcessor
from caesar.preprocess.core.exceptions import ProcessingError, ValidationError

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig


class MultiomePseudobulkProcessor(TransformationProcessor):
    """Create pseudobulked multiome data from single-cell multiome.

    This processor aggregates both ATAC coverage and RNA expression by cell type
    or cluster while maintaining the pairing between modalities. The output is
    stored in MultiomeDenseZarrIO format, enabling regulatory analysis at the
    cell type level.

    The processor:
    1. Loads MultiomeSparseZarr input with paired ATAC+RNA data
    2. Groups cells by labels/clusters (like PseudobulkProcessor)
    3. Aggregates ATAC coverage into dense tracks per cell type
    4. Aggregates RNA expression (sum/mean/median) per cell type
    5. Stores both modalities in unified MultiomeDenseZarr structure
    """

    @classmethod
    def get_command_name(cls) -> str:
        """Get the command name for CLI registration."""
        return "multiome-pseudobulk"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add multiome pseudobulk-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Multiome pseudobulk-specific arguments
        parser.add_argument(
            "-i", "--input", required=True, help="Input multiome sparse zarr path"
        )
        parser.add_argument(
            "-o",
            "--output",
            default="multiome_pseudobulk.zarr",
            help="Output multiome dense zarr path (default: multiome_pseudobulk.zarr)",
        )
        parser.add_argument(
            "--groupby",
            default="labels",
            help="Grouping variable (default: labels)",
        )
        parser.add_argument(
            "--min-cells",
            type=int,
            default=100,
            help="Minimum cells per group (default: 100)",
        )
        parser.add_argument(
            "--expression-aggregation",
            choices=["sum", "mean", "median"],
            default="sum",
            help="RNA expression aggregation method (default: sum)",
        )
        parser.add_argument(
            "--min-cells-for-expression",
            type=int,
            default=10,
            help="Minimum cells required for expression aggregation (default: 10)",
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return (
            "Create pseudobulked multiome data from single-cell multiome. "
            "Aggregates both ATAC and RNA by cell type while maintaining pairing."
        )

    def __init__(self, config: CaesarConfig):
        """Initialize MultiomePseudobulkProcessor.

        Args:
            config: Structured CaesarConfig instance
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store full config for direct access
        self.config = config

        # Get multiome pseudobulk configuration
        # First check if there's a specific multiome_pseudobulk config
        self.multiome_pseudobulk_config = getattr(config, "multiome_pseudobulk", None)

        # If not, use pseudobulk config as base
        if not self.multiome_pseudobulk_config:
            # Use standard pseudobulk config as fallback
            pseudobulk_config = getattr(config, "pseudobulk", None)
            if pseudobulk_config:
                self.groupby = pseudobulk_config.groupby
                self.min_cells_per_group = pseudobulk_config.min_cells_per_group
                self.min_libsize = pseudobulk_config.min_libsize
            else:
                # Defaults
                self.groupby = "labels"
                self.min_cells_per_group = 100
                self.min_libsize = 3_000_000
        else:
            # Use multiome-specific config
            self.groupby = self.multiome_pseudobulk_config.groupby
            self.min_cells_per_group = (
                self.multiome_pseudobulk_config.min_cells_per_group
            )
            self.min_libsize = getattr(
                self.multiome_pseudobulk_config, "min_libsize", 3_000_000
            )

        # RNA-specific aggregation settings
        self.expression_aggregation = getattr(
            self.multiome_pseudobulk_config, "expression_aggregation", "sum"
        )
        self.min_cells_for_expression = getattr(
            self.multiome_pseudobulk_config, "min_cells_for_expression", 10
        )

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate multiome pseudobulk configuration."""
        if self.min_cells_per_group < 1:
            raise ValidationError("min_cells_per_group must be at least 1")

        if self.min_libsize < 0:
            raise ValidationError("min_libsize must be non-negative")

        if self.min_cells_for_expression < 1:
            raise ValidationError("min_cells_for_expression must be at least 1")

        valid_aggregations = ["sum", "mean", "median"]
        if self.expression_aggregation not in valid_aggregations:
            raise ValidationError(
                f"Invalid expression_aggregation: {self.expression_aggregation}. "
                f"Must be one of {valid_aggregations}"
            )

    def process(self) -> dict[str, Any]:
        """Process multiome pseudobulk aggregation.

        Returns:
            Processing results including output path and statistics
        """
        self._start_processing()

        # Get and validate input path
        input_path = self.validate_input_path()
        self.logger.info(f"Processing multiome pseudobulk aggregation on {input_path}")

        # Open as MultiomeSparseZarr
        try:
            z = open_zarr(str(input_path), mode="r")
            if not isinstance(z, MultiomeSparseZarrIO):
                raise ValidationError(
                    f"Input must be MultiomeSparseZarr, got {type(z).__name__}"
                )
        except Exception as e:
            raise ValidationError(f"Failed to open input as MultiomeSparseZarr: {e}")

        # Get output path using base class method
        output_path = self.get_output_path("multiome_pseudobulk.zarr")

        # Initialize root group
        root = zarr.open_group(str(output_path), mode="a")

        # Process the multiome pseudobulk aggregation
        try:
            results = self._process_multiome_pseudobulk(z, root)

            # Store global metadata
            actual_assembly = z.assembly if hasattr(z, "assembly") else self.assembly
            if (
                actual_assembly == "auto"
                and hasattr(z, "_root")
                and "assembly" in z._root.attrs
            ):
                actual_assembly = z._root.attrs["assembly"]

            root.attrs.update(
                {
                    "n_groups": results["n_groups"],
                    "group_ids": sorted(results["group_names"]),
                    "atac_library_size": sum(results["atac_libsizes"].values()),
                    "total_rna_counts": results["total_rna_counts"],
                    "assembly": actual_assembly,
                    "groupby": self.groupby,
                    "expression_aggregation": self.expression_aggregation,
                    "completed": True,
                    "zarr_type": "multiome_dense",
                    "celltypes": sorted(results["group_names"]),
                }
            )

            self.logger.info(
                f"Completed multiome pseudobulk processing. "
                f"Created {results['n_groups']} groups with paired ATAC+RNA data"
            )

        except Exception as e:
            self.logger.error(f"Error during multiome pseudobulk processing: {e}")
            raise ProcessingError(f"Multiome pseudobulk processing failed: {e}") from e

        runtime = self._end_processing()

        return {
            "output_path": str(output_path),
            "n_groups": results["n_groups"],
            "groupby": self.groupby,
            "atac_library_sizes": results["atac_libsizes"],
            "total_rna_counts": results["total_rna_counts"],
            "runtime": runtime,
        }

    def _process_multiome_pseudobulk(
        self, z: MultiomeSparseZarrIO, root: zarr.Group
    ) -> dict[str, Any]:
        """Process multiome pseudobulk aggregation.

        Args:
            z: MultiomeSparseZarrIO instance
            root: Zarr root group for output

        Returns:
            Dictionary with processing results
        """
        # Get metadata for grouping - need to filter to shared cells only
        full_metadata = z.metadata
        shared_barcodes = z.shared_barcodes

        # Filter metadata to only include shared cells
        metadata = full_metadata[full_metadata.index.isin(shared_barcodes)].copy()

        # Update the cell_idx to match the shared cell indices
        metadata["cell_idx"] = z.shared_indices_atac

        group_column = self.groupby if self.groupby in metadata.columns else "labels"

        # Get unique groups
        labels = metadata[group_column].unique()
        valid_groups = [
            label for label in labels if label != "unassigned" and pd.notna(label)
        ]

        # Filter groups by minimum cell count
        group_counts = metadata[group_column].value_counts()
        filtered_groups = [
            group
            for group in valid_groups
            if group_counts.get(group, 0) >= self.min_cells_per_group
        ]

        self.logger.info(
            f"Found {len(filtered_groups)} groups after filtering "
            f"(min cells: {self.min_cells_per_group})"
        )

        # Initialize storage structures
        atac_libsizes = {}
        rna_stats = {}

        # Process ATAC data first (similar to PseudobulkProcessor)
        self.logger.info("Processing ATAC coverage aggregation...")
        atac_libsizes = self._aggregate_atac_coverage(
            z, root, filtered_groups, metadata
        )

        # Filter groups by minimum ATAC library size
        final_groups = []
        for group in filtered_groups:
            if group in atac_libsizes and atac_libsizes[group] >= self.min_libsize:
                final_groups.append(group)
            else:
                self.logger.info(
                    f"Skipping group {group} with ATAC library size "
                    f"{atac_libsizes.get(group, 0)} < {self.min_libsize}"
                )
                # Remove the group from root if it was created
                if group in root:
                    del root[group]

        self.logger.info(f"After ATAC filtering: {len(final_groups)} groups remaining")

        # Process RNA expression aggregation
        self.logger.info("Processing RNA expression aggregation...")
        rna_stats = self._aggregate_rna_expression(z, root, final_groups, metadata)

        # Store metadata for each group
        for group_name in final_groups:
            if group_name in root:
                root[group_name].attrs.update(
                    {
                        "library_size": atac_libsizes[group_name],
                        "celltype": group_name,
                        "assembly": z.assembly,
                        "n_cells": int(group_counts[group_name]),
                        "completed": True,
                    }
                )

                # Add RNA stats if available
                if group_name in rna_stats:
                    root[group_name].attrs.update(
                        {
                            "total_rna_counts": rna_stats[group_name]["total_counts"],
                            "n_expressed_genes": rna_stats[group_name]["n_genes"],
                            "rna_aggregation_method": self.expression_aggregation,
                        }
                    )

        return {
            "n_groups": len(final_groups),
            "group_names": final_groups,
            "atac_libsizes": {g: atac_libsizes[g] for g in final_groups},
            "total_rna_counts": sum(
                rna_stats.get(g, {}).get("total_counts", 0) for g in final_groups
            ),
        }

    def _aggregate_atac_coverage(
        self,
        z: MultiomeSparseZarrIO,
        root: zarr.Group,
        groups: list[str],
        metadata: pd.DataFrame,
    ) -> dict[str, int]:
        """Aggregate ATAC coverage by groups.

        Args:
            z: MultiomeSparseZarrIO instance
            root: Zarr root group for output
            groups: List of group names to process
            metadata: Cell metadata DataFrame

        Returns:
            Dictionary mapping group names to ATAC library sizes
        """
        # Create queries for each group
        # Since metadata is already filtered to shared cells and has updated cell_idx,
        # we can use MetadataQuery directly
        queries = {}
        for group in groups:
            group_cells = metadata[metadata[self.groupby] == group]
            if len(group_cells) >= self.min_cells_per_group:
                queries[group] = MetadataQuery(f"{self.groupby} == '{group}'")

        cluster_queries = MultiKeyQuery(queries)

        # Initialize arrays for each group and chromosome
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
                            self._process_atac_chunk,
                            z=z,
                            root=root,
                            cluster_queries=cluster_queries,
                            chrom=chrom,
                            start=start,
                            end=end,
                        )
                    )

            # Collect results
            libsizes = {group_name: 0 for group_name in cluster_queries.mapping}
            failed_chunks = []

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing ATAC chunks"
            ):
                try:
                    chunk_libsizes = future.result()
                    for group_name, libsize in chunk_libsizes.items():
                        libsizes[group_name] += libsize
                except Exception as e:
                    self.logger.error(f"Failed to process ATAC chunk: {e}")
                    failed_chunks.append(len(failed_chunks))

            if failed_chunks:
                raise RuntimeError(
                    f"Failed to process {len(failed_chunks)} ATAC chunks"
                )

        return libsizes

    def _process_atac_chunk(
        self, z, root, cluster_queries, chrom, start, end
    ) -> dict[str, int]:
        """Process a single ATAC chunk for all groups.

        Args:
            z: MultiomeSparseZarrIO instance
            root: Zarr root group for output
            cluster_queries: MultiKeyQuery object with group names as keys
            chrom: Chromosome name
            start: Start position
            end: End position

        Returns:
            Dictionary mapping group names to their chunk library sizes
        """
        try:
            chunk_libsizes = {}

            # Get tracks for all groups
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
                f"Failed to process ATAC chunk at {chrom}:{start}-{end}: {e}"
            )
            raise

    def _aggregate_rna_expression(
        self,
        z: MultiomeSparseZarrIO,
        root: zarr.Group,
        groups: list[str],
        metadata: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """Aggregate RNA expression by groups.

        Args:
            z: MultiomeSparseZarrIO instance
            root: Zarr root group for output
            groups: List of group names to process
            metadata: Cell metadata DataFrame

        Returns:
            Dictionary mapping group names to RNA statistics
        """
        # Load RNA expression data
        try:
            # Get RNA data for shared cells only
            adata = z.get_rna_for_shared_cells()
        except Exception as e:
            self.logger.warning(f"Failed to load RNA expression data: {e}")
            return {}

        # The RNA data is already filtered to shared cells
        # Now we need to align the grouping with the metadata

        # Add group labels to adata if not present
        if self.groupby not in adata.obs.columns:
            # Map barcodes to groups from metadata (which is already filtered to shared cells)
            barcode_to_group = dict(zip(metadata.index, metadata[self.groupby]))
            adata.obs[self.groupby] = adata.obs.index.map(barcode_to_group)

        rna_stats = {}

        for group_name in groups:
            # Get cells for this group
            group_mask = adata.obs[self.groupby] == group_name
            group_cells = adata[group_mask]

            n_cells = group_cells.n_obs
            if n_cells < self.min_cells_for_expression:
                self.logger.warning(
                    f"Skipping RNA aggregation for {group_name}: "
                    f"{n_cells} cells < {self.min_cells_for_expression}"
                )
                continue

            # Create expression group
            if group_name not in root:
                root.create_group(group_name)
            if "expression" not in root[group_name]:
                root[group_name].create_group("expression")

            expr_group = root[group_name]["expression"]

            # Get expression matrix
            X = group_cells.X
            if hasattr(X, "toarray"):
                X = X.toarray()

            # Aggregate expression based on method
            if self.expression_aggregation == "sum":
                aggregated = X.sum(axis=0)
            elif self.expression_aggregation == "mean":
                aggregated = X.mean(axis=0)
            elif self.expression_aggregation == "median":
                aggregated = np.median(X, axis=0)
            else:
                raise ValueError(
                    f"Unknown aggregation method: {self.expression_aggregation}"
                )

            # Ensure 1D array
            aggregated = np.asarray(aggregated).flatten()

            # Store raw counts (aggregated values)
            if "counts" not in expr_group:
                counts_arr = expr_group.create_array(
                    "counts",
                    shape=aggregated.shape,
                    dtype="f4",
                    compressors=BloscCodec(cname="zstd", clevel=4),
                )
                counts_arr[:] = aggregated

            # Calculate TPM normalization
            # For aggregated data, TPM = (counts / sum(counts)) * 1e6
            total_counts = aggregated.sum()
            if total_counts > 0:
                tpm = (aggregated / total_counts) * 1e6
            else:
                tpm = np.zeros_like(aggregated)

            if "tpm" not in expr_group:
                tpm_arr = expr_group.create_array(
                    "tpm",
                    shape=tpm.shape,
                    dtype="f4",
                    compressors=BloscCodec(cname="zstd", clevel=4),
                )
                tpm_arr[:] = tpm

            # Store gene metadata
            if "genes" not in expr_group:
                genes_group = expr_group.create_group("genes")

                # Store gene names
                gene_name_data = np.array(group_cells.var.index, dtype="S")
                gene_name_arr = genes_group.create_array(
                    "gene_name",
                    shape=gene_name_data.shape,
                    dtype=gene_name_data.dtype,
                    compressors=BloscCodec(cname="zstd", clevel=4),
                )
                gene_name_arr[:] = gene_name_data

                # Store additional gene info if available
                if "gene_ids" in group_cells.var.columns:
                    gene_id_data = np.array(group_cells.var["gene_ids"], dtype="S")
                    gene_id_arr = genes_group.create_array(
                        "gene_id",
                        shape=gene_id_data.shape,
                        dtype=gene_id_data.dtype,
                        compressors=BloscCodec(cname="zstd", clevel=4),
                    )
                    gene_id_arr[:] = gene_id_data

            # Calculate statistics
            n_expressed = (aggregated > 0).sum()
            rna_stats[group_name] = {
                "total_counts": float(total_counts),
                "n_genes": int(n_expressed),
                "n_cells": int(n_cells),
            }

            self.logger.info(
                f"Aggregated RNA for {group_name}: "
                f"{n_cells} cells, {n_expressed} expressed genes, "
                f"{total_counts:.0f} total counts"
            )

        return rna_stats
