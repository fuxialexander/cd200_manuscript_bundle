"""TSS accessibility processor for extracting accessibility profiles around TSS."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from caesar.preprocess.core.base import AnnotationProcessor
from caesar.preprocess.core.exceptions import ValidationError

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig, TSSAccessibilityConfig


class TSSAccessibilityProcessor(AnnotationProcessor):
    """TSS accessibility processor with direct configuration access.

    This processor extracts and aggregates accessibility signals around
    Transcription Start Sites (TSS). Supports two input types:

    1. **AnnData tile matrix (.h5ad)** - Fast tile-based extraction using
       PyRanges overlap. Recommended for most use cases.

    2. **Dense zarr store (.zarr)** - High-resolution extraction using
       get_gene_tss_track for base-pair resolution data.

    The extraction method is automatically determined based on input type.
    """

    @classmethod
    def get_command_name(cls) -> str:
        """Get the CLI command name for this processor."""
        return "tss-accessibility"

    @classmethod
    def get_description(cls) -> str:
        """Get the description for this processor."""
        return "Extract TSS accessibility profiles from tile count matrices"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add TSS accessibility-specific CLI arguments."""
        super().add_cli_args(parser)

        # Input - can be either h5ad (tile matrix) or zarr (dense store)
        parser.add_argument(
            "-i",
            "--input",
            required=False,
            help="Input file: either AnnData tile matrix (.h5ad) for fast extraction, "
            "or celltype dense Zarr store for high-resolution extraction",
        )

        # TSS parameters
        parser.add_argument(
            "--upstream",
            type=int,
            default=2000,
            help="Base pairs upstream of TSS (default: 2000)",
        )
        parser.add_argument(
            "--downstream",
            type=int,
            default=2000,
            help="Base pairs downstream of TSS (default: 2000)",
        )

        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing TSS accessibility data",
        )
        parser.add_argument(
            "--no-aggregate",
            action="store_true",
            help="Store per-gene profiles instead of aggregated (deprecated, use --per-site)",
        )
        parser.add_argument(
            "--gene-list",
            type=str,
            help="Path to file with gene list (one gene per line)",
        )
        parser.add_argument(
            "--gene-set",
            type=str,
            default="protein_coding",
            help="Predefined gene set: 'protein_coding', 'all' (default: protein_coding)",
        )
        parser.add_argument(
            "--assembly",
            type=str,
            help="Genome assembly (hg38, mm10, etc.) - auto-detected if not specified",
        )
        parser.add_argument(
            "--gencode-version",
            type=str,
            help="Gencode version (e.g., v44) - uses latest if not specified",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="tss_accessibility",
            help="Name of output dataset (default: tss_accessibility)",
        )
        parser.add_argument(
            "--tile-size",
            type=int,
            default=500,
            help="Size of tiles in base pairs (default: 500, for tile-based mode)",
        )
        # Output path
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            help="Output path for saving results (.h5ad for tile-based, .zarr for high-res)",
        )
        # New options for enhanced output
        parser.add_argument(
            "--groupby",
            type=str,
            help="Column in adata.obs to aggregate by (e.g., 'celltype', 'cluster'). "
            "Only for h5ad input. If not specified, returns per-cell accessibility.",
        )
        parser.add_argument(
            "--all-transcripts",
            action="store_true",
            help="Include TSS from all transcripts, not just canonical. "
            "Results in multiple TSS per gene for alternative promoter analysis.",
        )

    def __init__(self, config: CaesarConfig):
        """Initialize TSSAccessibilityProcessor."""
        super().__init__(config)

        self.config = config

        # Direct access to configuration
        self.tss_config: TSSAccessibilityConfig = config.tss_accessibility
        self.upstream = self.tss_config.upstream
        self.downstream = self.tss_config.downstream
        self.n_jobs = self.tss_config.n_jobs
        self.overwrite = self.tss_config.overwrite
        self.aggregate = self.tss_config.aggregate
        self.gene_list_file = self.tss_config.gene_list_file
        self.gene_set = self.tss_config.gene_set
        self.assembly = self.tss_config.assembly or config.assembly
        self.gencode_version = self.tss_config.gencode_version
        self.dataset_name = self.tss_config.dataset_name
        self.tile_size = self.tss_config.tile_size
        # New options
        self.groupby = getattr(self.tss_config, 'groupby', None)
        self.all_transcripts = getattr(self.tss_config, 'all_transcripts', False)

        self.validate_config()

    def validate_config(self) -> None:
        """Validate TSS accessibility configuration."""
        if self.upstream < 0:
            raise ValidationError("upstream must be non-negative")
        if self.downstream < 0:
            raise ValidationError("downstream must be non-negative")
        if self.upstream + self.downstream <= 0:
            raise ValidationError("Total window size must be positive")
        if self.n_jobs < 1:
            raise ValidationError("n_jobs must be at least 1")
        if self.gene_list_file:
            gene_list_path = Path(self.gene_list_file)
            if not gene_list_path.exists():
                raise ValidationError(f"Gene list file not found: {gene_list_path}")

    def _detect_input_type(self, input_path: Path) -> str:
        """Detect input type: h5ad (tile matrix), zarr (dense store), or bpcells.

        Returns:
            'h5ad' for AnnData tile matrix, 'zarr' for dense zarr store,
            'bpcells' for BPCells celltype dense store
        """
        if input_path.suffix == ".h5ad" or str(input_path).endswith(".h5ad"):
            return "h5ad"

        # Check if it's a zarr store
        if input_path.suffix == ".zarr" or (input_path / ".zgroup").exists():
            return "zarr"

        # Check if it's a BPCells store (has attrs.json but no .zgroup)
        attrs_path = input_path / "attrs.json"
        if attrs_path.exists() and not (input_path / ".zgroup").exists():
            return "bpcells"

        # Try to detect zarr by structure
        from caesar.io.zarr_io.base import ZarrIO

        try:
            zarr_type = ZarrIO.detect_store_type(input_path)
            if zarr_type in ["celltype_dense", "dense", "signal_dense"]:
                return "zarr"
        except Exception:
            pass

        raise ValueError(
            f"Cannot determine input type for {input_path}. "
            "Expected .h5ad (AnnData tile matrix), .zarr (dense store), "
            "or BPCells directory"
        )

    def _load_gencode(self, assembly: str | None = None):
        """Load Gencode annotation for the assembly."""
        from gcell.rna.gencode import Gencode

        if assembly is None:
            assembly = self.assembly

        if assembly == "auto" or assembly is None:
            raise ValidationError(
                "Could not determine assembly. Please specify --assembly"
            )

        self.logger.info(f"Loading Gencode for {assembly}...")
        # Gencode class handles default versions and M-prefix for mouse assemblies
        gencode = Gencode(assembly=assembly, version=self.gencode_version)
        return gencode

    def _get_gene_list(self, gencode) -> list[str]:
        """Get list of genes to process."""
        if self.gene_list_file:
            with open(self.gene_list_file) as f:
                gene_list = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(gene_list)} genes from {self.gene_list_file}")
            return gene_list

        if self.gene_set == "protein_coding":
            # Filter genes by type using gene_to_type dict
            gene_list = [
                gene for gene, gtype in gencode.gene_to_type.items()
                if gtype == "protein_coding"
            ]
            self.logger.info(f"Using {len(gene_list)} protein-coding genes")
        elif self.gene_set == "all":
            gene_list = list(gencode.gene_to_type.keys())
            self.logger.info(f"Using all {len(gene_list)} genes")
        else:
            # Default to protein_coding for unrecognized gene sets
            gene_list = [
                gene for gene, gtype in gencode.gene_to_type.items()
                if gtype == "protein_coding"
            ]
            self.logger.info(
                f"Gene set '{self.gene_set}' not recognized, using {len(gene_list)} protein-coding genes"
            )

        return gene_list

    def _process_h5ad_input(self, input_path: Path) -> dict[str, Any]:
        """Process tile-based extraction from h5ad input.

        Returns AnnData with:
        - obs: cells or celltypes (if groupby)
        - var: TSS sites with annotation (site_id, gene_name, transcript_id,
               chromosome, position, strand)
        - X: accessibility counts (obs x var)
        - varm['mean_profile']: per-site mean profile across observations
        - uns: metadata
        """
        import scanpy as sc

        from caesar.io.mixins.tss_tes_accessibility import (
            compute_accessibility_from_tiles,
        )

        self.logger.info(f"Loading tile AnnData from {input_path}")
        adata = sc.read_h5ad(input_path)
        self.logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} tiles")

        # Try to get assembly from adata
        assembly = self.assembly
        if assembly == "auto" or assembly is None:
            if "assembly" in adata.uns:
                assembly = adata.uns["assembly"]
                self.logger.info(f"Auto-detected assembly from adata: {assembly}")

        gencode = self._load_gencode(assembly)
        gene_list = self._get_gene_list(gencode)

        self.logger.info(
            f"Extracting TSS accessibility (tile-based) for {len(gene_list)} genes..."
        )
        if self.groupby:
            self.logger.info(f"Aggregating by '{self.groupby}'")
        if self.all_transcripts:
            self.logger.info("Including all transcript TSS (not just canonical)")

        # Use new function that returns AnnData
        result_adata = compute_accessibility_from_tiles(
            adata, gencode, gene_list, "tss",
            self.upstream, self.downstream, self.tile_size,
            groupby=self.groupby,
            include_all_transcripts=self.all_transcripts,
        )

        if result_adata is None:
            return {"status": "error", "reason": "no_valid_profiles"}

        # Determine output path
        output_path = self.get_output_path()
        if output_path is None:
            # Save alongside input as h5ad
            output_path = input_path.parent / f"{input_path.stem}_tss_accessibility.h5ad"
        else:
            output_path = Path(output_path)
            # Ensure it has .h5ad extension for tile-based output
            if output_path.suffix not in ['.h5ad', '.zarr']:
                output_path = output_path.with_suffix('.h5ad')

        # Check if output exists
        if output_path.exists() and not self.overwrite:
            self.logger.info(f"Output '{output_path}' exists. Skipping.")
            return {"status": "skipped", "reason": "output_exists"}

        self.logger.info(f"Saving results to {output_path}")
        result_adata.write_h5ad(output_path)

        n_sites = result_adata.n_vars
        n_genes = len(set(result_adata.var['gene_name']))
        n_obs = result_adata.n_obs

        return {
            "status": "success",
            "n_sites": n_sites,
            "n_genes": n_genes,
            "n_obs": n_obs,
            "dataset_name": self.dataset_name,
            "shape": result_adata.X.shape,
            "output_path": str(output_path),
            "method": "tile_based",
            "groupby": self.groupby,
            "all_transcripts": self.all_transcripts,
        }

    def _process_zarr_input(self, input_path: Path) -> dict[str, Any]:
        """Process high-resolution extraction from zarr input."""
        from caesar.io.zarr_io import open_zarr

        io = open_zarr(input_path, mode="a")

        # Get assembly from zarr
        assembly = self.assembly
        if assembly == "auto" or assembly is None:
            assembly = io.assembly
            self.logger.info(f"Auto-detected assembly: {assembly}")

        gencode = self._load_gencode(assembly)
        gene_list = self._get_gene_list(gencode)

        self.logger.info(
            f"Extracting TSS accessibility (high-resolution) for {len(gene_list)} genes..."
        )

        if not hasattr(io, "write_tss_accessibility"):
            raise AttributeError(
                f"{io.__class__.__name__} does not have TSSTESAccessibilityMixin."
            )

        results = io.write_tss_accessibility(
            gene_list=gene_list,
            gencode=gencode,
            upstream=self.upstream,
            downstream=self.downstream,
            n_jobs=self.n_jobs,
            overwrite=self.overwrite,
            dataset_name=self.dataset_name,
            aggregate=self.aggregate,
            adata=None,  # High-resolution mode
        )

        results["output_path"] = str(input_path)
        results["method"] = "high_resolution"
        return results

    def _process_bpcells_input(self, input_path: Path) -> dict[str, Any]:
        """Process tile-based extraction from BPCells celltype dense input.

        Uses the fast tile-based extraction method by creating a tile AnnData
        from the BPCells store. This avoids the memory-intensive high-resolution
        mode that queries base-pair resolution data per gene.
        """
        from caesar.io.mixins.tss_tes_accessibility import (
            compute_accessibility_from_tiles,
        )

        try:
            from caesar.io.bpcell_io import CelltypeDenseBPCellsIO
        except ImportError:
            raise ImportError(
                "BPCells support requires the bpcells package. "
                "Install with: pip install bpcells"
            ) from None

        io = CelltypeDenseBPCellsIO(str(input_path), mode="a")

        # Get assembly from BPCells store
        assembly = self.assembly
        if assembly == "auto" or assembly is None:
            assembly = io.assembly
            self.logger.info(f"Auto-detected assembly: {assembly}")

        gencode = self._load_gencode(assembly)
        gene_list = self._get_gene_list(gencode)

        self.logger.info(
            f"Extracting TSS accessibility (BPCells tile-based) for {len(gene_list)} genes..."
        )

        # Create tile AnnData from BPCells for fast tile-based extraction
        self.logger.info("Creating tile AnnData from BPCells store...")
        try:
            tile_adata = io.create_tile_adata(tile_size=self.tile_size)
        except Exception as e:
            self.logger.warning(
                f"Failed to create tile AnnData from BPCells: {e}. "
                "Falling back to high-resolution mode."
            )
            return self._process_bpcells_high_resolution(io, gencode, gene_list, input_path)

        self.logger.info(
            f"Created tile AnnData: {tile_adata.n_obs} celltypes × {tile_adata.n_vars} tiles"
        )

        # Use the tile-based extraction function
        result_adata = compute_accessibility_from_tiles(
            tile_adata, gencode, gene_list, "tss",
            self.upstream, self.downstream, self.tile_size,
            groupby=None,  # Already grouped by celltype in BPCells
            include_all_transcripts=self.all_transcripts,
        )

        if result_adata is None:
            return {"status": "error", "reason": "no_valid_profiles"}

        # Determine output path
        output_path = self.get_output_path()
        if output_path is None:
            output_path = input_path.parent / f"{input_path.name}_tss_accessibility.h5ad"
        else:
            output_path = Path(output_path)
            if output_path.suffix not in ['.h5ad', '.zarr']:
                output_path = output_path.with_suffix('.h5ad')

        # Check if output exists
        if output_path.exists() and not self.overwrite:
            self.logger.info(f"Output '{output_path}' exists. Skipping.")
            return {"status": "skipped", "reason": "output_exists"}

        self.logger.info(f"Saving results to {output_path}")
        result_adata.write_h5ad(output_path)

        n_sites = result_adata.n_vars
        n_genes = len(set(result_adata.var['gene_name']))
        n_obs = result_adata.n_obs

        return {
            "status": "success",
            "n_sites": n_sites,
            "n_genes": n_genes,
            "n_obs": n_obs,
            "dataset_name": self.dataset_name,
            "shape": result_adata.X.shape,
            "output_path": str(output_path),
            "method": "bpcells_tile_based",
            "all_transcripts": self.all_transcripts,
        }

    def _process_bpcells_high_resolution(
        self, io, gencode, gene_list: list[str], input_path: Path
    ) -> dict[str, Any]:
        """Fallback to high-resolution extraction from BPCells.

        This is the legacy method that queries base-pair resolution data per gene.
        It may be slow and memory-intensive for large gene lists.
        """
        self.logger.info(
            f"Extracting TSS accessibility (BPCells high-resolution) for {len(gene_list)} genes..."
        )

        if not hasattr(io, "write_tss_accessibility"):
            raise AttributeError(
                f"{io.__class__.__name__} does not have TSSTESAccessibilityMixin."
            )

        results = io.write_tss_accessibility(
            gene_list=gene_list,
            gencode=gencode,
            upstream=self.upstream,
            downstream=self.downstream,
            n_jobs=self.n_jobs,
            overwrite=self.overwrite,
            dataset_name=self.dataset_name,
            aggregate=self.aggregate,
            adata=None,  # High-resolution mode
        )

        results["output_path"] = str(input_path)
        results["method"] = "bpcells_high_resolution"
        return results

    def process(self) -> dict[str, Any]:
        """Process TSS accessibility extraction."""
        self._start_processing()

        skipped = self._check_enabled("tss_accessibility")
        if skipped:
            return skipped

        input_path = self.validate_input_path(self.get_target_path())
        input_type = self._detect_input_type(input_path)

        self.logger.info(f"Input type: {input_type}")

        if input_type == "h5ad":
            results = self._process_h5ad_input(input_path)
        elif input_type == "bpcells":
            results = self._process_bpcells_input(input_path)
        else:
            results = self._process_zarr_input(input_path)

        runtime = self._end_processing()

        output = {
            "status": results.get("status", "unknown"),
            "runtime": runtime,
            "input_path": str(input_path),
            "output_path": results.get("output_path", str(input_path)),
            "n_sites": results.get("n_sites", results.get("n_genes", 0)),
            "n_genes": results.get("n_genes", 0),
            "n_obs": results.get("n_obs", 0),
            "dataset_name": results.get("dataset_name", self.dataset_name),
            "upstream": self.upstream,
            "downstream": self.downstream,
            "aggregate": self.aggregate,
            "method": results.get("method", "unknown"),
            "groupby": results.get("groupby"),
            "all_transcripts": results.get("all_transcripts", False),
        }

        if results.get("status") == "success":
            method = results.get("method", "unknown")
            n_sites = results.get("n_sites", results.get("n_genes", 0))
            n_genes = results.get("n_genes", n_sites)
            n_obs = results.get("n_obs", 0)
            groupby = results.get("groupby")
            obs_label = "groups" if groupby else "cells"
            self.logger.info(
                f"TSS accessibility extraction complete ({method}): "
                f"{n_sites} sites from {n_genes} genes, {n_obs} {obs_label}"
            )
        elif results.get("status") == "skipped":
            self.logger.info(
                f"TSS accessibility extraction skipped: {results.get('reason', 'unknown')}"
            )
        else:
            self.logger.warning(
                f"TSS accessibility extraction failed: {results.get('reason', 'unknown')}"
            )

        return output
