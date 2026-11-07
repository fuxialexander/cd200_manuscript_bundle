"""MultiomeRNAProcessor for handling RNA-seq data in multiome experiments.

This processor handles different RNA input formats (h5ad, 10x_h5, 10x_mtx) for multiome
experiments, with proper QC, normalization, and HVG detection while preserving full gene
sets for pseudobulk analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata
import pandas as pd
import scanpy as sc

from caesar.preprocess.core.base import PreprocessingProcessor
from caesar.preprocess.core.exceptions import ProcessingError, ValidationError

if TYPE_CHECKING:
    from caesar.preprocess.core.config import CaesarConfig

logger = logging.getLogger(__name__)


class MultiomeRNAProcessor(PreprocessingProcessor):
    """Process RNA-seq data for multiome analysis into standardized AnnData format.

    Handles multiple input formats:
    - h5ad: Pre-processed AnnData files
    - 10x_h5: 10x Genomics HDF5 format
    - 10x_mtx: 10x Genomics Matrix Market format
    - auto: Auto-detect format
    """

    def __init__(self, config: CaesarConfig):
        """Initialize MultiomeRNAProcessor.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        # RNA-specific configuration from multiome_rna section
        self.rna_config = (
            config.multiome_rna if hasattr(config, "multiome_rna") else None
        )

        # If multiome_rna not present, try legacy multiome.rna config
        if not self.rna_config and hasattr(config, "multiome") and config.multiome:
            legacy_rna = config.multiome.get("rna", {})
            if legacy_rna:
                # Create MultiomeRNAConfig from legacy config
                from omegaconf import OmegaConf

                from caesar.preprocess.core.config import MultiomeRNAConfig

                self.rna_config = OmegaConf.to_object(
                    OmegaConf.merge(OmegaConf.structured(MultiomeRNAConfig), legacy_rna)
                )

        if not self.rna_config:
            # Use default MultiomeRNAConfig
            from caesar.preprocess.core.config import MultiomeRNAConfig

            self.rna_config = MultiomeRNAConfig()

        # Direct access to config attributes
        self.min_genes = self.rna_config.min_genes
        self.min_cells = self.rna_config.min_cells
        self.max_genes = self.rna_config.max_genes
        self.min_counts = self.rna_config.min_counts
        self.max_counts = self.rna_config.max_counts
        self.percent_mito = self.rna_config.percent_mito

        # Processing options
        self.normalization = self.rna_config.normalization
        self.n_top_genes = self.rna_config.n_top_genes
        self.calculate_hvg = self.rna_config.calculate_hvg

        # Format-specific options
        self.validate_only = self.rna_config.validate_only
        self.force_renormalize = self.rna_config.force_renormalize
        self.use_gene_symbols = self.rna_config.use_gene_symbols
        self.var_names = self.rna_config.var_names
        self.cache = self.rna_config.cache

        # Advanced options
        self.mito_prefix = self.rna_config.mito_prefix
        self.detect_doublets = self.rna_config.detect_doublets
        self.score_cell_cycle = self.rna_config.score_cell_cycle
        self.remove_ambient = self.rna_config.remove_ambient
        self.compute_embedding = self.rna_config.compute_embedding

        # Batch correction
        self.batch_key = self.rna_config.batch_key
        self.batch_correction = self.rna_config.batch_correction

        # Output options
        self.save_individual = self.rna_config.save_individual
        self.save_merged = self.rna_config.save_merged

    def validate_config(self) -> None:
        """Validate configuration."""
        # Check RNA data exists
        if not self.config.data.rna:
            raise ValidationError("No RNA data specified in data.rna")

        # Validate each RNA file
        for rna_file in self.config.data.rna:
            if isinstance(rna_file, str):
                path = Path(rna_file)
            elif isinstance(rna_file, dict):
                path = Path(rna_file.get("path", ""))
            else:
                raise ValidationError(f"Invalid RNA file specification: {rna_file}")

            if not path.exists():
                raise ValidationError(f"RNA file not found: {path}")

    def process(self) -> dict[str, Any]:
        """Process RNA data according to configuration.

        Returns:
            Processing results dictionary
        """
        self._start_processing()

        try:
            # Get RNA files
            rna_files = self._prepare_rna_files()

            # Process each sample
            adata_list = []
            for rna_file in rna_files:
                self.logger.info(f"Processing RNA sample: {rna_file['id']}")
                adata = self._process_single_rna(rna_file)

                if self.save_individual:
                    output_path = self.output_dir / f"{rna_file['id']}_rna.h5ad"
                    adata.write_h5ad(output_path)
                    self.logger.info(f"Saved individual result: {output_path}")

                adata_list.append(adata)

            # Merge if multiple samples
            if len(adata_list) > 1 and self.save_merged:
                adata_merged = self._merge_samples(adata_list, rna_files)
                output_path = self.output_dir / "merged_rna.h5ad"
                adata_merged.write_h5ad(output_path)
                self.logger.info(f"Saved merged result: {output_path}")
            elif len(adata_list) == 1:
                # Single sample, save as main output
                output_path = self.output_dir / "rna.h5ad"
                adata_list[0].write_h5ad(output_path)
                self.logger.info(f"Saved RNA data: {output_path}")

            runtime = self._end_processing()

            return {
                "success": True,
                "n_samples": len(rna_files),
                "runtime": runtime,
                "output_dir": str(self.output_dir),
            }

        except Exception as e:
            self.logger.error(f"RNA processing failed: {e}")
            raise ProcessingError(f"RNA processing failed: {e}") from e

    def _prepare_rna_files(self) -> list[dict[str, Any]]:
        """Prepare RNA file list for processing."""
        rna_files = []

        for i, rna_input in enumerate(self.config.data.rna):
            if isinstance(rna_input, str):
                # Simple path string
                rna_file = {
                    "path": rna_input,
                    "id": Path(rna_input).stem,
                    "format": "auto",
                }
            elif isinstance(rna_input, dict):
                # Dictionary with metadata
                rna_file = rna_input.copy()
                if "id" not in rna_file:
                    rna_file["id"] = Path(rna_file["path"]).stem
                if "format" not in rna_file:
                    rna_file["format"] = "auto"
            else:
                raise ValidationError(f"Invalid RNA input format: {type(rna_input)}")

            rna_files.append(rna_file)

        return rna_files

    def _process_single_rna(self, rna_file: dict[str, Any]) -> anndata.AnnData:
        """Process a single RNA sample."""
        path = Path(rna_file["path"])
        format_type = rna_file.get("format", "auto")
        sample_id = rna_file.get("id", path.stem)

        # Auto-detect format if needed
        if format_type == "auto":
            format_type = self._detect_format(path)

        # Load data based on format
        if format_type == "h5ad":
            adata = self._load_h5ad(path)
        elif format_type == "10x_h5":
            adata = self._load_10x_h5(path, rna_file)
        elif format_type == "10x_mtx":
            adata = self._load_10x_mtx(path, rna_file)
        else:
            raise ValueError(f"Unknown RNA format: {format_type}")

        # Add sample metadata
        adata.obs["sample_id"] = sample_id
        if "batch" in rna_file:
            adata.obs["batch"] = rna_file["batch"]

        # Process based on format and config
        if format_type == "h5ad" and self.validate_only:
            # Just validate pre-processed data
            self._validate_h5ad(adata)
        else:
            # Full processing pipeline
            adata = self._full_processing_pipeline(adata)

        return adata

    def _detect_format(self, path: Path) -> str:
        """Auto-detect RNA data format."""
        if path.suffix == ".h5ad":
            return "h5ad"
        elif path.suffix == ".h5":
            return "10x_h5"
        elif path.is_dir():
            # Check for MTX files using glob
            mtx_patterns = ["matrix.mtx", "matrix.mtx.gz"]
            for pattern in mtx_patterns:
                if path.glob(str(pattern)):
                    return "10x_mtx"

        raise ValueError(f"Cannot auto-detect format for: {path}")

    def _load_h5ad(self, path: Path) -> anndata.AnnData:
        """Load pre-processed h5ad file."""
        self.logger.info(f"Loading h5ad file: {path}")
        adata = sc.read_h5ad(path)

        # Ensure required slots exist
        if "n_genes" not in adata.obs.columns:
            adata.obs["n_genes"] = (adata.X > 0).sum(axis=1)
        if "n_counts" not in adata.obs.columns:
            adata.obs["n_counts"] = adata.X.sum(axis=1)

        return adata

    def _load_10x_h5(self, path: Path, rna_file: dict) -> anndata.AnnData:
        """Load 10x Genomics HDF5 format."""
        self.logger.info(f"Loading 10x h5 file: {path}")

        # Load with scanpy
        adata = sc.read_10x_h5(
            path,
            genome=rna_file.get("genome"),
            gex_only=True,  # Only gene expression
        )

        # Make variable names unique
        adata.var_names_make_unique()

        return adata

    def _load_10x_mtx(self, path: Path, rna_file: dict) -> anndata.AnnData:
        """Load 10x Genomics Matrix Market format."""
        self.logger.info(f"Loading 10x mtx directory: {path}")
        # auto detect prefix
        # find the mtx or mtx.gz file
        mtx_files = list(path.glob("*matrix.mtx*"))
        if len(mtx_files) == 0:
            raise ValueError(f"No mtx or mtx.gz file found in {path}")
        if len(mtx_files) > 1:
            raise ValueError(f"Multiple mtx or mtx.gz files found in {path}")
        mtx_file = mtx_files[0]
        # get prefix before the 'matrix'
        prefix = mtx_file.stem.split('matrix')[0]
        # Load with scanpy
        adata = sc.read_10x_mtx(
            path,
            var_names=self.var_names,
            cache=self.cache,
            gex_only=True,
            prefix=prefix,
        )

        # Make variable names unique
        adata.var_names_make_unique()

        return adata

    def _validate_h5ad(self, adata: anndata.AnnData):
        """Validate pre-processed h5ad meets requirements."""
        self.logger.info("Validating h5ad file...")

        # Check required observations
        required_obs = ["n_genes", "n_counts"]
        missing_obs = [col for col in required_obs if col not in adata.obs.columns]
        if missing_obs:
            raise ValueError(f"Missing required obs columns: {missing_obs}")

        # Check if normalization info exists
        if "normalization" not in adata.uns:
            self.logger.warning("No normalization info found in .uns")

        # Validate against QC thresholds
        n_cells_before = adata.n_obs

        # Apply filters (but don't modify if validate_only)
        mask = (
            (adata.obs["n_genes"] >= self.min_genes)
            & (adata.obs["n_counts"] >= self.min_counts)
            & (adata.obs["n_counts"] <= self.max_counts)
        )

        if "percent_mito" in adata.obs:
            mask &= adata.obs["percent_mito"] <= self.percent_mito

        n_cells_pass = mask.sum()
        self.logger.info(
            f"Validation: {n_cells_pass}/{n_cells_before} cells pass QC thresholds"
        )

        if n_cells_pass < self.min_cells:
            raise ValueError(
                f"Too few cells pass QC: {n_cells_pass} < {self.min_cells}"
            )

    def _full_processing_pipeline(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Full processing pipeline for raw data."""
        self.logger.info("Running full RNA processing pipeline...")

        # 1. Calculate QC metrics
        self._calculate_qc_metrics(adata)

        # 2. Apply QC filters
        adata = self._apply_qc_filters(adata)

        # 3. Normalization
        adata = self._normalize_data(adata)

        # 4. Find highly variable genes
        if self.calculate_hvg:
            self._find_variable_genes(adata)

        # 5. Optional: Advanced processing
        if self.detect_doublets:
            self._detect_doublets(adata)

        if self.score_cell_cycle:
            self._score_cell_cycle(adata)

        if self.remove_ambient:
            self._remove_ambient_rna(adata)

        # 6. Optional: Basic dimensionality reduction for QC visualization
        if self.compute_embedding:
            self._compute_basic_embedding(adata)

        # Store processing info
        adata.uns["multiome_rna_processing"] = {
            "normalization": self.normalization,
            "n_top_genes": self.n_top_genes,
            "min_genes": self.min_genes,
            "min_cells": self.min_cells,
            "percent_mito": self.percent_mito,
            "processing_date": str(pd.Timestamp.now()),
        }

        return adata

    def _calculate_qc_metrics(self, adata: anndata.AnnData):
        """Calculate quality control metrics following Scanpy best practices."""
        self.logger.info("Calculating QC metrics...")

        # Mitochondrial genes
        adata.var["mt"] = adata.var_names.str.startswith(self.mito_prefix)

        # Ribosomal genes (optional)
        adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))

        # Hemoglobin genes (optional, useful for detecting RBC contamination)
        adata.var["hb"] = adata.var_names.str.match("^HB[^(P)]")

        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt", "ribo", "hb"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )

        # Rename columns to match our convention
        adata.obs.rename(
            columns={
                "n_genes_by_counts": "n_genes",
                "total_counts": "n_counts",
                "pct_counts_mt": "percent_mito",
                "pct_counts_ribo": "percent_ribo",
                "pct_counts_hb": "percent_hb",
            },
            inplace=True,
        )

        # Store thresholds for reference
        adata.uns["qc_thresholds"] = {
            "min_genes": self.min_genes,
            "min_counts": self.min_counts,
            "max_counts": self.max_counts,
            "percent_mito": self.percent_mito,
        }

    def _apply_qc_filters(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Apply quality control filters following Scanpy best practices."""
        self.logger.info("Applying QC filters...")

        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars

        # Filter cells based on QC metrics
        # Start with permissive filtering as recommended
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        sc.pp.filter_genes(adata, min_cells=self.min_cells)

        # Additional cell filtering based on QC metrics
        # Using boolean indexing for more complex filters
        cell_subset = (
            (adata.obs["n_genes"] <= self.max_genes)
            & (adata.obs["n_counts"] >= self.min_counts)
            & (adata.obs["n_counts"] <= self.max_counts)
            & (adata.obs["percent_mito"] <= self.percent_mito)
        )

        # Apply the filter
        adata = adata[cell_subset, :].copy()

        self.logger.info(f"Filtered cells: {n_cells_before} → {adata.n_obs}")
        self.logger.info(f"Filtered genes: {n_genes_before} → {adata.n_vars}")

        # Store filtering stats
        adata.uns["filter_stats"] = {
            "n_cells_before": n_cells_before,
            "n_cells_after": adata.n_obs,
            "n_genes_before": n_genes_before,
            "n_genes_after": adata.n_vars,
        }

        return adata

    def _normalize_data(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Normalize expression data following Scanpy best practices."""
        self.logger.info(f"Normalizing data with method: {self.normalization}")

        # Store raw counts in a layer for later use
        # This is crucial for many downstream analyses
        adata.layers["counts"] = adata.X.copy()

        if self.normalization == "raw":
            # Keep raw counts - useful for some tools like scVI
            self.logger.info("Keeping raw counts without normalization")

        elif self.normalization in ["cpm", "log1p"]:
            # Standard normalization workflow
            # 1. Normalize every cell to have the same total count
            if self.normalization == "cpm":
                target_sum = 1e6
            else:  # log1p
                target_sum = 1e4

            sc.pp.normalize_total(adata, target_sum=target_sum)

            # 2. Logarithmize the data for log1p
            if self.normalization == "log1p":
                sc.pp.log1p(adata)

        elif self.normalization == "tpm":
            # TPM normalization - requires gene lengths
            if "gene_length" in adata.var.columns:
                # Proper TPM calculation
                adata.X = adata.X / adata.var["gene_length"].values
                sc.pp.normalize_total(adata, target_sum=1e6)
            else:
                self.logger.warning(
                    "TPM normalization requires gene lengths, using CPM instead"
                )
                sc.pp.normalize_total(adata, target_sum=1e6)

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

        # Store normalization parameters
        adata.uns["normalization"] = {
            "method": self.normalization,
            "target_sum": locals().get("target_sum", None),
        }

        # ALWAYS save the full normalized data before HVG selection
        # This is crucial for multiome: HVG is only for clustering,
        # but we need all genes for pseudobulk expression profiles
        adata.raw = adata.copy()
        self.logger.info(
            "Saved full gene set in .raw (required for pseudobulk expression)"
        )

        return adata

    def _find_variable_genes(self, adata: anndata.AnnData):
        """Find highly variable genes following Scanpy best practices.

        Note: This only marks HVGs for clustering. The full gene set is preserved
        in adata.raw for pseudobulk expression analysis.
        """
        self.logger.info(f"Finding top {self.n_top_genes} highly variable genes...")

        # Choose flavor based on normalization and data type
        if self.normalization == "log1p":
            # For log-normalized data
            if "counts" in adata.layers:
                # Use Seurat v3 which works on raw counts
                # This is more accurate as it models count data properly
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=self.n_top_genes,
                    subset=False,
                    flavor="seurat_v3",
                    layer="counts",  # Use raw counts layer
                )
            else:
                # Fallback to standard Seurat method
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=self.n_top_genes,
                    subset=False,
                    flavor="seurat",
                )
        else:
            # For non-normalized or CPM data
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.n_top_genes,
                subset=False,
                flavor="cell_ranger",
            )

        # Print HVG stats
        n_hvg = adata.var["highly_variable"].sum()
        self.logger.info(f"Found {n_hvg} highly variable genes")

    def _detect_doublets(self, adata: anndata.AnnData):
        """Detect potential doublets."""
        self.logger.info("Detecting doublets...")

        try:
            import scrublet

            # Run scrublet
            scrub = scrublet.Scrublet(
                adata.layers.get("counts", adata.X)
            )
            doublet_scores, predicted_doublets = scrub.scrub_doublets()

            adata.obs["doublet_score"] = doublet_scores
            adata.obs["predicted_doublet"] = predicted_doublets

        except ImportError:
            self.logger.warning("scrublet not installed, skipping doublet detection")

    def _score_cell_cycle(self, adata: anndata.AnnData):
        """Score cell cycle phase."""
        self.logger.info("Scoring cell cycle...")

        # Use built-in cell cycle genes
        if self.mito_prefix == "MT-":  # Human
            s_genes = [
                "MCM5",
                "PCNA",
                "TYMS",
                "FEN1",
                "MCM2",
                "MCM4",
                "RRM1",
                "UNG",
                "GINS2",
                "MCM6",
                "CDCA7",
                "DTL",
                "PRIM1",
                "UHRF1",
                "MLF1IP",
                "HELLS",
                "RFC2",
                "RPA2",
                "NASP",
                "RAD51AP1",
                "GMNN",
                "WDR76",
                "SLBP",
                "CCNE2",
            ]
            g2m_genes = [
                "HMGB2",
                "CDK1",
                "NUSAP1",
                "UBE2C",
                "BIRC5",
                "TPX2",
                "TOP2A",
                "NDC80",
                "CKS2",
                "NUF2",
                "CKS1B",
                "MKI67",
                "TMPO",
                "CENPF",
                "TACC3",
                "FAM64A",
                "SMC4",
                "CCNB2",
                "CKAP2L",
                "CKAP2",
                "AURKB",
                "BUB1",
                "KIF11",
            ]
        else:  # Mouse
            # Convert to mouse gene names (simplified - in practice use biomaRt)
            s_genes = [g.capitalize() for g in s_genes]
            g2m_genes = [g.capitalize() for g in g2m_genes]

        # Score cell cycle
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

    def _remove_ambient_rna(self, adata: anndata.AnnData):
        """Remove ambient RNA (soup)."""
        self.logger.info("Removing ambient RNA...")
        self.logger.warning(
            "Ambient RNA removal requires additional tools (SoupX/CellBender)"
        )
        # This would require integration with R (SoupX) or CellBender
        # For now, just flag it
        adata.uns["ambient_rna_removed"] = False

    def _compute_basic_embedding(self, adata: anndata.AnnData):
        """Compute basic dimensionality reduction for QC and visualization."""
        self.logger.info("Computing basic embedding for QC...")

        # Work on HVG subset if available
        if "highly_variable" in adata.var.columns:
            # Keep all genes but compute on HVG
            adata_hvg = adata[:, adata.var["highly_variable"]].copy()
        else:
            adata_hvg = adata.copy()

        # Scale data (important for PCA)
        sc.pp.scale(adata_hvg, max_value=10)

        # PCA
        sc.tl.pca(adata_hvg, svd_solver="arpack", n_comps=50)

        # Transfer PCA to main object
        adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
        adata.uns["pca"] = adata_hvg.uns["pca"]
        adata.varm["PCs"] = adata_hvg.varm.get("PCs", None)

        # Compute neighbors (needed for UMAP/clustering)
        sc.pp.neighbors(adata, n_neighbors=30, n_pcs=40)

        # UMAP for visualization
        sc.tl.umap(adata)

        self.logger.info("Computed PCA and UMAP embeddings")

    def _merge_samples(
        self, adata_list: list[anndata.AnnData], rna_files: list[dict]
    ) -> anndata.AnnData:
        """Merge multiple RNA samples."""
        self.logger.info("Merging samples...")

        adata_merged = anndata.concat(
            adata_list,
            join="outer",
            label="sample_id",
            keys=[rna_file["id"] for rna_file in rna_files],
        )

        # Batch correction if requested
        if self.batch_correction != "none" and self.batch_key:
            self.logger.info(f"Applying batch correction: {self.batch_correction}")

            if self.batch_correction == "harmony":
                try:
                    import scanpy.external as sce

                    # First need to do PCA
                    sc.pp.scale(adata_merged)
                    sc.tl.pca(adata_merged)
                    # Then harmony
                    sce.pp.harmony_integrate(adata_merged, self.batch_key)
                except ImportError:
                    self.logger.warning(
                        "harmony-py not installed, skipping batch correction"
                    )

            # Add other batch correction methods as needed

        return adata_merged

    @classmethod
    def get_command_name(cls) -> str:
        """Get the CLI command name for this processor."""
        return "multiome-rna"

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return "Process RNA-seq data for multiome experiments"

    @classmethod
    def add_cli_args(cls, parser) -> None:
        """Add multiome RNA-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # RNA-specific arguments
        parser.add_argument(
            "--rna",
            nargs="+",
            required=True,
            help="RNA expression files (h5ad/h5/mtx formats)",
        )
        parser.add_argument(
            "--normalization",
            choices=["raw", "cpm", "log1p", "tpm"],
            default="log1p",
            help="Normalization method (default: log1p)",
        )
        parser.add_argument(
            "--min-genes",
            type=int,
            default=200,
            help="Minimum genes per cell (default: 200)",
        )
        parser.add_argument(
            "--percent-mito",
            type=float,
            default=20.0,
            help="Maximum mitochondrial percentage (default: 20.0)",
        )
        parser.add_argument(
            "--n-top-genes",
            type=int,
            default=2000,
            help="Number of HVGs to detect (default: 2000)",
        )

