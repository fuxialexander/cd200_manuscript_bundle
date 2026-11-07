"""Add gene expression data to CelltypeDenseZarr.

This processor adds scRNA-seq expression data to existing CelltypeDenseZarr files,
computing pseudobulk expression profiles per cell type and mapping them to peaks
using gene annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyranges as pr
import scanpy as sc
from scipy.sparse import csr_matrix
from tqdm import tqdm

from caesar.io.zarr_io import CelltypeDenseZarrIO
from caesar.preprocess.core.base import BaseProcessor
from caesar.preprocess.core.exceptions import ProcessingError, ValidationError
from gcell.rna.gencode import Gencode

if TYPE_CHECKING:
    from caesar.preprocess.core.config import CaesarConfig


def count_to_log_tpm(count: np.ndarray) -> np.ndarray:
    """Convert raw counts to log TPM values.

    Args:
        count: Raw count array

    Returns:
        Log TPM values
    """
    return np.log10(count / count.sum() * 1e6 + 1)


class AddExpressionProcessor(BaseProcessor):
    """Processor for adding gene expression data to CelltypeDenseZarr.

    This processor:
    1. Reads scRNA-seq data (h5ad/h5/mtx formats)
    2. Calculates pseudobulk expression profiles per cell type
    3. Computes log TPM values
    4. Maps expression to peaks using gene annotations
    5. Stores expression for both strands
    6. Annotates peaks with TSS information
    """

    def __init__(self, config: CaesarConfig):
        """Initialize AddExpressionProcessor.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        # Expression-specific configuration
        self.expression_config = (
            config.add_expression if hasattr(config, "add_expression") else None
        )
        if not self.expression_config:
            raise ValidationError("No 'add_expression' configuration found")

        # Extract config values
        self.expression_data = self.expression_config.expression_data
        self.gene_annotations = self.expression_config.gene_annotations
        self.assembly = self.expression_config.assembly
        self.gencode_version = self.expression_config.gencode_version
        self.tss_window = self.expression_config.tss_window
        self.min_cells_per_type = self.expression_config.min_cells_per_type
        self.normalization_method = self.expression_config.normalization_method
        self.peak_name = self.expression_config.peak_name
        self.labels_column = self.expression_config.labels_column

        # Set input zarr path if provided
        if hasattr(self.expression_config, "input_zarr") and self.expression_config.input_zarr:
            self.input_zarr = Path(self.expression_config.input_zarr)
        else:
            self.input_zarr = None

    def validate_config(self) -> None:
        """Validate configuration."""
        # Check expression data exists
        if not self.expression_data:
            raise ValidationError("No expression data path specified")

        expr_path = Path(self.expression_data)
        if not expr_path.exists():
            raise ValidationError(f"Expression data not found: {expr_path}")

        # Validate file format
        valid_extensions = [".h5ad", ".h5", ".mtx"]
        if not any(str(expr_path).endswith(ext) for ext in valid_extensions):
            raise ValidationError(
                f"Expression data must be in h5ad, h5, or mtx format, got: {expr_path.suffix}"
            )

        # Check gene annotations or assembly
        if not self.gene_annotations and not self.assembly:
            raise ValidationError(
                "Either gene_annotations path or assembly must be specified"
            )

        if self.gene_annotations:
            annot_path = Path(self.gene_annotations)
            if not annot_path.exists():
                raise ValidationError(f"Gene annotations not found: {annot_path}")

            # Validate annotation format
            if annot_path.suffix not in [".gtf", ".gff", ".gff3", ".gz"]:
                raise ValidationError(
                    f"Gene annotations must be GTF or GFF format, got: {annot_path.suffix}"
                )

        # Validate input zarr
        if not self.input_zarr:
            raise ValidationError("No input zarr specified")

        if not self.input_zarr.exists():
            raise ValidationError(f"Input zarr not found: {self.input_zarr}")

    def load_expression_data(self) -> sc.AnnData:
        """Load expression data using scanpy.

        Returns:
            AnnData object with expression data
        """
        expr_path = Path(self.expression_data)

        self.logger.info(f"Loading expression data from {expr_path}")

        if expr_path.suffix == ".h5ad":
            adata = sc.read_h5ad(expr_path)
        elif expr_path.suffix == ".h5":
            adata = sc.read_10x_h5(expr_path)
        elif expr_path.suffix == ".mtx" or expr_path.name.endswith(".mtx.gz"):
            # For mtx format, assume standard 10x directory structure
            parent_dir = expr_path.parent
            adata = sc.read_10x_mtx(parent_dir)
        else:
            raise ValueError(f"Unsupported format: {expr_path.suffix}")

        self.logger.info(
            f"Loaded expression data: {adata.n_obs} cells x {adata.n_vars} genes"
        )

        return adata

    def load_gene_annotations(self) -> pd.DataFrame:
        """Load gene annotations from GTF/GFF file or using Gencode.

        Returns:
            DataFrame with gene annotations
        """
        if self.assembly:
            # Use Gencode class for automatic loading
            self.logger.info(
                f"Loading gene annotations using Gencode for assembly {self.assembly}"
            )

            # Initialize Gencode with optional version
            if self.gencode_version:
                gencode = Gencode(
                    assembly=self.assembly, gencode_version=self.gencode_version
                )
            else:
                gencode = Gencode(assembly=self.assembly)

            # Get gene dataframe from Gencode.gtf attribute
            gtf_df = gencode.gtf

            # Gencode.gtf has standard columns: Chromosome, Start, End, Strand, gene_name, gene_id
            # No need for additional processing

            self.logger.info(f"Loaded {len(gtf_df)} gene annotations from Gencode")

        else:
            # Fall back to manual loading from file
            annot_path = Path(self.gene_annotations)
            self.logger.info(f"Loading gene annotations from {annot_path}")

            # Use pyranges to read GTF/GFF
            if str(annot_path).endswith(".gz"):
                annotations = pr.read_gtf(str(annot_path))
            else:
                annotations = pr.read_gtf(str(annot_path))

            # Convert to DataFrame
            gtf_df = annotations.df

            # Filter for genes only
            if "Feature" in gtf_df.columns:
                gtf_df = gtf_df[gtf_df["Feature"] == "gene"]
            elif "feature" in gtf_df.columns:
                gtf_df = gtf_df[gtf_df["feature"] == "gene"]

            # Ensure required columns
            gene_name_col = None
            for col in ["gene_name", "gene_id", "Name", "ID"]:
                if col in gtf_df.columns:
                    gene_name_col = col
                    break

            if not gene_name_col:
                raise ValueError("No gene name column found in annotations")

            # Rename to standard column
            gtf_df["gene_name"] = gtf_df[gene_name_col]

            self.logger.info(f"Loaded {len(gtf_df)} gene annotations from file")

        return gtf_df

    def calculate_pseudobulk(self, adata: sc.AnnData) -> pd.DataFrame:
        """Calculate pseudobulk expression per cell type.

        Args:
            adata: AnnData object with expression and cell type labels

        Returns:
            DataFrame with genes as rows and cell types as columns
        """
        if self.labels_column not in adata.obs.columns:
            raise ValueError(
                f"Labels column '{self.labels_column}' not found in adata.obs"
            )

        pseudobulk = {}

        for celltype in adata.obs[self.labels_column].unique():
            # Get cells for this cell type
            mask = adata.obs[self.labels_column] == celltype
            n_cells = mask.sum()

            if n_cells < self.min_cells_per_type:
                self.logger.warning(
                    f"Skipping cell type {celltype} with only {n_cells} cells "
                    f"(min required: {self.min_cells_per_type})"
                )
                continue

            # Calculate total counts
            if hasattr(adata.X, "toarray"):
                total_count = np.array(adata[mask].X.sum(0)).flatten()
            else:
                total_count = np.array(adata[mask].X.sum(0)).flatten()

            # Convert to TPM
            if self.normalization_method == "tpm":
                expr_values = count_to_log_tpm(total_count)
            elif self.normalization_method == "cpm":
                # CPM normalization
                expr_values = np.log10(total_count / total_count.sum() * 1e6 + 1)
            else:
                raise ValueError(
                    f"Unknown normalization method: {self.normalization_method}"
                )

            pseudobulk[celltype] = expr_values

        if not pseudobulk:
            raise ValueError("No cell types passed filtering criteria")

        return pd.DataFrame(pseudobulk, index=adata.var_names)

    def process_expression_for_celltype(
        self,
        celltype_id: str,
        expression_df: pd.DataFrame,
        gtf_df: pd.DataFrame,
        cdz: CelltypeDenseZarrIO,
    ) -> None:
        """Process expression for a single cell type.

        Args:
            celltype_id: Cell type identifier
            expression_df: Expression data with this cell type as column
            gtf_df: Gene annotations
            cdz: CelltypeDenseZarr instance
        """
        if celltype_id not in expression_df.columns:
            self.logger.warning(f"Cell type {celltype_id} not found in expression data")
            return

        # Get expression for this cell type
        exp = expression_df[[celltype_id]].reset_index()
        exp.columns = ["gene_name", "TPM"]

        # Merge with gene annotations
        exp = pd.merge(
            gtf_df,
            exp[["gene_name", "TPM"]].drop_duplicates(),
            left_on="gene_name",
            right_on="gene_name",
            how="left",
        )

        # Fill missing values
        exp["TPM"] = exp["TPM"].fillna(0)

        # Convert strand to numeric (0 for +, 1 for -)
        exp["Strand_exp"] = exp["Strand"].map({"+": 0, "-": 1}).fillna(0).astype(int)

        # Write expression-annotated peaks
        self.write_expression_peaks(celltype_id, exp, cdz)

    def write_expression_peaks(
        self, celltype_id: str, exp_df: pd.DataFrame, cdz: CelltypeDenseZarrIO
    ) -> None:
        """Write expression-annotated peaks for a cell type.

        Args:
            celltype_id: Cell type identifier
            exp_df: Expression data with gene annotations
            cdz: CelltypeDenseZarr instance
        """
        # Get existing peaks
        source_peak_name = f"{self.peak_name}_open"
        peaks = cdz.get_peaks(celltype_id, name=source_peak_name)

        if peaks is None:
            self.logger.warning(
                f"No peaks found for {celltype_id} with name {source_peak_name}"
            )
            return

        # Extend gene coordinates by TSS window
        exp_extended = pr.PyRanges(exp_df).extend(self.tss_window)

        # Find overlaps between peaks and genes
        peaks_pr = pr.PyRanges(peaks.reset_index(drop=True).reset_index())
        overlap = peaks_pr.join(exp_extended, suffix="_exp", how="left", apply_strand_suffix=True).df

        # Clean up overlap results - handle both old and new pyranges column naming
        strand_col = "Strand_exp" if "Strand_exp" in overlap.columns else "Strand"
        overlap = overlap[
            ["index", "Chromosome", "Start", "End", "gene_name", strand_col, "TPM"]
        ].rename(columns={strand_col: "Strand_exp"})
        overlap = overlap.drop_duplicates()

        # Filter valid overlaps
        valid_overlap = overlap.query('gene_name != "-1" & TPM >= 0')

        # Get maximum expression per peak and strand
        if not valid_overlap.empty:
            # Convert strand to numeric: + -> 0 (positive), - -> 1 (negative)
            strand_map = {"+": 0, "-": 1, 0: 0, 1: 1}
            valid_overlap = valid_overlap.copy()
            valid_overlap["Strand_numeric"] = valid_overlap["Strand_exp"].map(
                lambda x: strand_map.get(x, 0)
            )

            max_expr = valid_overlap.groupby(["index", "Strand_numeric"])["TPM"].max()
            max_expr = max_expr.reset_index()

            # Create expression matrix (peaks x strands)
            row = max_expr["index"].values.astype(np.int64)
            col = max_expr["Strand_numeric"].values.astype(np.int64)
            data = max_expr["TPM"].values.astype(np.float32)

            exp_matrix = csr_matrix((data, (row, col)), shape=(len(peaks), 2)).toarray()
        else:
            exp_matrix = np.zeros((len(peaks), 2), dtype=np.float32)

        # Add expression columns to peaks
        peaks = peaks.reset_index(drop=True)
        peaks["Expression_positive"] = exp_matrix[:, 0]
        peaks["Expression_negative"] = exp_matrix[:, 1]

        # Mark TSS peaks
        peaks["TSS"] = 0
        if not valid_overlap.empty:
            tss_indices = valid_overlap["index"].unique()
            peaks.loc[tss_indices, "TSS"] = 1

        # Calculate normalized accessibility (aTPM) if Count column exists
        if "Count" in peaks.columns:
            peaks["aTPM"] = np.log10(peaks["Count"] / peaks["Count"].sum() * 1e5 + 1)
            peaks["aTPM"] = peaks["aTPM"] / peaks["aTPM"].max()
            columns_to_include = [
                "Start",
                "End",
                "Expression_positive",
                "Expression_negative",
                "aTPM",
                "Count",
                "TSS",
            ]
        else:
            columns_to_include = [
                "Start",
                "End",
                "Expression_positive",
                "Expression_negative",
                "TSS",
            ]

        # Write peaks with expression
        output_peak_name = f"{self.peak_name}_open_exp"
        cdz.write_peaks(
            celltype_id,
            peaks,
            name=output_peak_name,
            columns_to_include=columns_to_include,
            overwrite=True,
        )

        self.logger.info(
            f"Wrote expression-annotated peaks for {celltype_id}: "
            f"{(peaks['TSS'] == 1).sum()} TSS peaks out of {len(peaks)} total"
        )

    def process(self) -> dict[str, Any]:
        """Process expression data and add to CelltypeDenseZarr.

        Returns:
            Processing results
        """
        self._start_processing()

        try:
            # Load expression data
            adata = self.load_expression_data()

            # Load gene annotations
            gtf_df = self.load_gene_annotations()

            # Open CelltypeDenseZarr
            cdz = CelltypeDenseZarrIO(str(self.input_zarr), mode="a")

            # Calculate pseudobulk expression
            self.logger.info("Calculating pseudobulk expression per cell type")
            pseudobulk_df = self.calculate_pseudobulk(adata)

            # Process each cell type
            processed_celltypes = []
            for celltype_id in tqdm(cdz.ids, desc="Processing cell types"):
                # Match cell type ID to expression data
                # Handle potential naming differences
                matching_cols = [
                    col for col in pseudobulk_df.columns if celltype_id in col
                ]

                if not matching_cols:
                    self.logger.warning(f"No expression data found for {celltype_id}")
                    continue

                # Use first matching column
                expr_col = matching_cols[0]
                expr_df = pseudobulk_df[[expr_col]].copy()
                expr_df.columns = [celltype_id]

                self.process_expression_for_celltype(celltype_id, expr_df, gtf_df, cdz)
                processed_celltypes.append(celltype_id)

            self.logger.info(
                f"Completed expression annotation for {len(processed_celltypes)} cell types"
            )

        except Exception as e:
            self.logger.error(f"Error processing expression data: {e}")
            raise ProcessingError(f"Expression processing failed: {e}") from e

        runtime = self._end_processing()

        return {
            "input_zarr": str(self.input_zarr),
            "expression_data": str(self.expression_data),
            "gene_annotations": str(self.gene_annotations)
            if self.gene_annotations
            else "Gencode",
            "assembly": self.assembly if self.assembly else "from_file",
            "gencode_version": self.gencode_version if self.gencode_version else "auto",
            "n_celltypes_processed": len(processed_celltypes),
            "celltypes": processed_celltypes,
            "peak_name": f"{self.peak_name}_open_exp",
            "runtime": runtime,
        }
