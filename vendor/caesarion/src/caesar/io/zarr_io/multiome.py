from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyranges as pr
from scipy.stats import pearsonr, spearmanr

from ..aggregator import Aggregator, RawCoverageDict, SumAggregator
from ..query import BaseQuery, DefaultQuery, IndexQuery, KeyQuery, MetadataQuery
from .base import COMPRESSOR
from .dense import CelltypeDenseZarrIO
from .sparse import SingleSparseZarrIO


class MultiomeIO(ABC):
    """Abstract base class for multiome data operations.

    Provides backend-agnostic interface for multimodal (ATAC + RNA) data.
    Future implementations (e.g., BPCells) should inherit from this class.
    """

    @abstractmethod
    def get_n_shared_cells(self) -> int:
        """Return count of cells shared between modalities."""
        pass

    @abstractmethod
    def get_shared_cell_identifiers(self) -> list[str]:
        """Return identifiers (barcodes) for shared cells."""
        pass

    @abstractmethod
    def get_modality_indices(self, barcodes: list[str]) -> dict[str, np.ndarray]:
        """Map barcodes to indices in each modality.

        Args:
            barcodes: List of cell barcodes to map

        Returns:
            dict with keys 'atac' and 'rna' containing index arrays
        """
        pass

    @abstractmethod
    def get_paired_track(
        self,
        chr_name: str,
        start: int,
        end: int,
        query: BaseQuery = None,
        aggregator: Aggregator = None,
        gene_names: list[str] = None,
    ) -> dict[str, Any]:
        """Get paired ATAC and RNA data.

        Args:
            chr_name: Chromosome name
            start: Start position
            end: End position
            query: Query object for cell selection
            aggregator: Aggregator for combining data
            gene_names: Optional list of genes to subset RNA data

        Returns:
            Dictionary with 'atac' and 'rna' keys containing respective data
        """
        pass


class MultiomeSparseZarrIO(SingleSparseZarrIO, MultiomeIO):
    """
    Multiome IO that extends SingleSparseZarrIO with RNA data and barcode mapping.

    This class inherits all ATAC functionality from SingleSparseZarrIO and adds:
    - RNA expression data (rna.h5ad)
    - Barcode mapping between ATAC and RNA (barcode_mapping.csv)
    - Methods to access paired data for shared cells

    Directory structure example (/tmp/multiome_pbmc_output/multiome_sparse.zarr/):
    ```
    multiome_sparse.zarr/
    ├── barcodes/                  # Cell barcodes (inherited from SingleSparseZarrIO)
    ├── labels/                    # Cell labels/clusters
    ├── chrs/                      # ATAC-seq coverage data by chromosome
    │   ├── chr1/
    │   │   ├── chunk_0/
    │   │   └── ...
    │   └── ...
    ├── library_size/              # Per-chromosome library sizes
    ├── per_cell_libsize/          # Per-cell library sizes
    ├── tiles/                     # Genomic tiles for analysis
    ├── genome_tiles.h5ad          # Processed genomic tiles
    ├── rna.h5ad                   # Full RNA expression data (all cells)
    └── barcode_mapping.csv        # Maps shared cells between ATAC and RNA
    ```
    """

    def __init__(self, path: str, mode: str = "r", dtype: str = "float32"):
        """Initialize multiome IO.

        Args:
            path: Path to multiome zarr directory
            mode: Access mode (typically 'r' for multiome)
            dtype: Data type for ATAC data
        """
        # Initialize as SingleSparseZarrIO first
        super().__init__(path, mode=mode, dtype=dtype)

        # Check if this is actually a multiome zarr
        if not self.dataset.attrs.get("multiome", False):
            raise ValueError(f"Zarr at {path} is not a multiome dataset")

        # Load barcode mapping
        self.mapping_path = Path(path) / "barcode_mapping.csv"
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Barcode mapping not found at {self.mapping_path}")
        self.mapping = pd.read_csv(self.mapping_path)

        # Set up RNA data path
        self.rna_path = Path(path) / "rna.h5ad"
        if not self.rna_path.exists():
            raise FileNotFoundError(f"RNA expression data not found at {self.rna_path}")

        # Lazy-load RNA data
        self._rna_data = None

        # Get multiome-specific attributes
        self.n_shared_cells = self.dataset.attrs.get(
            "n_shared_cells", len(self.mapping)
        )
        self.atac_n_cells = self.dataset.attrs.get("atac_n_cells", len(self.barcodes))
        self.rna_n_cells = self.dataset.attrs.get("rna_n_cells", -1)

    @property
    def rna_data(self):
        """Lazy-load RNA data."""
        if self._rna_data is None:
            import scanpy as sc

            self._rna_data = sc.read_h5ad(self.rna_path)
        return self._rna_data

    @property
    def shared_barcodes(self):
        """Get list of shared barcodes."""
        return self.mapping["barcode"].tolist()

    @property
    def shared_indices_atac(self):
        """Get ATAC indices for shared cells."""
        return self.mapping["atac_idx"].values

    @property
    def shared_indices_rna(self):
        """Get RNA indices for shared cells."""
        return self.mapping["rna_idx"].values

    def get_atac_for_shared_cells(self, chr_name: str, start: int, end: int):
        """Get ATAC data for shared cells only.

        Args:
            chr_name: Chromosome name
            start: Start position
            end: End position

        Returns:
            Coverage array for shared cells only - summed across cells (n_positions,)
        """
        # Use IndexQuery to get data for shared cells only
        query = IndexQuery(self.shared_indices_atac)
        # Get coverage data (aggregated across cells)
        raw_coverage = self._get_coverage_for_query(
            chr_name, start, end, query=query, sparse=False
        )

        # Extract the data array from the dict
        if "default" in raw_coverage:
            return raw_coverage["default"]
        else:
            # Return the first value if multiple keys
            return list(raw_coverage.values())[0]

    def get_cell_by_position_matrix(
        self, chr_name: str, start: int, end: int, cell_indices: np.ndarray = None
    ) -> np.ndarray:
        """Get cell-by-position matrix for ATAC data.

        Args:
            chr_name: Chromosome name
            start: Start position
            end: End position
            cell_indices: Indices of cells to include (default: all shared cells)

        Returns:
            Matrix of shape (n_cells, n_positions) with binary values
        """
        if cell_indices is None:
            cell_indices = self.shared_indices_atac

        # Initialize output matrix
        n_cells = len(cell_indices)
        n_positions = end - start
        matrix = np.zeros((n_cells, n_positions), dtype=np.int8)

        # Get data chunk by chunk
        start_chunk = start // self.chunk_size
        end_chunk = (end - 1) // self.chunk_size

        for chunk_idx in range(start_chunk, end_chunk + 1):
            chunk_data = self.get_chunk(chr_name, chunk_idx)

            # Process each cell
            for i, cell_idx in enumerate(cell_indices):
                if cell_idx < chunk_data.shape[0]:
                    cell_data = chunk_data[cell_idx, :]
                    # Extract non-zero positions
                    positions = cell_data[cell_data != 0]
                    # Filter to positions in our range
                    in_range = (positions >= start) & (positions < end)
                    positions_in_range = positions[in_range] - start
                    # Mark positions in matrix
                    if len(positions_in_range) > 0:
                        matrix[i, positions_in_range] = 1

        return matrix

    def get_rna_for_shared_cells(self):
        """Get RNA expression matrix for shared cells only.

        Returns:
            AnnData object with only shared cells
        """
        return self.rna_data[self.shared_indices_rna].copy()

    def get_paired_data(
        self, chr_name: str, start: int, end: int, gene_names: list[str] = None
    ):
        """Get paired ATAC and RNA data for shared cells.

        Args:
            chr_name: Chromosome name for ATAC data
            start: Start position for ATAC data
            end: End position for ATAC data
            gene_names: Optional list of genes to subset RNA data

        Returns:
            Tuple of (atac_data, rna_data) for shared cells
        """
        # Get ATAC for shared cells
        atac_data = self.get_atac_for_shared_cells(chr_name, start, end)

        # Get RNA for shared cells
        rna_data = self.get_rna_for_shared_cells()

        # Subset genes if requested
        if gene_names is not None:
            genes_mask = rna_data.var_names.isin(gene_names)
            rna_data = rna_data[:, genes_mask]

        return atac_data, rna_data

    # MultiomeIO interface implementation
    def get_n_shared_cells(self) -> int:
        """Return count of cells shared between modalities."""
        return self.n_shared_cells

    def get_shared_cell_identifiers(self) -> list[str]:
        """Return identifiers (barcodes) for shared cells."""
        return self.shared_barcodes

    def get_modality_indices(self, barcodes: list[str]) -> dict[str, np.ndarray]:
        """Map barcodes to indices in each modality."""
        atac_indices = []
        rna_indices = []

        for bc in barcodes:
            if bc in self.mapping.set_index("barcode").index:
                row = self.mapping[self.mapping["barcode"] == bc].iloc[0]
                atac_indices.append(row["atac_idx"])
                rna_indices.append(row["rna_idx"])

        return {
            "atac": np.array(atac_indices, dtype=int),
            "rna": np.array(rna_indices, dtype=int),
        }

    def get_paired_track(
        self,
        chr_name: str,
        start: int,
        end: int,
        query: BaseQuery = None,
        aggregator: Aggregator = None,
        gene_names: list[str] = None,
    ) -> dict[str, Any]:
        """Get paired ATAC and RNA data with query support."""
        # Get ATAC track using standard method
        atac_track = self.get_track(
            chr_name,
            start,
            end,
            query=query,
            aggregator=aggregator or SumAggregator(),
            output_format="raw_array",
        )

        # Get RNA data for the same cells
        if query is None:
            rna_data = self.get_rna_for_shared_cells()
        else:
            # Resolve query to indices
            indices = self._resolve_query_to_rna_indices(query)
            rna_data = self.rna_data[indices]

        # Filter genes if requested
        if gene_names is not None:
            genes_mask = rna_data.var_names.isin(gene_names)
            rna_data = rna_data[:, genes_mask]

        return {"atac": atac_track, "rna": rna_data}

    # Override query handling for multiome
    def _get_coverage_for_query(
        self, chr_name: str, start: int, end: int, query: BaseQuery, sparse: bool
    ) -> RawCoverageDict:
        """Override to handle multiome-specific query logic."""

        # For DefaultQuery, return all shared cells
        if isinstance(query, DefaultQuery):
            # Get ATAC data for shared cells only
            full_coverage = super()._get_coverage_for_query(
                chr_name, start, end, IndexQuery(self.shared_indices_atac), sparse
            )
            return full_coverage

        # For KeyQuery with barcodes
        elif isinstance(query, KeyQuery):
            # Map barcodes to indices
            query_indices = self._resolve_barcode_query(query.keys)
            if not query_indices:
                return {"default": np.zeros((0, end - start), dtype=self.dtype)}

            # Convert to IndexQuery and delegate
            return super()._get_coverage_for_query(
                chr_name, start, end, IndexQuery(query_indices), sparse
            )

        # For MetadataQuery, filter on shared cells only
        elif isinstance(query, MetadataQuery):
            # Get metadata for shared cells
            shared_metadata = self.metadata.iloc[self.shared_indices_atac]

            # Apply query
            mask = shared_metadata.eval(query.query_string)
            filtered_indices = self.shared_indices_atac[mask]

            return super()._get_coverage_for_query(
                chr_name, start, end, IndexQuery(filtered_indices), sparse
            )

        # For IndexQuery, validate indices are within shared cells
        elif isinstance(query, IndexQuery):
            # Ensure indices refer to shared cells
            valid_indices = [
                idx for idx in query.indices if idx in self.shared_indices_atac
            ]

            if not valid_indices:
                return {"default": np.zeros((0, end - start), dtype=self.dtype)}

            return super()._get_coverage_for_query(
                chr_name, start, end, IndexQuery(valid_indices), sparse
            )

        # Delegate other query types to parent
        else:
            return super()._get_coverage_for_query(chr_name, start, end, query, sparse)

    def _resolve_barcode_query(self, barcodes: list[str]) -> list[int]:
        """Resolve barcode query to ATAC indices for shared cells."""
        indices = []
        barcode_to_idx = dict(zip(self.mapping["barcode"], self.mapping["atac_idx"]))

        for bc in barcodes:
            if bc in barcode_to_idx:
                indices.append(barcode_to_idx[bc])

        return indices

    def _resolve_query_to_rna_indices(self, query: BaseQuery) -> np.ndarray:
        """Resolve a query to RNA indices."""
        if isinstance(query, DefaultQuery):
            return self.shared_indices_rna

        elif isinstance(query, KeyQuery):
            # Map barcodes to RNA indices
            indices = []
            barcode_to_idx = dict(zip(self.mapping["barcode"], self.mapping["rna_idx"]))

            for bc in query.keys:
                if bc in barcode_to_idx:
                    indices.append(barcode_to_idx[bc])

            return np.array(indices, dtype=int)

        elif isinstance(query, IndexQuery):
            # Map ATAC indices to RNA indices
            atac_to_rna = dict(zip(self.mapping["atac_idx"], self.mapping["rna_idx"]))
            indices = [
                atac_to_rna.get(idx) for idx in query.indices if idx in atac_to_rna
            ]
            return np.array([i for i in indices if i is not None], dtype=int)

        elif isinstance(query, MetadataQuery):
            # Apply query to shared cells metadata
            shared_metadata = self.metadata.iloc[self.shared_indices_atac]
            mask = shared_metadata.eval(query.query_string)
            # Get corresponding RNA indices
            filtered_atac = self.shared_indices_atac[mask]
            atac_to_rna = dict(zip(self.mapping["atac_idx"], self.mapping["rna_idx"]))
            indices = [atac_to_rna[idx] for idx in filtered_atac if idx in atac_to_rna]
            return np.array(indices, dtype=int)

        else:
            # Default to all shared cells
            return self.shared_indices_rna

    def get_correlation_matrix(
        self,
        chr_name: str,
        start: int,
        end: int,
        gene_names: list[str],
        method: str = "pearson",
    ) -> pd.DataFrame:
        """Calculate correlation between ATAC peaks and gene expression.

        Args:
            chr_name: Chromosome name
            start: Start position
            end: End position
            gene_names: List of genes to correlate with ATAC signal
            method: Correlation method ('pearson' or 'spearman')

        Returns:
            DataFrame with correlations between peaks and genes
        """
        # Get ATAC data as cell-by-position matrix
        atac_data = self.get_cell_by_position_matrix(chr_name, start, end)

        # Get RNA data for shared cells
        rna_data = self.get_rna_for_shared_cells()
        if gene_names is not None:
            genes_mask = rna_data.var_names.isin(gene_names)
            rna_data = rna_data[:, genes_mask]

        # Extract expression matrix
        rna_matrix = rna_data.X
        if hasattr(rna_matrix, "toarray"):
            rna_matrix = rna_matrix.toarray()

        # Ensure ATAC is 2D: (n_cells, n_positions)
        if atac_data.ndim == 1:
            atac_data = atac_data.reshape(-1, 1)

        # Compute correlation matrix
        corr_func = pearsonr if method == "pearson" else spearmanr

        n_positions = atac_data.shape[1]
        n_genes = rna_matrix.shape[1]

        corr_matrix = np.zeros((n_positions, n_genes))
        pval_matrix = np.zeros((n_positions, n_genes))

        for i in range(n_positions):
            for j in range(n_genes):
                peak_signal = atac_data[:, i]
                gene_expr = rna_matrix[:, j]

                # Skip if no variation
                if np.std(peak_signal) == 0 or np.std(gene_expr) == 0:
                    corr_matrix[i, j] = 0
                    pval_matrix[i, j] = 1
                else:
                    corr_matrix[i, j], pval_matrix[i, j] = corr_func(
                        peak_signal, gene_expr
                    )

        # Return as DataFrame with informative names
        # Create position names based on actual data
        position_names = [
            f"{chr_name}:{start+i}-{start+i+1}" for i in range(n_positions)
        ]

        return pd.DataFrame(
            corr_matrix, index=position_names, columns=rna_data.var_names
        )


class MultiomeDenseZarrIO(CelltypeDenseZarrIO, MultiomeIO):
    """
    A class for handling pseudobulked multiome data (ATAC + RNA) stored in Zarr format.

    This class extends CelltypeDenseZarrIO to include pseudobulked RNA expression profiles
    alongside ATAC coverage data, enabling regulatory analysis at the cell type level.

    Directory structure example (/tmp/multiome_pbmc_output/multiome_pseudobulk.zarr/):
    ```
    multiome_pseudobulk.zarr/
    ├── cluster_0/                  # Data for cell type/cluster 0
    │   ├── chrs/                   # ATAC coverage (inherited from CelltypeDenseZarrIO)
    │   │   ├── chr1/
    │   │   │   ├── 0
    │   │   │   └── ...
    │   │   └── ...
    │   ├── cluster_peaks/          # Peak calls for this cluster
    │   └── expression/             # RNA pseudobulk data
    │       ├── counts/             # Raw counts
    │       │   └── 0
    │       └── tpm/                # TPM normalized
    │           └── 0
    ├── cluster_1/
    │   └── ...
    └── .zattrs
    ```

    Note: The actual directory name (e.g., 'cluster_0' vs 'celltype_0') depends on
    the labels used during pseudobulking.

    Attributes:
        _expression_profiles (dict): Cached expression profiles per cell type
        _gene_annotations (pd.DataFrame): Cached gene annotation data

    Usage:
        mdz = MultiomeDenseZarrIO('/path/to/multiome_dense.zarr')
        expr = mdz.get_expression_profile('cluster_0')
        de = mdz.get_differential_expression('cluster_0', 'cluster_1')
        links = mdz.get_peak_gene_links('cluster_0')
    """

    def __init__(self, path: str, mode="r", celltype_ids=None, chroms=None):
        """Initialize MultiomeDenseZarrIO.

        Args:
            path: Path to the Zarr file
            mode: Mode for opening Zarr files ('r' for read-only, 'a' for read/write)
            celltype_ids: List of cell type IDs to include. If None, all cell types are included
            chroms: List of chromosomes to include. If None, all chromosomes are included
        """
        super().__init__(path, mode, celltype_ids, chroms)
        self._expression_profiles = {}
        self._gene_annotations = None

    @property
    def expression_profiles(self) -> dict[str, np.ndarray]:
        """Dictionary mapping cell type to TPM expression profile."""
        if not self._expression_profiles:
            self._load_all_expression_profiles()
        return self._expression_profiles

    @property
    def gene_annotations(self) -> pd.DataFrame:
        """Gene annotation DataFrame with coordinates."""
        if self._gene_annotations is None:
            self._load_gene_annotations()
        return self._gene_annotations

    def _load_all_expression_profiles(self):
        """Load expression profiles for all cell types."""
        for celltype_id in self.ids:
            if celltype_id not in self._expression_profiles:
                self._expression_profiles[celltype_id] = self._load_expression_profile(
                    celltype_id
                )

    def _load_expression_profile(self, celltype_id: str) -> np.ndarray:
        """Load expression profile for a specific cell type.

        Args:
            celltype_id: Cell type identifier

        Returns:
            TPM normalized expression array
        """
        if celltype_id not in self.dataset:
            raise KeyError(f"Cell type '{celltype_id}' not found in dataset")

        celltype_group = self.dataset[celltype_id]

        # Check if expression data exists
        if "expression" not in celltype_group:
            logging.warning(f"No expression data found for cell type '{celltype_id}'")
            return np.array([])

        expr_group = celltype_group["expression"]

        # Return TPM normalized values
        if "tpm" in expr_group:
            return expr_group["tpm"][:]
        else:
            logging.warning(f"No TPM data found for cell type '{celltype_id}'")
            return np.array([])

    def _load_gene_annotations(self):
        """Load gene annotations from the first cell type with expression data."""
        for celltype_id in self.ids:
            celltype_group = self.dataset[celltype_id]

            if (
                "expression" in celltype_group
                and "genes" in celltype_group["expression"]
            ):
                genes_group = celltype_group["expression"]["genes"]

                # Build gene annotation DataFrame
                gene_data = {}

                # Common gene annotation fields
                annotation_fields = [
                    "gene_id",
                    "gene_name",
                    "chromosome",
                    "strand",
                    "start",
                    "end",
                    "gene_type",
                    "tss",
                ]

                for field in annotation_fields:
                    if field in genes_group:
                        gene_data[field] = genes_group[field][:]

                if gene_data:
                    self._gene_annotations = pd.DataFrame(gene_data)
                    return

        logging.warning("No gene annotations found in any cell type")
        self._gene_annotations = pd.DataFrame()

    def get_expression_profile(self, celltype_id: str) -> np.ndarray:
        """Get TPM expression profile for a specific cell type.

        Args:
            celltype_id: Cell type identifier

        Returns:
            TPM normalized expression array for all genes
        """
        if celltype_id not in self._expression_profiles:
            self._expression_profiles[celltype_id] = self._load_expression_profile(
                celltype_id
            )
        return self._expression_profiles[celltype_id]

    def get_raw_counts(self, celltype_id: str) -> np.ndarray:
        """Get raw count expression profile for a specific cell type.

        Args:
            celltype_id: Cell type identifier

        Returns:
            Raw count array for all genes
        """
        if celltype_id not in self.dataset:
            raise KeyError(f"Cell type '{celltype_id}' not found in dataset")

        celltype_group = self.dataset[celltype_id]

        if "expression" not in celltype_group:
            logging.warning(f"No expression data found for cell type '{celltype_id}'")
            return np.array([])

        expr_group = celltype_group["expression"]

        if "counts" in expr_group:
            return expr_group["counts"][:]
        else:
            logging.warning(f"No count data found for cell type '{celltype_id}'")
            return np.array([])

    def get_differential_expression(
        self, celltype1: str, celltype2: str, method: str = "ttest"
    ) -> pd.DataFrame:
        """Calculate differential expression between cell types.

        Args:
            celltype1: First cell type ID
            celltype2: Second cell type ID
            method: Statistical method ('ttest', 'wilcoxon')

        Returns:
            DataFrame with differential expression statistics
        """

        # Get expression profiles
        expr1 = self.get_expression_profile(celltype1)
        expr2 = self.get_expression_profile(celltype2)

        if len(expr1) != len(expr2):
            raise ValueError("Expression profiles have different lengths")

        # Get gene annotations
        genes = self.gene_annotations.copy()

        if len(genes) != len(expr1):
            raise ValueError(
                "Gene annotations and expression profiles have different lengths"
            )

        # Calculate fold change
        # Add pseudocount to avoid division by zero
        pseudocount = 0.1
        fold_change = (expr1 + pseudocount) / (expr2 + pseudocount)
        log2_fold_change = np.log2(fold_change)

        # Calculate p-values based on method
        if method == "ttest":
            # For single pseudobulk samples, we can't do a t-test
            # Instead, use expression difference as a proxy
            expr_diff = expr1 - expr2
            # Create pseudo p-values based on expression difference magnitude
            # This is a simplified approach - real DE would need replicates
            p_values = 1 / (1 + np.abs(expr_diff))
        else:
            raise NotImplementedError(
                f"Method '{method}' not implemented for pseudobulk data without replicates"
            )

        # Build results DataFrame
        de_results = genes.copy()
        de_results["expr_" + celltype1] = expr1
        de_results["expr_" + celltype2] = expr2
        de_results["log2_fold_change"] = log2_fold_change
        de_results["p_value"] = p_values

        # Sort by absolute log2 fold change
        de_results["abs_lfc"] = np.abs(log2_fold_change)
        de_results = de_results.sort_values("abs_lfc", ascending=False)
        de_results = de_results.drop("abs_lfc", axis=1)

        return de_results

    def get_peak_gene_links(
        self,
        celltype_id: str,
        correlation_threshold: float = 0.3,
        distance_threshold: int = 500000,
    ) -> pd.DataFrame:
        """Find correlated peak-gene pairs within a cell type.

        This method looks for peaks that have been annotated with expression data
        (e.g., by AddExpressionProcessor) and returns peak-gene associations.

        Args:
            celltype_id: Cell type to analyze
            correlation_threshold: Minimum correlation coefficient (not used in current implementation)
            distance_threshold: Maximum peak-gene distance

        Returns:
            DataFrame with peak-gene links
        """
        # Get peaks with expression annotation
        peaks_with_expr = self.get_peaks(celltype_id, name="peaks_open_exp")

        if peaks_with_expr is None:
            logging.warning(
                f"No peaks with expression found for cell type '{celltype_id}'"
            )
            return pd.DataFrame()

        # Filter by distance if TSS information is available
        if "distance_to_tss" in peaks_with_expr.columns:
            mask = peaks_with_expr["distance_to_tss"].abs() <= distance_threshold
            peaks_with_expr = peaks_with_expr[mask]

        # Select relevant columns for peak-gene links
        link_columns = [
            "Chromosome",
            "Start",
            "End",
            "nearest_gene",
            "nearest_gene_name",
            "distance_to_tss",
        ]

        # Add expression columns if available
        expr_columns = [
            col for col in peaks_with_expr.columns if col.startswith("expr_")
        ]
        link_columns.extend(expr_columns)

        # Filter to available columns
        available_columns = [
            col for col in link_columns if col in peaks_with_expr.columns
        ]

        if not available_columns:
            logging.warning("No relevant columns found for peak-gene links")
            return pd.DataFrame()

        links_df = peaks_with_expr[available_columns].copy()

        # Add cell type information
        links_df["celltype"] = celltype_id

        return links_df

    def get_cross_modality_correlation(
        self, chr: str, start: int, end: int, gene_id: str = None, window: int = 5000
    ) -> float:
        """Calculate correlation between ATAC signal and gene expression.

        Args:
            chr: Chromosome name
            start: Start position
            end: End position
            gene_id: Gene identifier for expression correlation
            window: Window size around gene TSS for ATAC signal

        Returns:
            Correlation coefficient between ATAC and RNA
        """
        if gene_id is None:
            raise ValueError("gene_id is required for correlation calculation")

        # Get gene information
        gene_info = self.gene_annotations[self.gene_annotations["gene_id"] == gene_id]

        if gene_info.empty:
            raise ValueError(f"Gene '{gene_id}' not found in annotations")

        gene_info = gene_info.iloc[0]

        # Calculate correlations across cell types
        atac_signals = []
        rna_expressions = []

        for celltype_id in self.ids:
            # Get ATAC signal around gene TSS
            tss = gene_info["tss"] if "tss" in gene_info else gene_info["start"]
            region_start = max(0, tss - window)
            region_end = tss + window

            try:
                atac_track = self.get_track(celltype_id, chr, region_start, region_end)
                atac_signal = np.mean(atac_track)
                atac_signals.append(atac_signal)

                # Get RNA expression
                expr_profile = self.get_expression_profile(celltype_id)
                gene_idx = self.gene_annotations[
                    self.gene_annotations["gene_id"] == gene_id
                ].index[0]

                if gene_idx < len(expr_profile):
                    rna_expressions.append(expr_profile[gene_idx])
                else:
                    rna_expressions.append(0)

            except Exception as e:
                logging.warning(f"Error getting data for {celltype_id}: {e}")
                continue

        if len(atac_signals) < 2:
            return 0.0

        # Calculate correlation
        correlation = np.corrcoef(atac_signals, rna_expressions)[0, 1]

        return correlation if not np.isnan(correlation) else 0.0

    def get_regulatory_potential(
        self,
        peak_regions: pd.DataFrame,
        distance_weight: str = "exponential",
        max_distance: int = 500000,
    ) -> pd.DataFrame:
        """Calculate regulatory potential linking peaks to genes.

        Args:
            peak_regions: DataFrame with peak coordinates (Chromosome, Start, End)
            distance_weight: Weighting scheme for distance ('exponential', 'linear')
            max_distance: Maximum distance to consider for peak-gene links

        Returns:
            DataFrame with regulatory potential scores
        """
        # Initialize results
        regulatory_scores = []

        # Get gene annotations
        genes = self.gene_annotations

        # Convert to PyRanges for efficient overlap
        peaks_pr = pr.PyRanges(peak_regions)

        # Process each gene
        for _, gene in genes.iterrows():
            if "tss" in gene:
                tss = gene["tss"]
            else:
                tss = gene["start"] if gene.get("strand", "+") == "+" else gene["end"]

            # Find peaks within max_distance of TSS
            nearby_peaks = peaks_pr[gene["chromosome"]].df

            if nearby_peaks.empty:
                continue

            # Calculate distances
            nearby_peaks["distance"] = np.minimum(
                np.abs(nearby_peaks["Start"] - tss), np.abs(nearby_peaks["End"] - tss)
            )

            # Filter by max distance
            nearby_peaks = nearby_peaks[nearby_peaks["distance"] <= max_distance]

            if nearby_peaks.empty:
                continue

            # Calculate weights based on distance
            if distance_weight == "exponential":
                # Exponential decay with distance
                decay_rate = 5e-5  # Decay parameter
                weights = np.exp(-decay_rate * nearby_peaks["distance"])
            elif distance_weight == "linear":
                # Linear decay
                weights = 1 - (nearby_peaks["distance"] / max_distance)
            else:
                raise ValueError(f"Unknown distance_weight: {distance_weight}")

            # Calculate regulatory potential for this gene
            for idx, (_, peak) in enumerate(nearby_peaks.iterrows()):
                regulatory_scores.append(
                    {
                        "gene_id": gene.get("gene_id", ""),
                        "gene_name": gene.get("gene_name", ""),
                        "peak_chr": peak["Chromosome"],
                        "peak_start": peak["Start"],
                        "peak_end": peak["End"],
                        "distance": peak["distance"],
                        "weight": weights[idx],
                        "regulatory_potential": weights[
                            idx
                        ],  # Can be modified with peak strength
                    }
                )

        return pd.DataFrame(regulatory_scores)

    def get_chromatin_potential(self, genes: list[str] = None) -> np.ndarray:
        """Calculate chromatin potential scores.

        Chromatin potential represents the accessibility-weighted
        activity of a gene based on nearby regulatory elements.

        Args:
            genes: List of gene IDs to calculate potential for.
                   If None, calculate for all genes.

        Returns:
            Array of chromatin potential scores
        """
        gene_annotations = self.gene_annotations

        if genes is not None:
            # Filter to requested genes
            mask = gene_annotations["gene_id"].isin(genes)
            gene_annotations = gene_annotations[mask]

        # Initialize chromatin potential array
        chromatin_potential = np.zeros((len(gene_annotations), len(self.ids)))

        # Process each gene
        for gene_idx, (_, gene) in enumerate(gene_annotations.iterrows()):
            chr_name = gene["chromosome"]

            # Define region around gene
            if "tss" in gene:
                tss = gene["tss"]
            else:
                tss = gene["start"] if gene.get("strand", "+") == "+" else gene["end"]

            # Use 100kb window around TSS
            window = 100000
            region_start = max(0, tss - window)
            region_end = tss + window

            # Get accessibility for each cell type
            for ct_idx, celltype_id in enumerate(self.ids):
                try:
                    # Get ATAC signal
                    atac_signal = self.get_track(
                        celltype_id, chr_name, region_start, region_end
                    )

                    # Weight by distance from TSS
                    positions = np.arange(region_start, region_end)
                    distances = np.abs(positions - tss)
                    weights = np.exp(-distances / 10000)  # Exponential decay

                    # Calculate weighted sum
                    if len(atac_signal) == len(weights):
                        chromatin_potential[gene_idx, ct_idx] = np.sum(
                            atac_signal * weights
                        )
                    else:
                        # Fallback to simple mean if lengths don't match
                        chromatin_potential[gene_idx, ct_idx] = np.mean(atac_signal)

                except Exception as e:
                    logging.warning(
                        f"Error calculating chromatin potential for {gene['gene_id']} in {celltype_id}: {e}"
                    )
                    chromatin_potential[gene_idx, ct_idx] = 0

        return chromatin_potential

    def get_multimodal_embedding(
        self,
        method: str = "wnn",
        n_components: int = 30,
        weight_atac: float = 0.5,
        weight_rna: float = 0.5,
    ) -> np.ndarray:
        """Generate joint embedding from ATAC and RNA modalities.

        Note: This method operates on pseudobulked data (cell types as samples),
        not single cells. The resulting embedding will have shape (n_celltypes, n_components).

        Args:
            method: Integration method (currently only 'wnn' is supported)
            n_components: Number of components in embedding
            weight_atac: Weight for ATAC modality
            weight_rna: Weight for RNA modality

        Returns:
            Joint embedding matrix (celltypes x components)
        """
        if method != "wnn":
            raise NotImplementedError(
                f"Method '{method}' not implemented. Only 'wnn' is supported."
            )

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Prepare RNA data matrix (genes x celltypes)
        rna_matrix = []
        for celltype_id in self.ids:
            expr = self.get_expression_profile(celltype_id)
            rna_matrix.append(expr)
        rna_matrix = np.array(rna_matrix).T  # genes x celltypes

        # Prepare ATAC data matrix (regions x celltypes)
        # Use accessibility at gene TSS regions as features
        genes = self.gene_annotations
        atac_matrix = np.zeros((len(genes), len(self.ids)))

        for gene_idx, (_, gene) in enumerate(genes.iterrows()):
            chr_name = gene["chromosome"]
            tss = gene.get("tss", gene["start"])

            # 2kb window around TSS
            region_start = max(0, tss - 1000)
            region_end = tss + 1000

            for ct_idx, celltype_id in enumerate(self.ids):
                try:
                    signal = self.get_track(
                        celltype_id, chr_name, region_start, region_end
                    )
                    atac_matrix[gene_idx, ct_idx] = np.mean(signal)
                except Exception:
                    atac_matrix[gene_idx, ct_idx] = 0

        # Transpose to celltypes x features
        rna_matrix = rna_matrix.T
        atac_matrix = atac_matrix.T

        # Standardize
        scaler = StandardScaler()
        rna_scaled = scaler.fit_transform(rna_matrix)
        atac_scaled = scaler.fit_transform(atac_matrix)

        # PCA on each modality
        pca_rna = PCA(n_components=n_components)
        pca_atac = PCA(n_components=n_components)

        rna_embedding = pca_rna.fit_transform(rna_scaled)
        atac_embedding = pca_atac.fit_transform(atac_scaled)

        # Weighted combination
        joint_embedding = weight_rna * rna_embedding + weight_atac * atac_embedding

        return joint_embedding

    def write_expression_data(
        self,
        celltype_id: str,
        counts: np.ndarray,
        tpm: np.ndarray,
        gene_info: pd.DataFrame,
    ):
        """Write expression data for a specific cell type.

        Args:
            celltype_id: Cell type identifier
            counts: Raw count array
            tpm: TPM normalized array
            gene_info: DataFrame with gene annotations
        """
        if self.mode not in ["w", "a"]:
            raise ValueError("Cannot write expression data in read-only mode")

        if celltype_id not in self.dataset:
            raise KeyError(f"Cell type '{celltype_id}' not found in dataset")

        celltype_group = self.dataset[celltype_id]

        # Create expression group if it doesn't exist
        if "expression" not in celltype_group:
            expr_group = celltype_group.create_group("expression")
        else:
            expr_group = celltype_group["expression"]

        # Write arrays
        if "counts" in expr_group:
            del expr_group["counts"]
        expr_group.create_array("counts", data=counts, compressors=COMPRESSOR)

        if "tpm" in expr_group:
            del expr_group["tpm"]
        expr_group.create_array("tpm", data=tpm, compressors=COMPRESSOR)

        # Write gene metadata
        if "genes" not in expr_group:
            genes_group = expr_group.create_group("genes")
        else:
            genes_group = expr_group["genes"]

        # Write each column as a separate array
        for col in gene_info.columns:
            if col in genes_group:
                del genes_group[col]
            genes_group.create_array(
                col, data=gene_info[col].values, compressors=COMPRESSOR
            )

        # Clear caches
        if celltype_id in self._expression_profiles:
            del self._expression_profiles[celltype_id]
        self._gene_annotations = None

        logging.info(f"Expression data written for cell type '{celltype_id}'")

    # MultiomeIO interface implementation
    def get_n_shared_cells(self) -> int:
        """For dense multiome, return number of cell types/clusters."""
        return len(self.ids)

    def get_shared_cell_identifiers(self) -> list[str]:
        """For dense multiome, return cell type/cluster identifiers."""
        return self.ids

    def get_modality_indices(self, barcodes: list[str]) -> dict[str, np.ndarray]:
        """For dense multiome, map cell type IDs to indices.

        Note: In dense format, 'barcodes' are actually cell type IDs.
        """
        indices = []
        for i, ct_id in enumerate(self.ids):
            if ct_id in barcodes:
                indices.append(i)

        # Same indices for both modalities in dense format
        idx_array = np.array(indices, dtype=int)
        return {"atac": idx_array, "rna": idx_array}

    def get_paired_track(
        self,
        chr_name: str,
        start: int,
        end: int,
        query: BaseQuery = None,
        aggregator: Aggregator = None,
        gene_names: list[str] = None,
    ) -> dict[str, Any]:
        """Get paired ATAC and RNA data for dense multiome.

        For dense data, returns dictionaries mapping cell types to their
        ATAC and RNA data.
        """
        # Get ATAC tracks
        atac_tracks = self.get_track(
            chr_name,
            start,
            end,
            query=query,
            aggregator=aggregator,
            output_format="raw_dict",
        )

        # Get RNA data - build dictionary of expression profiles
        rna_data = {}

        # Determine which cell types to include based on query
        if query is None or isinstance(query, DefaultQuery):
            celltype_ids = self.ids
        elif isinstance(query, KeyQuery):
            celltype_ids = [ct for ct in query.keys if ct in self.ids]
        else:
            # For other query types, use ATAC results to determine cell types
            celltype_ids = list(atac_tracks.keys())

        # Get expression profiles for selected cell types
        for ct_id in celltype_ids:
            if ct_id in self.ids:
                expr_profile = self.get_expression_profile(ct_id)
                if gene_names is not None and self.gene_annotations is not None:
                    # Filter to requested genes
                    gene_mask = self.gene_annotations["gene_name"].isin(gene_names)
                    rna_data[ct_id] = expr_profile[gene_mask]
                else:
                    rna_data[ct_id] = expr_profile

        return {"atac": atac_tracks, "rna": rna_data}
