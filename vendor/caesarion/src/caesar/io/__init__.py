"""
Caesar I/O Module - Backend-agnostic genomic data I/O operations.

This module provides a unified interface for reading, writing, and manipulating
genomic data across different storage backends (Zarr, BPCells, etc.).
"""

# Base classes
# Aggregator classes
from __future__ import annotations

from .aggregator import (
    AggregatedCoverage,
    Aggregator,
    ByKeyAggregator,
    ConcatAggregator,
    DataType,
    RawCoverageDict,
    SumAggregator,
)
from .genomic_io import GenomicIO

# Interface classes
from .interface import (
    CelltypeDenseIO,
    DenseIO,
    MultiSparseIO,
    SequenceDenseIO,
    SignalDenseIO,
    SingleSparseIO,
)

# Metadata I/O
from .metadata_io import Metadata

# Mixin classes
from .mixins import (
    CallPeaksMixin,
    ClusteringMixin,
    FootprintMixin,
    LibrarySizeAnnotationMixin,
    QuantileNormalizationMixin,
    TSSExtractionMixin,
    GenomicMaskingMixin,
)

# Normalizer classes
from .normalizer import (
    DictNormalizer,
    FactorNormalizer,
    NoNormalizer,
    Normalizer,
)

# Query classes
from .query import (
    BaseQuery,
    DefaultQuery,
    IndexQuery,
    KeyQuery,
    MetadataQuery,
    MultiKeyQuery,
)

# Peak I/O utilities - functions that don't require caesar.io imports
# These are defined here directly to avoid circular imports
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def read_peaks(peaks: str | pd.DataFrame) -> pd.DataFrame:
    """
    Read peaks from a file or DataFrame.

    Args:
        peaks (str | pd.DataFrame): Path to peaks file or DataFrame of peaks.
            Supported file formats: .bed, .narrowPeak.
            DataFrame requires columns: Chromosome, Start, End.

    Returns:
        pd.DataFrame: DataFrame containing peak regions with standard columns.

    Raises:
        ValueError: If input is not a valid path or DataFrame, or if DataFrame
                    lacks required columns.
        FileNotFoundError: If the provided file path does not exist.
    """
    if isinstance(peaks, str):
        if not os.path.exists(peaks):
            raise FileNotFoundError(f"Peak file not found: {peaks}")

        if peaks.endswith(".bed"):
            peaks_df = pd.read_csv(
                peaks,
                sep="\t",
                header=None,
                usecols=[0, 1, 2],
                names=["Chromosome", "Start", "End"],
            )
        elif peaks.endswith(".narrowPeak"):
            peaks_df = pd.read_csv(
                peaks,
                sep="\t",
                header=None,
                usecols=[0, 1, 2],
                names=["Chromosome", "Start", "End"],
            )
        else:
            raise ValueError(
                f"Unsupported peak file format for: {peaks}. Use .bed or .narrowPeak."
            )
    elif isinstance(peaks, pd.DataFrame):
        logging.info(
            "Reading peaks from DataFrame, ensuring columns: Chromosome, Start, End"
        )
        required_columns = ["Chromosome", "Start", "End"]
        if not all(col in peaks.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        peaks_df = peaks[required_columns].copy()
    else:
        raise ValueError(
            "Invalid input for peaks. Provide a file path (str) or a pandas DataFrame."
        )

    # Basic validation
    if not pd.api.types.is_numeric_dtype(
        peaks_df["Start"]
    ) or not pd.api.types.is_numeric_dtype(peaks_df["End"]):
        raise ValueError("Start and End columns must be numeric.")
    if (peaks_df["End"] < peaks_df["Start"]).any():
        logging.warning("Some peak End coordinates are less than Start coordinates.")

    return peaks_df


def prepare_peaks_for_processing(
    peaks: pd.DataFrame,
    sort: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Prepare peaks for processing with optional sorting and index tracking.

    This function prepares peaks for efficient processing by:
    1. Sorting by ["Chromosome", "Start"] for efficient processing
    2. Generating a reorder_index to restore original order after processing

    Matrix reordering is fast enough that we always generate the reorder_index
    even if peaks are already sorted, simplifying the code.

    Args:
        peaks: DataFrame with 'Chromosome', 'Start', 'End' columns
        sort: Whether to sort peaks (default: True). If False, returns peaks as-is
            with None reorder_index.

    Returns:
        Tuple of (sorted_peaks, reorder_index):
        - sorted_peaks: Peaks DataFrame, sorted if sort=True
        - reorder_index: Array to reorder matrix columns back to original order,
            or None if sort=False
    """
    if not sort:
        return peaks, None

    # Sort the peaks and get the indices that would sort them
    sorted_peaks = peaks.sort_values(["Chromosome", "Start"], kind="mergesort")

    # Get the original index positions after sorting
    sort_order = sorted_peaks.index.values

    # Reset index to [0, 1, 2, ...] for clean output
    sorted_peaks = sorted_peaks.reset_index(drop=True)

    # Generate reorder_index: map from sorted position back to original position
    reorder_index = np.argsort(sort_order)

    logging.debug(
        f"Sorted {len(peaks)} peaks for efficient processing. "
        f"Reorder index created for matrix column reordering."
    )

    return sorted_peaks, reorder_index


def reorder_matrix_columns(
    matrix: csr_matrix,
    reorder_index: np.ndarray | None,
) -> csr_matrix:
    """Reorder matrix columns using reorder index.

    This efficiently reorders matrix columns back to original peak order
    without sorting the peaks DataFrame itself.

    Args:
        matrix: Sparse matrix with shape (cells, peaks)
        reorder_index: Array of indices to reorder columns, or None if no reordering needed

    Returns:
        Reordered matrix with columns in original peak order
    """
    if reorder_index is None:
        return matrix

    # Reorder columns by selecting columns in reorder_index order
    reordered = matrix[:, reorder_index]

    logging.debug(f"Reordered matrix columns using reorder_index (shape: {reorder_index.shape})")

    return reordered


def ensure_peaks_match_matrix(
    peaks: pd.DataFrame,
    matrix: csr_matrix,
    stored_peaks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ensure peaks DataFrame matches matrix columns.

    This function ensures the peaks DataFrame matches the matrix column order,
    using stored peaks if available (more reliable) or sorting if needed.

    Args:
        peaks: Original peaks DataFrame
        matrix: Count matrix with shape (cells, peaks)
        stored_peaks: Peaks read back from storage (if available)

    Returns:
        Peaks DataFrame matching matrix column order

    Raises:
        ValueError: If peaks count doesn't match matrix columns
    """
    if stored_peaks is not None:
        # Use stored peaks - they match what's in the matrix
        result_peaks = stored_peaks
        logging.debug(f"Using stored peaks: {len(result_peaks)} peaks")
    else:
        # Fallback: use input peaks (should already be sorted)
        result_peaks = peaks.copy()
        logging.debug(f"Using input peaks: {len(result_peaks)} peaks")

    # Validate count matches
    if len(result_peaks) != matrix.shape[1]:
        raise ValueError(
            f"Peaks count ({len(result_peaks)}) does not match matrix columns "
            f"({matrix.shape[1]}). This indicates a mismatch between "
            "peaks and computed matrix."
        )

    return result_peaks


def sanitize_column_name(name: str) -> str:
    """Sanitize column name by replacing special characters with underscores."""
    return name.replace(".", "_").replace("-", "_")


# Zarr implementations
from .zarr_io import (
    CelltypeDenseZarrIO,
    DenseZarrIO,
    MotifDenseZarrIO,
    MultiSparseZarrIO,
    SignalDenseZarrIO,
    SingleSparseZarrIO,
    ZarrIO,
    open_zarr,
)

__all__ = [
    # Base classes
    "GenomicIO",
    # Interface classes
    "DenseIO",
    "SignalDenseIO",
    "SingleSparseIO",
    "MultiSparseIO",
    "SequenceDenseIO",
    "CelltypeDenseIO",
    # Query classes
    "BaseQuery",
    "DefaultQuery",
    "KeyQuery",
    "IndexQuery",
    "MetadataQuery",
    "MultiKeyQuery",
    # Aggregator classes
    "Aggregator",
    "SumAggregator",
    "ByKeyAggregator",
    "ConcatAggregator",
    "DataType",
    "RawCoverageDict",
    "AggregatedCoverage",
    # Normalizer classes
    "Normalizer",
    "NoNormalizer",
    "FactorNormalizer",
    "DictNormalizer",
    # Mixin classes
    "CallPeaksMixin",
    "FootprintMixin",
    "ClusteringMixin",
    "QuantileNormalizationMixin",
    "TSSExtractionMixin",
    "LibrarySizeAnnotationMixin",
    "GenomicMaskingMixin",
    # Zarr implementations
    "ZarrIO",
    "DenseZarrIO",
    "SignalDenseZarrIO",
    "SingleSparseZarrIO",
    "MultiSparseZarrIO",
    "MotifDenseZarrIO",
    "CelltypeDenseZarrIO",
    "open_zarr",
    # Metadata I/O
    "Metadata",
    # Peak I/O utilities
    "read_peaks",
    "prepare_peaks_for_processing",
    "reorder_matrix_columns",
    "ensure_peaks_match_matrix",
    "load_atac_io",
    "sanitize_column_name",
]

# load_atac_io requires caesar.io imports, so import it lazily from peak_utils
# This avoids circular imports since peak_utils imports from caesar.io at module level
def __getattr__(name: str):
    if name == "load_atac_io":
        from caesar.preprocess.utils.peak_utils import load_atac_io
        return load_atac_io
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
