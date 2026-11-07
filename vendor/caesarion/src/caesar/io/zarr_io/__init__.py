"""Zarr I/O module for Caesar.

This module provides zarr-based storage backends for various genomic data types.
"""
from __future__ import annotations

import logging
from pathlib import Path

import zarr

# Base classes and constants
from .base import COMPRESSOR, ZarrIO

# Dense format classes
from .dense import (
    CelltypeDenseZarrIO,
    DenseZarrIO,
    MotifDenseZarrIO,
    SequenceDenseZarrIO,
    SignalDenseZarrIO,
)

# Motif hits storage
from .motif_hits import MotifHitZarrIO

# Multiome format classes
from .multiome import MultiomeDenseZarrIO, MultiomeIO, MultiomeSparseZarrIO

# Sparse format classes
from .sparse import MultiSparseZarrIO, SingleSparseZarrIO


def detect_zarr_type(zarr_path: str | Path, mode: str = "r") -> str:
    """Detect the type of zarr store.

    First checks .zattrs for 'zarr_type' key, then falls back to structural detection.

    Args:
        zarr_path: Path to the zarr store
        mode: Mode to open zarr store (default: "r")

    Returns:
        str: Type of zarr store - one of:
            - "celltype_dense": CelltypeDenseZarrIO format
            - "multiome_dense": MultiomeDenseZarrIO format
            - "dense": DenseZarrIO format
            - "single_sparse": SingleSparseZarrIO format
            - "multiome_sparse": MultiomeSparseZarrIO format
            - "multi_sparse": MultiSparseZarrIO format
            - "signal_dense": SignalDenseZarrIO format
            - "sequence_dense": SequenceDenseZarrIO format

    Raises:
        ValueError: If zarr structure is unrecognized
    """
    zarr_path = Path(zarr_path)

    # Check for MultiSparse structure BEFORE trying to open as zarr group
    # MultiSparse is a directory containing multiple zarr stores, not a single zarr group
    if zarr_path.is_dir():
        # Check if this directory contains multiple .zarr subdirectories
        zarr_subdirs = list(zarr_path.glob("*.zarr"))
        if zarr_subdirs:
            # This looks like a multi_sparse directory
            # Verify by checking if any subdirectory is a valid zarr store
            for zarr_subdir in zarr_subdirs[:1]:  # Check first one
                try:
                    test_z = zarr.open_group(str(zarr_subdir), mode="r")
                    if "chrs" in test_z:
                        return "multi_sparse"
                except Exception:
                    pass

    try:
        # Open as group to check structure
        z = zarr.open_group(str(zarr_path), mode=mode)

        # First, check .zattrs for explicit zarr_type (most reliable)
        if "zarr_type" in z.attrs:
            zarr_type = z.attrs["zarr_type"]
            logging.debug(f"Found zarr_type in .zattrs: {zarr_type}")
            return zarr_type

        # Check for SequenceDense structure (has sequences group)
        if "sequences" in z:
            return "sequence_dense"

        # Check for SignalDense structure (has signals group)
        if "signals" in z:
            return "signal_dense"

        # Check for CelltypeDense or MultiomeDense structure (has cell type groups with chrs inside)
        # Exclude common metadata keys
        non_metadata_keys = [
            k
            for k in z
            if k not in ["chrs", "metadata", "peaks"] and not k.startswith(".")
        ]
        if non_metadata_keys:
            # Check if any of these groups contain 'chrs'
            # Note: zarr.Group works in both v2 and v3
            has_celltype_structure = any(
                isinstance(z[key], zarr.Group) and "chrs" in z[key]
                for key in non_metadata_keys
            )

            if has_celltype_structure:
                # Check if it's a multiome dense (has expression data)
                has_expression = any(
                    isinstance(z[key], zarr.Group) and "expression" in z[key]
                    for key in non_metadata_keys
                )

                if has_expression:
                    return "multiome_dense"
                else:
                    return "celltype_dense"

        # Check for MultiomeSparse structure (has barcodes, chrs, and rna.h5ad)
        if "barcodes" in z and "chrs" in z:
            # Check if RNA expression file exists
            rna_path = zarr_path / "rna.h5ad"
            if rna_path.exists():
                return "multiome_sparse"
            return "single_sparse"

        # Check for regular Dense structure (just has chrs at root)
        if "chrs" in z:
            return "dense"

        # Check for MultiSparse structure (directory with multiple zarr stores)
        if "chrs" not in z and any(zarr_path.glob("*/*.zarr")):
            return "multi_sparse"

        # If still unclear, try name-based detection as fallback
        path_str = str(zarr_path).lower()
        if "multiomedense" in path_str or "multiome_dense" in path_str:
            return "multiome_dense"
        elif "celltypedense" in path_str:
            return "celltype_dense"
        elif "multiomesparse" in path_str or "multiome_sparse" in path_str:
            return "multiome_sparse"
        elif "multisparse" in path_str:
            return "multi_sparse"
        elif "singlesparse" in path_str:
            return "single_sparse"
        elif "signaldense" in path_str:
            return "signal_dense"
        elif "sequencedense" in path_str:
            return "sequence_dense"
        elif "dense" in path_str:
            return "dense"

        raise ValueError(f"Unrecognized zarr structure in {zarr_path}")

    except Exception as e:
        logging.error(f"Error detecting zarr type for {zarr_path}: {e}")
        raise


def open_zarr(zarr_path: str | Path, mode: str = "r", **kwargs):
    """Open a zarr store with automatic type detection.

    Args:
        zarr_path: Path to the zarr store
        mode: Mode to open zarr store (default: "r")
        **kwargs: Additional arguments passed to the IO class constructor

    Returns:
        Appropriate ZarrIO subclass instance

    Raises:
        ValueError: If zarr type cannot be determined
    """
    zarr_type = detect_zarr_type(zarr_path, mode="r")  # Always detect in read mode

    type_to_class = {
        "celltype_dense": CelltypeDenseZarrIO,
        "multiome_dense": MultiomeDenseZarrIO,
        "dense": DenseZarrIO,
        "single_sparse": SingleSparseZarrIO,
        "multiome_sparse": MultiomeSparseZarrIO,
        "multi_sparse": MultiSparseZarrIO,
        "signal_dense": SignalDenseZarrIO,
        "sequence_dense": SequenceDenseZarrIO,
    }

    io_class = type_to_class.get(zarr_type)
    if io_class is None:
        raise ValueError(f"Unknown zarr type: {zarr_type}")

    return io_class(str(zarr_path), mode=mode, **kwargs)


__all__ = [
    # Base classes and constants
    "ZarrIO",
    "COMPRESSOR",
    # Dense format classes
    "DenseZarrIO",
    "SignalDenseZarrIO",
    "MotifDenseZarrIO",
    "SequenceDenseZarrIO",
    "CelltypeDenseZarrIO",
    # Motif hits storage
    "MotifHitZarrIO",
    # Sparse format classes
    "SingleSparseZarrIO",
    "MultiSparseZarrIO",
    # Multiome format classes
    "MultiomeSparseZarrIO",
    "MultiomeDenseZarrIO",
    "MultiomeIO",
    # Utility functions
    "detect_zarr_type",
    "open_zarr",
]
