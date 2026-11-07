"""Utilities for I/O operations.

This package provides backwards compatibility for imports from caesar.io.utils.
Peak-related functionality is now defined in caesar.io.__init__.py.
"""

from caesar.io import (
    ensure_peaks_match_matrix,
    load_atac_io,
    prepare_peaks_for_processing,
    read_peaks,
    reorder_matrix_columns,
    sanitize_column_name,
)

__all__ = [
    "read_peaks",
    "prepare_peaks_for_processing",
    "reorder_matrix_columns",
    "ensure_peaks_match_matrix",
    "load_atac_io",
    "sanitize_column_name",
]
