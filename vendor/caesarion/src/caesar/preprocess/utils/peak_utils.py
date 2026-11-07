"""Peak utilities for reading, ordering, and I/O operations.

This module provides shared logic for:
- Reading peak files from disk (BED, narrowPeak)
- Handling peak/tile ordering to ensure consistent behavior between Zarr and BPCells backends
- Loading ATAC data from different formats
- Utility functions for peak processing

Note: Most functions are now defined in caesar.io.__init__.py to avoid circular imports.
This module re-exports them for backward compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from caesar.io.interface import CelltypeDenseIO

logger = logging.getLogger(__name__)

# Import functions from caesar.io (defined there to avoid circular imports)
from caesar.io import (
    ensure_peaks_match_matrix,
    prepare_peaks_for_processing,
    read_peaks,
    reorder_matrix_columns,
    sanitize_column_name,
)


def load_atac_io(
    path: str,
    mode: str = "a",
    celltypes: Optional[list[str]] = None,
) -> CelltypeDenseIO:
    """Load ATAC data from either zarr or bpcells format.

    Args:
        path: Path to ATAC data (zarr file or bpcells directory)
        mode: File access mode ('r' for read-only, 'a' for read/write)
        celltypes: Optional list of celltype IDs to load

    Returns:
        CelltypeDenseIO instance (either CelltypeDenseZarrIO or CelltypeDenseBPCellsIO)

    Raises:
        ValueError: If the path cannot be loaded as either format
    """
    from caesar.io.zarr_io import CelltypeDenseZarrIO
    from caesar.io.bpcell_io import CelltypeDenseBPCellsIO
    from caesar.io.interface import CelltypeDenseIO

    path_obj = Path(path)

    # Detect input type
    if path.endswith(".zarr") or ".zarr" in str(path):
        io = CelltypeDenseZarrIO(path, mode=mode, celltype_ids=celltypes)
    elif "bpcells" in str(path).lower() or (
        path_obj.is_dir() and (path_obj / "attrs.json").exists()
    ):
        io = CelltypeDenseBPCellsIO(path, mode=mode, celltype_ids=celltypes)
    elif path_obj.suffix == ".zarr":
        io = CelltypeDenseZarrIO(path, mode=mode, celltype_ids=celltypes)
    else:
        # Try zarr first, then bpcells
        try:
            io = CelltypeDenseZarrIO(path, mode=mode, celltype_ids=celltypes)
        except Exception as e1:
            try:
                io = CelltypeDenseBPCellsIO(path, mode=mode, celltype_ids=celltypes)
            except Exception as e2:
                raise ValueError(
                    f"Could not load data as either zarr or bpcells format.\n"
                    f"Zarr error: {e1}\n"
                    f"BPCells error: {e2}\n"
                    f"Please ensure the path is correct and the format is supported."
                ) from e2

    return io
