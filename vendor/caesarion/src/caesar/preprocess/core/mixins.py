"""Common mixins for preprocessing components."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class OpenZarrMixin:
    """Mixin for zarr I/O operations."""

    def _open_zarr(self, zarr_path: str | Path, mode: str = "a"):
        """Open zarr store with automatic type detection.

        Args:
            zarr_path: Path to zarr store
            mode: Open mode (default: "a" for append)

        Returns:
            Opened zarr store
        """
        from caesar.io.zarr_io import open_zarr

        try:
            return open_zarr(zarr_path, mode=mode)
        except Exception as e:
            logging.warning(f"Error opening zarr with auto-detection: {e}")
            # Fallback to SingleSparseZarrIO
            from caesar.io.zarr_io import SingleSparseZarrIO

            return SingleSparseZarrIO(str(zarr_path), mode=mode)


class CompressorMixin:
    """Mixin for zarr compression settings."""

    @property
    def compressor(self):
        """Get standard zarr compressor."""
        if not hasattr(self, "_compressor"):
            from zarr.codecs import BloscCodec

            self._compressor = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")
        return self._compressor

    @property
    def default_chunks(self):
        """Get default chunk configuration."""
        if not hasattr(self, "_default_chunks"):
            from .constants import DEFAULT_CHUNK_SIZE

            self._default_chunks = DEFAULT_CHUNK_SIZE
        return self._default_chunks
