"""Common constants for preprocessing."""

from __future__ import annotations

from zarr.codecs import BloscCodec

# Standard compressor for all zarr operations
ZARR_COMPRESSOR = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")

# Default chunk sizes
DEFAULT_CHUNK_SIZE = 2_000_000
DEFAULT_TILE_SIZE = 500

# Default thresholds
DEFAULT_MIN_FRAGMENTS = 1000
DEFAULT_MAX_FRAGMENTS = 50000
DEFAULT_MIN_CELLS = 100

# Processing constants
DEFAULT_BATCH_SIZE = 100_000
DEFAULT_THREADS = 1

# Peak calling defaults
DEFAULT_MIN_PEAK_WIDTH = 200
DEFAULT_MAX_PEAK_WIDTH = 10000
DEFAULT_PEAK_CUTOFF = 0.05

# Clustering defaults
DEFAULT_N_NEIGHBORS = 30
DEFAULT_MIN_DIST = 0.3
DEFAULT_N_COMPONENTS = 50

# Library size defaults
DEFAULT_CHROMOSOMES_REGEX = r"^chr[0-9XY]+$"
