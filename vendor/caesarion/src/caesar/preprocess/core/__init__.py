"""Caesar preprocessing core module.

This module provides the base infrastructure for all preprocessing operations.
"""

from __future__ import annotations

from .base import BaseProcessor
from .chromosome_filter import ChromosomeFilter, get_chromosomes_to_process
from .tile_filtering import (
    create_column_indices,
    filter_tiles,
    filter_tiles_by_blacklist,
    filter_tiles_by_chromosomes,
)

# ConfigManager and ConfigSchema removed - not needed with new config system
from .constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_FRAGMENTS,
    DEFAULT_MIN_CELLS,
    DEFAULT_MIN_FRAGMENTS,
    ZARR_COMPRESSOR,
)
from .exceptions import ProcessingError, ProcessorError, ValidationError
from .mixins import CompressorMixin, OpenZarrMixin
from .output_spec import (
    SPEC_VERSION,
    AggregationInfo,
    InputFileInfo,
    OutputManifest,
    OutputPaths,
    PipelineInfo,
    SourceDataInfo,
    UnitInfo,
    check_output_completeness,
    create_accessibility_profiles_zarr,
    create_tile_counts_h5ad,
    generate_manifest,
    get_optional_components,
    get_required_components,
    validate_pseudobulk_output,
)

__all__ = [
    # Base
    "BaseProcessor",
    # Mixins
    "OpenZarrMixin",
    "CompressorMixin",
    # Constants
    "ZARR_COMPRESSOR",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MIN_FRAGMENTS",
    "DEFAULT_MAX_FRAGMENTS",
    "DEFAULT_MIN_CELLS",
    # Exceptions
    "ProcessorError",
    "ValidationError",
    "ProcessingError",
    # Chromosome filter
    "ChromosomeFilter",
    "get_chromosomes_to_process",
    # Tile filtering
    "filter_tiles",
    "filter_tiles_by_chromosomes",
    "filter_tiles_by_blacklist",
    "create_column_indices",
    # Output specification
    "SPEC_VERSION",
    "OutputManifest",
    "PipelineInfo",
    "AggregationInfo",
    "SourceDataInfo",
    "OutputPaths",
    "UnitInfo",
    "InputFileInfo",
    "validate_pseudobulk_output",
    "generate_manifest",
    "check_output_completeness",
    "get_required_components",
    "get_optional_components",
    "create_tile_counts_h5ad",
    "create_accessibility_profiles_zarr",
]
