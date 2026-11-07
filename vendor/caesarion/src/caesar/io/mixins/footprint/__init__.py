"""Footprinting functionality for TF binding analysis and ML feature extraction.

This module provides:
- Traditional TF footprinting analysis
- ML feature extraction from motif matches
- Numba-accelerated computational kernels
"""

from __future__ import annotations

from ._kernels import (
    _accumulate_fp_numba,
    _compute_prefix_sum,
    _compute_prefix_sum_from_sparse,
    _extract_and_aggregate_single_pass,
)
from ._ml_features import MLFeatureMixin
from ._traditional import TraditionalFootprintMixin

__all__ = [
    "FootprintMixin",
    "TraditionalFootprintMixin",
    "MLFeatureMixin",
    "_accumulate_fp_numba",
    "_compute_prefix_sum",
    "_compute_prefix_sum_from_sparse",
    "_extract_and_aggregate_single_pass",
]


class FootprintMixin(TraditionalFootprintMixin, MLFeatureMixin):
    """Combined footprinting functionality.

    This mixin provides both traditional TF footprinting and ML feature extraction
    capabilities for genomic data analysis.

    Attributes from TraditionalFootprintMixin:
        - get_genome_wide_footprint_all_motifs: Compute traditional footprints
        - _compute_footprint_in_region_for_all_motifs: Region-specific footprints
        - _compute_footprint_in_chunk_for_all_motifs: Chunk-based processing

    Attributes from MLFeatureMixin:
        - get_genome_wide_ml_features_prefix_sum: Extract ML-ready features
        - _compute_ml_features_prefix_sum: Region-specific ML features
        - _get_combined_track: Get combined track data
    """
    pass
