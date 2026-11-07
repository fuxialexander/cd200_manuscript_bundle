"""Backward compatibility facade for footprinting functionality.

This module provides backward compatibility for code importing from the old location.
All functionality has been moved to the footprint/ submodule for better organization.

New code should import from: caesar.io.mixins.footprint
"""

from __future__ import annotations

# Re-export everything from the new module structure
from .footprint import (
    FootprintMixin,
    MLFeatureMixin,
    TraditionalFootprintMixin,
    _accumulate_fp_numba,
    _compute_prefix_sum,
    _compute_prefix_sum_from_sparse,
    _extract_and_aggregate_single_pass,
)

__all__ = [
    "FootprintMixin",
    "TraditionalFootprintMixin",
    "MLFeatureMixin",
    "_accumulate_fp_numba",
    "_compute_prefix_sum",
    "_compute_prefix_sum_from_sparse",
    "_extract_and_aggregate_single_pass",
]
