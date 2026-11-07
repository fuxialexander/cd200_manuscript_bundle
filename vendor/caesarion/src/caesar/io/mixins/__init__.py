"""IO Mixins for Caesar.

This module provides various mixin classes that add functionality to IO classes.
"""
from __future__ import annotations

from .call_peaks import CallPeaksMixin
from .clustering import ClusteringMixin
from .footprint import FootprintMixin
from .genomic_masking import GenomicMaskingMixin
from .label import LabelMixin
from .library_size_annotation import LibrarySizeAnnotationMixin
from .quantile_normalization import QuantileNormalizationMixin
from .tss_extraction import TSSExtractionMixin
from .tss_tes_accessibility import (
    TSSAccessibilityMixin,
    TESAccessibilityMixin,
    TSSTESAccessibilityMixin,
)

__all__ = [
    "CallPeaksMixin",
    "ClusteringMixin",
    "FootprintMixin",
    "GenomicMaskingMixin",
    "LabelMixin",
    "LibrarySizeAnnotationMixin",
    "QuantileNormalizationMixin",
    "TSSExtractionMixin",
    "TSSAccessibilityMixin",
    "TESAccessibilityMixin",
    "TSSTESAccessibilityMixin",
]

