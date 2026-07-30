"""
Zarr-based storage system for deletion analysis results.

Enables efficient storage and retrieval of deletion analysis results,
avoiding repeated computation.
"""

import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import json
import logging

logger = logging.getLogger(__name__)


class DeletionResultCache:
    """
    Zarr-based cache for deletion analysis results.

    Storage structure:
        cache.zarr/
        ├── metadata/
        │   ├── variant_info (JSON metadata for each analysis)
        │   ├── output_heads (list of model output heads)
        │   └── analysis_params (deletion_size, stride, etc.)
        └── {variant_id}/
            ├── window_positions (int64 array)
            ├── deletion_effects/
            │   ├── {head_name} (float32 arrays)
            │   └── ...
            └── nucleotide_scores/  (optional)
                ├── {head_name} (float32 arrays)
                └── ...
    """

    def __init__(self, cache_path: str, mode: str = 'r'):
        """
        Initialize deletion result cache.

        Args:
            cache_path: Path to zarr cache directory
            mode: 'r' for read-only, 'a' for read-write, 'w' for create/overwrite
        """
        self.cache_path = Path(cache_path)
        self.mode = mode

        if mode == 'w':
            # Create new cache
            self.store = zarr.open_group(str(self.cache_path), mode='w')
            meta = self.store.create_group('metadata')
            meta.create_dataset('variant_info', shape=(0,), dtype=object,
                              object_codec=zarr.codecs.JSON())
        else:
            # Open existing cache
            if not self.cache_path.exists():
                raise FileNotFoundError(f"Cache not found: {cache_path}")
            self.store = zarr.open_group(str(self.cache_path), mode=mode)

    def _generate_variant_id(self, chromosome: str, position: int,
                           ref: str, alt: str) -> str:
        """Generate unique ID for variant."""
        return f"{chromosome}_{position}_{ref}_{alt}"

    def save_result(self, deletion_result: 'DeletionResult'):
        """
        Save deletion result to cache.

        Args:
            deletion_result: DeletionResult instance to cache
        """
        if self.mode == 'r':
            raise ValueError("Cannot write to cache in read-only mode")

        from .results import DeletionResult

        variant = deletion_result.variant
        variant_id = self._generate_variant_id(
            variant.chromosome,
            variant.position,
            variant.ref,
            variant.alt
        )

        # Create variant group
        if variant_id in self.store:
            del self.store[variant_id]
        var_group = self.store.create_group(variant_id)

        # Save metadata
        metadata = {
            'chromosome': variant.chromosome,
            'position': int(variant.position),
            'ref': variant.ref,
            'alt': variant.alt,
            'deletion_size': int(deletion_result.deletion_size),
            'stride': int(deletion_result.stride),
            'output_heads': deletion_result.output_heads,
            'region_start': int(deletion_result.region_start) if deletion_result.region_start else None,
            'region_end': int(deletion_result.region_end) if deletion_result.region_end else None,
            'has_nucleotide_scores': deletion_result.nucleotide_scores is not None
        }
        var_group.attrs.update(metadata)

        # Save window positions
        compressor = zarr.Blosc(cname='zstd', clevel=5)
        var_group.create_dataset('window_positions',
                                data=deletion_result.window_positions,
                                dtype=np.int64,
                                compressor=compressor)

        # Save deletion effects for each head
        effects_group = var_group.create_group('deletion_effects')
        for head_name, effects in deletion_result.deletion_effects.items():
            effects_group.create_dataset(head_name,
                                        data=effects,
                                        dtype=np.float32,
                                        compressor=compressor)

        # Save nucleotide scores if available
        if deletion_result.nucleotide_scores is not None:
            nuc_group = var_group.create_group('nucleotide_scores')
            for head_name, scores in deletion_result.nucleotide_scores.items():
                nuc_group.create_dataset(head_name,
                                        data=scores,
                                        dtype=np.float32,
                                        compressor=compressor)

        logger.info(f"Cached deletion result for {variant_id}")

    def load_result(self, chromosome: str, position: int,
                   ref: str, alt: str) -> Optional['DeletionResult']:
        """
        Load deletion result from cache.

        Args:
            chromosome: Chromosome name
            position: Variant position
            ref: Reference allele
            alt: Alternative allele

        Returns:
            DeletionResult if found, None otherwise
        """
        from .results import DeletionResult
        from .dataset import Variant

        variant_id = self._generate_variant_id(chromosome, position, ref, alt)

        if variant_id not in self.store:
            return None

        var_group = self.store[variant_id]
        metadata = dict(var_group.attrs)

        # Reconstruct variant
        variant = Variant(
            chromosome=metadata['chromosome'],
            position=metadata['position'],
            ref=metadata['ref'],
            alt=metadata['alt']
        )

        # Load deletion effects
        deletion_effects = {}
        for head_name in var_group['deletion_effects'].keys():
            deletion_effects[head_name] = var_group['deletion_effects'][head_name][:]

        # Load nucleotide scores if available
        nucleotide_scores = None
        if 'nucleotide_scores' in var_group:
            nucleotide_scores = {}
            for head_name in var_group['nucleotide_scores'].keys():
                nucleotide_scores[head_name] = var_group['nucleotide_scores'][head_name][:]

        # Reconstruct DeletionResult
        result = DeletionResult(
            variant=variant,
            deletion_effects=deletion_effects,
            window_positions=var_group['window_positions'][:],
            deletion_size=metadata['deletion_size'],
            stride=metadata['stride'],
            output_heads=metadata['output_heads'],
            nucleotide_scores=nucleotide_scores,
            region_start=metadata.get('region_start'),
            region_end=metadata.get('region_end')
        )

        logger.info(f"Loaded deletion result for {variant_id}")
        return result

    def has_result(self, chromosome: str, position: int,
                  ref: str, alt: str) -> bool:
        """Check if result exists in cache."""
        variant_id = self._generate_variant_id(chromosome, position, ref, alt)
        return variant_id in self.store

    def list_variants(self) -> List[Dict[str, any]]:
        """List all cached variants."""
        variants = []
        for variant_id in self.store.keys():
            if variant_id == 'metadata':
                continue
            var_group = self.store[variant_id]
            variants.append(dict(var_group.attrs))
        return variants

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        variants = self.list_variants()
        return {
            'n_variants': len(variants),
            'output_heads': list(set(h for v in variants for h in v.get('output_heads', []))),
            'total_size_mb': self._get_size_mb()
        }

    def _get_size_mb(self) -> float:
        """Calculate total cache size in MB."""
        total_bytes = 0
        for root, dirs, files in self.cache_path.walk():
            for file in files:
                total_bytes += (root / file).stat().st_size
        return total_bytes / (1024 ** 2)


def save_deletion_result(result: 'DeletionResult', output_path: str):
    """
    Convenience function to save a single deletion result.

    Args:
        result: DeletionResult to save
        output_path: Path for output zarr file

    Example:
        >>> save_deletion_result(del_result, 'my_analysis.zarr')
    """
    cache = DeletionResultCache(output_path, mode='w')
    cache.save_result(result)
    logger.info(f"Saved deletion result to {output_path}")


def load_deletion_result(cache_path: str, chromosome: str,
                        position: int, ref: str, alt: str) -> Optional['DeletionResult']:
    """
    Convenience function to load a single deletion result.

    Args:
        cache_path: Path to zarr cache
        chromosome: Chromosome name
        position: Variant position
        ref: Reference allele
        alt: Alternative allele

    Returns:
        DeletionResult if found, None otherwise

    Example:
        >>> result = load_deletion_result('cache.zarr', 'chr1', 1000000, 'A', 'G')
    """
    cache = DeletionResultCache(cache_path, mode='r')
    return cache.load_result(chromosome, position, ref, alt)


class BatchDeletionAnalyzer:
    """
    Helper class for batch deletion analysis with caching.

    Automatically caches results to avoid recomputation.
    """

    def __init__(self, analyzer: 'MutationAnalyzer', cache_path: Optional[str] = None):
        """
        Args:
            analyzer: MutationAnalyzer instance
            cache_path: Optional path to result cache
        """
        self.analyzer = analyzer
        self.cache = DeletionResultCache(cache_path, mode='a') if cache_path else None

    def analyze_variant(self, variant: 'Variant', **kwargs) -> 'DeletionResult':
        """
        Analyze variant with caching.

        Args:
            variant: Variant to analyze
            **kwargs: Arguments for sliding_window_deletion

        Returns:
            DeletionResult
        """
        # Check cache first
        if self.cache is not None:
            cached = self.cache.load_result(
                variant.chromosome,
                variant.position,
                variant.ref,
                variant.alt
            )
            if cached is not None:
                logger.info(f"Using cached result for {variant}")
                return cached

        # Run analysis
        logger.info(f"Running deletion analysis for {variant}")
        result = self.analyzer.sliding_window_deletion(variant, **kwargs)

        # Save to cache
        if self.cache is not None:
            self.cache.save_result(result)

        return result

    def analyze_batch(self, variants: List['Variant'], **kwargs) -> List['DeletionResult']:
        """
        Analyze multiple variants with caching.

        Args:
            variants: List of variants to analyze
            **kwargs: Arguments for sliding_window_deletion

        Returns:
            List of DeletionResults
        """
        results = []
        for variant in variants:
            result = self.analyze_variant(variant, **kwargs)
            results.append(result)
        return results
