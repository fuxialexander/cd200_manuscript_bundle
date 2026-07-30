"""
Mutation Analysis Framework

A unified, model-agnostic framework for genomic variant effect prediction.

This package provides:
- Simple model loading from PyTorch Lightning checkpoints
- Flexible dataset management with VCF/DataFrame support
- Comprehensive mutation analysis (variants, ISM, deletions)
- Type-safe result objects with built-in visualization
- Motif scanning result caching with zarr storage
- Modular TF filtering based on expression/DE
- Distributed nucleotide-level scoring for deletions
- Result caching to avoid recomputation

Example:
    >>> from mutation_analysis import load_model, MutationDataset, MutationAnalyzer
    >>> from caesar.io.zarr_io import SequenceDenseZarrIO
    >>>
    >>> # Load model
    >>> model, heads = load_model("model.ckpt")
    >>> print(f"Output heads: {heads}")
    >>>
    >>> # Load variants
    >>> genome = SequenceDenseZarrIO("hg38.zarr", mode='r')
    >>> dataset = MutationDataset.from_vcf("variants.vcf", genome)
    >>>
    >>> # Analyze with distributed scoring
    >>> analyzer = MutationAnalyzer(model, heads, dataset)
    >>> result = analyzer.sliding_window_deletion(dataset[0], distribute_effects=True)
    >>>
    >>> # Visualize with motif overlay
    >>> from mutation_analysis.motif_filters import create_annotation_filter
    >>> filt = create_annotation_filter('annot.csv', 'expressed', min_expression=10)
    >>> result.plot_deletion_effects('regression',
    ...     motif_cache='motifs.zarr', motif_filter=filt,
    ...     use_nucleotide_scores=True)
"""

__version__ = "0.1.0"
__author__ = "Development Team"

# Model loading
from .model import load_model, get_model_info

# Data classes
from .dataset import Variant, MutationDataset

# Analysis
from .analyzer import MutationAnalyzer

# Results
from .results import (
    BaseResult,
    VariantResult,
    ISMResult,
    DeletionResult,
)

# Utilities
from .metrics import calculate_effects

# Motif handling (optional imports - not required for basic usage)
try:
    from .motif_cache import MotifCache, MotifScanResult, create_cache_from_fimo
    from .motif_filters import (
        MotifFilter, PassThroughFilter, WhitelistFilter, BlacklistFilter,
        ExpressionFilter, MultiConditionExpressionFilter,
        DifferentialExpressionFilter, MultiConditionDEFilter,
        CustomFilter, CompositeFilter, create_annotation_filter
    )
    from .result_storage import (
        DeletionResultCache, save_deletion_result, load_deletion_result,
        BatchDeletionAnalyzer
    )
    _MOTIF_SUPPORT = True
except ImportError:
    _MOTIF_SUPPORT = False

# Phase 3 model support (optional)
try:
    from .phase3_loader import Phase3ModelLoader, load_phase3_model
    _PHASE3_SUPPORT = True
except ImportError:
    _PHASE3_SUPPORT = False

__all__ = [
    # Model loading
    'load_model',
    'get_model_info',

    # Data classes
    'Variant',
    'MutationDataset',

    # Analysis
    'MutationAnalyzer',

    # Results
    'BaseResult',
    'VariantResult',
    'ISMResult',
    'DeletionResult',

    # Utilities
    'calculate_effects',
]

# Add motif handling to __all__ if available
if _MOTIF_SUPPORT:
    __all__.extend([
        # Motif caching
        'MotifCache',
        'MotifScanResult',
        'create_cache_from_fimo',

        # Motif filtering
        'MotifFilter',
        'create_annotation_filter',

        # Result storage
        'DeletionResultCache',
        'save_deletion_result',
        'load_deletion_result',
        'BatchDeletionAnalyzer',
    ])

# Add Phase 3 support to __all__ if available
if _PHASE3_SUPPORT:
    __all__.extend([
        'Phase3ModelLoader',
        'load_phase3_model',
    ])