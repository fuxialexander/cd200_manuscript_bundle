"""Unified OmegaConf configuration for Caesar preprocessing.

This module provides a simplified, structured configuration system using OmegaConf
with proper type safety and validation. The configuration hierarchy is designed
to be intuitive and reduce complexity.

Key principles:
1. Single source of truth for all configuration parameters
2. Type-safe with validation
3. Clear hierarchy matching conceptual organization
4. Direct attribute access without complex mappings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


# Data input configurations
@dataclass
class BAMFile:
    """Configuration for a single BAM file."""

    path: str = MISSING
    id: str = MISSING
    celltype: str = MISSING


@dataclass
class FragmentFile:
    """Configuration for a single fragment file.

    For CUT&RUN experiments, additional fields support antibody target tracking
    and spike-in normalization. The spike_in_factor is typically calculated from
    spike-in read counts (e.g., E. coli DNA fragments).
    """

    path: str = MISSING
    id: str | None = None  # Optional - will be extracted from filename if not provided
    celltype: str | None = None  # Optional - used as fallback label when metadata not available
    metadata: str | None = None  # Path to metadata CSV/TSV file
    metadata_column: str | None = None  # Column name for cell type labels
    # CUT&RUN-specific metadata
    assay_type: str | None = None  # "atac", "cutandrun", "cutnrun", "chip" - auto-detected if None
    antibody_target: str | None = None  # Target protein for CUT&RUN (e.g., "H3K27ac", "CTCF")
    spike_in_factor: float | None = None  # Spike-in normalization factor (pre-calculated)


@dataclass
class DataConfig:
    """Data input configuration."""

    bam: Any | None = None  # BAMFile or List[BAMFile]
    fragments: Any | None = None  # FragmentFile or List[FragmentFile]
    bigwig: Any | None = None  # BigWigFile or List[BigWigFile]
    input: str | None = None  # For workflows that take zarr input
    metadata: str | None = None  # Cell metadata file
    rna: list[str] | None = None  # RNA expression files for multiome

    def normalize_inputs(self):
        """Normalize single files to lists.

        This method should be called after config loading to ensure
        consistent list-based access to file inputs.
        """
        # Convert single BAMFile to list
        if self.bam is not None and not isinstance(self.bam, list):
            self.bam = [self.bam]

        # Convert single FragmentFile to list
        if self.fragments is not None and not isinstance(self.fragments, list):
            self.fragments = [self.fragments]

        # Convert single BigWigFile to list
        if self.bigwig is not None and not isinstance(self.bigwig, list):
            self.bigwig = [self.bigwig]


@dataclass
class OutputConfig:
    """Output configuration."""

    dir: str = "./output"
    format: str = "zarr"
    backend: str = "zarr"  # zarr or bpcells
    compression: str = "zstd"


@dataclass
class ResourcesConfig:
    """Computational resources configuration."""

    n_jobs: int = 1
    memory_limit: str = "16G"
    temp_dir: str = "/tmp"


# Processor-specific configurations
@dataclass
class FragmentConfig:
    """Fragment processing configuration.

    CUT&RUN Support:
    ---------------
    For CUT&RUN experiments, set assay_type="cutandrun" to enable:
    - Fragment size validation (CUT&RUN typically has smaller fragments)
    - Spike-in chromosome filtering (excludes spike-in chromosomes from analysis)
    - Storage of antibody target and normalization factors in zarr metadata

    Spike-in normalization:
    - spike_in_chroms: List of spike-in chromosome names to exclude from main analysis
    - spike_in_factor_file: JSON file with pre-calculated normalization factors per sample
    - Spike-in factors are stored in zarr but NOT applied during processing
    """

    chunk_size: int = 2_000_000
    tile_size: int = 500
    min_fragments: int = 1000
    max_fragments: int = 50000
    min_cells: int = 100
    max_fragments_per_cell: int = 500
    barcode_tag: str = "CB"
    peak_file: str | None = None
    overwrite: bool = True  # Overwrite existing output (inverse of old skip_exist)
    dual_output: bool = False  # Output both sparse and dense formats
    output_formats: list[str] = field(
        default_factory=lambda: ["sparse"]
    )  # List of output formats
    use_metadata_filter: bool = False  # Whether to apply metadata filtering
    # Metadata column names (for unified metadata handling)
    barcode_col: str = "barcode"  # Column name for cell barcodes in metadata
    sample_id_col: str = "sample_id"  # Column name for sample IDs in metadata
    # Default output paths relative to output.dir
    output: str = "single_sparse.zarr"  # Default output for single-cell
    bulk_output: str = "celltype_dense.zarr"  # Default output for bulk

    # CUT&RUN-specific settings
    assay_type: str = "auto"  # "auto", "atac", "cutandrun", "cutnrun", "chip"
    spike_in_chroms: list[str] | None = None  # Spike-in chromosomes to filter (e.g., ["chrEcoli"])
    spike_in_factor_file: str | None = None  # JSON file with normalization factors per sample
    validate_fragment_sizes: bool = True  # Validate fragment size distributions by assay type
    expected_fragment_size_range: tuple[int, int] | None = None  # Override (min, max) fragment sizes

    def __post_init__(self):
        """Validate fragment configuration."""
        if self.min_fragments <= 0:
            raise ValueError("min_fragments must be greater than 0")
        if self.max_fragments <= self.min_fragments:
            raise ValueError("max_fragments must be greater than min_fragments")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be greater than 0")
        if self.min_cells <= 0:
            raise ValueError("min_cells must be greater than 0")
        # Validate assay_type
        valid_assay_types = ["auto", "atac", "cutandrun", "cutnrun", "chip"]
        if self.assay_type not in valid_assay_types:
            raise ValueError(
                f"assay_type must be one of {valid_assay_types}, got '{self.assay_type}'"
            )
        # Validate expected_fragment_size_range if provided
        if self.expected_fragment_size_range is not None:
            if len(self.expected_fragment_size_range) != 2:
                raise ValueError("expected_fragment_size_range must be a tuple of (min, max)")
            min_size, max_size = self.expected_fragment_size_range
            if min_size < 0 or max_size < 0:
                raise ValueError("expected_fragment_size_range values must be non-negative")
            if min_size >= max_size:
                raise ValueError("expected_fragment_size_range min must be less than max")


@dataclass
class BAMConfig:
    """BAM processing configuration."""

    single_end: bool = True
    signal_mode: str = "tn5"  # "tn5" for ATAC-seq/CUT&Tag, "chip" for ChIP-seq
    tn5_shift: bool = True  # Deprecated: use signal_mode instead. Kept for backwards compatibility.
    n_jobs: int = 16
    chunk_size: int = 2_000_000
    bin_size: int = 100  # Added bin_size field
    backend: str = "zarr"  # "zarr" or "bpcells" - backend to use for processing
    # Default output path relative to output.dir
    output: str = "celltype_dense.zarr"

    def __post_init__(self):
        """Validate BAM configuration."""
        if self.bin_size <= 0:
            raise ValueError("bin_size must be greater than 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be greater than 0")
        if self.backend not in ("zarr", "bpcells"):
            raise ValueError(f"backend must be 'zarr' or 'bpcells', got '{self.backend}'")
        if self.signal_mode not in ("tn5", "chip"):
            raise ValueError(f"signal_mode must be 'tn5' or 'chip', got '{self.signal_mode}'")


@dataclass
class BigWigConfig:
    """BigWig processing configuration."""

    chunk_size: int = 2_000_000
    n_jobs: int = 4
    # Default output path relative to output.dir
    output: str = "signal_dense.zarr"

    def __post_init__(self):
        """Validate BigWig configuration."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be greater than 0")


@dataclass
class ClusteringConfig:
    """Clustering configuration."""

    enabled: bool = True
    method: str = "lsi_leiden"
    n_components: int = 30
    resolution: float = 0.8
    device: str = "auto"
    blacklist_filtering: bool = False
    tile_size: int = 500
    total_features: int = 100000
    overwrite: bool = False
    min_genes: int = 200  # Added min_genes field
    n_neighbors: int = 30  # Added n_neighbors field

    def __post_init__(self):
        """Validate clustering configuration."""
        if self.min_genes <= 0:
            raise ValueError("min_genes must be greater than 0")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than 0")
        if self.n_components <= 0:
            raise ValueError("n_components must be greater than 0")
        if self.resolution <= 0:
            raise ValueError("resolution must be greater than 0")


@dataclass
class PseudobulkConfig:
    """Pseudobulk/aggregation configuration."""

    enabled: bool = True  # Whether pseudobulk aggregation is enabled
    groupby: str = "labels"  # Changed default from "clusters" to "labels"
    min_cells_per_group: int = 100
    min_libsize: int = 3_000_000
    overwrite: bool = False  # Overwrite existing pseudobulk output
    source: str | None = None  # For reprocessing workflows
    grouping_file: str | None = None  # Custom grouping file

    # Metadata handling
    metadata_column: str | None = (
        None  # Column to use from zarr metadata (None = use stored labels)
    )
    external_metadata: str | None = None  # External metadata file for regrouping
    barcode_column: str = "barcode"  # Column name for barcodes in external metadata

    # Sampling strategy fields
    sampling_strategy: str | None = None  # None, 'cell_count', or 'insertion_count'
    n_cells_per_group: int | None = None  # For cell_count strategy
    target_insertions: int | None = None  # For insertion_count strategy
    integrated: bool = False  # For multi-sample integrated processing

    # Expression data integration
    expression_data: str | None = None  # Path to scRNA-seq data (h5ad/h5/mtx)
    expression_groupby: str = "labels"  # Column name for expression grouping

    # Default I/O paths relative to output.dir
    input: str = "single_sparse.zarr"
    output: str = "celltype_dense.zarr"


@dataclass
class PeakCallingConfig:
    """Peak calling configuration."""

    enabled: bool = True  # Whether peak calling is enabled
    method: str = "caesar"
    qvalue: float = 0.05
    overwrite: bool = False  # Overwrite existing peak calls
    # Default input path relative to output.dir
    input: str = "celltype_dense.zarr"
    # Summit extraction options
    extract_summits: bool = True  # Whether to extract summits after peak calling
    summit_threshold: float = 0  # Minimum value threshold for summits
    summit_return_maxval: bool = True  # Include maximum values
    summit_return_sum: bool = True  # Include sum of values
    summit_recenter: bool = False  # Recenter peaks on summits
    # Background generation options
    generate_background: bool = False  # Whether to generate background regions
    background_tile_size: int = 2048  # Size of background tiles
    background_extend_bp: int = 2048  # Extension for overlap checking
    background_min_sum_value: float = 50  # Minimum SumValue for filteringwil
    background_max_quantile: float = 0.99  # Maximum quantile for filtering
    background_n_regions: int = 30000  # Maximum number of background regions
    background_reference_celltype: str | None = None  # Reference celltype for scoring
    # Peak classification options
    classify_peaks: bool = False  # Whether to classify peaks
    classification_min_sum_value: float = 100  # Minimum SumValue for filtering
    classification_min_max_value: float = 5  # Minimum MaxValue for 'for_profile'


@dataclass
class LibrarySizeConfig:
    """Library size annotation configuration."""

    enabled: bool = True  # Whether library size annotation is enabled
    n_jobs: int = 16
    overwrite: bool = False
    # Default input path relative to output.dir
    input: str = "celltype_dense.zarr"


@dataclass
class NormalizationConfig:
    """Normalization configuration."""

    enabled: bool = False
    method: str = "quantile"
    peak_file: str | None = None
    reference: str = "peaks"
    overwrite: bool = False  # Overwrite existing normalization params
    # Default input path relative to output.dir
    input: str = "celltype_dense.zarr"
    # Whether to write normalized data to a new file (default: False)
    # For BPCells: always False (only stores params, preserves raw counts)
    # For Zarr: if False, only stores params in attrs; if True, writes normalized.zarr
    write_output: bool = False


@dataclass
class FootprintConfig:
    """Footprint analysis configuration."""

    enabled: bool = True
    mode: str = "ml_features"  # "footprint" or "ml_features" (ml_features is more memory-efficient)
    motif_zarr: str | None = None
    cutoff_range: list[int] = field(default_factory=lambda: [6, 7])
    extend: int = 400
    n_processes: int = 32
    overwrite: bool = False  # Overwrite existing footprint results
    # ML feature extraction parameters (used when mode="ml_features")
    core_extend: int = 50  # Core window size (±bp) for ML features
    context_extend: int = 500  # Context window size (±bp) for ML features
    n_bins: int = 20  # Number of bins for shape reconstruction
    shift_offsets: list[int] | None = None  # Co-accessibility offset positions, defaults to [10, 20, 30]
    aggregate_by_motif: bool = True  # If True, returns averaged [CellType, Motif, Features]
    check_convergence_every: int = 10  # Check convergence every N chunks
    convergence_threshold: float = 1e-5  # Threshold for early stopping convergence
    # Custom motif database support
    motif_database_type: str = "zarr"  # Currently only zarr format is supported
    motif_scan_on_demand: bool = False  # Run motif scanning if motif_zarr doesn't exist
    motif_scan_config: dict | None = None  # Config for on-demand scanning
    # Visualization options
    visualize: bool = False
    visualization_output_dir: str | None = None
    motifs_to_visualize: list[str] | None = (
        None  # Specific motifs to visualize, None = all
    )
    # Default input path relative to output.dir
    input: str = "celltype_dense.zarr"


@dataclass
class TSSAccessibilityConfig:
    """TSS accessibility analysis configuration.

    Input can be either:
    - AnnData tile matrix (.h5ad) for fast tile-based extraction
    - Dense zarr store (.zarr) for high-resolution extraction

    The method is auto-determined based on input type.

    Output format:
    - For h5ad input: AnnData with obs=cells/groups, var=TSS sites with annotations
    - For zarr input: Dataset stored in the input zarr store

    Features:
    - groupby: Aggregate by cell type/cluster (h5ad input only)
    - all_transcripts: Include TSS from all transcripts for alternative promoter analysis
    """

    enabled: bool = True
    upstream: int = 2000  # Base pairs upstream of TSS
    downstream: int = 2000  # Base pairs downstream of TSS
    n_jobs: int = 16  # Number of parallel jobs (for high-res mode)
    overwrite: bool = False  # Overwrite existing results
    aggregate: bool = True  # Aggregate across genes (mean profile) - deprecated
    gene_list_file: str | None = None  # Path to file with gene list (one per line)
    gene_set: str | None = None  # Predefined gene set: "protein_coding", "all", or custom
    assembly: str | None = None  # Genome assembly (hg38, mm10, etc.) - auto-detected if None
    gencode_version: str | None = None  # Gencode version (e.g., "v44") - uses latest if None
    dataset_name: str = "tss_accessibility"  # Name of output dataset
    tile_size: int = 500  # Size of tiles in base pairs (for tile-based mode)
    # Default input path relative to output.dir (can be .h5ad or .zarr)
    input: str = "celltype_dense.zarr"
    # New options for enhanced output
    groupby: str | None = None  # Column in obs to aggregate by (e.g., "celltype", "cluster")
    all_transcripts: bool = False  # Include TSS from all transcripts, not just canonical


@dataclass
class TESAccessibilityConfig:
    """TES accessibility analysis configuration.

    Input can be either:
    - AnnData tile matrix (.h5ad) for fast tile-based extraction
    - Dense zarr store (.zarr) for high-resolution extraction

    The method is auto-determined based on input type.

    Output format:
    - For h5ad input: AnnData with obs=cells/groups, var=TES sites with annotations
    - For zarr input: Dataset stored in the input zarr store

    Features:
    - groupby: Aggregate by cell type/cluster (h5ad input only)
    - all_transcripts: Include TES from all transcripts for alternative termination analysis
    """

    enabled: bool = True
    upstream: int = 2000  # Base pairs upstream of TES
    downstream: int = 2000  # Base pairs downstream of TES
    n_jobs: int = 16  # Number of parallel jobs (for high-res mode)
    overwrite: bool = False  # Overwrite existing results
    aggregate: bool = True  # Aggregate across genes (mean profile) - deprecated
    gene_list_file: str | None = None  # Path to file with gene list (one per line)
    gene_set: str | None = None  # Predefined gene set: "protein_coding", "all", or custom
    assembly: str | None = None  # Genome assembly (hg38, mm10, etc.) - auto-detected if None
    gencode_version: str | None = None  # Gencode version (e.g., "v44") - uses latest if None
    dataset_name: str = "tes_accessibility"  # Name of output dataset
    tile_size: int = 500  # Size of tiles in base pairs (for tile-based mode)
    # Default input path relative to output.dir (can be .h5ad or .zarr)
    input: str = "celltype_dense.zarr"
    # New options for enhanced output
    groupby: str | None = None  # Column in obs to aggregate by (e.g., "celltype", "cluster")
    all_transcripts: bool = False  # Include TES from all transcripts, not just canonical


@dataclass
class BPCellsConfig:
    """BPCells processor configuration."""

    enabled: bool = True  # Whether BPCells processing is enabled
    output_type: str = "single_sparse"  # single_sparse, multi_sparse, celltype_dense, signal_dense, anndata, fragments
    shift_end: int = 1  # Standard ATAC-seq Tn5 shift
    threads: int | None = None  # None means use resources.n_jobs
    min_fragments: int = 1000  # Minimum fragments per cell
    max_fragments: int = 50000  # Maximum fragments per cell
    bin_size: int = 500  # Bin size in base pairs for binned matrices (used for anndata output)
    group_by: str = "celltype"  # For celltype_dense output
    use_clustering_labels: bool = False  # Use clustering labels for grouping (when available)
    save_metadata: bool = True  # Save cell metadata
    normalization_factor_file: str | None = None  # Store normalization factors
    overwrite: bool = False  # Overwrite existing output
    # Default output path relative to output.dir (varies by output_type)
    output: str = "bpcells"  # Will be suffixed with .bpcells
    # Metadata column flexibility
    barcode_col: str = "barcode"  # Column name for cell barcodes in metadata
    annotation_col: str = "celltype"  # Column name for cell annotations (celltype/annotation)
    sample_id_col: str = "sample_id"  # Column name for sample IDs in metadata (for non-unique barcodes)
    # Cluster mapping and filtering
    cluster_mapping_file: str | None = None  # JSON file mapping fine-grained to broad clusters
    target_groups: list[str] | None = None  # Filter to only process specific groups
    # Input format detection
    input_format: str = "auto"  # "10x", "bpcells", or "auto" (auto-detect)
    # Cleanup behavior
    cleanup_temp_dirs: bool = False  # Whether to remove temporary BPCells fragment directories after processing


@dataclass
class AddPeakConfig:
    """Add peak processor configuration."""

    peak_files: list[str] = field(
        default_factory=list
    )  # List of peak file paths/patterns
    mapping_pattern: str = (
        "{celltype}_peaks.narrowPeak"  # Pattern to extract metadata from filenames
    )
    merge_strategy: str = "union"  # union or intersection
    min_peak_count: int = 10  # Minimum count threshold for filtering
    normalize: bool = True  # Calculate aTPM normalization
    peak_name: str = "external_peaks"  # Name for storing in zarr
    columns_to_include: list[str] = field(
        default_factory=lambda: ["Start", "End"]
    )  # Columns to store from peak files


@dataclass
class AddExpressionConfig:
    """Add expression processor configuration."""

    expression_data: str = MISSING  # Path to scRNA-seq data (h5ad/h5/mtx)
    gene_annotations: str | None = (
        None  # Path to gene annotations (GTF/GFF) - optional if using Gencode
    )
    input_zarr: str | None = (
        None  # Input CelltypeDenseZarr path (can be set by workflow)
    )
    assembly: str | None = None  # Genome assembly (e.g., "hg38", "mm10") - for Gencode
    gencode_version: str | None = None  # Gencode version (e.g., "v44") - optional
    tss_window: int = 1000  # bp around TSS to consider for overlap
    min_cells_per_type: int = 100  # Minimum cells per cell type
    normalization_method: str = "tpm"  # tpm or cpm
    peak_name: str = "expression_peaks"  # Base name for storing peaks in zarr
    labels_column: str = "labels"  # Column name for cell type labels in adata.obs


@dataclass
class NonRedundantMotifConfig:
    """Non-redundant motif processor configuration."""

    motif_scan_results: str = MISSING  # Path to tabix-indexed BED.gz file
    motif_database: str = "vierstra_v2.0"  # Version of non-redundant set
    stranded: bool = True  # Store + and - strand motifs separately
    score_threshold: float = 5.0  # Minimum motif score to include
    chunk_size: int = 10_000_000  # Chunk size for processing
    # Default output path relative to output.dir
    output: str = "motif_scores.zarr"


@dataclass
class NonRedundantMotifCurationConfig:
    """Non-redundant motif curation processor configuration."""

    motif_zarr: str = (
        MISSING  # Path to motif scores zarr from NonRedundantMotifProcessor
    )
    genome_fasta: str = (
        MISSING  # Path to genome sequence zarr (SequenceDenseZarrIO format)
    )
    n_positive_samples: int = 100000  # Number of positive samples to extract
    n_negative_samples: int = 100000  # Number of negative samples to extract
    window_size: int = 200  # Window size around motif center (must be even)
    sampling_strategy: str = "balanced"  # balanced or weighted
    score_ranges: list[list[float]] = field(
        default_factory=lambda: [[5, 6], [6, 8], [8, 10], [10, 15], [15, 25]]
    )  # Score ranges for stratified sampling (each range is [min, max])
    samples_per_motif_per_range: int = 10  # Samples per motif per score range
    # Default output path relative to output.dir
    output: str = "motif_training_data.zarr"


@dataclass
class MultiomeRNAConfig:
    """Configuration for RNA processing in multiome context."""

    # Input validation
    min_genes: int = 200
    min_cells: int = 3
    max_genes: int = 10000
    min_counts: int = 500
    max_counts: int = 50000
    percent_mito: float = 20.0

    # Processing options
    normalization: str = "log1p"  # raw, cpm, tpm, log1p
    n_top_genes: int = 2000
    calculate_hvg: bool = True

    # Format-specific options
    validate_only: bool = False  # For h5ad files
    force_renormalize: bool = False
    use_gene_symbols: bool = True
    var_names: str = "gene_symbols"
    cache: bool = True

    # Advanced options
    mito_prefix: str = "MT-"  # Human: "MT-", Mouse: "mt-"
    detect_doublets: bool = False
    score_cell_cycle: bool = False
    remove_ambient: bool = False
    compute_embedding: bool = False  # Compute PCA/UMAP for QC

    # Additional QC thresholds (optional)
    max_percent_ribo: float | None = None  # e.g., 30.0
    max_percent_hb: float | None = None  # e.g., 5.0

    # Batch correction
    batch_key: str | None = None
    batch_correction: str = "none"  # none, harmony, combat, scanorama

    # Output options
    save_individual: bool = True
    save_merged: bool = True
    # Note: We always save raw data for multiome - needed for pseudobulk expression


@dataclass
class MultiomeConfig:
    """Configuration for multiome processing."""

    # ATAC configuration (delegates to FragmentConfig)
    atac: FragmentConfig = field(default_factory=FragmentConfig)

    # RNA configuration (delegates to MultiomeRNAConfig)
    rna: MultiomeRNAConfig = field(default_factory=MultiomeRNAConfig)

    # Barcode matching
    min_shared_cells: int = 100
    barcode_suffix: str | None = None  # e.g., "-1" for 10x data
    barcode_matching: str = "exact"  # exact, suffix, prefix

    # Output options
    compute_cross_modality_qc: bool = True
    save_intermediate: bool = False

    # Storage backend for ATAC data (zarr or bpcells)
    backend: str = "zarr"

    # Default output path relative to output.dir
    output: str = "multiome_sparse.zarr"


@dataclass
class MultiomePseudobulkConfig:
    """Configuration for multiome pseudobulk processing.

    Extends PseudobulkConfig functionality with RNA-specific options
    for creating pseudobulked multiome data.
    """

    # Inherit common pseudobulk settings
    groupby: str = "labels"  # Grouping variable (labels, clusters, etc.)
    min_cells_per_group: int = 100
    min_libsize: int = 3_000_000  # Minimum ATAC library size

    # RNA-specific aggregation settings
    expression_aggregation: str = "sum"  # sum, mean, or median
    min_cells_for_expression: int = 10  # Minimum cells for RNA aggregation

    # Default I/O paths relative to output.dir
    input: str = "multiome_sparse.zarr"
    output: str = "multiome_dense.zarr"


@dataclass
class CellStateTileCountConfig:
    """Configuration for generating full tile count matrix at cell-state level.

    This processor creates a complete tile count matrix aggregated at the cell-state
    level (cluster, metacell, or any custom grouping), immediately after cell-state
    assignment and BEFORE variable-tile shrinking.

    The output is a compact h5ad where:
    - .obs = cell-state metadata (cluster/metacell IDs, library sizes, etc.)
    - .var = all genomic tiles (Chromosome, Start, End)
    - .X = tile counts per cell-state (dense or sparse)

    This is required for:
    - GET 1.0 pipeline (peak count matrix extraction)
    - Peak × motif matrix generation
    - Downstream QC and coverage analysis
    """

    enabled: bool = False  # Must explicitly enable
    groupby: str = "labels"  # Column to group cells by (labels, metacell_id, etc.)
    tile_size: int = 500  # Tile size in base pairs
    min_cells_per_group: int = 1  # Minimum cells per group (use pseudobulk defaults if 0)
    min_libsize: int = 0  # Minimum library size per group (use pseudobulk defaults if 0)
    use_stored_tiles: bool = True  # Use tiles stored in zarr (recommended)
    apply_blacklist: bool = True  # Apply blacklist filtering to tiles
    sparse_output: bool = True  # Save as sparse matrix (recommended for large datasets)
    normalize: bool = False  # Apply CPM normalization
    overwrite: bool = False  # Overwrite existing output

    # Output options
    output_format: str = "h5ad"  # h5ad or zarr
    output_suffix: str = "cellstate_tiles"  # Suffix for output filename

    # Default I/O paths relative to output.dir
    input: str = "single_sparse.zarr"
    output: str = "cellstate_tile_counts.h5ad"

    # Metadata handling for external groupings (e.g., metacell assignments)
    external_metadata: str | None = None  # External CSV with groupings
    barcode_column: str = "barcode"  # Barcode column in external metadata

    def __post_init__(self):
        """Validate configuration."""
        if self.tile_size <= 0:
            raise ValueError("tile_size must be greater than 0")
        if self.output_format not in ("h5ad", "zarr"):
            raise ValueError(f"output_format must be 'h5ad' or 'zarr', got '{self.output_format}'")


@dataclass
class IsoLibsizeMetacellConfig:
    """Configuration for iso-libsize metacell sampling.

    This processor creates metacells with uniform ATAC library size
    using geometric sketching in embedding space. Can be used as an
    alternative to clustering-based pseudobulk aggregation.

    Works with both multiome data (using RNA embeddings) and standard
    single-cell ATAC (using LSI embeddings from tile matrix).
    """

    enabled: bool = False  # Must explicitly enable
    target_libsize: int = 5_000_000  # Target ATAC library size per metacell
    n_metacells: int | None = None  # Number of metacells (auto if None)
    embedding_key: str = "X_pca"  # X_pca for multiome, X_svd for standard ATAC
    n_neighbors: int = 50  # KNN neighbors for expansion
    min_libsize_fraction: float = 0.95  # Minimum fraction of target to accept
    max_cells_per_metacell: int = 2000  # Maximum cells per metacell
    seed: int = 42  # Random seed for reproducibility
    use_shared_cells_only: bool = True  # For multiome: only use shared cells
    tolerance: float = 0.05  # Tolerance for reaching target libsize

    # Clustering behavior option
    skip_clustering: bool = False  # If False, run clustering first for QC/viz

    # Output column name
    metacell_column: str = "metacell_id"  # Column name in metadata

    # Default I/O paths relative to output.dir
    input: str = "multiome_sparse.zarr"  # or single_sparse.zarr
    adata_path: str | None = None  # Optional external AnnData with embeddings

    # Overwrite existing results
    overwrite: bool = False  # Overwrite existing metacell assignments


@dataclass
class ChromosomeFilterConfig:
    """Chromosome filtering configuration."""

    max_length: int = 5
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["Un", "random", "_alt", "_fix"]
    )
    exclude_exact: list[str] = field(default_factory=lambda: ["chrM", "chrY"])
    include_patterns: list[str] = field(default_factory=list)
    include_exact: list[str] = field(default_factory=list)


@dataclass
class MetadataFilterConfig:
    """Metadata filter processor configuration."""

    dataset: str | None = None  # Dataset name for specific filters
    metadata_path: str | None = None  # Override default metadata path
    min_fragments: int = 1000  # Minimum fragments per cell
    max_fragments: int = 100000  # Maximum fragments per cell
    custom_filters: list[dict[str, Any]] | None = None  # Custom filter rules
    output_path: str | None = None  # Optional output path for filtered metadata


@dataclass
class SplitAnnotatorConfig:
    """Configuration for split annotation processor."""

    input_bed: str | None = None  # Optional - only required if split_annotator step is used
    cluster_file: str | None = None  # Optional - only required if split_annotator step is used
    output_bed: str | None = None  # Optional - only required if split_annotator step is used
    random_seed: int = 42
    val_chromosome: str = "chr8"
    interpolation_fraction: float = 0.1
    manifold_fraction: float = 0.5
    train_val_ratio: float = 0.9


@dataclass
class PeakDeduplicationConfig:
    """Configuration for peak deduplication processor."""

    input_bed: str | None = None  # Optional - only required if peak_deduplication step is used
    output_bed: str | None = None  # Optional - only required if peak_deduplication step is used
    genome_fasta: str | None = None  # Optional - only required if peak_deduplication step is used
    min_seq_id: float = 0.3
    coverage: float = 0.1
    script_dir: str | None = None  # Optional: directory containing mmseqs scripts


@dataclass
class DifferentialSignalConfig:
    """Configuration for differential signal calculation processor."""

    input_bed: str | None = None  # Optional - only required if differential_signal step is used
    output_bed: str | None = None  # Optional - only required if differential_signal step is used
    comparisons: list[list[str]] | None = None  # List of [condition1, condition2] pairs (OmegaConf converts tuples to lists)
    use_sanitized_names: bool = True  # Whether to sanitize column names
    normalization_factor: float = 1e5  # Normalization factor for log-normalization


@dataclass
class CNVDetectionConfig:
    """Configuration for CNV detection from ATAC-seq tile counts."""

    enabled: bool = True
    # Input source - can be zarr sparse, zarr dense, or bpcells fragments
    input: str | None = None  # Input store path (single_sparse.zarr or bpcells_fragments/)
    input_type: str = "auto"  # auto, zarr_sparse, zarr_dense, bpcells_fragments

    # Tile matrix generation (for bpcells fragments)
    tile_size: int = 500  # Tile size in bp
    regenerate_tiles: bool = False  # Force regenerate tiles even if exists

    # GC correction
    gc_correction: bool = True
    gc_method: str = "median_bin"  # median_bin or linear
    assembly: str = "auto"  # Genome assembly for GC content (hg38, mm10, etc.)

    # Smoothing
    smooth_method: str = "rolling"  # rolling or gaussian
    window_size: int = 21  # Window size in bins for smoothing

    # Segmentation
    joint_segment: bool = True  # Use joint segmentation (faster) vs per-cell
    min_segment_size: int = 10  # Minimum segment size in bins

    # CNV thresholds
    loss_threshold: float = -0.3  # Log2 ratio threshold for loss
    gain_threshold: float = 0.3  # Log2 ratio threshold for gain

    # Reference selection
    reference_cells: list[str] | None = None  # Cell IDs to use as diploid reference
    iterative: bool = True  # Use iterative detection with auto diploid detection
    diploid_threshold: float = 10.0  # Max % altered bins to consider diploid

    # Output
    overwrite: bool = False
    output_key: str = "cnv"  # Key for storing results in zarr

    def __post_init__(self):
        """Validate CNV detection configuration."""
        if self.tile_size <= 0:
            raise ValueError("tile_size must be greater than 0")
        if self.window_size <= 0:
            raise ValueError("window_size must be greater than 0")
        if self.min_segment_size <= 0:
            raise ValueError("min_segment_size must be greater than 0")
        if self.diploid_threshold <= 0 or self.diploid_threshold >= 100:
            raise ValueError("diploid_threshold must be between 0 and 100")


# ProcessingConfig removed - fields moved to specific configs
# type moved to WorkflowConfig, parameters moved to FragmentConfig/BAMConfig


@dataclass
class WorkflowStep:
    """Configuration for a single workflow step."""

    name: str = MISSING
    processor: str = MISSING
    description: str | None = None


@dataclass
class WorkflowConfig:
    """Workflow configuration."""

    name: str | None = None
    description: str | None = None
    type: str | None = None  # auto, single-cell, bulk - moved from ProcessingConfig
    steps: list[WorkflowStep] | None = None
    continue_on_error: bool = False


# Main unified configuration
@dataclass
class CaesarConfig:
    """Main unified configuration for Caesar preprocessing.

    This is the single source of truth for all configuration parameters.
    All processors receive the full config and extract what they need.

    Key design decisions:
    - Assembly is only at root level (no duplicates in processing)
    - All processor configs are at root level for easy access
    - Type-safe with validation methods
    - No complex parameter mappings needed
    """

    # Core configuration
    assembly: str = "auto"  # Single source of truth for assembly
    blacklist_file: str | None = None  # Assembly-specific blacklist file

    # Data and output
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)

    # Workflow configuration
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    # Processor-specific configurations
    fragments: FragmentConfig = field(default_factory=FragmentConfig)
    bam: BAMConfig = field(default_factory=BAMConfig)
    bigwig: BigWigConfig = field(default_factory=BigWigConfig)
    # BPCells processor configurations (step-specific)
    bpcells_import: BPCellsConfig | None = (
        None  # Step 1: Import fragments to BPCells format (output_type='fragments')
    )
    bpcells_anndata: BPCellsConfig | None = (
        None  # Step 2: Generate AnnData from BPCells fragments (output_type='anndata')
    )
    bpcells_pseudobulk: BPCellsConfig | None = (
        None  # Step 3: Create pseudobulk from BPCells fragments (output_type='celltype_dense')
    )
    # Legacy: bpcells for backward compatibility (maps to bpcells_anndata if not set)
    bpcells: BPCellsConfig = field(default_factory=BPCellsConfig)

    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    pseudobulk: PseudobulkConfig = field(default_factory=PseudobulkConfig)
    peaks: PeakCallingConfig = field(default_factory=PeakCallingConfig)
    library_size: LibrarySizeConfig = field(default_factory=LibrarySizeConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    footprints: FootprintConfig = field(default_factory=FootprintConfig)
    tss_accessibility: TSSAccessibilityConfig = field(
        default_factory=TSSAccessibilityConfig
    )  # TSS accessibility analysis config
    tes_accessibility: TESAccessibilityConfig = field(
        default_factory=TESAccessibilityConfig
    )  # TES accessibility analysis config
    add_peak: AddPeakConfig = field(default_factory=AddPeakConfig)
    add_expression: AddExpressionConfig | None = (
        None  # Optional expression annotation config
    )
    non_redundant_motif: NonRedundantMotifConfig | None = (
        None  # Optional non-redundant motif processing config
    )
    non_redundant_motif_curation: NonRedundantMotifCurationConfig | None = (
        None  # Optional non-redundant motif curation config
    )
    metadata_filter: MetadataFilterConfig = field(default_factory=MetadataFilterConfig)
    split_annotator: SplitAnnotatorConfig = field(default_factory=SplitAnnotatorConfig)
    peak_deduplication: PeakDeduplicationConfig = field(
        default_factory=PeakDeduplicationConfig
    )
    differential_signal: DifferentialSignalConfig = field(
        default_factory=DifferentialSignalConfig
    )
    cnv_detection: CNVDetectionConfig = field(
        default_factory=CNVDetectionConfig
    )  # CNV detection from tile counts
    multiome: MultiomeConfig = field(default_factory=MultiomeConfig)
    multiome_pseudobulk: MultiomePseudobulkConfig | None = (
        None  # Optional multiome pseudobulk config
    )
    isolibsize_metacell: IsoLibsizeMetacellConfig = field(
        default_factory=IsoLibsizeMetacellConfig
    )  # Iso-libsize metacell sampling config
    cellstate_tile_count: CellStateTileCountConfig = field(
        default_factory=CellStateTileCountConfig
    )  # Cell-state level full tile count matrix

    # Shared configurations
    chromosome_filter: ChromosomeFilterConfig = field(
        default_factory=ChromosomeFilterConfig
    )

    # Optional sections for advanced use
    multiome_rna: MultiomeRNAConfig | None = None  # RNA-specific multiome config
    test_mode: dict[str, Any] | None = None

    # Version info
    version: str = "2.0.0"


# Validation functions (separate from dataclasses to work with OmegaConf)
def validate_fragment_config(config: FragmentConfig) -> None:
    """Validate fragment configuration."""
    if config.min_fragments >= config.max_fragments:
        raise ValueError("min_fragments must be less than max_fragments")
    if config.min_cells < 1:
        raise ValueError("min_cells must be at least 1")
    if config.chunk_size < 1000:
        raise ValueError("chunk_size must be at least 1000")


def validate_bam_config(config: BAMConfig) -> None:
    """Validate BAM configuration."""
    if config.n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    if config.chunk_size < 1000:
        raise ValueError("chunk_size must be at least 1000")


def validate_caesar_config(config: CaesarConfig) -> None:
    """Validate the entire configuration."""
    # Validate processor-specific configs
    validate_fragment_config(config.fragments)
    validate_bam_config(config.bam)

    # Validate assembly
    valid_assemblies = ["auto", "hg38", "hg19", "mm10", "mm9"]
    if config.assembly not in valid_assemblies:
        raise ValueError(
            f"Invalid assembly: {config.assembly}. Must be one of {valid_assemblies}"
        )

    # Validate workflow type if specified
    if config.workflow.type:
        valid_types = ["auto", "single-cell", "bulk"]
        if config.workflow.type not in valid_types:
            raise ValueError(
                f"Invalid workflow type: {config.workflow.type}. Must be one of {valid_types}"
            )


# Predefined workflow templates
def create_single_cell_workflow() -> list[WorkflowStep]:
    """Create a single-cell ATAC-seq workflow with Zarr backend.

    This workflow provides a complete analysis pipeline using Zarr storage:
    1. Process fragments → Create sparse matrix from fragment files
    2. Cluster cells → Identify cell populations using TF-IDF + LSI + Leiden
    3. Create pseudobulk → Aggregate by cell clusters
    4. Call peaks → Identify regulatory regions
    5. Annotate library sizes → Calculate QC metrics
    6. Detect footprints → Identify TF binding sites
    7. Normalize → Apply normalization transformations

    This is the standard Zarr-based workflow for single-cell ATAC-seq analysis.
    """
    return [
        WorkflowStep(
            name="process_fragments",
            processor="fragments",
            description=(
                "Process fragment files to sparse Zarr matrix. "
                "Requires: fragments section with processing parameters"
            ),
        ),
        WorkflowStep(
            name="clustering",
            processor="clustering",
            description=(
                "Cluster cells using TF-IDF + LSI + Leiden on sparse matrix. "
                "Requires: clustering section with method and parameters"
            ),
        ),
        WorkflowStep(
            name="pseudobulk_by_clusters",
            processor="pseudobulk",
            description=(
                "Create pseudobulk by cell clusters. "
                "Requires: pseudobulk section with groupby parameter"
            ),
        ),
        WorkflowStep(
            name="call_peaks",
            processor="peaks",
            description=(
                "Call peaks using Caesar method on pseudobulk data. "
                "Requires: peaks section with method and qvalue parameters"
            ),
        ),
        WorkflowStep(
            name="annotate_library_sizes",
            processor="library_size_annotation",
            description=(
                "Calculate library sizes with blacklist filtering. "
                "Requires: library_size section with n_jobs parameter"
            ),
        ),
        WorkflowStep(
            name="detect_footprints",
            processor="footprints",
            description=(
                "Detect TF footprints on pseudobulk (before normalization). "
                "Requires: footprints section with motif_zarr and parameters"
            ),
        ),
        WorkflowStep(
            name="normalize",
            processor="normalization",
            description=(
                "Normalize signals using specified method. "
                "Requires: normalization section with method and reference parameters"
            ),
        ),
    ]


def create_bulk_workflow() -> list[WorkflowStep]:
    """Create a bulk ATAC/DNase-seq workflow."""
    return [
        WorkflowStep(
            name="process_data",
            processor="auto",  # Will be determined based on input type
            description="Process input files to coverage tracks",
        ),
        WorkflowStep(
            name="call_peaks",
            processor="peaks",
            description="Call peaks using Caesar method",
        ),
        WorkflowStep(
            name="annotate_library_sizes",
            processor="library_size_annotation",
            description="Calculate library sizes with blacklist filtering",
        ),
        WorkflowStep(
            name="detect_footprints",
            processor="footprints",
            description="Detect TF footprints on coverage (before normalization)",
        ),
        WorkflowStep(
            name="normalize", processor="normalization", description="Normalize signals"
        ),
    ]


def create_reprocessing_workflow() -> list[WorkflowStep]:
    """Create a reprocessing workflow for existing sparse zarr data.

    This workflow reprocesses existing sparse zarr with:
    1. Clustering with blacklist filtering
    2. Pseudobulk aggregation by clusters
    3. Library size annotation
    4. Peak calling
    5. Normalization
    """
    return [
        WorkflowStep(
            name="clustering",
            processor="clustering",
            description="Reprocess clustering with blacklist filtering",
        ),
        WorkflowStep(
            name="pseudobulk_by_clusters",
            processor="pseudobulk",
            description="Create pseudobulk by reprocessed clusters",
        ),
        WorkflowStep(
            name="annotate_library_sizes",
            processor="library_size_annotation",
            description="Calculate library sizes with blacklist filtering",
        ),
        WorkflowStep(
            name="call_peaks",
            processor="peaks",
            description="Call peaks on pseudobulk data",
        ),
        WorkflowStep(
            name="normalize",
            processor="normalization",
            description="Normalize signals",
        ),
    ]


def create_footprint_workflow() -> list[WorkflowStep]:
    """Create a footprint analysis workflow for existing celltype dense zarr."""
    return [
        WorkflowStep(
            name="detect_footprints",
            processor="footprints",
            description="Detect TF footprints on celltype dense zarr",
        ),
    ]


def create_reprocessing_isolibsize_workflow() -> list[WorkflowStep]:
    """Create a reprocessing workflow with iso-libsize metacell sampling.

    This workflow reprocesses existing sparse zarr data using iso-libsize
    metacell sampling for balanced pseudobulk generation. Unlike the standard
    reprocessing workflow which uses clustering-based pseudobulk, this creates
    metacells with uniform ATAC library size.

    Requirements:
    - Input sparse zarr must have embeddings (X_pca or X_svd) in linked h5ad
    - If embeddings don't exist, run clustering first to generate them

    The workflow skips clustering and goes directly to iso-libsize sampling,
    making it faster for data that already has embeddings computed.
    """
    return [
        WorkflowStep(
            name="isolibsize_metacell",
            processor="isolibsize_metacell",
            description="Generate balanced metacells with uniform library size",
        ),
        WorkflowStep(
            name="pseudobulk_by_metacells",
            processor="pseudobulk",
            description="Create pseudobulk by metacell assignments",
        ),
        WorkflowStep(
            name="annotate_library_sizes",
            processor="library_size_annotation",
            description="Calculate library sizes",
        ),
        WorkflowStep(
            name="call_peaks",
            processor="peaks",
            description="Call peaks on pseudobulk data",
        ),
        WorkflowStep(
            name="normalize",
            processor="normalization",
            description="Normalize signals",
        ),
        WorkflowStep(
            name="footprint",
            processor="footprints",
            description="TF footprinting analysis",
        ),
    ]


def create_reprocessing_isolibsize_bpcells_workflow() -> list[WorkflowStep]:
    """Create a reprocessing workflow with iso-libsize metacell sampling for BPCells.

    This workflow reprocesses existing BPCells sparse data using iso-libsize
    metacell sampling for balanced pseudobulk generation. Unlike the standard
    reprocessing workflow which uses clustering-based pseudobulk, this creates
    metacells with uniform ATAC library size.

    Requirements:
    - Input BPCells sparse store must have embeddings (X_svd) in linked h5ad
    - If embeddings don't exist, run clustering first to generate them

    The workflow skips clustering and goes directly to iso-libsize sampling,
    making it faster for data that already has embeddings computed.
    """
    return [
        WorkflowStep(
            name="isolibsize_metacell",
            processor="isolibsize_metacell",
            description="Generate balanced metacells with uniform library size (BPCells)",
        ),
        WorkflowStep(
            name="pseudobulk_celltype_dense",
            processor="bpcells",
            description=(
                "Create BPCells celltype_dense by metacell assignments. "
                "Requires: bpcells_pseudobulk section with output_type='celltype_dense', "
                "use_clustering_labels=false, annotation_col='metacell_id'"
            ),
        ),
        WorkflowStep(
            name="annotate_library_sizes",
            processor="library_size_annotation",
            description="Calculate library sizes on BPCells celltype_dense store",
        ),
        WorkflowStep(
            name="call_peaks",
            processor="peaks",
            description="Call peaks on BPCells celltype_dense pseudobulk",
        ),
        WorkflowStep(
            name="normalize",
            processor="normalization",
            description=(
                "Calculate normalization params on BPCells celltype_dense store. "
                "For BPCells: stores normalization params only (preserves raw counts)."
            ),
        ),
        WorkflowStep(
            name="footprint",
            processor="footprints",
            description="TF footprinting analysis on BPCells celltype_dense store",
        ),
    ]


def create_single_cell_bpcells_workflow() -> list[WorkflowStep]:
    """Create a single-cell ATAC-seq workflow with BPCells output.

    This workflow processes fragment files into efficient BPCells format.
    Supports multiple output types:
    - single_sparse: Per-cell sparse matrices (default)
    - multi_sparse: Separate outputs per fragment for MultiSparseBPCellsIO
    - celltype_dense: Celltype-aggregated pseudobulk (requires metadata)
    - signal_dense: Single pseudobulk signal

    Note: Peak calling is supported on dense outputs (celltype_dense/signal_dense).
    Configure bpcells.output_type in your config to choose the output format.
    """
    return [
        WorkflowStep(
            name="process_to_bpcells",
            processor="bpcells",
            description=(
                "Process fragment files to BPCells format. "
                "Output type (single_sparse/multi_sparse/celltype_dense/signal_dense) "
                "is controlled by bpcells.output_type config. "
                "For dense outputs, peak calling can be added as a downstream step."
            ),
        ),
        # Optional: Peak calling on dense outputs (celltype_dense or signal_dense)
        # Uncomment and configure bpcells.output_type appropriately:
        # WorkflowStep(
        #     name="call_peaks",
        #     processor="peaks",
        #     description="Call peaks on BPCells dense output (requires celltype_dense or signal_dense)",
        # ),
        # Note: Other downstream steps (clustering, pseudobulk, etc.) are NOT supported
        # with BPCells output. Use Zarr backend or BPCells IO classes for further analysis.
    ]


def create_bpcells_complete_workflow() -> list[WorkflowStep]:
    """Create complete single-cell ATAC-seq workflow with BPCells backend.

    This workflow provides a full analysis pipeline using BPCells:
    1. Import fragments → Convert 10x fragments to BPCells format (saved for reuse)
    2. Generate AnnData → Create binned genome-wide matrix from saved BPCells fragments
    3. Cluster cells → Identify cell populations
    4. Pseudobulk aggregation → Create celltype_dense by clusters using saved BPCells fragments
    5. Library size annotation → Calculate QC metrics
    6. Peak calling → Identify regulatory regions
    7. Normalize → Calculate normalization params (stored, preserves raw counts)

    This is equivalent to the Zarr single-cell workflow but uses BPCells
    for efficient storage and faster AnnData generation.

    Note: This workflow requires proper configuration of:
    - bpcells_import: for fragment import step
    - bpcells_anndata: for AnnData generation step
    - bpcells_pseudobulk: for pseudobulk aggregation step
    - normalization: for normalization step (now supports BPCells backend)
    """
    return [
        WorkflowStep(
            name="import_fragments",
            processor="bpcells",
            description=(
                "Import 10x fragment files to BPCells format. "
                "Saves BPCells fragments to output_dir/bpcells_fragments/ for reuse in subsequent steps. "
                "Requires: bpcells_import section with output_type='fragments'"
            ),
        ),
        WorkflowStep(
            name="generate_anndata",
            processor="bpcells",
            description=(
                "Generate AnnData from saved BPCells fragments (fast C++ path). "
                "Automatically uses saved BPCells fragments from import step. "
                "Requires: bpcells_anndata section with output_type='anndata'"
            ),
        ),
        WorkflowStep(
            name="clustering",
            processor="clustering",
            description=(
                "Cluster cells using TF-IDF + LSI + Leiden on AnnData. "
                "Supports AnnData file input from BPCells step. "
                "Outputs: updated anndata.h5ad with cluster labels, metadata.csv"
            ),
        ),
        WorkflowStep(
            name="pseudobulk_celltype_dense",
            processor="bpcells",
            description=(
                "Create celltype_dense pseudobulk using saved BPCells fragments. "
                "Automatically uses saved BPCells fragments from import step. "
                "Uses clustering labels from previous step. "
                "Requires: bpcells_pseudobulk section with output_type='celltype_dense', "
                "use_clustering_labels=true"
            ),
        ),
        WorkflowStep(
            name="annotate_library_sizes",
            processor="library_size_annotation",
            description=(
                "Calculate library sizes on BPCells celltype_dense store. "
                "Supports both Zarr and BPCells backends automatically."
            ),
        ),
        WorkflowStep(
            name="call_peaks",
            processor="peaks",
            description=(
                "Call peaks on BPCells celltype_dense pseudobulk. "
                "Supports both Zarr and BPCells backends automatically. "
                "Uses Caesar peak calling method."
            ),
        ),
        WorkflowStep(
            name="detect_footprints",
            processor="footprints",
            description=(
                "Detect TF footprints on BPCells pseudobulk (before normalization). "
                "Requires: footprints section with motif_zarr and parameters"
            ),
        ),
        WorkflowStep(
            name="normalize",
            processor="normalization",
            description=(
                "Calculate normalization params on BPCells celltype_dense store. "
                "For BPCells: stores normalization params only (preserves raw counts). "
                "Normalization is applied on-the-fly when using QuantileNormalizer with get_track(). "
                "Supports both Zarr and BPCells backends automatically. "
                "Requires: normalization section with method and reference parameters"
            ),
        ),
    ]


# Workflow templates registry
WORKFLOW_TEMPLATES = {
    # Single-cell workflows (Zarr backend)
    "single-cell": create_single_cell_workflow,
    "single-cell-complete": create_single_cell_workflow,  # Alias for backward compatibility
    "single-cell-simple": create_single_cell_workflow,  # Same as single-cell for now
    # BPCells workflows
    "bpcells-import-fragment": create_single_cell_bpcells_workflow,  # Simple fragment import to BPCells
    "bpcells-complete": create_bpcells_complete_workflow,  # Full pipeline with BPCells backend
    # Bulk workflows
    "bulk": create_bulk_workflow,
    "bulk-simple": create_bulk_workflow,  # Same as bulk for now
    # Specialized workflows
    "reprocessing": create_reprocessing_workflow,
    "reprocessing-isolibsize": create_reprocessing_isolibsize_workflow,
    "reprocessing-isolibsize-bpcells": create_reprocessing_isolibsize_bpcells_workflow,
    "footprint": create_footprint_workflow,
}


# Register configurations with Hydra ConfigStore
def register_configs():
    """Register all configuration dataclasses with Hydra ConfigStore.

    This function should be called when using Hydra for CLI generation.
    """
    cs = ConfigStore.instance()

    # Register all configuration dataclasses
    cs.store(name="base_caesar", node=CaesarConfig)
    cs.store(group="data", name="base", node=DataConfig)
    cs.store(group="data", name="fragment_file", node=FragmentFile)
    cs.store(group="data", name="bam_file", node=BAMFile)
    cs.store(group="output", name="base", node=OutputConfig)
    cs.store(group="resources", name="base", node=ResourcesConfig)
    cs.store(group="fragments", name="base", node=FragmentConfig)
    cs.store(group="bam", name="base", node=BAMConfig)
    cs.store(group="bpcells", name="base", node=BPCellsConfig)
    cs.store(group="clustering", name="base", node=ClusteringConfig)
    cs.store(group="pseudobulk", name="base", node=PseudobulkConfig)
    cs.store(group="peaks", name="base", node=PeakCallingConfig)
    cs.store(group="library_size", name="base", node=LibrarySizeConfig)
    cs.store(group="normalization", name="base", node=NormalizationConfig)
    cs.store(group="footprints", name="base", node=FootprintConfig)
    cs.store(group="add_peak", name="base", node=AddPeakConfig)
    cs.store(group="add_expression", name="base", node=AddExpressionConfig)
    cs.store(group="non_redundant_motif", name="base", node=NonRedundantMotifConfig)
    cs.store(
        group="non_redundant_motif_curation",
        name="base",
        node=NonRedundantMotifCurationConfig,
    )
    cs.store(group="metadata_filter", name="base", node=MetadataFilterConfig)
    cs.store(group="multiome", name="base", node=MultiomeConfig)
    cs.store(group="multiome_rna", name="base", node=MultiomeRNAConfig)
    cs.store(group="multiome_pseudobulk", name="base", node=MultiomePseudobulkConfig)
    cs.store(group="isolibsize_metacell", name="base", node=IsoLibsizeMetacellConfig)
    cs.store(group="chromosome_filter", name="base", node=ChromosomeFilterConfig)
    cs.store(group="workflow", name="base", node=WorkflowConfig)


# Helper functions for config management
def export_config(cfg: DictConfig, path: str) -> None:
    """Export configuration to YAML file.

    Args:
        cfg: Hydra DictConfig to export
        path: Path to save the config file
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to regular dict for clean YAML output
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Save to YAML
    with open(output_path, "w") as f:
        OmegaConf.save(config_dict, f)

    print(f"Configuration exported to: {output_path}")


def load_caesar_config(config_path: str | None, overrides: list[str] = None) -> DictConfig:
    """Load Caesar configuration from file with optional overrides.

    Args:
        config_path: Path to YAML configuration file (if None, uses empty config)
        overrides: List of Hydra-style overrides (e.g., ["data.fragments.min_fragments=2000"])

    Returns:
        Loaded and validated configuration
    """
    # Load base config
    if config_path is not None:
        cfg = OmegaConf.load(config_path)
    else:
        # Create empty config when no file is provided
        cfg = OmegaConf.create({})

    # Apply overrides if provided
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Create structured config for validation
    schema = OmegaConf.structured(CaesarConfig)
    cfg = OmegaConf.merge(schema, cfg)

    # Normalize data inputs if present
    if hasattr(cfg, "data") and cfg.data is not None:
        data_obj = OmegaConf.to_object(cfg.data)
        if isinstance(data_obj, DataConfig):
            data_obj.normalize_inputs()
            # Update the config with normalized data
            cfg.data = OmegaConf.structured(data_obj)

    # Validate the configuration
    caesar_cfg = OmegaConf.to_object(cfg)
    if isinstance(caesar_cfg, CaesarConfig):
        validate_caesar_config(caesar_cfg)

    return cfg
