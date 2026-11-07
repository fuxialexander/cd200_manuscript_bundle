## Design Document: Caesar IO Module (`src/caesar/io/`)

**Authors:** Xi Fu
**Last Updated:** 2025-12-22

### 1. Overview

This document outlines the design of the `src/caesar/io/` module. Its primary purpose is to provide a unified, backend-agnostic Application Programming Interface (API) for accessing and interacting with various genomic data storage formats. The module supports multiple backends (Zarr and BPCells) and can be extended to support additional formats. It defines abstract base classes, common data access patterns, query mechanisms, aggregation/normalization strategies, and concrete implementations for specific data types and storage backends. Mixins are used to add specialized analysis capabilities (e.g., footprinting, peak calling, clustering) to relevant classes.

The design emphasizes flexibility, maintainability, and extensibility by employing:
- Abstract Base Classes (ABCs) for backend-agnostic interfaces
- Strategy pattern for aggregation and normalization
- Explicit Query objects for data selection
- Mixins for composable analysis capabilities
- Separate backend implementations (Zarr, BPCells)

### 2. Module Structure

The `src/caesar/io/` module is organized into several layers:

#### 2.1. Core Files
*   **`genomic_io.py`**: Defines the `GenomicIO` abstract base class
*   **`interface.py`**: Defines data-type-specific abstract classes (`DenseIO`, `SignalDenseIO`, `SingleSparseIO`, `MultiSparseIO`, `SequenceDenseIO`, `CelltypeDenseIO`)
*   **`query.py`**: Query object definitions (`BaseQuery`, `DefaultQuery`, `KeyQuery`, `IndexQuery`, `MetadataQuery`, `MultiKeyQuery`)
*   **`aggregator.py`**: Aggregation strategies (`Aggregator`, `SumAggregator`, `ByKeyAggregator`, `ConcatAggregator`)
*   **`normalizer.py`**: Normalization strategies (`Normalizer`, `NoNormalizer`, `FactorNormalizer`, `DictNormalizer`, `ListNormalizer`)
*   **`metadata_io.py`**: Metadata handling (`Metadata` class)
*   **`adata_io.py`**: AnnData integration utilities
*   **`peak_ordering.py`**: Peak file reading utilities
*   **`parallel_processing.py`**: Parallel processing utilities

#### 2.2. Backend Implementations
*   **`zarr_io/`**: Zarr backend implementations
    *   `base.py`: `ZarrIO` abstract base class
    *   `dense.py`: Dense Zarr implementations (`DenseZarrIO`, `SignalDenseZarrIO`, `MotifDenseZarrIO`, `SequenceDenseZarrIO`, `CelltypeDenseZarrIO`)
    *   `sparse.py`: Sparse Zarr implementations (`SingleSparseZarrIO`, `MultiSparseZarrIO`)
    *   `multiome.py`: Multiome implementations (`MultiomeSparseZarrIO`, `MultiomeDenseZarrIO`)
    *   `motif_hits.py`: Motif hit storage (`MotifHitZarrIO`)
*   **`bpcell_io.py`**: BPCells backend implementations (`BPCellsIO`, `CelltypeDenseBPCellsIO`, `SignalDenseBPCellsIO`, `SingleSparseBPCellsIO`, `MultiSparseBPCellsIO`)

#### 2.3. Mixins
*   **`mixins/`**: Reusable analysis capabilities
    *   `call_peaks.py`: `CallPeaksMixin` - peak calling functionality
    *   `clustering.py`: `ClusteringMixin` - single-cell clustering
    *   `footprint.py`: `FootprintMixin` - TF footprinting analysis
    *   `tss_extraction.py`: `TSSExtractionMixin` - TSS-centered signal extraction
    *   `quantile_normalization.py`: `QuantileNormalizationMixin` - quantile normalization
    *   `library_size_annotation.py`: `LibrarySizeAnnotationMixin` - library size annotation
    *   `genomic_masking.py`: `GenomicMaskingMixin` - genomic region masking
    *   `label.py`: `LabelMixin` - cell/sample labeling utilities

### 3. Core Components

#### 3.1. `GenomicIO` Abstract Base Class (`genomic_io.py`)

*   **Purpose:** Defines the fundamental contract that all backend IO implementations must adhere to. Ensures interoperability across different storage backends.
*   **Location:** `src/caesar/io/genomic_io.py`
*   **Inheritance:** Inherits from `abc.ABC`.
*   **Key Attributes:**
    *   `path`, `mode`, `dtype`: Basic initialization parameters.
    *   `assembly`: Genome assembly identifier (e.g., "hg38"), determined by `_init_data_structure`.
    *   `assembly_chrom_sizes`: Dictionary mapping chromosome names to lengths for the entire detected assembly.
    *   `_active_chroms`: List of chromosome names currently considered active (intersection of assembly and data, potentially filtered).
    *   `chunk_size`: Default chunk size used for processing, set by `_init_chunking`.
    *   `dir_chunked`: Boolean indicating directory-based chunking (backend-specific).
    *   `_libsize`, `_per_cell_libsize`: Internal storage for lazily calculated library sizes.
    *   `_gencode`: Cached Gencode object for gene annotation and name resolution.
    *   `_init_chunking_done`: Boolean flag indicating if `_init_chunking` has completed.
*   **Key Properties:**
    *   `data_chroms` (Abstract): Must return the list of chromosomes present in the specific data source.
    *   `chroms`: Returns the sorted list of *active* chromosome names (`_active_chroms`).
    *   `chrom_sizes`: Returns a dictionary of sizes for *active* chromosomes, derived from `assembly_chrom_sizes`.
    *   `chrom_n_chunks`: Returns the number of chunks per *active* chromosome based on `chrom_sizes` and `chunk_size`.
    *   `libsize`: Returns total library size(s). If library size annotation has been performed, returns non-blacklist counts; otherwise calculated via `_calculate_libsize` on first access.
    *   `per_cell_libsize`: Returns per-cell library sizes (for sparse), calculated via `_calculate_per_cell_libsize` on first access.
    *   `gencode`: Returns cached Gencode object for gene annotation, lazily initialized based on `assembly`.
*   **Abstract Methods (Must be implemented by subclasses):**
    *   `_init_data_structure()`: Handles loading metadata, determining data layout, setting `assembly`, loading `assembly_chrom_sizes`, determining `data_chroms`, setting `_active_chroms`, potentially calling `_init_chunking`.
    *   `_init_chunking()`: Sets `chunk_size` and marks chunking initialization state (`_init_chunking_done`).
    *   `_setup_dataset()`: Sets up the dataset (e.g., opens file handles).
    *   `_load_attr(key)`: Loads an attribute from the backend.
    *   `_save_attr(key, value)`: Saves an attribute to the backend.
    *   `_get_coverage_for_query()`: **Core backend method.** Retrieves data for a specific region based on a `BaseQuery` object. **Crucially, it must always return a `RawCoverageDict` (`Dict[str, Union[np.ndarray, csr_matrix]]`)**, even if the query results in a single logical group (e.g., `{\'default\': data}`).
    *   `get_default_track_obj()`: Returns a default `Track` object representing the dataset's primary view.
    *   `_get_all_ids()`: Returns identifiers (e.g., samples, cell types, ['default']) managed by the instance.
    *   `_calculate_libsize()`: Calculates total library size(s) and stores in `_libsize`.
    *   `_calculate_per_cell_libsize()`: Calculates library sizes per cell (for sparse) and stores in `_per_cell_libsize`.
*   **Concrete Methods (Implemented in `GenomicIO`):**
    *   `_determine_assembly()`: Determines the assembly based on backend metadata.
    *   `_load_assembly_chrom_sizes()`: Helper to fetch chromosome sizes based on `self.assembly`.
    *   `_parse_query()`: Helper to convert user input (None, str, list, dict, np.ndarray) into `BaseQuery` objects.
    *   `get_track()`: Unified public method for data retrieval (see Section 5).
    *   `get_track_for_regions()`: Iterates over regions, calling `get_track`.
    *   `get_gene_track()` / `get_gene_tss_track()`: Calculate gene/TSS regions and call `get_track`. Accepts either `Gene` object or gene name string (auto-resolved via `gencode`).
    *   `_get_smoothed_track()`: Convenience method to get a smoothed `np.ndarray`.
    *   `get_track_obj()` / `get_gene_track_obj()` / `get_gene_tss_track_obj()`: Convenience methods calling `get_track` with `output_format='track_object'`.
    *   `leave_out_chromosomes()`: Modifies the `_active_chroms` set in-place to exclude specified chromosomes.

#### 3.2. Query Objects (`query.py`)

*   **Purpose:** Provide explicit, type-safe representations of different data selection methods. Decouples query definition from backend execution.
*   **Location:** `src/caesar/io/query.py`
*   **Classes:**
    *   `BaseQuery(ABC)`: Abstract base class for all query types.
    *   `DefaultQuery`: Selects the default dataset view (e.g., all cells, single sample).
    *   `IndexQuery`: Selects based on integer indices (e.g., cell indices).
    *   `KeyQuery`: Selects based on one or more string keys (e.g., sample names, cell type IDs).
    *   `MetadataQuery`: Selects based on a metadata filter string (e.g., pandas query expression).
    *   `MultiKeyQuery`: Represents multiple named groups for aggregation (e.g., pseudobulk), mapping output group names to underlying keys/indices/queries.

#### 3.3. Strategy Patterns: Aggregation & Normalization

*   **Purpose:** Encapsulate algorithms for aggregation and normalization, allowing them to be varied independently from the main data access logic (Strategy Design Pattern).

##### Aggregators (`aggregator.py`)
*   **Location:** `src/caesar/io/aggregator.py`
*   **Classes:**
    *   `Aggregator(ABC)`: Defines `aggregate(RawCoverageDict) -> AggregatedCoverage` and `get_track_keys(RawCoverageDict) -> List[str]`.
    *   `SumAggregator`: Sums values in the input dict. Returns `np.ndarray` or `csr_matrix`. Track key is typically singular (e.g., 'aggregated_sum').
    *   `ByKeyAggregator`: Returns the input dict unchanged. Track keys match input keys.
    *   `ConcatAggregator`: Stacks/concatenates values. Returns `np.ndarray` or `List[csr_matrix]`. Track keys match sorted input keys.

##### Normalizers (`normalizer.py`)
*   **Location:** `src/caesar/io/normalizer.py`
*   **Classes:**
    *   `Normalizer(ABC)`: Defines `prepare_norm_factors(List[str]) -> NormFactors | None`.
    *   `NoNormalizer`: Returns `None` (no normalization).
    *   `FactorNormalizer`: Applies a single normalization factor to all specified track keys.
    *   `DictNormalizer`: Applies different factors specified in a dictionary, matching by track key.
    *   `ListNormalizer`: Applies normalization factors from a list, matched by track index.

#### 3.4. Data-Type-Specific IO Classes (`interface.py`)

*   **Purpose:** Provide abstract implementations for different types of genomic data (dense signal, celltype profiles, sequence, sparse single-cell, sparse multi-sample). These classes define the interface and common behavior but are still backend-agnostic.
*   **Location:** `src/caesar/io/interface.py`

##### Mixins Used
*   `FootprintMixin`: Adds footprinting analysis capabilities (genome-wide, motif-specific).
*   `TSSExtractionMixin`: Adds methods for extracting TSS-centered signals.
*   `QuantileNormalizationMixin`: Adds quantile normalization functionality.
*   `CallPeaksMixin`: Adds peak calling capabilities using MACS2/MACS3.
*   `ClusteringMixin`: Adds cell clustering functionality (primarily for sparse data).
*   `LibrarySizeAnnotationMixin`: Adds library size annotation and blacklist region handling.
*   `GenomicMaskingMixin`: Adds genomic region masking capabilities.
*   `LabelMixin`: Adds cell/sample labeling utilities.

##### Interface Classes
*   **`DenseIO`**: Base class for dense data.
    *   Mixins: `FootprintMixin`, `TSSExtractionMixin`, `LibrarySizeAnnotationMixin`
    *   Implements: `_get_all_ids` (returns `['default']`), `_calculate_libsize` (no-op), `_calculate_per_cell_libsize` (no-op), `get_default_track_obj`
    *   Methods: `get_default_footprint`

*   **`SignalDenseIO`**: For dense signal tracks (e.g., bulk ATAC-seq, ChIP-seq).
    *   Inherits from: `DenseIO`
    *   Mixins: `QuantileNormalizationMixin`, `CallPeaksMixin`, `FootprintMixin`, `LibrarySizeAnnotationMixin`, `GenomicMaskingMixin`
    *   Overrides: `get_default_track_obj` (normalizes by library size, applies smoothing)
    *   Methods: `get_high_quality_summits`, `_process_chunk_get_summits`

*   **`CelltypeDenseIO`**: For dense data structured by cell types.
    *   Inherits from: `DenseIO`
    *   Mixins: `CallPeaksMixin`, `FootprintMixin`, `QuantileNormalizationMixin`, `TSSExtractionMixin`, `LibrarySizeAnnotationMixin`, `LabelMixin`
    *   Implements: `subset` (abstract method for subsetting cell types)
    *   Properties: `metadata`
    *   Methods: `sample_track`, `get_high_quality_summits`, `_process_chunk_get_summits`, `get_peak_counts`, `_process_chunk_get_peak_counts`, `_load_normalized_chunk_for_all_id`, `leave_out_celltypes`, `leave_out_celltypes_with_pattern`, `subset_celltypes_with_data_name`, `filter_by_min_depth`
    *   Overrides: `_get_all_ids`, `get_default_track_obj`

*   **`SequenceDenseIO`**: For dense sequence data (e.g., one-hot encoded DNA).
    *   Inherits from: `DenseIO`
    *   Mixins: `FootprintMixin`
    *   Methods: `gc_content`, `_compute_gc_content`, `get_peak_gc_content`

*   **`SingleSparseIO`**: For sparse single-cell data from a single sample/source.
    *   Inherits from: `GenomicIO`
    *   Mixins: `CallPeaksMixin`, `FootprintMixin`, `QuantileNormalizationMixin`, `TSSExtractionMixin`, `ClusteringMixin`, `LibrarySizeAnnotationMixin`, `LabelMixin`
    *   Methods: `_metadata_query`, `_resolve_query_to_indices`, `get_normalization_factor`, `get_peaks`, `get_count_matrix`, `create_adata`
    *   Overrides: `get_default_track_obj` (for sparse sum)

*   **`MultiSparseIO`**: For sparse single-cell data aggregated from multiple samples/sources.
    *   Inherits from: `GenomicIO`
    *   Mixins: `CallPeaksMixin`, `FootprintMixin`, `QuantileNormalizationMixin`, `TSSExtractionMixin`, `ClusteringMixin`, `LibrarySizeAnnotationMixin`, `LabelMixin`
    *   Manages: Multiple underlying IO instances (backend-specific)
    *   Methods: `get_label_counts`, `get_label_distribution`, `get_cell_qc_metrics`, `create_adata`
    *   Overrides: `data_chroms` (intersection), `_get_all_ids`, `get_default_track_obj`, `_calculate_libsize`, `_calculate_per_cell_libsize`

### 4. Backend Implementations

The module provides two complete backend implementations: Zarr and BPCells.

#### 4.1. Zarr Backend (`zarr_io/`)

*   **Location:** `src/caesar/io/zarr_io/`
*   **Base Class:** `ZarrIO` (in `base.py`) - inherits from `GenomicIO` and provides Zarr-specific initialization and data access methods.

##### Zarr Dense Implementations (`dense.py`)
*   **`DenseZarrIO`**: Zarr implementation of `DenseIO`
*   **`SignalDenseZarrIO`**: Zarr implementation of `SignalDenseIO` for bulk signal tracks
*   **`MotifDenseZarrIO`**: Specialized class for motif data (motif IDs, scores, PWMs)
    *   Methods: `get_motif_ids`, `get_motif_names`, `get_motifs_by_name`, `get_tf_names`, `filter_by_tf`, etc.
*   **`SequenceDenseZarrIO`**: Zarr implementation of `SequenceDenseIO` for DNA sequences
*   **`CelltypeDenseZarrIO`**: Zarr implementation of `CelltypeDenseIO` for cell-type-structured data

##### Zarr Sparse Implementations (`sparse.py`)
*   **`SingleSparseZarrIO`**: Zarr implementation of `SingleSparseIO`
    *   Storage format: Compressed sparse row (CSR) matrices per chromosome
    *   Metadata: Cell barcodes, clustering results, QC metrics
*   **`MultiSparseZarrIO`**: Zarr implementation of `MultiSparseIO`
    *   Manages: Dictionary of `SingleSparseZarrIO` instances
    *   Aggregation: Cross-sample queries and pseudobulking

##### Zarr Multiome Implementations (`multiome.py`)
*   **`MultiomeIO`** (Abstract): Base mixin for multiome data (paired ATAC+RNA)
    *   Methods for RNA data access: `get_rna_expression`, `get_highly_variable_genes`, etc.
*   **`MultiomeSparseZarrIO`**: Extends `SingleSparseZarrIO` with RNA expression data
    *   Storage: ATAC in CSR format, RNA in separate group
    *   Methods: `add_rna_expression`, `get_rna_track`, `get_gene_expression`
*   **`MultiomeDenseZarrIO`**: Extends `CelltypeDenseZarrIO` with pseudobulked RNA profiles
    *   Storage: Both ATAC and RNA pseudobulked by cell type

##### Other Zarr Classes
*   **`MotifHitZarrIO`** (`motif_hits.py`): Storage for motif hit locations
    *   Methods: `add_hits`, `get_hits`, `get_hit_counts`

#### 4.2. BPCells Backend (`bpcell_io.py`)

*   **Location:** `src/caesar/io/bpcell_io.py`
*   **Base Class:** `BPCellsIO` - inherits from `GenomicIO` and provides BPCells-specific data access
*   **Storage Format:** BPCells uses a custom binary format optimized for sparse genomic data with fast random access

##### BPCells Implementations
*   **`CelltypeDenseBPCellsIO`**: BPCells implementation of `CelltypeDenseIO`
    *   Uses `MultiPrecalculatedInsertionMatrix` for efficient cell-type-aggregated access
*   **`SignalDenseBPCellsIO`**: BPCells implementation of `SignalDenseIO`
*   **`SingleSparseBPCellsIO`**: BPCells implementation of `SingleSparseIO`
    *   Direct access to fragment files
*   **`MultiSparseBPCellsIO`**: BPCells implementation of `MultiSparseIO`
    *   Manages multiple fragment sources
*   **`MultiomeSparseBPCellsIO`**: Extends `SingleSparseBPCellsIO` with RNA expression data
    *   Storage: ATAC in BPCells format, RNA in separate h5ad file
    *   Methods: `get_rna_for_shared_cells`, `get_gene_expression`, `compute_atac_rna_correlation`

##### BPCells Utilities
*   **`MultiPrecalculatedInsertionMatrix`**: Wrapper for backward compatibility with Caesar
*   **`build_cell_groups_legacy`**: Legacy wrapper for building cell groups
*   **`precalculate_insertion_counts_legacy`**: Legacy wrapper for precalculating insertion matrices

### 5. Core Data Access Workflow (`get_track`)

The primary user interaction for data retrieval is via the unified `GenomicIO.get_track` method. The workflow is as follows:

1.  **Input:** User provides genomic coordinates (`chr_name`, `start`, `end`), a query specification (`query`), and optionally `aggregator`, `normalizer`, `sparse` flag, `output_format`, `track_label`, and `**kwargs` (for `Track` constructor).
2.  **Input Validation:** Checks aggregator/normalizer types, `output_format`, and if `chr_name` is in active chromosomes.
3.  **Query Parsing:** `_parse_query` converts the user `query` (e.g., `None`, list, string, dict) into a specific `BaseQuery` object.
4.  **Backend Retrieval:** The parsed `BaseQuery` is passed to the backend-specific `_get_coverage_for_query` method, which interacts with the storage and **always** returns a `RawCoverageDict`.
5.  **Aggregation:** The chosen `Aggregator` strategy processes the `RawCoverageDict` to produce `aggregated_data`. The aggregator also determines the `track_keys` expected in the output.
6.  **Normalization & Formatting (Conditional):**
    *   If `output_format == 'track_object'`:
        *   The aggregated data is structured into a dictionary suitable for the `Track` constructor, using appropriate keys (potentially derived from `track_label` and `aggregator.get_track_keys`).
        *   The `Normalizer` strategy generates the `normalize_factor` dictionary based on the final track keys.
        *   A `caesar.analysis.track.Track` object is instantiated with the structured tracks and normalization factors. `**kwargs` are passed to the `Track` constructor.
    *   If `output_format == 'raw_array'`:
        *   Checks if `aggregated_data` is a single array/matrix (e.g., from `SumAggregator`). Raises `TypeError` otherwise.
        *   Applies normalization **only** if `normalizer` is `FactorNormalizer`. `DictNormalizer` is ignored with a warning.
        *   Returns the potentially normalized array/matrix.
    *   If `output_format == 'raw_dict'`:
        *   Checks if `aggregated_data` is a dictionary (e.g., from `ByKeyAggregator`). If it's a single array, it wraps it in a dict using a key from the aggregator. Raises `TypeError` otherwise.
        *   Returns the aggregated dictionary **without** applying normalization.
    *   If `output_format == 'raw_list'`:
        *   Checks if `aggregated_data` is a list (e.g., from `ConcatAggregator` with sparse data). Raises `TypeError` otherwise.
        *   Returns the aggregated list **without** applying normalization.
7.  **Return:** The method returns the final data in the format specified by `output_format`.

### 6. Utility Modules

#### 6.1. AnnData Integration (`adata_io.py`)
*   **Purpose:** Convert Caesar sparse data to AnnData format for compatibility with scanpy/scverse ecosystem
*   **Key Functions:**
    *   `prepare_adata`: Preprocess AnnData for analysis (filtering, normalization, PCA, clustering)
    *   `add_gene_activity_matrix`: Create gene activity scores from ATAC peaks
    *   `add_gene_activity_matrix_use_encode_re2g`: Use ENCODE regulatory element-to-gene links

#### 6.2. Metadata (`metadata_io.py`)
*   **Purpose:** Handle metadata tables associated with samples and cells
*   **Class:** `Metadata`
    *   Methods for loading, filtering, and querying metadata
    *   Integration with query system for metadata-based data selection

#### 6.3. Peak Ordering (`peak_ordering.py`)
*   **Purpose:** Read and process peak files in various formats
*   **Function:** `read_peaks`
    *   Supports BED, narrowPeak, and other genomic interval formats
    *   Returns standardized DataFrame with chromosome, start, end columns

#### 6.4. Parallel Processing (`parallel_processing.py`)
*   **Purpose:** Utilities for parallel data processing
*   **Features:** Process pool management, chunk-based parallel operations

### 7. Extensibility

*   **New Backends:**
    1. Create a new base class inheriting from `GenomicIO` (e.g., `NewBackendIO`)
    2. Implement all abstract methods, especially `_get_coverage_for_query`
    3. Create concrete classes inheriting from both the data-type interface (e.g., `DenseIO`) and your backend base
    4. Ensure `_init_data_structure` and `_init_chunking` are correctly implemented
    5. Add to `__init__.py` exports

*   **New Query Types:**
    1. Define a new subclass of `BaseQuery`
    2. Update `_parse_query` if simple user input should map to it
    3. Implement handling in backend `_get_coverage_for_query` methods
    4. For sparse backends, update `_resolve_query_to_indices` if needed

*   **New Aggregation Methods:**
    1. Create a new subclass of `Aggregator`
    2. Implement `aggregate()` and `get_track_keys()` methods
    3. Users can pass instances to `get_track`
    4. Update `get_track` formatting logic if needed for `output_format='track_object'`

*   **New Normalization Methods:**
    1. Create a new subclass of `Normalizer`
    2. Implement `prepare_norm_factors()` method
    3. Users can pass instances to `get_track`
    4. Update `get_track` formatting logic if needed

*   **New Analysis Capabilities:**
    1. Create new Mixin classes in `mixins/` directory
    2. Implement analysis methods (ensure they work with backend-agnostic interfaces)
    3. Add mixins to relevant data-type classes in `interface.py`
    4. Export from `mixins/__init__.py`

### 8. Dependencies

*   **Internal:**
    *   `caesar.analysis.track`: Track visualization and manipulation
    *   `caesar.io.mixins`: Analysis mixins
    *   `caesar.config`: Configuration management
    *   `gcell.dna.genome.ChromSize`: Chromosome size utilities
    *   `gcell.rna.gene.Gene`, `gcell.rna.gencode.Gencode`: Gene annotation

*   **External:**
    *   Core: `numpy`, `pandas`, `scipy` (sparse, ndimage)
    *   Storage: `zarr` (for Zarr backend), `bpcells` (for BPCells backend, optional)
    *   Analysis: `scanpy`, `anndata` (for single-cell analysis)
    *   Utilities: `logging`, `abc`, `typing`, `concurrent.futures`, `tqdm`, `re`
    *   Genomics: `pyranges` (for genomic interval operations)

### 9. Key Design Decisions

1. **Backend Abstraction:** The split between `GenomicIO` (backend-agnostic base) and specific backends (`ZarrIO`, `BPCellsIO`) allows for clean separation of concerns and easy addition of new storage formats.

2. **Data-Type Interfaces:** The `interface.py` classes (`DenseIO`, `SingleSparseIO`, etc.) provide a middle layer that defines behavior for different data types while remaining backend-agnostic.

3. **Mixin Architecture:** Analysis capabilities are added through mixins rather than inheritance, allowing flexible composition of features across different data types and backends.

4. **Query Objects:** Explicit query objects provide type safety and clear semantics for data selection, making it easier to add new query types and debug data access.

5. **Strategy Pattern:** Aggregators and Normalizers use the strategy pattern, allowing algorithms to be varied independently and making the system more testable and maintainable.

6. **Lazy Initialization:** Properties like `libsize`, `gencode`, and `per_cell_libsize` are calculated only when first accessed, improving performance for operations that don't need them.

7. **Gene Name Resolution:** Automatic resolution of gene names to `Gene` objects via the cached `gencode` property simplifies the API for gene-centric operations.

### 10. Version History

#### Version 2.0 (2025-12-22)
**Major Updates:**
- Complete restructuring to reflect current implementation
- Added comprehensive documentation of module structure
- Documented both Zarr and BPCells backend implementations
- Added section on multiome support
- Documented all current mixins including new ones:
  - `LibrarySizeAnnotationMixin`
  - `GenomicMaskingMixin`
  - `LabelMixin`
- Added utility modules documentation (AnnData integration, metadata, peak ordering, parallel processing)
- Updated inheritance hierarchy to reflect split between `genomic_io.py` and `interface.py`
- Added backend implementation details for Zarr and BPCells
- Documented `ListNormalizer` addition
- Added key design decisions section
- Updated extensibility guidelines with step-by-step instructions

**Key Changes:**
1. Module organization now clearly separated into:
   - Core abstractions (`genomic_io.py`, `interface.py`)
   - Backend implementations (`zarr_io/`, `bpcell_io.py`)
   - Supporting modules (query, aggregator, normalizer, metadata, etc.)
   - Analysis mixins (`mixins/`)

2. Complete documentation of all IO classes:
   - Base: `GenomicIO`
   - Interfaces: `DenseIO`, `SignalDenseIO`, `CelltypeDenseIO`, `SequenceDenseIO`, `SingleSparseIO`, `MultiSparseIO`
   - Zarr implementations: 9 classes including multiome support
   - BPCells implementations: 4 classes with compatibility wrappers

3. Enhanced extensibility documentation with concrete steps for adding new backends, queries, aggregators, normalizers, and analysis capabilities

#### Version 1.0 (Original)
- Initial design document focused on `interface.py`
- Basic documentation of `GenomicIO` and data-type classes
- Query and strategy pattern documentation

