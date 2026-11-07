from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any
from collections.abc import Sequence
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.sparse import csr_matrix

# Conditional import for type checking
if TYPE_CHECKING:
    from gcell.rna.gene import Gene
    from gcell.rna.gencode import Gencode

from gcell.dna.genome import ChromSize

from ..analysis.track import Track
from .aggregator import (
    AggregatedCoverage,
    Aggregator,
    ByKeyAggregator,
    ConcatAggregator,
    DataType,
    RawCoverageDict,
    SumAggregator,
)
from .normalizer import (
    DictNormalizer,
    FactorNormalizer,
    ListNormalizer,
    NoNormalizer,
    Normalizer,
    QuantileNormalizer,
)
from .query import (
    BaseQuery,
    DefaultQuery,
    IndexQuery,
    KeyQuery,
    MetadataQuery,
    MultiKeyQuery,
)


class GenomicIO(ABC):
    """Abstract base class for backend-agnostic genomic data I/O operations.

    Provides a unified interface for accessing and interacting with various genomic
    data storage formats (e.g., Zarr, BPCells, potential future formats). It defines
    the core contract that all backend-specific IO implementations must adhere to,
    ensuring interoperability and facilitating extensibility.

    The design utilizes Abstract Base Classes (ABCs), Strategy patterns for
    aggregation and normalization, and explicit Query objects (`BaseQuery`
    subclasses) to decouple data retrieval logic from specific backend implementations.

    Key concepts include:
        - Backend Agnosticism: Core methods operate on abstract interfaces.
        - Query Objects: Represent data selection criteria explicitly.
        - Strategy Pattern: Aggregators and Normalizers encapsulate algorithms.
        - Chromosome Management: Clear distinction between assembly-wide chromosome
          information (`assembly_chrom_sizes`), chromosomes present in the specific
          data source (`data_chroms`), and the currently active/used set of
          chromosomes (`_active_chroms`, accessed via the `chroms` property).
        - Lazy Loading: Properties like `libsize`, `per_cell_libsize`, and `gencode`
          are calculated/initialized only when first accessed.
        - Gene Name Resolution: Gene-related methods (`get_gene_track`, `get_gene_tss_track`,
          etc.) can accept either a `Gene` object or a gene name string. When a string
          is provided, it is automatically resolved to a `Gene` object using the
          cached `gencode` property.

    Subclasses are responsible for implementing abstract methods to handle the
    specifics of their underlying storage format, including data structure
    initialization, chunking logic, attribute loading/saving, and the core
    data retrieval method `_get_coverage_for_query`. Mixins can be added to
    subclasses to provide specialized analytical capabilities (e.g., footprinting).

    Attributes:
        path (str): Path to the data source (file or directory).
        mode (str): File opening mode ('r', 'a', 'r+', 'w').
        dtype (str): Expected data type of the genomic signal (e.g., 'float32').
        assembly (str | None): Genome assembly identifier (e.g., 'hg38'), determined
            by the subclass during initialization (`_init_data_structure`).
        assembly_chrom_sizes (dict[str, int] | None): Dictionary mapping all
            chromosome names to their lengths for the detected `assembly`. Loaded
            by `_load_assembly_chrom_sizes`.
        _active_chroms (list[str] | None): List of chromosome names currently
            considered active for processing. This is typically the intersection
            of `assembly_chrom_sizes.keys()` and `data_chroms`, potentially
            further filtered by methods like `leave_out_chromosomes`. Set by
            subclass during initialization (`_init_data_structure`). Access via
            the `chroms` property.
        chunk_size (int | None): Default chunk size used for processing, set by
            the subclass during `_init_chunking`.
        dir_chunked (bool | None): Indicates if the backend data is chunked by
            directory (relevant for specific Zarr formats). Set by subclass.
        _libsize (float | dict | None): Internal storage for total library size(s).
            Lazily calculated via the `libsize` property using `_calculate_libsize`.
        _per_cell_libsize (pd.Series | dict | None): Internal storage for per-cell
            library sizes (primarily for sparse formats). Lazily calculated via the
            `per_cell_libsize` property using `_calculate_per_cell_libsize`.
        _gencode (Gencode | None): Cached Gencode object for gene annotation and
            gene name resolution. Lazily initialized via the `gencode` property
            based on `self.assembly` when first accessed.
        _init_chunking_done (bool): Flag indicating if `_init_chunking` has run.

    Args:
        path (str): Path to the data file/directory.
        mode (str): Mode for opening files ('r': read-only, 'a': read/write,
                    'r+': read/write, 'w': write). Defaults to "r".
        dtype (str): Data type for the genomic data (e.g., 'float32', 'int8').
                     Defaults to "float32".

    Raises:
        ValueError: If an invalid `mode` is provided.
    """

    def __init__(self, path: str, mode: str = "r", dtype: str = "float32"):
        """
        Initialize base GenomicIO object.

        Deferred initialization: Core data structures, assembly information,
        and chunking details are set up by subclasses calling internal methods
        like `_init_data_structure()` and `_init_chunking()`.

        Args:
            path (str): Path to the data file/directory.
            mode (str): Mode for opening files ('r': read-only, 'a': append/write,
                        'r+': read/write, 'w': write). Defaults to "r".
            dtype (str): Data type for the genomic data (e.g., 'float32', 'int8').
                         Defaults to "float32".

        Raises:
            ValueError: If an invalid `mode` is provided.
        """
        if mode not in ["r", "a", "r+", "w"]:
            raise ValueError("Mode must be 'r', 'a', 'r+', or 'w'")

        self.path = path
        self.mode = mode
        self.dtype = dtype

        # --- Attributes related to Chromosomes ---
        # To be set by _init_data_structure in subclasses
        self.assembly: str | None = None  # Genome assembly (e.g., 'hg38')

        self.assembly_chrom_sizes: dict[str, int] | None = (
            None  # Full chromosome sizes for the assembly
        )
        self._active_chroms: list[str] | None = (
            None  # List of currently active chromosome names
        )

        # --- Other attributes to be set by subclasses ---
        self.chunk_size: int | None = (
            None  # Default chunk size for iteration/processing
        )
        self.dir_chunked: bool | None = (
            None  # Whether data is stored chunked by directory (relevant for some Zarr formats)
        )
        # --- Lazily calculated attributes ---
        self._libsize: float | dict | None = None
        self._per_cell_libsize: pd.Series | dict | None = None
        self._gencode: Gencode | None = None  # Cached Gencode object for gene resolution

        # Initialization is deferred to subclasses calling:
        # self._init_data_structure()
        # self._init_chunking()

    # --- Chromosome related Properties ---
    @property
    def data_chroms(self) -> list[str]:
        """
        Abstract property: Returns a sorted list of chromosome names
        present in the underlying data source. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the data_chroms property.")

    @property
    def chroms(self) -> list[str]:
        """Returns a sorted list of *active* chromosome names for this IO object."""
        # Ensures _active_chroms is initialized by _init_data_structure
        if self._active_chroms is None:
            # Trigger initialization if not already done explicitly
            # This might happen if properties are accessed before subclass init completes fully
            # or if init fails silently.
            logging.warning(
                "Accessing 'chroms' property before _active_chroms is set. Attempting initialization."
            )
            try:
                self._init_data_structure()
                if self._active_chroms is None:  # Check again after init attempt
                    logging.error("Initialization failed to set _active_chroms.")
                    return []
            except Exception as e:
                logging.error(f"Error during implicit initialization for 'chroms': {e}")
                return []

        # Return a copy to prevent external modification
        return sorted(list(self._active_chroms))

    @property
    def chrom_sizes(self) -> dict[str, int]:
        """Returns a dictionary of sizes for the *active* chromosomes."""
        if self._active_chroms is None or self.assembly_chrom_sizes is None:
            # Trigger initialization if not already done explicitly
            logging.warning(
                "Accessing 'chrom_sizes' property before chromosomes are fully initialized. Attempting initialization."
            )
            try:
                self._init_data_structure()
                if self._active_chroms is None or self.assembly_chrom_sizes is None:
                    logging.error("Initialization failed to set chromosome info.")
                    return {}
            except Exception as e:
                logging.error(
                    f"Error during implicit initialization for 'chrom_sizes': {e}"
                )
                return {}

        return {
            chrom: self.assembly_chrom_sizes[chrom]
            for chrom in self._active_chroms
            if chrom in self.assembly_chrom_sizes  # Ensure key exists
        }

    @property
    def chrom_n_chunks(self) -> dict[str, int] | None:
        """Returns a dictionary mapping *active* chromosome names to the number of chunks."""
        active_sizes = self.chrom_sizes  # Use the property for active sizes
        if not active_sizes or self.chunk_size is None:
            # Attempt initialization if needed
            logging.warning(
                "Accessing 'chrom_n_chunks' before chromosomes/chunk_size initialized. Attempting initialization."
            )
            try:
                if (
                    not hasattr(self, "_init_chunking_done")
                    or not self._init_chunking_done
                ):
                    self._init_chunking()
                if not self.chrom_sizes or self.chunk_size is None:
                    logging.error(
                        "Initialization failed to set chromosome sizes or chunk size."
                    )
                    return None
                active_sizes = self.chrom_sizes  # Re-fetch after potential init
            except Exception as e:
                logging.error(
                    f"Error during implicit initialization for 'chrom_n_chunks': {e}"
                )
                return None

        n_chunks = {}
        for chrom, size in active_sizes.items():
            # Ensure size and chunk_size are valid before division
            if size >= 0 and self.chunk_size > 0:
                n_chunks[chrom] = (size + self.chunk_size - 1) // self.chunk_size
            else:
                logging.warning(
                    f"Invalid size ({size}) or chunk_size ({self.chunk_size}) for chrom {chrom}. Setting chunks to 0."
                )
                n_chunks[chrom] = 0
        return n_chunks

    # --- Library Size Properties ---

    @property
    def libsize(self) -> float | dict | None:
        """
        Total library size or dictionary of library sizes (e.g., per cell type).

        If library size annotation has been performed (via LibrarySizeAnnotationMixin),
        returns the non-blacklist library size. Otherwise, calculates library size
        using the standard method.

        Returns
        -------
        float | dict | None
            Library size value(s). For annotated data, returns non-blacklist counts.
        """
        if self._libsize is None:
            # Check if library size annotation exists
            # For Zarr: attrs are in self.dataset.attrs
            # For BPCells: attrs are in self.attrs (loaded from attrs.json)
            attrs = {}
            if hasattr(self, "dataset") and self.dataset is not None:
                attrs = getattr(self.dataset, "attrs", {})
            # Also check self.attrs for BPCellsIO (which loads from attrs.json)
            if hasattr(self, "attrs") and isinstance(self.attrs, dict):
                # Merge attrs, with self.attrs taking precedence for BPCells
                attrs = {**attrs, **self.attrs}

            # Check for annotated library sizes
            if (
                "library_size_annotated" in attrs
                and attrs["library_size_annotated"]
            ):
                # Single-track format
                if "library_size_nonblacklist" in attrs and not isinstance(
                    attrs["library_size_nonblacklist"], dict
                ):
                    self._libsize = attrs["library_size_nonblacklist"]
                    logging.info("Using annotated non-blacklist library size.")
                # Multi-track format
                elif "library_size_nonblacklist" in attrs and isinstance(
                    attrs["library_size_nonblacklist"], dict
                ):
                    self._libsize = attrs["library_size_nonblacklist"]
                    logging.info("Using annotated non-blacklist library sizes.")
                else:
                    # Fallback to standard calculation
                    logging.info(
                        "Library size annotation found but non-blacklist values missing. "
                        "Falling back to standard calculation."
                    )
                    self._libsize = self._calculate_libsize()
            else:
                # No annotation found, calculate normally
                self._libsize = self._calculate_libsize()
                if self.mode in ["r"]:
                    logging.info(
                        "Library size calculated without blacklist filtering. "
                        "Consider running the library size annotator for more accurate values: "
                        "zarr_io.annotate_library_sizes() or bpcells_io.annotate_library_sizes()"
                    )

        return self._libsize

    @property
    def per_cell_libsize(self) -> pd.Series | dict | None:
        """
        Per-cell library sizes (for sparse formats).
        Calculated on first access if not already computed.
        """
        if self._per_cell_libsize is None:
            self._per_cell_libsize = self._calculate_per_cell_libsize()
        return self._per_cell_libsize

    def _get_default_gencode_version(self, assembly: str) -> int:
        """
        Determine the default GENCODE version for a given assembly.

        GENCODE uses different version numbering schemes for human and mouse:
        - Human (hg38, hg19): Numeric versions (e.g., 40, 44)
        - Mouse (mm10, mm39): M-prefixed versions (e.g., M10, M38) which correspond
          to integer versions (10, 38) in the API

        Parameters
        ----------
        assembly : str
            Genome assembly identifier (e.g., 'hg38', 'mm10', 'mm39').

        Returns
        -------
        int
            Default GENCODE version number for the assembly.
        """
        assembly_lower = assembly.lower()

        # Human assemblies - use version 40 as default (widely used, stable)
        if assembly_lower in ["hg38", "grch38"]:
            return 40
        elif assembly_lower in ["hg19", "grch37"]:
            return 40  # Also version 40 for hg19 (lifted over)

        # Mouse assemblies - use appropriate M-version defaults
        elif assembly_lower in ["mm10"]:
            return 10  # M10 for mm10
        elif assembly_lower in ["mm39"]:
            return 38  # M38 for mm39 (latest for mm39)
        else:
            # Default fallback - try version 40 for unknown assemblies
            # This might work for some assemblies, or will fail gracefully
            logging.warning(
                f"Unknown assembly '{assembly}', using default GENCODE version 40. "
                "This may not be appropriate for this assembly."
            )
            return 40

    @property
    def gencode(self) -> Gencode | None:
        """
        Lazily loaded and cached Gencode object for gene annotation.

        The Gencode object is automatically initialized based on `self.assembly`
        when first accessed. It is cached for subsequent use to avoid repeated
        initialization. The default GENCODE version is automatically determined
        based on the assembly:
        - Human assemblies (hg38, hg19): version 40
        - Mouse mm10: version 10 (M10)
        - Mouse mm39: version 38 (M38)

        Returns
        -------
        Gencode | None
            A Gencode instance for the current assembly, or None if assembly
            is not set or Gencode cannot be imported/initialized.

        Raises
        ------
        ImportError
            If the `gcell` package is not installed.
        ValueError
            If `self.assembly` is not set or is invalid.
        """
        if self._gencode is None:
            if self.assembly is None:
                logging.warning(
                    "Cannot initialize Gencode: assembly is not set. "
                    "Set assembly attribute or call _init_data_structure() first."
                )
                return None

            try:
                from gcell.rna.gencode import Gencode

                # Determine appropriate default version based on assembly
                default_version = self._get_default_gencode_version(self.assembly)
                self._gencode = Gencode(assembly=self.assembly, version=default_version)
                logging.debug(
                    f"Initialized Gencode object for assembly {self.assembly} (version {default_version})"
                )
            except ImportError as e:
                logging.error(
                    f"Failed to import Gencode from gcell: {e}. "
                    "Please install gcell package to use gene name resolution."
                )
                return None
            except Exception as e:
                logging.error(
                    f"Failed to initialize Gencode for assembly {self.assembly}: {e}"
                )
                return None

        return self._gencode

    @staticmethod
    def detect_store_type(path: str | Path) -> str | None:
        """Detect store type from path without instantiating any IO class.

        Tries both Zarr and BPCells backends to determine the store type.
        This is a convenience method that delegates to backend-specific classmethods.

        Args:
            path: Path to the store

        Returns
        -------
        str | None
            Store type identifier (e.g., "celltype_dense", "bpcells_celltype_dense"),
            or None if the store type cannot be determined.
        """
        from pathlib import Path
        from caesar.io.zarr_io.base import ZarrIO
        from caesar.io.bpcell_io import BPCellsIO

        if path is None:
            raise ValueError("Cannot detect store type: path is None")
        
        path = Path(path)

        # Try Zarr first
        zarr_type = ZarrIO.detect_store_type(path)
        if zarr_type is not None:
            return zarr_type

        # Try BPCells
        bpcells_type = BPCellsIO.detect_store_type(path)
        if bpcells_type is not None:
            return bpcells_type

        return None

    @property
    def store_type(self) -> str | None:
        """
        Store type identifier for the underlying data format.

        Returns a string identifying the store type (e.g., "celltype_dense", "single_sparse")
        based on backend-specific metadata. This property provides a unified interface
        for determining the store type across different backends (Zarr, BPCells, etc.).

        Returns
        -------
        str | None
            Store type identifier, or None if the store type cannot be determined.

        Notes
        -----
        This property delegates to the backend-specific `_get_store_type()` method,
        which checks metadata files (.zattrs for Zarr, attrs.json for BPCells) first,
        then falls back to structural detection if metadata is not available.
        """
        return self._get_store_type()

    @abstractmethod
    def _get_store_type(self) -> str | None:
        """
        Backend-specific method to determine the store type.

        Should check metadata files first (e.g., .zattrs for Zarr, attrs.json for BPCells),
        then fall back to structural detection if metadata is not available.

        Returns
        -------
        str | None
            Store type identifier, or None if the store type cannot be determined.
        """
        raise NotImplementedError("Subclasses must implement _get_store_type.")

    def _resolve_gene(self, gene: Gene | str) -> Gene:
        """
        Resolve a gene identifier (Gene object or gene name string) to a Gene object.

        If `gene` is already a Gene object, it is returned as-is. If it is a string,
        the cached Gencode object is used to look up and return the corresponding
        Gene object.

        Parameters
        ----------
        gene : Gene | str
            Either a Gene object or a gene name string (e.g., "TP53", "MYC").

        Returns
        -------
        Gene
            A Gene object corresponding to the input.

        Raises
        ------
        ValueError
            If `gene` is a string but cannot be resolved to a Gene object,
            or if Gencode is not available.
        TypeError
            If `gene` is neither a Gene object nor a string.
        """
        if isinstance(gene, str):
            gencode_obj = self.gencode
            if gencode_obj is None:
                raise ValueError(
                    "Cannot resolve gene name: Gencode object is not available. "
                    "Ensure assembly is set and gcell package is installed."
                )

            try:
                return gencode_obj.get_gene(gene)
            except (KeyError, IndexError, AttributeError) as e:
                raise ValueError(
                    f"Gene '{gene}' not found in Gencode annotation for assembly {self.assembly}"
                ) from e
        elif hasattr(gene, "genomic_range"):  # Check if it's a Gene-like object
            return gene
        else:
            raise TypeError(
                f"Expected Gene object or gene name string, got {type(gene)}"
            )

    # --- Helper to load dataset and assembly information ---

    @abstractmethod
    def _setup_dataset(self) -> None:
        """Sets up the dataset."""
        raise NotImplementedError("Subclasses must implement _setup_dataset.")

    def _determine_assembly(self) -> None:
        """Determine and set the genome assembly identifier (`self.assembly`).

        Attempts to load the assembly information from the dataset's attributes
        (using `_load_attr`). If loading fails or the dataset cannot be accessed,
        it logs a warning and falls back to a default assembly (e.g., "hg38").
        This method should be called during the initialization process, typically
        within `_init_data_structure`.

        Side Effects:
            Sets the `self.assembly` attribute. Logs warnings on failure or fallback.
        """
        if self.dataset is not None:
            try:
                # Ensure attrs is accessed safely
                self.assembly = self._load_attr("assembly")
            except Exception as e:
                logging.warning(
                    f"Error accessing Zarr attributes for assembly: {e}, using hg38 as fallback"
                )
                self.assembly = "hg38"  # Fallback
            if self.assembly is None:
                logging.warning(
                    "Error accessing Zarr attributes for assembly: using hg38 as fallback"
                )
                self.assembly = "hg38"  # Fallback
        else:
            logging.warning(
                f"Zarr dataset at {self.path} could not be loaded, using hg38 as fallback"
            )
            self.assembly = "hg38"  # Default if dataset failed to load

    def _load_assembly_chrom_sizes(self) -> None:
        """Loads chromosome sizes based on self.assembly."""
        if self.assembly:
            try:
                self.assembly_chrom_sizes = ChromSize(self.assembly).dict
                logging.debug(f"Loaded chromosome sizes for assembly: {self.assembly}")
            except Exception as e:
                logging.warning(
                    f"Could not load chromosome sizes for assembly '{self.assembly}': {e}. "
                    f"Assembly information might be incomplete."
                )
                self.assembly_chrom_sizes = {}  # Set to empty dict on failure
        else:
            logging.warning(
                "No assembly specified. Cannot load assembly chromosome sizes."
            )
            self.assembly_chrom_sizes = {}  # Set to empty dict if no assembly

    # --- Abstract Methods (Must be implemented by subclasses) ---

    @abstractmethod
    def _init_data_structure(self) -> None:
        """
        Initialize data structure specific attributes.
        Should set `self.assembly`, load `self.assembly_chrom_sizes`,
        determine `data_chroms`, set `self._active_chroms`, and potentially call
        `_init_chunking`. Loads metadata, identifies dataset structure.
        """
        # Base implementation attempt to load assembly sizes
        self._load_assembly_chrom_sizes()
        # Subclasses must implement the rest, including setting _active_chroms

    @abstractmethod
    def _init_chunking(self, first_chrom: str | None = None) -> None:
        """
        Initialize chunking-related attributes.
        Should set `self.chunk_size`. Might calculate `chrom_n_chunks` or rely
        on the property implementation. Marks initialization state.
        """
        self._init_chunking_done = True  # Mark as done

    @abstractmethod
    def _load_attr(self, key) -> None:
        """Load attributes from the backend storage."""
        pass

    @abstractmethod
    def _save_attr(self, key, value) -> None:
        """Save attributes to the backend storage."""
        pass

    @abstractmethod
    def _get_coverage_for_query(
        self, chr_name: str, start: int, end: int, query: BaseQuery, sparse: bool
    ) -> RawCoverageDict:
        """
        Retrieve coverage data from the backend for a given region and query object.

        Args:
            chr_name: Chromosome name.
            start: Start coordinate (0-based).
            end: End coordinate (exclusive).
            query: A BaseQuery object specifying the data subset.
            sparse: Whether to return data in sparse format (if supported).

        Returns:
            A dictionary mapping group identifiers (str) to their corresponding
            coverage data (np.ndarray or csr_matrix). Even for queries resulting
            in a single effective group, a dict should be returned (e.g., {'default': data}).
        """
        pass

    @abstractmethod
    def get_default_track_obj(
        self,
        chr_name: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> Track:
        """
        Get a default Track object, typically representing aggregated or
        primary signal for the dataset over the specified region.

        Args:
            chr_name: Chromosome name.
            start: Start coordinate.
            end: End coordinate.

        Returns:
            A caesar.analysis.Track instance.
        """
        pass

    @abstractmethod
    def _get_all_ids(self) -> list | dict:
        """
        Return all primary identifiers managed by this IO instance.
        - For single-sample/dense data, might return a default list like ['default'].
        - For multi-sample or celltype data, returns list/dict keys representing
          the samples or celltypes.
        """
        pass

    @abstractmethod
    def _calculate_libsize(self) -> None:
        """
        Calculate and store the library size(s) in `self._libsize`.
        Implementation depends heavily on the data format (dense vs sparse,
        single vs multi-sample). Should handle potential errors gracefully.
        Should always return the value, rather than setting self._libsize.
        """
        # Ensure subclasses implement this and set self._libsize
        raise NotImplementedError("Subclasses must implement _calculate_libsize.")

    @abstractmethod
    def _calculate_per_cell_libsize(self) -> None:
        """
        Calculate and store per-cell library sizes in `self._per_cell_libsize`.
        Only relevant for sparse formats. Should handle potential errors gracefully.
        Should always return the value, rather than setting self._per_cell_libsize.
        """
        # Ensure subclasses implement this and set self._per_cell_libsize
        # For non-sparse IO, this can be a no-op setting self._per_cell_libsize = None
        raise NotImplementedError(
            "Subclasses must implement _calculate_per_cell_libsize."
        )

    # --- Helper for Query Parsing ---
    def _parse_query(self, query: Any) -> BaseQuery:
        """Converts user-provided query input into a BaseQuery object."""
        if isinstance(query, BaseQuery):
            return query
        elif query is None:
            return DefaultQuery()
        elif isinstance(query, str):
            # Assume metadata query for strings
            return MetadataQuery(query)
        elif isinstance(query, list | tuple | np.ndarray | pd.Series):
            # Try converting to int indices first
            try:
                indices = np.asarray(query, dtype=int)
                # Basic sanity check for negative indices
                if np.any(indices < 0):
                    raise ValueError("Indices cannot be negative.")
                return IndexQuery(indices)
            except (ValueError, TypeError):
                # If not integers, assume they are keys (e.g., sample names)
                keys = [str(k) for k in query]
                return KeyQuery(keys)
        elif isinstance(query, dict):
            # Assume MultiKeyQuery structure {group_name: keys/indices}
            # We might need deeper validation here depending on expected structure
            return MultiKeyQuery(query)
        else:
            raise TypeError(
                f"Unsupported query type: {type(query)}. Use None, str, list/tuple, dict, or BaseQuery object."
            )

    def _parse_aggregator(self, aggregator):
        """Convert string aggregators to objects for backward compatibility"""
        if isinstance(aggregator, str):
            aggregator_map = {
                "sum": SumAggregator(),
                "by_key": ByKeyAggregator(),  # Legacy support
                "concat": ConcatAggregator(),
            }
            if aggregator not in aggregator_map:
                raise ValueError(
                    f"Unknown aggregator: {aggregator}. Choose from: sum, by_key, concat."
                )
            return aggregator_map[aggregator]
        return aggregator or SumAggregator()  # Default to sum

    def _parse_normalizer(self, normalizer):
        """Convert string normalizers to objects for convenience.
        
        Supported string shortcuts:
        - 'none' or 'no': NoNormalizer (no normalization)
        - 'qn' or 'quantile': QuantileNormalizer (uses normalization params from IO object)
        
        Args:
            normalizer: Normalizer instance or string shortcut
            
        Returns:
            Normalizer instance
        """
        if isinstance(normalizer, str):
            normalizer_lower = normalizer.lower()
            if normalizer_lower in ["none", "no"]:
                return NoNormalizer()
            elif normalizer_lower in ["qn", "quantile"]:
                # QuantileNormalizer needs the IO object to load normalization params
                return QuantileNormalizer(self)
            else:
                raise ValueError(
                    f"Unknown normalizer shortcut: {normalizer}. "
                    f"Choose from: 'none'/'no', 'qn'/'quantile'. "
                    f"Or provide a Normalizer instance directly."
                )
        return normalizer or NoNormalizer()  # Default to no normalization

    # --- Get Track Methods ---

    def get_track(
        self,
        chr_name: str,
        start: int,
        end: int,
        query: Any = None,
        aggregator: str | Aggregator = SumAggregator(),
        normalizer: Normalizer | str = NoNormalizer(),
        normalize_factor: float | dict | None = None,  # Compatibility parameter
        sparse: bool = False,
        output_format: str = "track_object",  # 'track_object', 'raw_array', 'raw_dict', 'raw_list'
        track_label: str | None = None,
        conv_size: int | None = None,  # Add conv_size parameter
        **kwargs,
    ) -> Track | AggregatedCoverage:
        """
        Retrieve, aggregate, normalize, and format coverage data for a genomic region.

        This is the central method for accessing genomic coverage data. It performs
        the following steps:
        1. Parses the `query` into a `BaseQuery` object.
        2. Retrieves raw coverage data using `_get_coverage_for_query`.
        3. Aggregates the raw data using the specified `aggregator`.
        4. Applies convolution smoothing if `conv_size` is provided.
        5. Applies normalization using the specified `normalizer`.
        6. Formats the result according to `output_format`.

        Parameters
        ----------
        chr_name : str
            Chromosome name (must be in `self.chroms`).
        start : int
            Start coordinate (0-based, inclusive).
        end : int
            End coordinate (0-based, exclusive).
        query : Any, optional
            Specifies the subset of data to retrieve.
            Can be None (default), str (metadata query), list/tuple/array
            (indices or keys), dict (MultiKeyQuery), or a `BaseQuery`
            subclass instance. Defaults to None (interpreted as `DefaultQuery`).
        aggregator : str | Aggregator, optional
            An `Aggregator` instance or string ('sum', 'by_key', 'concat') to combine
            data retrieved by the query. Defaults to `SumAggregator()`.
        normalizer : Normalizer | str, optional
            A `Normalizer` instance or string shortcut to apply to the aggregated data.
            String shortcuts:
            - 'none' or 'no': No normalization (default)
            - 'qn' or 'quantile': Quantile normalization using params from IO object
            Normalization applicability depends on the `output_format`.
            Defaults to `NoNormalizer()`.
        normalize_factor : float | dict | None, optional
            Compatibility parameter for backward compatibility with older code.
            If provided, overrides the `normalizer` parameter:
            - float/int: Converted to `FactorNormalizer(normalize_factor)`
            - dict: Converted to `DictNormalizer(normalize_factor)`
            - None: No effect (uses the provided `normalizer`)
            Defaults to None.
        sparse : bool, optional
            Whether to request sparse data format (e.g., csr_matrix) from the backend.
            Backend support varies. Defaults to False.
        output_format : str, optional
            Controls the return type:
            - 'track_object': Return a `Track` object (default). Normalization
              and convolution (via `Track` constructor) are applied.
            - 'raw_array': Return a single `np.ndarray` or `csr_matrix`.
              Requires an aggregator like `SumAggregator`. Normalization applied.
              Convolution is applied if `conv_size` is provided.
            - 'raw_dict': Return a dictionary `{key: data}` (e.g., from
              `ByKeyAggregator`). Normalization and convolution are applied
              if specified.
            - 'raw_list': Return a list of arrays or matrices (e.g., from
              `ConcatAggregator`). Normalization and convolution are applied
              if specified.
        track_label : str | None, optional
            Base label for tracks when `output_format='track_object'`.
            Usage depends on the aggregator. Defaults to None.
        conv_size : int | None, optional
            Size of the convolution window for smoothing. Applied to all output formats
            when provided. For 'track_object', passed to the `Track` constructor.
            Defaults to None (no smoothing).
        **kwargs
            Additional keyword arguments passed *only* to the `Track`
            constructor when `output_format='track_object'`.

        Returns
        -------
        Track | AggregatedCoverage
            The processed coverage data. The exact type depends on `output_format`:
            - `Track`: If `output_format='track_object'`.
            - `np.ndarray | csr_matrix`: If `output_format='raw_array'`.
            - `dict[str, np.ndarray | csr_matrix]`: If `output_format='raw_dict'`.
            - `list[np.ndarray | csr_matrix]`: If `output_format='raw_list'`.

        Raises
        ------
        TypeError
            If `aggregator` or `normalizer` are not valid instances, or
            if the aggregated data type is incompatible with the
            requested `output_format`.
        ValueError
            If `output_format` is invalid, `chr_name` is not active,
            or coordinates are invalid.
        NotImplementedError
            If the backend does not support sparse data when
            `sparse=True` is requested.
        Exception
            Propagates exceptions from backend data retrieval,
            aggregation, or normalization steps.
        """
        # --- Input Validation ---
        # Parse aggregator (handles string conversion)
        aggregator = self._parse_aggregator(aggregator)
        if not isinstance(aggregator, Aggregator):
            raise TypeError(
                "aggregator must be an instance of the Aggregator base class."
            )

        # If normalize_factor is provided, convert to appropriate Normalizer (for backward compatibility)
        if normalize_factor is not None:
            if isinstance(normalize_factor, int | float):
                normalizer = FactorNormalizer(normalize_factor)
            elif isinstance(normalize_factor, dict):
                normalizer = DictNormalizer(normalize_factor)
            elif isinstance(
                normalize_factor, list | Sequence | np.ndarray
            ) and not isinstance(normalize_factor, str):
                normalizer = ListNormalizer(normalize_factor)
            else:
                raise TypeError(
                    f"normalize_factor must be a float, int, dict, list, sequence, or numpy array, got {type(normalize_factor)}"
                )
        else:
            # Parse normalizer (handles string conversion)
            normalizer = self._parse_normalizer(normalizer)

        if not isinstance(normalizer, Normalizer):
            raise TypeError(
                "normalizer must be an instance of the Normalizer base class."
            )
        valid_formats = ["track_object", "raw_array", "raw_dict", "raw_list"]
        if output_format not in valid_formats:
            raise ValueError(
                f"Invalid output_format '{output_format}'. Choose from: {valid_formats}"
            )
        if chr_name not in self.chroms:  # Check against active chromosomes
            raise ValueError(
                f"Chromosome '{chr_name}' not found in active chromosomes for this dataset: {self.chroms}"
            )

        # --- Core Logic ---
        # 1. Parse Query and Get Raw Data
        parsed_query = self._parse_query(query)
        raw_coverage_dict = self._get_coverage_for_query(
            chr_name=chr_name, start=start, end=end, query=parsed_query, sparse=sparse
        )

        # 2. Aggregate Data
        aggregated_data = aggregator.aggregate(raw_coverage_dict)
        track_keys = aggregator.get_track_keys(raw_coverage_dict)

        # Handle empty aggregation result - return zeros with correct shape
        region_size = end - start
        if isinstance(aggregated_data, np.ndarray) and aggregated_data.size == 0:
            if sparse:
                aggregated_data = csr_matrix((1, region_size), dtype=self.dtype)
            else:
                aggregated_data = np.zeros(region_size, dtype=self.dtype)

        # --- Output Formatting ---
        if output_format == "track_object":
            # 3a. Prepare 'tracks' dict for Track constructor
            tracks_for_obj: dict[str, DataType] = {}
            final_track_keys: list[str] = []

            if isinstance(aggregator, SumAggregator):
                key = track_label if track_label is not None else "default_sum"
                if isinstance(aggregated_data, np.ndarray | csr_matrix):
                    tracks_for_obj[key] = aggregated_data
                    final_track_keys.append(key)
            elif isinstance(aggregator, ByKeyAggregator):
                if isinstance(aggregated_data, dict):
                    tracks_for_obj = aggregated_data
                    final_track_keys = (
                        track_keys  # these are the keys from the aggregator
                    )
            elif isinstance(aggregator, ConcatAggregator):
                base_label = track_label if track_label is not None else "track"
                # track_keys from ConcatAggregator are the original keys that were concatenated
                # final_track_keys need to be generated if track_keys is not suitable
                if isinstance(
                    aggregated_data, list
                ):  # sparse concat, one matrix per original key
                    if len(track_keys) == len(aggregated_data):
                        final_track_keys = track_keys
                        for i, tk in enumerate(final_track_keys):
                            tracks_for_obj[tk] = aggregated_data[i]
                    else:
                        # if keys dont match data, generate generic keys
                        final_track_keys = [
                            f"{base_label}_{i}" for i in range(len(aggregated_data))
                        ]
                        for i, tk in enumerate(final_track_keys):
                            tracks_for_obj[tk] = aggregated_data[i]
                        logging.warning(
                            "ConcatAggregator (sparse): track_keys from aggregator did not match data length. Generated generic keys."
                        )
                elif isinstance(aggregated_data, np.ndarray):  # Dense concat
                    if aggregated_data.ndim == 0 and aggregated_data.size == 0:
                        pass  # empty
                    elif aggregated_data.ndim == 1:  # single track result from concat
                        key = (
                            track_label
                            if track_label is not None
                            else (track_keys[0] if track_keys else "concat_track")
                        )
                        tracks_for_obj[key] = aggregated_data
                        final_track_keys.append(key)
                    elif aggregated_data.ndim == 2:  # (n_tracks, length)
                        if len(track_keys) == aggregated_data.shape[0]:
                            final_track_keys = track_keys
                        else:
                            final_track_keys = [
                                f"{base_label}_{i}"
                                for i in range(aggregated_data.shape[0])
                            ]
                            if track_keys:
                                logging.warning(
                                    "ConcatAggregator (dense): track_keys from aggregator did not match data shape. Generated generic keys."
                                )
                        for i, tk in enumerate(final_track_keys):
                            tracks_for_obj[tk] = aggregated_data[i]
                    else:
                        logging.warning("Concat key/data shape mismatch (dense)")

            # 4a. Prepare Normalization Factors for Track
            # Ensure final_track_keys are used for preparing norm_factors
            # norm_factors = normalizer.prepare_norm_factors(final_track_keys) # No longer needed for Track construction

            # 5a. Create Track Object
            # Pass the normalizer object directly to the Track constructor
            return Track(
                chrom=chr_name,
                start=start,
                end=end,
                assembly=self.assembly if self.assembly else "unknown",
                tracks=tracks_for_obj,  # These are the raw aggregated tracks
                normalizer=normalizer,  # Pass the Normalizer object
                conv_size=conv_size,  # Pass conv_size to Track constructor
                **kwargs,
            )

        # --- Handle raw output formats with unified convolution and normalization ---
        else:  # raw_array, raw_dict, raw_list
            processed_data = aggregated_data

            # 3b. Apply Convolution if conv_size is provided
            if conv_size is not None and conv_size > 1:

                kernel = np.ones(conv_size) / conv_size
                if output_format == "raw_array":

                    if isinstance(processed_data, np.ndarray):
                        processed_data = processed_data.astype(np.float32)
                        if processed_data.ndim == 1 and processed_data.size > 0:
                            processed_data = convolve1d(
                                processed_data, kernel, mode="nearest", axis=0
                            )
                        elif (
                            processed_data.ndim == 2 and processed_data.shape[1] > 0
                        ):  # (N, M) apply along M
                            processed_data = convolve1d(
                                processed_data, kernel, mode="nearest", axis=1
                            )
                        elif processed_data.size == 0:
                            pass  # no data to convolve
                        else:
                            logging.warning(
                                f"Convolution for raw_array: Unsupported ndarray shape {processed_data.shape}. Skipping."
                            )
                    elif isinstance(processed_data, csr_matrix):
                        logging.warning(
                            "Convolution for raw_array: Not directly supported for csr_matrix. Skipping."
                        )
                elif output_format == "raw_dict":
                    if isinstance(processed_data, dict):
                        for key, value in processed_data.items():
                            if isinstance(value, np.ndarray):
                                value = value.astype(np.float32)
                                if value.ndim == 1 and value.size > 0:
                                    processed_data[key] = convolve1d(
                                        value, kernel, mode="nearest", axis=0
                                    )
                                elif value.ndim == 2 and value.shape[1] > 0:
                                    processed_data[key] = convolve1d(
                                        value, kernel, mode="nearest", axis=1
                                    )
                                elif value.size == 0:
                                    pass  # no data to convolve
                                else:
                                    logging.warning(
                                        f"Convolution for raw_dict key '{key}': Unsupported ndarray shape {value.shape}. Skipping."
                                    )
                            elif isinstance(value, csr_matrix):
                                logging.warning(
                                    f"Convolution for raw_dict key '{key}': Not supported for csr_matrix. Skipping."
                                )
                elif output_format == "raw_list":
                    if isinstance(processed_data, list):
                        new_list = []
                        for i, item in enumerate(processed_data):
                            if isinstance(item, np.ndarray):
                                if item.ndim == 1 and item.size > 0:
                                    new_list.append(
                                        convolve1d(
                                            item, kernel, mode="nearest", axis=0
                                        )
                                    )
                                elif item.ndim == 2 and item.shape[1] > 0:
                                    new_list.append(
                                        convolve1d(
                                            item, kernel, mode="nearest", axis=1
                                        )
                                    )
                                elif item.size == 0:
                                    new_list.append(item)  # keep empty item
                                else:
                                    logging.warning(
                                        f"Convolution for raw_list item {i}: Unsupported ndarray shape {item.shape}. Keeping original."
                                    )
                                    new_list.append(item)
                            elif isinstance(item, csr_matrix):
                                logging.warning(
                                    f"Convolution for raw_list item {i}: Not supported for csr_matrix. Keeping original."
                                )
                                new_list.append(item)
                            else:
                                new_list.append(
                                    item
                                )  # keep non-array/csr items as is
                        processed_data = new_list

            # 4b. Apply Normalization via normalizer.transform
            normalized_data = normalizer.transform(processed_data)

            # 5b. Validate output type and return
            if output_format == "raw_array":
                if not isinstance(normalized_data, np.ndarray | csr_matrix):
                    # This might occur if normalizer.transform changed the type unexpectedly
                    # or if initial aggregation was not array/csr_matrix and it wasn't caught.
                    raise TypeError(
                        f"Output_format 'raw_array' expects np.ndarray or csr_matrix, "
                        f"but received {type(normalized_data)} after normalization."
                    )
                return normalized_data
            elif output_format == "raw_dict":
                if not isinstance(normalized_data, dict):
                    # If aggregation resulted in single array and normalizer kept it that way.
                    if isinstance(
                        normalized_data, np.ndarray | csr_matrix
                    ) and isinstance(aggregator, SumAggregator):
                        # Try to get a key. If track_keys is empty, use a default.
                        key_for_single_array = (
                            track_keys[0]
                            if track_keys
                            else track_label
                            if track_label
                            else "default_track"
                        )
                        logging.info(
                            f"raw_dict output: wrapping single array into dict with key '{key_for_single_array}'."
                        )
                        return {key_for_single_array: normalized_data}
                    raise TypeError(
                        f"Output_format 'raw_dict' expects dict, "
                        f"but received {type(normalized_data)} after normalization."
                    )
                return normalized_data
            elif output_format == "raw_list":
                if not isinstance(normalized_data, list):
                    raise TypeError(
                        f"Output_format 'raw_list' expects list, "
                        f"but received {type(normalized_data)} after normalization."
                    )
                return normalized_data
            else:
                # Should be caught by initial validation, but as a fallback
                raise ValueError(
                    f"Internal error: Unhandled output_format '{output_format}' after processing"
                )

    def get_track_for_regions(
        self, chr_name: str, region_list: list[tuple[int, int]], **kwargs
    ) -> list[Track | AggregatedCoverage]:
        """Retrieves track data for multiple regions on the same chromosome.

        Optimized to load data in chunks rather than retrieving each region
        individually. Groups regions by chunk, loads each chunk once, then
        extracts all regions belonging to that chunk.

        Args:
            chr_name (str): Chromosome name (must be in `self.chroms`).
            region_list (list[tuple[int, int]]): A list where each element is a
                tuple `(start, end)` representing a genomic region.
                Coordinates are 0-based (start inclusive, end exclusive).
            **kwargs: Additional arguments passed directly to the underlying
                `get_track` call for each region. Common arguments include
                `query`, `aggregator`, `normalizer`, `sparse`, `output_format`.

        Returns:
            list[Track | AggregatedCoverage]: A list containing the processed track
            data for each valid region in `region_list`. The type of each element
            corresponds to the return type of `get_track` based on the provided
            `kwargs` (defaults to `Track` object if `output_format` is not set).

        Raises:
            ValueError: If `chr_name` is not found in active chromosomes.
                        Propagates exceptions from `get_track`.
        """
        # Validate chromosome
        active_sizes = self.chrom_sizes
        if chr_name not in active_sizes:
            raise ValueError(
                f"Chromosome {chr_name} not found in active chromosome sizes"
            )

        chr_size = active_sizes[chr_name]

        # Filter and validate regions
        valid_regions = []
        region_indices = []  # Track original indices for result ordering

        for i, (start, end) in enumerate(region_list):
            # Validate coordinates
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
            ):
                logging.warning(
                    f"Skipping invalid region ({start}, {end}) on {chr_name}"
                )
                continue

            # Clip to chromosome bounds
            start = max(0, start)
            end = min(chr_size, end)

            if start >= end:
                logging.warning(
                    f"Skipping empty region after bounds check ({start}, {end}) on {chr_name}"
                )
                continue

            valid_regions.append((start, end))
            region_indices.append(i)

        if not valid_regions:
            return []

        # Group regions by chunk
        chunk_size = getattr(self, "chunk_size", 2000000)  # Default 1MB chunks
        chunk_regions = {}  # chunk_idx -> [(region_idx, start, end, start_in_chunk, end_in_chunk)]

        for region_idx, (start, end) in enumerate(valid_regions):
            start_chunk = start // chunk_size
            end_chunk = (end - 1) // chunk_size  # -1 because end is exclusive

            # Handle regions that span multiple chunks
            for chunk_idx in range(start_chunk, end_chunk + 1):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, chr_size)

                # Calculate region boundaries within this chunk
                region_start_in_chunk = max(0, start - chunk_start)
                region_end_in_chunk = min(chunk_end - chunk_start, end - chunk_start)

                if region_start_in_chunk < region_end_in_chunk:
                    if chunk_idx not in chunk_regions:
                        chunk_regions[chunk_idx] = []

                    chunk_regions[chunk_idx].append(
                        (
                            region_idx,
                            start,
                            end,
                            region_start_in_chunk,
                            region_end_in_chunk,
                        )
                    )

        # Initialize results array - handle case where all regions have same length
        region_lengths = [end - start for start, end in valid_regions]
        all_same_length = len(set(region_lengths)) == 1

        if all_same_length:
            # Pre-allocate fixed-size array
            region_length = region_lengths[0]
            # Determine output format to know result type
            output_format = kwargs.get("output_format", "track")
            if output_format == "raw_array":
                # Get a sample to determine the actual shape for sequence data
                sample_track = self.get_track(
                    chr_name, valid_regions[0][0], valid_regions[0][1], **kwargs
                )
                if sample_track.ndim > 1:
                    # For sequence data with shape (length, features)
                    results = np.zeros(
                        (len(valid_regions), region_length, sample_track.shape[1]),
                        dtype=self.dtype,
                    )
                else:
                    # For scalar data with shape (length,)
                    results = np.zeros(
                        (len(valid_regions), region_length), dtype=self.dtype
                    )
            else:
                results = [None] * len(valid_regions)
        else:
            results = [None] * len(valid_regions)

        # Process each chunk
        for chunk_idx, regions_in_chunk in chunk_regions.items():
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, chr_size)

            # Load entire chunk
            chunk_data = self.get_track(chr_name, chunk_start, chunk_end, **kwargs)

            # Extract regions from chunk
            for (
                region_idx,
                orig_start,
                orig_end,
                start_in_chunk,
                end_in_chunk,
            ) in regions_in_chunk:
                if isinstance(chunk_data, np.ndarray):
                    region_data = chunk_data[start_in_chunk:end_in_chunk]

                    # For regions spanning multiple chunks, we need to concatenate
                    if isinstance(results, list):
                        # Dynamic list allocation
                        if results[region_idx] is None:
                            if chunk_data.dtype != self.dtype:
                                region_data = region_data.astype(self.dtype)
                            results[region_idx] = region_data
                        else:
                            # Concatenate with existing data
                            if chunk_data.dtype != self.dtype:
                                region_data = region_data.astype(self.dtype)
                            results[region_idx] = np.concatenate(
                                [results[region_idx], region_data]
                            )
                    else:
                        # Pre-allocated array - direct assignment
                        if chunk_data.dtype != self.dtype:
                            region_data = region_data.astype(self.dtype)
                        results[region_idx] = region_data
                else:
                    # Handle non-array types (Track objects, etc.)
                    # Fall back to individual get_track calls for complex types
                    if results[region_idx] is None:
                        results[region_idx] = self.get_track(
                            chr_name, orig_start, orig_end, **kwargs
                        )

        return results

    def get_track_for_peaks(
        self, peaks: pd.DataFrame, **kwargs
    ) -> list[Track | AggregatedCoverage]:
        """Retrieves track data for a list of peaks."""
        tracks = []
        for chr_name in peaks["Chromosome"].unique():
            chr_peaks = peaks[peaks["Chromosome"] == chr_name]
            tracks.extend(
                self.get_track_for_regions(
                    chr_name, chr_peaks[["Start", "End"]].values.tolist(), **kwargs
                )
            )
        return tracks

    def get_gene_track(
        self, gene: Gene | str, upstream: int = 1000, downstream: int = 1000, **kwargs
    ) -> Track | AggregatedCoverage:  # Return type depends on kwargs['output_format']
        """Retrieves track data for a region defined by a gene +/- flanking regions.

        Calculates the genomic interval based on the gene's start, end, and strand,
        adding the specified `upstream` and `downstream` padding. Coordinates are
        adjusted to stay within chromosome boundaries. Calls `get_track` for the
        calculated region.

        Args:
            gene (Gene | str): A `gcell.rna.gene.Gene` object (or compatible object)
                with `chrom`, `start`, `end`, and `strand` attributes, or a gene name
                string (e.g., "TP53", "MYC"). If a string is provided, the gene will
                be resolved using the Gencode object associated with this IO instance's
                assembly. The Gencode object is lazily loaded and cached on first use.
            upstream (int, optional): Number of base pairs upstream of the gene
                (relative to transcription start site direction) to include.
                Defaults to 1000.
            downstream (int, optional): Number of base pairs downstream of the gene
                (relative to transcription end site direction) to include.
                Defaults to 1000.
            **kwargs: Additional arguments passed directly to the underlying
                `get_track` call. Common arguments include `query`, `aggregator`,
                `normalizer`, `sparse`, `output_format`. If
                `output_format='track_object'`, gene metadata ('gene_name',
                'gene_strand') is added to the Track's metadata if not already present.

        Returns:
            Track | AggregatedCoverage: The processed track data for the gene region.
            The type depends on `output_format` in `kwargs` (defaults to `Track`).
            Returns an empty result (e.g., empty Track object or empty aggregation)
            if the calculated region is empty after bounds checking.

        Raises:
            ValueError: If `gene` is a string but cannot be resolved to a Gene object,
                        or if coordinates are invalid or chromosome is not found.
            TypeError: If `gene` is neither a Gene object nor a string.
            Exception: Propagates exceptions from `get_track`.
        """
        # Resolve gene name string to Gene object if needed
        gene = self._resolve_gene(gene)

        g_chrom, g_start, g_end, g_strand = gene.genomic_range

        if g_strand == "+":
            start = g_start - upstream
            end = g_end + downstream
        elif g_strand == "-":
            start = g_start - downstream
            end = g_end + upstream
        else:
            # Handle unstranded case if necessary, e.g., symmetrical padding
            logging.warning(
                f"Gene {getattr(gene, 'name', '')} has no strand info. Applying padding symmetrically."
            )
            start = g_start - upstream
            end = g_end + downstream

        # Basic coordinate sanity check
        start = max(0, start)
        # Check against active chromosome bounds
        active_sizes = self.chrom_sizes
        if g_chrom in active_sizes:
            end = min(active_sizes[g_chrom], end)
        elif self.assembly_chrom_sizes and g_chrom in self.assembly_chrom_sizes:
            # Fallback to assembly size if chrom is known but not active (e.g. after subsetting)
            # This might allow calculation even if the chrom isn't "active" for iteration
            logging.debug(
                f"Chromosome {g_chrom} not active, using assembly size for bounds."
            )
            end = min(self.assembly_chrom_sizes[g_chrom], end)
        else:
            logging.warning(
                f"Chromosome {g_chrom} not found in known sizes. Cannot apply upper bound."
            )

        if start >= end:
            logging.warning(
                f"Calculated region for gene {getattr(gene, 'name', '')} is empty ({start}, {end}). Returning empty result."
            )
            # Handle empty based on output format requested in kwargs
            output_format = kwargs.get(
                "output_format", "track_object"
            )  # Default if not specified
            if output_format == "track_object":
                return Track(
                    chrom=g_chrom,
                    start=start,
                    end=end,
                    assembly=self.assembly or "unknown",
                    tracks={},
                )
            else:
                # For raw formats, delegate empty creation to aggregator
                aggregator = kwargs.get("aggregator", SumAggregator())
                if not isinstance(aggregator, Aggregator):
                    aggregator = SumAggregator()
                return aggregator.aggregate({})

        return self.get_track(g_chrom, start, end, **kwargs)

    def get_gene_tss_track(
        self, gene: Gene | str, upstream: int = 1000, downstream: int = 1000, **kwargs
    ) -> Track | AggregatedCoverage:
        """Retrieves track data centered around a gene's Transcription Start Site (TSS).

        Calculates the genomic interval centered on the TSS (start for '+' strand,
        end for '-' strand), adding `upstream` and `downstream` padding relative
        to the TSS position. Coordinates are adjusted to stay within chromosome
        boundaries. Calls `get_track` for the calculated region.

        Args:
            gene (Gene | str): A `gcell.rna.gene.Gene` object (or compatible object)
                with `chrom`, `start`, `end`, and `strand` attributes, or a gene name
                string (e.g., "TP53", "MYC"). If a string is provided, the gene will
                be resolved using the Gencode object associated with this IO instance's
                assembly. The Gencode object is lazily loaded and cached on first use.
            upstream (int, optional): Number of base pairs upstream of the TSS to
                include. Defaults to 1000.
            downstream (int, optional): Number of base pairs downstream of the TSS to
                include. Defaults to 1000.
            **kwargs: Additional arguments passed directly to the underlying
                `get_track` call. Common arguments include `query`, `aggregator`,
                `normalizer`, `sparse`, `output_format`. If
                `output_format='track_object'`, gene/TSS metadata ('gene_name',
                'gene_strand', 'tss_coord') is added to the Track's metadata if
                not already present.

        Returns:
            Track | AggregatedCoverage: The processed track data for the TSS region.
            The type depends on `output_format` in `kwargs` (defaults to `Track`).
            Returns an empty result (e.g., empty Track object or empty aggregation)
            if the calculated region is empty after bounds checking.

        Raises:
            ValueError: If `gene` is a string but cannot be resolved to a Gene object,
                        or if coordinates are invalid or chromosome is not found.
            TypeError: If `gene` is neither a Gene object nor a string.
            Exception: Propagates exceptions from `get_track`.
        """
        # Resolve gene name string to Gene object if needed
        gene = self._resolve_gene(gene)

        g_chrom, g_start, g_end, g_strand = gene.genomic_range

        # Get TSS coordinate using gene.tss_coordinate property (symmetric to TES)
        try:
            tss = gene.tss_coordinate
        except (ValueError, AttributeError) as e:
            # Fallback: use tss_list directly
            logging.warning(
                f"Could not get TSS coordinate for gene {getattr(gene, 'name', '')}: {e}. "
                f"Falling back to tss_list."
            )
            if g_strand == "+":
                tss = gene.tss_list.Start.min()
            elif g_strand == "-":
                tss = gene.tss_list.Start.max()
            else:
                tss = gene.tss_list.Start.min()

        if g_strand == "+":
            start = tss - upstream
            end = tss + downstream
        elif g_strand == "-":
            # For '-' strand, upstream/downstream are reversed relative to transcription direction
            start = tss - downstream
            end = tss + upstream
        else:
            logging.warning(
                f"Gene {getattr(gene, 'name', '')} has no strand info for TSS track. Using start coordinate as TSS."
            )
            start = tss - upstream
            end = tss + downstream

        # Basic coordinate sanity check
        start = max(0, start)
        # Check against active chromosome bounds
        active_sizes = self.chrom_sizes
        if g_chrom in active_sizes:
            end = min(active_sizes[g_chrom], end)
        elif self.assembly_chrom_sizes and g_chrom in self.assembly_chrom_sizes:
            # Fallback check as in get_gene_track
            logging.debug(
                f"Chromosome {g_chrom} not active, using assembly size for bounds."
            )
            end = min(self.assembly_chrom_sizes[g_chrom], end)
        else:
            logging.warning(
                f"Chromosome {g_chrom} not found in known sizes. Cannot apply upper bound."
            )

        if start >= end:
            logging.warning(
                f"Calculated TSS region for gene {getattr(gene, 'name', '')} is empty ({start}, {end}). Returning empty result."
            )
            # Handle empty based on output format requested in kwargs
            output_format = kwargs.get(
                "output_format", "track_object"
            )  # Default if not specified
            if output_format == "track_object":
                return Track(
                    chrom=g_chrom,
                    start=start,
                    end=end,
                    assembly=self.assembly or "unknown",
                    tracks={},
                )
            else:
                # For raw formats, delegate empty creation to aggregator
                aggregator = kwargs.get("aggregator", SumAggregator())
                if not isinstance(aggregator, Aggregator):
                    aggregator = SumAggregator()
                return aggregator.aggregate({})

        return self.get_track(g_chrom, start, end, **kwargs)

    def get_gene_tes_track(
        self, gene: Gene | str, upstream: int = 1000, downstream: int = 1000, **kwargs
    ) -> Track | AggregatedCoverage:
        """Retrieves track data centered around a gene's Transcription End Site (TES).

        Calculates the genomic interval centered on the TES (end for '+' strand,
        start for '-' strand), adding `upstream` and `downstream` padding relative
        to the TES position. Coordinates are adjusted to stay within chromosome
        boundaries. Calls `get_track` for the calculated region.

        Args:
            gene (Gene | str): A `gcell.rna.gene.Gene` object (or compatible object)
                with `chrom`, `start`, `end`, and `strand` attributes, or a gene name
                string (e.g., "TP53", "MYC"). If a string is provided, the gene will
                be resolved using the Gencode object associated with this IO instance's
                assembly. The Gencode object is lazily loaded and cached on first use.
            upstream (int, optional): Number of base pairs upstream of the TES to
                include. For '+' strand genes, this extends before the TES (toward the
                gene body). For '-' strand genes, this extends after the TES (away from
                the gene body). Defaults to 1000.
            downstream (int, optional): Number of base pairs downstream of the TES to
                include. For '+' strand genes, this extends after the TES (away from the
                gene body). For '-' strand genes, this extends before the TES (toward the
                gene body). Defaults to 1000.
            **kwargs: Additional arguments passed directly to the underlying
                `get_track` call. Common arguments include `query`, `aggregator`,
                `normalizer`, `sparse`, `output_format`. If
                `output_format='track_object'`, gene/TES metadata ('gene_name',
                'gene_strand', 'tes_coord') is added to the Track's metadata if
                not already present.

        Returns:
            Track | AggregatedCoverage: The processed track data for the TES region.
            The type depends on `output_format` in `kwargs` (defaults to `Track`).
            Returns an empty result (e.g., empty Track object or empty aggregation)
            if the calculated region is empty after bounds checking.

        Raises:
            ValueError: If `gene` is a string but cannot be resolved to a Gene object,
                        or if coordinates are invalid or chromosome is not found.
            TypeError: If `gene` is neither a Gene object nor a string.
            Exception: Propagates exceptions from `get_track`.
        """
        # Resolve gene name string to Gene object if needed
        gene = self._resolve_gene(gene)

        g_chrom, g_start, g_end, g_strand = gene.genomic_range

        # Get TES coordinate using gene.tes property (symmetric to TSS)
        try:
            tes = gene.tes
        except (ValueError, AttributeError) as e:
            # Final fallback: use genomic_range
            logging.warning(
                f"Could not get TES for gene {getattr(gene, 'name', '')}: {e}. "
                f"Falling back to genomic_range."
            )
            if g_strand == "+":
                tes = g_end  # TES is at the end for plus strand genes
            elif g_strand == "-":
                tes = g_start  # TES is at the start for minus strand genes
            else:
                tes = g_end

        if g_strand == "+":
            start = tes - upstream
            end = tes + downstream
        elif g_strand == "-":
            # For '-' strand, upstream/downstream are reversed relative to transcription direction
            start = tes - downstream
            end = tes + upstream
        else:
            logging.warning(
                f"Gene {getattr(gene, 'name', '')} has no strand info for TES track. Using end coordinate as TES."
            )
            start = tes - upstream
            end = tes + downstream

        # Basic coordinate sanity check
        start = max(0, start)
        # Check against active chromosome bounds
        active_sizes = self.chrom_sizes
        if g_chrom in active_sizes:
            end = min(active_sizes[g_chrom], end)
        elif self.assembly_chrom_sizes and g_chrom in self.assembly_chrom_sizes:
            # Fallback check as in get_gene_track
            logging.debug(
                f"Chromosome {g_chrom} not active, using assembly size for bounds."
            )
            end = min(self.assembly_chrom_sizes[g_chrom], end)
        else:
            logging.warning(
                f"Chromosome {g_chrom} not found in known sizes. Cannot apply upper bound."
            )

        if start >= end:
            logging.warning(
                f"Calculated TES region for gene {getattr(gene, 'name', '')} is empty ({start}, {end}). Returning empty result."
            )
            # Handle empty based on output format requested in kwargs
            output_format = kwargs.get(
                "output_format", "track_object"
            )  # Default if not specified
            if output_format == "track_object":
                return Track(
                    chrom=g_chrom,
                    start=start,
                    end=end,
                    assembly=self.assembly or "unknown",
                    tracks={},
                )
            else:
                # For raw formats, delegate empty creation to aggregator
                aggregator = kwargs.get("aggregator", SumAggregator())
                if not isinstance(aggregator, Aggregator):
                    aggregator = SumAggregator()
                return aggregator.aggregate({})

        return self.get_track(g_chrom, start, end, **kwargs)

    def _get_smoothed_track(
        self, chr_name: str, start: int, end: int, conv_size: int = 200, **kwargs
    ) -> np.ndarray:
        """Retrieves and smooths track data for a region, returning a NumPy array.

        This is a convenience method that:
        1. Calls `get_track` with `output_format='raw_array'`, `sparse=False`,
           and `aggregator=ConcatAggregator()` to get dense data, potentially
           concatenated if the query yields multiple results (though typically
           used with queries yielding a single track).
        2. Transposes the result if needed (assumes signal along columns).
        3. Converts the data type to `np.float32`.
        4. Applies 1D convolution using `scipy.ndimage.convolve1d` with a
           uniform kernel (`np.ones(conv_size) / conv_size`) and 'nearest'
           boundary mode.

        Args:
            chr_name (str): Chromosome name.
            start (int): Start coordinate (0-based, inclusive).
            end (int): End coordinate (0-based, exclusive).
            conv_size (int, optional): The size of the moving average window
                for smoothing. Defaults to 200.
            **kwargs: Additional arguments passed to the underlying `get_track` call
                (e.g., `query`, `normalizer`).

        Returns:
            np.ndarray: A 1D or 2D NumPy array containing the smoothed signal(s),
                with dtype `np.float32`. The shape depends on the result of the
                underlying `get_track` call before convolution.

        Raises:
            Exception: Propagates exceptions from `get_track` or `convolve1d`.
        """
        track = self.get_track(
            chr_name,
            start,
            end,
            output_format="raw_array",
            sparse=False,
            aggregator=ConcatAggregator(),
            **kwargs,
        ).T.astype(np.float32)

        # convolve with a window of size conv_size
        return convolve1d(track, np.ones(conv_size) / conv_size, mode="nearest", axis=0)

    def get_track_obj(self, chr_name: str, start: int, end: int, **kwargs) -> Track:
        """Convenience method to get a Track object for a region."""
        result = self.get_track(
            chr_name, start, end, output_format="track_object", **kwargs
        )
        if not isinstance(result, Track):
            raise TypeError(
                f"Expected Track object, but get_track returned {type(result)}"
            )
        return result

    def get_gene_track_obj(
        self, gene: Gene | str, upstream: int = 1000, downstream: int = 1000, **kwargs
    ) -> Track:
        """Convenience method to get a Track object for a gene region.

        Args:
            gene (Gene | str): A Gene object or gene name string. See `get_gene_track`
                for details on gene name resolution.
            upstream (int, optional): Upstream padding. Defaults to 1000.
            downstream (int, optional): Downstream padding. Defaults to 1000.
            **kwargs: Additional arguments passed to `get_gene_track`.

        Returns:
            Track: A Track object for the gene region.
        """
        result = self.get_gene_track(
            gene, upstream, downstream, output_format="track_object", **kwargs
        )
        # Type checking reinforcement (should always be Track, but helps IDE/mypy)
        if not isinstance(result, Track):
            raise TypeError(
                f"Expected Track object, but get_gene_track returned {type(result)}"
            )
        return result

    def get_gene_tss_track_obj(
        self, gene: Gene | str, upstream: int = 1000, downstream: int = 1000, **kwargs
    ) -> Track:
        """Convenience method to get a Track object for a gene TSS region.

        Args:
            gene (Gene | str): A Gene object or gene name string. See `get_gene_tss_track`
                for details on gene name resolution.
            upstream (int, optional): Upstream padding. Defaults to 1000.
            downstream (int, optional): Downstream padding. Defaults to 1000.
            **kwargs: Additional arguments passed to `get_gene_tss_track`.

        Returns:
            Track: A Track object for the gene TSS region.
        """
        result = self.get_gene_tss_track(
            gene, upstream, downstream, output_format="track_object", **kwargs
        )
        # Type checking reinforcement
        if not isinstance(result, Track):
            raise TypeError(
                f"Expected Track object, but get_gene_tss_track returned {type(result)}"
            )
        return result

    def get_gene_tes_track_obj(
        self, gene: Gene | str, upstream: int = 1000, downstream: int = 1000, **kwargs
    ) -> Track:
        """Convenience method to get a Track object for a gene TES region.

        Args:
            gene (Gene | str): A Gene object or gene name string. See `get_gene_tes_track`
                for details on gene name resolution.
            upstream (int, optional): Upstream padding. Defaults to 1000.
            downstream (int, optional): Downstream padding. Defaults to 1000.
            **kwargs: Additional arguments passed to `get_gene_tes_track`.

        Returns:
            Track: A Track object for the gene TES region.
        """
        result = self.get_gene_tes_track(
            gene, upstream, downstream, output_format="track_object", **kwargs
        )
        # Type checking reinforcement
        if not isinstance(result, Track):
            raise TypeError(
                f"Expected Track object, but get_gene_tes_track returned {type(result)}"
            )
        return result

    def get_default_gene_track_obj(
        self, gene: Gene | str, upstream: int = 1000, downstream: int = 1000
    ) -> Track:
        """Get a default Track object for a gene region, typically representing aggregated or
        primary signal for the dataset.

        Similar to `get_default_track_obj`, but uses a gene to determine the coordinates.
        The gene region is calculated with the specified upstream and downstream padding,
        similar to `get_gene_track_obj`.

        Args:
            gene (Gene | str): A Gene object or gene name string. See `get_gene_track`
                for details on gene name resolution.
            upstream (int, optional): Number of base pairs upstream of the gene
                (relative to transcription start site direction) to include.
                Defaults to 1000.
            downstream (int, optional): Number of base pairs downstream of the gene
                (relative to transcription end site direction) to include.
                Defaults to 1000.

        Returns:
            Track: A caesar.analysis.Track instance representing the default track
                for the gene region.

        Raises:
            ValueError: If `gene` is a string but cannot be resolved to a Gene object,
                        or if coordinates are invalid or chromosome is not found.
            TypeError: If `gene` is neither a Gene object nor a string.
        """
        # Resolve gene name string to Gene object if needed
        gene = self._resolve_gene(gene)

        g_chrom, g_start, g_end, g_strand = gene.genomic_range

        if g_strand == "+":
            start = g_start - upstream
            end = g_end + downstream
        elif g_strand == "-":
            start = g_start - downstream
            end = g_end + upstream
        else:
            # Handle unstranded case if necessary, e.g., symmetrical padding
            logging.warning(
                f"Gene {getattr(gene, 'name', '')} has no strand info. Applying padding symmetrically."
            )
            start = g_start - upstream
            end = g_end + downstream

        # Basic coordinate sanity check
        start = max(0, start)
        # Check against active chromosome bounds
        active_sizes = self.chrom_sizes
        if g_chrom in active_sizes:
            end = min(active_sizes[g_chrom], end)
        elif self.assembly_chrom_sizes and g_chrom in self.assembly_chrom_sizes:
            # Fallback to assembly size if chrom is known but not active (e.g. after subsetting)
            logging.debug(
                f"Chromosome {g_chrom} not active, using assembly size for bounds."
            )
            end = min(self.assembly_chrom_sizes[g_chrom], end)
        else:
            logging.warning(
                f"Chromosome {g_chrom} not found in known sizes. Cannot apply upper bound."
            )

        if start >= end:
            logging.warning(
                f"Calculated region for gene {getattr(gene, 'name', '')} is empty ({start}, {end}). Returning empty Track."
            )
            return Track(
                chrom=g_chrom,
                start=start,
                end=end,
                assembly=self.assembly or "unknown",
                tracks={},
            )

        return self.get_default_track_obj(chr_name=g_chrom, start=start, end=end)

    def get_default_gene_tss_track_obj(
        self, gene: Gene | str, upstream: int = 1000, downstream: int = 1000
    ) -> Track:
        """Get a default Track object centered around a gene's Transcription Start Site (TSS),
        typically representing aggregated or primary signal for the dataset.

        Similar to `get_default_track_obj`, but uses a gene's TSS to determine the coordinates.
        The TSS region is calculated with the specified upstream and downstream padding,
        similar to `get_gene_tss_track_obj`.

        Args:
            gene (Gene | str): A Gene object or gene name string. See `get_gene_tss_track`
                for details on gene name resolution.
            upstream (int, optional): Number of base pairs upstream of the TSS to
                include. Defaults to 1000.
            downstream (int, optional): Number of base pairs downstream of the TSS to
                include. Defaults to 1000.

        Returns:
            Track: A caesar.analysis.Track instance representing the default track
                for the gene TSS region.

        Raises:
            ValueError: If `gene` is a string but cannot be resolved to a Gene object,
                        or if coordinates are invalid or chromosome is not found.
            TypeError: If `gene` is neither a Gene object nor a string.
        """
        # Resolve gene name string to Gene object if needed
        gene = self._resolve_gene(gene)

        g_chrom, g_start, g_end, g_strand = gene.genomic_range

        # Get TSS coordinate using gene.tss_coordinate property (symmetric to TES)
        try:
            tss = gene.tss_coordinate
        except (ValueError, AttributeError) as e:
            # Fallback: use tss_list directly
            logging.warning(
                f"Could not get TSS coordinate for gene {getattr(gene, 'name', '')}: {e}. "
                f"Falling back to tss_list."
            )
            if g_strand == "+":
                tss = gene.tss_list.Start.min()
            elif g_strand == "-":
                tss = gene.tss_list.Start.max()
            else:
                tss = gene.tss_list.Start.min()

        if g_strand == "+":
            start = tss - upstream
            end = tss + downstream
        elif g_strand == "-":
            # For '-' strand, upstream/downstream are reversed relative to transcription direction
            start = tss - downstream
            end = tss + upstream
        else:
            logging.warning(
                f"Gene {getattr(gene, 'name', '')} has no strand info for TSS track. Using start coordinate as TSS."
            )
            start = tss - upstream
            end = tss + downstream

        # Basic coordinate sanity check
        start = max(0, start)
        # Check against active chromosome bounds
        active_sizes = self.chrom_sizes
        if g_chrom in active_sizes:
            end = min(active_sizes[g_chrom], end)
        elif self.assembly_chrom_sizes and g_chrom in self.assembly_chrom_sizes:
            # Fallback to assembly size if chrom is known but not active (e.g. after subsetting)
            logging.debug(
                f"Chromosome {g_chrom} not active, using assembly size for bounds."
            )
            end = min(self.assembly_chrom_sizes[g_chrom], end)
        else:
            logging.warning(
                f"Chromosome {g_chrom} not found in known sizes. Cannot apply upper bound."
            )

        if start >= end:
            logging.warning(
                f"Calculated TSS region for gene {getattr(gene, 'name', '')} is empty ({start}, {end}). Returning empty Track."
            )
            return Track(
                chrom=g_chrom,
                start=start,
                end=end,
                assembly=self.assembly or "unknown",
                tracks={},
            )

        return self.get_default_track_obj(chr_name=g_chrom, start=start, end=end)

    def get_default_gene_tes_track_obj(
        self, gene: Gene | str, upstream: int = 1000, downstream: int = 1000
    ) -> Track:
        """Get a default Track object centered around a gene's Transcription End Site (TES),
        typically representing aggregated or primary signal for the dataset.

        Similar to `get_default_track_obj`, but uses a gene's TES to determine the coordinates.
        The TES region is calculated with the specified upstream and downstream padding,
        similar to `get_gene_tes_track`.

        Args:
            gene (Gene | str): A Gene object or gene name string. See `get_gene_tes_track`
                for details on gene name resolution.
            upstream (int, optional): Number of base pairs upstream of the TES to
                include. For '+' strand genes, this extends before the TES (toward the
                gene body). For '-' strand genes, this extends after the TES (away from
                the gene body). Defaults to 1000.
            downstream (int, optional): Number of base pairs downstream of the TES to
                include. For '+' strand genes, this extends after the TES (away from the
                gene body). For '-' strand genes, this extends before the TES (toward the
                gene body). Defaults to 1000.

        Returns:
            Track: A caesar.analysis.Track instance representing the default track
                for the gene TES region.

        Raises:
            ValueError: If `gene` is a string but cannot be resolved to a Gene object,
                        or if coordinates are invalid or chromosome is not found.
            TypeError: If `gene` is neither a Gene object nor a string.
        """
        # Resolve gene name string to Gene object if needed
        gene = self._resolve_gene(gene)

        g_chrom, g_start, g_end, g_strand = gene.genomic_range

        # Get TES coordinate using gene.tes property (symmetric to TSS)
        try:
            tes = gene.tes
        except (ValueError, AttributeError) as e:
            # Final fallback: use genomic_range
            logging.warning(
                f"Could not get TES for gene {getattr(gene, 'name', '')}: {e}. "
                f"Falling back to genomic_range."
            )
            if g_strand == "+":
                tes = g_end  # TES is at the end for plus strand genes
            elif g_strand == "-":
                tes = g_start  # TES is at the start for minus strand genes
            else:
                tes = g_end

        if g_strand == "+":
            start = tes - upstream
            end = tes + downstream
        elif g_strand == "-":
            # For '-' strand, upstream/downstream are reversed relative to transcription direction
            start = tes - downstream
            end = tes + upstream
        else:
            logging.warning(
                f"Gene {getattr(gene, 'name', '')} has no strand info for TES track. Using end coordinate as TES."
            )
            start = tes - upstream
            end = tes + downstream

        # Basic coordinate sanity check
        start = max(0, start)
        # Check against active chromosome bounds
        active_sizes = self.chrom_sizes
        if g_chrom in active_sizes:
            end = min(active_sizes[g_chrom], end)
        elif self.assembly_chrom_sizes and g_chrom in self.assembly_chrom_sizes:
            # Fallback to assembly size if chrom is known but not active (e.g. after subsetting)
            logging.debug(
                f"Chromosome {g_chrom} not active, using assembly size for bounds."
            )
            end = min(self.assembly_chrom_sizes[g_chrom], end)
        else:
            logging.warning(
                f"Chromosome {g_chrom} not found in known sizes. Cannot apply upper bound."
            )

        if start >= end:
            logging.warning(
                f"Calculated TES region for gene {getattr(gene, 'name', '')} is empty ({start}, {end}). Returning empty Track."
            )
            return Track(
                chrom=g_chrom,
                start=start,
                end=end,
                assembly=self.assembly or "unknown",
                tracks={},
            )

        return self.get_default_track_obj(chr_name=g_chrom, start=start, end=end)

    # --- In-place Chromosome subsetting ---

    def leave_out_chromosomes(
        self, chroms: Sequence[str], inverse: bool = False
    ) -> None:
        """
        Modifies the active chromosomes for this IO object in-place.

        If inverse=True, keeps only the chromosomes specified in `chroms`
        (subsetting).
        If inverse=False (default), removes the chromosomes specified in `chroms`
        from the active set (exclusion).

        Args:
            chroms: A sequence of chromosome names to keep (if inverse=True) or
                    remove (if inverse=False).
            inverse: If True, performs subsetting (keep only). If False, performs
                     exclusion (remove).
        """
        if self._active_chroms is None:
            logging.warning(
                "Cannot modify active chromosomes: _active_chroms is not initialized."
            )
            return

        try:
            target_set = set(chroms)
        except TypeError:
            logging.error("chroms must be convertible to a set.")
            return

        current_active_set = set(self._active_chroms)

        if inverse:
            # Keep only specified chromosomes (subsetting)
            new_active_chroms = sorted(
                list(current_active_set.intersection(target_set))
            )
            action_desc = "Subset/Keep"
            original_len = len(current_active_set)
            requested_len = len(target_set)
            final_len = len(new_active_chroms)

            if not new_active_chroms:
                logging.warning("Subset resulted in no active chromosomes.")
            elif final_len < requested_len:
                removed_count = original_len - final_len
                not_found_count = len(target_set - current_active_set)
                logging.info(
                    f"{action_desc} chromosomes applied. {final_len} remaining active. "
                    f"{removed_count} original chromosomes removed. "
                    f"{not_found_count} requested chromosomes were not active."
                )
            else:  # Subset didn't remove anything unexpected
                removed_count = original_len - final_len
                logging.info(
                    f"{action_desc} chromosomes applied. {final_len} remaining active. "
                    f"{removed_count} original chromosomes removed."
                )

        else:
            # Remove specified chromosomes (exclusion)
            new_active_chroms = sorted(list(current_active_set.difference(target_set)))
            action_desc = "Exclude/Remove"
            original_len = len(current_active_set)
            final_len = len(new_active_chroms)
            change_count = original_len - final_len

            if not new_active_chroms:
                logging.warning(
                    f"{action_desc} chromosomes resulted in no active chromosomes."
                )
            else:
                logging.info(
                    f"{action_desc} chromosomes applied. {final_len} remaining active. "
                    f"{change_count} chromosomes removed."
                )

        self._active_chroms = new_active_chroms
        # Properties will reflect the change on next access.

    def get_genome_browser_track(self, chr_name, start, end, gene_annot=None, **kwargs):
        """Get track optimized for genome browser visualization with gene annotations.

        This method provides full flexibility for data retrieval, allowing you to specify
        custom queries, aggregators, and other parameters via `**kwargs`. It uses `get_track`
        internally, which supports querying specific cell types, samples, or applying custom
        aggregation strategies.

        **When to use:**
        - You need to query specific subsets of data (e.g., specific cell types, samples)
        - You want to apply custom aggregators (e.g., `ByKeyAggregator`, `ConcatAggregator`)
        - You need fine-grained control over data retrieval parameters
        - You want to pass custom queries via `query` parameter

        **When NOT to use:**
        - You just want the default/aggregated track for the entire dataset
        - You want automatic gene annotation (use `plot_default_genome_browser_track` instead)

        Parameters
        ----------
        chr_name : str
            Chromosome name
        start : int
            Start position (0-based, inclusive)
        end : int
            End position (0-based, exclusive)
        gene_annot : Gencode, optional
            Gene annotation object for plotting gene bodies. Must be provided explicitly.
            Defaults to None.
        **kwargs : dict
            Additional arguments passed to `get_track` and `plot_tracks_with_genebody`.
            Common `get_track` parameters:
            - `query`: Custom query object or specification (e.g., cell type names, indices)
            - `aggregator`: Aggregator instance (e.g., `ByKeyAggregator()`, `SumAggregator()`)
            - `normalizer`: Normalizer instance (overrides default)
            - `conv_size`: Convolution window size (default: 200)
            - `sparse`: Whether to request sparse format
            Common `plot_tracks_with_genebody` parameters:
            - `isoforms`: Show gene isoforms (default: True)
            - `genes_to_highlight`: List of gene names to highlight
            - `protein_coding_only`: Show only protein-coding genes (default: True)
            - `autoscale`: Auto-scale the y-axis (default: True)
            - `gene_track_height`: Height of the gene track (default: 2)

        Returns
        -------
        Track
            Track object with visualization methods applied (if `gene_annot` is provided)

        Examples
        --------
        >>> # Plot specific cell types
        >>> io.get_genome_browser_track(
        ...     'chr1', 100000, 200000,
        ...     gene_annot=io.gencode,
        ...     query=['T_cell', 'B_cell'],
        ...     aggregator=ByKeyAggregator()
        ... )
        """
        # Get track object with appropriate settings
        # Handle both dict (multi-track) and numeric (single-track) libsize
        default_normalizer = None
        if hasattr(self, "libsize") and self.libsize:
            if isinstance(self.libsize, dict):
                default_normalizer = DictNormalizer(self.libsize)
            else:
                default_normalizer = FactorNormalizer(self.libsize)
        else:
            default_normalizer = NoNormalizer()

        track = self.get_track(
            chr_name,
            start,
            end,
            output_format="track_object",
            conv_size=kwargs.pop("conv_size", 200),  # Smooth for visualization
            normalizer=kwargs.pop("normalizer", default_normalizer),
            **kwargs,
        )

        # If gene annotation provided, use the enhanced visualization
        if gene_annot is not None:
            # Extract plot-specific kwargs
            plot_kwargs = {
                "protein_coding_only": kwargs.pop("protein_coding_only", True),
                "isoforms": kwargs.pop("isoforms", True),
                "genes_to_highlight": kwargs.pop("genes_to_highlight", None),
                "gene_track_height": kwargs.pop("gene_track_height", 2),
            }
            track.plot_tracks_with_genebody(gene_annot=gene_annot, **plot_kwargs)

        return track

    def plot_default_genome_browser_track(
        self,
        chr_name: str,
        start: int,
        end: int,
        gene_annot: Gencode | None = None,
        conv_size: int = 200,
        normalizer: Normalizer | None = None,
        **plot_kwargs,
    ) -> Track:
        """Get and plot the default aggregated track for a genomic region with gene annotations.

        This method retrieves the default/aggregated track for the entire dataset (no custom queries)
        and automatically plots it with gene annotations. It's optimized for quick visualization
        of genomic regions when you want to see the overall signal pattern.

        **When to use:**
        - You want to visualize the default aggregated track for a specific genomic region
        - You know the exact coordinates (chromosome, start, end)
        - You want automatic gene annotation (uses `self.gencode` if available)
        - You don't need to query specific subsets of data

        **When NOT to use:**
        - You need to query specific cell types or samples (use `get_genome_browser_track` instead)
        - You want to work with a gene region (use `plot_default_gene_track` instead)

        **Key differences from `get_genome_browser_track`:**
        - Uses default track (no custom queries/aggregators)
        - Automatically uses `self.gencode` if `gene_annot` is not provided
        - Simpler API focused on default visualization
        - `conv_size` and `normalizer` are explicit parameters

        Parameters
        ----------
        chr_name : str
            Chromosome name
        start : int
            Start position (0-based, inclusive)
        end : int
            End position (0-based, exclusive)
        gene_annot : Gencode, optional
            Gene annotation object for plotting gene bodies. If None,
            automatically uses `self.gencode` if available. Defaults to None.
        conv_size : int, optional
            Size of the convolution window for smoothing. Defaults to 200.
        normalizer : Normalizer, optional
            Normalizer to apply to the track. If None, automatically uses:
            - `DictNormalizer(self.libsize)` if `libsize` is a dict (multi-track format)
            - `FactorNormalizer(self.libsize)` if `libsize` is numeric (single-track format)
            - `NoNormalizer()` if `libsize` is not available
            Defaults to None.
        **plot_kwargs : dict
            Additional arguments passed to `plot_tracks_with_genebody`.
            Common options include:
            - `isoforms` (bool): Show gene isoforms. Defaults to True.
            - `genes_to_highlight` (list[str]): List of gene names to highlight.
            - `protein_coding_only` (bool): Show only protein-coding genes. Defaults to True.
            - `autoscale` (bool): Auto-scale the y-axis. Defaults to True.
            - `gene_track_height` (float): Height of the gene track. Defaults to 2.

        Returns
        -------
        Track
            Track object with visualization applied (if `gene_annot` or `self.gencode` is available).

        Examples
        --------
        >>> # Simple usage with automatic gene annotation
        >>> io.plot_default_genome_browser_track('chr1', 100000, 200000)
        >>>
        >>> # With custom plot options
        >>> io.plot_default_genome_browser_track(
        ...     'chr1', 100000, 200000,
        ...     isoforms=True,
        ...     genes_to_highlight=['MYC', 'TP53']
        ... )
        """
        # Set up default normalizer if not provided
        # Handle both dict (multi-track) and numeric (single-track) libsize
        if normalizer is None:
            if hasattr(self, "libsize") and self.libsize:
                if isinstance(self.libsize, dict):
                    normalizer = DictNormalizer(self.libsize)
                else:
                    normalizer = FactorNormalizer(self.libsize)
            else:
                normalizer = NoNormalizer()

        # Get track with default query/aggregator (equivalent to default track)
        # Pass conv_size and normalizer to Track constructor via get_track
        track = self.get_track(
            chr_name,
            start,
            end,
            output_format="track_object",
            conv_size=conv_size,
            normalizer=normalizer,
        )

        # Use self.gencode as default if gene_annot not provided
        if gene_annot is None:
            gene_annot = self.gencode

        # Extract plot-specific kwargs with defaults
        plot_kwargs_final = {
            "protein_coding_only": plot_kwargs.pop("protein_coding_only", True),
            "isoforms": plot_kwargs.pop("isoforms", True),
            "genes_to_highlight": plot_kwargs.pop("genes_to_highlight", None),
            "gene_track_height": plot_kwargs.pop("gene_track_height", 2),
            **plot_kwargs,  # Pass any remaining kwargs
        }

        if gene_annot is not None:
            track.plot_tracks_with_genebody(gene_annot=gene_annot, **plot_kwargs_final)

        return track

    def plot_default_gene_track(
        self,
        gene: Gene | str,
        upstream: int = 1000,
        downstream: int = 1000,
        gene_annot: Gencode | None = None,
        **plot_kwargs,
    ) -> Track:
        """Get and plot the default aggregated track for a gene region with gene annotations.

        This method automatically calculates the genomic region for a gene (including upstream
        and downstream padding) and plots the default aggregated track. It's the most convenient
        way to visualize gene regions when you know the gene name but not the exact coordinates.

        **When to use:**
        - You want to visualize the entire gene region (gene body + flanking regions)
        - You know the gene name but not the exact genomic coordinates
        - You want automatic gene annotation (uses `self.gencode` if available)
        - You want the default aggregated track for the entire dataset

        **When NOT to use:**
        - You want to focus on the TSS region only (use `plot_default_gene_tss_track` instead)
        - You know the exact coordinates (use `plot_default_genome_browser_track` instead)
        - You need to query specific cell types or samples (use `get_genome_browser_track` instead)

        **Key features:**
        - Automatically resolves gene names to coordinates
        - Handles strand-specific padding (upstream/downstream relative to transcription direction)
        - Uses default aggregated track (no custom queries)
        - Automatically uses `self.gencode` for gene annotations if available

        Parameters
        ----------
        gene : Gene | str
            A Gene object or gene name string (e.g., "MYC", "TP53"). Gene names are
            automatically resolved using `self.gencode`. See `get_gene_track` for details.
        upstream : int, optional
            Number of base pairs upstream of the gene (relative to transcription start site
            direction) to include. For '+' strand genes, this extends before the start;
            for '-' strand genes, this extends after the end. Defaults to 1000.
        downstream : int, optional
            Number of base pairs downstream of the gene (relative to transcription end site
            direction) to include. For '+' strand genes, this extends after the end;
            for '-' strand genes, this extends before the start. Defaults to 1000.
        gene_annot : Gencode, optional
            Gene annotation object for plotting gene bodies. If None,
            automatically uses `self.gencode` if available. Defaults to None.
        **plot_kwargs : dict
            Additional arguments passed to `plot_tracks_with_genebody`.
            Common options include:
            - `isoforms` (bool): Show gene isoforms. Defaults to True.
            - `genes_to_highlight` (list[str]): List of gene names to highlight.
            - `protein_coding_only` (bool): Show only protein-coding genes. Defaults to True.
            - `autoscale` (bool): Auto-scale the y-axis. Defaults to True.
            - `gene_track_height` (float): Height of the gene track. Defaults to 2.

        Returns
        -------
        Track
            Track object with visualization applied (if `gene_annot` or `self.gencode` is available).

        Examples
        --------
        >>> # Simple usage with automatic gene annotation
        >>> io.plot_default_gene_track('MYC')
        >>>
        >>> # With custom padding and plot options
        >>> io.plot_default_gene_track(
        ...     'PRDM1',
        ...     upstream=20000,
        ...     downstream=120000,
        ...     isoforms=True,
        ...     genes_to_highlight=['PRDM1'],
        ...     autoscale=False
        ... )
        """
        track = self.get_default_gene_track_obj(
            gene=gene, upstream=upstream, downstream=downstream
        )

        # Use self.gencode as default if gene_annot not provided
        if gene_annot is None:
            gene_annot = self.gencode

        if gene_annot is not None:
            track.plot_tracks_with_genebody(gene_annot=gene_annot, **plot_kwargs)

        return track

    def plot_default_gene_tss_track(
        self,
        gene: Gene | str,
        upstream: int = 1000,
        downstream: int = 1000,
        gene_annot: Gencode | None = None,
        **plot_kwargs,
    ) -> Track:
        """Get and plot the default aggregated track centered on a gene's TSS with gene annotations.

        This method automatically calculates the genomic region centered on a gene's Transcription
        Start Site (TSS) and plots the default aggregated track. It's ideal for visualizing
        promoter regions and TSS-proximal regulatory elements.

        **When to use:**
        - You want to focus on the promoter/TSS region of a gene
        - You're interested in TSS-proximal regulatory elements (e.g., transcription factor binding)
        - You know the gene name but not the exact TSS coordinates
        - You want automatic gene annotation (uses `self.gencode` if available)
        - You want the default aggregated track for the entire dataset

        **When NOT to use:**
        - You want to visualize the entire gene body (use `plot_default_gene_track` instead)
        - You know the exact coordinates (use `plot_default_genome_browser_track` instead)
        - You need to query specific cell types or samples (use `get_genome_browser_track` instead)

        **Key features:**
        - Automatically resolves gene names to coordinates
        - Calculates TSS position based on strand (start for '+' strand, end for '-' strand)
        - Centers the region on the TSS with symmetric or asymmetric padding
        - Uses default aggregated track (no custom queries)
        - Automatically uses `self.gencode` for gene annotations if available

        Parameters
        ----------
        gene : Gene | str
            A Gene object or gene name string (e.g., "MYC", "TP53"). Gene names are
            automatically resolved using `self.gencode`. See `get_gene_tss_track` for details.
        upstream : int, optional
            Number of base pairs upstream of the TSS to include. For '+' strand genes,
            this extends before the start coordinate; for '-' strand genes, this extends
            after the end coordinate. Defaults to 1000.
        downstream : int, optional
            Number of base pairs downstream of the TSS to include. For '+' strand genes,
            this extends after the start coordinate; for '-' strand genes, this extends
            before the end coordinate. Defaults to 1000.
        gene_annot : Gencode, optional
            Gene annotation object for plotting gene bodies. If None,
            automatically uses `self.gencode` if available. Defaults to None.
        **plot_kwargs : dict
            Additional arguments passed to `plot_tracks_with_genebody`.
            Common options include:
            - `isoforms` (bool): Show gene isoforms. Defaults to True.
            - `genes_to_highlight` (list[str]): List of gene names to highlight.
            - `protein_coding_only` (bool): Show only protein-coding genes. Defaults to True.
            - `autoscale` (bool): Auto-scale the y-axis. Defaults to True.
            - `gene_track_height` (float): Height of the gene track. Defaults to 2.

        Returns
        -------
        Track
            Track object with visualization applied (if `gene_annot` or `self.gencode` is available).

        Examples
        --------
        >>> # Simple usage with automatic gene annotation
        >>> io.plot_default_gene_tss_track('MYC')
        >>>
        >>> # With asymmetric padding (more downstream)
        >>> io.plot_default_gene_tss_track(
        ...     'TP53',
        ...     upstream=2000,
        ...     downstream=5000,
        ...     isoforms=True,
        ...     genes_to_highlight=['TP53']
        ... )
        """
        track = self.get_default_gene_tss_track_obj(
            gene=gene, upstream=upstream, downstream=downstream
        )

        # Use self.gencode as default if gene_annot not provided
        if gene_annot is None:
            gene_annot = self.gencode

        if gene_annot is not None:
            track.plot_tracks_with_genebody(gene_annot=gene_annot, **plot_kwargs)

        return track

    def plot_default_gene_tes_track(
        self,
        gene: Gene | str,
        upstream: int = 1000,
        downstream: int = 1000,
        gene_annot: Gencode | None = None,
        **plot_kwargs,
    ) -> Track:
        """Get and plot the default aggregated track centered on a gene's TES with gene annotations.

        This method automatically calculates the genomic region centered on a gene's Transcription
        End Site (TES) and plots the default aggregated track. It's ideal for visualizing
        3' end regions, polyadenylation sites, and TES-proximal regulatory elements.

        **When to use:**
        - You want to focus on the 3' end/TES region of a gene
        - You're interested in TES-proximal regulatory elements (e.g., polyadenylation signals)
        - You know the gene name but not the exact TES coordinates
        - You want automatic gene annotation (uses `self.gencode` if available)
        - You want the default aggregated track for the entire dataset

        **When NOT to use:**
        - You want to visualize the entire gene body (use `plot_default_gene_track` instead)
        - You want to focus on the promoter/TSS (use `plot_default_gene_tss_track` instead)
        - You know the exact coordinates (use `plot_default_genome_browser_track` instead)
        - You need to query specific cell types or samples (use `get_genome_browser_track` instead)

        **Key features:**
        - Automatically resolves gene names to coordinates
        - Calculates TES position based on strand (end for '+' strand, start for '-' strand)
        - Centers the region on the TES with symmetric or asymmetric padding
        - Uses default aggregated track (no custom queries)
        - Automatically uses `self.gencode` for gene annotations if available

        Parameters
        ----------
        gene : Gene | str
            A Gene object or gene name string (e.g., "MYC", "TP53"). Gene names are
            automatically resolved using `self.gencode`. See `get_gene_tes_track` for details.
        upstream : int, optional
            Number of base pairs upstream of the TES to include. For '+' strand genes,
            this extends before the TES (toward the gene body); for '-' strand genes,
            this extends after the TES (away from the gene body). Defaults to 1000.
        downstream : int, optional
            Number of base pairs downstream of the TES to include. For '+' strand genes,
            this extends after the TES (away from the gene body); for '-' strand genes,
            this extends before the TES (toward the gene body). Defaults to 1000.
        gene_annot : Gencode, optional
            Gene annotation object for plotting gene bodies. If None,
            automatically uses `self.gencode` if available. Defaults to None.
        **plot_kwargs : dict
            Additional arguments passed to `plot_tracks_with_genebody`.
            Common options include:
            - `isoforms` (bool): Show gene isoforms. Defaults to True.
            - `genes_to_highlight` (list[str]): List of gene names to highlight.
            - `protein_coding_only` (bool): Show only protein-coding genes. Defaults to True.
            - `autoscale` (bool): Auto-scale the y-axis. Defaults to True.
            - `gene_track_height` (float): Height of the gene track. Defaults to 2.

        Returns
        -------
        Track
            Track object with visualization applied (if `gene_annot` or `self.gencode` is available).

        Examples
        --------
        >>> # Simple usage with automatic gene annotation
        >>> io.plot_default_gene_tes_track('MYC')
        >>>
        >>> # With asymmetric padding (more upstream toward gene body)
        >>> io.plot_default_gene_tes_track(
        ...     'TP53',
        ...     upstream=5000,
        ...     downstream=2000,
        ...     isoforms=True,
        ...     genes_to_highlight=['TP53']
        ... )
        """
        track = self.get_default_gene_tes_track_obj(
            gene=gene, upstream=upstream, downstream=downstream
        )

        # Use self.gencode as default if gene_annot not provided
        if gene_annot is None:
            gene_annot = self.gencode

        if gene_annot is not None:
            track.plot_tracks_with_genebody(gene_annot=gene_annot, **plot_kwargs)

        return track
