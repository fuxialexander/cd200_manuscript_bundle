# src/caesar/io/bpcells_io.py

from __future__ import annotations

import concurrent.futures
from concurrent.futures import as_completed
import json
import logging
import os
from abc import abstractmethod
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from tqdm import tqdm

# Conditional import for BPCells
try:
    import bpcells.cpp
    from bpcells.experimental import PrecalculatedInsertionMatrix
    from bpcells.experimental.matrix import DirMatrix

    HAS_BPCELLS = True
except ImportError:
    HAS_BPCELLS = False


def validate_celltype_dense_compatibility(input_dirs: list[str]) -> dict:
    """Validate that celltype_dense matrices are compatible for concatenation.

    Args:
        input_dirs: List of paths to celltype_dense BPCells matrix directories

    Returns:
        dict with validation results:
            - compatible: bool indicating if matrices can be concatenated
            - chrom_offsets: shared chrom_offsets if compatible
            - group_names: list of group names from each matrix
            - library_sizes: list of library sizes from each matrix
            - shapes: list of shapes from each matrix
            - error: error message if not compatible

    Raises:
        ValueError: If required files are missing
    """
    input_dirs = [str(os.path.abspath(os.path.expanduser(d))) for d in input_dirs]

    all_chrom_offsets = []
    all_group_names = []
    all_library_sizes = []
    all_shapes = []

    for i, input_dir in enumerate(input_dirs):
        # Load chrom_offsets
        chrom_offsets_path = os.path.join(input_dir, "chrom_offsets.json")
        if not os.path.exists(chrom_offsets_path):
            raise ValueError(f"Missing chrom_offsets.json in {input_dir}")
        with open(chrom_offsets_path) as f:
            all_chrom_offsets.append(json.load(f))

        # Load group_names
        group_names_path = os.path.join(input_dir, "group_names.json")
        row_names_path = os.path.join(input_dir, "row_names")
        if os.path.exists(group_names_path):
            with open(group_names_path) as f:
                all_group_names.append(json.load(f))
        elif os.path.exists(row_names_path):
            with open(row_names_path) as f:
                all_group_names.append([line.strip() for line in f if line.strip()])
        else:
            raise ValueError(f"Missing group_names.json or row_names in {input_dir}")

        # Load library_size
        library_size_path = os.path.join(input_dir, "library_size.json")
        if os.path.exists(library_size_path):
            with open(library_size_path) as f:
                lib_data = json.load(f)
                if isinstance(lib_data, dict) and "library_sizes" in lib_data:
                    all_library_sizes.append(lib_data["library_sizes"])
                elif isinstance(lib_data, list):
                    all_library_sizes.append(lib_data)
                else:
                    raise ValueError(f"Unexpected library_size.json format in {input_dir}")
        else:
            all_library_sizes.append(None)

        # Load shape
        shape = tuple(np.fromfile(os.path.join(input_dir, "shape"), np.uint32, offset=8))
        all_shapes.append(shape)

    # Check compatibility
    reference_offsets = all_chrom_offsets[0]
    for i, offsets in enumerate(all_chrom_offsets[1:], 1):
        if offsets != reference_offsets:
            ref_chroms = sorted(reference_offsets.keys())
            cur_chroms = sorted(offsets.keys())
            if ref_chroms != cur_chroms:
                return {
                    "compatible": False,
                    "error": (
                        f"Matrix {i} has different chromosomes than matrix 0.\n"
                        f"Matrix 0: {ref_chroms}\n"
                        f"Matrix {i}: {cur_chroms}"
                    ),
                    "chrom_offsets": all_chrom_offsets,
                    "group_names": all_group_names,
                    "library_sizes": all_library_sizes,
                    "shapes": all_shapes,
                }
            else:
                diff_chroms = [c for c in ref_chroms if offsets[c] != reference_offsets[c]]
                return {
                    "compatible": False,
                    "error": (
                        f"Matrix {i} has different chrom_offsets (different chromosome ordering).\n"
                        f"Differing chromosomes: {diff_chroms[:5]}..."
                    ),
                    "chrom_offsets": all_chrom_offsets,
                    "group_names": all_group_names,
                    "library_sizes": all_library_sizes,
                    "shapes": all_shapes,
                }

    # Check column dimensions
    reference_cols = all_shapes[0][1]
    for i, shape in enumerate(all_shapes[1:], 1):
        if shape[1] != reference_cols:
            return {
                "compatible": False,
                "error": (
                    f"Matrix {i} has {shape[1]:,} columns but matrix 0 has {reference_cols:,}.\n"
                    "Matrices must have the same genome size."
                ),
                "chrom_offsets": all_chrom_offsets,
                "group_names": all_group_names,
                "library_sizes": all_library_sizes,
                "shapes": all_shapes,
            }

    return {
        "compatible": True,
        "chrom_offsets": reference_offsets,
        "group_names": all_group_names,
        "library_sizes": all_library_sizes,
        "shapes": all_shapes,
        "error": None,
    }


def create_concatenated_celltype_dense_view(
    input_dirs: list[str],
    collapse_group: bool = True,
) -> "MultiPrecalculatedInsertionMatrix":
    """Create a virtual concatenated view of multiple celltype_dense matrices.

    This creates a MultiPrecalculatedInsertionMatrix that lazily queries all input
    matrices and concatenates results. This is the recommended approach for combining
    celltype_dense matrices as it:
    - Works with matrices that have different coordinate systems
    - Doesn't require copying data
    - Supports collapse_group for aggregating duplicate group names

    For matrices with the same coordinate system, queries are concatenated at runtime.
    For matrices with different coordinate systems, each is queried independently
    using its own coordinate mapping.

    Args:
        input_dirs: List of paths to celltype_dense BPCells matrix directories
        collapse_group: If True, aggregate library sizes and counts for groups
            with the same name across matrices

    Returns:
        MultiPrecalculatedInsertionMatrix that provides a unified view

    Example:
        >>> matrix = create_concatenated_celltype_dense_view(
        ...     ["/path/to/matrix1", "/path/to/matrix2"],
        ...     collapse_group=True
        ... )
        >>> counts = matrix.get_counts(regions)
    """
    if not HAS_BPCELLS:
        raise ImportError("BPCells is required")

    input_dirs = [str(os.path.abspath(os.path.expanduser(d))) for d in input_dirs]
    return MultiPrecalculatedInsertionMatrix(input_dirs, collapse_group=collapse_group)


def _extract_sample_name_from_path(input_dir: str) -> str:
    """Extract sample name from celltype_dense directory path.

    Handles common patterns:
    - .../get_data/get_data_fragments/A001-C-002-D_20200702/output/hg38/celltype_dense_bpcells/
      -> A001-C-002-D_20200702
    - .../study_name/sample_id/output/assembly/celltype_dense_bpcells/
      -> sample_id
    - .../output_dir/celltype_dense_bpcells/
      -> output_dir

    Args:
        input_dir: Path to celltype_dense directory

    Returns:
        Extracted sample name
    """
    path = Path(input_dir).resolve()
    parts = path.parts

    # Look for pattern: .../sample_id/output/assembly/celltype_dense_bpcells
    # Go backwards from celltype_dense_bpcells to find sample_id
    for i, part in enumerate(parts):
        if "celltype_dense" in part.lower():
            # Check if we have output/assembly pattern before this
            if i >= 3 and parts[i - 2] == "output":
                return parts[i - 3]  # sample_id is 3 levels up
            elif i >= 2:
                return parts[i - 2]  # fallback: 2 levels up
            elif i >= 1:
                return parts[i - 1]  # fallback: 1 level up

    # Last resort: use parent directory name
    return path.parent.name


def concat_celltype_dense_matrices(
    input_dirs: list[str],
    output_dir: str,
    validate_chrom_offsets: bool = True,
    prefix_with_sample_name: bool = False,
    threads: int = 32,
    batch_size: int = 32,
) -> str:
    """Physically concatenate multiple celltype_dense BPCells matrices.

    This function creates a single concatenated matrix by:
    1. Validating that all input matrices have compatible coordinate systems
    2. Using PrecalculatedInsertionMatrix's internal format to concatenate
    3. Merging metadata files (group_names, library_sizes)

    NOTE: Physical concatenation of precalculated insertion matrices requires
    compatible coordinate systems (identical chrom_offsets). For matrices with
    different coordinate systems, use create_concatenated_celltype_dense_view()
    which provides lazy concatenation at query time.

    Args:
        input_dirs: List of paths to celltype_dense BPCells matrix directories
        output_dir: Path for the output concatenated matrix
        validate_chrom_offsets: If True (default), require identical chrom_offsets.
        prefix_with_sample_name: If True, prefix group names with sample name
            extracted from directory path (e.g., "A001-C-002-D_20200702_cluster_0").
            This ensures unique group names across samples.
        threads: Number of threads for BPCells C++ concatenation (default: 32)
        batch_size: Batch size for BPCells C++ concatenation (default: 32)

    Returns:
        Path to the output directory

    Raises:
        ValueError: If matrices have incompatible coordinate systems
        ImportError: If BPCells is not available
        NotImplementedError: Physical concatenation of precalculated matrices
            is not yet supported - use create_concatenated_celltype_dense_view()

    Example:
        >>> # For compatible matrices with unique group names:
        >>> output = concat_celltype_dense_matrices(
        ...     ["/path/to/matrix1", "/path/to/matrix2"],
        ...     "/path/to/output",
        ...     prefix_with_sample_name=True
        ... )
    """
    if not HAS_BPCELLS:
        raise ImportError("BPCells is required for concat_celltype_dense_matrices")

    if len(input_dirs) < 2:
        raise ValueError("At least 2 input directories are required for concatenation")

    # Normalize paths
    input_dirs = [str(os.path.abspath(os.path.expanduser(d))) for d in input_dirs]
    output_dir = str(os.path.abspath(os.path.expanduser(output_dir)))

    # Validate compatibility
    validation = validate_celltype_dense_compatibility(input_dirs)
    if not validation["compatible"]:
        raise ValueError(
            f"Matrices are not compatible for physical concatenation:\n"
            f"{validation['error']}\n\n"
            f"Use create_concatenated_celltype_dense_view() for lazy concatenation instead."
        )

    # Try to use the C++ function for physical concatenation if available
    has_cpp_concat = hasattr(bpcells.cpp, "write_matrix_dir_from_concat_experimental")

    if has_cpp_concat:
        # C++ function creates the directory, so ensure it doesn't exist
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        # Use the C++ function to physically concatenate the matrices
        # concat_rows=True because rows = pseudobulks, columns = genome positions
        bpcells.cpp.write_matrix_dir_from_concat_experimental(
            input_dirs, output_dir, True, threads=threads, batch_size=batch_size
        )
        logging.info(f"Physically concatenated {len(input_dirs)} matrices into {output_dir}")
    else:
        # For lazy mode, we need to create the output directory
        os.makedirs(output_dir, exist_ok=True)
        # Write source matrices list for lazy loading via MultiPrecalculatedInsertionMatrix
        with open(os.path.join(output_dir, "source_matrices.json"), "w") as f:
            json.dump(input_dirs, f, indent=2)
        logging.info(
            f"Created virtual concatenated view in {output_dir}\n"
            f"Note: Physical concatenation requires BPCells with "
            f"write_matrix_dir_from_concat_experimental support."
        )

    # Write merged metadata
    all_group_names = validation["group_names"]
    all_library_sizes = validation["library_sizes"]

    # Merge group_names, optionally prefixing with sample name
    merged_group_names = []
    for i, (input_dir, names) in enumerate(zip(input_dirs, all_group_names)):
        if prefix_with_sample_name:
            sample_name = _extract_sample_name_from_path(input_dir)
            prefixed_names = [f"{sample_name}_{name}" for name in names]
            merged_group_names.extend(prefixed_names)
        else:
            merged_group_names.extend(names)
    with open(os.path.join(output_dir, "group_names.json"), "w") as f:
        json.dump(merged_group_names, f, indent=2)

    # Write row_names for compatibility
    with open(os.path.join(output_dir, "row_names"), "w") as f:
        f.write("\n".join(merged_group_names))

    # Merge library_sizes
    if all(ls is not None for ls in all_library_sizes):
        merged_library_sizes = []
        for ls in all_library_sizes:
            merged_library_sizes.extend(ls)
        with open(os.path.join(output_dir, "library_size.json"), "w") as f:
            json.dump({"library_sizes": merged_library_sizes}, f, indent=2)

    # Write chrom_offsets
    with open(os.path.join(output_dir, "chrom_offsets.json"), "w") as f:
        json.dump(validation["chrom_offsets"], f, indent=2)

    # Copy version file from first matrix
    version_src = os.path.join(input_dirs[0], "version")
    if os.path.exists(version_src):
        import shutil
        shutil.copy(version_src, os.path.join(output_dir, "version"))

    # Merge attrs.json
    merged_attrs = {"source_matrices": input_dirs, "concatenated": True}
    for input_dir in input_dirs:
        attrs_path = os.path.join(input_dir, "attrs.json")
        if os.path.exists(attrs_path):
            with open(attrs_path) as f:
                attrs = json.load(f)
                if "assembly" not in merged_attrs and "assembly" in attrs:
                    merged_attrs["assembly"] = attrs["assembly"]

    with open(os.path.join(output_dir, "attrs.json"), "w") as f:
        json.dump(merged_attrs, f, indent=2)

    # Calculate combined shape
    total_rows = sum(s[0] for s in validation["shapes"])
    cols = validation["shapes"][0][1]
    shape_data = np.array([total_rows, cols], dtype=np.uint32)

    # Write shape file with proper header
    with open(os.path.join(output_dir, "shape"), "wb") as f:
        # Write header matching BPCells format
        f.write(b"UINT32v1")  # 8 bytes header
        shape_data.tofile(f)

    logging.info(
        f"Concatenation complete: {output_dir}\n"
        f"Total groups: {len(merged_group_names)}\n"
        f"Physical concat: {has_cpp_concat}"
    )

    return output_dir


class MultiPrecalculatedInsertionMatrix:
    """Wrapper class that provides backward compatibility for Caesar.

    This wrapper adapts the new unified bpcells.experimental.PrecalculatedInsertionMatrix
    to provide the collapse_group functionality that was previously available in older
    BPCells versions but has been removed from the current API.

    The collapse_group feature is essential for CelltypeDenseBPCellsIO, where multiple
    matrices with the same group name need to be aggregated (e.g., bulk replicates or
    technical duplicates of the same cell type).

    When collapse_group=False (default):
        Acts as a transparent pass-through to PrecalculatedInsertionMatrix

    When collapse_group=True:
        - Aggregates library sizes for groups with the same name
        - Sums counts across duplicate groups
        - Maintains unique group names for consistent indexing
    """

    def __init__(self, matrices, collapse_group: bool = False):
        """
        Initialize MultiPrecalculatedInsertionMatrix wrapper.

        Args:
            matrices: List of paths to matrix directories or PrecalculatedInsertionMatrix objects
            collapse_group: If True, sum up library sizes and counts for groups with the same name
        """
        # Convert paths to PrecalculatedInsertionMatrix objects if needed
        if not isinstance(matrices, list):
            matrices = [matrices]

        # Use the unified PrecalculatedInsertionMatrix directly
        self._backend = PrecalculatedInsertionMatrix(matrices)
        self._collapse_group = collapse_group

        if collapse_group:
            # Implement collapse functionality for Caesar compatibility
            self._setup_collapse_group()
        else:
            # Direct passthrough
            self._group_names = self._backend.group_names
            self._library_size = self._backend.library_size
            self._shape = self._backend.shape

    def _setup_collapse_group(self):
        """Set up group collapsing for backward compatibility."""
        all_group_names = self._backend.group_names
        all_library_sizes = self._backend.library_size

        # Create mapping from unique group names to their original indices
        self._unique_group_names = []
        self._group_indices_map = {}

        for i, name in enumerate(all_group_names):
            if name not in self._group_indices_map:
                self._group_indices_map[name] = []
                self._unique_group_names.append(name)
            self._group_indices_map[name].append(i)

        # Calculate combined library size by summing up sizes for the same group
        self._library_size = np.array(
            [
                np.sum(all_library_sizes[self._group_indices_map[name]])
                for name in self._unique_group_names
            ]
        )

        # Combined shape with unique groups
        self._shape = (
            len(self._unique_group_names),  # Number of unique groups
            self._backend.shape[1],  # Genome size
        )

        self._group_names = self._unique_group_names

    @property
    def shape(self):
        return self._shape

    @property
    def library_size(self):
        return self._library_size

    @property
    def group_names(self):
        return self._group_names

    @property
    def _chrom_offsets(self):
        """For backward compatibility with Caesar code."""
        return self._backend._chrom_offsets

    def get_counts(self, regions):
        """Get counts with optional group collapsing."""
        if not self._collapse_group:
            # Direct passthrough
            return self._backend.get_counts(regions)

        # Get counts from backend
        all_counts = self._backend.get_counts(regions)

        # Collapse counts by group name
        num_regions, _, region_size = all_counts.shape
        collapsed_counts = np.zeros(
            (num_regions, len(self._unique_group_names), region_size), dtype=np.int32
        )

        for i, name in enumerate(self._unique_group_names):
            indices = self._group_indices_map[name]
            # Sum counts for the same group
            collapsed_counts[:, i, :] = np.sum(all_counts[:, indices, :], axis=1)

        return collapsed_counts


# Wrapper functions for BPCells functions with updated signatures
def build_cell_groups_legacy(fragments: str, cell_ids, group_ids, group_order):
    """
    Legacy wrapper for build_cell_groups that returns numpy array instead of pd.Categorical.
    Maintains backward compatibility with existing Caesar code.
    """
    try:
        from bpcells.experimental.fragments import build_cell_groups

        # Use the new function that returns pd.Categorical
        build_cell_groups(
            fragments, cell_ids, group_ids, group_order
        )

        # Convert back to numpy array format for compatibility
        group_index_lookup = {g: i for i, g in enumerate(group_order)}
        cell_index_lookup = {
            c: i for i, c in enumerate(bpcells.cpp.cell_names_fragments_dir(fragments))
        }

        ret = np.full(len(cell_index_lookup), -1, np.int32)
        for cell_id, group_id in zip(cell_ids, group_ids):
            if cell_id in cell_index_lookup and group_id in group_index_lookup:
                ret[cell_index_lookup[cell_id]] = group_index_lookup[group_id]

        return ret

    except ImportError:
        # Fallback if BPCells is not available
        return None


def precalculate_insertion_counts_legacy(
    fragments: str,
    output_dir: str,
    cell_groups,
    chrom_sizes,
    threads: int = 0,
    group_names=None,
):
    """
    Legacy wrapper for precalculate_insertion_counts that accepts numpy arrays.
    Converts numpy array to pd.Categorical if needed for the new API.
    """
    try:
        from bpcells.experimental.fragments import precalculate_insertion_counts

        if isinstance(cell_groups, np.ndarray) and group_names is not None:
            # Convert numpy array to pd.Categorical for new API
            # Map indices to group names
            group_assignments = [
                group_names[idx] if idx >= 0 and idx < len(group_names) else None
                for idx in cell_groups
            ]
            categorical_groups = pd.Categorical(
                group_assignments, categories=group_names, ordered=True
            )
            return precalculate_insertion_counts(
                fragments, output_dir, categorical_groups, chrom_sizes, threads
            )
        else:
            # Pass through if already in correct format
            return precalculate_insertion_counts(
                fragments, output_dir, cell_groups, chrom_sizes, threads, group_names
            )

    except ImportError:
        return None


from typing import TYPE_CHECKING

from .interface import (
    BaseQuery,
    CelltypeDenseIO,
    DefaultQuery,
    GenomicIO,
    IndexQuery,
    KeyQuery,
    MetadataQuery,
    MultiKeyQuery,
    MultiSparseIO,
    SignalDenseIO,
    SingleSparseIO,
)
from . import read_peaks
from .parallel_processing import process_chromosome_with_chunks

# Import ChromSize for on-the-fly tile generation
try:
    from gcell.dna.genome import ChromSize
    HAS_GCELL = True
except ImportError:
    HAS_GCELL = False
    ChromSize = None

if TYPE_CHECKING:
    from .aggregator import RawCoverageDict


class BPCellsIO(GenomicIO):
    """
    Abstract base class for BPCells-based genomic data I/O.
    Uses bpcells.experimental.PrecalculatedInsertionMatrix as backend.
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        dtype: str = "float32",
        chroms: list[str] | None = None,
    ):
        """Initialize BPCellsIO object."""
        super().__init__(path, mode, dtype)

        self._setup_dataset()
        self._determine_assembly()
        if chroms is not None:
            self._active_chroms = set(chroms).intersection(self.data_chroms)
            if len(self._active_chroms) == 0:
                raise ValueError(f"No valid chromosomes found in {chroms}")
        else:
            self._active_chroms = self.data_chroms

        self._init_data_structure()

    @property
    def data_chroms(self) -> list[str]:
        """Returns chromosomes present in the BPCells data."""
        if self.dataset is None:
            return []
        return sorted(list(self.dataset._chrom_offsets.keys()))

    def _setup_dataset(self) -> None:
        """Load the BPCells matrix and attributes."""
        # BPCells matrix and group ordering
        self.dataset = None
        self.ids = None
        self.group_names_from_file = None  # Store group_names.json if available
        self.attrs_path = os.path.join(self.path, "attrs.json")
        self.attrs = {}
        self.assembly = None

        # First, try to load group_names.json (for celltype_dense format)
        # This takes priority over matrix row names as it contains meaningful names
        group_names_path = os.path.join(self.path, "group_names.json")
        if os.path.exists(group_names_path):
            try:
                with open(group_names_path) as f:
                    self.group_names_from_file = json.load(f)
                logging.info(f"Loaded {len(self.group_names_from_file)} group names from group_names.json")
            except Exception as e:
                logging.warning(f"Error reading group_names.json: {e}")

        # Try to load barcodes.tsv as the primary source for cell IDs (for sparse formats)
        barcodes_path = os.path.join(self.path, "barcodes.tsv")
        if os.path.exists(barcodes_path):
            try:
                self.ids = (
                    pd.read_csv(barcodes_path, header=None).iloc[:, 0].tolist()
                )
                logging.info(f"Loaded {len(self.ids)} barcodes from barcodes.tsv")
            except Exception as e:
                logging.warning(f"Error reading barcodes.tsv: {e}")

        # Try to load attrs.json
        if os.path.exists(self.attrs_path):
            try:
                with open(self.attrs_path) as f:
                    self.attrs = json.load(f)
                self.assembly = self.attrs.get("assembly")
            except Exception as e:
                logging.warning(f"Error reading attrs.json: {e}")

        # Try to load normalization_factors.json and merge into attrs as normalization_params
        # This allows QuantileNormalizer to access normalization params via _load_attr
        normalization_factors_path = os.path.join(self.path, "normalization_factors.json")
        if os.path.exists(normalization_factors_path):
            try:
                with open(normalization_factors_path) as f:
                    normalization_factors = json.load(f)
                    # If normalization_params already exists in attrs, merge them
                    # Otherwise, use normalization_factors as normalization_params
                    if "normalization_params" in self.attrs:
                        # Merge: attrs takes precedence, but fill missing keys from normalization_factors
                        existing_params = self.attrs["normalization_params"]
                        if isinstance(existing_params, dict) and isinstance(normalization_factors, dict):
                            merged_params = normalization_factors.copy()
                            merged_params.update(existing_params)
                            self.attrs["normalization_params"] = merged_params
                        else:
                            # If structure doesn't match, prefer attrs version
                            logging.warning(
                                "normalization_params in attrs.json has different structure than "
                                "normalization_factors.json. Using attrs.json version."
                            )
                    else:
                        # Use normalization_factors as normalization_params
                        # If it's already in the expected format (with background/scale_factor),
                        # use it directly; otherwise wrap it
                        if isinstance(normalization_factors, dict) and "background" in normalization_factors and "scale_factor" in normalization_factors:
                            self.attrs["normalization_params"] = normalization_factors
                        else:
                            # Assume it's a simple factor dict and convert to expected format
                            # This handles backward compatibility with simple factor files
                            logging.info(
                                "normalization_factors.json doesn't have background/scale_factor structure. "
                                "Storing as-is. For quantile normalization, ensure format matches "
                                "{'background': {...}, 'scale_factor': {...}}"
                            )
                            self.attrs["normalization_params"] = normalization_factors
                logging.info("Loaded normalization factors from normalization_factors.json")
            except Exception as e:
                logging.warning(f"Error reading normalization_factors.json: {e}")

        # Check if BPCells is available and try to load the dataset
        if HAS_BPCELLS:
            try:
                if "*" in self.path:
                    files = glob(self.path)
                    # Only set collapse_group=True for CelltypeDenseBPCellsIO
                    collapse_group = self.__class__.__name__ == "CelltypeDenseBPCellsIO"
                    self.dataset = MultiPrecalculatedInsertionMatrix(
                        files, collapse_group=collapse_group
                    )
                    self.attrs_path = os.path.join(files[0], "attrs.json")
                else:
                    self.dataset = PrecalculatedInsertionMatrix(self.path)
                logging.info(f"Loaded BPCells matrix from: {self.path}")

                # Priority order for IDs:
                # 1. group_names.json (for celltype_dense) - most meaningful
                # 2. barcodes.tsv (for sparse formats)
                # 3. dataset.group_names (fallback to matrix row names)
                if self.group_names_from_file is not None:
                    self.ids = self.group_names_from_file
                    logging.info(f"Using {len(self.ids)} IDs from group_names.json")
                elif self.ids is None and hasattr(self.dataset, "group_names"):
                    self.ids = self.dataset.group_names
                    logging.info(f"Loaded {len(self.ids)} IDs from BPCells dataset")

            except (NameError, Exception) as e:
                logging.error(f"Failed to load BPCells matrix or attributes: {e}")
                self.dataset = None
        else:
            logging.warning("BPCells not available, loading from files only")

    def _init_chunking(self, first_chrom: str | None = None) -> None:
        """Set up chunking parameters."""
        self.chunk_size = 2000000  # Standard chunk size
        self.dir_chunked = False  # BPCells doesn't use dir chunking
        self._init_chunking_done = True

    def _load_attr(self, key):
        """Load an attribute from the attrs dictionary."""
        return self.attrs.get(key)

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-JSON-serializable types to native Python types.
        
        Recursively converts numpy scalars, arrays, and other types to JSON-serializable formats.
        
        Args:
            obj: Object to convert (can be dict, list, numpy scalar, etc.)
            
        Returns:
            JSON-serializable version of the object
        """
        import numpy as np
        
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert numpy scalar to Python native type
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj  # Already JSON-serializable
        else:
            # Try to convert unknown types
            try:
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                else:
                    return str(obj)  # Fallback to string representation
            except Exception:
                return str(obj)

    def _save_attr(self, key, value):
        """Save an attribute to attrs.json, preserving existing attributes.

        This method ensures that existing attributes (like 'assembly') are not
        lost when saving new attributes. It reloads attrs.json before writing
        to merge with any attributes that might have been added externally or
        if the initial load failed.

        The merge strategy:
        1. Reload from disk to get the latest persisted state
        2. Merge in-memory attrs (preserves any recent _save_attr calls)
        3. Update with the new value
        4. Write back to disk
        """
        if self.mode == "r":
            logging.warning("Cannot save attributes: BPCellsIO is in read-only mode.")
            return

        # Reload attrs.json to ensure we have the latest persisted version
        # This prevents losing attributes if the initial load failed or if
        # attributes were added by another process
        if os.path.exists(self.attrs_path):
            try:
                with open(self.attrs_path, "r") as f:
                    existing_attrs = json.load(f)
                    # Start with disk version, then merge in-memory changes
                    # This ensures disk attributes are preserved
                    merged_attrs = existing_attrs.copy()
                    merged_attrs.update(self.attrs)
                    self.attrs = merged_attrs
            except Exception as e:
                logging.warning(
                    f"Error reloading attrs.json before save: {e}. "
                    "Proceeding with in-memory attributes only."
                )

        # Convert value to JSON-serializable format (handles numpy types)
        json_serializable_value = self._convert_to_json_serializable(value)

        # Update the attribute
        self.attrs[key] = json_serializable_value

        # Write the merged attributes back to file
        with open(self.attrs_path, "w") as f:
            json.dump(self.attrs, f, indent=4)

        # If saving normalization_params, also write to normalization_factors.json for backward compatibility
        if key == "normalization_params":
            normalization_factors_path = os.path.join(self.path, "normalization_factors.json")
            try:
                with open(normalization_factors_path, "w") as f:
                    json.dump(json_serializable_value, f, indent=4)
                logging.info(f"Also saved normalization_params to {normalization_factors_path}")
            except Exception as e:
                logging.warning(f"Error saving normalization_params to normalization_factors.json: {e}")

    @staticmethod
    def _detect_store_type_from_path(path: Path, path_str: str = None, attrs: dict = None) -> str | None:
        """Core logic for detecting BPCells store type from path.

        Args:
            path: Path object to the BPCells store
            path_str: Optional string representation of the path (for glob pattern detection)
            attrs: Optional pre-loaded attrs dictionary (to avoid re-reading attrs.json)

        Returns:
            Store type identifier or None if cannot be determined.
        """
        if path_str is None:
            path_str = str(path)

        # Load attrs if not provided
        if attrs is None:
            attrs_json_path = path / "attrs.json"
            if attrs_json_path.exists():
                try:
                    with open(attrs_json_path) as f:
                        attrs = json.load(f)
                except Exception:
                    attrs = {}
            else:
                attrs = {}

        # Check for explicit store_type
        if "store_type" in attrs:
            return attrs["store_type"]

        # Check class field (e.g., "CelltypeDenseBPCellsIO")
        if "class" in attrs:
            class_name = attrs["class"]
            class_type_map = {
                "CelltypeDense": "bpcells_celltype_dense",
                "SignalDense": "bpcells_signal_dense",
                "SingleSparse": "bpcells_single_sparse",
                "MultiSparse": "bpcells_multi_sparse",
            }
            for pattern, store_type in class_type_map.items():
                if pattern in class_name:
                    return store_type

        # Fallback to structural detection
        has_version = (path / "version").exists()
        has_val_data = (path / "val_data").exists()
        has_group_names = (path / "group_names.json").exists()
        has_index_data = (path / "index_data").exists()

        if has_version and has_val_data:
            if has_group_names:
                return "bpcells_celltype_dense"
            if has_index_data:
                return "bpcells_signal_dense"

        # Check for sparse format (has metadata.csv)
        if (path / "metadata.csv").exists():
            if "*" in path_str or len(list(path.parent.glob(f"{path.name}/*/metadata.csv"))) > 1:
                return "bpcells_multi_sparse"
            return "bpcells_single_sparse"

        return None

    @classmethod
    def detect_store_type(cls, path: str | Path) -> str | None:
        """Detect BPCells store type from path without instantiating.

        First checks attrs.json for 'store_type' or 'class' fields, then falls back
        to structural detection based on file presence.

        Args:
            path: Path to the BPCells store

        Returns:
            Store type identifier (e.g., "bpcells_celltype_dense", "bpcells_signal_dense"),
            or None if the store type cannot be determined or path is not a BPCells store.
        """
        return cls._detect_store_type_from_path(Path(path))

    def _get_store_type(self) -> str | None:
        """Get store type from BPCells metadata or structural detection.

        First checks attrs.json for 'store_type' or 'class' fields, then falls back
        to structural detection based on file presence.

        Returns:
            Store type identifier (e.g., "bpcells_celltype_dense", "bpcells_signal_dense"),
            or None if the store type cannot be determined.
        """
        return self._detect_store_type_from_path(Path(self.path), self.path, self.attrs)

    @abstractmethod
    def _get_group_indices_for_query(self, query: BaseQuery) -> dict[str, np.ndarray]:
        """
        Resolve a query into a dictionary mapping output group names
        to indices in the BPCells matrix's second dimension.
        """
        pass

    def _get_coverage_for_query(
        self, chr_name: str, start: int, end: int, query: BaseQuery, sparse: bool
    ) -> RawCoverageDict:
        """
        Retrieve coverage data for a given region and query.
        Returns a dictionary mapping group names to coverage data.
        """
        if self.dataset is None:
            raise RuntimeError("BPCells matrix is not loaded.")
        if chr_name not in self.chroms:
            raise ValueError(
                f"Chromosome '{chr_name}' not found in active chromosomes."
            )

        # Prepare query region
        query_regions = pd.DataFrame(
            {
                "chrom": [chr_name],
                "start": [max(0, start)],
                "end": [end],
            }
        )

        # Resolve query to matrix indices
        group_indices_map = self._get_group_indices_for_query(query)
        if not group_indices_map:
            return {}

        # Get all unique indices needed
        all_indices = sorted(
            list(set(idx for indices in group_indices_map.values() for idx in indices))
        )
        if not all_indices:
            return {}

        # Query BPCells matrix
        try:
            raw_counts = self.dataset.get_counts(query_regions)
            selected_data = raw_counts[0, all_indices, :]
        except Exception as e:
            logging.error(f"BPCells get_counts failed: {e}")
            return {}

        # Map indices back to group names
        idx_map = {
            original_idx: new_idx for new_idx, original_idx in enumerate(all_indices)
        }
        remapped_indices = {}
        for group_name, original_indices in group_indices_map.items():
            remapped = np.array(
                [idx_map[oi] for oi in original_indices if oi in idx_map], dtype=int
            )
            if remapped.size > 0:
                remapped_indices[group_name] = remapped

        # Process and return results
        return self._handle_query_result_dict(
            remapped_indices, selected_data, start, end, sparse
        )

    def _get_raw_coverage_for_regions(self, regions: pd.DataFrame) -> np.ndarray:
        """Get raw coverage for a list of regions."""
        return self.dataset.get_counts(regions).transpose(
            0, 2, 1
        )  # (region, position, pseudobulk)

    def _handle_query_result_dict(
        self,
        group_indices_map: dict[str, np.ndarray],
        region_data: np.ndarray,
        start: int,
        end: int,
        sparse: bool,
    ) -> dict:
        """Process query results into a dictionary of coverage data."""
        result_dict = {}

        if not group_indices_map or region_data.size == 0:
            return result_dict

        for group_name, indices in group_indices_map.items():
            if not indices.size:
                continue

            # Extract and aggregate data for this group
            group_data = region_data[indices]
            if group_data.shape[0] > 1:
                aggregated_data = group_data.sum(axis=0)
            else:
                aggregated_data = group_data.flatten()

            # Cast to desired dtype and sparsify if needed
            final_data = aggregated_data.astype(self.dtype)
            result_dict[group_name] = csr_matrix(final_data) if sparse else final_data

        return result_dict

    def _calculate_chunk_signal(self, chunk_idx):
        try:
            start_idx = chunk_idx * self.chunk_size
            region_size = min(self.chunk_size, self.dataset.shape[1] - start_idx)

            # Call BPCells C++ function directly
            result = bpcells.cpp.query_precalculated_pseudobulk_coverage(
                self.dataset._dir, [start_idx], region_size
            )

            # Reshape to get dimensions (region, pseudobulks, position)
            result = (
                result.reshape((region_size, 1, -1), order="F")
                .transpose((1, 2, 0))[0, :, :]
                .sum(axis=1)
            )  # Sum across the positions, keep pseudobulks

            return result
        except Exception as e:
            logging.error(f"Error processing chunk {chunk_idx}: {e}")
            return None

    def _calculate_genome_wide_signal(self) -> np.ndarray:
        """
        Calculate genome-wide signal for all groups in the matrix.
        Returns an array with signal values for each group index.
        """
        if self.dataset is None or self.assembly_chrom_sizes is None:
            return 0

        # Get number of groups in matrix
        n_groups = self.dataset.shape[0] if hasattr(self.dataset, "shape") else 0
        if n_groups == 0:
            return 0

        # Calculate total genome size
        total_genome_size = self.dataset.shape[1]

        # Process chunks in parallel with ProcessPoolExecutor
        total_signal = np.zeros(n_groups, dtype=np.float64)
        n_chunks = total_genome_size // self.chunk_size

        if n_chunks == 0:
            return total_signal

        # Use up to available CPUs but no more than chunks
        n_processes = min(os.cpu_count() or 1, n_chunks)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_processes
        ) as executor:
            futures = [
                executor.submit(self._calculate_chunk_signal, i)
                for i in range(n_chunks)
            ]

            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=n_chunks,
                desc="Calculating genome-wide signal",
            ):
                result = future.result()
                if result is not None:
                    total_signal += result

        return total_signal

    def _try_load_libsize_from_attrs(self, attr_key="library_sizes"):
        """Try to load library sizes from attributes."""
        precomputed = self._load_attr(attr_key)
        if precomputed is not None:
            if isinstance(precomputed, dict):
                return precomputed
            elif isinstance(precomputed, int | float):
                return float(precomputed)
        return None

    def _try_load_libsize_from_bpcells(self):
        return self.dataset.library_size

    def _process_libsize_result(
        self,
        precomputed,
        ids: list[str] = None,
        idx_mapping: dict[str, int] = None,
    ) -> float | dict | None:
        """Process library size data into appropriate format.

        This helper converts various library size formats (dict, list, numpy array,
        scalar) into a consistent output format based on the provided IDs.

        Args:
            precomputed: Raw library size data (dict, list, numpy array, or scalar)
            ids: List of IDs to filter/map library sizes for
            idx_mapping: Optional mapping from ID to index (for array-based precomputed)

        Returns:
            Processed library size as dict, float, or None
        """
        if precomputed is None:
            return None

        if isinstance(precomputed, dict):
            if ids is not None:
                return {k: precomputed[k] for k in ids if k in precomputed}
            return precomputed

        if isinstance(precomputed, list | np.ndarray):
            if ids is not None and idx_mapping is not None:
                return {
                    k: precomputed[idx_mapping[k]]
                    for k in ids
                    if k in idx_mapping and idx_mapping[k] < len(precomputed)
                }
            if ids is not None:
                return {k: precomputed[i] for i, k in enumerate(ids) if i < len(precomputed)}
            return precomputed

        if isinstance(precomputed, int | float):
            return float(precomputed)

        return None

    def _save_libsize_to_attrs(self, libsize, attr_key="library_sizes"):
        """Save library sizes to attributes."""
        if self.mode != "r" and attr_key not in self.attrs:
            self._save_attr(attr_key, libsize)

    def _prepare_bed_dataframe(
        self,
        peaks: str | pd.DataFrame,
        name: str,
        columns_to_include: list[str] = None,
    ) -> pd.DataFrame:
        """Prepare peaks DataFrame for BED file output.

        Args:
            peaks: Peaks as DataFrame or file path
            name: Name identifier for the peaks
            columns_to_include: Additional columns to include beyond required ones

        Returns:
            DataFrame ready for BED file output
        """
        if columns_to_include is None:
            columns_to_include = ["Start", "End"]

        peaks_df = read_peaks(peaks)
        if not all(col in peaks_df.columns for col in ["Chromosome", "Start", "End"]):
            raise ValueError(
                "Peaks DataFrame must contain 'Chromosome', 'Start', 'End' columns."
            )

        # Select columns to include - avoid duplicates
        required_cols = ["Chromosome", "Start", "End"]
        available_cols = [
            col for col in columns_to_include
            if col in peaks_df.columns and col not in required_cols
        ]
        bed_cols = required_cols + available_cols

        # Sort peaks by chromosome and start position
        peaks_df = peaks_df.sort_values(["Chromosome", "Start"]).reset_index(drop=True)

        # Prepare BED format
        bed_df = peaks_df[bed_cols].copy()

        # Add name column if not present (BED format requirement)
        if "name" not in bed_df.columns:
            bed_df["name"] = [f"{name}_{i}" for i in range(len(bed_df))]

        # Ensure name is in the right position (4th column in BED format)
        if "name" in bed_df.columns:
            cols = ["Chromosome", "Start", "End", "name"]
            other_cols = [c for c in bed_df.columns if c not in cols]
            bed_df = bed_df[cols + other_cols]

        return bed_df

    def _read_bed_file(
        self,
        bed_path: str,
        name: str,
        peak_cols_key: str = None,
    ) -> pd.DataFrame | None:
        """Read peaks from a BED file.

        Args:
            bed_path: Path to the BED file
            name: Name identifier for the peaks
            peak_cols_key: Key to look up stored column names in attrs

        Returns:
            DataFrame with peaks or None if reading failed
        """
        if not os.path.exists(bed_path):
            return None

        try:
            # Try to get stored column names
            if peak_cols_key and peak_cols_key in self.attrs:
                column_names = self.attrs[peak_cols_key]
                return pd.read_csv(
                    bed_path, sep="\t", header=None, names=column_names
                )

            # Fallback to legacy behavior
            try:
                peaks_df = pd.read_csv(
                    bed_path, sep="\t", header=None,
                    names=["Chromosome", "Start", "End", "name"]
                )
            except (ValueError, pd.errors.ParserError):
                peaks_df = pd.read_csv(
                    bed_path, sep="\t", header=None,
                    names=["Chromosome", "Start", "End"]
                )
                peaks_df["name"] = [f"{name}_{i}" for i in range(len(peaks_df))]

            return peaks_df
        except Exception as e:
            logging.warning(f"Failed to read peaks from {bed_path}: {e}")
            return None

    def annotate_library_sizes(
        self,
        blacklist_file: str | None = None,
        n_jobs: int = 16,
        overwrite: bool = True,
    ) -> dict[str, dict[str, float]]:
        """
        Annotate BPCells store with library size calculations accounting for blacklist.

        This method calculates three library size metrics:
        - library_size_total: Total signal across all regions
        - library_size_blacklist: Signal in blacklisted regions
        - library_size_nonblacklist: Signal in non-blacklisted regions (default)

        Parameters
        ----------
        blacklist_file : str, optional
            Path to blacklist BED file. If None, uses assembly-specific blacklist.
        n_jobs : int, default=16
            Number of parallel processes (not used for BPCells, kept for API compatibility).
        overwrite : bool, default=True
            Whether to overwrite existing library size annotations.

        Returns
        -------
        dict
            Library size results for all tracks/groups with structure:
            {track_id: {'total': float, 'blacklist': float, 'nonblacklist': float}}
        """
        if self.mode not in ["a", "r+", "w"]:
            raise ValueError("Cannot annotate in read-only mode")

        # Check if already annotated
        if not overwrite:
            existing = self._load_attr("library_size_nonblacklist")
            if existing is not None:
                logging.info(
                    "Library sizes already annotated. Use overwrite=True to recalculate."
                )
                return self._load_library_size_annotations()

        if self.dataset is None:
            raise RuntimeError("BPCells matrix is not loaded.")

        logging.info("Calculating library sizes with blacklist annotation...")

        # Load blacklist regions
        if blacklist_file is not None:
            blacklist = pd.read_csv(blacklist_file, sep="\t", header=None)
            if len(blacklist.columns) >= 3:
                blacklist.columns = ["chrom", "start", "end"] + list(
                    blacklist.columns[3:]
                )
            else:
                raise ValueError(
                    f"Blacklist file must have at least 3 columns (chrom, start, end). "
                    f"Found {len(blacklist.columns)} columns."
                )
        else:
            try:
                from gcell.dna.genome import read_blacklist

                blacklist_df = read_blacklist(self.assembly)
                blacklist = blacklist_df.rename(
                    columns={"Chromosome": "chrom", "Start": "start", "End": "end"}
                )
            except Exception as e:
                logging.warning(
                    f"Could not load blacklist for assembly {self.assembly}: {e}. "
                    "Using empty blacklist."
                )
                blacklist = pd.DataFrame(columns=["chrom", "start", "end"])

        # Get all track/group IDs to process
        all_ids = self._get_all_ids()
        if isinstance(all_ids, dict):
            all_ids = list(all_ids.keys())
        elif not isinstance(all_ids, list):
            all_ids = [all_ids]

        # Get total library sizes from dataset
        # library_size is a numpy array with one value per group
        total_library_sizes = self.dataset.library_size
        if total_library_sizes is None or len(total_library_sizes) == 0:
            logging.warning(
                "No library sizes found in dataset. Calculating from genome-wide signal..."
            )
            # Fallback: calculate genome-wide signal
            total_library_sizes = self._calculate_genome_wide_signal()
            if total_library_sizes is None or len(total_library_sizes) == 0:
                raise RuntimeError(
                    "Could not determine total library sizes. "
                    "Dataset may not have library_size attribute."
                )

        # Map group names to indices
        group_names = self.dataset.group_names if hasattr(self.dataset, "group_names") else None
        if group_names is None:
            # Fallback: use numeric indices
            group_names = [str(i) for i in range(len(total_library_sizes))]

        # Create mapping from track_id to dataset index
        track_to_idx = {}
        for i, group_name in enumerate(group_names):
            if group_name in all_ids:
                track_to_idx[group_name] = i

        # If no matches, assume indices match order
        if not track_to_idx:
            if len(all_ids) == len(total_library_sizes):
                track_to_idx = {all_ids[i]: i for i in range(len(all_ids))}
            else:
                logging.warning(
                    f"Mismatch between track IDs ({len(all_ids)}) and library sizes ({len(total_library_sizes)}). "
                    "Using first {len(all_ids)} library sizes."
                )
                track_to_idx = {all_ids[i]: i for i in range(min(len(all_ids), len(total_library_sizes)))}

        # Calculate blacklist library sizes
        blacklist_library_sizes = np.zeros(len(total_library_sizes), dtype=np.float64)

        if len(blacklist) > 0:
            # Filter blacklist to only chromosomes present in dataset
            available_chroms = set(self.data_chroms) if hasattr(self, "data_chroms") else set()
            if available_chroms:
                blacklist = blacklist[blacklist["chrom"].isin(available_chroms)].copy()

            if len(blacklist) > 0:
                logging.info(f"Querying {len(blacklist)} blacklist regions...")
                logging.debug(
                    f"Blacklist columns: {list(blacklist.columns)}, "
                    f"shape: {blacklist.shape}, "
                    f"first row: {blacklist.iloc[0].to_dict() if len(blacklist) > 0 else 'N/A'}"
                )

                # BPCells requires all regions to be the same size for batch queries
                # Group regions by size and query each group separately
                try:
                    blacklist['region_size'] = blacklist['end'] - blacklist['start']
                    region_sizes = blacklist['region_size'].unique()

                    logging.info(f"Found {len(region_sizes)} unique region sizes in blacklist")

                    for region_size in region_sizes:
                        size_subset = blacklist[blacklist['region_size'] == region_size].copy()

                        if len(size_subset) == 0:
                            continue

                        try:
                            # Query this batch of same-sized regions
                            subset_counts = self.dataset.get_counts(size_subset[['chrom', 'start', 'end']])

                            # Sum across basepair dimension (axis=2) and regions (axis=0)
                            subset_totals = subset_counts.sum(axis=2).sum(axis=0).astype(np.float64)
                            blacklist_library_sizes += subset_totals

                        except Exception as e:
                            logging.warning(
                                f"Failed to query {len(size_subset)} regions of size {region_size}: {e}. "
                                f"Skipping these regions."
                            )
                            continue

                    logging.info(
                        f"Calculated blacklist library sizes: "
                        f"min={blacklist_library_sizes.min():.0f}, "
                        f"max={blacklist_library_sizes.max():.0f}, "
                        f"mean={blacklist_library_sizes.mean():.0f}"
                    )
                except Exception as e:
                    import traceback
                    logging.error(
                        f"Failed to query blacklist regions: {type(e).__name__}: {e}\n"
                        f"Blacklist DataFrame info:\n"
                        f"  - Columns: {list(blacklist.columns)}\n"
                        f"  - Shape: {blacklist.shape}\n"
                        f"  - Sample (first 3 rows):\n{blacklist.head(3)}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
                    logging.warning("Using zero blacklist library sizes.")
                    blacklist_library_sizes = np.zeros(len(total_library_sizes), dtype=np.float64)
        else:
            logging.info("No blacklist regions provided. Blacklist library sizes will be zero.")

        # Calculate non-blacklist library sizes
        nonblacklist_library_sizes = total_library_sizes - blacklist_library_sizes

        # Build results dictionary
        results = {}
        for track_id, idx in track_to_idx.items():
            if idx < len(total_library_sizes):
                results[track_id] = {
                    "total": float(total_library_sizes[idx]),
                    "blacklist": float(blacklist_library_sizes[idx]),
                    "nonblacklist": float(nonblacklist_library_sizes[idx]),
                }

        # Store results in attrs.json
        # For single-track IO classes
        if len(results) == 1 and "default" in results:
            for metric, value in results["default"].items():
                self._save_attr(f"library_size_{metric}", value)
        else:
            # For multi-track IO classes (like CelltypeDenseBPCellsIO)
            # Store as dictionaries
            self._save_attr(
                "library_size_total",
                {k: v["total"] for k, v in results.items()},
            )
            self._save_attr(
                "library_size_blacklist",
                {k: v["blacklist"] for k, v in results.items()},
            )
            self._save_attr(
                "library_size_nonblacklist",
                {k: v["nonblacklist"] for k, v in results.items()},
            )

        # Mark as annotated
        self._save_attr("library_size_annotated", True)
        self._save_attr("library_size_annotation_date", pd.Timestamp.now().isoformat())

        logging.info("Library size annotation complete.")
        return results

    def _load_library_size_annotations(self) -> dict[str, dict[str, float]]:
        """Load existing library size annotations from attrs.json."""
        results = {}

        # Check for single-track format
        total = self._load_attr("library_size_total")
        if total is not None and not isinstance(total, dict):
            results["default"] = {
                "total": float(total),
                "blacklist": float(self._load_attr("library_size_blacklist") or 0),
                "nonblacklist": float(self._load_attr("library_size_nonblacklist") or 0),
            }
        # Check for multi-track format
        elif total is not None and isinstance(total, dict):
            blacklist = self._load_attr("library_size_blacklist") or {}
            nonblacklist = self._load_attr("library_size_nonblacklist") or {}

            for track_id in total.keys():
                results[track_id] = {
                    "total": float(total.get(track_id, 0)),
                    "blacklist": float(blacklist.get(track_id, 0) if isinstance(blacklist, dict) else 0),
                    "nonblacklist": float(nonblacklist.get(track_id, 0) if isinstance(nonblacklist, dict) else 0),
                }

        return results

    def _resolve_query_keys(
        self, query: BaseQuery, valid_keys: list[str]
    ) -> dict[str, list[str]]:
        """Helper to resolve query to output groups and their keys."""
        result = {}

        if isinstance(query, DefaultQuery):
            for key in valid_keys:
                result[key] = [key]

        elif isinstance(query, KeyQuery):
            for key in query.keys:
                if key in valid_keys:
                    result[key] = [key]

        elif isinstance(query, MultiKeyQuery):
            for group_name, keys in query.mapping.items():
                if isinstance(keys, str):
                    key_list = [keys]
                elif isinstance(keys, list | tuple | np.ndarray):
                    key_list = list(keys)
                else:
                    continue

                valid_group_keys = [k for k in key_list if k in valid_keys]
                if valid_group_keys:
                    result[group_name] = valid_group_keys

        return result

    def _determine_assembly(self) -> None:
        """Determine and set the genome assembly identifier for BPCells.

        Overrides GenomicIO._determine_assembly() to handle BPCells-specific
        attribute loading. The assembly is already attempted to be loaded in
        _setup_dataset() from attrs.json. If it wasn't found, we fall back to
        hg38 with a backend-appropriate warning message.
        """
        # Assembly may have already been loaded in _setup_dataset() from attrs.json
        if self.assembly is not None:
            return

        # Try to load from attrs if dataset exists
        if self.dataset is not None:
            try:
                self.assembly = self._load_attr("assembly")
            except Exception as e:
                logging.warning(
                    f"Error accessing BPCells attributes for assembly: {e}, using hg38 as fallback"
                )
                self.assembly = "hg38"

            if self.assembly is None:
                logging.warning(
                    "Assembly not found in BPCells attrs.json: using hg38 as fallback. "
                    "Consider ensuring attrs.json contains 'assembly' key."
                )
                self.assembly = "hg38"
        else:
            logging.warning(
                f"BPCells dataset at {self.path} could not be loaded, using hg38 as fallback"
            )
            self.assembly = "hg38"

    def _init_data_structure(self):
        """Initialize data structure specific attributes.
        Should set `self.assembly`, load `self.assembly_chrom_sizes`,
        determine `data_chroms`, set `self._active_chroms`, and potentially call
        `_init_chunking`. Loads metadata, identifies dataset structure."""
        self._load_assembly_chrom_sizes()
        self._init_chunking()


class CelltypeDenseBPCellsIO(BPCellsIO, CelltypeDenseIO):
    """Handles Celltype Dense data stored in BPCells format.

    This class provides pseudobulked or aggregated data organized by cell type,
    with efficient access via BPCells' MultiPrecalculatedInsertionMatrix backend.

    Key Features:
        - Uses collapse_group=True for MultiPrecalculatedInsertionMatrix to aggregate
          duplicate cell type groups
        - Supports subsetting to specific cell types
        - Provides efficient library size calculation from BPCells attributes
        - Handles group_names.json for meaningful cell type names vs. numeric indices

    Storage Format:
        - BPCells binary format with precalculated insertion matrices
        - group_names.json: Contains meaningful cell type names
        - attrs.json: Contains metadata and configuration
        - Supports wildcard paths for multi-file inputs
    """

    def __init__(self, path: str, mode: str = "r", celltype_ids=None, chroms=None):
        super().__init__(path, mode, dtype="float32", chroms=chroms)

        # Load celltypes - prioritize group_names.json over matrix row names
        # group_names.json contains meaningful names (e.g., 'bulk.sample_0')
        # while dataset.group_names may contain numeric indices (e.g., '0')
        if self.group_names_from_file is not None:
            self.celltypes = self.group_names_from_file
            logging.info(f"Using celltypes from group_names.json: {self.celltypes}")
        elif self.dataset is not None and hasattr(self.dataset, "group_names"):
            self.celltypes = self.dataset.group_names
            logging.info(f"Using celltypes from dataset.group_names: {self.celltypes}")
        else:
            raise ValueError("Could not load 'celltypes' from group_names.json or dataset")

        # Filter celltypes if requested
        if celltype_ids is not None:
            self.ids = [ct for ct in self.celltypes if ct in celltype_ids]
            if not self.ids:
                logging.warning("Specified celltype_ids resulted in empty selection.")
        else:
            self.ids = self.celltypes
        self.n_celltypes = len(self.ids)

        # Store mapping from celltype names to matrix row indices
        # This is needed because the matrix may use numeric indices while we use meaningful names
        if self.dataset is not None and hasattr(self.dataset, "group_names"):
            # Map our celltype names to matrix indices
            matrix_group_names = self.dataset.group_names
            self.celltype_to_idx = {}
            for i, ct_name in enumerate(self.celltypes):
                # Try to find matching index in matrix
                # If group_names.json was used, we need to map by position
                # (assuming they're in the same order)
                if i < len(matrix_group_names):
                    self.celltype_to_idx[ct_name] = i
                else:
                    # Fallback: try to find by name match
                    try:
                        idx = matrix_group_names.index(ct_name)
                        self.celltype_to_idx[ct_name] = idx
                    except ValueError:
                        # If no match, use position (assumes same order)
                        self.celltype_to_idx[ct_name] = i
        else:
            # Fallback: assume 1:1 mapping by position
            self.celltype_to_idx = {ct: i for i, ct in enumerate(self.celltypes)}

    def _get_group_indices_for_query(self, query: BaseQuery) -> dict[str, np.ndarray]:
        """Map query to celltype indices."""
        if isinstance(query, IndexQuery | MetadataQuery | MultiKeyQuery):
            raise NotImplementedError(
                f"{type(query)} not supported for CelltypeDenseBPCellsIO."
            )

        # Get mapping from celltype name to ORIGINAL matrix index
        # This is critical for proper subsetting
        mapping = self.celltype_to_idx

        # Resolve query to groups and their keys
        groups_to_keys = self._resolve_query_keys(query, self.ids)

        # Convert keys to indices
        result = {}
        for group_name, keys in groups_to_keys.items():
            indices = []
            for key in keys:
                if key in mapping:
                    indices.append(mapping[key])
            if indices:
                result[group_name] = np.array(indices, dtype=int)

        return result

    def _calculate_libsize(self) -> float | dict | None:
        """Calculate and return library size(s)."""
        try:
            precomputed = self._try_load_libsize_from_bpcells()
        except Exception:
            precomputed = self._try_load_libsize_from_attrs("library_sizes")

        result = self._process_libsize_result(precomputed, self.ids, self.celltype_to_idx)
        if result is None:
            logging.warning("Failed to load library size from the dataset or attributes")
        return result

    def _calculate_per_cell_libsize(self) -> None:
        """Not applicable for CelltypeDense."""
        logging.warning("Not applicable for CelltypeDense.")
        return None

    def subset(self, celltype_ids):
        """Create a subset with only the specified celltypes."""
        if not isinstance(celltype_ids, list | set):
            raise TypeError("celltype_ids must be a list or set.")

        # Filter against available celltypes
        valid_subset = [ct for ct in celltype_ids if ct in self.ids]

        if not valid_subset:
            logging.warning("Subset resulted in no valid cell types.")

        # Create new instance with filtered celltypes
        return CelltypeDenseBPCellsIO(
            self.path, mode=self.mode, celltype_ids=valid_subset, chroms=self.chroms
        )

    def write_peaks(
        self,
        id: str | None,
        peaks: str | pd.DataFrame,
        name="peaks",
        columns_to_include=["Start", "End"],
        overwrite=False,
    ):
        """Write peaks for a specific celltype ID to a BED file.

        Args:
            id: Celltype ID to write peaks for. If None, writes peaks for all celltypes.
            peaks: Peaks as DataFrame or file path
            name: Name identifier for the peaks (used as filename: {id}_{name}.bed)
            columns_to_include: Columns to include in the BED file (default: ["Start", "End"])
            overwrite: Whether to overwrite existing peaks file

        Note:
            This saves peaks to {self.path}/{id}_{name}.bed for each celltype,
            similar to how CelltypeDenseZarrIO saves peaks to zarr[id][name]/regions/{chrom}.
        """
        if self.mode == "r":
            logging.warning("Cannot write peaks: BPCellsIO is in read-only mode.")
            return

        # Handle None id - write for all celltypes
        if id is None:
            for celltype_id in self.ids:
                self.write_peaks(celltype_id, peaks, name, columns_to_include, overwrite)
            return

        # Validate celltype ID
        if id not in self.ids:
            logging.error(f"Celltype ID '{id}' not found in dataset.")
            return

        bed_path = os.path.join(self.path, f"{id}_{name}.bed")

        # Check if file exists and overwrite flag
        if os.path.exists(bed_path) and not overwrite:
            logging.warning(
                f"Peaks file {bed_path} already exists. Use overwrite=True to replace it."
            )
            return

        # Use helper to prepare BED DataFrame
        bed_df = self._prepare_bed_dataframe(peaks, name, columns_to_include)

        # Write BED file
        bed_df.to_csv(bed_path, sep="\t", header=False, index=False)
        logging.info(f"Saved {len(bed_df)} peaks for celltype '{id}' to {bed_path}")

        # Store peak name in attrs for reference
        peak_attr_key = f"peak_names_{id}"
        if peak_attr_key not in self.attrs:
            self.attrs[peak_attr_key] = []
        if name not in self.attrs[peak_attr_key]:
            self.attrs[peak_attr_key].append(name)
            self._save_attr(peak_attr_key, self.attrs[peak_attr_key])

        # Store column names for this peak set
        peak_cols_key = f"peak_columns_{id}_{name}"
        self.attrs[peak_cols_key] = list(bed_df.columns)
        self._save_attr(peak_cols_key, self.attrs[peak_cols_key])

    def get_peaks(self, id: str, name="peaks", chrom=None):
        """Get peaks for a specific celltype ID from a saved BED file.

        Args:
            id: Celltype ID to get peaks for
            name: Name identifier for the peaks (looks for {id}_{name}.bed in self.path)
            chrom: Optional chromosome to filter peaks by

        Returns:
            DataFrame with peaks or None if not found
        """
        if id not in self.ids:
            logging.error(f"Celltype ID '{id}' not found in dataset.")
            return None

        bed_path = os.path.join(self.path, f"{id}_{name}.bed")
        peak_cols_key = f"peak_columns_{id}_{name}"

        peaks_df = self._read_bed_file(bed_path, name, peak_cols_key)
        if peaks_df is None:
            logging.info(f"No peaks file found for celltype '{id}' with name '{name}' at {bed_path}")
            return None

        # Optional filtering by chromosome
        if chrom is not None and "Chromosome" in peaks_df.columns:
            peaks_df = peaks_df[peaks_df["Chromosome"] == chrom]
            if peaks_df.empty:
                logging.info(f"No peaks found for {name} on chromosome {chrom} for celltype {id}")
                return None

        logging.info(f"Loaded {len(peaks_df)} peaks for celltype '{id}' from {bed_path}")
        return peaks_df

    def __repr__(self):
        return f"CelltypeDenseBPCellsIO(path={self.path},\n\tmode={self.mode},\n\tcelltype_ids={self.ids},\n\tchroms={self.chroms})"

    def create_tile_adata(
        self,
        tile_size: int = 500,
        blacklist: str | pd.DataFrame | None = None,
    ):
        """Create tile count AnnData for TSS/TES accessibility extraction.

        This creates an AnnData with genome tiles as features and celltypes as
        observations, suitable for use with the fast tile-based TSS/TES
        accessibility extraction method.

        Parameters:
            tile_size: Size of tiles in base pairs (default: 500)
            blacklist: Optional blacklist file or DataFrame for filtering tiles.
                If None, uses assembly-specific blacklist. Set to False to skip.

        Returns:
            AnnData with:
                - X: tile count matrix (celltypes × tiles)
                - obs: celltype metadata
                - var: tile positions (Chromosome, Start, End)
                - uns: metadata including assembly
        """
        from scanpy import AnnData as sc_AnnData

        from gcell.dna.genome import ChromSize

        from caesar.preprocess.core.tile_filtering import filter_tiles

        if self.dataset is None:
            raise RuntimeError("BPCells matrix not loaded.")

        if not hasattr(self, "assembly") or self.assembly is None:
            raise ValueError(
                "Assembly not available. Cannot generate tiles. "
                "Please ensure the BPCells store has assembly metadata."
            )

        logging.info(f"Generating tiles ({tile_size}bp) for assembly {self.assembly}")

        # Generate tiles
        chrom_sizes = ChromSize(self.assembly)
        tiles_df = chrom_sizes.tiling_region(tile_size)

        # Apply chromosome filtering to match what's in the BPCells store
        if self.data_chroms is not None:
            tiles_df = tiles_df[tiles_df["Chromosome"].isin(self.data_chroms)]

        # Apply blacklist filtering if requested
        if blacklist is not False:
            tiles_df = filter_tiles(
                tiles_df,
                blacklist=blacklist,
                assembly=self.assembly,
            )

        if tiles_df.empty:
            raise ValueError("No tiles remaining after filtering.")

        # Reset index and add tile index
        tiles_df = tiles_df.reset_index(drop=True)
        n_tiles = len(tiles_df)
        logging.info(f"Processing {n_tiles} tiles for {len(self.ids)} celltypes")

        # Get counts for all tiles using the BPCells get_counts method
        # This is memory-efficient as BPCells handles the chunking internally
        query_regions = tiles_df[["Chromosome", "Start", "End"]].rename(
            columns={"Chromosome": "chrom", "Start": "start", "End": "end"}
        )

        try:
            # Get counts: returns (n_regions, n_groups, region_size)
            counts = self.dataset.get_counts(query_regions)

            # Sum across the region width to get total counts per tile
            # Shape: (n_tiles, n_celltypes)
            tile_counts = counts.sum(axis=2, dtype="float32").T

            # Subset to only the celltypes in self.ids (in case of filtering)
            if len(self.ids) < len(self.celltypes):
                ct_indices = [self.celltype_to_idx[ct] for ct in self.ids]
                tile_counts = tile_counts[ct_indices, :]

        except Exception as e:
            logging.error(f"Failed to get tile counts from BPCells: {e}")
            raise

        # Create obs DataFrame with celltype information
        obs_df = pd.DataFrame({"celltype": self.ids}, index=self.ids)

        # Create var DataFrame with tile positions
        var_df = tiles_df[["Chromosome", "Start", "End"]].copy()
        var_df["tile_idx"] = np.arange(n_tiles)
        var_df.index = [f"{row.Chromosome}:{row.Start}-{row.End}" for _, row in var_df.iterrows()]
        var_df.index.name = "tile_id"

        # Create AnnData
        adata = sc_AnnData(
            X=tile_counts,
            obs=obs_df,
            var=var_df,
        )

        # Add metadata
        adata.uns["assembly"] = self.assembly
        adata.uns["tile_size"] = tile_size
        adata.uns["source"] = "CelltypeDenseBPCellsIO"
        adata.uns["path"] = str(self.path)

        logging.info(
            f"Created tile AnnData: {adata.n_obs} celltypes × {adata.n_vars} tiles"
        )

        return adata


class SignalDenseBPCellsIO(BPCellsIO, SignalDenseIO):
    """Handles Signal Dense (pseudobulk) data stored in BPCells format."""

    def __init__(self, path: str, mode: str = "r", chroms: list[str] | None = None):
        super().__init__(path, mode, dtype="float32", chroms=chroms)
        self.ids = ["default"]
        self._init_data_structure()
        self._init_chunking()

    def _get_group_indices_for_query(self, query: BaseQuery) -> dict[str, np.ndarray]:
        """Map query to the single group index."""
        if not isinstance(query, DefaultQuery):
            raise NotImplementedError(
                f"SignalDenseBPCellsIO only supports DefaultQuery, not {type(query)}"
            )

        return {"default": np.zeros(1, dtype=int)}

    def _calculate_libsize(self) -> float | dict | None:
        """Calculate and return library size(s)."""
        try:
            precomputed = self._try_load_libsize_from_bpcells()
        except Exception:
            precomputed = self._try_load_libsize_from_attrs("library_sizes")

        result = self._process_libsize_result(precomputed, self.ids)
        if result is None:
            logging.warning("Failed to load library size from the dataset or attributes")
        return result

    def _calculate_per_cell_libsize(self) -> None:
        """Not applicable for SignalDense."""
        logging.warning("Not applicable for SignalDense.")
        return None

    def __repr__(self):
        return f"SignalDenseBPCellsIO(path={self.path},\n\tmode={self.mode},\n\tchroms={self.chroms})"


class SingleSparseBPCellsIO(BPCellsIO, SingleSparseIO):
    """Handles Single Sparse (cells as groups) data stored in BPCells format.

    This class provides a complete BPCells backend implementation for single-cell
    sparse data, with full compatibility with the Caesar IO interface.

    Mixins Inherited:
        - ClusteringMixin: Provides clustering methods (create_and_cluster_tiles_adata, etc.)
        - CallPeaksMixin: Peak calling functionality
        - FootprintMixin: TF footprinting analysis
        - QuantileNormalizationMixin: Quantile normalization
        - TSSExtractionMixin: TSS-centered signal extraction
        - LabelMixin: Cell/sample labeling utilities
        - LibrarySizeAnnotationMixin: Library size annotation

    Implemented Methods:
        - create_adata(): Creates AnnData objects from peaks/tiles
        - get_count_matrix(): Computes count matrices with chunked processing and memory optimization
        - write_peaks(): Saves peaks as BED files (with special handling for tiles)
        - get_peaks(): Loads peaks from BED files OR generates tiles on-the-fly for "tiles" peak_name
        - _load_metadata_from_files(): Loads cell metadata from CSV files
        - _calculate_libsize(): Calculates total and per-cell library sizes

    Special Features:
        - On-the-fly tile generation for peak_name="tiles" to avoid large disk files
        - Automatic cell filtering based on library size thresholds
        - Efficient chunked processing to handle large datasets within memory constraints
        - Support for metadata queries and cell subset selection

    Storage Format:
        - BPCells binary format with precalculated insertion matrices
        - Metadata stored in metadata.csv
        - Peaks stored as BED files ({peak_name}.bed)
        - Attributes stored in attrs.json
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        peaks_name: str = None,
        chroms: list[str] | None = None,
    ):
        # Define paths for metadata
        # Set up cell attributes
        super().__init__(path, mode, dtype="int32", chroms=chroms)

        # Initialize data structure
        self._init_data_structure()

        # Set n_cells after ids is properly initialized
        self.n_cells = len(self.ids) if self.ids is not None else 0

        # Load peaks if requested
        self.peaks_name = peaks_name
        if peaks_name is not None:
            self.peaks = self.get_peaks(peaks_name)

    def _init_data_structure(self):
        """
        Initialize sparse data specific attributes and chunking.
        """
        super()._init_data_structure()
        self._init_chunking()

        # Initialize per-cell library size before loading metadata
        self._per_cell_libsize = self._calculate_per_cell_libsize()

        # Load metadata from files
        self.metadata = self._load_metadata_from_files()

        # Calculate libsize
        self._libsize = self._calculate_libsize()

    def _load_metadata_from_files(self) -> pd.DataFrame:
        """Load cell metadata from CSV and TSV files."""
        df = None
        self.metadata_path = os.path.join(self.path, "metadata.csv")
        if os.path.exists(self.metadata_path):
            df = pd.read_csv(self.metadata_path)
            if "barcodes" not in df.columns:
                if "barcode" in df.columns:
                    df.rename(columns={"barcode": "barcodes"}, inplace=True)
                else:
                    raise ValueError("Metadata CSV must contain a 'barcodes' column.")

            # Sort according to barcodes.tsv if available
            if self.ids:
                if set(df["barcodes"]) == set(self.ids):
                    barcode_order = {bc: i for i, bc in enumerate(self.ids)}
                    df["_sort_idx"] = df["barcodes"].map(barcode_order)
                    df = (
                        df.sort_values("_sort_idx")
                        .drop("_sort_idx", axis=1)
                        .reset_index(drop=True)
                    )
                else:
                    df = df.sort_values("barcodes").reset_index(drop=True)
            else:
                df = df.sort_values("barcodes").reset_index(drop=True)

        # Create minimal DataFrame if no CSV exists
        if df is None:
            if not self.ids:
                raise FileNotFoundError(
                    f"Missing both metadata.csv and barcodes.tsv in {self.path}"
                )
            df = pd.DataFrame({"barcodes": self.ids})

        # Add required columns
        data_key = self._load_attr("data_key") or os.path.basename(self.path)
        if "cell_idx" not in df.columns:
            df["cell_idx"] = range(len(df))
        if "data_key" not in df.columns:
            df["data_key"] = data_key
        if "per_cell_libsize" not in df.columns:
            df["per_cell_libsize"] = np.nan

        # Calculate per-cell library sizes
        if self._per_cell_libsize is not None:
            df["per_cell_libsize"] = df["barcodes"].map(self._per_cell_libsize)

        # Calculate and save label-level library sizes
        if "labels" in df.columns:
            libsize = (
                df.groupby("labels")
                .agg({"per_cell_libsize": "sum", "barcodes": "count"})
                .rename(
                    columns={"per_cell_libsize": "libsize", "barcodes": "cell_counts"}
                )
            )
            libsize["frag_per_cell"] = (
                libsize["libsize"] / libsize["cell_counts"]
            ).astype(int)
            self._save_libsize_to_attrs(libsize.to_dict()["libsize"], "libsize_data")

        return df

    def _calculate_libsize(self) -> float | None:
        """Calculate per-label total library size from metadata."""
        return self._load_attr("libsize_data")

    def _calculate_per_cell_libsize(self) -> dict | float | None:
        """Calculate per-cell library size from metadata."""
        precomputed = self._try_load_libsize_with_fallback()

        result = self._process_libsize_result(precomputed, self.ids)
        if result is None:
            logging.warning("Failed to load library size from the dataset or attributes")
        return result

    def _try_load_libsize_with_fallback(self):
        """Try loading library sizes from multiple sources."""
        # Try BPCells dataset first
        try:
            precomputed = self._try_load_libsize_from_bpcells()
            if precomputed is not None:
                return precomputed
        except Exception:
            pass

        # Try attrs.json
        try:
            precomputed = self._try_load_libsize_from_attrs("library_sizes")
            if precomputed is not None:
                return precomputed
        except Exception:
            pass

        # Try library_size.json file directly
        library_size_path = os.path.join(self.path, "library_size.json")
        if os.path.exists(library_size_path):
            try:
                with open(library_size_path) as f:
                    libsize_data = json.load(f)
                return libsize_data.get("library_sizes")
            except Exception as e:
                logging.warning(f"Error reading library_size.json: {e}")

        return None

    def _get_all_ids(self):
        return self.ids

    def _resolve_query_to_barcodes(self, query: BaseQuery) -> list[str]:
        """Resolve a query to a list of barcodes.

        For KeyQuery, keys are treated as label values (from the 'labels' column in metadata),
        consistent with how MultiSparseZarrIO handles KeyQuery.
        """
        if isinstance(query, DefaultQuery):
            return self.ids
        elif isinstance(query, IndexQuery):
            if len(query.indices) == 0:
                return []
            try:
                return self.metadata.iloc[query.indices]["barcodes"].tolist()
            except IndexError:
                return []
        elif isinstance(query, KeyQuery):
            # KeyQuery keys are treated as labels (cell type/cluster labels)
            # This is consistent with MultiSparseZarrIO behavior
            if "labels" in self.metadata.columns:
                valid_keys = [k for k in query.keys if k in self.metadata["labels"].unique()]
                if valid_keys:
                    return self.metadata[self.metadata["labels"].isin(valid_keys)]["barcodes"].tolist()
            # Fallback: try matching against barcodes directly
            return [bc for bc in query.keys if bc in self.ids]
        elif isinstance(query, MetadataQuery):
            try:
                return self.metadata.query(query.query_string)["barcodes"].tolist()
            except Exception as e:
                logging.warning(f"Metadata query failed: {e}")
                return []
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

    def _get_group_indices_for_query(self, query: BaseQuery) -> dict[str, np.ndarray]:
        """Map query to cell indices."""
        barcode_to_idx = {bc: i for i, bc in enumerate(self.ids)}
        result = {}

        if isinstance(query, MultiKeyQuery):
            # Process each group separately
            for group_name, sub_query in query.mapping.items():
                barcodes = self._resolve_query_to_barcodes(sub_query)
                indices = [
                    barcode_to_idx[bc] for bc in barcodes if bc in barcode_to_idx
                ]
                if indices:
                    result[group_name] = np.array(indices, dtype=int)
        else:
            # Single output group
            barcodes = self._resolve_query_to_barcodes(query)
            indices = [barcode_to_idx[bc] for bc in barcodes if bc in barcode_to_idx]
            if indices:
                result["default"] = np.array(indices, dtype=int)

        return result

    def write_peaks(
        self,
        peaks: str | pd.DataFrame,
        name: str = "peaks",
        columns_to_include=["Start", "End"],
    ):
        """Write peaks to a BED file in the BPCells directory.

        Args:
            peaks: Peaks as DataFrame or file path
            name: Name identifier for the peaks (used as filename: {name}.bed)
            columns_to_include: Columns to include in the BED file (default: ["Start", "End"])

        Note:
            This saves peaks to {self.path}/{name}.bed, similar to how zarr
            saves peaks to zarr[f'{name}/regions/{chrom}'].
            For "tiles", this is skipped to avoid large disk files - tiles are
            generated on-the-fly when needed via get_peaks("tiles").
        """
        # Skip saving tiles to disk - they're generated on-the-fly
        if name == "tiles":
            logging.debug(
                "Skipping save for 'tiles' - tiles are generated on-the-fly to avoid large disk files"
            )
            return

        if self.mode == "r":
            logging.warning("Cannot write peaks: BPCellsIO is in read-only mode.")
            return

        # Use helper to prepare BED DataFrame
        bed_df = self._prepare_bed_dataframe(peaks, name, columns_to_include)

        # Write BED file
        bed_path = os.path.join(self.path, f"{name}.bed")
        bed_df.to_csv(bed_path, sep="\t", header=False, index=False)
        logging.info(f"Saved {len(bed_df)} peaks to {bed_path}")

        # Store peak name in attrs for reference
        if "peak_names" not in self.attrs:
            self.attrs["peak_names"] = []
        if name not in self.attrs["peak_names"]:
            self.attrs["peak_names"].append(name)
            self._save_attr("peak_names", self.attrs["peak_names"])

        # Store column names for this peak set
        peak_cols_key = f"peak_columns_{name}"
        self.attrs[peak_cols_key] = list(bed_df.columns)
        self._save_attr(peak_cols_key, self.attrs[peak_cols_key])

    def get_peaks(self, name="peaks") -> pd.DataFrame | None:
        """Get peaks from a saved BED file, external file, or generate on-the-fly for tiles.

        Args:
            name: Name identifier for the peaks (looks for {name}.bed in self.path).
                If name is "tiles", generates tiles on-the-fly (500bp) instead of loading from disk.

        Returns:
            DataFrame with peaks or None if not found
        """
        # Special handling for "tiles" - generate on-the-fly to avoid large disk files
        if name == "tiles":
            tiles = self._generate_tiles_on_the_fly()
            if tiles is not None:
                return tiles
            # Fall through to check for saved tiles.bed file

        # Check for saved peaks in the BPCells directory
        bed_path = os.path.join(self.path, f"{name}.bed")
        peak_cols_key = f"peak_columns_{name}"

        peaks_df = self._read_bed_file(bed_path, name, peak_cols_key)
        if peaks_df is not None:
            logging.info(f"Loaded {len(peaks_df)} peaks from {bed_path}")
            return peaks_df

        # Fall back to checking self.peaks_name (external file)
        if self.peaks_name and os.path.exists(self.peaks_name):
            try:
                return read_peaks(self.peaks_name)
            except Exception as e:
                logging.error(f"Failed to read peaks from {self.peaks_name}: {e}")
                return None

        if hasattr(self, "peaks") and isinstance(self.peaks, pd.DataFrame):
            return self.peaks

        logging.warning(f"No peaks file associated with name '{name}'.")
        return None

    def _generate_tiles_on_the_fly(self) -> pd.DataFrame | None:
        """Generate genomic tiles on-the-fly for clustering.

        Returns:
            DataFrame with tiles or None if generation failed
        """
        if not HAS_GCELL or ChromSize is None:
            logging.warning(
                "gcell not available. Cannot generate tiles on-the-fly. "
                "Falling back to checking for saved tiles.bed file."
            )
            return None

        if not hasattr(self, "assembly") or self.assembly is None:
            logging.warning("Assembly not available. Cannot generate tiles on-the-fly.")
            return None

        try:
            logging.info(f"Generating tiles on-the-fly (500bp) for assembly {self.assembly}")
            from caesar.preprocess.core.tile_filtering import filter_tiles

            chrom_sizes = ChromSize(self.assembly)
            tiles = chrom_sizes.tiling_region(500)

            # Apply shared filtering logic (chromosomes + blacklist)
            tiles = filter_tiles(
                tiles,
                blacklist=None,  # Blacklist filtering handled by clustering config
                assembly=self.assembly,
            )

            # Filter to only chromosomes that exist in BPCells data
            if hasattr(self, "data_chroms") and self.data_chroms:
                available_chroms = set(self.data_chroms)
                tiles = tiles[tiles["Chromosome"].isin(available_chroms)]
                logging.info(
                    f"Filtered tiles to {len(available_chroms)} chromosomes "
                    f"available in BPCells data"
                )

            logging.info(f"Generated {len(tiles)} tiles on-the-fly")
            return tiles
        except Exception as e:
            logging.warning(
                f"Failed to generate tiles on-the-fly: {e}. "
                "Falling back to checking for saved tiles.bed file."
            )
            return None

    def _process_chunk(
        self,
        chrom: str,
        chunk_start: int,
        chunk_end_pos: int,
        chunk_peaks: pd.DataFrame,
        valid_cell_indices: np.ndarray | None,
        min_libsize: int,
        max_libsize: int,
    ) -> tuple[csr_matrix, list] | None:
        """
        Process a single chunk of peaks and return count matrix with peak indices.

        Args:
            chrom: Chromosome name
            chunk_start: Start position of chunk
            chunk_end_pos: End position of chunk
            chunk_peaks: DataFrame of peaks overlapping this chunk
            valid_cell_indices: Array of valid cell indices (None if all cells)
            min_libsize: Minimum library size (for error messages)
            max_libsize: Maximum library size (for error messages)

        Returns:
            Tuple of (sparse count matrix for this chunk, array of original indices),
            or None if processing failed
        """
        query_regions = chunk_peaks[["Chromosome", "Start", "End"]].rename(
            columns={"Chromosome": "chrom", "Start": "start", "End": "end"}
        )

        try:
            # Get counts for this chunk
            counts = self.dataset.get_counts(query_regions)

            # Sum across basepair dimension and transpose
            peak_counts = counts.sum(axis=2, dtype=self.dtype).T

            # First filter to cells in self.ids (if dataset has more cells than self.ids)
            if hasattr(self, '_dataset_cell_indices') and self._dataset_cell_indices is not None:
                peak_counts = peak_counts[self._dataset_cell_indices, :]

            # Then filter cells based on library size
            if valid_cell_indices is not None:
                peak_counts = peak_counts[valid_cell_indices, :]

            # Return matrix and original indices together
            original_indices = chunk_peaks["original_indices"].values
            return csr_matrix(peak_counts), original_indices
        except Exception as e:
            error_msg = str(e)
            if "bad_alloc" in error_msg.lower():
                logging.error(
                    f"Memory allocation error for {chrom} chunk {chunk_start}-{chunk_end_pos}. "
                    f"The BPCells C++ backend queries ALL cells ({self.n_cells} cells) even with chunking. "
                    f"To fix this, you must re-precalculate the insertion matrix with filtered cells:\n"
                    f"1. Calculate library_size from fragments or use PrecalculatedInsertionMatrix.library_size\n"
                    f"2. Filter cells (e.g., library_size >= {min_libsize} and <= {max_libsize})\n"
                    f"3. Re-precalculate using keeper_cells in import_10x_fragments or "
                    f"filtered cell_groups in precalculate_insertion_counts"
                )
            else:
                logging.error(f"BPCells query failed for {chrom} chunk {chunk_start}-{chunk_end_pos}: {e}")
            # Return None to indicate failure - caller will create empty matrix
            return None

    def get_count_matrix(
        self, peaks=None, peak_name="peak", overwrite: bool = False, name=None, **kwargs
    ) -> csr_matrix:
        """
        Get count matrix for the given peaks.

        Args:
            peaks: Peaks as DataFrame or file path. If None, will try to load
                peaks using peak_name.
            peak_name: Name identifier for the peaks. Used for storing/retrieving
                peaks. Default is 'peak'.
            overwrite: Whether to overwrite existing saved peaks. Default is False.
            name: Alias for peak_name (for compatibility with ClusteringMixin).
                If provided, takes precedence over peak_name.

        Returns:
            Sparse count matrix (cells x peaks)

        Note:
            If peaks are provided and not already saved, they will be saved
            using write_peaks() for future retrieval, similar to how zarr
            saves peaks to zarr[f'{peak_name}/regions/{chrom}'].
        """
        if self.dataset is None:
            raise RuntimeError("BPCells matrix not loaded.")

        # Handle 'name' parameter for compatibility with ClusteringMixin
        if name is not None:
            peak_name = name

        # Try to load existing peaks if not provided
        if peaks is None:
            peaks_df = self.get_peaks(peak_name)
            if peaks_df is None:
                raise ValueError(
                    f"No peaks found for peak_name '{peak_name}' and no peaks provided."
                )
        else:
            peaks_df = read_peaks(peaks)

        # Save peaks if they don't exist or if overwrite is True
        # Skip saving tiles - they're generated on-the-fly
        if peak_name != "tiles":
            bed_path = os.path.join(self.path, f"{peak_name}.bed")
            if overwrite or not os.path.exists(bed_path):
                if self.mode != "r":
                    logging.info(f"Saving peaks as {peak_name}.bed")
                    self.write_peaks(peaks_df, peak_name)
                elif overwrite:
                    logging.warning(
                        "Cannot save peaks: BPCellsIO is in read-only mode. "
                        "Skipping peak save."
                    )
        else:
            logging.debug(
                "Skipping save for 'tiles' - tiles are generated on-the-fly to avoid large disk files"
            )
        if not all(col in peaks_df.columns for col in ["Chromosome", "Start", "End"]):
            raise ValueError(
                "Peaks DataFrame must contain 'Chromosome', 'Start', 'End' columns."
            )

        # Use peaks as-is (no sorting) - matrix columns will match this order
        # Store original_indices for chunk processing (BPCells-specific)
        # This is needed for chunk-based processing to track peak order
        peaks_df["original_indices"] = peaks_df.index.values
        peaks_df["length"] = peaks_df["End"] - peaks_df["Start"]
        unique_lengths = peaks_df["length"].unique()
        n_peaks = len(peaks_df)

        if len(unique_lengths) == 1:
            # Optimized case: all peaks have same length
            # For very large peak sets (e.g., genome tiles), process in chunks
            # and filter cells after each chunk to avoid memory issues
            chunk_size = kwargs.get("chunk_size", 500000)  # Default 500kbp chunks (can be overridden)
            min_libsize = kwargs.get("min_libsize", 1000)
            max_libsize = kwargs.get("max_libsize", 50000)

            if n_peaks > 1000000:  # Threshold for chunking
                chunk_size_mb = chunk_size / 1000000
                chunk_size_kb = chunk_size / 1000
                if chunk_size >= 1000000:
                    chunk_str = f"{chunk_size_mb:.1f}M"
                else:
                    chunk_str = f"{chunk_size_kb:.0f}k"
                logging.info(
                    f"Large peak set detected ({n_peaks} peaks). "
                    f"Processing in {chunk_str} bp chunks with cell filtering "
                    f"(library_size >= {min_libsize} and <= {max_libsize})."
                )

                # First, map self.ids to dataset indices (if dataset has group_names)
                # But we need to check library_size length to determine valid range
                dataset_cell_indices = None
                library_size_len = None
                try:
                    if hasattr(self.dataset, "library_size") and self.dataset.library_size is not None:
                        library_size_len = len(self.dataset.library_size)
                except:
                    pass

                if self.ids is not None and hasattr(self.dataset, "group_names") and self.dataset.group_names is not None:
                    # Create mapping from self.ids to dataset indices
                    id_to_dataset_idx = {id_val: idx for idx, id_val in enumerate(self.dataset.group_names)}
                    dataset_cell_indices = np.array([id_to_dataset_idx[id_val] for id_val in self.ids if id_val in id_to_dataset_idx])

                    # Filter to valid indices based on library_size length if available
                    if library_size_len is not None:
                        valid_mask = dataset_cell_indices < library_size_len
                        if not np.all(valid_mask):
                            invalid_count = np.sum(~valid_mask)
                            logging.warning(
                                f"{invalid_count} cells from self.ids map to dataset indices >= library_size length ({library_size_len}), filtering them out"
                            )
                            dataset_cell_indices = dataset_cell_indices[valid_mask]

                    if len(dataset_cell_indices) < len(self.ids):
                        logging.warning(
                            f"Only {len(dataset_cell_indices)}/{len(self.ids)} cells from self.ids are valid in dataset"
                        )
                elif self.ids is not None:
                    # If no group_names, assume self.ids indices match dataset indices
                    # But still respect library_size length
                    if library_size_len is not None:
                        dataset_cell_indices = np.arange(min(len(self.ids), library_size_len))
                    else:
                        dataset_cell_indices = np.arange(len(self.ids))

                # Store dataset_cell_indices for use in _process_chunk
                self._dataset_cell_indices = dataset_cell_indices

                # Get library sizes for cell filtering
                try:
                    library_sizes = self.dataset.library_size
                    if library_sizes is not None:
                        # If we have dataset_cell_indices, filter library_sizes first
                        if dataset_cell_indices is not None:
                            # Ensure indices are within bounds
                            max_idx = len(library_sizes) - 1
                            valid_indices_mask = (dataset_cell_indices >= 0) & (dataset_cell_indices <= max_idx)
                            if not np.all(valid_indices_mask):
                                invalid_count = np.sum(~valid_indices_mask)
                                logging.warning(
                                    f"{invalid_count} dataset cell indices out of bounds, filtering them out"
                                )
                                dataset_cell_indices = dataset_cell_indices[valid_indices_mask]
                                # Update stored indices
                                self._dataset_cell_indices = dataset_cell_indices

                            if len(dataset_cell_indices) > 0:
                                library_sizes = library_sizes[dataset_cell_indices]
                                # Create boolean mask for valid cells within our subset
                                valid_cells_mask = (library_sizes >= min_libsize) & (library_sizes <= max_libsize)
                                # valid_cell_indices are positions in self.ids (after filtering)
                                valid_cell_indices = np.where(valid_cells_mask)[0]
                                logging.info(
                                    f"Filtering cells: {len(valid_cell_indices)}/{len(dataset_cell_indices)} "
                                    f"cells pass library_size filter (from {len(self.ids)} total cells in self.ids)"
                                )
                            else:
                                logging.warning("No valid dataset cell indices after filtering")
                                valid_cell_indices = None
                        else:
                            # Create boolean mask for valid cells
                            valid_cells_mask = (library_sizes >= min_libsize) & (library_sizes <= max_libsize)
                            valid_cell_indices = np.where(valid_cells_mask)[0]
                            logging.info(
                                f"Filtering cells: {len(valid_cell_indices)}/{len(library_sizes)} "
                                f"cells pass library_size filter"
                            )
                    else:
                        logging.warning("Library sizes not available, processing all cells")
                        valid_cell_indices = None
                except Exception as e:
                    logging.warning(f"Could not get library sizes: {e}. Processing all cells.")
                    valid_cell_indices = None

                # Process chromosome by chromosome using shared processing logic
                # BPCells-specific: concatenate chromosomes with hstack after processing
                chromosome_results = []

                def bpcell_chunk_processor(chromosome, chunk_start, chunk_end_pos, chunk_peaks, **kwargs):
                    """BPCells chunk processor adapter for shared processing."""
                    # Extract BPCells-specific kwargs
                    valid_cell_indices = kwargs.get("valid_cell_indices")
                    min_libsize = kwargs.get("min_libsize", 0)
                    max_libsize = kwargs.get("max_libsize", None)

                    # Call BPCells-specific chunk processor
                    result = self._process_chunk(
                        chromosome,
                        chunk_start,
                        chunk_end_pos,
                        chunk_peaks,
                        valid_cell_indices,
                        min_libsize,
                        max_libsize,
                    )

                    if result is None:
                        # Create empty matrix for failed chunk
                        n_cells_chunk = len(valid_cell_indices) if valid_cell_indices is not None else self.n_cells
                        original_indices = chunk_peaks["original_indices"].values
                        return csr_matrix((n_cells_chunk, len(chunk_peaks)), dtype=self.dtype), original_indices

                    return result

                for chrom in tqdm(peaks_df["Chromosome"].unique(), desc="Processing chromosomes"):
                    chrom_peaks = peaks_df[peaks_df["Chromosome"] == chrom].copy()
                    if len(chrom_peaks) == 0:
                        continue

                    # Use shared chromosome processing with sparse concat + reorder
                    chrom_matrix, chrom_peaks_reordered = process_chromosome_with_chunks(
                        chromosome=chrom,
                        peaks_chr=chrom_peaks,
                        process_chunk_fn=bpcell_chunk_processor,
                        chunk_size=chunk_size,
                        chunk_starts=None,  # Auto-generate from peaks
                        n_workers=1,  # Use ThreadPoolExecutor from shared code
                        executor_class=concurrent.futures.ThreadPoolExecutor,
                        valid_cell_indices=valid_cell_indices,
                        min_libsize=min_libsize,
                        max_libsize=max_libsize,
                    )

                    chromosome_results.append(chrom_matrix)

                # BPCells-specific: Concatenate all chromosomes with hstack
                if chromosome_results:
                    result = hstack(chromosome_results)
                    return result
                else:
                    n_cells_result = len(valid_cell_indices) if valid_cell_indices is not None else self.n_cells
                    return csr_matrix((n_cells_result, n_peaks), dtype=self.dtype)
            else:
                # Small peak set: process all at once, but still filter cells if library_size available
                query_regions = peaks_df[["Chromosome", "Start", "End"]].rename(
                    columns={"Chromosome": "chrom", "Start": "start", "End": "end"}
                )

                try:
                    # Get counts for all regions
                    counts = self.dataset.get_counts(query_regions)

                    # Sum across basepair dimension and transpose
                    peak_counts = counts.sum(axis=2, dtype=self.dtype).T

                    # First filter to cells in self.ids (if dataset has more cells than self.ids)
                    if self.ids is not None and hasattr(self.dataset, "group_names") and self.dataset.group_names is not None:
                        # Create mapping from self.ids to dataset indices
                        id_to_dataset_idx = {id_val: idx for idx, id_val in enumerate(self.dataset.group_names)}
                        dataset_cell_indices = np.array([id_to_dataset_idx[id_val] for id_val in self.ids if id_val in id_to_dataset_idx])
                        if len(dataset_cell_indices) > 0:
                            peak_counts = peak_counts[dataset_cell_indices, :]

                    # Filter cells if library_size available
                    try:
                        library_sizes = self.dataset.library_size
                        if library_sizes is not None:
                            min_libsize = kwargs.get("min_libsize", 1000)
                            max_libsize = kwargs.get("max_libsize", 50000)
                            # If we filtered to dataset_cell_indices, use those library sizes
                            if 'dataset_cell_indices' in locals() and dataset_cell_indices is not None:
                                library_sizes = library_sizes[dataset_cell_indices]
                            valid_cells_mask = (library_sizes >= min_libsize) & (library_sizes <= max_libsize)
                            valid_cell_indices = np.where(valid_cells_mask)[0]
                            if len(valid_cell_indices) < len(library_sizes):
                                logging.info(
                                    f"Filtering cells: {len(valid_cell_indices)}/{len(library_sizes)} "
                                    f"cells pass library_size filter"
                                )
                                peak_counts = peak_counts[valid_cell_indices, :]
                    except Exception:
                        pass  # Continue without filtering if library_size not available

                    # Return as sparse matrix
                    return csr_matrix(peak_counts)

                except Exception as e:
                    logging.error(f"BPCells query failed: {e}")
                    return csr_matrix((self.n_cells, n_peaks), dtype=self.dtype)

        else:
            # Slower case: peaks have different lengths
            logging.warning("Peaks have multiple lengths. Processing group by group.")
            results = {}
            original_indices = peaks_df.index.tolist()

            for length, group in tqdm(
                peaks_df.groupby("length"), desc="Processing peak lengths"
            ):
                query_regions = group[["Chromosome", "Start", "End"]].rename(
                    columns={"Chromosome": "chrom", "Start": "start", "End": "end"}
                )
                group_indices = group.index.tolist()

                try:
                    # Get counts for this group
                    counts = self.dataset.get_counts(query_regions)

                    # Sum and transpose
                    peak_counts = counts.sum(axis=2, dtype=self.dtype).T

                    # Store each peak column
                    for i, idx in enumerate(group_indices):
                        results[idx] = csr_matrix(peak_counts[:, i : i + 1])

                except Exception as e:
                    logging.error(f"Query failed for length {length}: {e}")
                    for idx in group_indices:
                        results[idx] = csr_matrix((self.n_cells, 1), dtype=self.dtype)

            # Reassemble matrix in original order
            if not results:
                return csr_matrix((self.n_cells, n_peaks), dtype=self.dtype)

            columns = [
                results.get(idx, csr_matrix((self.n_cells, 1), dtype=self.dtype))
                for idx in original_indices
            ]
            return hstack(columns, format="csr")

    def create_adata(
        self,
        peak_name="genome_tiles",
        peaks=None,
    ):
        """
        Create an AnnData object from the BPCells data.

        This method computes or retrieves the count matrix for the given peaks
        and creates an AnnData object suitable for downstream analysis.

        Args:
            peak_name (str): Name identifier for the peaks. Used to retrieve
                peaks if they were previously loaded. Default is 'genome_tiles'.
            peaks (str | pd.DataFrame, optional): The peaks to get the count matrix for.
                Can be a file path or a DataFrame. Must contain 'Chromosome', 'Start',
                and 'End' columns. If None, will try to retrieve peaks using peak_name.

        Returns:
            AnnData: An AnnData object containing:
                - X: Sparse count matrix (cells x peaks)
                - obs: Cell metadata
                - var: Peak information (if peaks are provided)

        Note:
            This method uses the existing get_count_matrix method which leverages
            BPCells' get_counts for efficient computation. The implementation is
            similar to MultiSparseZarrIO.create_adata but for a single sample.
        """
        from scanpy import AnnData

        # Get peaks if not provided
        if peaks is None:
            peaks = self.get_peaks(peak_name)
            if peaks is None:
                raise ValueError(
                    f"No peaks found for peak_name '{peak_name}' and no peaks provided. "
                    "Please provide peaks as a DataFrame or file path."
                )
        # Only sample if we have more than 1000 peaks
        if len(peaks) > 1000:
            peaks = peaks.sample(1000)
        # Get count matrix using the existing method
        # This uses self.dataset.get_counts internally (from PrecalculatedInsertionMatrix)
        # Disable libsize filtering to ensure count_matrix has same rows as metadata
        count_matrix = self.get_count_matrix(
            peaks=peaks, peak_name=peak_name, chunk_size=2000000,
            min_libsize=0, max_libsize=float('inf')
        )

        # Ensure peaks is a DataFrame for var
        if isinstance(peaks, str):
            peaks_df = read_peaks(peaks)
        else:
            peaks_df = peaks.copy()

        # Ensure peaks DataFrame has proper index
        if peaks_df.index.name is None or peaks_df.index.name == "index":
            peaks_df.index.name = "peak_id"

        # Align metadata with count_matrix rows
        # The count_matrix might have fewer rows if some cells don't exist in the dataset
        obs_df = self.metadata.copy()
        if count_matrix.shape[0] != len(obs_df):
            # The count_matrix rows correspond to cells that exist in both self.ids and dataset
            # We need to filter metadata to match
            if hasattr(self.dataset, "group_names") and self.dataset.group_names is not None:
                # Filter to cells that exist in the dataset
                valid_ids = [id_val for id_val in self.ids if id_val in set(self.dataset.group_names)]
                obs_df = obs_df[obs_df["barcodes"].isin(valid_ids)].reset_index(drop=True)

        # Create AnnData object
        adata = AnnData(X=count_matrix, obs=obs_df, var=peaks_df)

        return adata


class MultiSparseBPCellsIO(BPCellsIO, MultiSparseIO):
    """Handles multiple SingleSparseBPCellsIO datasets with unified access.

    This class provides a complete BPCells backend implementation for multi-sample
    sparse data, managing multiple SingleSparseBPCellsIO datasets with unified
    access across all samples.

    Mixins Inherited:
        - ClusteringMixin: Provides clustering methods across all samples
        - CallPeaksMixin: Peak calling functionality
        - FootprintMixin: TF footprinting analysis
        - QuantileNormalizationMixin: Quantile normalization
        - TSSExtractionMixin: TSS-centered signal extraction
        - LabelMixin: Cell/sample labeling utilities
        - LibrarySizeAnnotationMixin: Library size annotation

    Implemented Methods:
        - create_adata(): Creates AnnData objects from peaks/tiles across all samples
        - get_count_matrix(): Computes count matrices across all samples with aggregation
        - write_peaks(): Saves peaks as BED files for each sample
        - get_peaks(): Aggregates peaks from all samples
        - _load_metadata_from_files(): Loads cell metadata across all samples
        - _calculate_libsize(): Calculates library sizes across all samples

    Key Features:
        - Manages multiple BPCells datasets as a unified collection
        - Supports per-sample and cross-sample operations
        - Automatic metadata aggregation across samples
        - Sample-aware cell indexing and filtering

    Storage Format:
        - Multiple BPCells directories, one per sample
        - Each sample has its own metadata.csv, peaks BED files, and attrs.json
        - Unified access layer for cross-sample analysis
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        dtype: str = "float32",
        pattern: str = "single_sparse_*",
        peaks_name: str = None,
        chroms: list[str] | None = None,
    ):
        # Initialize GenomicIO first
        super(MultiSparseIO, self).__init__(path, mode, dtype)

        # Setup parameters
        self.parent_dir = path
        self.pattern = pattern
        self.peaks_name = peaks_name

        # Find sample directories
        from glob import glob

        self.sample_dirs = sorted(glob(os.path.join(self.parent_dir, self.pattern)))
        if not self.sample_dirs:
            raise FileNotFoundError(
                f"No sample directories matching '{pattern}' found in {self.parent_dir}"
            )

        # Load child datasets
        self.bpcells_dict = self._load_samples(chroms)
        if not self.bpcells_dict:
            raise RuntimeError("Failed to load any child BPCells datasets.")

        # Setup from first child
        self._setup_from_first_child()

        # Combine metadata and set IDs
        self.metadata = self._combine_metadata()
        self.ids = list(self.bpcells_dict.keys())

        # Initialize structures
        self._init_data_structure()
        self._init_chunking()

    def _load_samples(self, chroms) -> dict[str, SingleSparseBPCellsIO]:
        """Load SingleSparseBPCellsIO instances for each sample."""
        bpcells_dict = {}
        for sample_dir in tqdm(self.sample_dirs, desc="Loading BPCells samples"):
            sample_name = os.path.basename(sample_dir)
            try:
                bp_io = SingleSparseBPCellsIO(
                    sample_dir,
                    mode=self.mode,
                    peaks_name=self.peaks_name,
                    chroms=chroms,
                )
                bpcells_dict[sample_name] = bp_io
            except Exception as e:
                logging.error(f"Failed to load sample '{sample_name}': {e}")
        return bpcells_dict

    def _setup_from_first_child(self):
        """Copy common attributes from first child dataset."""
        self.first_bpcells = next(iter(self.bpcells_dict.values()))
        self.assembly = self.first_bpcells.assembly
        self._active_chroms = self.first_bpcells.chroms
        self.assembly_chrom_sizes = self.first_bpcells.assembly_chrom_sizes
        self.dataset = None  # Not applicable at multi level

    def _combine_metadata(self) -> pd.DataFrame:
        """Combine metadata from all child datasets."""
        all_meta = []
        for sample_name, bp_io in self.bpcells_dict.items():
            meta = bp_io.metadata.copy()
            meta["data_key"] = sample_name
            all_meta.append(meta)

        if not all_meta:
            return pd.DataFrame()

        combined = pd.concat(all_meta, ignore_index=True)
        combined["global_cell_idx"] = combined.index
        return combined

    def _init_data_structure(self):
        """Initialize combined data structure."""
        self._per_cell_libsize = self._calculate_per_cell_libsize()
        # Calculate libsize, but don't fail if metadata doesn't have required columns
        # (libsize is not required for clustering operations)
        try:
            self._libsize = self._calculate_libsize()
        except Exception as e:
            logging.debug(f"Could not calculate libsize (not required for clustering): {e}")
            self._libsize = {}

    @property
    def data_chroms(self) -> list[str]:
        """Return common chromosomes across all datasets."""
        if not self.bpcells_dict:
            return []
        common_chroms = set(self.first_bpcells.data_chroms)
        for bp_io in self.bpcells_dict.values():
            common_chroms.intersection_update(bp_io.data_chroms)
        return sorted(list(common_chroms))

    def _init_chunking(self, first_chrom: str | None = None) -> None:
        """Set up chunking parameters."""
        self.chunk_size = 2000000  # Standard chunk size
        self.dir_chunked = False  # BPCells doesn't use dir chunking
        self._init_chunking_done = True

    def _load_attr(self, key):
        """Load an attribute from the first child dataset."""
        if self.first_bpcells:
            return self.first_bpcells._load_attr(key)
        return None

    def _save_attr(self, key, value):
        """Save is not applicable for multi-sparse."""
        logging.warning(
            "MultiSparseBPCellsIO does not support saving attributes directly."
        )

    def _resolve_query_to_sample_indices(
        self, query: BaseQuery
    ) -> dict[str, dict[str, np.ndarray]]:
        """Map query to sample-specific indices."""
        query_map = {}

        if isinstance(query, DefaultQuery):
            # Each sample becomes an output group
            for sample_name, bp_io in self.bpcells_dict.items():
                query_map[sample_name] = {sample_name: np.arange(len(bp_io.metadata))}

        elif isinstance(query, IndexQuery):
            # Indices refer to global metadata
            target_meta = self.metadata.iloc[query.indices]
            grouped = (
                target_meta.groupby("data_key")["cell_idx"]
                .apply(lambda x: x.values)
                .to_dict()
            )
            query_map["default"] = grouped

        elif isinstance(query, KeyQuery):
            # Keys are sample names
            for sample_name in query.keys:
                if sample_name in self.bpcells_dict:
                    query_map[sample_name] = {
                        sample_name: np.arange(
                            len(self.bpcells_dict[sample_name].metadata)
                        )
                    }

        elif isinstance(query, MetadataQuery):
            # Apply metadata query to combined metadata
            try:
                target_meta = self.metadata.query(query.query_string)
                grouped = (
                    target_meta.groupby("data_key")["cell_idx"]
                    .apply(lambda x: x.values)
                    .to_dict()
                )
                query_map["default"] = grouped
            except Exception as e:
                logging.warning(f"Metadata query failed: {e}")

        elif isinstance(query, MultiKeyQuery):
            # Map each output group to its respective samples/cells
            for group_name, group_query in query.mapping.items():
                sub_result = self._resolve_query_to_sample_indices(group_query)
                if sub_result:
                    # Flatten all subgroup results into a single group
                    merged = {}
                    for sub_map in sub_result.values():
                        for sample, indices in sub_map.items():
                            if sample in merged:
                                merged[sample] = np.concatenate(
                                    [merged[sample], indices]
                                )
                            else:
                                merged[sample] = indices
                    query_map[group_name] = merged

        return query_map

    def _get_coverage_for_query(
        self, chr_name: str, start: int, end: int, query: BaseQuery, sparse: bool
    ) -> RawCoverageDict:
        """
        Retrieve coverage data for a given region and query across all samples.
        Returns a dictionary mapping group names to coverage data.
        """
        # Map query to sample-specific indices
        sample_indices_map = self._resolve_query_to_sample_indices(query)
        if not sample_indices_map:
            return {}

        # Process each output group
        result_dict = {}
        for group_name, sample_indices in sample_indices_map.items():
            group_results = []

            # Get data from each sample
            for sample_name, indices in sample_indices.items():
                if sample_name in self.bpcells_dict and len(indices) > 0:
                    bp_io = self.bpcells_dict[sample_name]

                    # Create an IndexQuery for this sample's cells
                    sample_query = IndexQuery(indices)

                    # Get coverage from the sample
                    sample_coverage = bp_io._get_coverage_for_query(
                        chr_name, start, end, sample_query, sparse
                    )

                    # Add sample's data to group results
                    if sample_coverage and "default" in sample_coverage:
                        group_results.append(sample_coverage["default"])

            # Combine results for this group
            if group_results:
                if sparse:
                    # For sparse data, vstack the matrices
                    combined = vstack(group_results, format="csr")
                else:
                    # For dense data, sum the arrays
                    combined = np.sum(group_results, axis=0)

                result_dict[group_name] = combined

        return result_dict

    def _get_group_indices_for_query(self, query: BaseQuery) -> dict[str, np.ndarray]:
        """Not directly applicable for MultiSparseBPCellsIO."""
        raise NotImplementedError(
            "MultiSparseBPCellsIO doesn't implement _get_group_indices_for_query directly, it uses the member single sparse bpcells io"
        )

    def get_label_counts(
        self, threshold: int = 200, normalize: bool = False
    ) -> pd.Series:
        """
        Count cells by sample/label.

        Parameters:
        -----------
        threshold : int
            Minimum UMI count for a cell to be included
        normalize : bool
            Whether to normalize counts to proportions

        Returns:
        --------
        pd.Series
            Counts or proportions per sample
        """
        # Count cells per sample, possibly filtering by UMI threshold
        counts = {}
        for sample_name, bp_io in self.bpcells_dict.items():
            if threshold > 0 and bp_io._per_cell_libsize is not None:
                valid_cells = (bp_io._per_cell_libsize >= threshold).sum()
                counts[sample_name] = valid_cells
            else:
                counts[sample_name] = len(bp_io.metadata)

        result = pd.Series(counts)

        # Normalize if requested
        if normalize and result.sum() > 0:
            result = result / result.sum()

        return result

    def get_label_distribution(self, by: str = "sample") -> pd.DataFrame:
        """
        Get distribution of cells across samples and/or metadata categories.

        Parameters:
        -----------
        by : str
            Column in metadata to group by, defaults to "sample"

        Returns:
        --------
        pd.DataFrame
            Distribution of cells
        """
        if by == "sample":
            # Simple count per sample
            counts = self.get_label_counts(threshold=0)
            return pd.DataFrame(
                {
                    "sample": counts.index,
                    "count": counts.values,
                    "proportion": counts.values / counts.sum()
                    if counts.sum() > 0
                    else 0,
                }
            )
        else:
            # Group by metadata column
            if by not in self.metadata.columns:
                raise ValueError(f"Column '{by}' not found in metadata")

            grouped = self.metadata.groupby(by).size()
            total = grouped.sum()

            return pd.DataFrame(
                {
                    by: grouped.index,
                    "count": grouped.values,
                    "proportion": grouped.values / total if total > 0 else 0,
                }
            )

    def get_cell_qc_metrics(self) -> pd.DataFrame:
        """
        Calculate quality control metrics for cells across all samples.

        Returns:
        --------
        pd.DataFrame
            DataFrame with QC metrics per cell
        """
        # Start with combined metadata
        qc_df = self.metadata.copy()

        # Add library size if not already present
        if (
            "per_cell_libsize" not in qc_df.columns
            or qc_df["per_cell_libsize"].isna().all()
        ) and self._per_cell_libsize is not None:
            # Map by barcode
            barcode_to_libsize = self._per_cell_libsize.to_dict()
            qc_df["per_cell_libsize"] = qc_df.apply(
                lambda row: barcode_to_libsize.get(
                    f"{row['data_key']}:{row['barcodes']}", np.nan
                ),
                axis=1,
            )

        # Calculate additional metrics if needed
        if "z_score_libsize" not in qc_df.columns:
            # Helper function to calculate z-scores safely
            def z_score(x):
                if len(x) <= 1 or x.std() == 0:
                    return np.zeros_like(x)
                return (x - x.mean()) / x.std()

            # Calculate z-scores per sample
            if "per_cell_libsize" in qc_df.columns:
                qc_df["z_score_libsize"] = qc_df.groupby("data_key")[
                    "per_cell_libsize"
                ].transform(z_score)

        return qc_df

    def create_adata(
        self,
        peak_name="genome_tiles",
        peaks=None,
    ):
        """
        Create an AnnData object from the BPCells data across all samples.

        This method computes or retrieves the count matrix for the given peaks
        across all samples and creates a combined AnnData object.

        Args:
            peak_name (str): Name identifier for the peaks. Used to retrieve
                peaks if they were previously loaded. Default is 'genome_tiles'.
            peaks (str | pd.DataFrame | dict, optional): The peaks to get the count matrix for.
                Can be a file path, DataFrame, or dict mapping sample names to peaks.
                If None, will try to retrieve peaks using peak_name.

        Returns:
            AnnData: An AnnData object containing:
                - X: Sparse count matrix (all cells x peaks)
                - obs: Combined cell metadata from all samples
                - var: Peak information (if peaks are provided)

        Note:
            This method combines count matrices from all samples using vstack,
            similar to MultiSparseZarrIO.create_adata.
        """
        from scanpy import AnnData

        ds = []
        for data_key in tqdm(self.bpcells_dict):
            bp_io = self.bpcells_dict[data_key]
            # Handle dict of peaks (different peaks per sample)
            if isinstance(peaks, dict):
                sample_peaks = peaks.get(data_key, None)
            else:
                sample_peaks = peaks

            dcs = bp_io.get_count_matrix(peaks=sample_peaks, peak_name=peak_name)
            ds.append(dcs)

        ds = vstack(ds)

        if peak_name is not None:
            logging.info(
                "Loading peaks...assuming all samples have the same peaks with the given peak_name"
            )
            # Get peaks from first sample (assuming all samples have same peaks)
            first_bp_io = next(iter(self.bpcells_dict.values()))
            peaks_df = first_bp_io.get_peaks(peak_name)
            if peaks_df is not None:
                adata = AnnData(X=ds, obs=self.metadata, var=peaks_df)
            else:
                logging.warning(f"Could not load peaks for peak_name '{peak_name}'")
                adata = AnnData(X=ds, obs=self.metadata)
        else:
            adata = AnnData(X=ds, obs=self.metadata)

        return adata

    def get_count_matrix(
        self,
        peaks: str | pd.DataFrame | dict = None,
        peak_name="peaks",
        overwrite: bool = False,
    ):
        """
        Get count matrix for the given peaks across all samples.

        Args:
            peaks: Peaks as DataFrame, file path, or dict mapping sample names to peaks.
                If dict, each sample gets its own peaks. If None, will try to load
                peaks using peak_name.
            peak_name: Name identifier for the peaks. Used for storing/retrieving
                peaks. Default is 'peaks'.
            overwrite: Whether to overwrite existing saved peaks. Default is False.

        Returns:
            Sparse count matrix (all cells x peaks) stacked across all samples
        """
        ds = []
        for data_key in self.bpcells_dict:
            bp_io = self.bpcells_dict[data_key]
            # Handle dict of peaks (different peaks per sample)
            if isinstance(peaks, dict):
                sample_peaks = peaks.get(data_key, None)
            else:
                sample_peaks = peaks

            dcs = bp_io.get_count_matrix(
                peaks=sample_peaks, peak_name=peak_name, overwrite=overwrite
            )
            ds.append(dcs)
        ds = vstack(ds)
        return ds

    def write_peaks(self, peaks: str | pd.DataFrame | dict, name="peaks"):
        """
        Write peaks to BED files for each sample.

        Args:
            peaks: Peaks as DataFrame, file path, or dict mapping sample names to peaks.
                If dict, each sample gets its own peaks written.
            name: Name identifier for the peaks (used as filename: {name}.bed)

        Note:
            This saves peaks to {sample_path}/{name}.bed for each sample,
            similar to how MultiSparseZarrIO saves peaks to each zarr file.
        """
        if isinstance(peaks, dict):
            for data_key, peaks_df in peaks.items():
                if data_key not in self.bpcells_dict:
                    logging.warning(
                        f"Data key {data_key} not found in bpcells_dict, skipping writing peaks."
                    )
                    continue
                bp_io = self.bpcells_dict[data_key]
                bp_io.write_peaks(peaks_df, name)
        else:
            for data_key in self.bpcells_dict:
                bp_io = self.bpcells_dict[data_key]
                bp_io.write_peaks(peaks, name)
        return

    def get_peaks(self, name="peaks"):
        """
        Get peaks for each sample by calling get_peaks on each SingleSparseBPCellsIO.

        Args:
            name: Name identifier for the peaks (looks for {name}.bed in each sample directory).
                If name is "tiles", generates tiles on-the-fly (500bp) instead of loading from disk.

        Returns:
            dict: Dictionary mapping sample names to peak DataFrames
        """
        all_peaks = {}
        for data_key in self.bpcells_dict:
            bp_io = self.bpcells_dict[data_key]
            peaks_df = bp_io.get_peaks(name)
            if peaks_df is not None:
                all_peaks[data_key] = peaks_df
            else:
                logging.debug(f"No peaks '{name}' found for data key {data_key}.")
        if not all_peaks:
            logging.warning(f"No peaks found for name '{name}' across any data key.")
        return all_peaks


# =============================================================================
# Multiome BPCells Support
# =============================================================================


class MultiomeSparseBPCellsIO(SingleSparseBPCellsIO):
    """
    Multiome IO that extends SingleSparseBPCellsIO with RNA data and barcode mapping.

    This class inherits all ATAC functionality from SingleSparseBPCellsIO and adds:
    - RNA expression data (rna.h5ad)
    - Barcode mapping between ATAC and RNA (barcode_mapping.csv)
    - Methods to access paired data for shared cells

    Directory structure example:
    ```
    multiome_sparse_bpcells/
    ├── bpcells binary files     # ATAC-seq data (BPCells format)
    ├── attrs.json               # Metadata including multiome flags
    ├── metadata.csv             # Cell metadata
    ├── rna.h5ad                 # Full RNA expression data (all cells)
    └── barcode_mapping.csv      # Maps shared cells between ATAC and RNA
    ```

    This enables BPCells backend support for multiome data, complementing
    the existing MultiomeSparseZarrIO implementation.
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        peaks_name: str = None,
        chroms: list[str] | None = None,
    ):
        """Initialize multiome BPCells IO.

        Args:
            path: Path to multiome BPCells directory
            mode: Access mode (typically 'r' for multiome)
            peaks_name: Optional peaks file name to load
            chroms: Optional list of chromosomes to use
        """
        # Initialize as SingleSparseBPCellsIO first
        super().__init__(path, mode=mode, peaks_name=peaks_name, chroms=chroms)

        # Check if this is actually a multiome dataset
        if not self.attrs.get("multiome", False):
            raise ValueError(f"BPCells directory at {path} is not a multiome dataset")

        # Load barcode mapping
        self.mapping_path = Path(path) / "barcode_mapping.csv"
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Barcode mapping not found at {self.mapping_path}")
        self.mapping = pd.read_csv(self.mapping_path)

        # Set up RNA data path
        self.rna_path = Path(path) / "rna.h5ad"
        if not self.rna_path.exists():
            raise FileNotFoundError(f"RNA expression data not found at {self.rna_path}")

        # Lazy-load RNA data
        self._rna_data = None

        # Get multiome-specific attributes
        self.n_shared_cells = self.attrs.get("n_shared_cells", len(self.mapping))
        self.atac_n_cells = self.attrs.get("atac_n_cells", len(self.ids))
        self.rna_n_cells = self.attrs.get("rna_n_cells", -1)

        logging.info(
            f"Loaded multiome BPCells: {self.n_shared_cells} shared cells "
            f"(ATAC: {self.atac_n_cells}, RNA: {self.rna_n_cells})"
        )

    @property
    def rna_data(self):
        """Lazy-load RNA data."""
        if self._rna_data is None:
            import scanpy as sc

            self._rna_data = sc.read_h5ad(self.rna_path)
        return self._rna_data

    @property
    def shared_barcodes(self) -> list[str]:
        """Get list of shared barcodes."""
        return self.mapping["barcode"].tolist()

    @property
    def shared_indices_atac(self) -> np.ndarray:
        """Get ATAC indices for shared cells."""
        return self.mapping["atac_idx"].values

    @property
    def shared_indices_rna(self) -> np.ndarray:
        """Get RNA indices for shared cells."""
        return self.mapping["rna_idx"].values

    # MultiomeIO interface methods

    def get_n_shared_cells(self) -> int:
        """Return count of cells shared between modalities."""
        return self.n_shared_cells

    def get_shared_cell_identifiers(self) -> list[str]:
        """Return identifiers (barcodes) for shared cells."""
        return self.shared_barcodes

    def get_modality_indices(self, barcodes: list[str]) -> dict[str, np.ndarray]:
        """Map barcodes to indices in each modality.

        Args:
            barcodes: List of cell barcodes to map

        Returns:
            dict with keys 'atac' and 'rna' containing index arrays
        """
        # Create lookup from barcode to indices
        barcode_to_atac = dict(zip(self.mapping["barcode"], self.mapping["atac_idx"]))
        barcode_to_rna = dict(zip(self.mapping["barcode"], self.mapping["rna_idx"]))

        atac_indices = []
        rna_indices = []

        for bc in barcodes:
            if bc in barcode_to_atac:
                atac_indices.append(barcode_to_atac[bc])
                rna_indices.append(barcode_to_rna[bc])

        return {
            "atac": np.array(atac_indices, dtype=np.int32),
            "rna": np.array(rna_indices, dtype=np.int32),
        }

    def get_atac_for_shared_cells(
        self, chr_name: str, start: int, end: int
    ) -> np.ndarray:
        """Get ATAC data for shared cells only.

        Args:
            chr_name: Chromosome name
            start: Start position
            end: End position

        Returns:
            Coverage array for shared cells only - summed across cells (n_positions,)
        """
        from ..query import IndexQuery

        # Use IndexQuery to get data for shared cells only
        query = IndexQuery(self.shared_indices_atac)
        # Get coverage data using parent's method
        raw_coverage = self._get_coverage_for_query(
            chr_name, start, end, query=query, sparse=False
        )
        return raw_coverage

    def get_rna_for_shared_cells(
        self, gene_names: list[str] | None = None
    ) -> "np.ndarray | pd.DataFrame":
        """Get RNA expression for shared cells only.

        Args:
            gene_names: Optional list of genes to subset

        Returns:
            Expression matrix for shared cells (n_shared_cells x n_genes)
        """
        rna = self.rna_data

        # Subset to shared cells
        shared_rna = rna[self.shared_indices_rna, :]

        # Optionally subset genes
        if gene_names is not None:
            gene_mask = rna.var_names.isin(gene_names)
            shared_rna = shared_rna[:, gene_mask]

        return shared_rna.X

    def get_paired_track(
        self,
        chr_name: str,
        start: int,
        end: int,
        query=None,
        aggregator=None,
        gene_names: list[str] | None = None,
    ) -> dict:
        """Get paired ATAC and RNA data.

        Args:
            chr_name: Chromosome name
            start: Start position
            end: End position
            query: Query object for cell selection (applied to shared cells)
            aggregator: Aggregator for combining data
            gene_names: Optional list of genes to subset RNA data

        Returns:
            Dictionary with 'atac' and 'rna' keys containing respective data
        """
        # Get ATAC track for shared cells
        atac_data = self.get_atac_for_shared_cells(chr_name, start, end)

        # Get RNA data for shared cells
        rna_data = self.get_rna_for_shared_cells(gene_names=gene_names)

        return {
            "atac": atac_data,
            "rna": rna_data,
            "shared_barcodes": self.shared_barcodes,
            "n_shared_cells": self.n_shared_cells,
        }

    def get_gene_expression(
        self, gene_name: str, return_barcodes: bool = False
    ) -> np.ndarray | tuple[np.ndarray, list[str]]:
        """Get expression values for a specific gene across shared cells.

        Args:
            gene_name: Gene name to query
            return_barcodes: If True, also return barcode list

        Returns:
            Expression array for shared cells, optionally with barcodes
        """
        rna = self.rna_data

        if gene_name not in rna.var_names:
            raise ValueError(f"Gene '{gene_name}' not found in RNA data")

        gene_idx = rna.var_names.get_loc(gene_name)
        expression = rna[self.shared_indices_rna, gene_idx].X

        if hasattr(expression, "toarray"):
            expression = expression.toarray().flatten()
        else:
            expression = np.asarray(expression).flatten()

        if return_barcodes:
            return expression, self.shared_barcodes
        return expression

    def compute_atac_rna_correlation(
        self,
        chr_name: str,
        start: int,
        end: int,
        gene_name: str,
        method: str = "pearson",
    ) -> dict:
        """Compute correlation between ATAC signal and gene expression.

        Args:
            chr_name: Chromosome for ATAC region
            start: Start position
            end: End position
            gene_name: Gene to correlate with
            method: Correlation method ('pearson' or 'spearman')

        Returns:
            dict with correlation, p-value, and sample size
        """
        from scipy.stats import pearsonr, spearmanr

        # Get ATAC coverage per cell (not aggregated)
        from ..query import IndexQuery

        query = IndexQuery(self.shared_indices_atac)

        # Get per-cell coverage
        atac_per_cell = []
        for idx in self.shared_indices_atac:
            cell_query = IndexQuery([idx])
            coverage = self._get_coverage_for_query(
                chr_name, start, end, query=cell_query, sparse=False
            )
            atac_per_cell.append(np.sum(coverage))

        atac_values = np.array(atac_per_cell)

        # Get gene expression
        rna_values = self.get_gene_expression(gene_name)

        # Compute correlation
        if method == "pearson":
            corr, pval = pearsonr(atac_values, rna_values)
        elif method == "spearman":
            corr, pval = spearmanr(atac_values, rna_values)
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            "correlation": corr,
            "p_value": pval,
            "n_cells": len(atac_values),
            "method": method,
            "region": f"{chr_name}:{start}-{end}",
            "gene": gene_name,
        }

    def __repr__(self):
        return (
            f"MultiomeSparseBPCellsIO(path={self.path}, "
            f"n_shared_cells={self.n_shared_cells}, "
            f"atac_cells={self.atac_n_cells}, rna_cells={self.rna_n_cells})"
        )
