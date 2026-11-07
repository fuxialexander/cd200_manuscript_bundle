from __future__ import annotations

import concurrent.futures
import gc
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import pandas as pd
import pyranges as pr
import zarr
from intervaltree import IntervalTree
from scipy.sparse import csr_matrix, hstack, vstack
from tqdm import tqdm

from ..interface import MultiSparseIO, SingleSparseIO
from ..query import (
    BaseQuery,
    DefaultQuery,
    IndexQuery,
    KeyQuery,
    MetadataQuery,
    MultiKeyQuery,
)
from ..parallel_processing import process_chromosome_with_chunks
from .. import read_peaks  # read_peaks is defined before zarr_io import in io/__init__.py
from .base import COMPRESSOR, ZarrIO

# Type aliases
RawCoverageDict = dict[str, np.ndarray | csr_matrix]


class SingleSparseZarrIO(
    ZarrIO,
    SingleSparseIO,
):
    """
    A class for handling single-cell ATAC-seq data stored in Zarr format.

    Expected directory structure:
    ```
    sample.zarr/
    ├── .zgroup                      # Zarr group marker
    ├── .zattrs                      # Root attributes (assembly, data_key, chunk_size, etc.)
    ├── barcodes/                    # Cell barcode array
    │   ├── .zarray
    │   └── 0
    ├── labels/                      # Cell type/cluster labels
    │   ├── .zarray
    │   └── 0
    ├── per_cell_libsize/            # Per-cell library sizes
    │   ├── .zarray
    │   └── 0
    ├── chrs/                        # Chromosome data
    │   ├── chr1/
    │   │   ├── chunk_0/             # Fragment data chunks
    │   │   │   ├── .zarray
    │   │   │   └── 0, 1, 2...
    │   │   ├── chunk_0_end_mask/    # End masks for paired reads
    │   │   │   ├── .zarray
    │   │   │   └── 0
    │   │   └── chunk_1/, chunk_2/...
    │   └── chr2/, chr3/...
    ├── library_size/                # Per-chunk library sizes
    │   ├── chr1/
    │   │   ├── .zarray
    │   │   └── 0
    │   └── chr2/, chr3/...
    ├── tiles/                       # Genome tiles for clustering (optional)
    │   ├── .zgroup
    │   ├── regions/                 # Tile coordinates
    │   │   └── chr1/, chr2/...
    │   ├── count_sparse/            # Sparse count matrices
    │   │   └── chr1/, chr2/...
    │   └── count_sparse_chunk/      # Chunked sparse counts
    │       └── chr1/, chr2/...
    ├── genome_tiles.h5ad            # AnnData object for clustering (optional)
    └── [peak_name]/                 # Custom peak sets (optional)
        ├── .zgroup
        ├── regions/                 # Peak coordinates
        │   └── chr1/, chr2/...
        └── count_sparse/            # Peak count matrices
            └── chr1/, chr2/...
    ```
    Attributes stored in .zattrs:
    - assembly: Genome assembly (e.g., "hg38", "mm10")
    - data_key: Sample identifier
    - chunk_size: Fragment chunk size (default: 2000000)
    - multiome: Boolean flag for multiome data
    - has_rna: Boolean flag for RNA data presence
    - n_cells: Total number of cells
    - n_shared_cells: Number of cells with both ATAC and RNA (multiome only)
    - processing_date: Processing timestamp
    - clustering_completed: Boolean flag for clustering status
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        dtype: str = "float32",
        peaks_name: str = None,
        chroms: list[str] | None = None,
    ):
        super().__init__(path, mode, dtype, chroms)
        self.peaks_name = peaks_name
        if peaks_name is not None:
            self.peaks = self.get_peaks(peaks_name)
        if self.dataset is not None:
            self._init_data_structure()
        else:
            logging.error(
                f"Dataset at {path} could not be loaded. SingleSparseZarrIO initialization incomplete."
            )

    def _init_data_structure(self):
        """
        Initialize sparse data specific attributes and chunking.
        """
        super()._init_data_structure()
        self._init_chunking()

        # Load metadata
        self.data_key = self.dataset.attrs["data_key"]
        self.barcodes = self.dataset["barcodes"][:]

        try:
            labels = self.dataset["labels"][:]
        except KeyError:
            labels = np.repeat(self.data_key, len(self.barcodes))

        # Initialize metadata DataFrame
        df = pd.DataFrame(
            {
                "barcodes": self.barcodes,
                "labels": labels,
                "data_key": self.data_key,
                "per_cell_libsize": list(self.per_cell_libsize.values()),
            },
            index=self.barcodes,
        )
        df.index.name = "barcodes"
        df["cell_idx"] = np.arange(len(df))
        self.metadata = df

        # Set number of cells
        self.n_cells = self.dataset["chrs"]["chr1"]["chunk_0"].shape[0]
        # Get chunk size from attributes or default
        # Moved self.chunk_size access AFTER _init_chunking is called by ZarrIO.__init__
        # self.chunk_size = self.dataset.attrs.get("chunk_size", 2000000) # Redundant if _init_chunking works

    def _init_chunking(self, first_chrom: str | None = None) -> None:
        """Initialize chunking for sparse Zarr."""
        # super()._init_chunking(first_chrom)  # Call base class chunking init if needed
        # Chunk size is a bit different for sparse Zarr
        self.chunk_size = self._load_attr("chunk_size")
        logging.info(f"Sparse Zarr chunk size set to: {self.chunk_size}")

    def _calculate_libsize(self):
        """
        Calculate per-label total library size from metadata.
        """
        # Calculate from metadata
        libsize = (
            self.metadata.groupby("labels")
            .agg({"per_cell_libsize": "sum", "barcodes": "count"})
            .rename(columns={"per_cell_libsize": "libsize", "barcodes": "cell_counts"})
        )

        libsize["frag_per_cell"] = (libsize["libsize"] / libsize["cell_counts"]).astype(
            int
        )

        # Write to zarr attr if not exists
        if "libsize_data" not in self.dataset.attrs and self.mode == "a":
            zarr.open(self.path, mode="a").attrs["libsize_data"] = libsize.to_dict()

        return libsize

    def _calculate_per_cell_libsize(self):
        if "per_cell_libsize" in self.dataset:
            libsize_i = self.dataset["per_cell_libsize"][:].astype(int)
        else:
            # Initialize as numpy array of zeros with length matching number of barcodes
            libsize_i = np.zeros(len(self.barcodes), dtype=int)
            for f in glob(self.path + "/library_size/*/*"):
                libsize_i += zarr.load(f)
            # store to zarr
            if self.mode == "a":
                self.dataset.create_array(
                    "per_cell_libsize", data=libsize_i.astype("int32"), overwrite=True
                )
            libsize_i = libsize_i.astype(int)
        # convert to dict with barcodes as keys
        libsize_d = dict(zip(self.barcodes, libsize_i))
        return libsize_d

    def _get_all_ids(self):
        return self.barcodes

    def get_chunk(self, chr_name: str, chunk_idx: int) -> zarr.Array:
        """
        Get a chunk of the ATAC-seq data for a specific chromosome.

        Parameters:
        chr_name (str): The name of the chromosome.
        chunk_idx (int): The index of the chunk.

        Returns:
        zarr.core.Array: The ATAC-seq data for the specified chromosome chunk.
        """
        return self.dataset[f"chrs/{chr_name}/chunk_{chunk_idx}"]

    def _get_coverage_for_query(
        self, chr_name, start, end, query: BaseQuery, sparse=False
    ):
        """
        Retrieve coverage for a specific region, parsing the BaseQuery object.

        Args:
            chr_name: Chromosome name.
            start: Start coordinate (0-based).
            end: End coordinate (exclusive).
            query: A BaseQuery object specifying the data subset.
            sparse: Whether to return data in sparse format (csr_matrix).

        Returns:
            A dictionary mapping group identifiers (str) to their corresponding
            coverage data (np.ndarray or csr_matrix). Even for single queries,
            returns {'default': data}.
        """
        start_chunk = start // self.chunk_size
        end_chunk = (end - 1) // self.chunk_size

        resolved_query_arg: np.ndarray | slice | dict[str, np.ndarray | slice]
        is_multi_key = False
        coverage: np.ndarray | dict[str, np.ndarray]

        # 1. Resolve Query into indices or dict of indices
        if isinstance(query, MultiKeyQuery):
            is_multi_key = True
            resolved_indices_dict = {}
            logging.debug(f"Resolving MultiKeyQuery with {len(query.mapping)} groups.")
            for group_key, sub_query_spec in query.mapping.items():
                if isinstance(sub_query_spec, BaseQuery):
                    resolved_indices_dict[group_key] = self._resolve_query_to_indices(
                        sub_query_spec
                    )
                elif isinstance(sub_query_spec, list | tuple | np.ndarray | pd.Series):
                    # Attempt to treat as indices first, then keys
                    try:
                        indices = np.asarray(sub_query_spec, dtype=int)
                        if np.any(indices < 0):
                            raise ValueError("Negative indices")
                        # TODO: Add validation that indices are within [0, n_cells)
                        resolved_indices_dict[group_key] = indices
                        logging.debug(
                            f"Resolved group '{group_key}' using direct indices."
                        )
                    except (ValueError, TypeError):
                        keys = [str(k) for k in sub_query_spec]
                        resolved_indices_dict[group_key] = self.metadata[
                            self.metadata.index.isin(keys)
                        ]["cell_idx"].values
                        logging.debug(
                            f"Resolved group '{group_key}' using keys matching 'barcodes'."
                        )
                else:
                    raise TypeError(
                        f"Unsupported sub-query type '{type(sub_query_spec)}' in MultiKeyQuery for group '{group_key}'"
                    )
            resolved_query_arg = resolved_indices_dict
            coverage = {
                key: np.zeros(end - start, dtype=self.dtype)
                for key in resolved_indices_dict
            }
        else:  # Handle single queries (Default, Index, Key, Metadata)
            is_multi_key = False
            resolved_indices = self._resolve_query_to_indices(query)
            resolved_query_arg = resolved_indices
            coverage = np.zeros(end - start, dtype=self.dtype)

        # Internal helper function for chunk processing
        def _extract_chunk(
            chunk_idx, chr_name, resolved_indices_arg, start, end, get_chunk_func
        ):
            """Extracts data for a single chunk based on resolved indices."""
            try:
                chunk_data = get_chunk_func(chr_name, chunk_idx)
            except (KeyError, IndexError) as e:
                # Chunk doesn't exist (likely last chunk beyond chromosome end)
                logging.debug(f"Chunk {chunk_idx} not found for {chr_name}: {e}")
                if isinstance(resolved_indices_arg, dict):
                    return {key: np.array([], dtype=np.int64) for key in resolved_indices_arg}
                else:
                    return np.array([], dtype=np.int64)

            chunk_num_cells = chunk_data.shape[0]
            chunk_coverage_result: np.ndarray | dict[str, np.ndarray]
            if isinstance(resolved_indices_arg, dict):  # MultiKeyQuery case
                chunk_coverage_result = {}
                for key, cell_idx_array in resolved_indices_arg.items():
                    # cell_idx_array should be np.ndarray (or potentially slice)
                    if isinstance(
                        cell_idx_array, slice
                    ):  # e.g. slice(None) from DefaultQuery within MultiKey
                        selected_cells = chunk_data[cell_idx_array]
                    elif cell_idx_array is not None and cell_idx_array.size > 0:
                        # Ensure indices are integers for indexing
                        int_indices = cell_idx_array.astype(int)
                        # Validate indices are within chunk bounds
                        valid_mask = (int_indices >= 0) & (int_indices < chunk_num_cells)
                        if not np.all(valid_mask):
                            logging.debug(
                                f"Filtering {np.sum(~valid_mask)} out-of-bounds indices "
                                f"(chunk {chunk_idx} has {chunk_num_cells} cells)"
                            )
                            int_indices = int_indices[valid_mask]
                        if int_indices.size > 0:
                            selected_cells = chunk_data.oindex[int_indices, :]
                        else:
                            selected_cells = np.array([], dtype=chunk_data.dtype)
                    else:  # Empty selection for this key
                        selected_cells = np.array([], dtype=chunk_data.dtype)

                    # Flatten and filter coordinates
                    if selected_cells.size == 0:
                        region_data = np.array([], dtype=np.int64)  # Index type
                    else:
                        flattened = selected_cells.reshape(-1)
                        # Assuming 0 is padding/invalid in sparse Zarr, filter it out.
                        # If 0 is a valid coordinate, this needs adjustment.
                        flattened = flattened[flattened != 0]
                        # Filter coordinates within the requested [start, end) range
                        in_range_mask = np.logical_and(
                            flattened >= start, flattened < end
                        )
                        # Calculate relative indices within the output array
                        region_data = flattened[in_range_mask] - start
                    chunk_coverage_result[key] = region_data

            else:  # Single query case (slice or np.ndarray)
                resolved_indices = resolved_indices_arg
                if isinstance(
                    resolved_indices, slice
                ):  # Should only be slice(None) if DefaultQuery used arange
                    selected_cells = chunk_data.oindex[resolved_indices]
                elif resolved_indices is not None and resolved_indices.size > 0:
                    int_indices = resolved_indices.astype(int)
                    # Validate indices are within chunk bounds
                    valid_mask = (int_indices >= 0) & (int_indices < chunk_num_cells)
                    if not np.all(valid_mask):
                        logging.debug(
                            f"Filtering {np.sum(~valid_mask)} out-of-bounds indices "
                            f"(chunk {chunk_idx} has {chunk_num_cells} cells)"
                        )
                        int_indices = int_indices[valid_mask]
                    if int_indices.size > 0:
                        selected_cells = chunk_data.oindex[int_indices, :]
                    else:
                        selected_cells = np.array([], dtype=chunk_data.dtype)
                else:  # Empty selection
                    selected_cells = np.array([], dtype=chunk_data.dtype)

                if selected_cells.size == 0:
                    region_data = np.array([], dtype=np.int64)
                else:
                    flattened = selected_cells.reshape(-1)
                    flattened = flattened[flattened != 0]  # Assume 0 padding
                    in_range_mask = np.logical_and(flattened >= start, flattened < end)
                    region_data = flattened[in_range_mask] - start
                chunk_coverage_result = region_data
            return chunk_coverage_result

        # 2. Process Chunks in Parallel
        # Validate chunk range against available chunks
        try:
            available_chunks = list(self.dataset[f"chrs/{chr_name}"].keys())
            available_chunk_indices = [
                int(chunk_key.split("_")[1])
                for chunk_key in available_chunks
                if chunk_key.startswith("chunk_") and chunk_key.split("_")[1].isdigit()
            ]
            if available_chunk_indices:
                max_chunk_idx = max(available_chunk_indices)
                # Clip end_chunk to available chunks
                end_chunk = min(end_chunk, max_chunk_idx)
        except (KeyError, ValueError) as e:
            logging.debug(f"Could not validate chunk range for {chr_name}: {e}")
            # Continue with original range, _extract_chunk will handle missing chunks

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for chunk_idx in range(start_chunk, end_chunk + 1):
                futures.append(
                    executor.submit(
                        _extract_chunk,
                        chunk_idx,
                        chr_name,
                        resolved_query_arg,  # Pass resolved indices/dict
                        start,
                        end,
                        self.get_chunk,  # Pass method directly
                    )
                )

            # 3. Aggregate Results
            for future in concurrent.futures.as_completed(futures):
                chunk_result = future.result()  # dict or array of relative indices
                if is_multi_key:  # chunk_result is a dict
                    assert isinstance(coverage, dict)  # Type assertion for mypy
                    for key, region_data in chunk_result.items():
                        # region_data contains relative indices for this chunk
                        # Use np.add.at for efficient in-place updates
                        if region_data.size > 0:
                            # Ensure indices are valid (should be >= 0 from calculation)
                            valid_indices = region_data[region_data >= 0]
                            if valid_indices.size > 0:
                                np.add.at(coverage[key], valid_indices, 1)
                else:  # chunk_result is a single array
                    assert isinstance(coverage, np.ndarray)  # Type assertion
                    region_data = chunk_result
                    if region_data.size > 0:
                        valid_indices = region_data[region_data >= 0]
                        if valid_indices.size > 0:
                            np.add.at(coverage, valid_indices, 1)

        # 4. Final Formatting (Return RawCoverageDict)
        final_coverage_dict: dict[str, np.ndarray | csr_matrix] = {}
        if is_multi_key:
            assert isinstance(coverage, dict)
            for key, cov_array in coverage.items():
                final_coverage_dict[key] = (
                    csr_matrix(cov_array) if sparse else cov_array
                )
        else:
            assert isinstance(coverage, np.ndarray)
            final_coverage = csr_matrix(coverage) if sparse else coverage
            final_coverage_dict["default"] = final_coverage  # Always return dict
        return final_coverage_dict

    def get_count_for_region(
        self, cell_idx, chr_name: str, start: int, end: int, chunk_size: int = 2000000
    ):
        """
        Get the count of fragments for the specified region and cell indices.

        Parameters:
        cell_idx (np.ndarray): Indices of the selected cells.
        chr_name (str): The name of the chromosome.
        start (int): The start position of the region.
        end (int): The end position of the region.
        chunk_size (int): Size of the chunk.

        Returns:
        np.ndarray: The count of fragments for the specified region.
        """
        start_chunk = start // chunk_size
        end_chunk = end // chunk_size
        count = 0

        for chunk_idx in range(start_chunk, end_chunk + 1):
            chunk_data = self.get_chunk(chr_name, chunk_idx)
            selected_cells = chunk_data.oindex[cell_idx, :]
            flattened = selected_cells.reshape(-1)
            # Replace 0 with -1
            flattened[flattened == 0] = -1
            region_data = flattened[np.logical_and(flattened >= start, flattened < end)]
            count += len(region_data)

        return count

    def _compute_count_matrix_for_chr_per_chunk(
        self, chr_name, peaks_chunk, chunk_size, chunk_idx, dataset, peak_start_idx
    ):
        """Process a single chunk for peak counting"""
        chunk_idx = int(chunk_idx.split("_")[-1])
        chunk_data = self.get_chunk(chr_name, chunk_idx)
        n_cells = chunk_data.shape[0]

        # Create IntervalTree for efficient overlap detection
        peak_tree = IntervalTree.from_tuples(
            (row.Start, row.End, i) for i, row in peaks_chunk.iterrows()
        )

        peak_count = np.zeros((n_cells, len(peaks_chunk)), dtype=int)

        for cell_num, cell in enumerate(chunk_data):
            for element in cell:
                if element == 0:  # Skip padding zeros
                    break
                overlapping = peak_tree[element]
                for interval in overlapping:
                    peak_index = interval.data
                    peak_count[cell_num, peak_index - peak_start_idx] += 1

        return peak_count, peak_start_idx

    def _zarr_chunk_processor(
        self,
        chromosome: str,
        chunk_start: int,
        chunk_end_pos: int,
        chunk_peaks: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Zarr-specific chunk processor adapter for shared chromosome processing.

        This wraps _compute_count_matrix_for_chr_per_chunk to match the shared interface.
        """
        # Find the chunk index that corresponds to this genomic position
        chunk_idx_num = chunk_start // self.chunk_size
        chunk_idx = f"chunk_{chunk_idx_num}"

        # Check if this chunk exists in the dataset
        if chunk_idx not in self.dataset[f"chrs/{chromosome}"]:
            return None

        # Get peak start index (global position in peaks_chr)
        if chunk_peaks.empty:
            return None

        peak_start_idx = chunk_peaks.index[0]

        # Call the original chunk processor
        try:
            chunk_peak_count, _ = self._compute_count_matrix_for_chr_per_chunk(
                chromosome,
                chunk_peaks,
                self.chunk_size,
                chunk_idx,
                self.dataset,
                peak_start_idx,
            )
            # Return with original indices from chunk_peaks (use stored original_indices if available)
            if "original_indices" in chunk_peaks.columns:
                original_indices = chunk_peaks["original_indices"].values
            else:
                original_indices = chunk_peaks.index.values
            return chunk_peak_count, original_indices
        except Exception as e:
            logging.error(f"Error processing Zarr chunk {chunk_idx}: {e}")
            return None

    def _compute_count_matrix_for_chr(
        self,
        chr_name,
        peaks,
        peak_name=None,
        chunk_size: int = 2000000,
        write_to_zarr=False,
    ) -> csr_matrix:
        """
        Get the count matrix for the specified chromosome using shared processing logic.

        Parameters:
        chr_name (str): The name of the chromosome.
        peaks (pd.DataFrame): The peaks to get the count matrix for. Must contain 'Chromosome', 'Start', and 'End' columns.
        chunk_size (int): Size of the chunk.
        write_to_zarr (bool): Whether to write the count matrix to Zarr. Defaults to False.
        peak_name (str): The name of the peak for writing to Zarr. Required if write_to_zarr is True.

        Returns:
        scipy.sparse.csr_matrix: The count matrix for the specified chromosome.
        """
        peaks_chr = (
            peaks.query("Chromosome == @chr_name")
            .sort_values(["Start"])
            .reset_index(drop=True)
        )

        if peaks_chr.empty:
            logging.warning(f"No peaks for {chr_name}")
            return csr_matrix((self.n_cells, 0))

        logging.info(f"Processing {chr_name} with {len(peaks_chr)} peaks")

        # Get chunk starts from dataset (Zarr-specific)
        chunk_starts = []
        for chunk_idx in self.dataset[f"chrs/{chr_name}"]:
            if chunk_idx.endswith("_end_mask"):
                continue
            chunk_idx_num = int(chunk_idx.split("_")[-1])
            chunk_start = chunk_idx_num * chunk_size
            chunk_starts.append(chunk_start)

        # Use shared chromosome processing with sparse concat + reorder
        count_sparse, reordered_peaks = process_chromosome_with_chunks(
            chromosome=chr_name,
            peaks_chr=peaks_chr,
            process_chunk_fn=self._zarr_chunk_processor,
            chunk_size=chunk_size,
            chunk_starts=chunk_starts if chunk_starts else None,
            n_workers=1,  # Use ProcessPoolExecutor from shared code
            executor_class=ProcessPoolExecutor,
        )

        # Zarr-specific: Write to storage if requested
        if write_to_zarr:
            if peak_name is None:
                raise ValueError("Please provide peak_name to write to zarr")

            if count_sparse.sum() > 0:
                row, col = count_sparse.nonzero()
                data = count_sparse.data
                count_sparse_data = np.vstack((row, col, data)).T
                # create group if not exists
                if f"{peak_name}/count_sparse" not in self.dataset:
                    self.dataset.create_group(f"{peak_name}/count_sparse")
                self.dataset[f"{peak_name}/count_sparse"].create_array(
                    chr_name,
                    data=count_sparse_data.astype("u4"),
                    chunks=(10000, count_sparse_data.shape[1]),
                    compressors=COMPRESSOR,
                    overwrite=True,
                )
            else:
                logging.info(f"Warning: peak count for {chr_name} is all zeros")

        return count_sparse

    def _compute_count_matrix(
        self,
        peaks,
        peak_name=None,
        chunk_size: int = 2000000,
        write_to_zarr: bool = False,
    ) -> np.ndarray:
        """
        Get the count matrix for the specified peaks.

        Parameters:
        peaks (pd.DataFrame): The peaks to get the count matrix for. Must contain 'Chromosome', 'Start', and 'End' columns and need to be sorted.
        chunk_size (int): Size of the chunk.
        create_zarr_datasets (bool): Whether to create Zarr datasets. Defaults to True.

        Returns:
        np.ndarray: The count matrix for the specified peaks.
        """
        for chr_name in tqdm(peaks.Chromosome.unique()):
            if chr_name in self.dataset["chrs"]:
                self._compute_count_matrix_for_chr(
                    chr_name, peaks, peak_name, chunk_size, write_to_zarr
                )
        return

    def get_count_matrix(
        self,
        peaks: str | pd.DataFrame,
        peak_name="peak",
        chunk_size: int = 2000000,
        overwrite: bool = False,
    ) -> np.ndarray:
        """
        Get the count matrix for the specified peaks.

        This method computes or retrieves the count matrix for the given peaks across all cells.
        If the count matrix doesn't exist, it will be computed and stored in the Zarr file.

        Args:
            peaks (str|pd.DataFrame): The peaks to get the count matrix for. Can be a file path or a DataFrame.
                Must contain 'Chromosome', 'Start', and 'End' columns and need to be sorted.
            peak_name (str): Name of the peak set. Used for storing/retrieving data in Zarr. Default is 'peak'.
            chunk_size (int): Size of data chunks for processing. Default is 2000000.

        Returns:
            scipy.sparse.csr_matrix: The count matrix for the specified peaks.

        Note:
            This method can be computationally intensive for large datasets or many peaks.
        """

        if overwrite or peak_name not in self.dataset:
            logging.info(f"Computing count matrix for {peak_name}")
            peaks = read_peaks(peaks)
            self.write_peaks(peaks, peak_name)
            if overwrite or f"{peak_name}/count_sparse" not in self.dataset:
                logging.info(f"{peak_name}/count_sparse not found in {self.data_key}")
                self._compute_count_matrix(
                    peaks=peaks,
                    peak_name=peak_name,
                    chunk_size=chunk_size,
                    write_to_zarr=True,
                )
        ds = []
        chroms = self.dataset[f"{peak_name}/count_sparse"].keys()
        # version sort for numerical chromosomes, while put other like X, Y, M at the end
        chroms = sorted(
            chroms,
            key=lambda x: (
                int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 1000,
                x,
            ),
        )
        for chrom in tqdm(chroms):
            d = self.dataset[f"{peak_name}/count_sparse"][chrom][:]
            # read shape from self.barcodes and length of self.dataset[f"{peak_name}/regions"]
            # a is (N, [row, col, data]), convert to csr_matrix
            shape = (
                self.n_cells,
                self.dataset[f"{peak_name}/regions/{chrom}"].attrs["n_regions"],
            )
            d = csr_matrix((d[:, 2], (d[:, 0], d[:, 1])), shape=shape)
            ds.append(d)
        ds = hstack(ds)
        return ds

    def write_peaks(self, peaks: str | pd.DataFrame, name: str):
        """
        Write peaks to a zarr[f'{name}/regions/{chrom}'] using the internal helper.
        """
        self._write_peaks_internal(
            group=self.dataset,
            peaks=peaks,
            name=name,
            columns_to_include=["Start", "End"],
            overwrite=True,
            dtype="i4",
            compressors=COMPRESSOR,
        )

    def get_peaks(self, name="peaks"):
        """
        Get peaks from a zarr[f'{name}/regions/{chrom}'] using the internal helper.
        """
        # Use the internal static method
        peaks_df = self._get_peaks_internal(group=self.dataset, name=name)
        if peaks_df is None:
            logging.info(f"No peaks found for {name} in {self.data_key}")
        return peaks_df

    def remove_peaks(self, peak_name: str):
        """Remove peaks group f'{peak_name}' from the zarr file using the internal helper."""
        if self.mode == "r":
            logging.error("Cannot remove peaks: Zarr dataset is read-only.")
            return False
        # Use the internal static method
        return self._remove_peaks_internal(group=self.dataset, name=peak_name)

    def _load_attributes(self, chr, chunk_idx, attr):
        """Get stored zarr attributes for a given chromosome and chunk index"""
        return self.dataset[f"chrs/{chr}/chunk_{chunk_idx}"].attrs[attr]

    def get_frip(self, chr, chunk_idx):
        """Get FRiP for a given chromosome and chunk index"""
        return self._load_attributes(chr, chunk_idx, "FRiP")

    def get_libsize_for_chr_chunk(self, chr_name: str, chunk_idx: int):
        """Get library size for a given chromosome and chunk index from zarr"""
        return self.dataset[f"library_size/{chr_name}"][:, chunk_idx]

    def _resolve_query_to_indices(self, query: BaseQuery) -> np.ndarray | slice:
        """
        Resolve a query to cell indices.

        Args:
            query: A BaseQuery object specifying the data subset.

        Returns:
            np.ndarray or slice: The resolved indices for cell selection.
        """
        if isinstance(query, DefaultQuery):
            # Return all cells
            return slice(None)

        elif isinstance(query, IndexQuery):
            # Return the specified indices directly
            return query.indices

        elif isinstance(query, KeyQuery):
            # Find cells matching the specified barcodes
            mask = self.metadata.index.isin(query.keys)
            return self.metadata[mask]["cell_idx"].to_numpy()

        elif isinstance(query, MetadataQuery):
            # Use pandas query to filter cells
            filtered_df = self.metadata.query(query.query_string)
            return filtered_df["cell_idx"].to_numpy()

        else:
            raise TypeError(f"Unsupported query type: {type(query)}")


class MultiSparseZarrIO(ZarrIO, MultiSparseIO):
    """
    A class for handling and processing multi-sample sparse data stored in Zarr format.

    This class provides functionality to load, query, and manipulate multiple Zarr files
    containing sparse genomic data. It's designed to work with single-cell genomic data,
    offering methods for retrieving tracks, creating AnnData objects, and handling peaks.

    Expected directory structure:
    ```
    multi_sample_dir/
    ├── sample1.zarr/                # SingleSparseZarrIO structure
    │   ├── .zgroup
    │   ├── .zattrs
    │   ├── barcodes/
    │   ├── labels/
    │   ├── chrs/
    │   └── ...
    ├── sample2.zarr/                # SingleSparseZarrIO structure
    │   ├── .zgroup
    │   ├── .zattrs
    │   ├── barcodes/
    │   ├── labels/
    │   ├── chrs/
    │   └── ...
    └── sample3.zarr/                # SingleSparseZarrIO structure
        └── ...
    ```
    Each *.zarr subdirectory follows the SingleSparseZarrIO structure.
    The MultiSparseZarrIO class loads all zarr files matching the pattern
    (default: "*.zarr") and provides unified access across all samples.

    Attributes:
        zarr_dir (str): Path to the directory containing Zarr files.
        mode (str): Mode for opening Zarr files ('r' for read-only, 'w' for write).
        zarr_files (list): List of Zarr file paths.
        peaks_name (str): Name of the peaks dataset within the Zarr files.
        zarr_dict (dict): Dictionary mapping data keys to SingleSparseZarrIO objects.
        ids (list): List of data keys (identifiers) for the loaded Zarr files.
        chroms (list): List of chromosome names.
        chrom_n_chunks (dict): Dictionary mapping chromosome names to number of chunks.
        metadata (pd.DataFrame): Combined metadata from all loaded Zarr files.
        assembly (str): Genome assembly name.

    Methods:
        get_label_counts(threshold=200): Get label counts above a specified threshold.
        query_barcodes(query, min_cell_num=100): Query metadata by labels.
        get_track(cell_idx_dict, chr_name, start, end, chunk_size=2000000): Retrieve genomic tracks.
        get_track_obj(chr_name, start, end, ids=None, normalized=True): Get a Track object.
        get_track_for_regions(cell_idx_dict, chr_name, region_list, chunk_size=2000000): Get tracks for specific regions.
        create_adata(peak_name='peak', peaks=None): Create an AnnData object from the Zarr data.
        write_peaks(peaks, name='peaks'): Write peak information to Zarr files.
        get_peaks(name='peaks'): Retrieve peak information from Zarr files.
        get_merged_peaks(name='peaks_p0.01', data_keys=None): Get merged peaks across multiple samples.

    Properties:
        libsize: Total library size for each label.

    Usage:
        mz = MultiSparseZarrIO('/path/to/zarr/files')
        adata = mz.create_adata()
        tracks = mz.get_track_obj('chr1', 1000000, 2000000)
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        dtype: str = "float32",
        pattern: str = "*.zarr",
        quick_look: bool = False,
        peaks_name: str = None,
        chroms: list[str] | None = None,
    ):
        """
        Initialize a MultiSparseZarrIO object.

        Parameters:
        path (str): Path to the directory containing Zarr files.
        mode (str): Mode for opening Zarr files ('r' for read-only, 'a' for read/write). Default is 'r'.
        dtype (str): Data type for the genomic data. Default is 'float32'.
        pattern (str): Glob pattern for finding Zarr files. Default is "*.zarr".
        quick_look (bool): Whether to load only the first 3 samples. Default is False.
        peaks_name (str): Name of the peaks dataset to use. Default is None.
        """
        self.zarr_dir = path
        self.pattern = pattern
        self.quick_look = quick_look
        self.peaks_name = peaks_name
        super().__init__(path, mode, dtype, chroms)
        self._init_data_structure()

    @property
    def data_chroms(self) -> list[str]:
        """
        Abstract property: Returns a sorted list of chromosome names
        present in the underlying data source. Must be implemented by subclasses.
        """
        return self.first_zarr.data_chroms

    def _setup_dataset(self):
        """
        Setup the dataset.
        """
        self.zarr_files = glob(f"{self.zarr_dir}/{self.pattern}")
        if self.quick_look:
            self.zarr_files = self.zarr_files[0:3]
        # Load all zarr files into dictionary
        self.zarr_dict = self._load_all_zarr()
        # Set basic attributes from first zarr file
        self.first_zarr = next(iter(self.zarr_dict.values()))
        # use first zarr as proxy to set basic attributes (Note that this assumes all zarr files have the same assembly and chunk_size)
        self.dataset = self.first_zarr.dataset

    def _determine_assembly(self):
        """
        Determine the assembly from the first zarr file.
        """
        if not self.zarr_dict:
            self.assembly = None  # Or a default value
            logging.warning("Cannot determine assembly: zarr_dict is empty.")
            return

        first_assembly = self.first_zarr.assembly
        self.assembly = self.first_zarr.assembly
        # --- Verification Step ---
        for data_key, zarr_io in self.zarr_dict.items():
            if zarr_io.assembly != first_assembly:
                logging.error(
                    f"Inconsistent assembly detected! "
                    f"First sample '{next(iter(self.zarr_dict))}': '{first_assembly}', "
                    f"Sample '{data_key}': '{zarr_io.assembly}'."
                )
                # Decide on behavior: raise error or just warn and proceed with first?
                # Example: Raising an error
                raise ValueError(
                    "Inconsistent assemblies found across input Zarr files."
                )
                # Example: Warning and using first (already set self.assembly)
                # logging.warning(f"Using assembly '{first_assembly}' from the first sample.")
                break  # stop checking after first inconsistency

    def _init_data_structure(self):
        """
        Initialize data structure specific attributes.
        """
        super()._init_data_structure()
        self.ids = list(self.zarr_dict.keys())
        self.chunk_size = self.first_zarr.chunk_size
        # Initialize metadata
        dfs = []
        for data_key in self.zarr_dict:
            z = self.zarr_dict[data_key]
            dfs.append(z.metadata)
        # Concatenate metadata preserving barcode index
        self.metadata = pd.concat(dfs)
        # Reset cell_idx to be sequential across all samples
        self.metadata["cell_idx"] = np.arange(len(self.metadata))

        # Initialize peaks if name provided
        if self.peaks_name is not None:
            self.peaks = self.get_peaks(self.peaks_name)

    def _load_all_zarr(self) -> dict[str, SingleSparseZarrIO]:
        """
        Load all Zarr files.

        Returns:
        pd.DataFrame: A DataFrame containing the barcodes, labels, and data keys of all Zarr files.
        """
        dfs = []
        zarr_dict = {}
        if len(self.zarr_files) == 0:
            logging.warning("No Zarr files found in the directory.")
            return {}
        for zarr_file in tqdm(self.zarr_files):
            z = SingleSparseZarrIO(
                zarr_file, mode=self.mode, peaks_name=self.peaks_name
            )
            dfs.append(z.metadata)
            zarr_dict[z.data_key] = z
        self.assembly = z.assembly
        # Concatenate metadata preserving barcode index
        self.metadata = pd.concat(dfs)
        # Reset cell_idx to be sequential across all samples
        self.metadata["cell_idx"] = np.arange(len(self.metadata))
        return zarr_dict

    def _get_coverage_for_query(
        self,
        chr_name: str,
        start: int,
        end: int,
        query: BaseQuery,
        sparse: bool = False,
    ) -> RawCoverageDict:
        """
        Retrieve coverage data for a specific region and query across multiple samples.

        This method maps the abstract BaseQuery to the specific cell indices within
        each relevant underlying SingleSparseZarrIO instance and aggregates the results
        *within* each output group defined by the query.

        Args:
            chr_name: Chromosome name.
            start: Start coordinate (0-based).
            end: End coordinate (exclusive).
            query: A BaseQuery object specifying the data subset.
            sparse: Whether to return data in sparse format (if supported).

        Returns:
            A dictionary mapping group identifiers (str) to their corresponding
            coverage data (np.ndarray or csr_matrix).
        """
        # 1. Map the BaseQuery to a dictionary: {output_group_key: {data_key: local_indices}}
        query_map: dict[str, dict[str, np.ndarray]] = {}

        if isinstance(query, DefaultQuery):
            # Default: Treat each sample as a separate group
            query_map = {
                dk: {
                    dk: self.metadata[self.metadata["data_key"] == dk][
                        "cell_idx"
                    ].values
                }
                for dk in self.ids
            }
        elif isinstance(query, IndexQuery):
            # Indices refer to the global combined metadata
            target_meta = self.metadata.iloc[query.indices]
            grouped_indices = (
                target_meta.groupby("data_key")["cell_idx"].apply(np.array).to_dict()
            )
            # Result is a single group 'default' containing data from selected indices across samples
            query_map["default"] = grouped_indices
        elif isinstance(query, KeyQuery):
            # Assume keys refer to labels for MultiSparse context
            valid_keys = [
                k for k in query.keys if k in self.metadata["labels"].unique()
            ]
            if not valid_keys:
                logging.warning(
                    f"None of the keys {query.keys} found in data_keys {self.ids}."
                )
                return {}
            # Each valid key becomes an output group, selecting all cells for that sample
            query_map = {
                dk: {
                    dk: self.metadata[self.metadata["labels"] == dk]["cell_idx"].values
                }
                for dk in valid_keys
            }
        elif isinstance(query, MetadataQuery):
            # Query the combined metadata
            try:
                target_meta = self.metadata.query(query.query_string)
                grouped_indices = (
                    target_meta.groupby("data_key")["cell_idx"]
                    .apply(np.array)
                    .to_dict()
                )
                # Result is a single group 'default' containing data from selected cells
                query_map["default"] = grouped_indices
            except Exception as e:
                logging.error(f"Metadata query '{query.query_string}' failed: {e}")
                return {}
        elif isinstance(query, MultiKeyQuery):
            # Mapping is {output_group: keys/indices/subquery}
            for group_key, sub_query_spec in query.mapping.items():
                target_meta = pd.DataFrame()  # To collect metadata rows for this group
                grouped_indices = {}

                # Resolve the sub-query spec to target metadata rows
                if isinstance(sub_query_spec, IndexQuery):
                    target_meta = self.metadata.iloc[sub_query_spec.indices]
                elif isinstance(sub_query_spec, KeyQuery):
                    # Assume keys are data_keys (samples)
                    target_meta = self.metadata[
                        self.metadata["data_key"].isin(sub_query_spec.keys)
                    ]
                elif isinstance(sub_query_spec, MetadataQuery):
                    try:
                        target_meta = self.metadata.query(sub_query_spec.query_string)
                    except Exception as e:
                        logging.warning(f"Sub-query failed for group {group_key}: {e}")
                        target_meta = pd.DataFrame()  # Empty df
                elif isinstance(sub_query_spec, list | np.ndarray):  # Assume indices
                    try:
                        indices = np.asarray(sub_query_spec, dtype=int)
                        target_meta = self.metadata.iloc[indices]
                    except (ValueError, TypeError, IndexError):
                        logging.warning(
                            f"Could not interpret {sub_query_spec} as indices for group {group_key}"
                        )
                        target_meta = pd.DataFrame()  # Empty df
                else:
                    logging.warning(
                        f"Unsupported sub-query type {type(sub_query_spec)} in MultiKeyQuery for group {group_key}"
                    )
                    target_meta = pd.DataFrame()  # Empty df

                # Group the target metadata by data_key to get local indices
                if not target_meta.empty:
                    grouped_indices = (
                        target_meta.groupby("data_key")["cell_idx"]
                        .apply(np.array)
                        .to_dict()
                    )

                query_map[group_key] = grouped_indices

        # --- 2. Fetch and Aggregate Data *Within* Each Output Group ---
        final_coverage_dict: RawCoverageDict = {}

        # Iterate through the planned query structure
        for output_group_key, group_data_map in query_map.items():
            coverages_for_group = []  # Collect coverage arrays/matrices for this group

            for data_key, local_indices in group_data_map.items():
                if data_key not in self.zarr_dict:
                    logging.warning(
                        f"Data key {data_key} not found in zarr_dict for group {output_group_key}. Skipping."
                    )
                    continue
                if local_indices is None or len(local_indices) == 0:
                    logging.debug(
                        f"No cells selected for data key {data_key} in group {output_group_key}. Skipping retrieval."
                    )
                    continue  # Skip if no cells selected for this sample in this group

                z_io = self.zarr_dict[data_key]
                try:
                    # Fetch coverage for these specific indices within the single sample
                    # Use get_track with IndexQuery and SumAggregator to get the summed coverage for these cells
                    coverage_dict = z_io._get_coverage_for_query(
                        chr_name=chr_name,
                        start=start,
                        end=end,
                        query=IndexQuery(local_indices),  # Query by local indices
                        sparse=sparse,
                    )
                    if coverage_dict:
                        # Take the first value, assuming SumAggregator produced one output
                        coverage_data = next(iter(coverage_dict.values()))
                        if coverage_data is not None:
                            coverages_for_group.append(coverage_data)
                    else:
                        logging.debug(
                            f"No coverage data returned for {data_key} in group {output_group_key}"
                        )

                except Exception as e:
                    logging.error(
                        f"Error getting track for {data_key} in group {output_group_key}: {e}"
                    )

            # --- 3. Aggregate results across samples *for this output group* ---
            if not coverages_for_group:
                logging.warning(
                    f"No coverage data found for any sample in output group {output_group_key}."
                )
                # Store empty result for this group if needed, or skip
                # Let's store an empty array of the correct shape/type
                shape = (end - start,)
                empty_data = np.zeros(shape, dtype=self.dtype)
                final_coverage_dict[output_group_key] = (
                    csr_matrix(empty_data) if sparse else empty_data
                )
                continue

            # Aggregate the collected coverages (summation is standard)
            coverages_for_group[0]
            try:
                if sparse:
                    if all(isinstance(v, csr_matrix) for v in coverages_for_group):
                        stacked = vstack(coverages_for_group, format="csr")
                        # Keep final result as sparse matrix if requested sparse
                        final_coverage_dict[output_group_key] = csr_matrix(
                            stacked.sum(axis=0)
                        )  # Keep as sparse row matrix (1, N)
                    else:
                        # Attempt to convert non-sparse to sparse before summing? Or error?
                        # For now, error and return empty
                        logging.error(
                            f"Cannot sum sparse and non-sparse data for group {output_group_key}"
                        )
                        shape = (end - start,)
                        empty_data = np.zeros(shape, dtype=self.dtype)
                        final_coverage_dict[output_group_key] = csr_matrix(empty_data)
                else:  # Dense output requested
                    # Ensure all components are numpy arrays
                    if all(isinstance(v, np.ndarray) for v in coverages_for_group):
                        # Ensure all arrays have the same length before stacking
                        expected_len = end - start
                        valid_arrays = [
                            v for v in coverages_for_group if v.shape == (expected_len,)
                        ]
                        if not valid_arrays:
                            logging.warning(
                                f"No valid arrays to sum for group {output_group_key}"
                            )
                            final_coverage_dict[output_group_key] = np.zeros(
                                expected_len, dtype=self.dtype
                            )
                            continue
                        # Sum dense arrays
                        stacked_arrays = np.stack(valid_arrays, axis=0)
                        final_coverage_dict[output_group_key] = np.sum(
                            stacked_arrays, axis=0
                        )
                    else:
                        # This might happen if sparse=True was used internally but sparse=False requested here
                        # Convert sparse results to dense before summing
                        dense_arrays = []
                        expected_len = end - start
                        for v in coverages_for_group:
                            if isinstance(v, csr_matrix):
                                arr = (
                                    v.toarray().flatten()
                                )  # Convert sparse to dense 1D
                                if arr.shape == (expected_len,):
                                    dense_arrays.append(arr)
                            elif isinstance(v, np.ndarray) and v.shape == (
                                expected_len,
                            ):
                                dense_arrays.append(v)
                            else:
                                logging.warning(
                                    f"Skipping incompatible data type {type(v)} or shape in group {output_group_key}"
                                )

                        if not dense_arrays:
                            logging.warning(
                                f"No valid dense arrays to sum for group {output_group_key}"
                            )
                            final_coverage_dict[output_group_key] = np.zeros(
                                expected_len, dtype=self.dtype
                            )
                            continue

                        stacked_arrays = np.stack(dense_arrays, axis=0)
                        final_coverage_dict[output_group_key] = np.sum(
                            stacked_arrays, axis=0
                        )

            except Exception as agg_e:
                logging.error(
                    f"Error aggregating results for output group {output_group_key}: {agg_e}"
                )
                # Store empty result on error
                shape = (end - start,)
                empty_data = np.zeros(shape, dtype=self.dtype)
                final_coverage_dict[output_group_key] = (
                    csr_matrix(empty_data) if sparse else empty_data
                )

        return final_coverage_dict

    def create_adata(
        self,
        peak_name="tiles",
        peaks=None,
    ):
        """
        Create an AnnData object from the data stored in the Zarr dictionary.

        Returns:
            An AnnData object containing the data.
        """
        # import anndata as ad
        from scanpy import AnnData

        # if peak_name in self.dataset:
        #     adata = ad.read_zarr(self.zarr_dir + "/" + peak_name)
        #     return adata
        ds = []
        for data_key in tqdm(self.zarr_dict):
            z = self.zarr_dict[data_key]
            dcs = z.get_count_matrix(peaks=peaks, peak_name=peak_name)
            ds.append(dcs)
        ds = vstack(ds)
        if peak_name is not None:
            logging.info(
                "Loading peaks...assuming all samples have the same peaks with the given peak_name"
            )
            peaks = z.get_peaks(peak_name)
            adata = AnnData(X=ds, obs=self.metadata, var=peaks)
        else:
            adata = AnnData(X=ds, obs=self.metadata)
        return adata

    def get_count_matrix(
        self,
        peaks: str | pd.DataFrame | dict,
        peak_name="peaks",
        overwrite: bool = False,
    ):
        ds = []
        for data_key in self.zarr_dict:
            z = self.zarr_dict[data_key]
            dcs = z.get_count_matrix(
                peaks=peaks, peak_name=peak_name, overwrite=overwrite
            )
            ds.append(dcs)
        ds = vstack(ds)
        return ds

    def write_peaks(self, peaks: str | pd.DataFrame | dict, name="peaks"):
        if isinstance(peaks, dict):
            for data_key, peaks_df in peaks.items():
                if data_key not in self.zarr_dict:
                    logging.warning(
                        f"Data key {data_key} not found in zarr_dict, skipping writing peaks."
                    )
                    continue
                z = self.zarr_dict[data_key]
                z.write_peaks(peaks_df, name)
        else:
            for data_key in self.zarr_dict:
                z = self.zarr_dict[data_key]
                z.write_peaks(peaks, name)
        return

    def get_peaks(self, name="peaks"):
        """Get peaks for each data_key by calling get_peaks on each SingleSparseZarrIO."""
        all_peaks = {}
        for data_key in self.zarr_dict:
            z = self.zarr_dict[data_key]
            # Call the public get_peaks method of SingleSparseZarrIO
            peaks_df = z.get_peaks(name)
            if peaks_df is not None:
                all_peaks[data_key] = peaks_df
            else:
                logging.debug(f"No peaks '{name}' found for data key {data_key}.")
        if not all_peaks:
            logging.warning(f"No peaks found for name '{name}' across any data key.")
        return all_peaks

    def remove_peaks(self, name: str):
        """Remove peaks group f'{name}' for each data_key by calling remove_peaks on each SingleSparseZarrIO."""
        results = {}
        for data_key in self.zarr_dict:
            z = self.zarr_dict[data_key]
            # Call the public remove_peaks method of SingleSparseZarrIO
            results[data_key] = z.remove_peaks(name)
        return results  # Returns dict indicating success/failure per data_key

    def get_merged_peaks(self, name="peaks_p0.01", data_keys=None):
        if data_keys is None:
            data_keys = list(self.zarr_dict.keys())
        peaks = [pr.PyRanges(self.zarr_dict[k].get_peaks(name)) for k in data_keys]
        peaks = pr.concat(peaks).merge(slack=-100)
        return peaks

    def get_sample_stats(self) -> pd.DataFrame:
        """
        Get basic statistics for each sample/data key.

        Returns:
        pd.DataFrame: DataFrame containing per-sample statistics including:
            - Number of cells
            - Total library size
            - Mean reads per cell
            - Number of unique labels
            - Most common label
        """
        stats = []
        for data_key in self.zarr_dict:
            sample_cells = self.metadata[self.metadata.data_key == data_key]
            stats.append(
                {
                    "data_key": data_key,
                    "n_cells": len(sample_cells),
                    "total_libsize": sample_cells.per_cell_libsize.sum(),
                    "mean_reads_per_cell": sample_cells.per_cell_libsize.mean(),
                    "n_unique_labels": sample_cells.labels.nunique(),
                    "most_common_label": sample_cells.labels.mode().iloc[0],
                    "median_reads_per_cell": sample_cells.per_cell_libsize.median(),
                    "std_reads_per_cell": sample_cells.per_cell_libsize.std(),
                }
            )
        return pd.DataFrame(stats)

    def filter_cells(
        self,
        min_libsize: int = None,
        max_libsize: int = None,
        labels: list = None,
        min_cells_per_label: int = None,
    ) -> MultiSparseZarrIO:
        """
        Create a new MultiSparseZarrIO object with filtered cells.

        Parameters:
        min_libsize (int): Minimum library size (reads per cell)
        max_libsize (int): Maximum library size (reads per cell)
        labels (list): List of labels to keep
        min_cells_per_label (int): Minimum number of cells required for a label

        Returns:
        MultiSparseZarrIO: A new object containing only the filtered cells
        """
        mask = pd.Series(True, index=self.metadata.index)

        if min_libsize is not None:
            mask &= self.metadata.per_cell_libsize >= min_libsize

        if max_libsize is not None:
            mask &= self.metadata.per_cell_libsize <= max_libsize

        if labels is not None:
            mask &= self.metadata.labels.isin(labels)

        if min_cells_per_label is not None:
            valid_labels = self.get_label_counts(threshold=min_cells_per_label).index
            mask &= self.metadata.labels.isin(valid_labels)

        # Create new object with filtered cells
        filtered_metadata = self.metadata[mask].copy()
        filtered_data_keys = filtered_metadata.data_key.unique()

        # Create new object with only the filtered samples
        filtered_obj = self.subset(filtered_data_keys)
        return filtered_obj

    def summarize(self) -> None:
        """
        Print a comprehensive summary of the dataset.
        """
        print("MultiSparseZarrIO Dataset Summary")
        print("================================")
        print(f"Assembly: {self.assembly}")
        print(f"Number of samples: {len(self.zarr_dict)}")
        print(f"Total number of cells: {len(self.metadata)}")
        print(f"Number of unique labels: {self.metadata.labels.nunique()}")
        print("\nLabel counts:")
        print(self.get_label_counts().to_string())
        print("\nSample statistics:")
        print(self.get_sample_stats().to_string())
        print(f"\nChromosomes: {', '.join(self.chroms)}")
        if self.peaks_name:
            print(f"\nPeaks loaded: {self.peaks_name}")
