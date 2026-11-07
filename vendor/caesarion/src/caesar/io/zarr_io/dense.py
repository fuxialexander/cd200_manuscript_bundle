from __future__ import annotations

import concurrent.futures
import logging
from concurrent.futures import ProcessPoolExecutor
from os.path import basename
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyranges as pr
import zarr
from intervaltree import IntervalTree
from scipy.sparse import csr_matrix
from tqdm import tqdm

from ..interface import (
    BaseQuery,
    CelltypeDenseIO,
    DefaultQuery,
    DenseIO,
    IndexQuery,
    KeyQuery,
    MetadataQuery,
    MultiKeyQuery,
    SequenceDenseIO,
    SignalDenseIO,
)
from .base import COMPRESSOR, ZarrIO

if TYPE_CHECKING:
    from ..aggregator import RawCoverageDict


class DenseZarrIO(ZarrIO, DenseIO):
    """
    Handles dense genomic data stored in Zarr format, implementing the GenomicIO interface.

    Expected directory structure:
    ```
    /path/to/dense.zarr/
    ├── .zgroup                    # Zarr group metadata
    ├── .zattrs                    # Root attributes (assembly, etc.)
    └── chrs/                      # Chromosome data group
        ├── .zgroup
        ├── chr1/                  # Data for chromosome 1
        │   ├── .zarray           # Array metadata
        │   └── 0, 1, 2, ...      # Chunked data files
        ├── chr2/
        │   ├── .zarray
        │   └── 0, 1, 2, ...
        └── ...
    ```
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        dtype: str = "float32",
        chroms: list[str] | None = None,
    ):
        super().__init__(path, mode, dtype, chroms)
        if self.dataset is not None:
            self._init_data_structure()
        else:
            logging.error(
                f"Dataset at {path} could not be loaded. DenseZarrIO initialization incomplete."
            )
            # Set defaults for essential attributes if dataset loading failed
            self.feature_size = None
            self.chroms = []

    def _init_data_structure(self):
        """
        Initialize dense data specific attributes and chunking.
        Filters chrom_sizes based on actual data.
        """
        # call super to initialize the self.chrom_sizes from assembly
        super()._init_data_structure()
        # Call _init_chunking with a valid chromosome from the dataset
        if self.chroms:
            self._init_chunking(first_chrom=self.chroms[0])
        else:
            logging.error("No valid chromosomes found to initialize chunking.")
            return

        # Determine feature size (assuming consistent across chunks/chroms)
        if (
            self.chunk_size is not None and self.chroms
        ):  # Proceed only if chunking was successful
            try:
                first_chrom_ref = self.dataset["chrs"][self.chroms[0]]
                if self.dir_chunked:
                    # Directory-based chunking
                    shape = first_chrom_ref["chunk_0"].shape
                    self.feature_size = shape[1] if len(shape) == 2 else 1
                else:
                    # Internal chunking
                    shape = first_chrom_ref.shape
                    self.feature_size = shape[1] if len(shape) == 2 else 1
            except (KeyError, IndexError, AttributeError) as e:
                logging.warning(f"Could not determine feature size: {e}")
                self.feature_size = None
        else:
            self.feature_size = None

    def _get_coverage_for_query(
        self, chr_name: str, start: int, end: int, query: BaseQuery, sparse: bool
    ) -> RawCoverageDict:
        """
        Retrieve coverage for a dense Zarr region.
        Only handles DefaultQuery, others raise errors.
        Returns RawCoverageDict: {'default': data}.
        """
        if not isinstance(query, DefaultQuery):
            raise NotImplementedError(
                f"DenseZarrIO only supports DefaultQuery, not {type(query)}"
            )

        if self.dataset is None or "chrs" not in self.dataset:
            raise OSError("Zarr dataset or 'chrs' group not loaded.")

        if chr_name not in self.dataset["chrs"]:
            raise ValueError(f"Invalid chromosome: {chr_name}")

        # Adjust start/end to be within bounds (using chrom_sizes)
        data_sizes = self.chrom_sizes
        if chr_name not in data_sizes:
            logging.warning(
                f"Chromosome {chr_name} size unknown, proceeding without bounds check."
            )
            chrom_size = float("inf")  # Effectively no upper bound
        else:
            chrom_size = data_sizes[chr_name]

        start = max(0, start)
        end = min(chrom_size, end)

        if start >= end:
            logging.warning(
                f"Query region [{start}, {end}) on {chr_name} is empty after bounds check."
            )
            # Return empty data of the correct type/shape
            shape = (0,) if self.feature_size == 1 else (0, self.feature_size)
            empty_data = np.array([], dtype=self.dtype).reshape(shape)
            if sparse:
                empty_data = csr_matrix(empty_data)
            return {"default": empty_data}

        # --- Data Retrieval Logic ---
        coverage_full: np.ndarray | None = None
        # Check if data is loaded into memory (optional optimization)
        if hasattr(self, "data") and self.data and chr_name in self.data:
            try:
                coverage_full = self.data[chr_name][start:end]
            except Exception as e:
                logging.warning(
                    f"Error accessing in-memory data for {chr_name}: {e}. Reading from disk."
                )
                coverage_full = None  # Fallback to disk reading

        if coverage_full is None:
            # Read from disk
            try:
                chrom_ref = self.dataset["chrs"][chr_name]
                if not self.dir_chunked:
                    # New format - internal chunking
                    coverage_full = chrom_ref[start:end]
                else:
                    # Old format - directory-based chunking
                    if self.chunk_size is None or self.chunk_size <= 0:
                        raise ValueError("Invalid chunk size for directory chunking.")

                    chunk_start_idx = start // self.chunk_size
                    chunk_end_idx = (end - 1) // self.chunk_size

                    # Check chunk indices validity (more robustly)
                    num_chunks_in_chrom = len(chrom_ref)
                    if chunk_start_idx >= num_chunks_in_chrom or chunk_end_idx < 0:
                        raise IndexError(
                            f"Chunk indices [{chunk_start_idx}, {chunk_end_idx}] out of bounds for {chr_name} (0-{num_chunks_in_chrom - 1})"
                        )
                    chunk_start_idx = max(0, chunk_start_idx)
                    chunk_end_idx = min(num_chunks_in_chrom - 1, chunk_end_idx)

                    chunks_to_load = []
                    first_chunk_offset = start % self.chunk_size
                    last_chunk_len = (end - 1) % self.chunk_size + 1

                    for chunk_idx in range(chunk_start_idx, chunk_end_idx + 1):
                        chunk_data = chrom_ref[f"chunk_{chunk_idx}"][:]
                        if chunk_idx == chunk_start_idx and chunk_idx == chunk_end_idx:
                            # Single chunk case
                            chunk_data = chunk_data[first_chunk_offset:last_chunk_len]
                        elif chunk_idx == chunk_start_idx:
                            chunk_data = chunk_data[first_chunk_offset:]
                        elif chunk_idx == chunk_end_idx:
                            chunk_data = chunk_data[:last_chunk_len]
                        # Else: Middle chunks are taken entirely

                        chunks_to_load.append(chunk_data)

                    if not chunks_to_load:
                        # Should be caught by start >= end check, but as safety
                        shape = (
                            (0,)
                            if self.feature_size == 1
                            else (0, self.feature_size or 1)
                        )
                        coverage_full = np.array([], dtype=self.dtype).reshape(shape)
                    else:
                        coverage_full = np.concatenate(chunks_to_load, axis=0)

            except (OSError, KeyError, IndexError, ValueError, AttributeError) as e:
                logging.error(
                    f"Error reading Zarr data for {chr_name}[{start}:{end}]: {e}"
                )
                raise OSError(
                    f"Failed to read Zarr data for {chr_name}[{start}:{end}]"
                ) from e

        # Ensure correct dtype and sparsity
        try:
            result_data = coverage_full.astype(self.dtype)
            if sparse:
                result_data = csr_matrix(result_data)
        except AttributeError as e:
            # Handle case where coverage_full might be None or failed to load correctly
            logging.error(
                f"Coverage data for {chr_name}[{start}:{end}] is invalid: {e}"
            )
            shape = (0,) if self.feature_size == 1 else (0, self.feature_size or 1)
            result_data = np.array([], dtype=self.dtype).reshape(shape)
            if sparse:
                result_data = csr_matrix(result_data)

        return {"default": result_data}

    def load_to_memory_sparse(self):
        """Loads the entire dataset into memory as sparse matrices (per chromosome)."""
        if self.dataset is None:
            raise OSError("Cannot load to memory: Zarr dataset not loaded.")
        self.data = {}
        from scipy.sparse import vstack

        for chr_name in self.chroms:
            try:
                chrom_ref = self.dataset["chrs"][chr_name]
                if not self.dir_chunked:
                    self.data[chr_name] = csr_matrix(chrom_ref[:])
                else:
                    # concatenate chunks for directory chunking
                    track_chunks = []
                    for chunk_idx in range(len(chrom_ref)):
                        chunk_data = chrom_ref[f"chunk_{chunk_idx}"][:].astype(
                            self.dtype
                        )
                        track_chunks.append(csr_matrix(chunk_data))
                    if track_chunks:
                        self.data[chr_name] = vstack(track_chunks)
                    else:
                        self.data[chr_name] = csr_matrix(
                            (0, self.feature_size or 0), dtype=self.dtype
                        )  # Empty sparse matrix
            except (KeyError, IndexError, AttributeError) as e:
                logging.warning(
                    f"Failed to load chromosome {chr_name} to sparse memory: {e}"
                )

    def load_to_memory_dense(self):
        """Loads the entire dataset into memory as dense numpy arrays (per chromosome)."""
        if self.dataset is None:
            raise OSError("Cannot load to memory: Zarr dataset not loaded.")
        self.data = {}
        for chr_name in self.chroms:
            try:
                chrom_ref = self.dataset["chrs"][chr_name]
                if not self.dir_chunked:
                    self.data[chr_name] = chrom_ref[:].astype(self.dtype)
                else:
                    # concatenate chunks for directory chunking
                    track_chunks = []
                    for chunk_idx in range(len(chrom_ref)):
                        chunk_data = chrom_ref[f"chunk_{chunk_idx}"][:].astype(
                            self.dtype
                        )
                        track_chunks.append(chunk_data)
                    if track_chunks:
                        self.data[chr_name] = np.concatenate(track_chunks, axis=0)
                    else:
                        shape = (
                            (0,)
                            if self.feature_size == 1
                            else (0, self.feature_size or 1)
                        )
                        self.data[chr_name] = np.array([], dtype=self.dtype).reshape(
                            shape
                        )  # Empty dense array
            except (KeyError, IndexError, AttributeError) as e:
                logging.warning(
                    f"Failed to load chromosome {chr_name} to dense memory: {e}"
                )


class SignalDenseZarrIO(DenseZarrIO, SignalDenseIO):
    """
    A class for handling dense genomic data stored in Zarr format.

    This class provides functionality to load, query, and manipulate dense genomic data
    stored in Zarr files. It's designed to work with chromosome-wise data, offering
    methods for retrieving tracks, handling peaks, and processing transcription start
    site (TSS) data.

    Expected directory structure:
    ```
    /path/to/signal_dense.zarr/
    ├── .zgroup                    # Zarr group metadata
    ├── .zattrs                    # Root attributes (assembly, library_size, etc.)
    ├── chrs/                      # Chromosome data group
    │   ├── .zgroup
    │   ├── chr1/                  # Data for chromosome 1
    │   │   ├── .zarray           # Array metadata
    │   │   └── 0, 1, 2, ...      # Chunked data files
    │   ├── chr2/
    │   │   ├── .zarray
    │   │   └── 0, 1, 2, ...
    │   └── ...
    ├── peaks/                     # Optional peaks data
        ├── .zgroup
        └── regions/
            ├── chr1/
            │   ├── .zarray
            │   └── 0
            └── ...
    ```
    """

    def __init__(
        self, path: str, mode="r", dtype="float32", chroms: list[str] | None = None
    ):
        super().__init__(path, mode, dtype, chroms)

    def _calculate_libsize(self):
        """
        Calculate total library size for dense data.
        """
        if "library_size" in self.dataset.attrs:
            return self.dataset.attrs["library_size"]
        # old libsize attribute
        elif "libsize" in self.dataset.attrs:
            return self.dataset.attrs["libsize"]
        total_size = 0
        for chr_name in self.dataset["chrs"]:
            if not self.dir_chunked:
                total_size += self.dataset["chrs"][chr_name][:].sum()
            else:
                for chunk_idx in range(self.chrom_n_chunks[chr_name]):
                    total_size += self.dataset["chrs"][chr_name][f"chunk_{chunk_idx}"][
                        :
                    ].sum()
        # write libsize to attrs
        z = zarr.open(self.path, mode="a")
        z.attrs["library_size"] = total_size
        return total_size

    def write_peaks(self, peaks: str | pd.DataFrame, name="peaks", overwrite=False):
        """
        Write peaks to Zarr using the internal helper.
        Ensures path is f'{name}/regions/{chrom}'.
        """
        self._write_peaks_internal(
            group=self.dataset,
            peaks=peaks,
            name=name,
            columns_to_include=["Start", "End"],  # Standard columns
            overwrite=overwrite,
            dtype="i4",  # Standard dtype (was i8)
            compressors=COMPRESSOR,  # Standard compressor
        )

    def get_peaks(self, name="peaks") -> pd.DataFrame | None:
        """
        Get peaks from Zarr using the internal helper.
        Reads from f'{name}/regions/{chrom}'.
        """
        peaks_df = self._get_peaks_internal(group=self.dataset, name=name)
        if peaks_df is None:
            logging.info(f"No peaks found for {name} in {self.path}")
        return peaks_df

    def remove_peaks(self, name: str) -> bool:
        """
        Remove peaks group f'{name}' from the Zarr file using the internal helper.
        """
        if self.mode == "r":
            logging.error("Cannot remove peaks: Zarr dataset is read-only.")
            return False
        return self._remove_peaks_internal(group=self.dataset, name=name)


class MotifDenseZarrIO(DenseZarrIO):
    """
    A specialized class for handling motif data stored in Zarr format, inheriting from DenseZarrIO.

    This class adds functionality to precompute motif mode lengths and efficiently compute
    motif match indices using score quantiles and precomputed motif lengths.
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        dtype: str = "float32",
        chroms: list[str] | None = None,
    ):
        super().__init__(path, mode, dtype, chroms)

    def _init_data_structure(self):
        """
        Initialize motif data specific attributes and chunking.
        """
        super()._init_data_structure()
        # Get number of motifs from first chromosome's data
        self.n_motifs = self.get_track(
            self.chroms[0], 10000, 11000, output_format="raw_array"
        ).shape[1]

        # Initialize motif-specific attributes
        self._motif_mode_lengths = None
        self._motif_quantiles = None

    @property
    def motif_mode_lengths(self):
        """Get or compute motif mode lengths."""
        if self._motif_mode_lengths is None:
            if self.dataset.attrs.get("motif_mode_lengths", None) is not None:
                self._motif_mode_lengths = self.dataset.attrs["motif_mode_lengths"]
            else:
                self._motif_mode_lengths = self.precompute_motif_mode_length()
        return self._motif_mode_lengths

    @property
    def motif_quantiles(self):
        """Get or compute motif quantiles."""
        if self._motif_quantiles is None:
            if self.dataset.attrs.get("motif_quantiles", None) is not None:
                self._motif_quantiles = self.dataset["quantile_data"]
            else:
                self._motif_quantiles = self.compute_score_quantiles_and_match_indices(
                    quantiles=[0.99, 0.9]
                )
        return self._motif_quantiles

    def precompute_motif_mode_length_process_chunk(
        self, chunk_idx, min_length, chromosome, chunk_size, chrom_size
    ):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, chrom_size)
        motif_density_all = self.get_track(
            chromosome, start, end, output_format="raw_array"
        )
        local_lengths = [0] * self.n_motifs

        for motif_idx in range(self.n_motifs):
            motif_density = motif_density_all[:, motif_idx]
            matches = motif_density >= 5  # Threshold at 5 for detecting motif presence

            match_starts = np.where(np.diff(matches.astype(int)) == 1)[0] + 1
            match_ends = np.where(np.diff(matches.astype(int)) == -1)[0] + 1

            if len(match_starts) == 0 or len(match_ends) == 0:
                continue

            if match_ends[0] < match_starts[0]:
                match_ends = match_ends[1:]
            if len(match_starts) > len(match_ends):
                match_starts = match_starts[:-1]

            match_lengths = match_ends - match_starts
            match_lengths = match_lengths[match_lengths >= min_length]

            if len(match_lengths) > 0:
                mode_length = np.bincount(match_lengths).argmax()
                local_lengths[motif_idx] = max(local_lengths[motif_idx], mode_length)

        return local_lengths

    def precompute_motif_mode_length(
        self, chromosome="chr20", min_length=5, chunk_size=50000, n_processes=None
    ):
        """
        Precompute the mode length of motifs for a specific chromosome and save as a Zarr attribute.
        This takes around 3 minutes.

        Args:
            chromosome (str): The chromosome to sample for computing mode lengths.
            min_length (int): Minimum acceptable motif length.
            chunk_size (int): Size of chunks to process iteratively to avoid memory issues.
            n_processes (int): Number of processes to use for parallel computation.
        """
        import concurrent.futures
        from collections import Counter

        # Initialize list of lists to store all lengths for each motif
        all_motif_lengths = [[] for _ in range(self.n_motifs)]
        chrom_size = self.chrom_sizes[chromosome]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_processes
        ) as executor:
            futures = [
                executor.submit(
                    self.precompute_motif_mode_length_process_chunk,
                    chunk_idx,
                    min_length,
                    chromosome,
                    chunk_size,
                    chrom_size,
                )
                for chunk_idx in range(self.chrom_n_chunks[chromosome])
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Precomputing motif mode lengths",
                total=self.chrom_n_chunks[chromosome],
            ):
                chunk_lengths = future.result()
                # Append non-zero lengths from each chunk
                for i in range(self.n_motifs):
                    if chunk_lengths[i] > 0:
                        all_motif_lengths[i].append(chunk_lengths[i])

        # Calculate mode across all chunks for each motif
        motif_lengths = []
        for lengths in all_motif_lengths:
            if lengths:  # If we found any valid lengths
                counter = Counter(lengths)
                mode_length = counter.most_common(1)[0][0]
                motif_lengths.append(mode_length)
            else:  # If no valid lengths found
                motif_lengths.append(0)

        self.dataset.attrs["motif_mode_lengths"] = motif_lengths

    def compute_motif_match_indices_sparse_process_chunk(
        self, chunk_idx, chunk_size, chromosome, chrom_size, motif_mode_lengths
    ):
        """
        Process a single chromosome chunk to compute motif match indices efficiently.
        Args:
            chunk_idx (int): Index of the current chunk.
            chunk_size (int): Size of each chunk.
            chromosome (str): Chromosome being processed.
            chrom_size (int): Total size of the chromosome.
            motif_mode_lengths (list): Precomputed mode lengths for motifs.
        Returns:
            tuple: (chunk_idx, sparse_data), where sparse_data is a numpy array of shape (n_matches, 3).
        """
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, chrom_size)
        motif_density_all = self.get_track(
            chromosome, start, end, output_format="raw_array"
        )
        n_motifs = motif_density_all.shape[1]

        sparse_data = []

        # Process all motifs in parallel using vectorized operations
        for motif_idx in range(n_motifs):
            motif_density = motif_density_all[:, motif_idx]
            motif_length = motif_mode_lengths[motif_idx]

            if motif_length == 0:  # Skip motifs with zero-length mode
                continue

            # Identify matches
            matches = motif_density >= 5  # Threshold for detecting matches
            if not np.any(matches):  # Skip if no matches
                continue

            # Compute match starts and ends
            match_starts = np.where(np.diff(matches.astype(int)) == 1)[0] + 1
            match_ends = np.where(np.diff(matches.astype(int)) == -1)[0] + 1

            # Adjust boundary conditions
            if matches[0]:
                match_starts = np.insert(match_starts, 0, 0)
            if matches[-1]:
                match_ends = np.append(match_ends, matches.size)

            match_lengths = match_ends - match_starts

            # Filter matches by precomputed length
            valid_matches = match_lengths == motif_length
            if not np.any(valid_matches):
                continue

            valid_starts = match_starts[valid_matches]
            valid_scores = [
                np.max(motif_density[start:end])
                for start, end in zip(valid_starts, match_ends[valid_matches])
            ]

            # Append matches to sparse_data
            sparse_data.extend(
                [
                    [motif_idx, start + pos, score]
                    for pos, score in zip(valid_starts, valid_scores)
                ]
            )

        return chunk_idx, np.array(sparse_data)

    def compute_motif_match_indices_sparse(
        self,
        save_to_zarr=True,
        zarr_group_name="motif_match_data",
        chunk_size=50000,
        n_processes=None,
    ):
        """
        Compute motif match indices and store in sparse format.

        Args:
            save_to_zarr (bool): Whether to save results to a Zarr dataset.
            zarr_group_name (str): Name of the Zarr group for storing the results.
            chunk_size (int): Size of chunks for processing.
            n_processes (int): Number of processes for parallel computation.

        Returns:
            str: Path to the saved Zarr group if `save_to_zarr` is True, otherwise None.
        """

        motif_mode_lengths = self.dataset.attrs.get("motif_mode_lengths", None)
        if motif_mode_lengths is None:
            raise ValueError("Motif mode lengths have not been precomputed.")

        zarr_group = None
        if save_to_zarr:
            zarr_group = self.dataset.require_group(zarr_group_name, overwrite=True)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_processes
        ) as executor:
            for chrom in self.chrom_sizes:
                chrom_size = self.chrom_sizes[chrom]
                chrom_n_chunks = (chrom_size + chunk_size - 1) // chunk_size

                if save_to_zarr:
                    chrom_group = zarr_group.require_group(chrom, overwrite=True)

                futures = [
                    executor.submit(
                        self.compute_motif_match_indices_sparse_process_chunk,
                        chunk_idx,
                        chunk_size,
                        chrom,
                        chrom_size,
                        motif_mode_lengths,
                    )
                    for chunk_idx in range(chrom_n_chunks)
                ]

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    desc="Computing motif match indices",
                    total=len(futures),
                ):
                    chunk_idx, sparse_data = future.result()

                    if save_to_zarr:
                        # Save each chunk separately, e.g., chrs/chr1/chunk_1
                        chrom_group.create_array(
                            f"chunk_{chunk_idx}",
                            data=sparse_data,
                            shape=sparse_data.shape,
                        )

        # set chunk_size attribute
        zarr_group.attrs["chunk_size"] = chunk_size
        return zarr_group_name if save_to_zarr else None

    def get_chunk_motif_match_indices(
        self,
        chrom,
        chunk_idx,
        zarr_group_name="motif_match_data",
        cutoff=5,
        motif_idx=None,
    ):
        arr = self.dataset[f"{zarr_group_name}/{chrom}/chunk_{chunk_idx}"][:]
        if arr.shape[0] == 0:
            return np.array([])
        arr = arr[arr[:, 2] >= cutoff]
        if motif_idx is not None:
            arr = arr[arr[:, 0] == motif_idx]
        return arr

    def get_motif_match_indices(
        self,
        chrom,
        start,
        end,
        zarr_group_name="motif_match_data",
        cutoff=5,
        motif_idx=None,
        regions: pd.DataFrame | None = None,
    ):
        chunk_size = self.dataset[zarr_group_name].attrs["chunk_size"]
        start_chunk_idx = start // chunk_size
        end_chunk_idx = end // chunk_size
        arr = []
        if regions is not None:
            regions = regions[regions["Chromosome"] == chrom].query(
                f"Start > {start} and End < {end}"
            )
            region_trees = IntervalTree.from_tuples(
                regions[["Start", "End"]].values.tolist()
            )

        # Include end_chunk_idx in the range to handle small regions correctly
        for chunk_idx in range(start_chunk_idx, end_chunk_idx + 1):
            res = self.get_chunk_motif_match_indices(
                chrom, chunk_idx, motif_idx=motif_idx, cutoff=cutoff
            )
            if res.shape[0] > 0:
                arr.append(res)

        if len(arr) > 0:
            arr = np.concatenate(arr)
            if regions is not None:
                arr_bool = np.vectorize(lambda x: region_trees.overlaps(x))(arr[:, 1])
                arr = arr[arr_bool]
            # remove positions that are out of range
            arr = arr[arr[:, 1] >= start]
            arr = arr[arr[:, 1] < end]
            return arr
        return np.array([])


class SequenceDenseZarrIO(DenseZarrIO, SequenceDenseIO):
    """DenseZarrIO of genomic sequences, additionaly support GC content computation,
    default to int8."""

    def __init__(
        self, path: str, mode="r", dtype="int8", chroms: list[str] | None = None
    ):
        super().__init__(path, mode=mode, dtype=dtype, chroms=chroms)


class CelltypeDenseZarrIO(
    ZarrIO,
    CelltypeDenseIO,
):
    """
    A class for handling dense genomic data for multiple cell types stored in Zarr format.

    This class provides functionality to load, query, and manipulate dense genomic data
    for multiple cell types stored in a single Zarr file. It offers methods for retrieving
    tracks, handling peaks, processing transcription start site (TSS) data, and performing
    various data operations across cell types.

    Expected directory structure:
    ```
    /path/to/celltype_dense.zarr/
    ├── .zgroup                    # Zarr group metadata
    ├── .zattrs                    # Root attributes (assembly, celltypes, etc.)
    ├── cluster_0/                 # Data for first cell type/cluster
    │   ├── .zgroup
    │   ├── .zattrs               # Cell type attributes (library_size, n_cells, etc.)
    │   ├── chrs/                 # Chromosome data
    │   │   ├── .zgroup
    │   │   ├── chr1/
    │   │   │   ├── .zarray
    │   │   │   └── 0, 1, 2, ... # Chunked coverage data
    │   │   └── ...
    │   ├── cluster_peaks/        # Called peaks for this cell type
    │       ├── .zgroup
    │       └── regions/
    │           ├── chr1/
    │           │   ├── .zarray
    │           │   └── 0
    │           └── ...
    ├── cluster_1/                # Data for second cell type
    │   ├── .zgroup
    │   ├── .zattrs
    │   ├── chrs/
    │   ├── cluster_peaks/
    └── ...                       # Additional cell types
    ```

    Attributes:
        data_key (str): Identifier for the dataset, derived from the Zarr file name.
        path (str): Path to the Zarr file.
        mode (str): Mode for opening Zarr files ('r' for read-only, 'a' for read/write).
        dataset (zarr.Group): The loaded Zarr dataset.
        ids (list): List of cell type identifiers in the dataset.
        assembly (str): Genome assembly name (e.g., 'hg38').
        celltypes (list): List of cell type names.
        n_samples (list): Number of cells or samples for each cell type.
        n_celltypes (int): Total number of cell types.
        chunk_size (int): Size of data chunks.
        chrom_n_chunks (dict): Dictionary mapping chromosome names to number of chunks.
        chroms (list): List of chromosome names in the dataset.

    Properties:
        libsize (dict): Library size for each cell type.
        metadata (pd.DataFrame): Metadata for all cell types in the dataset.

    Methods:
        non_redundant_celltypes(method='max_depth'): Get non-redundant cell types based on specified method.
        filter_by_min_depth(method="depth_2048"): Filter cell types by minimum depth.
        rename_peaks(old_name, new_name): Rename peaks in the Zarr file.
        subset(celltype_ids): Create a subset of the dataset with specified cell types.
        subset_chromosomes(chroms): Create a subset of the dataset with specified chromosomes.
        leave_out_chromosomes(chroms, inverse=False): Create a subset excluding specified chromosomes.
        leave_out_celltypes(celltype_ids, inverse=False): Create a subset excluding specified cell types.
        leave_out_celltypes_with_pattern(pattern, inverse=False): Create a subset based on cell type ID patterns.
        subset_celltypes_with_data_name(data_name): Get cell types with specific data name.
        get_track(celltype_id, chr_name, start, end, sparse=False): Retrieve a genomic track for a specific region and cell type.
        get_track_obj(chr_name, start, end, ids=None, normalized=True): Get a Track object for multiple cell types.
        sample_track(chr_name, start, end): Get a random track from a random cell type.
        write_peaks(id, peaks, name='peaks', columns_to_include=['Start', 'End'], overwrite=False): Write peak information to the Zarr file.
        compute_tss_track(gene_id, gencode): Compute TSS track for a specific gene across all cell types.
        write_tf_tss_tracks_parallel(tf_list, gencode, overwrite=False, n_jobs=64): Write TSS tracks for multiple transcription factors in parallel.
        get_peaks(id, name='peaks_p0.01', chrom=None): Retrieve peak information for a specific cell type.
        get_peak_counts(celltype_id, name='peaks_p0.01', chrom=None): Count insertions in peaks for a given cell type and chromosome.
        call_peaks(track_id='agg', min_length=200, merge_length=0, blacklist=None): Call peaks across the genome for a specific track.
        quantile_normalization_peak_based(peak_file, blacklist_file): Perform quantile normalization on the track data for all cell types.

    Usage:
        cz = CelltypeDenseZarrIO('/path/to/zarr/file.zarr')
        track = cz.get_track('celltype1', 'chr1', 1000000, 2000000)
        cz.write_peaks('celltype1', peaks_df, name='my_peaks')
    """

    def __init__(self, path: str, mode="r", celltype_ids=None, chroms=None):
        """
        Initialize a CelltypeDenseZarrIO object. Reads root attributes like 'assembly'
        and 'cluster_ids' (or similar) if available.

        Args:
            path (str): Path to the Zarr file.
            mode (str): Mode for opening Zarr files ('r' for read-only, 'a' for read/write). Default is 'r'.
            celltype_ids (list, optional): List of cell type IDs to include. If None, all cell types are included.
            chroms (list, optional): List of chromosomes to include. If None, all chromosomes are included.
        """
        # Store initialization parameters first
        self._init_celltype_ids = celltype_ids
        # Call parent class __init__ which loads dataset and calls _init_data_structure
        super().__init__(
            path, mode, dtype="float32", chroms=chroms
        )  # DenseZarrIO __init__ calls _init_data_structure
        self._init_data_structure()

    def __repr__(self):
        libsize = pd.DataFrame(self.libsize.items(), columns=["celltype", "libsize"])
        return f"""
<CelltypeDenseZarrIO: {self.data_key}; mode: {self.mode}>
Number of celltypes: {self.n_celltypes}
Assembly: {self.assembly}
Chromosomes: {",".join(self.chroms)}
Libsize:\n{libsize}
"""

    @property
    def data_chroms(self) -> list[str]:
        """
        Abstract property: Returns a sorted list of chromosome names
        present in the underlying data source. Must be implemented by subclasses.
        """
        ids = list(self.dataset.keys())
        try:
            return list(self.dataset[ids[0]]["chrs"].keys())
        except KeyError:
            logging.error("'chrs' group not found in Zarr dataset.")
            return []

    def _init_ids(self):
        """Initialize celltype IDs from root attributes or top-level keys."""
        # Try finding a list of celltype/cluster IDs in root attrs
        # Common names might be 'cluster_ids', 'celltype_ids', 'ids', 'keys' etc.
        # Add more potential keys if needed for your datasets.
        potential_id_keys = [
            "cluster_ids",
            "celltype_ids",
            "ids",
            "keys",
            "datakey",
            "data_key",
        ]
        all_available_ids = None
        root_attrs = self.dataset.attrs
        for key in potential_id_keys:
            if key in root_attrs:
                ids_from_attr = root_attrs[key]
                if isinstance(ids_from_attr, list | tuple | np.ndarray):
                    all_available_ids = list(ids_from_attr)
                    logging.info(f"Read cell type IDs from root attribute '{key}'.")
                    break
        # Fallback: Get IDs from top-level keys if not found in attributes
        if all_available_ids is None:
            all_available_ids = list(self.dataset.keys())
            logging.info("Reading cell type IDs from top-level Zarr keys.")

        # --- Filter IDs based on init args ---
        if self._init_celltype_ids is not None:
            # Ensure IDs from init args are strings for comparison
            init_ids_str = [str(id) for id in self._init_celltype_ids]
            self.ids = [id for id in all_available_ids if str(id) in init_ids_str]
            invalid_ids = [
                id for id in init_ids_str if str(id) not in all_available_ids
            ]
            if invalid_ids:
                logging.warning(
                    f"Provided celltype IDs not found in dataset: {invalid_ids}"
                )
        else:
            self.ids = all_available_ids

        if not self.ids:
            raise ValueError("No valid cell types found or specified.")

    def _init_data_structure(self):
        """Initialize data structure specific attributes for CelltypeDense.
        Reads assembly and celltype list from root .zattrs if possible.
        """
        super()._init_data_structure()
        self.data_key = basename(self.path)
        if self.dataset is None:
            raise ValueError(
                "Cannot initialize CelltypeDenseZarrIO data structure: Zarr dataset not loaded."
            )

        self._init_ids()
        self.n_celltypes = len(self.ids)
        self.first_zarr = self.dataset[self.ids[0]]

        self._init_chunking()

    def _init_chunking(self, first_chrom=None):
        """Initialize chunking-related attributes."""
        return super()._init_chunking(first_chrom, self.first_zarr)

    def _calculate_libsize(self):
        """Calculate library sizes if not already present in attributes."""
        try:
            if "library_size" in self.first_zarr.attrs:
                return {id: self.dataset[id].attrs["library_size"] for id in self.ids}

            # old libsize attribute
            elif "libsize" in self.first_zarr.attrs:
                return {id: self.dataset[id].attrs["libsize"] for id in self.ids}
        except KeyError:
            pass

        # Calculate library sizes in parallel
        with ProcessPoolExecutor() as executor:
            args = [(id, self.path) for id in self.ids]
            results = tqdm(executor.map(self.calculate_libsize_for_id, args))

        # Store results
        _libsize = {}
        with zarr.open_group(self.path, mode="a") as root:
            for id, libsize_id in results:
                root[id].attrs["library_size"] = libsize_id
                _libsize[id] = libsize_id

        return _libsize

    def calculate_libsize_for_id(self, args):
        id, root_path = args
        root = zarr.open_group(root_path, mode="a")
        libsize_id = 0
        for chrom in root[id]["chrs"]:
            if self.dir_chunked:
                for chunk in root[id]["chrs"][chrom]:
                    libsize_id += root[id]["chrs"][chrom][chunk][:].sum()
            else:
                libsize_id += root[id]["chrs"][chrom][:].sum()
        return id, libsize_id

    def load_to_memory_dense(self):
        """Load the dense data to memory as nested dicts of numpy arrays.

        This replaces self.dataset with a plain nested dict structure that
        supports the same access patterns as zarr groups:
        - self.dataset[id]["chrs"][chr_name][start:end] for slicing
        - self.dataset[id]["chrs"][chr_name][f"chunk_{i}"] for dir_chunked

        Note:
            After calling this method, self.dataset will be a dict rather than
            a zarr.Group. This is intentional for zarr v3 compatibility and
            improved memory efficiency. Code should not rely on zarr-specific
            methods (like .attrs) after loading to memory.
        """
        if self.dataset is None:
            raise OSError("Cannot load to memory: Zarr dataset not loaded.")

        data = {}
        for id in self.ids:
            data[id] = {"chrs": {}}
            for chr_name in self.chroms:
                try:
                    chrom_ref = self.dataset[id]["chrs"][chr_name]
                    if not self.dir_chunked:
                        # Load full array - numpy slicing works identically
                        data[id]["chrs"][chr_name] = chrom_ref[:]
                    else:
                        # Load chunks into dict for dir_chunked format
                        data[id]["chrs"][chr_name] = {
                            f"chunk_{i}": chrom_ref[f"chunk_{i}"][:]
                            for i in range(len(chrom_ref))
                        }
                except (KeyError, IndexError, AttributeError) as e:
                    logging.warning(f"Failed to load {id}/{chr_name} to memory: {e}")

        self.dataset = data

    def subset(self, celltype_ids):
        """Creates a new CelltypeDenseZarrIO instance with a subset of cell types."""
        if not isinstance(celltype_ids, list):
            celltype_ids = list(celltype_ids)
        # Filter against currently available IDs in *this* instance
        valid_subset_ids = [id for id in celltype_ids if id in self.ids]
        if not valid_subset_ids:
            logging.error("Subset resulted in no valid cell types.")
            # Return an empty/non-functional instance or raise error?
            # Let's return a new instance with the empty list.
        elif len(valid_subset_ids) < len(celltype_ids):
            logging.warning(
                "Some requested subset IDs were not found in the current object."
            )

        # Pass the *filtered* list of IDs and current chroms
        return CelltypeDenseZarrIO(
            self.path, mode=self.mode, celltype_ids=valid_subset_ids, chroms=self.chroms
        )

    def _get_coverage_for_id(self, id, chr_name, start, end, sparse):
        if not self.dir_chunked:
            # New format - internal chunking
            coverage_full = self.dataset[id]["chrs"][chr_name][start:end]
        else:
            # Old format - directory-based chunking
            chunk_start = start // self.chunk_size
            chunk_end = (end - 1) // self.chunk_size
            coverage_full = []
            for chunk_idx in range(chunk_start, chunk_end + 1):
                chunk = self.dataset[id]["chrs"][chr_name][f"chunk_{chunk_idx}"]
                coverage_full.append(chunk)
            coverage_full = np.concatenate(coverage_full)
            query_size = end - start
            query_start = start % self.chunk_size
            coverage_full = coverage_full[query_start : query_start + query_size]

        if sparse:
            coverage_full = csr_matrix(coverage_full)
        return coverage_full

    def _get_coverage_for_query(
        self, chr_name: str, start: int, end: int, query: BaseQuery, sparse: bool
    ) -> RawCoverageDict:
        """
        Retrieve coverage data based on the query object. Returns a dictionary
        mapping celltype IDs to their coverage data.
        """
        results: RawCoverageDict = {}

        if isinstance(query, DefaultQuery):
            target_ids = self.ids  # Get all configured IDs for this object
            for cell_id in target_ids:
                try:
                    coverage_full = self._get_coverage_for_id(
                        cell_id, chr_name, start, end, sparse
                    )
                    results[cell_id] = coverage_full
                except Exception as e:
                    logging.warning(
                        f"Error fetching coverage for default ID {cell_id}: {e}"
                    )

        elif isinstance(query, KeyQuery):
            # Use only the keys that are valid for this object
            target_ids = [key for key in query.keys if key in self.ids]
            invalid_keys = [key for key in query.keys if key not in self.ids]
            if invalid_keys:
                logging.warning(
                    f"Requested keys not available in this dataset: {invalid_keys}"
                )
            for cell_id in target_ids:
                try:
                    coverage_full = self._get_coverage_for_id(
                        cell_id, chr_name, start, end, sparse
                    )
                    results[cell_id] = (
                        coverage_full  # Keyed by celltype ID for KeyQuery
                    )
                except Exception as e:
                    logging.warning(f"Error fetching coverage for key {cell_id}: {e}")

        elif isinstance(query, IndexQuery | MetadataQuery | MultiKeyQuery):
            raise NotImplementedError(
                f"{type(query)} is not supported for CelltypeDenseZarrIO."
            )
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        # Ensure all results have the expected length (end - start)
        # This check is important after aggregation for MultiKeyQuery as well
        expected_len = end - start

        final_results = {}
        for key, data in results.items():
            # Check the length dimension (axis 0 for dense, shape[1] for sparse row matrix if aggregation returns that)
            current_len = (
                data.shape[0]
                if isinstance(data, np.ndarray)
                else (
                    data.shape[1]
                    if isinstance(data, csr_matrix) and data.shape[0] == 1
                    else data.shape[0]
                )
            )

            if current_len == expected_len:
                final_results[key] = data
            else:
                # If shape is (0,) or similar empty, create appropriate empty array/matrix
                if data.size == 0:
                    logging.warning(
                        f"Coverage data for key '{key}' is empty. Creating empty placeholder."
                    )
                    if sparse:
                        final_results[key] = csr_matrix(
                            (0, expected_len), dtype=data.dtype
                        )
                    else:
                        final_results[key] = np.zeros(
                            (expected_len,), dtype=data.dtype
                        )  # Make it (expected_len,)
                else:
                    logging.warning(
                        f"Coverage data for key '{key}' has unexpected length {current_len} (expected {expected_len}). Shape: {data.shape}. Skipping."
                    )

        return final_results

    def write_peaks(
        self,
        id: str | None,
        peaks: str | pd.DataFrame,
        name="peaks",
        columns_to_include=["Start", "End"],
        overwrite=False,
    ):
        """Write peaks for a specific celltype ID using the internal helper."""

        if id is None:
            for id in self.ids:
                self.write_peaks(id, peaks, name, columns_to_include, overwrite)
        if id not in self.dataset:
            logging.error(f"Celltype ID '{id}' not found in dataset.")
            return

        target_group = self.dataset[id]

        self._write_peaks_internal(
            group=target_group,
            peaks=peaks,
            name=name,
            columns_to_include=columns_to_include,
            overwrite=overwrite,
            dtype="i4",  # Use standard dtype
            compressors=COMPRESSOR,
        )

    def get_peaks(self, id: str, name="cluster_peaks", chrom=None):
        """Get peaks for a specific celltype ID using the internal helper."""
        if id not in self.dataset:
            logging.error(f"Celltype ID '{id}' not found in dataset.")
            return None

        target_group = self.dataset[id]
        peaks_df = self._get_peaks_internal(group=target_group, name=name)

        if peaks_df is None:
            logging.info(f"No peaks found for {name} under celltype {id}")
            return None

        # Optional filtering by chromosome (after retrieving all peaks for the ID)
        if chrom is not None:
            if "Chromosome" in peaks_df:
                peaks_df = peaks_df[peaks_df["Chromosome"] == chrom]
                if peaks_df.empty:
                    logging.info(
                        f"No peaks found for {name} on chromosome {chrom} for celltype {id}"
                    )
                    return None  # Return None if filtering results in empty df
            else:
                logging.warning(
                    "Cannot filter by chromosome: 'Chromosome' column not found."
                )

        return peaks_df

    def remove_peaks(self, id: str | None, name: str = "cluster_peaks") -> bool:
        """Remove peaks group f'{name}' for a specific celltype ID using the internal helper."""

        if id not in self.dataset:
            logging.error(f"Celltype ID '{id}' not found in dataset.")
            return False
        if id is None:  # loop over all ids
            for id in self.ids:
                target_group = self.dataset[id]
                self._remove_peaks_internal(group=target_group, name=name)
            return True
        else:
            target_group = self.dataset[id]
            return self._remove_peaks_internal(group=target_group, name=name)

    def get_union_peaks(self, name="cluster_peaks"):
        """
        Get union peaks for all celltypes
        """
        peaks = []
        for id in self.ids:
            peaks.append(self.get_peaks(id, name))
        peaks = pd.concat(peaks)
        peaks = pr.PyRanges(peaks).merge().sort().df
        return peaks
