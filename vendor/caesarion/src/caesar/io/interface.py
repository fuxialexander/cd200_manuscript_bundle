from __future__ import annotations

import logging
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

# Conditional import for type checking
if TYPE_CHECKING:
    from caesar.io.zarr_io import MotifDenseZarrIO

    from ..analysis.track import Track

import re

from .aggregator import (
    Aggregator,
    ByKeyAggregator,
    ConcatAggregator,
    SumAggregator,
)
from .genomic_io import GenomicIO
from .mixins import (
    CallPeaksMixin,
    ClusteringMixin,
    FootprintMixin,
    LabelMixin,
    LibrarySizeAnnotationMixin,
    QuantileNormalizationMixin,
    TSSExtractionMixin,
    GenomicMaskingMixin,
    TSSTESAccessibilityMixin,
)
from .normalizer import (
    DictNormalizer,
    FactorNormalizer,
    NoNormalizer,
    Normalizer,
)
from .query import (
    BaseQuery,
    DefaultQuery,
    IndexQuery,
    KeyQuery,
    MetadataQuery,
    MultiKeyQuery,
)


class DenseIO(
    GenomicIO, FootprintMixin, TSSExtractionMixin, LibrarySizeAnnotationMixin
):
    """GenomicIO subclass for handling dense genomic data.

    This class serves as a base for representing genomic data stored in a dense
    format, typically as a single numerical array or matrix per chromosome or
    region. It assumes a simpler structure compared to sparse or multi-sample
    data, often representing a single signal track (like bulk ATAC-seq or ChIP-seq).

    It inherits the core functionality from `GenomicIO` and adds specific mixins
    relevant to dense signal analysis:
        - `FootprintMixin`: Provides methods for footprinting analysis.
        - `TSSExtractionMixin`: Provides methods for extracting signals centered
          around Transcription Start Sites (TSSs).

    Key Characteristics:
        - Implements `_get_all_ids` to typically return `['default']`, indicating a
          single primary data view.
        - Provides basic implementations for `_calculate_libsize` and
          `_calculate_per_cell_libsize` (often as no-ops or simple sums, expecting
          subclasses like `SignalDenseIO` to refine if necessary).
        - `get_default_track_obj` provides a basic way to retrieve the track data
          without specific normalization applied by default.

    Subclasses like `SignalDenseIO`, `SequenceDenseIO`, and `CelltypeDenseIO` further
    specialize this class for specific types of dense data and add more
    sophisticated methods and normalization strategies.
    """

    def _get_all_ids(self) -> list[str]:
        """Returns the identifier(s) associated with this dense data instance.

        For standard dense data (e.g., bulk signal), this typically represents a single
        conceptual entity. By convention, it returns `['default']`.

        Returns:
            list[str]: A list containing the identifier, usually `['default']`.
        """
        return ["default"]

    def _calculate_libsize(self) -> None:
        """Calculate and store the library size (`self._libsize`).

        This base implementation for `DenseIO` returns `None` and does not set
        `self._libsize`. Subclasses (like `SignalDenseIO`) that handle data
        where library size (e.g., total signal sum) is meaningful should override
        this method to perform the calculation and return the value.

        Returns:
            None: The base implementation does not compute a library size.
        """
        # Base implementation assumes libsize might not be meaningful or
        # needs specific calculation in subclass.
        # Subclasses should override and set self._libsize appropriately.
        return None

    def _calculate_per_cell_libsize(self) -> None:
        """Calculate and store per-cell library sizes (`self._per_cell_libsize`).

        The concept of per-cell library size is generally not applicable to
        standard dense genomic data. This base implementation returns `None`
        and does not set `self._per_cell_libsize`.

        Subclasses dealing with specific structures that might have a per-unit
        size concept (e.g., `CelltypeDenseIO` *before* aggregation) could
        potentially override this, but it typically remains `None` for `DenseIO`.

        Returns:
            None: Indicates per-cell library size is not applicable/calculated.
        """
        # Per-cell libsize is not relevant for standard dense data.
        # Subclasses dealing with specific structures (like CelltypeDenseIO
        # before aggregation) might override, but generally, this is None.
        return None

    def get_default_track_obj(
        self,
        chr_name: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> Track:
        """Get a default `Track` object representing the raw dense signal.

        Fetches the track data for the specified region using the base `get_track`
        method. By default, this retrieves the raw signal without applying any
        specific normalization or aggregation beyond what `get_track` does
        with default parameters when called with `output_format='track_object'`.
        The resulting `Track` object will contain the signal under the key 'default'.

        Subclasses (like `SignalDenseIO`) often override this to provide a more
        meaningful default view (e.g., normalized and smoothed signal).

        Args:
            chr_name (str | None): Chromosome name. Must be provided.
            start (int | None): Start coordinate (0-based). Must be provided.
            end (int | None): End coordinate (exclusive). Must be provided.

        Returns:
            Track: A `caesar.analysis.Track` object containing the raw signal
                   in a track named 'default'.

        Raises:
            ValueError: If `chr_name`, `start`, or `end` are not provided.
            Exception: Propagates exceptions from the underlying `get_track` call.
        """
        # Check if required arguments are provided
        if chr_name is None or start is None or end is None:
            raise ValueError(
                "chr_name, start, and end must be provided for get_default_track_obj"
            )

        return self.get_track(
            chr_name=chr_name,
            start=start,
            end=end,
            output_format="track_object",
            track_label="default",  # Basic label for the raw dense track
            # No specific aggregator or normalizer needed for default raw view
        )

    def get_default_footprint(
        self,
        m: MotifDenseZarrIO,
        n_processes: int = 16,
        extend: int = 250,
        query: BaseQuery = DefaultQuery(),
        aggregator: Aggregator = SumAggregator(),
        normalizer: Normalizer = NoNormalizer(),
        output_format: str = "raw_array",
    ):
        """Get a default footprint for all motifs, normalized by library size.

        This method calculates genome-wide footprints for all motifs and applies
        library size normalization at the end. It uses SumAggregator by default
        and NoNormalizer during processing to optimize runtime.

        Parameters
        ----------
        m : MotifDenseZarrIO
            Motif IO object containing motif information.
        n_processes : int, optional
            Number of processes to use for parallel computation, by default 16.
        extend : int, optional
            Number of bases to extend around motif match, by default 250.
        query : BaseQuery, optional
            Query to filter data, by default DefaultQuery().
        aggregator : Aggregator, optional
            Aggregator to use, by default SumAggregator().
        normalizer : Normalizer, optional
            Normalizer to use, by default NoNormalizer().
        output_format : str, optional
            Format of the output, by default "raw_array".

        Returns
        -------
        numpy.ndarray
            Footprint data normalized by library size and scaled by 1e8.
        """
        fp = self.get_genome_wide_footprint_all_motifs(
            m, n_processes, extend, query, aggregator, normalizer, output_format
        )
        # normalize by the library size at the end
        libsize = self.libsize if self.libsize is not None else 1
        return fp / libsize * 1e8


class SignalDenseIO(
    DenseIO,
    QuantileNormalizationMixin,
    CallPeaksMixin,
    FootprintMixin,
    LibrarySizeAnnotationMixin,
    GenomicMaskingMixin,
):
    """GenomicIO subclass specialized for dense genomic signal tracks.

    This class builds upon `DenseIO` to specifically handle dense signal data,
    such as bulk ATAC-seq or ChIP-seq read coverage.

    It inherits capabilities from:
        - `DenseIO`: Base dense data handling.
        - `QuantileNormalizationMixin`: Adds quantile normalization capability.
        - `CallPeaksMixin`: Adds peak calling methods.
        - `FootprintMixin`: Adds TF footprinting analysis capabilities.
        - `LibrarySizeAnnotationMixin`: Adds library size annotation support.

    Key Features:
        - Overrides `get_default_track_obj` to provide a normalized view by default.
          It applies library size normalization using `FactorNormalizer` and
          includes smoothing via convolution (default `conv_size=200`).

    This class is suitable for representing and analyzing quantitative signal
    tracks where normalization by total signal (library size) is a common
    requirement.
    """

    def get_default_track_obj(
        self,
        chr_name=None,
        start=None,
        end=None,
    ) -> Track:
        """Get a default `Track` object, normalized by library size and smoothed.

        Overrides the base `DenseIO.get_default_track_obj` implementation to
        provide a view more suitable for quantitative signal analysis (e.g.,
        bulk ATAC-seq, ChIP-seq).

        This default view includes:
        - **Normalization**: Applies library size normalization using
          `FactorNormalizer` based on the instance's `libsize` property.
          If `libsize` is not available or zero, `NoNormalizer` is used.
        - **Smoothing**: Applies smoothing by default, passing `conv_size=200`
          to the `Track` constructor.

        Args:
            chr_name (str | None): Chromosome name. Must be provided.
            start (int | None): Start coordinate (0-based). Must be provided.
            end (int | None): End coordinate (exclusive). Must be provided.

        Returns:
            Track: A `caesar.analysis.Track` object containing the normalized
                   and smoothed signal, typically under the key 'default'.

        Raises:
            ValueError: If `chr_name`, `start`, or `end` are not provided.
            Exception: Propagates exceptions from the underlying `get_track` call.
        """
        # Check if required arguments are provided
        if chr_name is None or start is None or end is None:
            raise ValueError(
                "chr_name, start, and end must be provided for get_default_track_obj"
            )

        # Use instance's libsize for normalization
        normalize_factor = (
            self.libsize if self.libsize is not None and self.libsize > 0 else 1
        )
        normalizer = (
            FactorNormalizer(normalize_factor)
            if normalize_factor != 1
            else NoNormalizer()
        )

        # Call the base get_track method with specific normalization and smoothing
        track = self.get_track(
            chr_name,
            start,
            end,
            output_format="track_object",
            normalizer=normalizer,
            conv_size=200,  # Apply default smoothing
            track_label="default",  # Label for the normalized track
        )
        return track

    def get_high_quality_summits(
        self,
        peaks: pd.DataFrame,
        threshold: int = 50,
        return_dataframe: bool = False,
        extend: int = 1024,
        return_maxval: bool = False,
        return_sum: bool = False,
        sum_over_extended: bool = False,
    ) -> dict[str, list[int]] | pd.DataFrame:
        """Identify high-quality peak summits using parallel processing.

        For each peak region, this method identifies the genomic position with the
        maximum signal value that exceeds a given threshold. It processes the genome
        chromosome by chromosome and chunk by chunk using a ProcessPoolExecutor for
        parallelism.

        Args:
            peaks: A pandas DataFrame defining peak regions. Must contain columns
                   'Chromosome', 'Start', and 'End'.
            threshold: The minimum signal value required for a position to be
                       considered a high-quality summit.
            return_dataframe: If True, returns a DataFrame with columns 'Chromosome',
                             'Start', 'End', 'Summit'. If False, returns a dictionary
                             mapping chromosomes to lists of summit positions.
            extend: Number of base pairs to extend on each side of the summit
                    when creating the Start and End coordinates in the DataFrame.
                    Only used when return_dataframe is True. Default is 1024.
            return_maxval: If True, includes the maximum signal value in the output.
                          When return_dataframe is True, adds a 'MaxValue' column.
                          When return_dataframe is False, returns a tuple of (position, value)
                          for each summit.
            return_sum: If True, includes the sum of signal values within the peak region.
                       When return_dataframe is True, adds a 'SumValue' column.
                       When return_dataframe is False, extends the tuple to include sum value.
            sum_over_extended: If True and return_sum is True, calculates the sum over
                              the extended region (summit +/- extend) instead of the
                              original peak region. Uses parallel processing for efficiency.
                              Only used when return_dataframe is True. Default is False.

        Returns:
            If return_dataframe is False:
                A dictionary mapping chromosome names to lists of summit positions
                (or tuples of (position, value) if return_maxval is True, or
                (position, value, sum) if both return_maxval and return_sum are True).
                Each summit position is a 0-based genomic coordinate.
            If return_dataframe is True:
                A pandas DataFrame with columns:
                - Chromosome: Chromosome name
                - Start: Summit position minus extend
                - End: Summit position plus extend
                - Summit: The summit position (0-based)
                - MaxValue: The maximum signal value at the summit (if return_maxval is True)
                - SumValue: The sum of signal values within the peak region, or within
                           the extended region if sum_over_extended is True (if return_sum is True)

        Raises:
            ValueError: If the peaks DataFrame is missing required columns.
            Exception: Propagates exceptions from parallel processing.
        """
        # Initialize result dictionary
        summits: dict[str, list[int] | list[tuple[int, float]]] = {}

        # Ensure required columns are in the peaks DataFrame
        if not {"Chromosome", "Start", "End"}.issubset(peaks.columns):
            raise ValueError(
                "Peaks DataFrame must contain 'Chromosome', 'Start', 'End' columns."
            )

        # Process each chromosome present in the peaks DataFrame
        for chrom in tqdm(
            peaks["Chromosome"].unique(), desc="Processing Chromosomes for Summits"
        ):
            if chrom not in self.chrom_sizes:
                logging.warning(
                    f"Chromosome {chrom} from peaks not found in active chromosomes. Skipping."
                )
                continue

            chrom_peaks = peaks.query("Chromosome == @chrom")
            if chrom_peaks.empty:
                continue  # Skip chromosome if no peaks are defined

            n_chunks = self.chrom_n_chunks.get(chrom)
            if n_chunks is None:
                logging.warning(
                    f"No chunk information for chromosome {chrom}. Skipping."
                )
                continue

            # Prepare arguments for parallel processing
            chunk_args = [
                (chrom, chunk_idx, chrom_peaks, threshold, return_maxval, return_sum)
                for chunk_idx in range(n_chunks)
            ]

            # Process chunks in parallel
            chunk_results: list[list[int] | list[tuple[int, float]]] = []
            try:
                with ProcessPoolExecutor() as executor:
                    chunk_results = list(
                        tqdm(
                            executor.map(self._process_chunk_get_summits, chunk_args),
                            total=len(chunk_args),
                        )
                    )
            except Exception as e:
                logging.error(
                    f"Error during parallel summit processing for {chrom}: {e}"
                )
                continue

            # Collect results for this chromosome
            chrom_summits = []
            for chunk_result in chunk_results:
                chrom_summits.extend(chunk_result)

            if chrom_summits:
                # Sort by position if not returning maxval, otherwise sort by position within tuples
                if return_maxval:
                    chrom_summits.sort(key=lambda x: x[0])
                else:
                    chrom_summits.sort()
                summits[chrom] = chrom_summits

        if return_dataframe:
            # Convert dictionary to DataFrame
            rows = []
            for chrom, summit_list in summits.items():
                for summit_data in summit_list:
                    if return_maxval and return_sum:
                        summit, maxval, sumval = summit_data
                    elif return_maxval:
                        summit, maxval = summit_data
                        sumval = None
                    elif return_sum:
                        summit, sumval = summit_data
                        maxval = None
                    else:
                        summit = summit_data
                        maxval = None
                        sumval = None

                    start = max(0, summit - extend)
                    end = summit + extend
                    # Check against chromosome size if available
                    if chrom in self.chrom_sizes:
                        end = min(end, self.chrom_sizes[chrom])

                    row_data = {
                        "Chromosome": chrom,
                        "Start": start,
                        "End": end,
                        "Summit": summit,
                    }
                    if return_maxval:
                        row_data["MaxValue"] = maxval
                    if return_sum:
                        row_data["SumValue"] = sumval
                    rows.append(row_data)

            result_df = pd.DataFrame(rows)

            # Recalculate sums over extended regions in parallel if requested
            if return_sum and sum_over_extended and len(result_df) > 0:
                result_df = self._calculate_region_sums_parallel(result_df)

            return result_df

        return summits

    def _process_chunk_get_summits(
        self, args: tuple[str, int, pd.DataFrame, int, bool, bool]
    ) -> list[int] | list[tuple[int, float]]:
        """Process a single chunk to find high-quality summits.

        Args:
            args: A tuple containing:
                - chrom: Chromosome name
                - chunk_idx: Index of the chunk to process
                - peaks: DataFrame containing peak regions
                - threshold: Minimum signal value for a summit
                - return_maxval: Whether to return maximum values
                - return_sum: Whether to return sum values

        Returns:
            A list of summit positions (0-based) or (position, value) tuples if return_maxval is True,
            or (position, value, sum) tuples if both return_maxval and return_sum are True.
        """
        chrom, chunk_idx, peaks, threshold, return_maxval, return_sum = args
        active_sizes = self.chrom_sizes  # Use active sizes for bounds checking

        if self.chunk_size is None or chrom not in active_sizes:
            logging.error(
                f"Chunk size ({self.chunk_size}) or chromosome '{chrom}' info missing for summit calculation in chunk {chunk_idx}."
            )
            return []

        chunk_start = chunk_idx * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, active_sizes[chrom])

        # Filter peaks to those overlapping this chunk
        chunk_peaks = peaks[(peaks["End"] > chunk_start) & (peaks["Start"] < chunk_end)]

        # Skip data loading if no peaks overlap the chunk
        if chunk_peaks.empty:
            return []

        # Get track data
        try:
            track_data = self.get_track(
                chr_name=chrom,
                start=chunk_start,
                end=chunk_end,
                output_format="raw_array",
                sparse=False,  # Ensure dense output
            )
        except Exception as e:
            logging.error(
                f"Error fetching track for summit calculation ({chrom}, chunk {chunk_idx}): {e}"
            )
            return []

        if not isinstance(track_data, np.ndarray) or track_data.ndim != 1:
            logging.error(
                f"Invalid track data format for summit calculation ({chrom}, chunk {chunk_idx})."
            )
            return []

        # Process each peak in the chunk
        summits = []
        for _, peak in chunk_peaks.iterrows():
            # Convert peak coordinates to chunk-relative coordinates
            peak_start = max(0, peak["Start"] - chunk_start)
            peak_end = min(len(track_data), peak["End"] - chunk_start)

            # Skip if peak is outside chunk bounds
            if peak_start >= len(track_data) or peak_end <= 0:
                continue

            # Get the maximum value and its position within the peak
            peak_data = track_data[peak_start:peak_end]
            if len(peak_data) == 0:
                continue

            max_val = np.max(peak_data)
            if max_val >= threshold:
                max_pos = np.argmax(peak_data) + peak_start + chunk_start

                if return_maxval and return_sum:
                    sum_val = float(np.sum(peak_data))
                    summits.append((max_pos, float(max_val), sum_val))
                elif return_maxval:
                    summits.append((max_pos, float(max_val)))
                elif return_sum:
                    sum_val = float(np.sum(peak_data))
                    summits.append((max_pos, sum_val))
                else:
                    summits.append(max_pos)

        return summits

    def _calculate_region_sums_parallel(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate sums over regions in parallel.

        This method recalculates the SumValue column for a DataFrame containing
        region coordinates. It processes data chromosome by chromosome and chunk
        by chunk using parallel processing for efficiency.

        Args:
            df: A pandas DataFrame with columns 'Chromosome', 'Start', 'End',
                and 'SumValue'. The 'SumValue' column will be updated with the
                recalculated sums.

        Returns:
            The input DataFrame with 'SumValue' column updated to contain the sum
            of signal values within each region.
        """
        if df.empty or "SumValue" not in df.columns:
            return df

        # Create a unique index for merging results back
        df = df.reset_index(drop=True)
        df["_row_idx"] = df.index

        # Process each chromosome
        all_results = []
        for chrom in tqdm(
            df["Chromosome"].unique(), desc="Calculating sums over extended regions"
        ):
            if chrom not in self.chrom_sizes:
                continue

            chrom_df = df[df["Chromosome"] == chrom]
            if chrom_df.empty:
                continue

            n_chunks = self.chrom_n_chunks.get(chrom)
            if n_chunks is None:
                continue

            # Prepare arguments for parallel processing
            chunk_args = [
                (chrom, chunk_idx, chrom_df)
                for chunk_idx in range(n_chunks)
            ]

            # Process chunks in parallel
            try:
                with ProcessPoolExecutor() as executor:
                    chunk_results = list(
                        executor.map(
                            self._process_chunk_calculate_region_sums, chunk_args
                        )
                    )
                # Flatten results
                for chunk_result in chunk_results:
                    all_results.extend(chunk_result)
            except Exception as e:
                logging.error(
                    f"Error during parallel sum calculation for {chrom}: {e}"
                )
                continue

        # Update SumValue in the original DataFrame
        if all_results:
            # results are (row_idx, sum_value) tuples
            # Group by row_idx and sum (for regions spanning multiple chunks)
            sum_dict: dict[int, float] = {}
            for row_idx, sum_val in all_results:
                if row_idx in sum_dict:
                    sum_dict[row_idx] += sum_val
                else:
                    sum_dict[row_idx] = sum_val

            # Update the DataFrame
            for row_idx, sum_val in sum_dict.items():
                df.loc[row_idx, "SumValue"] = sum_val

        # Remove temporary index column
        df = df.drop(columns=["_row_idx"])
        return df

    def _process_chunk_calculate_region_sums(
        self, args: tuple[str, int, pd.DataFrame]
    ) -> list[tuple[int, float]]:
        """Process a single chunk to calculate region sums.

        Args:
            args: A tuple containing:
                - chrom: Chromosome name
                - chunk_idx: Index of the chunk to process
                - regions_df: DataFrame with columns 'Chromosome', 'Start', 'End',
                             '_row_idx'

        Returns:
            A list of tuples (row_idx, sum_value) for regions overlapping this chunk.
        """
        chrom, chunk_idx, regions_df = args
        active_sizes = self.chrom_sizes

        if self.chunk_size is None or chrom not in active_sizes:
            return []

        chunk_start = chunk_idx * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, active_sizes[chrom])

        # Filter regions to those overlapping this chunk
        chunk_regions = regions_df[
            (regions_df["End"] > chunk_start) & (regions_df["Start"] < chunk_end)
        ]

        if chunk_regions.empty:
            return []

        # Load track data for this chunk
        try:
            track_data = self.get_track(
                chr_name=chrom,
                start=chunk_start,
                end=chunk_end,
                output_format="raw_array",
                sparse=False,
            )
        except Exception as e:
            logging.error(
                f"Error loading track data for sum calculation ({chrom}, chunk {chunk_idx}): {e}"
            )
            return []

        if not isinstance(track_data, np.ndarray) or track_data.ndim != 1:
            return []

        track_len = len(track_data)

        # Calculate sums for each region
        results = []
        for _, row in chunk_regions.iterrows():
            region_start = row["Start"]
            region_end = row["End"]
            row_idx = row["_row_idx"]

            # Calculate slice indices relative to chunk start
            rel_start = max(0, region_start - chunk_start)
            rel_end = min(track_len, region_end - chunk_start)

            if rel_start < rel_end:
                try:
                    sum_val = float(np.sum(track_data[rel_start:rel_end]))
                    results.append((row_idx, sum_val))
                except Exception:
                    results.append((row_idx, 0.0))

        return results

    def get_count_matrix(self, peaks=None, peak_name=None, **kwargs):
        """
        Get the count matrix for the given peaks.

        Abstract method that must be implemented by subclasses. Both Zarr and
        BPCells backends provide complete implementations.

        Args:
            peaks (str | pd.DataFrame | None): Specification of peaks.
                Could be a DataFrame with peak coordinates or an identifier
                to retrieve peaks stored within the object.
            peak_name (str | None): Optional name or key associated with the peaks,
                potentially used if `peaks` is an identifier.
            **kwargs: Additional backend-specific arguments.

        Returns:
            scipy.sparse.csr_matrix: The cell-by-peak count matrix.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        Note:
            Implementations should handle peak sorting, chunking for memory
            efficiency, and cell filtering based on library size thresholds.
        """
        pass

    def get_peaks(self, peak_name="peak"):
        """
        Get the peaks from the dataset.

        Abstract method that must be implemented by subclasses. Both Zarr and
        BPCells backends provide complete implementations.

        Args:
            peak_name (str, optional): Identifier for the specific peak set
                to retrieve. Defaults to "peak".

        Returns:
            pd.DataFrame: A DataFrame containing peak information with at least
                'Chromosome', 'Start', 'End' columns.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        Note:
            - Zarr backend stores peaks in zarr arrays under peak_name groups
            - BPCells backend stores peaks as BED files (e.g., {peak_name}.bed)
            - Special handling: peak_name="tiles" may generate tiles on-the-fly
              in some implementations to avoid large disk files
        """
        pass

    def create_adata(self, peak_name="peak", peak=None):
        """
        Create an AnnData object from the data stored in the Zarr dictionary.

        Constructs an `anndata.AnnData` object, typically using a cell-by-peak
        count matrix as the main data (`X`). Cell metadata (`obs`) is usually
        taken from `self.metadata`, and peak information (`var`) is retrieved
        using `get_peaks`.

        Requires the `scanpy` library (`AnnData`).

        Args:
            peak_name (str, optional): Identifier for the peak set to use for
                creating the count matrix and `var` annotation. Defaults to "peak".
            peak (pd.DataFrame | None, optional): Optionally provide the peak
                DataFrame directly instead of retrieving it via `peak_name`.
                Defaults to None.

        Returns:
            AnnData: An AnnData object representing the single-cell data.

        Raises:
            ImportError: If `scanpy` is not installed.
            NotImplementedError: If `get_count_matrix` or `get_peaks` are not
                               implemented by the subclass.
            Exception: Propagates exceptions from data retrieval or AnnData creation.
        """
        from scanpy import AnnData

        ds = self.get_count_matrix(peaks=peak, peak_name=peak_name)
        peak = self.get_peaks(peak_name)
        adata = AnnData(X=ds, obs=self.metadata, var=peak)
        return adata


class SingleSparseIO(
    GenomicIO,
    CallPeaksMixin,
    FootprintMixin,
    QuantileNormalizationMixin,
    TSSExtractionMixin,
    ClusteringMixin,
    LabelMixin,
):
    """Provides IO operations for single-sample sparse genomic data.

    This class specializes `GenomicIO` for handling sparse data formats (like
    those representing single-cell assays, e.g., scATAC-seq) where data comes
    from a single sample or experimental source. It assumes the underlying
    data backend can provide sparse matrices (e.g., `scipy.sparse.csr_matrix`).

    It integrates functionalities from various mixins for common analyses:
    - `CallPeaksMixin`: Peak calling capabilities.
    - `FootprintMixin`: TF footprinting analysis.
    - `QuantileNormalizationMixin`: Quantile normalization.
    - `TSSExtractionMixin`: Signal extraction around TSSs.
    - `ClusteringMixin`: Cell clustering methods.

    Typically, this class expects associated cell metadata (e.g., barcodes,
    clusters, library sizes) to be available, often loaded into the `metadata`
    attribute during initialization by the backend-specific implementation.

    Args:
        path (str): Path to the data source (e.g., Zarr directory, file).
        mode (str): File opening mode (e.g., 'r' for read).
        assembly (str | None, optional): Genome assembly identifier (e.g., "hg38").
            If None, attempts to infer from data. Defaults to None.
        chunk_size (int | None, optional): Desired chunk size for processing.
            If None, determined by backend. Defaults to None.
        subset_chroms (List[str] | None, optional): List of chromosomes to limit
            operations to. If None, uses all common chromosomes. Defaults to None.
        dtype (type | str | None, optional): Expected data type. Defaults to None.
        **kwargs: Additional keyword arguments passed to the specific backend
            initializer.

    Attributes:
        metadata (pd.DataFrame | None): DataFrame containing cell metadata.
            Expected to have an index aligning with cell order and potentially
            columns like 'labels', 'per_cell_libsize'. Loaded by the backend.
        n_cells (int): The total number of cells in the dataset. Often derived
            from the shape of the metadata or the data dimensions.
    """

    def _metadata_query(self, query_str):
        """Performs a query on the cell metadata DataFrame.

        Applies a filtering expression to the `metadata` attribute, which is
        expected to be a pandas DataFrame containing cell annotations. This is
        a helper method used internally, for example, by
        `_resolve_query_to_indices` when handling `MetadataQuery` objects.

        Args:
            query_str (str): A query string compatible with pandas.DataFrame.query().

        Returns:
            np.ndarray: The index values (typically integers or labels) from the
                metadata DataFrame corresponding to rows that match the query.

        Raises:
            AttributeError: If `self.metadata` is not set or is not a pandas DataFrame
                that supports the `.query()` method.
            Exception: Propagates exceptions raised by `pandas.DataFrame.query()` if
                the `query_str` is invalid (e.g., SyntaxError, NameError).
        """
        # Ensure metadata exists and is a DataFrame before querying
        if not hasattr(self, "metadata") or not isinstance(self.metadata, pd.DataFrame):
            raise AttributeError(
                "Metadata attribute is missing or not a pandas DataFrame."
            )
        try:
            return self.metadata.query(query_str).index.values
        except Exception as e:
            logging.error(f"Error executing metadata query '{query_str}': {e}")
            raise  # Re-raise the exception after logging

    def get_normalization_factor(self, ids=None):
        """
        Get normalization factor based on cell id query.

        Parameters
        ----------
        ids : array-like or None
            Cell indices to compute normalization factor for.
            If None, uses all cells.

        Returns
        -------
        float
            Sum of per-cell library sizes for selected cells, or 0.0 if error.
        """
        if (
            not hasattr(self, "metadata")
            or self.metadata is None
            or "per_cell_libsize" not in self.metadata.columns
            or not isinstance(self.per_cell_libsize, dict)
        ):
            logging.error(
                "Cannot get normalization factor: metadata or 'per_cell_libsize' missing."
            )
            return 0.0

        if ids is None:
            # Assuming default means all cells represented in the current metadata index
            ids = self.metadata.index.values
            # Alternative: if default should ignore filtering applied to metadata
            # ids = np.arange(self.n_cells)

        try:
            # Use .loc for potentially non-contiguous or non-integer indices if metadata index isn't guaranteed 0..N-1
            if isinstance(self.per_cell_libsize, dict):
                selected_libsize = pd.Series(self.per_cell_libsize).loc[ids]
            else:
                selected_libsize = self.metadata.loc[ids, "per_cell_libsize"]
            return float(selected_libsize.sum())
        except KeyError:
            logging.warning(
                "Some requested IDs not found in metadata index for normalization factor calculation."
            )
            # Attempt with iloc if ids are integer positions
            try:
                if np.issubdtype(np.array(ids).dtype, np.integer):
                    selected_libsize = self.metadata["per_cell_libsize"].iloc[ids]
                    return float(selected_libsize.sum())
                else:
                    logging.error(
                        "Cannot calculate normalization factor with non-integer IDs for iloc."
                    )
                    return 0.0
            except Exception as e:
                logging.error(f"Error calculating normalization factor: {e}")
                return 0.0
        except Exception as e:
            logging.error(f"Error calculating normalization factor: {e}")
            return 0.0

    def get_default_track_obj(self, chr_name=None, start=None, end=None) -> Track:
        # Check if metadata has 'labels' column
        if (
            hasattr(self, "metadata")
            and isinstance(self.metadata, pd.DataFrame)
            and "labels" in self.metadata.columns
        ):
            # Get unique labels and filter out cluster_unassigned
            unique_labels = self.metadata["labels"].unique()
            unique_labels = [label for label in unique_labels if label != "cluster_unassigned"]

            # Create a MultiKeyQuery with MetadataQuery for each label
            label_queries = {
                label: MetadataQuery(f"labels == '{label}'") for label in unique_labels
            }
            multi_key_query = MultiKeyQuery(label_queries)

            # Filter libsize dict to only include labels in queries (defensive)
            libsize_dict = None
            if isinstance(self.libsize, dict):
                libsize_dict = {k: v for k, v in self.libsize.items() if k != "cluster_unassigned" and k in unique_labels}

            # Use ByKeyAggregator and DictNormalizer with filtered self.libsize
            return self.get_track_obj(
                chr_name,
                start,
                end,
                query=multi_key_query,
                aggregator=ByKeyAggregator(),
                normalizer=DictNormalizer(libsize_dict)
                if libsize_dict and len(libsize_dict) > 0
                else NoNormalizer(),
            )
        else:
            # Fall back to original implementation
            default_indices = self._resolve_query_to_indices(DefaultQuery())
            norm_factor = self.get_normalization_factor(ids=default_indices)

            # Pass resolved indices to get_track_obj (inherited/base method)
            return self.get_track_obj(  # Call the base class get_track_obj here
                chr_name,
                start,
                end,
                query=IndexQuery(default_indices),  # Pass specific indices
                normalizer=FactorNormalizer(norm_factor)
                if norm_factor > 0
                else NoNormalizer(),
                track_label="default_sum",  # Example label
            )

    def get_count_matrix(self, peaks=None, peak_name=None, **kwargs):
        """
        Get the count matrix for the given peaks.

        Abstract method that must be implemented by subclasses. Both Zarr and
        BPCells backends provide complete implementations.

        Args:
            peaks (str | pd.DataFrame | None): Specification of peaks.
                Could be a DataFrame with peak coordinates or an identifier
                to retrieve peaks stored within the object.
            peak_name (str | None): Optional name or key associated with the peaks,
                potentially used if `peaks` is an identifier.
            **kwargs: Additional backend-specific arguments.

        Returns:
            scipy.sparse.csr_matrix: The cell-by-peak count matrix.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        Note:
            Implementations should handle peak sorting, chunking for memory
            efficiency, and cell filtering based on library size thresholds.
        """
        pass

    def get_peaks(self, peak_name="peak"):
        """
        Get the peaks from the dataset.

        Abstract method that must be implemented by subclasses. Both Zarr and
        BPCells backends provide complete implementations.

        Args:
            peak_name (str, optional): Identifier for the specific peak set
                to retrieve. Defaults to "peak".

        Returns:
            pd.DataFrame: A DataFrame containing peak information with at least
                'Chromosome', 'Start', 'End' columns.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        Note:
            - Zarr backend stores peaks in zarr arrays under peak_name groups
            - BPCells backend stores peaks as BED files (e.g., {peak_name}.bed)
            - Special handling: peak_name="tiles" may generate tiles on-the-fly
              in some implementations to avoid large disk files
        """
        pass

    def create_adata(self, peak_name="peak", peak=None):
        """
        Create an AnnData object from the data stored in the Zarr dictionary.

        Constructs an `anndata.AnnData` object, typically using a cell-by-peak
        count matrix as the main data (`X`). Cell metadata (`obs`) is usually
        taken from `self.metadata`, and peak information (`var`) is retrieved
        using `get_peaks`.

        Requires the `scanpy` library (`AnnData`).

        Args:
            peak_name (str, optional): Identifier for the peak set to use for
                creating the count matrix and `var` annotation. Defaults to "peak".
            peak (pd.DataFrame | None, optional): Optionally provide the peak
                DataFrame directly instead of retrieving it via `peak_name`.
                Defaults to None.

        Returns:
            AnnData: An AnnData object representing the single-cell data.

        Raises:
            ImportError: If `scanpy` is not installed.
            NotImplementedError: If `get_count_matrix` or `get_peaks` are not
                               implemented by the subclass.
            Exception: Propagates exceptions from data retrieval or AnnData creation.
        """
        from scanpy import AnnData

        ds = self.get_count_matrix(peaks=peak, peak_name=peak_name)
        peak = self.get_peaks(peak_name)
        adata = AnnData(X=ds, obs=self.metadata, var=peak)
        return adata

    def get_libsize_dict(self, groupby="labels"):
        """Generate library size dictionary based on metadata grouping

        Parameters
        ----------
        groupby : str, default='labels'
            Column name in metadata to group by

        Returns
        -------
        dict
            Dictionary mapping group names to total library sizes
        """
        if not hasattr(self, "metadata") or groupby not in self.metadata.columns:
            raise ValueError(f"No '{groupby}' column in metadata")

        # Calculate per-group library sizes
        libsize_dict = {}
        for group in self.metadata[groupby].unique():
            if pd.isna(group):
                continue
            group_cells = self.metadata[self.metadata[groupby] == group].index
            if hasattr(self, "per_cell_libsize") and isinstance(
                self.per_cell_libsize, dict
            ):
                # Use pre-computed per-cell library sizes (dict format)
                libsize_dict[group] = sum(
                    self.per_cell_libsize.get(bc, 0)
                    for bc in self.metadata.loc[group_cells, "barcodes"]
                )
            elif hasattr(self, "per_cell_libsize") and isinstance(
                self.per_cell_libsize, pd.Series
            ):
                # Use pre-computed per-cell library sizes (Series format)
                libsize_dict[group] = self.per_cell_libsize.loc[group_cells].sum()
            else:
                # Fall back to calculating from data
                libsize_dict[group] = self.get_normalization_factor(group_cells)

        return libsize_dict

    @property
    def libsize_by_labels(self):
        """Get library sizes grouped by labels (convenience property)"""
        if hasattr(self, "_libsize_by_labels"):
            return self._libsize_by_labels
        self._libsize_by_labels = self.get_libsize_dict("labels")
        return self._libsize_by_labels

    def get_pseudobulk_track(self, chr_name, start, end, groupby="labels", **kwargs):
        """Get pseudobulk tracks grouped by metadata column"""
        if not hasattr(self, "metadata") or groupby not in self.metadata.columns:
            raise ValueError(f"No '{groupby}' column in metadata")

        # Auto-construct query and normalizers
        groups = self.metadata[groupby].unique()
        queries = {
            group: MetadataQuery(f"{groupby} == '{group}'")
            for group in groups
            if not pd.isna(group)
        }
        query = MultiKeyQuery(queries)

        # Get per-group library sizes
        norm_factors = self.get_libsize_dict(groupby)

        return self.get_track(
            chr_name,
            start,
            end,
            query=query,
            aggregator=ByKeyAggregator(),
            normalizer=DictNormalizer(norm_factors),
            **kwargs,
        )


class MultiSparseIO(
    GenomicIO,
    CallPeaksMixin,
    FootprintMixin,
    QuantileNormalizationMixin,
    TSSExtractionMixin,
    ClusteringMixin,
    LabelMixin,
):
    """Provides IO operations for aggregated sparse genomic data from multiple sources.

    This class extends `GenomicIO` to manage and query sparse genomic data
    aggregated from multiple independent samples, experiments, or sources
    (e.g., multiple scATAC-seq datasets).

    It typically orchestrates operations across a collection of underlying
    `GenomicIO` instances (often `SingleSparseIO`), frequently stored in an
    attribute like `zarr_dict` (though the exact mechanism depends on the backend
    implementation).

    Key functionalities include:
    - Aggregating metadata from all sources into a single `metadata` DataFrame,
      often adding a 'data_key' column to distinguish origins.
    - Calculating properties like `data_chroms` by finding the intersection of
      chromosomes present across all sources.
    - Implementing query resolution (`_get_coverage_for_query`) to potentially fetch
      and combine data from multiple underlying storage locations based on the query.
    - Calculating grouped library sizes (`_calculate_libsize`) across samples/labels.
    - Providing methods for analyzing multi-sample characteristics, such as
      label counts (`get_label_counts`), label distributions (`get_label_distribution`),
      and cell QC metrics across samples (`get_cell_qc_metrics`).

    Like `SingleSparseIO`, it integrates functionalities from various mixins:
    - `CallPeaksMixin`: Peak calling capabilities (potentially aggregated).
    - `FootprintMixin`: TF footprinting analysis.
    - `QuantileNormalizationMixin`: Quantile normalization.
    - `TSSExtractionMixin`: Signal extraction around TSSs.
    - `ClusteringMixin`: Cell clustering methods.

    Args:
        path (str | dict): Path to the data source. This might be a single path
            if the backend handles multi-sample structure internally, or often a
            dictionary mapping sample IDs to individual data paths if initialized
            externally (specifics depend on backend/factory function).
        mode (str): File opening mode (e.g., 'r' for read).
        assembly (str | None, optional): Genome assembly identifier. Defaults to None.
        chunk_size (int | None, optional): Chunk size. Defaults to None.
        subset_chroms (List[str] | None, optional): Chromosomes to subset. Defaults to None.
        dtype (type | str | None, optional): Data type. Defaults to None.
        **kwargs: Additional backend-specific keyword arguments.

    Attributes:
        zarr_dict (dict[str, GenomicIO] | None): Dictionary mapping data keys
            (e.g., sample IDs) to their corresponding `GenomicIO` instances.
            This attribute name is common but may vary by implementation.
        metadata (pd.DataFrame | None): Aggregated DataFrame containing cell metadata
            from all sources, typically including a 'data_key' column.
        n_cells (int): Total number of cells across all managed datasets.
    """

    # --- Chromosome properties need careful handling for MultiSparse ---
    @property
    def data_chroms(self) -> list[str]:
        """Returns intersection of data chromosomes from all underlying stores."""
        if not self.zarr_dict:
            return []
        all_data_chroms = [set(z.data_chroms) for z in self.zarr_dict.values()]
        if not all_data_chroms:
            return []
        intersection_chroms = set.intersection(*all_data_chroms)
        return sorted(list(intersection_chroms))

    # Note: _active_chroms, assembly_chrom_sizes are typically set during _init_data_structure
    # based on consistency check and intersection across zarr_dict.
    # Properties like chroms, chrom_sizes, chrom_n_chunks inherited from GenomicIO
    # should work correctly if _active_chroms and assembly_chrom_sizes are set properly.

    def _calculate_libsize(self):
        """Calculate library size aggregated per label/group across all managed sources.

        Computes the sum of `per_cell_libsize` for all cells belonging to each
        unique label found in the `self.metadata['labels']` column. This is useful
        for normalization or analysis grouped by cell type, cluster, or other
        categorical annotations.

        Assumes `self.metadata` is a pandas DataFrame containing both a 'labels'
        column and a 'per_cell_libsize' column.

        Returns:
            dict[str, float]: A dictionary mapping each unique label to the sum of
                              library sizes for cells with that label. Returns an
                              empty dictionary if required metadata columns are
                              missing or if an error occurs during calculation.

        Side Effects:
            Logs an error if required metadata columns are missing.
            Logs an error if the calculation fails.
        """
        if (
            not hasattr(self, "metadata")
            or self.metadata is None
            or "labels" not in self.metadata.columns
            or "per_cell_libsize" not in self.metadata.columns
        ):
            # This is not an error - libsize calculation is optional and not required for clustering
            logging.debug(
                "Cannot calculate grouped libsize: metadata, 'labels', or 'per_cell_libsize' missing. "
                "This is optional and not required for clustering operations."
            )
            self._libsize = {}  # Set to empty dict
            return self._libsize

        logging.info("Calculating library size per label group...")
        try:
            # Ensure per_cell_libsize is numeric
            numeric_libsize = pd.to_numeric(
                self.metadata["per_cell_libsize"], errors="coerce"
            ).fillna(0)
            return (
                self.metadata.groupby("labels")
                .apply(lambda x: numeric_libsize.loc[x.index].sum())
                .to_dict()
            )
        except Exception as e:
            logging.error(f"Error calculating library size per label group: {e}")
            return {}  # Set to empty dict on error

    def _calculate_per_cell_libsize(self):
        """Retrieve or calculate per-cell library sizes for all cells.

        This method aims to return a pandas Series containing the library size
        for each individual cell across all aggregated datasets.

        Typically, this relies on the `self.metadata` DataFrame having been correctly
        populated during initialization (e.g., by concatenating metadata from
        individual sources) and containing a 'per_cell_libsize' column.

        Returns:
            pd.Series | None: A pandas Series indexed by cell identifier, where
                              values are the per-cell library sizes. Returns `None`
                              if the metadata or the 'per_cell_libsize' column
                              is missing.

        Side Effects:
            Logs a warning if the metadata or column is missing.
        """
        # This usually relies on the metadata DataFrame being correctly populated
        # during _init_data_structure by concatenating metadata from individual stores.
        if (
            hasattr(self, "metadata")
            and self.metadata is not None
            and "per_cell_libsize" in self.metadata.columns
        ):
            return self.metadata["per_cell_libsize"]
        else:
            logging.warning(
                "Cannot assign per_cell_libsize: metadata or column missing."
            )
            return None

    def _get_all_ids(self):
        """Returns a sorted list of unique identifiers for the managed data sources.

        These identifiers typically correspond to the keys used to manage the
        individual datasets within this multi-source object (e.g., the keys
        of an internal dictionary like `self.zarr_dict`). They often represent
        sample names, experiment IDs, or batch identifiers.

        Returns:
            list[str]: A sorted list of the unique data source identifiers.
                       Returns an empty list if no sources are managed.
        """
        # Assumes zarr_dict keys are the data_keys/sample IDs
        return sorted(list(self.zarr_dict.keys())) if hasattr(self, "zarr_dict") else []

    def get_default_track_obj(self, chr_name=None, start=None, end=None) -> Track:
        """Get a default `Track` object representing aggregated signal across all cells.

        Provides a default view for the multi-sparse dataset, typically showing
        the sum of signals across *all* cells from *all* managed data sources,
        normalized by the total library size of all these cells.

        Steps:
        1. Calculates the total library size across all cells (sum of
           `self.per_cell_libsize` or sum of grouped `self.libsize`).
        2. Calls `get_track_obj` using `DefaultQuery()` (which signals the
           backend implementation to retrieve and aggregate data from all sources)
           and `SumAggregator()`.
        3. Applies normalization using `FactorNormalizer` with the calculated
           total library size.

        Args:
            chr_name (str | None): Chromosome name. Must be provided.
            start (int | None): Start coordinate (0-based). Must be provided.
            end (int | None): End coordinate (exclusive). Must be provided.

        Returns:
            Track: A `caesar.analysis.Track` object containing the normalized
                   total signal, typically under a label like 'total_signal'.

        Raises:
            ValueError: If `chr_name`, `start`, or `end` are not provided.
            Exception: Propagates exceptions from underlying methods like
                       `get_track_obj` or library size calculation.

        Side Effects:
            Logs a warning if the total library size cannot be determined for normalization.
        """
        # Check if metadata has 'labels' column
        if (
            hasattr(self, "metadata")
            and isinstance(self.metadata, pd.DataFrame)
            and "labels" in self.metadata.columns
        ):
            # Get unique labels and filter out cluster_unassigned
            unique_labels = self.metadata["labels"].unique()
            unique_labels = [label for label in unique_labels if label != "cluster_unassigned"]

            # Create a MultiKeyQuery with MetadataQuery for each label
            label_queries = {
                label: MetadataQuery(f"labels == '{label}'") for label in unique_labels
            }
            multi_key_query = MultiKeyQuery(label_queries)

            # Filter libsize dict to only include labels in queries (defensive)
            libsize_dict = None
            if isinstance(self.libsize, dict):
                libsize_dict = {k: v for k, v in self.libsize.items() if k != "cluster_unassigned" and k in unique_labels}

            # Use ByKeyAggregator and DictNormalizer with filtered self.libsize
            return self.get_track_obj(
                chr_name,
                start,
                end,
                query=multi_key_query,
                aggregator=ByKeyAggregator(),
                normalizer=DictNormalizer(libsize_dict)
                if libsize_dict and len(libsize_dict) > 0
                else NoNormalizer(),
            )
        else:
            # MultiSparse default might mean summing across *all* cells from *all* samples.
            # Query needs to specify this. MultiKeyQuery might be suitable if per-sample sum is needed.
            # For a single summed track:
            total_libsize = 0.0
            if isinstance(self.per_cell_libsize, pd.Series):
                total_libsize = float(self.per_cell_libsize.sum())
            elif isinstance(
                self.libsize, dict
            ):  # Sum of group libsizes if per-cell unavailable
                total_libsize = float(sum(self.libsize.values()))
            else:
                logging.warning(
                    "Cannot determine total libsize for default track normalization."
                )

            # Use DefaultQuery to get data via _get_coverage_for_query,
            # which should handle merging across the zarr_dict.
            # Use SumAggregator.
            return self.get_track_obj(
                chr_name,
                start,
                end,
                query=DefaultQuery(),  # Signal to _get_coverage to fetch from all
                aggregator=SumAggregator(),
                normalizer=FactorNormalizer(total_libsize)
                if total_libsize > 0
                else NoNormalizer(),
                track_label="total_signal",  # Example label
            )

    def get_label_counts(
        self, threshold: int = 200, normalize: bool = False
    ) -> pd.Series:
        """Get the counts of cells for each label, optionally filtered by a threshold.

        Calculates the number of cells associated with each unique label in the
        `self.metadata['labels']` column. Can filter out labels with fewer than
        `threshold` cells and optionally normalize the counts to fractions.

        Args:
            threshold (int, optional): The minimum number of cells a label must
                have to be included in the result. Set to 0 or negative to disable
                filtering. Defaults to 200.
            normalize (bool, optional): If True, return the counts as fractions
                of the total number of cells passing the threshold. If False,
                return raw counts. Defaults to False.

        Returns:
            pd.Series: A pandas Series indexed by label, containing the cell counts
                       or fractions. Returns an empty Series if the required metadata
                       or 'labels' column is missing, or if an error occurs.

        Side Effects:
            Logs an error if metadata or the 'labels' column is missing.
            Logs an error if the calculation fails.
        """
        if (
            not hasattr(self, "metadata")
            or self.metadata is None
            or "labels" not in self.metadata.columns
        ):
            logging.error(
                "Cannot get label counts: metadata or 'labels' column missing."
            )
            return pd.Series(dtype=int)  # Return empty Series

        try:
            counts = self.metadata.labels.value_counts()
            if threshold > 0:
                counts = counts[counts >= threshold]
            if normalize:
                total = counts.sum()
                counts = (
                    counts / total if total > 0 else counts
                )  # Avoid division by zero
        except Exception as e:
            logging.error(f"Error calculating label counts: {e}")
            return pd.Series(dtype=int)  # Return empty Series

        return counts

    def get_label_distribution(self, by: str = "sample") -> pd.DataFrame:
        """Calculate the distribution of labels by sample, or samples by label.

        Computes a cross-tabulation to show the proportional distribution of
        cell labels within each sample, or conversely, the distribution of samples
        contributing to each cell label.

        Assumes `self.metadata` contains 'labels' and 'data_key' columns,
        where 'data_key' represents the sample or source identifier.

        Args:
            by (str, optional): Specifies the grouping for normalization:
                - 'sample': Normalizes across rows (samples). Each row sums to 1,
                  showing the fraction of each label within that sample.
                - 'label': Normalizes across columns (labels). Each column sums to 1,
                  showing the fraction of each sample within that label.
                Defaults to "sample".

        Returns:
            pd.DataFrame: A pandas DataFrame containing the cross-tabulated
                          proportions. Rows are indexed by the 'by' group (sample or
                          label), and columns by the other category. Returns an
                          empty DataFrame if required metadata columns are missing
                          or if `by` is invalid or an error occurs.

        Side Effects:
            Logs an error if metadata or required columns are missing.
            Logs an error if `by` argument is invalid.
            Logs an error if the calculation fails.
        """
        if (
            not hasattr(self, "metadata")
            or self.metadata is None
            or "labels" not in self.metadata.columns
            or "data_key" not in self.metadata.columns
        ):
            logging.error(
                "Cannot get label distribution: metadata, 'labels', or 'data_key' column missing."
            )
            return pd.DataFrame()  # Return empty DataFrame

        try:
            if by == "sample":
                # Get label distribution within each sample
                dist = pd.crosstab(
                    self.metadata.data_key, self.metadata.labels, normalize="index"
                )
            elif by == "label":
                # Get sample distribution within each label
                dist = pd.crosstab(
                    self.metadata.labels, self.metadata.data_key, normalize="index"
                )
            else:
                logging.error(
                    f"Invalid 'by' argument for get_label_distribution: {by}. Choose 'sample' or 'label'."
                )
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error calculating label distribution: {e}")
            return pd.DataFrame()

        return dist

    def get_cell_qc_metrics(self) -> pd.DataFrame:
        """Calculate and return common quality control (QC) metrics per cell.

        Computes several QC metrics based on the aggregated `self.metadata`,
        focusing primarily on library size (`per_cell_libsize`) statistics
        relative to the cell's sample of origin (`data_key`).

        Metrics calculated:
        - Library size (`per_cell_libsize`, made numeric)
        - Sample identifier (`data_key`)
        - Cell label (`labels`, if present)
        - Library size percentile rank within its sample.
        - Library size Z-score within its sample.

        Assumes `self.metadata` contains 'per_cell_libsize' and 'data_key' columns.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the original metadata plus
                          the calculated QC columns (`libsize_percentile`,
                          `libsize_zscore`, `per_cell_libsize_numeric`). Returns an
                          empty DataFrame if required metadata columns are missing
                          or if an error occurs during calculation.

        Side Effects:
            Logs an error if required metadata columns are missing.
            Logs an error if calculation fails.
            Handles potential division by zero in Z-score calculation for samples
            with zero standard deviation in library size.
        """
        if (
            not hasattr(self, "metadata")
            or self.metadata is None
            or "per_cell_libsize" not in self.metadata.columns
            or "data_key" not in self.metadata.columns
        ):
            logging.error(
                "Cannot get QC metrics: metadata, 'per_cell_libsize', or 'data_key' column missing."
            )
            return pd.DataFrame()  # Return empty DataFrame

        qc = self.metadata.copy()

        # Calculate metrics safely
        try:
            # Ensure libsize is numeric
            qc["per_cell_libsize_numeric"] = pd.to_numeric(
                qc["per_cell_libsize"], errors="coerce"
            )

            # Add percentile ranks within each sample
            qc["libsize_percentile"] = qc.groupby("data_key")[
                "per_cell_libsize_numeric"
            ].transform(
                lambda x: pd.qcut(
                    x.rank(method="first"), q=100, labels=False, duplicates="drop"
                )  # rank needed for qcut
            )

            # Add z-scores within each sample
            def z_score(x):
                # Calculate z-score only if std dev is not zero
                std_dev = x.std()
                if std_dev == 0:
                    return np.zeros_like(x)  # Return zeros if no variation
                else:
                    return (x - x.mean()) / std_dev

            qc["libsize_zscore"] = qc.groupby("data_key")[
                "per_cell_libsize_numeric"
            ].transform(z_score)

            # Clean up intermediate numeric column if desired
            # qc = qc.drop(columns=['per_cell_libsize_numeric'])

        except Exception as e:
            logging.error(f"Error calculating QC metrics: {e}")
            # Return the initial copy or an empty frame depending on desired behavior on error
            return pd.DataFrame()

        return qc

    def get_libsize_dict(self, groupby="labels"):
        """Generate library size dictionary based on metadata grouping

        Parameters
        ----------
        groupby : str, default='labels'
            Column name in metadata to group by

        Returns
        -------
        dict
            Dictionary mapping group names to total library sizes
        """
        if not hasattr(self, "metadata") or groupby not in self.metadata.columns:
            raise ValueError(f"No '{groupby}' column in metadata")

        # Calculate per-group library sizes
        libsize_dict = {}
        for group in self.metadata[groupby].unique():
            if pd.isna(group):
                continue
            group_cells = self.metadata[self.metadata[groupby] == group].index
            # MultiSparseIO typically has per_cell_libsize in metadata
            if "per_cell_libsize" in self.metadata.columns:
                # Convert to numeric and sum
                group_libsizes = pd.to_numeric(
                    self.metadata.loc[group_cells, "per_cell_libsize"], errors="coerce"
                ).fillna(0)
                libsize_dict[group] = float(group_libsizes.sum())
            else:
                # Fall back to using libsize property if available
                if hasattr(self, "libsize") and isinstance(self.libsize, dict):
                    libsize_dict[group] = self.libsize.get(group, 0.0)
                else:
                    logging.warning(
                        f"Cannot calculate library size for group '{group}'"
                    )
                    libsize_dict[group] = 0.0

        return libsize_dict

    @property
    def libsize_by_labels(self):
        """Get library sizes grouped by labels (convenience property)"""
        if hasattr(self, "_libsize_by_labels"):
            return self._libsize_by_labels
        self._libsize_by_labels = self.get_libsize_dict("labels")
        return self._libsize_by_labels

    def get_pseudobulk_track(self, chr_name, start, end, groupby="labels", **kwargs):
        """Get pseudobulk tracks grouped by metadata column"""
        if not hasattr(self, "metadata") or groupby not in self.metadata.columns:
            raise ValueError(f"No '{groupby}' column in metadata")

        # Auto-construct query and normalizers
        groups = self.metadata[groupby].unique()
        queries = {
            group: MetadataQuery(f"{groupby} == '{group}'")
            for group in groups
            if not pd.isna(group)
        }
        query = MultiKeyQuery(queries)

        # Get per-group library sizes
        norm_factors = self.get_libsize_dict(groupby)

        return self.get_track(
            chr_name,
            start,
            end,
            query=query,
            aggregator=ByKeyAggregator(),
            normalizer=DictNormalizer(norm_factors),
            **kwargs,
        )


class SequenceDenseIO(DenseIO, FootprintMixin):
    """GenomicIO subclass specialized for dense sequence representations.

    This class extends `DenseIO` to handle genomic data where the signal represents
    sequence information directly, typically stored densely. A common use case is
    one-hot encoded DNA sequences (e.g., ACGT represented as [1,0,0,0], [0,1,0,0],
    etc.), where each position along the genome has a vector corresponding to the
    nucleotide present.

    It inherits capabilities from:
        - `DenseIO`: Base dense data handling.
        - `FootprintMixin`: Adds footprinting analysis capabilities, which can be
          relevant for analyzing sequence motifs in conjunction with accessibility data.

    Key Features:
        - Assumes the data represents sequence, often with a specific encoding
          (like one-hot). The default `dtype` is often 'int8'.
        - Provides sequence-specific methods like `gc_content` to calculate the
          G+C percentage within a region and `get_peak_gc_content` to calculate it
          over a set of peak intervals.

    Args:
        path (str): Path to the data source (e.g., Zarr directory).
        mode (str, optional): File opening mode. Defaults to 'r'.
        dtype (str, optional): Data type, typically 'int8' for one-hot encoding.
            Defaults to 'int8'.
        assembly (str | None, optional): Genome assembly. Defaults to None.
        chunk_size (int | None, optional): Chunk size. Defaults to None.
        subset_chroms (List[str] | None, optional): Chromosomes to subset. Defaults to None.
        **kwargs: Additional backend-specific keyword arguments.
    """

    def __init__(self, path: str, mode="r", dtype="int8"):
        """Initializes the SequenceDenseIO object.

        Calls the parent `GenomicIO` initializer but specifically sets the default
        `dtype` to 'int8', which is commonly used for one-hot encoded sequence
        representations (e.g., A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]).

        Args:
            path (str): Path to the data file/directory.
            mode (str, optional): Mode for opening files ('r', 'a', 'r+', 'w').
                                Defaults to "r".
            dtype (str, optional): Data type for sequence data. Overrides the
                                 `GenomicIO` default and sets to "int8".
        """
        # Call the parent __init__ but override the default dtype
        super().__init__(path, mode=mode, dtype=dtype)

    def gc_content(self, chr, start, end):
        """Calculate the GC content (fraction of G and C bases) for a region.

        Retrieves the sequence data for the specified region, assuming a
        one-hot encoding where the columns represent A, C, G, T.
        It sums the values in the C (index 1) and G (index 2) columns
        and divides by the total length of the region.

        Args:
            chr (str): Chromosome name.
            start (int): Start coordinate (0-based, inclusive).
            end (int): End coordinate (0-based, exclusive).

        Returns:
            float: The fraction of Guanine (G) and Cytosine (C) bases in the
                   specified region. Returns NaN if the region length is zero.

        Raises:
            Exception: Propagates exceptions from `get_track` if data retrieval fails.
            IndexError: If the retrieved track does not have at least 3 columns
                        (assuming ACGT encoding).
        """
        seq = self.get_track(chr, start, end)
        # one hot ACGT
        # set A T to 0, C G to 1
        gc = seq[:, 1] + seq[:, 2]
        return gc.sum() / len(gc)

    def _compute_gc_content(self, iter):
        """Helper function to compute GC content for a peak from DataFrame row.

        Designed to be used with `pandas.DataFrame.iterrows()` or similar iteration
        that yields `(index, row)` tuples, where the row contains 'Chromosome',
        'Start', and 'End' columns.

        Args:
            iter (tuple): A tuple containing `(index, row)`, where `row` is a
                          pandas Series or similar object with 'Chromosome',
                          'Start', and 'End' attributes/keys.

        Returns:
            float: The GC content calculated by calling `self.gc_content` with
                   the coordinates from the row.
        """
        i, row = iter
        return self.gc_content(row.Chromosome, row.Start, row.End)

    def get_peak_gc_content(self, peaks, n_jobs=None):
        """Calculate GC content for a list of genomic regions (peaks) in parallel.

        Iterates through a DataFrame of peaks, calculates the GC content for each
        peak region using `_compute_gc_content`, and utilizes a process pool
        (`concurrent.futures.ProcessPoolExecutor`) for parallel computation.

        Args:
            peaks (pd.DataFrame): A DataFrame where each row represents a peak
                and must contain columns named 'Chromosome', 'Start', and 'End'.
            n_jobs (int | None, optional): The maximum number of worker processes
                to use. If None, `ProcessPoolExecutor` typically defaults based
                on the number of CPUs. Defaults to None.

        Returns:
            list[float]: A list containing the calculated GC content for each peak
                         in the input `peaks` DataFrame, in the same order.

        Side Effects:
            Prints a warning if the number of peaks is large (> 100,000).
            Uses `tqdm` to display a progress bar if available.

        Raises:
            AttributeError: If the `peaks` DataFrame is missing required columns.
            Exception: Propagates exceptions from the parallel execution or
                       `_compute_gc_content`.
        """
        if len(peaks) > 100000:
            print("Warning: large number of peaks, this may take a while")
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(n_jobs) as executor:
            results = list(
                tqdm(
                    executor.map(self._compute_gc_content, peaks.iterrows()),
                    total=len(peaks),
                )
            )

        return results


class CelltypeDenseIO(
    DenseIO,
    CallPeaksMixin,
    FootprintMixin,
    QuantileNormalizationMixin,
    TSSExtractionMixin,
    LibrarySizeAnnotationMixin,
    GenomicMaskingMixin,
    TSSTESAccessibilityMixin,
):
    """GenomicIO subclass for dense data explicitly grouped by cell type or sample.

    This class extends `DenseIO` to manage dense genomic data where the signal
    is inherently structured into distinct groups, typically representing different
    cell types, tissues, or experimental samples within a single dataset (e.g.,
    aggregated bulk ATAC-seq profiles per cell type).

    It assumes the backend stores data in a way that allows retrieval based on
    these group identifiers (referred to as 'IDs' or 'celltype IDs' within the class,
    although they could represent other groupings).

    It inherits functionalities from various mixins:
    - `DenseIO`: Base dense data handling.
    - `CallPeaksMixin`: Peak calling capabilities (often applied per group).
    - `FootprintMixin`: Footprinting analysis (potentially per group).
    - `QuantileNormalizationMixin`: Quantile normalization.
    - `TSSExtractionMixin`: Signal extraction around TSSs.

    Key Features:
    - Manages a list of group identifiers (`self.ids`).
    - Often provides a `metadata` property derived from these IDs, parsing them
      to extract structured information (e.g., celltype, data_key, sample, depth).
    - Includes an abstract `subset` method that subclasses must implement to create
      a new instance containing only specified group IDs.
    - Provides methods for querying and manipulating data based on these group IDs,
      such as `get_high_quality_summits`, `get_peak_counts`, `leave_out_celltypes`,
      `filter_by_min_depth`, etc.
    - `get_default_track_obj` typically returns a `Track` containing normalized data
      for *all* managed cell types/groups, using `ByKeyAggregator`.

    Args:
        path (str): Path to the data source.
        mode (str, optional): File opening mode. Defaults to 'r'.
        dtype (str, optional): Data type. Defaults to 'float32'.
        assembly (str | None, optional): Genome assembly. Defaults to None.
        chunk_size (int | None, optional): Chunk size. Defaults to None.
        subset_chroms (List[str] | None, optional): Chromosomes to subset. Defaults to None.
        **kwargs: Additional backend-specific keyword arguments.

    Attributes:
        ids (List[str]): List of unique identifiers for the cell types or groups
            managed by this instance. Loaded by the backend implementation.
    """

    @abstractmethod
    def subset(self, ids: list[str]) -> CelltypeDenseIO:
        """Create a new instance containing only the specified group IDs.

        This abstract method must be implemented by subclasses. It should return
        a new instance of the class, configured to represent only the data
        associated with the provided identifiers.

        Args:
            ids: A list of group identifiers (e.g., cell type IDs) to include
                 in the subsetted object.

        Returns:
            A new instance of the `CelltypeDenseIO` subclass containing data
            only for the specified IDs.
        """
        pass

    @property
    def metadata(self):
        """Generate a pandas DataFrame containing metadata derived from group IDs.

        Assumes IDs follow a specific format (e.g., "celltype.data_key.sample.depth")
        and parses them accordingly.

        Returns:
            pandas.DataFrame: A DataFrame with columns 'celltypes', 'data_key',
                              'sample', 'depth' (as int), and 'id' (original ID).
        """
        # Implementation assumes a specific ID format: "celltype.data_key.sample.depth"
        # This might need adjustment depending on the actual ID structure.
        try:
            metadata = pd.DataFrame(
                [id.split(".") + [id] for id in self.ids],
                columns=["celltypes", "data_key", "sample", "depth", "id"],
            )
            metadata["depth"] = metadata["depth"].astype("int")
        except (ValueError, IndexError) as e:
            logging.warning(
                f"Could not parse all IDs into metadata DataFrame assuming format "
                f"'celltype.data_key.sample.depth'. Error: {e}. "
                f"Returning DataFrame with 'id' column only."
            )
            metadata = pd.DataFrame({"id": self.ids})
        return metadata

    def _get_all_ids(self) -> dict[str, str]:
        """Return a dictionary mapping group IDs to themselves.

        Used internally by the query parsing mechanism when a `DefaultQuery` or
        `KeyQuery` needs to resolve all available IDs.

        Returns:
            A dictionary where keys and values are the group identifiers managed
            by this instance (`self.ids`).
        """
        return {i: i for i in self.ids}

    def get_default_track_obj(
        self,
        chr_name: str,
        start: int,
        end: int,
        conv_size: int = 100,
    ) -> Track:
        """Get a Track object containing normalized data for all managed groups.

        Retrieves data for the specified genomic region for every group ID
        managed by this instance. It uses `ByKeyAggregator` to keep the data
        for each group separate within the resulting `Track` object and
        applies library size normalization using `DictNormalizer` based on
        `self.libsize`.

        Args:
            chr_name: Chromosome name.
            start: Start coordinate (0-based).
            end: End coordinate (exclusive).

        Returns:
            A `Track` object containing normalized tracks for all groups.
            Each track within the object corresponds to a group ID.

        Raises:
            ValueError: If `chr_name`, `start`, or `end` are not provided.
        """
        # Check if required arguments are provided (redundant with type hints but explicit)
        if chr_name is None or start is None or end is None:
            raise ValueError(
                "chr_name, start, and end must be provided for get_default_track_obj"
            )

        # Use ByKeyAggregator to get all tracks in the output dict
        # Use DictNormalizer with self.libsize if it's a dictionary
        norm = (
            DictNormalizer(self.libsize)
            if isinstance(self.libsize, dict)
            else NoNormalizer()
        )
        return self.get_track(
            chr_name=chr_name,
            start=start,
            end=end,
            query=DefaultQuery(),  # Query all available cell types/groups
            aggregator=ByKeyAggregator(),  # Keep tracks separate
            normalizer=norm,
            conv_size=conv_size,
            output_format="track_object",
        )

    def get_default_footprint(
        self,
        m: MotifDenseZarrIO,
        n_processes: int = 16,
        extend: int = 250,
        query: BaseQuery = DefaultQuery(),
        aggregator: Aggregator = ConcatAggregator(),
        normalizer: Normalizer = NoNormalizer(),
        output_format: str = "raw_array",
    ):
        """Get a default footprint for all motifs, normalized by library size.

        This method calculates genome-wide footprints for all motifs and applies
        library size normalization. It uses ConcatAggregator by default to maintain
        separate profiles for each group/cell type. Returns a numpy array of shape
        (n_motifs, n_bases, n_groups).

        Parameters
        ----------
        m : MotifDenseZarrIO
            Motif IO object containing motif information.
        n_processes : int, optional
            Number of processes to use for parallel computation, by default 16.
        extend : int, optional
            Number of bases to extend around motif match, by default 250.
        query : BaseQuery, optional
            Query to filter data, by default DefaultQuery().
        aggregator : Aggregator, optional
            Aggregator to use, by default ConcatAggregator().
        normalizer : Normalizer, optional
            Normalizer to use, by default NoNormalizer().
        output_format : str, optional
            Format of the output, by default "raw_array".

        Returns
        -------
        numpy.ndarray
            Footprint data normalized by library size and scaled by 1e8.
        """
        fp = self.get_genome_wide_footprint_all_motifs(
            m, n_processes, extend, query, aggregator, normalizer, output_format
        )
        # normalize by the library size # a dict
        libsize = np.array(list(self.libsize.values()))
        return fp / libsize[None, None, :] * 1e8

    def sample_track(self, chr_name: str, start: int, end: int) -> np.ndarray:
        """Retrieve track data for a randomly selected group ID.

        Selects one group ID at random from `self.ids` and fetches the
        raw (non-aggregated, non-normalized) track data for that specific
        group in the given genomic region.

        Args:
            chr_name: Chromosome name.
            start: Start coordinate (0-based).
            end: End coordinate (exclusive).

        Returns:
            A NumPy array containing the raw signal data for the randomly
            chosen group in the specified region.
        """
        # Note: The original code used self.celltypes, assuming self.ids was intended.
        if not self.ids:
            logging.warning("No IDs available to sample from.")
            # Depending on desired behavior, could return None, empty array, or raise error.
            # Returning empty array for now.
            return np.array([], dtype=self.dtype)

        random_id = np.random.choice(self.ids)
        # Fetch raw data for the single selected ID
        return self.get_track(
            chr_name=chr_name,
            start=start,
            end=end,
            query=KeyQuery(random_id),
            aggregator=SumAggregator(),  # Aggregates the single group
            output_format="raw_array",
        )

    def get_high_quality_summits(
        self,
        peaks: pd.DataFrame,
        threshold: int = 50,
        return_dataframe: bool = False,
        extend: int = 1024,
        return_maxval: bool = False,
        return_sum: bool = False,
        recenter: bool = True,
        sum_over_extended: bool = False,
    ) -> dict[str, dict[str, list[int]]] | pd.DataFrame:
        """Identify high-quality peak summits across all groups using parallel processing.

        For each group (ID), this method identifies the genomic position with the
        maximum signal value that exceeds a given threshold within each peak region.
        It processes the genome chromosome by chromosome and chunk by chunk using
        a ProcessPoolExecutor for parallelism.

        Args:
            peaks: A pandas DataFrame defining peak regions. Must contain columns
                   'Chromosome', 'Start', and 'End'.
            threshold: The minimum signal value required for a position to be
                       considered a high-quality summit for a group.
            return_dataframe: If True, returns a DataFrame with columns 'Chromosome',
                             'Start', 'End', 'Summit', 'Group'. If False, returns
                             a nested dictionary mapping groups to chromosomes to
                             lists of summit positions.
            extend: Number of base pairs to extend on each side of the summit
                    when creating the Start and End coordinates in the DataFrame.
                    Only used when return_dataframe is True and recenter is True. Default is 1024.
            return_maxval: If True, includes the maximum signal value in the output.
                          When return_dataframe is True, adds a 'MaxValue' column.
                          When return_dataframe is False, returns a tuple of (position, value)
                          for each summit.
            return_sum: If True, includes the sum of signal values within the peak region.
                       When return_dataframe is True, adds a 'SumValue' column.
                       When return_dataframe is False, extends the tuple to include sum value.
            recenter: If True (default), creates new Start/End coordinates by extending
                     around the summit position using the extend parameter. If False,
                     preserves the original Start/End coordinates from the input peaks
                     DataFrame. Only used when return_dataframe is True.
            sum_over_extended: If True and return_sum is True, calculates the sum over
                              the final output region (extended or original depending on
                              recenter) instead of the original peak region. Uses parallel
                              processing for efficiency. Only used when return_dataframe
                              is True. Default is False.

        Returns:
            If return_dataframe is False:
                A nested dictionary structured as {group_id: {chromosome: [summit_pos1, ...]}}.
                Contains the 0-based genomic coordinates of the identified high-quality
                summits for each group and chromosome (or tuples of (position, value)
                if return_maxval is True, or (position, value, sum) if both return_maxval
                and return_sum are True).
            If return_dataframe is True:
                A pandas DataFrame with columns:
                - Chromosome: Chromosome name
                - Start: Summit position minus extend (if recenter) or original Start
                - End: Summit position plus extend (if recenter) or original End
                - Summit: The summit position (0-based)
                - Group: The group ID
                - MaxValue: The maximum signal value at the summit (if return_maxval is True)
                - SumValue: The sum of signal values within the peak region, or within
                           the output region if sum_over_extended is True (if return_sum is True)

        Raises:
            ValueError: If the peaks DataFrame is missing required columns.
            Exception: Propagates exceptions from parallel processing.
        """
        # Initialize result dictionary
        cluster_summits: dict[str, dict[str, list[tuple]]] = {
            cluster_id: {} for cluster_id in self.ids
        }

        # Ensure required columns are in the peaks DataFrame
        if not {"Chromosome", "Start", "End"}.issubset(peaks.columns):
            raise ValueError(
                "Peaks DataFrame must contain 'Chromosome', 'Start', 'End' columns."
            )

        # Process each chromosome present in the peaks DataFrame
        for chrom in tqdm(
            peaks["Chromosome"].unique(), desc="Processing Chromosomes for Summits"
        ):
            if chrom not in self.chrom_sizes:
                logging.warning(
                    f"Chromosome {chrom} from peaks not found in active chromosomes. Skipping."
                )
                continue

            chrom_peaks = peaks.query("Chromosome == @chrom")
            if chrom_peaks.empty:
                continue  # Skip chromosome if no peaks are defined

            n_chunks = self.chrom_n_chunks.get(chrom)
            if n_chunks is None:
                logging.warning(
                    f"No chunk information for chromosome {chrom}. Skipping."
                )
                continue

            # Prepare arguments for parallel processing
            chunk_args = [
                (chrom, chunk_idx, chrom_peaks, threshold, return_maxval, return_sum)
                for chunk_idx in range(n_chunks)
            ]

            # Process chunks in parallel
            chunk_results: list[dict[str, list[tuple]]] = []
            try:
                with ProcessPoolExecutor() as executor:
                    chunk_results = list(
                        tqdm(
                            executor.map(self._process_chunk_get_summits, chunk_args),
                            total=len(chunk_args),
                        )
                    )
            except Exception as e:
                logging.error(
                    f"Error during parallel summit processing for {chrom}: {e}"
                )
                continue

            # Collect results for this chromosome
            for cluster_id in self.ids:
                # Ensure chromosome entry exists for the cluster
                if chrom not in cluster_summits[cluster_id]:
                    cluster_summits[cluster_id][chrom] = []

                # Aggregate summit positions from all chunk results for this cluster
                for chunk_result in chunk_results:
                    if cluster_id in chunk_result:
                        cluster_summits[cluster_id][chrom].extend(
                            chunk_result[cluster_id]
                        )
                # Sort the summit positions (all results are now tuples with coordinates)
                if cluster_summits[cluster_id][chrom]:
                    cluster_summits[cluster_id][chrom].sort(key=lambda x: x[0])

        if return_dataframe:
            # Convert nested dictionary to DataFrame
            rows = []
            for group_id, chrom_dict in cluster_summits.items():
                for chrom, summit_list in chrom_dict.items():
                    for summit_data in summit_list:
                        # Parse the tuple based on what values were requested
                        if return_maxval and return_sum:
                            summit, maxval, sumval, peak_start, peak_end = summit_data
                        elif return_maxval:
                            summit, maxval, peak_start, peak_end = summit_data
                            sumval = None
                        elif return_sum:
                            summit, sumval, peak_start, peak_end = summit_data
                            maxval = None
                        else:
                            summit, peak_start, peak_end = summit_data
                            maxval = None
                            sumval = None

                        if recenter:
                            # Use extended coordinates around the summit
                            start = max(0, summit - extend)
                            end = summit + extend
                            # Check against chromosome size if available
                            if chrom in self.chrom_sizes:
                                end = min(end, self.chrom_sizes[chrom])
                        else:
                            # Use the precomputed original peak coordinates
                            start = peak_start
                            end = peak_end

                        row_data = {
                            "Chromosome": chrom,
                            "Start": start,
                            "End": end,
                            "Summit": summit,
                            "Group": group_id,
                        }
                        if return_maxval:
                            row_data["MaxValue"] = maxval
                        if return_sum:
                            # Store original sum; will be recalculated if sum_over_extended
                            row_data["SumValue"] = sumval
                        rows.append(row_data)

            result_df = pd.DataFrame(rows)

            # Recalculate sums over extended regions in parallel if requested
            if return_sum and sum_over_extended and len(result_df) > 0:
                result_df = self._calculate_region_sums_parallel(result_df)

            return result_df

        return cluster_summits

    def _process_chunk_get_summits(
        self, args: tuple[str, int, pd.DataFrame, int, bool, bool]
    ) -> dict[str, list[int] | list[tuple[int, float]]]:
        """Process a single chunk to find high-quality summits for each group.

        Args:
            args: A tuple containing:
                - chrom: Chromosome name
                - chunk_idx: Index of the chunk to process
                - peaks: DataFrame containing peak regions
                - threshold: Minimum signal value for a summit
                - return_maxval: Whether to return maximum values
                - return_sum: Whether to return sum values

        Returns:
            A dictionary mapping group IDs to lists of tuples containing:
            - (position, peak_start, peak_end) if neither return_maxval nor return_sum
            - (position, value, peak_start, peak_end) if return_maxval is True
            - (position, sum, peak_start, peak_end) if return_sum is True
            - (position, value, sum, peak_start, peak_end) if both return_maxval and return_sum are True
        """
        chrom, chunk_idx, peaks, threshold, return_maxval, return_sum = args
        active_sizes = self.chrom_sizes  # Use active sizes for bounds checking

        if self.chunk_size is None or chrom not in active_sizes:
            logging.error(
                f"Chunk size ({self.chunk_size}) or chromosome '{chrom}' info missing for summit calculation in chunk {chunk_idx}."
            )
            return {}

        chunk_start = chunk_idx * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, active_sizes[chrom])

        # Filter peaks to those overlapping this chunk
        chunk_peaks = peaks[(peaks["End"] > chunk_start) & (peaks["Start"] < chunk_end)]

        # Skip data loading if no peaks overlap the chunk
        if chunk_peaks.empty:
            return {}

        # Get track data for all groups
        try:
            track_data = self._load_raw_chunk_for_all_id(chrom, chunk_idx)
        except Exception as e:
            logging.error(
                f"Error fetching track for summit calculation ({chrom}, chunk {chunk_idx}): {e}"
            )
            return {}

        # Process each peak in the chunk
        cluster_summits: dict[str, list[int] | list[tuple[int, float]]] = {
            cluster_id: [] for cluster_id in self.ids
        }

        for _, peak in chunk_peaks.iterrows():
            # Convert peak coordinates to chunk-relative coordinates
            peak_start = max(0, peak["Start"] - chunk_start)
            peak_end = min(len(track_data[self.ids[0]]), peak["End"] - chunk_start)

            # Skip if peak is outside chunk bounds
            if peak_start >= len(track_data[self.ids[0]]) or peak_end <= 0:
                continue

            # For each group, find the maximum value and its position within the peak
            for cluster_id in self.ids:
                if cluster_id not in track_data:
                    continue

                peak_data = track_data[cluster_id][peak_start:peak_end]
                if len(peak_data) == 0:
                    continue

                max_val = np.max(peak_data)
                if max_val >= threshold:
                    max_pos = np.argmax(peak_data) + peak_start + chunk_start

                    # Always include original peak coordinates for efficiency
                    peak_start_coord = int(peak["Start"])
                    peak_end_coord = int(peak["End"])

                    if return_maxval and return_sum:
                        sum_val = float(np.sum(peak_data))
                        cluster_summits[cluster_id].append((max_pos, float(max_val), sum_val, peak_start_coord, peak_end_coord))
                    elif return_maxval:
                        cluster_summits[cluster_id].append((max_pos, float(max_val), peak_start_coord, peak_end_coord))
                    elif return_sum:
                        sum_val = float(np.sum(peak_data))
                        cluster_summits[cluster_id].append((max_pos, sum_val, peak_start_coord, peak_end_coord))
                    else:
                        cluster_summits[cluster_id].append((max_pos, peak_start_coord, peak_end_coord))

        return cluster_summits

    def _calculate_region_sums_parallel(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate sums over regions for all groups in parallel.

        This method recalculates the SumValue column for a DataFrame containing
        region coordinates and group IDs. It processes data chromosome by chromosome
        and chunk by chunk using parallel processing for efficiency.

        Args:
            df: A pandas DataFrame with columns 'Chromosome', 'Start', 'End', 'Group',
                and 'SumValue'. The 'SumValue' column will be updated with the
                recalculated sums.

        Returns:
            The input DataFrame with 'SumValue' column updated to contain the sum
            of signal values within each region for the corresponding group.
        """
        if df.empty or "SumValue" not in df.columns:
            return df

        # Create a unique index for merging results back
        df = df.reset_index(drop=True)
        df["_row_idx"] = df.index

        # Process each chromosome
        all_results = []
        for chrom in tqdm(
            df["Chromosome"].unique(), desc="Calculating sums over extended regions"
        ):
            if chrom not in self.chrom_sizes:
                continue

            chrom_df = df[df["Chromosome"] == chrom]
            if chrom_df.empty:
                continue

            n_chunks = self.chrom_n_chunks.get(chrom)
            if n_chunks is None:
                continue

            # Prepare arguments for parallel processing
            chunk_args = [
                (chrom, chunk_idx, chrom_df)
                for chunk_idx in range(n_chunks)
            ]

            # Process chunks in parallel
            try:
                with ProcessPoolExecutor() as executor:
                    chunk_results = list(
                        executor.map(
                            self._process_chunk_calculate_region_sums, chunk_args
                        )
                    )
                # Flatten results
                for chunk_result in chunk_results:
                    all_results.extend(chunk_result)
            except Exception as e:
                logging.error(
                    f"Error during parallel sum calculation for {chrom}: {e}"
                )
                continue

        # Update SumValue in the original DataFrame
        if all_results:
            # results are (row_idx, sum_value) tuples
            # Group by row_idx and sum (for regions spanning multiple chunks)
            sum_dict: dict[int, float] = {}
            for row_idx, sum_val in all_results:
                if row_idx in sum_dict:
                    sum_dict[row_idx] += sum_val
                else:
                    sum_dict[row_idx] = sum_val

            # Update the DataFrame
            for row_idx, sum_val in sum_dict.items():
                df.loc[row_idx, "SumValue"] = sum_val

        # Remove temporary index column
        df = df.drop(columns=["_row_idx"])
        return df

    def _process_chunk_calculate_region_sums(
        self, args: tuple[str, int, pd.DataFrame]
    ) -> list[tuple[int, float]]:
        """Process a single chunk to calculate region sums for all groups.

        Args:
            args: A tuple containing:
                - chrom: Chromosome name
                - chunk_idx: Index of the chunk to process
                - regions_df: DataFrame with columns 'Chromosome', 'Start', 'End',
                             'Group', '_row_idx'

        Returns:
            A list of tuples (row_idx, sum_value) for regions overlapping this chunk.
        """
        chrom, chunk_idx, regions_df = args
        active_sizes = self.chrom_sizes

        if self.chunk_size is None or chrom not in active_sizes:
            return []

        chunk_start = chunk_idx * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, active_sizes[chrom])

        # Filter regions to those overlapping this chunk
        chunk_regions = regions_df[
            (regions_df["End"] > chunk_start) & (regions_df["Start"] < chunk_end)
        ]

        if chunk_regions.empty:
            return []

        # Load track data for all groups in this chunk
        try:
            track_data = self._load_raw_chunk_for_all_id(chrom, chunk_idx)
        except Exception as e:
            logging.error(
                f"Error loading track data for sum calculation ({chrom}, chunk {chunk_idx}): {e}"
            )
            return []

        # Calculate sums for each region
        results = []
        for _, row in chunk_regions.iterrows():
            region_start = row["Start"]
            region_end = row["End"]
            group_id = row["Group"]
            row_idx = row["_row_idx"]

            if group_id not in track_data:
                continue

            group_track = track_data[group_id]
            track_len = len(group_track)

            # Calculate slice indices relative to chunk start
            rel_start = max(0, region_start - chunk_start)
            rel_end = min(track_len, region_end - chunk_start)

            if rel_start < rel_end:
                try:
                    sum_val = float(np.sum(group_track[rel_start:rel_end]))
                    results.append((row_idx, sum_val))
                except Exception:
                    results.append((row_idx, 0.0))

        return results

    def get_peak_counts(
        self, celltype_id: str, name: str = "peaks_p0.01", chrom: str | None = None
    ) -> pd.DataFrame:
        """Calculate the total signal count within defined peaks for a specific group.

        Retrieves peak definitions using `self.get_peaks` and then sums the
        signal values within those peak boundaries for the specified `celltype_id`.
        Processes data chromosome by chromosome and chunk by chunk in parallel.

        Args:
            celltype_id: The identifier of the group for which to calculate counts.
            name: The name identifier for the peak set to use (passed to `self.get_peaks`).
            chrom: Optional. If provided, calculates counts only for this chromosome.
                   If None (default), calculates for all chromosomes found in the
                   peak set.

        Returns:
            A pandas DataFrame with columns 'Chromosome', 'Start', 'End', 'Count',
            where 'Start' and 'End' define the peak region and 'Count' is the
            summed signal for `celltype_id` within that peak.

        Raises:
            ValueError: If the specified `celltype_id` is not found in `self.ids`.
            TypeError: If `get_peaks` does not return a pandas DataFrame.
            Exception: Propagates exceptions from `get_peaks` or parallel processing.
        """
        if celltype_id not in self.ids:
            raise ValueError(
                f"Celltype ID '{celltype_id}' not found in available IDs: {self.ids}"
            )

        # Get peaks for the specified chromosome or all chromosomes
        # Assumes self.get_peaks exists (from CallPeaksMixin or similar)
        # and returns a DataFrame with 'Chromosome', 'Start', 'End'.
        try:
            peaks = self.get_peaks(celltype_id, name, chrom)  # type: ignore
            if not isinstance(peaks, pd.DataFrame):
                raise TypeError(
                    f"Expected pandas DataFrame from get_peaks, but got {type(peaks)}"
                )
            if not {"Chromosome", "Start", "End"}.issubset(peaks.columns):
                raise ValueError(
                    "Peaks DataFrame from get_peaks must contain 'Chromosome', 'Start', 'End' columns."
                )
        except AttributeError:
            raise NotImplementedError(
                "The 'get_peaks' method is required but not implemented or inherited by this class."
            )
        except Exception as e:
            logging.error(
                f"Error calling get_peaks for {celltype_id}, name={name}, chrom={chrom}: {e}"
            )
            raise  # Re-raise the original error

        # Process each chunk in parallel and collect peak counts
        chr_results = []
        chromosomes_to_process = [chrom] if chrom else peaks["Chromosome"].unique()

        for current_chrom in tqdm(
            chromosomes_to_process, desc=f"Processing {celltype_id} Peak Counts"
        ):
            if current_chrom not in self.chrom_sizes:
                logging.warning(
                    f"Chromosome {current_chrom} from peaks not found in active chromosomes. Skipping."
                )
                continue

            chrom_peaks = peaks.query("Chromosome == @current_chrom")
            if chrom_peaks.empty:
                continue  # Skip chromosome if no peaks defined

            n_chunks = self.chrom_n_chunks.get(current_chrom)
            if n_chunks is None:
                logging.warning(
                    f"No chunk information for chromosome {current_chrom}. Skipping."
                )
                continue

            # Prepare arguments for parallel chunk processing
            chunk_args = [
                (current_chrom, idx, chrom_peaks, celltype_id)
                for idx in range(n_chunks)
            ]

            all_peak_counts_for_chrom: list[tuple[int, int, float]] = []
            try:
                with ProcessPoolExecutor() as executor:
                    results = list(
                        executor.map(self._process_chunk_get_peak_counts, chunk_args)
                    )
                # Flatten the list of lists results from chunks
                for chunk_result in results:
                    all_peak_counts_for_chrom.extend(chunk_result)
            except Exception as e:
                logging.error(
                    f"Error during parallel peak count processing for {celltype_id}, {current_chrom}: {e}"
                )
                # Decide how to handle: continue, raise, etc. Continuing for now.
                continue

            # Aggregate counts by peak for the current chromosome
            if all_peak_counts_for_chrom:
                peak_counts_df = pd.DataFrame(
                    all_peak_counts_for_chrom, columns=["Start", "End", "Count"]
                )
                # Group by peak definition and sum counts (handles peaks spanning chunks)
                peak_counts_df = (
                    peak_counts_df.groupby(["Start", "End"])["Count"]
                    .sum()
                    .reset_index()
                )
                peak_counts_df["Chromosome"] = current_chrom
                chr_results.append(peak_counts_df)

        # Concatenate results from all chromosomes
        if not chr_results:
            # Return empty DataFrame with correct columns if no results
            return pd.DataFrame(columns=["Chromosome", "Start", "End", "Count"])
        else:
            final_peak_counts_df = pd.concat(chr_results, ignore_index=True)
            # Ensure correct column order
            return final_peak_counts_df[["Chromosome", "Start", "End", "Count"]]

    def _process_chunk_get_peak_counts(
        self, args: tuple[str, int, pd.DataFrame, str]
    ) -> list[tuple[int, int, float]]:
        """Process a single chunk to calculate peak counts for a specific group.

        Helper function for `get_peak_counts`. Fetches the raw track data for the
        specified group ID within the chunk and sums the signal within the
        boundaries of any peaks that overlap the chunk.

        Args:
            args: A tuple containing:
                chrom (str): Chromosome name.
                chunk_idx (int): Index of the chunk to process.
                peaks (pd.DataFrame): DataFrame of peaks filtered for the current
                                      chromosome.
                celltype_id (str): The group ID for which to calculate counts.

        Returns:
            A list of tuples, where each tuple is (peak_start, peak_end, count).
            'count' is the sum of the signal for `celltype_id` within that peak's
            region intersecting the current chunk. A peak spanning multiple chunks
            will appear in the results of each chunk it overlaps.
        """
        chrom, chunk_idx, peaks, celltype_id = args
        active_sizes = self.chrom_sizes  # Use active sizes for bounds checking

        if self.chunk_size is None or chrom not in active_sizes:
            logging.error(
                f"Chunk size ({self.chunk_size}) or chromosome '{chrom}' info missing for peak count calculation in chunk {chunk_idx}."
            )
            return []

        chunk_start = chunk_idx * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, active_sizes[chrom])

        # Filter peaks to those overlapping this chunk
        peaks_in_chunk = peaks[
            (peaks["Start"] < chunk_end)
            & (peaks["End"] > chunk_start)
            # Redundant check if peaks were pre-filtered, but safe
            & (peaks["Chromosome"] == chrom)
        ]

        # Fetch track data only if there are relevant peaks in the chunk
        if peaks_in_chunk.empty:
            # logging.debug(f"No peaks overlap chunk {chunk_idx} on {chrom}. Skipping track loading.")
            return []

        # Use get_track with raw_array output and specify the single celltype ID
        try:
            track = self.get_track(
                chr_name=chrom,
                start=chunk_start,
                end=chunk_end,
                query=KeyQuery(celltype_id),  # Query only the specific group
                aggregator=SumAggregator(),  # Aggregates the single group's data
                output_format="raw_array",  # Expect a 1D array for the chunk
                sparse=False,  # Ensure dense for easier indexing
            )
            if not isinstance(track, np.ndarray) or track.ndim != 1:
                raise TypeError(
                    f"Expected 1D numpy array, got {type(track)} with shape {getattr(track, 'shape', 'N/A')}"
                )

        except Exception as e:
            logging.error(
                f"Error fetching track for peak counts ({celltype_id}, {chrom}, chunk {chunk_idx}): {e}"
            )
            return []  # Return empty list on error

        # Sum up values in each peak region within the chunk
        peak_counts: list[tuple[int, int, float]] = []
        track_len = len(track)  # Length of the data fetched for this chunk

        for _, peak in peaks_in_chunk.iterrows():
            peak_start, peak_end = peak["Start"], peak["End"]

            # Calculate slice indices relative to the start of the chunk's data array
            # Peak region: [peak_start, peak_end)
            # Chunk region: [chunk_start, chunk_end)
            # Data array covers chunk region, indexed 0 to track_len-1
            # Start index in array: max(0, peak_start - chunk_start)
            # End index in array: min(track_len, peak_end - chunk_start)
            rel_start = max(0, peak_start - chunk_start)
            rel_end = min(track_len, peak_end - chunk_start)

            # Ensure the slice is valid (rel_start < rel_end) and within bounds
            if rel_start < rel_end:
                # Sum the track values within the calculated slice
                try:
                    count = np.sum(track[rel_start:rel_end])
                    # Append original peak coordinates and the calculated count
                    peak_counts.append((peak["Start"], peak["End"], float(count)))
                except IndexError as e:
                    logging.error(
                        f"IndexError during peak count summation for peak {peak_start}-{peak_end} "
                        f"in chunk {chunk_idx} ({chrom}). Slice: [{rel_start}:{rel_end}], Track length: {track_len}. Error: {e}"
                    )
                    # Append with 0 count or handle differently
                    peak_counts.append((peak["Start"], peak["End"], 0.0))

            else:
                # If slice is invalid (e.g., peak ends before chunk starts, though filtered earlier), append 0 count.
                # This case might indicate an issue with filtering or coordinate logic.
                # logging.debug(f"Invalid slice [{rel_start}:{rel_end}] for peak {peak_start}-{peak_end} in chunk {chunk_idx} ({chrom}). Appending 0 count.")
                peak_counts.append((peak["Start"], peak["End"], 0.0))

        return peak_counts

    def _load_normalized_chunk_for_all_id(
        self, chr_name: str, chunk_idx: int,
    ) -> dict[str, np.ndarray]:
        """Load and normalize data for a single chunk across all groups.

        Fetches data for the specified chunk (`chr_name`, `chunk_idx`) for all
        managed group IDs. It leverages `get_default_track_obj` internally,
        which handles fetching data for all IDs and applying library size
        normalization (`DictNormalizer`).

        Args:
            chr_name: Chromosome name.
            chunk_idx: Index of the chunk to load.

        Returns:
            A dictionary mapping group ID to its corresponding normalized
            1D NumPy array for the specified chunk. Returns an empty dictionary
            if chunk information is missing or an error occurs during data loading.
        """
        active_sizes = self.chrom_sizes  # Use active sizes for bounds checking

        if self.chunk_size is None or chr_name not in active_sizes:
            logging.error(
                f"Chunk size ({self.chunk_size}) or chromosome '{chr_name}' info missing for loading normalized chunk {chunk_idx}."
            )
            return {}

        chunk_start = chunk_idx * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, active_sizes[chr_name])

        # Use get_default_track_obj as it handles normalization and aggregation correctly
        try:
            # This retrieves a Track object containing normalized data for all IDs
            track_obj = self.get_default_track_obj(chr_name, chunk_start, chunk_end)
            # Return the dictionary of tracks held by the Track object
            return track_obj.tracks if track_obj and track_obj.tracks else {}
        except ValueError as e:
            # Catch potential ValueError from get_default_track_obj if coords invalid (shouldn't happen here)
            logging.error(
                f"ValueError loading normalized chunk {chunk_idx} for {chr_name}: {e}"
            )
            return {}
        except Exception as e:
            logging.error(
                f"Unexpected error loading normalized chunk {chunk_idx} for {chr_name}: {e}"
            )
            return {}  # Return empty dict on other errors

    def _load_raw_chunk_for_all_id(
        self, chr_name: str, chunk_idx: int,
    ) -> dict[str, np.ndarray]:
        """Load raw (unnormalized) data for a single chunk across all groups.

        Fetches raw data for the specified chunk (`chr_name`, `chunk_idx`) for all
        managed group IDs. Unlike `_load_normalized_chunk_for_all_id`, this method
        uses `get_track` with `output_format='raw_dict'` to get unnormalized data,
        similar to how SignalDenseIO handles summit detection.

        Args:
            chr_name: Chromosome name.
            chunk_idx: Index of the chunk to load.

        Returns:
            A dictionary mapping group ID to its corresponding raw (unnormalized)
            1D NumPy array for the specified chunk. Returns an empty dictionary
            if chunk information is missing or an error occurs during data loading.
        """
        active_sizes = self.chrom_sizes  # Use active sizes for bounds checking

        if self.chunk_size is None or chr_name not in active_sizes:
            logging.error(
                f"Chunk size ({self.chunk_size}) or chromosome '{chr_name}' info missing for loading raw chunk {chunk_idx}."
            )
            return {}

        chunk_start = chunk_idx * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, active_sizes[chr_name])

        # Use get_track with raw_dict to get unnormalized data for all groups
        try:
            # Use ByKeyAggregator to separate data by group and NoNormalizer for raw data
            from .aggregator import ByKeyAggregator
            from .normalizer import NoNormalizer
            from .query import DefaultQuery

            track_data = self.get_track(
                chr_name=chr_name,
                start=chunk_start,
                end=chunk_end,
                query=DefaultQuery(),  # Query all available groups
                aggregator=ByKeyAggregator(),  # Keep data separate by group
                normalizer=NoNormalizer(),  # No normalization - raw data
                output_format="raw_dict",  # Return dictionary format
                sparse=False,  # Ensure dense output
            )
            return track_data if isinstance(track_data, dict) else {}
        except ValueError as e:
            # Catch potential ValueError from get_track if coords invalid (shouldn't happen here)
            logging.error(
                f"ValueError loading raw chunk {chunk_idx} for {chr_name}: {e}"
            )
            return {}
        except Exception as e:
            logging.error(
                f"Unexpected error loading raw chunk {chunk_idx} for {chr_name}: {e}"
            )
            return {}  # Return empty dict on other errors

    def leave_out_celltypes(
        self, celltype_ids: list[str] | set[str], inverse: bool = False
    ) -> CelltypeDenseIO:
        """Create a subset instance excluding or including specific group IDs.

        Calls the abstract `subset` method to generate a new instance.

        Args:
            celltype_ids: A list or set of group identifiers to either exclude
                          (if `inverse` is False) or exclusively include
                          (if `inverse` is True).
            inverse: If False (default), the returned subset contains all IDs
                     *except* those in `celltype_ids`. If True, the returned
                     subset contains *only* the IDs in `celltype_ids`.

        Returns:
            A new `CelltypeDenseIO` instance representing the subset.

        Raises:
            NotImplementedError: If the `subset` method hasn't been implemented
                                 by the concrete subclass.
        """
        current_ids = set(self.ids)
        ids_to_process = set(celltype_ids)

        if not inverse:
            # Keep IDs NOT in the input list
            final_ids = list(current_ids.difference(ids_to_process))
        else:
            # Keep ONLY IDs that ARE in the input list AND exist in the object
            final_ids = list(current_ids.intersection(ids_to_process))

        if len(final_ids) == len(self.ids):
            logging.warning("leave_out_celltypes resulted in no change to the ID list.")
        elif not final_ids:
            logging.warning("leave_out_celltypes resulted in an empty ID list.")

        # The `subset` method is abstract and must be implemented by subclasses.
        # This call might raise NotImplementedError if not implemented.
        return self.subset(final_ids)

    def leave_out_celltypes_with_pattern(
        self, pattern: str, inverse: bool = False
    ) -> CelltypeDenseIO:
        """Create a subset instance excluding or including IDs matching a regex pattern.

        Filters the current group IDs (`self.ids`) based on whether they match the
        provided regular expression pattern. Calls the abstract `subset` method
        to generate the new instance.

        Args:
            pattern: The regular expression pattern (string) to match against group IDs.
            inverse: If False (default), the returned subset contains IDs that
                     *do not* match the pattern. If True, the returned subset
                     contains *only* IDs that *do* match the pattern.

        Returns:
            A new `CelltypeDenseIO` instance representing the subset.

        Raises:
            re.error: If the provided `pattern` is an invalid regular expression.
            NotImplementedError: If the `subset` method hasn't been implemented
                                 by the concrete subclass.
        """
        try:
            # Compile the pattern once for efficiency and error checking
            compiled_pattern = re.compile(pattern)
        except re.error as e:
            logging.error(f"Invalid regex pattern provided: '{pattern}'. Error: {e}")
            raise  # Re-raise the regex error

        # Filter IDs based on regex search and the inverse flag
        if inverse:
            # Keep only IDs that match the pattern
            filtered_ids = [id for id in self.ids if compiled_pattern.search(id)]
        else:
            # Keep only IDs that do NOT match the pattern
            filtered_ids = [id for id in self.ids if not compiled_pattern.search(id)]

        if len(filtered_ids) == len(self.ids):
            logging.warning(
                "leave_out_celltypes_with_pattern resulted in no change to the ID list."
            )
        elif not filtered_ids:
            logging.warning(
                "leave_out_celltypes_with_pattern resulted in an empty ID list."
            )

        # The `subset` method is abstract and must be implemented by subclasses.
        return self.subset(filtered_ids)

    def subset_celltypes_with_data_name(
        self, data_name: str = "peaks_q0.01_tissue_open_exp"
    ) -> CelltypeDenseIO:
        """Create a subset instance containing only groups that have a specific data key.

        Filters the current group IDs based on the presence of `data_name` as a
        key within an assumed underlying data structure (`self.dataset`).

        Note: This method relies on a non-standard `self.dataset` attribute,
        likely specific to certain backend implementations (e.g., Zarr). Its
        functionality depends heavily on how the backend stores data per group ID.

        Args:
            data_name: The key to check for within each group's data structure.

        Returns:
            A new `CelltypeDenseIO` instance containing only groups possessing
            the specified `data_name`.

        Raises:
            AttributeError: If `self.dataset` does not exist or does not behave
                            like a dictionary mapping IDs to objects with a `keys()` method.
            NotImplementedError: If the `subset` method hasn't been implemented.
        """
        celltype_ids = []
        try:
            # This assumes self.dataset is a dict-like structure mapping ids
            # to objects that have a .keys() method (like Zarr groups).
            # This is backend-specific and might fail.
            if not hasattr(self, "dataset"):
                raise AttributeError(
                    "This method requires a 'dataset' attribute, which is not defined in the base class or currently available."
                )

            for id_ in self.ids:
                # Check if the id exists as a key in dataset and if data_name is in its keys
                if id_ in self.dataset and data_name in self.dataset[id_]:  # type: ignore
                    celltype_ids.append(id_)
        except AttributeError as e:
            logging.error(
                f"Failed to check for data_name '{data_name}'. Does 'self.dataset' exist and support key checking per ID? Error: {e}"
            )
            raise  # Re-raise the error as it indicates a fundamental incompatibility
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while checking for data_name '{data_name}': {e}"
            )
            # Decide whether to raise or return an empty subset
            return self.subset([])  # Return empty subset on other errors

        if not celltype_ids:
            logging.warning(
                f"No cell types found containing the data key: '{data_name}'"
            )

        return self.subset(celltype_ids)

    def filter_by_min_depth(self, cutoff: float = 3000000) -> CelltypeDenseIO:
        """Create a subset instance including only groups meeting a minimum library size.

        Filters the current group IDs based on their calculated library sizes stored
        in the `self.libsize` dictionary. Only groups with a library size greater
        than or equal to the `cutoff` are included in the resulting subset.

        Args:
            cutoff: The minimum library size (total signal count) required for
                    a group to be included.

        Returns:
            A new `CelltypeDenseIO` instance containing only groups that meet
            the minimum depth requirement.

        Raises:
            TypeError: If `self.libsize` is not a dictionary (e.g., if it's a
                       single value for a non-grouped dataset, though unlikely here).
            NotImplementedError: If the `subset` method hasn't been implemented.
        """
        if not isinstance(self.libsize, dict):
            logging.error(
                f"Cannot filter by min depth because self.libsize is not a dictionary (type: {type(self.libsize)}). "
                f"Filtering requires per-group library sizes."
            )
            # Return self or raise error? Returning self to avoid breaking workflows, but log error.
            return self  # Or raise TypeError("self.libsize must be a dictionary for filter_by_min_depth")

        # Filter IDs based on the library size stored in self.libsize
        filtered_ids = [
            id for id, size in self.libsize.items() if id in self.ids and size >= cutoff
        ]

        if len(filtered_ids) == len(self.ids):
            logging.warning("filter_by_min_depth resulted in no change to the ID list.")
        elif not filtered_ids:
            logging.warning(
                f"filter_by_min_depth with cutoff {cutoff} resulted in an empty ID list."
            )

        return self.subset(filtered_ids)
