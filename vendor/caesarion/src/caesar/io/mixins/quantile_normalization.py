"""Quantile normalization mixin for genomic signal data."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyranges as pr
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from gcell.dna.genome import ChromSize, read_blacklist

# Define compression for zarr storage
COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")


class QuantileNormalizationMixin:
    """
    A mixin class that provides methods for peak-based quantile normalization of genomic signal data.

    This mixin implements a comprehensive approach to normalize genomic signal data by:
    1. Identifying peak and non-peak regions across the genome
    2. Computing background levels from non-peak regions
    3. Determining scale factors from peak regions (using high percentiles)
    4. Applying normalization formula: (signal - background) / scale_factor

    The normalization process helps reduce technical biases and makes signals comparable
    across different samples, cell types, or experimental conditions.

    Requirements:
        - The child class must implement or provide:
            1) self.assembly (str): Name of the genome assembly (e.g., 'hg38').
            2) self.chunk_size (int): Size of data chunks.
            3) self.chroms (list): List of chromosome names.
            4) self.chrom_n_chunks (dict): Mapping from chrom -> number_of_chunks.
            5) self.chrom_sizes (dict): Mapping from chrom -> chromosome size.
            6) self.get_track(chr_name, start, end, query, aggregator) -> np.ndarray:
               A method to get the coverage/track data for a genomic region.
            7) self._get_all_ids() -> list or dict: Method to get all track identifiers.

    Usage:
        - Inherit from this mixin in your Zarr I/O classes
        - Call `quantile_normalization_peak_based(...)` to perform normalization
        - Optionally save normalized data to a new Zarr store
    """

    def _get_peak_regions(self, peak_file, blacklist_file=None):
        """
        Process peak and blacklist files to get peak and non-peak regions.

        Args:
            peak_file (str): Path to a BED file containing peak regions.
            blacklist_file (str, optional): Path to a BED file containing blacklist regions.

        Returns:
            tuple: (peaks, non_peaks) as PyRanges objects
        """
        # Read the peak file
        peaks = pr.read_bed(peak_file)

        # Read the blacklist file (if provided)
        if blacklist_file is not None:
            blacklist = pr.PyRanges(blacklist_file)
        else:
            try:
                blacklist = pr.PyRanges(read_blacklist(self.assembly))
            except Exception:
                blacklist = pr.PyRanges(
                    pd.DataFrame(columns=["Chromosome", "Start", "End"])
                )

        # Merge peaks and blacklist
        combined = pr.concat([peaks, blacklist], strand=False).merge()

        # Subtract out peaks/blacklist from the full chromosome sizes
        chrom_size = ChromSize(self.assembly).as_pyranges()
        non_peaks = chrom_size.subtract(combined)

        return peaks, non_peaks

    def _process_chunk_quantile_normalization_peak_based(self, args):
        """
        Internal method run in parallel on each chunk.

        Args:
            args: A tuple of
                  (chr_name, chunk_idx, track_kwargs, peaks_chr, non_peaks_chr)
        Returns:
            peak_mean_list (list of float): The average coverage per peak region in this chunk.
            non_peak_sum (float): Sum of average coverage in the non-peak regions of this chunk.
            non_peak_count (int): How many non-peak regions were encountered in this chunk.
        """
        chr_name, chunk_idx, conv_size, track_kwargs, peaks_chr, non_peaks_chr = args

        chunk_start = chunk_idx * self.chunk_size
        chunk_end = (chunk_idx + 1) * self.chunk_size
        chrom_size = self.chrom_sizes[chr_name]
        if chunk_end > chrom_size:
            chunk_end = chrom_size

        # Get track data for this chunk
        track = self._get_smoothed_track(
            chr_name, chunk_start, chunk_end, conv_size, **track_kwargs
        )

        # Process peak regions
        peak_mean_list = self._compute_peak_means(
            peaks_chr, chunk_start, chunk_end, track
        )

        # Process non-peak regions
        non_peak_sum, non_peak_count = self._compute_non_peak_stats(
            non_peaks_chr, chunk_start, chunk_end, track
        )

        return peak_mean_list, non_peak_sum, non_peak_count

    def _compute_peak_means(self, peaks_chr, chunk_start, chunk_end, track):
        """Compute mean coverage in peak regions."""
        peak_mean_list = []
        for _, row in peaks_chr.iterrows():
            start, end = row.Start, row.End
            # Overlapping region check with the chunk
            if start < chunk_end and end > chunk_start:
                rel_start = max(0, start - chunk_start)
                rel_end = min(self.chunk_size, end - chunk_start)
                peak_mean_list.append(track[rel_start:rel_end].mean(0))
        return peak_mean_list

    def _compute_non_peak_stats(self, non_peaks_chr, chunk_start, chunk_end, track):
        """Compute statistics for non-peak regions."""
        non_peak_sum = 0.0
        non_peak_count = 0
        for _, row in non_peaks_chr.iterrows():
            start, end = row.Start, row.End
            # Overlapping region check with the chunk
            if start < chunk_end and end > chunk_start:
                rel_start = max(0, start - chunk_start)
                rel_end = min(self.chunk_size, end - chunk_start)
                non_peak_sum += track[rel_start:rel_end].mean(0)
                non_peak_count += 1
        return non_peak_sum, non_peak_count

    def _prepare_chunk_tasks(self, peaks, non_peaks, conv_size, track_kwargs):
        """Prepare tasks for parallel processing of chunks."""
        args_list = []
        for chr_name in self.chroms:
            peaks_chr = peaks.df.query("Chromosome == @chr_name")
            non_peaks_chr = non_peaks.df.query("Chromosome == @chr_name")

            n_chunks = self.chrom_n_chunks[chr_name]
            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * self.chunk_size
                chunk_end = (chunk_idx + 1) * self.chunk_size
                if chunk_end > self.chrom_sizes[chr_name]:
                    chunk_end = self.chrom_sizes[chr_name]

                peaks_chr_i = peaks_chr[
                    (peaks_chr["Start"] < chunk_end) & (peaks_chr["End"] > chunk_start)
                ]
                non_peaks_chr_i = non_peaks_chr[
                    (non_peaks_chr["Start"] < chunk_end)
                    & (non_peaks_chr["End"] > chunk_start)
                ]
                args_list.append(
                    (
                        chr_name,
                        chunk_idx,
                        conv_size,
                        track_kwargs,
                        peaks_chr_i,
                        non_peaks_chr_i,
                    )
                )
        return args_list

    def _calculate_normalization_factors(
        self, peak_mean_lists, non_peak_sums, non_peak_counts, all_ids
    ):
        """Calculate background and scale factors from collected data."""
        # Mean coverage in non-peak regions
        if non_peak_counts == 0:
            # Edge case: no non-peaks or no data
            # Create zero array with shape matching the number of IDs
            background = np.zeros(len(all_ids))
        else:
            background = non_peak_sums / non_peak_counts

        # Derive a percentile from the peak coverage distribution
        pm_array = np.stack(peak_mean_lists)
        print(pm_array.shape)
        if len(pm_array) == 0:
            scale_factor = np.zeros(len(all_ids))
        else:
            # E.g. average from 99th to 99.9th percentile
            quantiles = np.percentile(pm_array, np.arange(99, 99.9, 0.1), axis=0)
            scale_factor = quantiles.mean(0)

        print("scale_factor: ", scale_factor)
        print("background: ", background)

        return background, scale_factor

    def _prepare_output_zarr(self, output_zarr_path, all_ids):
        """Prepare the output Zarr store structure."""
        # Use zarr_format=2 for backward compatibility
        root = zarr.open_group(output_zarr_path, mode="w", zarr_format=2)

        # Copy root attributes from source zarr
        if hasattr(self.dataset, "attrs"):
            root.attrs.update(self.dataset.attrs)

        for identifier in all_ids:
            root.create_group(identifier)
            root[identifier].create_group("chrs")

            # Copy group attributes if they exist
            if identifier in self.dataset and hasattr(
                self.dataset[identifier], "attrs"
            ):
                root[identifier].attrs.update(self.dataset[identifier].attrs)

            for chrom in self.chroms:
                root[identifier]["chrs"].create_array(
                    chrom,
                    shape=(self.chrom_sizes[chrom],),
                    dtype="float32",
                    chunks=self.chunk_size,
                    compressors=COMPRESSOR,
                )

        # Ensure essential attributes are set
        if "chrs" not in root.attrs:
            root.attrs["chrs"] = self.chroms

        # Copy assembly info if available
        if hasattr(self, "assembly"):
            root.attrs["assembly"] = self.assembly
        elif "assembly" in self.dataset.attrs:
            root.attrs["assembly"] = self.dataset.attrs["assembly"]

        return root

    def _prepare_normalization_tasks(
        self, root, all_ids, background, scale_factor, conv_size
    ):
        """Prepare tasks for parallel normalization of chunks."""
        arg_list = []
        for chrom in self.chroms:
            for start in range(0, self.chrom_sizes[chrom], self.chunk_size):
                end = min(start + self.chunk_size, self.chrom_sizes[chrom])
                arg_list.append(
                    (
                        root,
                        all_ids,
                        chrom,
                        start,
                        end,
                        conv_size,
                        background,
                        scale_factor,
                    )
                )
        return arg_list

    def quantile_normalization_peak_based(
        self,
        peak_file: str,
        blacklist_file: str = None,
        n_jobs: int = 16,
        conv_size: int = 20,
        output_zarr_path: str | None = None,
        **track_kwargs,
    ):
        """
        Perform quantile normalization on genomic signal data based on peak and non-peak regions.

        This method implements a comprehensive normalization approach that:
        1. Identifies peak regions from the provided peak file
        2. Excludes blacklisted regions (if provided)
        3. Computes background levels from non-peak regions
        4. Calculates scale factors from high percentiles of peak region signals
        5. Applies the normalization formula: (signal - background) / scale_factor
        6. Optionally saves normalized data to a new Zarr store

        The normalization helps reduce technical biases and makes signals comparable
        across different samples, cell types, or experimental conditions.

        Parameters
        ----------
        peak_file : str
            Path to a BED file containing peak regions used for normalization
        blacklist_file : str, optional
            Path to a BED file containing regions to exclude from analysis.
            If None, will attempt to download standard blacklist for the assembly.
        n_jobs : int, default=16
            Number of parallel processes to use for computation
        conv_size : int, default=20
            Size of convolution window for signal smoothing
        output_zarr_path : str, optional
            Path to save normalized data as a Zarr store.
            If None, normalization factors are returned but data is not saved.
        **track_kwargs
            Additional arguments passed to get_track() method, including:
            - query: Specify which track(s) to normalize

        Returns
        -------
        dict
            A dictionary with normalization parameters:
            - "background": dict mapping track IDs to background values
            - "scale_factor": dict mapping track IDs to scale factors

        Notes
        -----
        - Background is calculated as the mean signal in non-peak regions
        - Scale factor is calculated as the mean of 99th-99.9th percentiles in peak regions
        - The normalization process handles each chromosome in chunks for memory efficiency
        - If output_zarr_path is provided, normalized data will be saved with the same
          structure as the original data, but with normalization applied
        """
        # Get track identifiers
        if track_kwargs.get("query") is not None:
            all_ids = track_kwargs["query"]
        else:
            all_ids = self._get_all_ids()
        if isinstance(all_ids, dict):
            all_ids = list(all_ids.keys())
        elif not isinstance(all_ids, list):
            all_ids = [all_ids]

        # Process peak and blacklist files
        peaks, non_peaks = self._get_peak_regions(peak_file, blacklist_file)

        # Prepare chunk tasks
        args_list = self._prepare_chunk_tasks(peaks, non_peaks, conv_size, track_kwargs)

        # For storing results
        peak_mean_lists = []
        non_peak_sums = 0
        non_peak_counts = 0

        # Execute in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self._process_chunk_quantile_normalization_peak_based, args_list
                    ),
                    total=len(args_list),
                    desc="Calculating normalization factors",
                )
            )

        # Aggregate results
        for peak_mean_list, non_peak_sum, non_peak_count in results:
            peak_mean_lists.extend(peak_mean_list)
            non_peak_sums += non_peak_sum
            non_peak_counts += non_peak_count

        # Calculate normalization factors
        background, scale_factor = self._calculate_normalization_factors(
            peak_mean_lists, non_peak_sums, non_peak_counts, all_ids
        )

        # Save normalized data if output path is provided
        if output_zarr_path is not None:
            # Prepare output Zarr structure
            root = self._prepare_output_zarr(output_zarr_path, all_ids)

            # Prepare normalization tasks
            arg_list = self._prepare_normalization_tasks(
                root, all_ids, background, scale_factor, conv_size
            )

            # Apply normalization to each chunk in parallel
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                list(
                    tqdm(
                        executor.map(self._apply_normalization_chunk, arg_list),
                        total=len(arg_list),
                        desc="Applying normalization",
                    )
                )

        background_dict = {id: background[i] for i, id in enumerate(all_ids)}
        scale_factor_dict = {id: scale_factor[i] for i, id in enumerate(all_ids)}
        norm_factor = {
            "background": background_dict,
            "scale_factor": scale_factor_dict,
        }
        # Save normalization parameters as attrs
        if output_zarr_path is not None:
            root.attrs["normalization_params"] = norm_factor
        
        # Save normalization params - use _save_attr if available (for BPCells), otherwise use dataset.attrs (for Zarr)
        if hasattr(self, "_save_attr"):
            # BPCells backend: use _save_attr which saves to attrs.json
            self._save_attr("normalization_params", norm_factor)
        elif hasattr(self, "dataset") and hasattr(self.dataset, "attrs"):
            # Zarr backend: use dataset.attrs directly
            self.dataset.attrs["normalization_params"] = norm_factor
            # for CelltypeDenseZarrIO - also save per-track normalization params
            for id in all_ids:
                if hasattr(self.dataset, "__getitem__") and id in self.dataset:
                    try:
                        norm_factor_for_id = {
                            "background": background_dict[id],
                            "scale_factor": scale_factor_dict[id],
                        }
                        if hasattr(self.dataset[id], "attrs"):
                            self.dataset[id].attrs["normalization_params"] = norm_factor_for_id
                    except (TypeError, AttributeError):
                        # Skip if dataset[id] doesn't support attrs (e.g., BPCells)
                        pass
        else:
            # Fallback: try to use _save_attr if it exists
            if hasattr(self, "_save_attr"):
                self._save_attr("normalization_params", norm_factor)
            else:
                logging.warning(
                    "Could not save normalization_params: "
                    "IO object doesn't have _save_attr or dataset.attrs"
                )
        return norm_factor

    def _apply_normalization_chunk(
        self,
        args: tuple,
    ) -> None:
        """
        Apply normalization to a single chunk of data.

        Args:
            args (tuple): A tuple containing:
                - root: Zarr root group
                - identifier: Track identifier
                - chrom: Chromosome name
                - start: Start position
                - end: End position
                - conv_size: Convolution window size
                - background: Background values array
                - scale_factor: Scale factor array

        The normalization formula applied is:
            normalized_signal = (signal - background) / scale_factor
        """
        # Get signal
        root, all_ids, chrom, start, end, conv_size, background, scale_factor = args
        signal = self._get_smoothed_track(chrom, start, end, conv_size=conv_size)

        # Apply normalization
        signal = signal - background.reshape(1, -1)
        signal = signal / scale_factor.reshape(1, -1)

        # Store normalized signal
        for i, identifier in enumerate(all_ids):
            root[identifier]["chrs"][chrom][start:end] = signal[:, i]

