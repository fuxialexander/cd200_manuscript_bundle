"""Library size annotation mixin for genomic IO classes."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm
from zarr.codecs import BloscCodec

from gcell.dna.genome import read_blacklist

from ..aggregator import ByKeyAggregator

COMPRESSOR = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")


class LibrarySizeAnnotationMixin:
    """
    Mixin class for annotating zarr stores with library size calculations that account for blacklist regions.

    This mixin calculates three library size metrics:
    - library_size_total: Total signal across all regions
    - library_size_blacklist: Signal in blacklisted regions
    - library_size_nonblacklist: Signal in non-blacklisted regions (default)

    The calculations are performed on existing dense zarr data, allowing for
    post-hoc correction without reprocessing raw fragments.
    """

    def _calculate_library_sizes_with_blacklist(
        self,
        blacklist_file: str | None = None,
        n_jobs: int = 16,
        chunk_size: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Calculate library sizes accounting for blacklist regions.

        This method uses an optimized approach similar to BPCells:
        1. Reads pre-computed total library sizes from zarr attributes
        2. Only queries blacklist regions from zarr arrays (chunked if needed)
        3. Calculates non-blacklist as total - blacklist

        Parameters
        ----------
        blacklist_file : str, optional
            Path to blacklist BED file. If None, uses assembly-specific blacklist.
        n_jobs : int, default=16
            Number of parallel processes to use.
        chunk_size : int, optional
            Chunk size for processing. If None, uses self.chunk_size.

        Returns
        -------
        dict
            Nested dictionary with structure:
            {track_id: {'total': float, 'blacklist': float, 'nonblacklist': float}}
        """
        # Load blacklist regions
        if blacklist_file is not None:
            blacklist = pd.read_csv(blacklist_file, sep="\t", header=None)
            blacklist.columns = ["Chromosome", "Start", "End"] + list(
                blacklist.columns[3:]
            )
        else:
            try:
                blacklist = read_blacklist(self.assembly)
            except Exception:
                logging.warning(
                    f"Could not load blacklist for assembly {self.assembly}. Using empty blacklist."
                )
                blacklist = pd.DataFrame(columns=["Chromosome", "Start", "End"])

        # Get all track IDs to process
        all_ids = self._get_all_ids()
        if isinstance(all_ids, dict):
            all_ids = list(all_ids.keys())
        elif not isinstance(all_ids, list):
            all_ids = [all_ids]

        # Try to load pre-computed total library sizes from attributes
        total_library_sizes = {}
        
        # Check for single-track format (SignalDenseZarrIO)
        # Single-track stores library_size at root level
        if "library_size" in self.dataset.attrs:
            # Single-track format
            if len(all_ids) == 1:
                track_id = all_ids[0]
                total_library_sizes[track_id] = float(self.dataset.attrs["library_size"])
            else:
                # If multiple tracks but library_size at root, use for all tracks
                # (unlikely but handle gracefully)
                for track_id in all_ids:
                    total_library_sizes[track_id] = float(self.dataset.attrs["library_size"])
        elif "libsize" in self.dataset.attrs:
            # Old attribute name
            if len(all_ids) == 1:
                track_id = all_ids[0]
                total_library_sizes[track_id] = float(self.dataset.attrs["libsize"])
            else:
                for track_id in all_ids:
                    total_library_sizes[track_id] = float(self.dataset.attrs["libsize"])
        else:
            # Multi-track format (CelltypeDenseZarrIO)
            # Each track stores library_size in its own group
            for track_id in all_ids:
                if track_id in self.dataset:
                    if "library_size" in self.dataset[track_id].attrs:
                        total_library_sizes[track_id] = float(self.dataset[track_id].attrs["library_size"])
                    elif "libsize" in self.dataset[track_id].attrs:
                        total_library_sizes[track_id] = float(self.dataset[track_id].attrs["libsize"])

        # If we couldn't load pre-computed library sizes, fall back to calculating from all data
        if not total_library_sizes or len(total_library_sizes) != len(all_ids):
            logging.warning(
                "Pre-computed library sizes not found in attributes. "
                "Falling back to calculating from all zarr data (slower)."
            )
            return self._calculate_library_sizes_from_all_data(
                blacklist, all_ids, n_jobs, chunk_size
            )

        logging.info(
            f"Using pre-computed library sizes from attributes. "
            f"Only querying {len(blacklist)} blacklist regions."
        )

        # Filter blacklist to only chromosomes present in dataset
        available_chroms = set(self.chroms) if hasattr(self, "chroms") else set()
        if available_chroms:
            blacklist = blacklist[blacklist["Chromosome"].isin(available_chroms)].copy()

        # Calculate blacklist library sizes by only querying blacklist regions
        blacklist_library_sizes = self._calculate_blacklist_library_sizes(
            blacklist, all_ids, n_jobs, chunk_size
        )

        # Calculate non-blacklist library sizes
        results = {}
        for track_id in all_ids:
            total = total_library_sizes.get(track_id, 0.0)
            blacklist_size = blacklist_library_sizes.get(track_id, 0.0)
            nonblacklist = total - blacklist_size

            results[track_id] = {
                "total": total,
                "blacklist": blacklist_size,
                "nonblacklist": nonblacklist,
            }

        return results

    def _calculate_blacklist_library_sizes(
        self,
        blacklist: pd.DataFrame,
        all_ids: list[str],
        n_jobs: int = 16,
        chunk_size: int | None = None,
    ) -> dict[str, float]:
        """
        Calculate library sizes for blacklist regions only.

        Parameters
        ----------
        blacklist : pd.DataFrame
            DataFrame with columns ["Chromosome", "Start", "End"]
        all_ids : list[str]
            List of track IDs to process
        n_jobs : int
            Number of parallel processes
        chunk_size : int, optional
            Chunk size for processing

        Returns
        -------
        dict[str, float]
            Dictionary mapping track_id to blacklist library size
        """
        if len(blacklist) == 0:
            logging.info("No blacklist regions provided. Blacklist library sizes will be zero.")
            return {track_id: 0.0 for track_id in all_ids}

        # Use default chunk size if not provided
        if chunk_size is None:
            chunk_size = self.chunk_size

        # Handle edge case: no valid chunk_size
        if chunk_size is None or chunk_size <= 0:
            logging.warning(
                "Invalid or missing chunk_size for blacklist calculation. "
                "Using fallback of 1000000."
            )
            chunk_size = 1000000

        # Initialize results
        blacklist_sizes = {track_id: 0.0 for track_id in all_ids}

        # Prepare tasks for parallel processing - only for blacklist regions
        tasks = []
        for chrom in blacklist["Chromosome"].unique():
            chrom_blacklist = blacklist[blacklist["Chromosome"] == chrom]
            if chrom not in self.chroms:
                continue

            # Process each blacklist region
            for _, region in chrom_blacklist.iterrows():
                start = region["Start"]
                end = region["End"]
                # Skip invalid regions
                if end <= start:
                    continue
                # Split large regions into chunks if needed
                for chunk_start in range(start, end, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, end)
                    tasks.append((chrom, chunk_start, chunk_end, all_ids))

        if not tasks:
            return blacklist_sizes

        logging.info(f"Querying {len(tasks)} blacklist region chunks...")

        # Process blacklist regions in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self._process_blacklist_chunk_library_sizes, task)
                for task in tasks
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Calculating blacklist library sizes",
            ):
                chunk_results = future.result()
                # Aggregate results
                for track_id, value in chunk_results.items():
                    blacklist_sizes[track_id] += value

        logging.info(
            f"Calculated blacklist library sizes: "
            f"min={min(blacklist_sizes.values()):.0f}, "
            f"max={max(blacklist_sizes.values()):.0f}, "
            f"mean={sum(blacklist_sizes.values())/len(blacklist_sizes):.0f}"
        )

        return blacklist_sizes

    def _process_blacklist_chunk_library_sizes(self, args) -> dict[str, float]:
        """
        Process a single blacklist region chunk to calculate library sizes.

        Parameters
        ----------
        args : tuple
            (chrom, chunk_start, chunk_end, track_ids)

        Returns
        -------
        dict[str, float]
            Dictionary mapping track_id to blacklist library size for this chunk
        """
        chrom, chunk_start, chunk_end, track_ids = args

        try:
            # Get track data for this blacklist region only
            if hasattr(self, "_is_single_track") or (
                isinstance(track_ids, list) and track_ids == ["default"]
            ):
                track_data = self.get_track(
                    chrom, chunk_start, chunk_end, output_format="raw_dict"
                )
            else:
                track_data = self.get_track(
                    chrom,
                    chunk_start,
                    chunk_end,
                    query=track_ids,
                    aggregator=ByKeyAggregator(),
                    output_format="raw_dict",
                )
        except (KeyError, Exception) as e:
            # Return zero counts if we can't read this region
            logging.debug(
                f"Could not read blacklist region {chrom}:{chunk_start}-{chunk_end}: {e}"
            )
            if isinstance(track_ids, list):
                return {track_id: 0.0 for track_id in track_ids}
            else:
                return {"default": 0.0}

        # Calculate library sizes for each track
        results = {}

        # Handle single-track case
        if "aggregated_sum" in track_data and len(track_ids) == 1:
            signal = track_data["aggregated_sum"].astype(np.float32)
            track_id = track_ids[0]

            if isinstance(signal, csr_matrix):
                signal = signal.toarray().flatten()
            elif signal.ndim > 1:
                signal = signal.flatten()

            results[track_id] = float(np.sum(signal.astype(np.float32)))
        else:
            # Multi-track case
            for track_id, signal in track_data.items():
                if isinstance(signal, csr_matrix):
                    signal = signal.toarray().flatten().astype(np.float32)
                elif signal.ndim > 1:
                    signal = signal.flatten().astype(np.float32)

                results[track_id] = float(np.sum(signal.astype(np.float32)))

        return results

    def _calculate_library_sizes_from_all_data(
        self,
        blacklist: pd.DataFrame,
        all_ids: list[str],
        n_jobs: int = 16,
        chunk_size: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Fallback method: Calculate library sizes by reading all data (original implementation).

        This is used when pre-computed library sizes are not available in attributes.
        """
        # Initialize results
        results = {
            track_id: {"total": 0.0, "blacklist": 0.0, "nonblacklist": 0.0}
            for track_id in all_ids
        }

        # Use default chunk size if not provided
        if chunk_size is None:
            chunk_size = self.chunk_size

        # Handle edge case: no valid chunk_size or empty data
        if chunk_size is None or chunk_size <= 0:
            logging.warning(
                "Invalid or missing chunk_size. Using fallback of 1000000 for library size calculation."
            )
            chunk_size = 1000000

        # Prepare tasks for parallel processing
        tasks = []
        total_chunks = 0
        for chrom in self.chroms:
            chrom_blacklist = blacklist[blacklist["Chromosome"] == chrom]
            chrom_size = self.chrom_sizes.get(chrom, 0)

            # Skip empty chromosomes
            if chrom_size <= 0:
                continue

            for chunk_start in range(0, chrom_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, chrom_size)
                tasks.append((chrom, chunk_start, chunk_end, chrom_blacklist, all_ids))
                total_chunks += 1

        logging.info(f"Processing {total_chunks} chunks with chunk_size={chunk_size}")

        # Handle edge case: no tasks to process (all chromosomes empty or no data)
        if total_chunks == 0:
            logging.info("No data to process. Returning zero library sizes.")
            return results

        # Process chunks in parallel
        skipped_chunks = 0
        processed_chunks = 0

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self._process_chunk_library_sizes, task)
                for task in tasks
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Calculating library sizes",
            ):
                chunk_results = future.result()

                # Check if this chunk was processed successfully
                if chunk_results and any(
                    metrics.get("total", 0) > 0 for metrics in chunk_results.values()
                ):
                    processed_chunks += 1
                else:
                    # Check if all values are zero (likely skipped)
                    all_zero = all(
                        all(v == 0.0 for v in metrics.values())
                        for metrics in chunk_results.values()
                    )
                    if all_zero:
                        skipped_chunks += 1
                    else:
                        processed_chunks += 1

                # Aggregate results
                for track_id, metrics in chunk_results.items():
                    for metric, value in metrics.items():
                        results[track_id][metric] += value

        if skipped_chunks > 0:
            logging.warning(
                f"Library size calculation completed with {skipped_chunks} chunks skipped "
                f"due to missing data. Processed {processed_chunks}/{total_chunks} chunks successfully."
            )
        else:
            logging.info(
                f"Library size calculation completed successfully. "
                f"Processed all {processed_chunks} chunks."
            )

        return results

    def _process_chunk_library_sizes(self, args):
        """
        Process a single chunk to calculate library sizes.

        Parameters
        ----------
        args : tuple
            (chrom, chunk_start, chunk_end, chrom_blacklist, track_ids)

        Returns
        -------
        dict
            Chunk-level library size calculations
        """
        chrom, chunk_start, chunk_end, chrom_blacklist, track_ids = args

        try:
            # Get track data for this chunk - using raw_dict format
            # For DenseIO (single track), we don't need to specify track_ids
            if hasattr(self, "_is_single_track") or (
                isinstance(track_ids, list) and track_ids == ["default"]
            ):
                track_data = self.get_track(
                    chrom, chunk_start, chunk_end, output_format="raw_dict"
                )
            else:
                track_data = self.get_track(
                    chrom,
                    chunk_start,
                    chunk_end,
                    query=track_ids,
                    aggregator=ByKeyAggregator(),
                    output_format="raw_dict",
                )
        except KeyError as e:
            # Handle specific KeyError for missing chunks
            logging.warning(
                f"Missing chunk in Zarr data for {chrom}:{chunk_start}-{chunk_end}: {e}. "
                f"This chunk will be skipped in library size calculation."
            )
            # Return zero counts for this chunk if we can't read it
            results = {}
            for track_id in track_ids if isinstance(track_ids, list) else ["default"]:
                results[track_id] = {
                    "total": 0.0,
                    "blacklist": 0.0,
                    "nonblacklist": 0.0,
                }
            return results
        except Exception as e:
            # Handle other exceptions
            logging.error(
                f"Unexpected error reading track data for {chrom}:{chunk_start}-{chunk_end}: {type(e).__name__}: {e}"
            )
            # Return zero counts for this chunk if we can't read it
            results = {}
            for track_id in track_ids if isinstance(track_ids, list) else ["default"]:
                results[track_id] = {
                    "total": 0.0,
                    "blacklist": 0.0,
                    "nonblacklist": 0.0,
                }
            return results

        # Create blacklist mask for this chunk
        chunk_length = chunk_end - chunk_start
        blacklist_mask = np.zeros(chunk_length, dtype=bool)

        for _, region in chrom_blacklist.iterrows():
            # Calculate overlap with chunk
            overlap_start = max(0, region["Start"] - chunk_start)
            overlap_end = min(chunk_length, region["End"] - chunk_start)
            if overlap_start < overlap_end:
                blacklist_mask[overlap_start:overlap_end] = True

        # Calculate metrics for each track
        results = {}

        # Handle single-track case where data is wrapped as 'aggregated_sum'
        if "aggregated_sum" in track_data and len(track_ids) == 1:
            signal = track_data["aggregated_sum"].astype(np.float32)
            track_id = track_ids[0]

            if isinstance(signal, csr_matrix):
                signal = signal.toarray().flatten()
            elif signal.ndim > 1:
                signal = signal.flatten()

            # Check for empty signal
            if signal.size == 0:
                results[track_id] = {
                    "total": 0.0,
                    "blacklist": 0.0,
                    "nonblacklist": 0.0,
                }
            else:
                # Adjust blacklist mask to match signal size
                if signal.size < blacklist_mask.size:
                    blacklist_mask_adjusted = blacklist_mask[: signal.size]
                else:
                    blacklist_mask_adjusted = blacklist_mask

                total = float(np.sum(signal.astype(np.float32)))
                blacklist = float(np.sum(signal[blacklist_mask_adjusted].astype(np.float32)))
                nonblacklist = total - blacklist

                results[track_id] = {
                    "total": total,
                    "blacklist": blacklist,
                    "nonblacklist": nonblacklist,
                }
        else:
            # Multi-track case
            for track_id, signal in track_data.items():
                if isinstance(signal, csr_matrix):
                    signal = signal.toarray().flatten().astype(np.float32)
                elif signal.ndim > 1:
                    signal = signal.flatten().astype(np.float32)

                # Check for empty signal
                if signal.size == 0:
                    results[track_id] = {
                        "total": 0.0,
                        "blacklist": 0.0,
                        "nonblacklist": 0.0,
                    }
                    continue

                # Adjust blacklist mask to match signal size
                if signal.size < blacklist_mask.size:
                    blacklist_mask_adjusted = blacklist_mask[: signal.size]
                else:
                    blacklist_mask_adjusted = blacklist_mask

                total = float(np.sum(signal.astype(np.float32)))
                blacklist = float(np.sum(signal[blacklist_mask_adjusted].astype(np.float32)))
                nonblacklist = total - blacklist

                results[track_id] = {
                    "total": total,
                    "blacklist": blacklist,
                    "nonblacklist": nonblacklist,
                }

        return results

    def annotate_library_sizes(
        self,
        blacklist_file: str | None = None,
        n_jobs: int = 16,
        overwrite: bool = True,
    ) -> dict[str, dict[str, float]]:
        """
        Annotate zarr store with library size calculations accounting for blacklist.

        Parameters
        ----------
        blacklist_file : str, optional
            Path to blacklist BED file. If None, uses assembly-specific blacklist.
        n_jobs : int, default=16
            Number of parallel processes.
        overwrite : bool, default=True
            Whether to overwrite existing library size annotations.

        Returns
        -------
        dict
            Library size results for all tracks
        """
        if self.mode not in ["a", "r+", "w"]:
            raise ValueError("Cannot annotate in read-only mode")

        # Check if already annotated
        if not overwrite and self._load_attr("library_size_nonblacklist"):
            logging.info(
                "Library sizes already annotated. Use overwrite=True to recalculate."
            )
            return self._load_library_size_annotations()

        logging.info("Calculating library sizes with blacklist annotation...")
        results = self._calculate_library_sizes_with_blacklist(
            blacklist_file=blacklist_file, n_jobs=n_jobs
        )

        # Store results as attributes
        # Convert all values to Python float to ensure JSON serialization works in zarr v3
        # For single-track IO classes
        if len(results) == 1 and "default" in results:
            for metric, value in results["default"].items():
                self.dataset.attrs[f"library_size_{metric}"] = float(value)
        else:
            # For multi-track IO classes (like CelltypeDenseZarrIO)
            # Store at both dataset level and individual track level
            self.dataset.attrs["library_size_total"] = {
                k: float(v["total"]) for k, v in results.items()
            }
            self.dataset.attrs["library_size_blacklist"] = {
                k: float(v["blacklist"]) for k, v in results.items()
            }
            self.dataset.attrs["library_size_nonblacklist"] = {
                k: float(v["nonblacklist"]) for k, v in results.items()
            }

            # Also store at individual group level if applicable
            for track_id, metrics in results.items():
                if track_id in self.dataset:
                    for metric, value in metrics.items():
                        self.dataset[track_id].attrs[f"library_size_{metric}"] = float(value)
                    # Also set the simple library_size attribute that CelltypeDenseZarrIO expects
                    self.dataset[track_id].attrs["library_size"] = float(metrics[
                        "nonblacklist"
                    ])

        self.dataset.attrs["library_size_annotated"] = True
        self.dataset.attrs["library_size_annotation_date"] = (
            pd.Timestamp.now().isoformat()
        )

        logging.info("Library size annotation complete.")
        return results

    def _load_library_size_annotations(self) -> dict[str, dict[str, float]]:
        """Load existing library size annotations from zarr attributes."""
        results = {}

        # Check for single-track format
        if "library_size_total" in self.dataset.attrs and not isinstance(
            self.dataset.attrs["library_size_total"], dict
        ):
            results["default"] = {
                "total": self.dataset.attrs.get("library_size_total", 0),
                "blacklist": self.dataset.attrs.get("library_size_blacklist", 0),
                "nonblacklist": self.dataset.attrs.get("library_size_nonblacklist", 0),
            }
        # Check for multi-track format
        elif "library_size_nonblacklist" in self.dataset.attrs and isinstance(
            self.dataset.attrs["library_size_nonblacklist"], dict
        ):
            for track_id in self.dataset.attrs["library_size_nonblacklist"]:
                results[track_id] = {
                    "total": self.dataset.attrs["library_size_total"].get(track_id, 0),
                    "blacklist": self.dataset.attrs["library_size_blacklist"].get(
                        track_id, 0
                    ),
                    "nonblacklist": self.dataset.attrs["library_size_nonblacklist"].get(
                        track_id, 0
                    ),
                }

        return results
