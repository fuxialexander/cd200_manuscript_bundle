from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyranges as pr
from tqdm import tqdm

if TYPE_CHECKING:
    from ...analysis.track import Track


class CallPeaksMixin:
    """
    Mixin class for calling peaks on tracks.
    """

    def get_track_obj(self, chrom, start, end, **track_kwargs) -> Track:
        """
        Get a track object for the specified region and selected indices.
        """
        NotImplementedError("Subclasses must implement get_track_obj()")

    def _process_chunk_call_peaks(self, args):
        chrom, start, end, track_id, min_length, merge_length, track_kwargs, subdivide_long_peaks = args
        if track_kwargs is None:
            track_kwargs = {}
        track = self.get_track_obj(chrom, start, end, **track_kwargs)

        # Initialize peaks dictionary
        peaks_dict = {}

        # Handle different track_id cases
        if track_id == "agg":
            peaks = track.get_local_iqr_peak(
                track_id, min_length=min_length, merge_length=merge_length
            )
            if peaks is not None and len(peaks) > 0:
                peaks_dict["agg"] = peaks
        elif isinstance(track_id, list):
            for tid in track_id:
                peaks = track.get_local_iqr_peak(
                    tid, min_length=min_length, merge_length=merge_length
                )
                if peaks is not None and len(peaks) > 0:
                    peaks_dict[tid] = peaks

        # Return None if no peaks found
        if not peaks_dict:
            return None

        # Process each set of peaks
        for tid in peaks_dict:
            peaks = peaks_dict[tid]

            if subdivide_long_peaks:
                long_peaks = peaks[peaks[:, 1] - peaks[:, 0] > 2048]
                short_peaks = peaks[peaks[:, 1] - peaks[:, 0] <= 2048]

                # Process long peaks
                local_peaks = []
                if long_peaks.shape[0] > 0:
                    for long_peaks_start, long_peaks_end in long_peaks:
                        local_track = self.get_track_obj(
                            chrom, long_peaks_start, long_peaks_end, **track_kwargs
                        )
                        local_peak = local_track.get_local_iqr_peak(
                            tid, min_length=200, merge_length=0, iqr_factor=-0.1
                        )
                        if local_peak.shape[0] > 0 and np.isnan(local_peak).sum() == 0:
                            local_peaks.append(local_peak)
                        else:
                            local_peaks.append(np.array([[long_peaks_start, long_peaks_end]]))
                    local_peaks = np.concatenate(local_peaks)
                    peaks_dict[tid] = np.concatenate([short_peaks, local_peaks])
            # If not subdividing, keep all peaks as-is (peaks_dict[tid] remains unchanged)

        return peaks_dict

    def _process_chromosome_call_peaks(
        self,
        chrom,
        n_chunk,
        track_id,
        min_length,
        merge_length,
        track_kwargs,
        subdivide_long_peaks=True,
        n_jobs=16,
    ):
        peaks_by_id = {}
        chunk_size = self.chunk_size
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            # Process full chunks
            for i in range(n_chunk):
                args = (
                    chrom,
                    i * chunk_size,
                    min((i + 1) * chunk_size, self.chrom_sizes[chrom]),
                    track_id,
                    min_length,
                    merge_length,
                    track_kwargs,
                    subdivide_long_peaks,
                )
                futures.append(executor.submit(self._process_chunk_call_peaks, args))
            # Process overlapping chunks
            for i in range(n_chunk - 1):
                args = (
                    chrom,
                    i * chunk_size + chunk_size // 2,
                    min(
                        (i + 1) * chunk_size + chunk_size // 2, self.chrom_sizes[chrom]
                    ),
                    track_id,
                    min_length,
                    merge_length,
                    track_kwargs,
                    subdivide_long_peaks,
                )
                futures.append(executor.submit(self._process_chunk_call_peaks, args))

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    # Handle peaks_dict from result
                    for tid, peaks in result.items():
                        if tid not in peaks_by_id:
                            peaks_by_id[tid] = []
                        peaks_by_id[tid].append(peaks)

        # Process peaks for each track ID
        results = {}
        for tid, peaks_list in peaks_by_id.items():
            if peaks_list:
                peaks = np.concatenate(peaks_list)
                peaks_df = pd.DataFrame(peaks, columns=["Start", "End"])
                peaks_df["Chromosome"] = chrom
                peaks_df = peaks_df[["Chromosome", "Start", "End"]]
                results[tid] = pr.PyRanges(peaks_df).merge().df

        # If track_id is not a list, return just that result
        if not isinstance(track_id, list):
            return results.get(track_id)

        # If track_id is a list, return all results
        return results if results else None

    def call_peaks(
        self,
        track_id="agg",
        min_length=200,
        merge_length=50,
        blacklist=None,
        subdivide_long_peaks=True,
        n_jobs=16,
        **track_kwargs,
    ):
        """
        Call peaks on tracks.

        Parameters
        ----------
        track_id : str or list
            Track identifier. Default is "agg".
        min_length : int
            Minimum length of peaks. Default is 200.
        merge_length : int
            Length to merge nearby peaks. Default is 50.
        blacklist : str or None
            Path to blacklist file. Default is None.
        subdivide_long_peaks : bool
            Whether to subdivide long peaks (>5000 bp) into smaller sub-peaks.
            Default is True.
        n_jobs : int
            Number of parallel jobs. Default is 16.
        **track_kwargs : dict
            Additional arguments passed to get_track_obj()
        """
        all_peaks = []
        track_id = [track_id] if not isinstance(track_id, list) else track_id
        for chrom, n_chunk in tqdm(
            self.chrom_n_chunks.items(), desc="Processing chromosomes"
        ):
            chromosome_peaks = self._process_chromosome_call_peaks(
                chrom, n_chunk, track_id, min_length, merge_length, track_kwargs, subdivide_long_peaks, n_jobs
            )
            if chromosome_peaks is not None:
                all_peaks.append(chromosome_peaks)

        if all_peaks:
            # Handle case where all_peaks is list of DataFrames
            if isinstance(all_peaks[0], pd.DataFrame):
                all_peaks_merged = pd.concat(all_peaks)
            # Handle case where all_peaks is list of dicts of DataFrames
            else:
                all_peaks_merged = {}
                for i in range(len(all_peaks)):
                    for tid in all_peaks[i]:
                        if tid not in all_peaks_merged:
                            all_peaks_merged[tid] = []
                        all_peaks_merged[tid].append(all_peaks[i][tid])
                for tid in all_peaks_merged:
                    all_peaks_merged[tid] = pd.concat(all_peaks_merged[tid])

            if blacklist is None:
                try:
                    blacklist = pd.read_csv(
                        f"https://raw.githubusercontent.com/Boyle-Lab/Blacklist/refs/heads/master/lists/{self.assembly}-blacklist.v2.bed.gz",
                        sep="\t",
                        header=None,
                    )
                    blacklist.columns = ["Chromosome", "Start", "End", "name"]
                except Exception as e:
                    print(f"Error loading blacklist: {e}, using all peaks")
                    blacklist = pd.DataFrame(
                        columns=["Chromosome", "Start", "End", "name"]
                    )
            else:
                blacklist = pd.read_csv(blacklist, sep="\t", header=None)
                blacklist.columns = ["Chromosome", "Start", "End", "name"]

            # Filter out peaks that overlap with blacklist using pyranges
            pr_blacklist = pr.PyRanges(blacklist)

            if isinstance(all_peaks_merged, dict):
                # Handle dict case - filter each track separately
                filtered_peaks = {}
                for tid, peaks_df in all_peaks_merged.items():
                    pr_peaks = pr.PyRanges(peaks_df)
                    filtered_peaks[tid] = pr_peaks.subtract(pr_blacklist).df
                return filtered_peaks
            else:
                # Handle DataFrame case
                pr_peaks = pr.PyRanges(all_peaks_merged)
                filtered_peaks = pr_peaks.subtract(pr_blacklist).df
                return filtered_peaks
        else:
            return pd.DataFrame(columns=["Chromosome", "Start", "End"])
