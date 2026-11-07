"""Traditional TF footprinting methods."""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from ._kernels import _accumulate_fp_numba

if TYPE_CHECKING:
    import pandas as pd

    from ...zarr_io import MotifDenseZarrIO, MotifHitZarrIO


class TraditionalFootprintMixin:
    """Mixin for traditional TF footprinting analysis."""

    def _update_memmap_footprints(
        self, memmap_paths, local_footprints, cutoff, n_motifs, extend, extra_dims
    ):
        """Update memmap arrays with local footprint data.

        Parameters
        ----------
        memmap_paths
            List of memmap file paths
        local_footprints
            Local footprint data to add to memmap
        cutoff
            Score cutoff(s) for motif matches
        n_motifs
            Number of motifs
        extend
            Number of bases to extend around motif match
        extra_dims
            Extra dimensions for the footprint array

        Returns
        -------
        None
        """
        if memmap_paths is not None:
            if isinstance(cutoff, float | int):
                fp_memmap = np.memmap(
                    memmap_paths[0],
                    dtype=np.float32,
                    mode="r+",
                    shape=(n_motifs, 2 * extend, *extra_dims),
                )
                fp_memmap += local_footprints
                fp_memmap.flush()
            else:
                n_cutoffs = len(cutoff)
                fp_memmap = np.memmap(
                    memmap_paths[0],
                    dtype=np.float32,
                    mode="r+",
                    shape=(n_cutoffs, n_motifs, 2 * extend, *extra_dims),
                )
                fp_memmap += local_footprints
                fp_memmap.flush()

        return local_footprints if memmap_paths is None else None

    def _compute_footprint_in_region_for_all_motifs(
        self,
        m: MotifDenseZarrIO | MotifHitZarrIO,
        chromosome: str,
        start: int,
        end: int,
        cutoff: float | np.ndarray = 5,
        extend: int = 200,
        motif_idx: np.ndarray = None,
        zarr_group_name="motif_match_data",
        memmap_paths=None,
        regions=None,
        **track_kwargs,
    ):
        """
        Compute footprints for all motifs in a specific genomic region.

        Parameters
        ----------
        m : MotifDenseZarrIO
            Motif object containing match data
        chromosome : str
            Chromosome name
        start : int
            Start position of the region
        end : int
            End position of the region
        cutoff : float or np.ndarray
            Score cutoff(s) for motif matches
        extend : int
            Number of bases to extend around motif match
        motif_idx : np.ndarray, optional
            Indices of motifs to process, if None, process all motifs
        zarr_group_name : str
            Name of zarr group containing motif match data
        memmap_paths : list, optional
            Paths to memmap files for storing results
        regions : pd.DataFrame, optional
            Regions to restrict motif matches to
        **track_kwargs
            Additional arguments passed to get_track()

        Returns
        -------
        np.ndarray or None
            Footprint array if memmap_paths is None, otherwise None
        """
        n_motifs = m.n_motifs if motif_idx is None else len(motif_idx)

        track = self.get_track(
            chromosome, start, end, output_format="raw_array", **track_kwargs
        )
        # move sequence dimension to axis 0
        seq_dim = np.where(np.array(track.shape) == end - start)[0][0]
        track = track.transpose(
            seq_dim, *np.arange(seq_dim), *np.arange(seq_dim + 1, track.ndim)
        )
        extra_dims = track.shape[1:] if track.ndim > 1 else ()

        # prepare matches & local_footprints
        is_multi = not isinstance(cutoff, float | int)
        if is_multi:
            cutoff = np.sort(cutoff)
            n_cutoffs = len(cutoff)

        matches = m.get_motif_match_indices(
            chromosome,
            start,
            end,
            zarr_group_name=zarr_group_name,
            cutoff=(cutoff if not is_multi else cutoff[0]),
            regions=regions,
        )

        # init output
        if not is_multi:
            local_footprints = np.zeros(
                (n_motifs, 2 * extend, *extra_dims), dtype=np.float32
            )
        else:
            local_footprints = np.zeros(
                (n_cutoffs, n_motifs, 2 * extend, *extra_dims), dtype=np.float32
            )

        if matches.size == 0:
            return self._update_memmap_footprints(
                memmap_paths, local_footprints, cutoff, n_motifs, extend, extra_dims
            )

        # optional sub‐set of motifs
        if motif_idx is not None:
            matches = matches[np.isin(matches[:, 0], motif_idx)]

        # filter by the lowest cutoff
        matches = matches[matches[:, 2] >= (cutoff if not is_multi else cutoff[0])]

        # compute relative positions & filter OOB
        rel_pos = matches[:, 1].astype(np.int64) - start
        ok = (rel_pos >= extend) & (rel_pos + extend <= track.shape[0])
        rel_pos = rel_pos[ok]
        motif_ids = matches[ok, 0].astype(np.int64)
        scores = matches[ok, 2].astype(np.float32)

        # single‐cutoff → one JIT call
        if not is_multi:
            # ensure float32 track for JIT
            track32 = track.astype(np.float32)
            fp32 = _accumulate_fp_numba(
                track32, rel_pos, motif_ids, scores, n_motifs, extend, float(cutoff)
            )
            local_footprints[:] = fp32

        # multi‐cutoff → loop over thresholds
        else:
            track32 = track.astype(np.float32)
            for c_idx, thr in enumerate(cutoff):
                # pass only the hits above this threshold
                mask = scores >= thr
                if not mask.any():
                    continue
                fp32 = _accumulate_fp_numba(
                    track32,
                    rel_pos[mask],
                    motif_ids[mask],
                    scores[mask],
                    n_motifs,
                    extend,
                    float(thr),
                )
                local_footprints[c_idx] = fp32

        return self._update_memmap_footprints(
            memmap_paths, local_footprints, cutoff, n_motifs, extend, extra_dims
        )

    def _compute_footprint_in_chunk_for_all_motifs(
        self,
        m: MotifDenseZarrIO | MotifHitZarrIO,
        chromosome: str,
        chunk_idx: int,
        cutoff: float | np.ndarray = 5,
        extend: int = 200,
        motif_idx: np.ndarray = None,
        zarr_group_name="motif_match_data",
        memmap_paths=None,
        regions=None,
        **track_kwargs,
    ):
        """
        Compute footprints for all motifs in a specific chunk.

        This is a wrapper around _compute_footprint_region_for_all_motifs that
        calculates the chunk boundaries based on the chunk index.

        Parameters
        ----------
        m : MotifDenseZarrIO
            Motif object containing match data
        chromosome : str
            Chromosome name
        chunk_idx : int
            Index of the chunk to process
        cutoff : float or np.ndarray
            Score cutoff(s) for motif matches
        extend : int
            Number of bases to extend around motif match
        motif_idx : np.ndarray, optional
            Indices of motifs to process, if None, process all motifs
        zarr_group_name : str
            Name of zarr group containing motif match data
        memmap_paths : list, optional
            Paths to memmap files for storing results
        regions : pd.DataFrame, optional
            Regions to restrict motif matches to
        **track_kwargs
            Additional arguments passed to get_track()

        Returns
        -------
        np.ndarray or None
            Footprint array if memmap_paths is None, otherwise None
        """
        chunk_start = chunk_idx * self.chunk_size
        chunk_end = min((chunk_idx + 1) * self.chunk_size, self.chrom_sizes[chromosome])

        return self._compute_footprint_in_region_for_all_motifs(
            m,
            chromosome,
            chunk_start,
            chunk_end,
            cutoff,
            extend,
            motif_idx,
            zarr_group_name,
            memmap_paths,
            regions,
            **track_kwargs,
        )

    def get_genome_wide_footprint_all_motifs(
        self,
        m: MotifDenseZarrIO | MotifHitZarrIO,
        n_processes: int = None,
        cutoff: float | np.ndarray = 5,
        extend: int = 250,
        motif_idx: np.ndarray = None,
        zarr_group_name="motif_match_data",
        regions: pd.DataFrame | None = None,
        **track_kwargs,
    ):
        """Get genome-wide footprint for selected motifs using precomputed match indices.

        Parameters
        ----------
        m
            MotifDenseZarrIO instance containing motif data
        n_processes
            Number of processes to use. If None, uses CPU count - 1.
        cutoff
            Score cutoff(s) for motif matches. Can be a single float or array of cutoffs.
            Default is 5.
        extend
            Number of bases to extend around motif match. Default is 200.
        motif_idx
            Array of motif indices to process. If None, process all motifs.
        zarr_group_name
            Name of zarr group containing motif match data. Default is "motif_match_data".
        **track_kwargs : dict
            Additional arguments passed to get_track_obj()

        Returns
        -------
        For single cutoff: Array of shape (n_motifs, 2*extend, *extra_dims) containing footprints
        For multiple cutoffs: Array of shape (n_cutoffs, n_motifs, 2*extend, *extra_dims)
        """
        if n_processes is None:
            n_processes = os.cpu_count() - 1

        n_motifs = m.n_motifs if motif_idx is None else len(motif_idx)

        # Get track shape from a sample track
        sample_track = self.get_track(
            list(self.chrom_sizes.keys())[0],
            0,
            1000,
            output_format="raw_array",
            **track_kwargs,
        )
        # always put the sequence dimension first
        seq_dim = np.where(np.array(sample_track.shape) == 1000)[0][0]
        sample_track = sample_track.transpose(
            seq_dim,
            *np.arange(sample_track.ndim)[0:seq_dim],
            *np.arange(sample_track.ndim)[seq_dim + 1 :],
        )
        extra_dims = sample_track.shape[1:] if len(sample_track.shape) > 1 else ()

        if not isinstance(cutoff, float | int):
            cutoff = np.sort(cutoff)
            n_cutoffs = len(cutoff)


        # Create temporary directory for memmap files
        temp_dir = tempfile.mkdtemp()

        try:
            # Create memory-mapped file
            memmap_path = os.path.join(temp_dir, "footprint.memmap")
            if isinstance(cutoff, float | int):
                footprints = np.memmap(
                    memmap_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(n_motifs, 2 * extend, *extra_dims),
                )
            else:
                footprints = np.memmap(
                    memmap_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(n_cutoffs, n_motifs, 2 * extend, *extra_dims),
                )
            footprints.fill(0)
            memmap_paths = [memmap_path]

            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                futures = []
                for chrom, n_chunks in self.chrom_n_chunks.items():
                    for chunk_idx in range(n_chunks):
                        if regions is not None:
                            regions_chunk = regions.query(
                                f"(Chromosome == @chrom) & (Start > {chunk_idx * self.chunk_size}) & (End < {(chunk_idx + 1) * self.chunk_size})"
                            )
                        else:
                            regions_chunk = None
                        futures.append(
                            executor.submit(
                                self._compute_footprint_in_chunk_for_all_motifs,
                                m,
                                chrom,
                                chunk_idx,
                                cutoff,
                                extend,
                                motif_idx,
                                zarr_group_name,
                                memmap_paths,
                                regions_chunk,
                                **track_kwargs,
                            )
                        )

                for future in tqdm(futures):
                    future.result()

            # Copy results from memmap
            result = footprints.copy()
            return result

        finally:
            # Clean up memmap files
            import shutil

            shutil.rmtree(temp_dir)
