"""ML feature extraction methods for motif analysis."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import issparse
from tqdm import tqdm

from ._kernels import (
    _compute_prefix_sum,
    _compute_prefix_sum_from_sparse,
    _extract_and_aggregate_single_pass,
)

if TYPE_CHECKING:
    import pandas as pd

    from ...zarr_io import MotifDenseZarrIO, MotifHitZarrIO


class MLFeatureMixin:
    """Mixin for ML feature extraction from motif matches."""

    def _get_combined_track(
        self,
        chromosome: str,
        start: int,
        end: int,
        **track_kwargs,
    ) -> np.ndarray:
        """
        Get combined track (non-strand-specific).

        Args:
            chromosome: Chromosome name
            start: Start position
            end: End position
            **track_kwargs: Additional arguments for get_track()

        Returns:
            Combined track array
            Shape: (L, C), dtype float32
        """
        # Default to concat aggregator if not specified, to preserve cell types
        if "aggregator" not in track_kwargs:
            track_kwargs["aggregator"] = "concat"

        # Get combined track
        combined_track = self.get_track(
            chromosome, start, end, output_format="raw_array", **track_kwargs
        )

        # Ensure (L, C) shape
        if combined_track.ndim == 1:
            # Add cell dimension if missing
            combined_track = combined_track[:, np.newaxis]
        else:
            # Move sequence dimension to axis 0
            seq_dim = np.where(np.array(combined_track.shape) == end - start)[0][0]
            combined_track = combined_track.transpose(
                seq_dim,
                *np.arange(seq_dim),
                *np.arange(seq_dim + 1, combined_track.ndim),
            )
            # Flatten extra dimensions if any (e.g. into a single 'C' dimension)
            if combined_track.ndim > 2:
                combined_track = combined_track.reshape(combined_track.shape[0], -1)

        # Return as float32
        combined_track = combined_track.astype(np.float32)
        return combined_track

    def _compute_ml_features_prefix_sum(
        self,
        m: MotifDenseZarrIO | MotifHitZarrIO,
        chromosome: str,
        start: int,
        end: int,
        cutoff: float = 5,
        motif_idx: np.ndarray = None,
        core_extend: int = 50,
        context_extend: int = 500,
        n_bins: int = 20,
        shift_offsets: np.ndarray = None,
        zarr_group_name: str = "motif_match_data",
        regions: pd.DataFrame | None = None,
        aggregate_by_motif: bool = True,
        **track_kwargs,
        ) -> dict:
        """
        Compute ML-ready features using prefix sum lookup approach.

        Args:
            m: Motif IO object
            chromosome: Chromosome name
            start: Start position
            end: End position
            cutoff: Score cutoff
            motif_idx: Subset of motifs to process
            core_extend: Core window size (±bp)
            context_extend: Context window size (±bp)
            n_bins: Number of bins for shape reconstruction
            shift_offsets: Co-accessibility offset positions
            zarr_group_name: Zarr group name
            regions: Optional regions to restrict matches
            aggregate_by_motif: If True, returns aggregated tensor [CellType, Motif, Features]
            **track_kwargs: Additional arguments for get_track()

        Returns:
            Dictionary with aggregated feature arrays: {"agg_tensor", "motif_counts"}
        """
        if shift_offsets is None:
            shift_offsets = np.array([10, 20, 30], dtype=np.int32)

        n_motifs_total = m.n_motifs

        # Get combined track (L, C), non-strand-specific
        combined_track = self._get_combined_track(
            chromosome, start, end, **track_kwargs
        )

        # Convert to float32 for memory efficiency
        # Check if track is sparse (SciPy)
        if issparse(combined_track):
            # For sparse matrices, convert data array to float32
            combined_track = combined_track.astype(np.float32)
        else:
            # For dense arrays, convert to float32
            combined_track = combined_track.astype(np.float32)

        track_length, n_cells = combined_track.shape

        # Compute prefix sum (single track, non-strand-specific)
        if issparse(combined_track):
            # CSC is best for column-wise prefix sum
            track_csc = combined_track.tocsc()
            prefix_sum = _compute_prefix_sum_from_sparse(
                track_csc.data, track_csc.indices, track_csc.indptr, track_csc.shape
            )
        else:
            prefix_sum = _compute_prefix_sum(combined_track)


        # Get motif matches
        # CRITICAL: Pass motif_idx to the backend to avoid fetching millions of irrelevant hits
        matches = m.get_motif_match_indices(
            chromosome,
            start,
            end,
            zarr_group_name=zarr_group_name,
            cutoff=cutoff,
            regions=regions,
            motif_idx=motif_idx
        )

        n_features = 2 + n_bins + len(shift_offsets)

        if matches.size == 0:
            # No matches in this chunk
            return {
                "agg_tensor": np.zeros(
                    (n_cells, n_motifs_total, n_features), dtype=np.float32
                ),
                "motif_counts": np.zeros(n_motifs_total, dtype=np.float32),
            }

        rel_pos = matches[:, 1].astype(np.int64) - start
        ok = (rel_pos >= context_extend) & (rel_pos + context_extend <= track_length)
        rel_pos = rel_pos[ok]
        motif_ids = matches[ok, 0].astype(np.int64)
        scores = matches[ok, 2].astype(np.float64)

        # Allocate a single tensor and counts array and aggregate in-place
        out_tensor = np.zeros(
            (n_cells, n_motifs_total, n_features), dtype=np.float32
        )
        out_counts = np.zeros(n_motifs_total, dtype=np.float32)

        _extract_and_aggregate_single_pass(
            prefix_sum,
            rel_pos,
            motif_ids,
            scores,
            n_motifs_total,
            track_length,
            core_extend,
            context_extend,
            n_bins,
            shift_offsets.astype(np.int32),
            float(cutoff),
            out_tensor,
            out_counts,
        )

        # IMPORTANT: We return sums and counts; global aggregation will handle averaging.
        return {"agg_tensor": out_tensor, "motif_counts": out_counts}

    def get_genome_wide_ml_features_prefix_sum(
        self,
        m: MotifDenseZarrIO | MotifHitZarrIO,
        n_processes: int = None,
        cutoff: float = 5,
        motif_idx: np.ndarray = None,
        core_extend: int = 50,
        context_extend: int = 500,
        n_bins: int = 20,
        shift_offsets: np.ndarray = None,
        zarr_group_name: str = "motif_match_data",
        regions: pd.DataFrame | None = None,
        aggregate_by_motif: bool = True,
        check_convergence_every: int = 10,
        convergence_threshold: float = 1e-5,
        **track_kwargs,
    ) -> dict | np.ndarray:
        """
        Get genome-wide ML features using prefix sum approach.

        Args:
            aggregate_by_motif: If True (default), returns averaged [CellType, Motif, Features] array.
            check_convergence_every: Check convergence every N chunks (default: 10).
            convergence_threshold: Threshold for early stopping convergence (default: 1e-5).
        """
        if n_processes is None:
            n_processes = os.cpu_count() - 1

        n_motifs = m.n_motifs

        # Sample to get n_cells
        sample_track = self._get_combined_track(
            list(self.chrom_sizes.keys())[0], 0, 1000, **track_kwargs
        )
        n_cells = sample_track.shape[1]

        n_shifts = len(shift_offsets) if shift_offsets is not None else 0
        n_features = 2 + n_bins + n_shifts

        total_agg = np.zeros(
            (n_cells, n_motifs, n_features), dtype=np.float32
        )
        total_counts = np.zeros(n_motifs, dtype=np.float32)
        prev_avg: np.ndarray | None = None

        # Collect all chunk tasks
        chunk_tasks = []
        for chrom, n_chunks in self.chrom_n_chunks.items():
            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * self.chunk_size
                chunk_end = min((chunk_idx + 1) * self.chunk_size, self.chrom_sizes[chrom])

                if regions is not None:
                    # Simple region filter
                    regions_chunk = regions.query(f"Chromosome == '{chrom}'")
                else:
                    regions_chunk = None

                chunk_tasks.append((chrom, chunk_start, chunk_end, regions_chunk))

        # Process chunks with early stopping for convergence
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = {}
            submitted_tasks = set()
            processed_chunks = 0
            chunks_since_check = 0
            next_task_idx = 0

            # Process results with convergence checking
            pbar = tqdm(total=len(chunk_tasks), desc="Extracting ML features")
            converged = False

            # Submit initial batch
            while next_task_idx < len(chunk_tasks) and len(futures) < n_processes * 2 and not converged:
                task_idx = next_task_idx
                chrom, chunk_start, chunk_end, regions_chunk = chunk_tasks[task_idx]
                future = executor.submit(
                    self._compute_ml_features_prefix_sum,
                    m, chrom, chunk_start, chunk_end, cutoff, motif_idx,
                    core_extend, context_extend, n_bins, shift_offsets,
                    zarr_group_name, regions_chunk,
                    aggregate_by_motif=aggregate_by_motif, **track_kwargs
                )
                futures[future] = task_idx
                submitted_tasks.add(task_idx)
                next_task_idx += 1

            # Process completed futures with convergence checking
            # Use a while loop to continuously process futures as they complete
            while futures and not converged:
                # Wait for at least one future to complete using as_completed
                # This will block until at least one future completes
                for future in as_completed(futures.keys()):
                    # Process this completed future
                    if converged:
                        future.cancel()
                        break

                    task_idx = futures.pop(future)
                    try:
                        res = future.result()
                        processed_chunks += 1
                        chunks_since_check += 1

                        # Accumulate raw sums and counts from each chunk
                        total_agg += res["agg_tensor"]
                        total_counts += res["motif_counts"]

                        # Batched convergence checking to reduce overhead
                        if (
                            chunks_since_check >= check_convergence_every
                            and processed_chunks >= 20
                        ):
                            chunks_since_check = 0

                            # Compute current running average only for motifs with data
                            current_avg = np.zeros_like(total_agg)
                            denom = total_counts[np.newaxis, :, np.newaxis]
                            np.divide(
                                total_agg,
                                denom,
                                out=current_avg,
                                where=denom > 0,
                            )

                            if prev_avg is not None:
                                mask = total_counts > 0
                                if mask.any():
                                    delta = np.abs(
                                        current_avg[:, mask, :] - prev_avg[:, mask, :]
                                    )
                                    max_delta = float(np.max(delta))

                                    if max_delta < convergence_threshold:
                                        converged = True
                                        pbar.set_description(
                                            f"Converged! Max delta: {max_delta:.2e} < {convergence_threshold:.2e}"
                                        )
                                        pbar.write("")
                                        pbar.write("=" * 60)
                                        pbar.write(
                                            "EARLY STOPPING: Convergence reached!"
                                        )
                                        pbar.write(
                                            f"  Max delta: {max_delta:.2e} < threshold: {convergence_threshold:.2e}"
                                        )
                                        pbar.write(
                                            f"  Processed: {processed_chunks}/{len(chunk_tasks)} chunks ({processed_chunks/len(chunk_tasks)*100:.1f}%)"
                                        )
                                        pbar.write("=" * 60)
                                        for f in list(futures.keys()):
                                            f.cancel()
                                        break

                            prev_avg = current_avg

                        pbar.update(1)

                    except Exception as e:
                        pbar.write(f"Error processing chunk {task_idx}: {e}")
                        pbar.update(1)

                    # Submit new tasks if not converged
                    if not converged and next_task_idx < len(chunk_tasks):
                        task_idx = next_task_idx
                        chrom, chunk_start, chunk_end, regions_chunk = chunk_tasks[task_idx]
                        future = executor.submit(
                            self._compute_ml_features_prefix_sum,
                            m, chrom, chunk_start, chunk_end, cutoff, motif_idx,
                            core_extend, context_extend, n_bins, shift_offsets,
                            zarr_group_name, regions_chunk,
                            aggregate_by_motif=aggregate_by_motif, **track_kwargs
                        )
                        futures[future] = task_idx
                        submitted_tasks.add(task_idx)
                        next_task_idx += 1

                    # Break out of inner loop to re-check futures dict
                    break

            pbar.close()

            # Report final status
            if converged:
                print(f"\n✓ Early stopping: Converged after processing {processed_chunks}/{len(chunk_tasks)} chunks")
            else:
                print(f"\n✓ Completed: Processed all {processed_chunks}/{len(chunk_tasks)} chunks")

        # Final global average over all processed chunks
        denom = total_counts[np.newaxis, :, np.newaxis]
        np.divide(total_agg, denom, out=total_agg, where=denom > 0)
        return total_agg
