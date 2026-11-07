"""Numba-accelerated kernels for footprinting and ML feature extraction."""

from __future__ import annotations

import numpy as np
from numba import njit


# -----------------------------------------------------------------------------
# Numbaized accumulation for a _single_ cutoff
# -----------------------------------------------------------------------------
@njit(cache=True)
def _accumulate_fp_numba(
    track: np.ndarray,  # shape (L, ...)
    rel_pos: np.ndarray,  # shape (N_hits,)
    motif_ids: np.ndarray,  # shape (N_hits,)
    scores: np.ndarray,  # shape (N_hits,)
    n_motifs: int,
    extend: int,
    cutoff: float,
):
    """
    For each hit k, if scores[k] >= cutoff, add
    track[rel_pos[k]-extend : rel_pos[k]+extend]
    into footprints[motif_ids[k], :].
    """
    track.shape[0]
    # allocate float32 accumulator
    fp = np.zeros((n_motifs, 2 * extend) + track.shape[1:], dtype=np.float32)

    for k in range(rel_pos.shape[0]):
        sc = scores[k]
        if sc < cutoff:
            continue
        start = rel_pos[k] - extend
        # bounds are assumed pre‐filtered so 0 <= start <= start+2*extend <= L
        m_idx = motif_ids[k]
        for i in range(2 * extend):
            fp[m_idx, i] += track[start + i]

    return fp


# -----------------------------------------------------------------------------
# Prefix Sum (Integral Image) Lookup - Strand-Aware ML Features
# -----------------------------------------------------------------------------
@njit(cache=True)
def _compute_prefix_sum_from_sparse(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Compute prefix sum from CSC sparse matrix components.
    Uses float32 for memory efficiency (50% reduction vs float64).
    """
    L, C = shape
    prefix_sum = np.zeros((L + 1, C), dtype=np.float32)
    for j in range(C):
        for k in range(indptr[j], indptr[j+1]):
            row = indices[k]
            if row < L:
                prefix_sum[row + 1, j] += data[k]

        for i in range(L):
            prefix_sum[i + 1, j] += prefix_sum[i, j]
    return prefix_sum


@njit(cache=True)
def _compute_prefix_sum(track: np.ndarray) -> np.ndarray:
    """
    Compute prefix sum (cumulative sum) for efficient window lookups.
    Supports both 1D (L,) and 2D (L, C) dense tracks.
    Uses float32 for memory efficiency (50% reduction vs float64).
    """
    if track.ndim == 1:
        res = np.zeros(track.shape[0] + 1, dtype=np.float32)
        res[1:] = np.cumsum(track.astype(np.float32))
        return res
    else:
        L, C = track.shape
        res = np.zeros((L + 1, C), dtype=np.float32)
        for j in range(C):
            current_sum = 0.0
            for i in range(L):
                current_sum += track[i, j]
                res[i + 1, j] = current_sum
        return res


@njit(cache=True)
def _extract_and_aggregate_single_pass(
    prefix_sum: np.ndarray,
    positions: np.ndarray,
    motif_ids: np.ndarray,
    scores: np.ndarray,
    n_motifs: int,
    track_length: int,
    core_extend: int,
    context_extend: int,
    n_bins: int,
    shift_offsets: np.ndarray,
    cutoff: float,
    out_tensor: np.ndarray,  # (n_cells, n_motifs, n_features)
    out_counts: np.ndarray,  # (n_motifs,)
) -> None:
    """
    Single-pass, memory-efficient feature extraction and aggregation.

    Directly accumulates features into the provided `out_tensor` in-place.

    Feature layout in the last dimension (non-strand-specific):
        0:  core_signal
        1:  occupancy
        2 .. 2 + n_bins:        shape
        2 + n_bins .. end:      coaccess (len(shift_offsets))
    """
    valid_mask = scores >= cutoff
    n_valid = np.sum(valid_mask)
    if n_valid == 0:
        return

    valid_positions = positions[valid_mask]
    valid_motif_ids = motif_ids[valid_mask]

    if prefix_sum.ndim == 1:
        n_cells = 1
        has_cell_dim = False
    else:
        n_cells = prefix_sum.shape[1]
        has_cell_dim = True

    n_shifts = len(shift_offsets)
    bin_size = (2 * context_extend) // n_bins

    idx_core = 0
    idx_occ = 1
    idx_shape_start = 2
    idx_coacc_start = 2 + n_bins

    for i in range(n_valid):
        pos = valid_positions[i]
        m_idx = valid_motif_ids[i]
        out_counts[m_idx] += 1.0

        c_s = max(0, pos - core_extend)
        c_e = min(track_length, pos + core_extend)
        x_s = max(0, pos - context_extend)
        x_e = min(track_length, pos + context_extend)

        for j in range(n_cells):
            if has_cell_dim:
                core_signal = prefix_sum[c_e, j] - prefix_sum[c_s, j]
                context_signal = prefix_sum[x_e, j] - prefix_sum[x_s, j]
            else:
                core_signal = prefix_sum[c_e] - prefix_sum[c_s]
                context_signal = prefix_sum[x_e] - prefix_sum[x_s]

            out_tensor[j, m_idx, idx_core] += core_signal

            if context_signal > 0.0:
                out_tensor[j, m_idx, idx_occ] += core_signal / context_signal

            for b in range(n_bins):
                b_s = max(0, pos - context_extend + b * bin_size)
                b_e = min(track_length, pos - context_extend + (b + 1) * bin_size)

                if has_cell_dim:
                    val = prefix_sum[b_e, j] - prefix_sum[b_s, j]
                else:
                    val = prefix_sum[b_e] - prefix_sum[b_s]

                out_tensor[j, m_idx, idx_shape_start + b] += val

            for s in range(n_shifts):
                off = shift_offsets[s]
                s_s = max(0, pos + off - 5)
                s_e = min(track_length, pos + off + 5)

                if has_cell_dim:
                    val = prefix_sum[s_e, j] - prefix_sum[s_s, j]
                else:
                    val = prefix_sum[s_e] - prefix_sum[s_s]

                out_tensor[j, m_idx, idx_coacc_start + s] += val
