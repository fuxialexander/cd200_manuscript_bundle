"""
Core FIMO p-value calculation and utility functions.

This module contains the low-level, performance-critical functions for
calculating FIMO score-to-p-value mappings and for processing motif data.
Many of these functions are accelerated with Numba.
"""

import math
import numba
import numpy as np

@numba.njit('float64(float64, float64)', fastmath=True, cache=True)
def logaddexp2(x, y):
    """Calculate the logaddexp in a numerically stable manner in base 2."""
    if x == float("-inf") and y == float("-inf"):
        return float("-inf")
    if x == float("inf") or y == float("inf"):
        return float("inf")
    vmax, vmin = max(x, y), min(x, y)
    return vmax + math.log2(math.pow(2, vmin - vmax) + 1)


@numba.njit(cache=True)
def _pwm_to_mapping(log_pwm, bin_size):
    """Calculate score <-> log p-value mappings using NRDB background frequencies."""
    n, l = log_pwm.shape

    # NRDB background frequencies
    log_bg_A = math.log2(0.281774)
    log_bg_C = math.log2(0.222020)
    log_bg_G = math.log2(0.228876)
    log_bg_T = math.log2(0.267330)

    int_log_pwm = np.round(log_pwm / bin_size).astype(np.int32)

    smallest, largest = 9999999, -9999999
    log_pwm_min_csum, log_pwm_max_csum = 0, 0
    for i in range(l):
        log_pwm_min = 9999999
        log_pwm_max = -9999999

        for j in range(n):
            log_pwm_min = min(log_pwm_min, int_log_pwm[j, i])
            log_pwm_max = max(log_pwm_max, int_log_pwm[j, i])

        log_pwm_min_csum += log_pwm_min
        log_pwm_max_csum += log_pwm_max

        smallest = min(smallest, log_pwm_min_csum)
        largest = max(largest, log_pwm_max_csum)

    largest += l

    logpdf_size = largest - smallest + 1
    logpdf = np.empty(logpdf_size, dtype=np.float64)
    old_logpdf = np.full(logpdf_size, -np.inf, dtype=np.float64)

    # Map nucleotide index to background log frequency
    log_bg_values = [log_bg_A, log_bg_C, log_bg_G, log_bg_T]

    for i in range(n):
        idx = int_log_pwm[i, 0] - smallest
        old_logpdf[idx] = logaddexp2(old_logpdf[idx], log_bg_values[i])

    for i in range(1, l):
        for j in range(logpdf_size):
            logpdf[j] = -np.inf

        for j in range(logpdf_size):
            x = old_logpdf[j]
            if x != -np.inf:
                for k in range(n):
                    idx = j + int_log_pwm[k, i]
                    if 0 <= idx < logpdf_size:
                        logpdf[idx] = logaddexp2(logpdf[idx], log_bg_values[k] + x)

        for j in range(logpdf_size):
            old_logpdf[j] = logpdf[j]

    for i in range(logpdf_size - 2, -1, -1):
        logpdf[i] = logaddexp2(logpdf[i], logpdf[i + 1])

    return smallest, logpdf

def trim_motif_padding(motif_log_odds: np.ndarray) -> np.ndarray:
    """Remove zero-padding positions from motif log-odds to get actual motif length."""
    valid_positions = []
    for pos in range(motif_log_odds.shape[1]):
        column = motif_log_odds[:, pos]
        is_all_zero = np.allclose(column, 0.0, atol=1e-6)
        is_all_same_negative = np.allclose(column, column[0], atol=1e-6) and column[0] < -3.0

        if not (is_all_zero or is_all_same_negative):
            valid_positions.append(pos)

    if not valid_positions:
        return motif_log_odds[:, :0]

    start_idx = valid_positions[0]
    end_idx = valid_positions[-1] + 1
    return motif_log_odds[:, start_idx:end_idx]

def pvalue_from_logpdf(scores: np.ndarray, smallest: int, logpdf: np.ndarray, bin_size: float) -> np.ndarray:
    """
    Vectorized conversion of raw scores to p-values using a precomputed lookup table.
    """
    score_indices = (scores / bin_size).astype(int) - smallest
    
    # Clip indices to be within the bounds of the logpdf table
    score_indices = np.clip(score_indices, 0, len(logpdf) - 1)
    
    log_pvalues = logpdf[score_indices]
    
    return 2.0 ** log_pvalues
