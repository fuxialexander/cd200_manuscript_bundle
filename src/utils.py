"""
Filename: utils.py
Purpose: Utility functions for motif analysis and signal processing in genomics

Main Components:
- read_pwms_simple: Reads Position Weight Matrix files with filtering options
- pad_motifs_simple: Pads PWM motifs to uniform length for batch processing
- smooth_signal: Smooths genomic signals using 1D convolution (supports 2D/3D tensors)

Dependencies: torch, numpy, pathlib, logging
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


def read_pwms_simple(pwm_file, max_motifs=None, filter_prefix=None):
    """Simple PWM reader for testing.
    
    Args:
        pwm_file: Path to PWM file
        max_motifs: Maximum number of motifs to load
        filter_prefix: If provided, only load motifs whose names start with this prefix
    """
    motifs = {}
    current_motif = None
    pwm_lines = []

    with Path(pwm_file).open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_motif and pwm_lines:
                    # Check if we should include this motif based on filter_prefix
                    include_motif = True
                    if filter_prefix is not None:
                        include_motif = current_motif.startswith(filter_prefix)
                    
                    if include_motif:
                        pwm = np.array(
                            [[float(x) for x in row.split("\t")] for row in pwm_lines]
                        )
                        motifs[current_motif] = pwm
                        logger.info(f"Loaded motif: {current_motif}")
                        if max_motifs and len(motifs) >= max_motifs:
                            break
                current_motif = line[1:]
                pwm_lines = []
            elif line and current_motif:
                pwm_lines.append(line)

    # Handle the last motif
    if current_motif and pwm_lines and (not max_motifs or len(motifs) < max_motifs):
        include_motif = True
        if filter_prefix is not None:
            include_motif = current_motif.startswith(filter_prefix)
        
        if include_motif:
            pwm = np.array([[float(x) for x in row.split("\t")] for row in pwm_lines])
            motifs[current_motif] = pwm
            logger.info(f"Loaded motif: {current_motif}")

    logger.info(f"Total motifs loaded: {len(motifs)}")
    if filter_prefix:
        logger.info(f"Filtered for motifs starting with: {filter_prefix}")
    
    return motifs


def pad_motifs_simple(motifs_dict):
    """Simple motif padding for testing."""
    max_length = max(pwm.shape[0] for pwm in motifs_dict.values())
    
    padded_kernels = []
    pad_value = 0.0

    for motif_name, pwm in motifs_dict.items():
        orig_len = pwm.shape[0]
        total_pad = max_length - orig_len
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left

        padded_pwm = np.pad(
            pwm, ((pad_left, pad_right), (0, 0)), "constant", constant_values=pad_value
        )
        padded_kernels.append(padded_pwm)

    return np.array(padded_kernels), max_length


def smooth_signal(signal, window_size):
    """Smooth a signal using a 1D convolution with a uniform kernel.
    
    Args:
        signal: Input signal tensor
                - 2D: (batch, length) -> (batch, length)
                - 3D: (batch, samples, length) -> (batch, samples, length)
        window_size: Size of smoothing window
    
    Returns:
        Smoothed signal with same shape as input
    """
    if window_size <= 1:
        return signal
    
    if window_size % 2 == 0:
        window_size += 1

    # Create a uniform kernel with the same dtype and device as the signal
    kernel = torch.ones(1, 1, window_size, device=signal.device, dtype=signal.dtype) / window_size
    
    # Handle both 2D and 3D inputs
    if signal.dim() == 2:
        # Original 2D case: (batch, length) -> (batch, 1, length)
        smoothed = F.conv1d(signal.unsqueeze(1), kernel, padding='same')
        return smoothed.squeeze(1)
    elif signal.dim() == 3:
        # New 3D case: (batch, samples, length)
        batch_size, num_samples, seq_len = signal.shape
        # Reshape to (batch*samples, 1, length) for conv1d
        signal_reshaped = signal.view(batch_size * num_samples, 1, seq_len)
        smoothed = F.conv1d(signal_reshaped, kernel, padding='same')
        # Reshape back to (batch, samples, length)
        return smoothed.view(batch_size, num_samples, seq_len)
    else:
        raise ValueError(f"smooth_signal expects 2D or 3D input, got {signal.dim()}D") 