"""
Metrics for mutation effect calculation.

All functions are pure (no side effects) and return dictionaries
with calculated metrics.
"""

from typing import Dict, List, Optional
import torch
import numpy as np


def calculate_effects(ref_outputs: Dict[str, torch.Tensor],
                     alt_outputs: Dict[str, torch.Tensor],
                     output_heads: List[str],
                     core_region: Optional[tuple] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate effects for all output heads automatically.

    Args:
        ref_outputs: Reference predictions {head_name: tensor}
        alt_outputs: Alternative predictions {head_name: tensor}
        output_heads: List of output head names
                     Supports hybrid modes like 'regression_differential', 'count_differential'
        core_region: Optional (start, end) indices to slice predictions before calculating effects.
                    For regression head, this slices the spatial dimension.

    Returns:
        Nested dict: {head_name: {metric_name: value}}

        Each head_name dict contains:
        - 'primary_metric': Main effect metric for this head type
        - Additional metrics specific to head type

    Example:
        >>> effects = calculate_effects(ref_out, alt_out, ['count', 'regression'])
        >>> effects['count']['log_fold_change']
        0.342
        >>> effects['regression']['max_difference']
        1.567
        >>> # Hybrid mode with core region
        >>> effects = calculate_effects(ref_out, alt_out, ['regression_differential'],
        ...                            core_region=(900, 1148))
        >>> effects['regression_differential']['hybrid_score']
        0.128
    """
    effects = {}

    for head in output_heads:
        # Handle hybrid modes (e.g., 'regression_differential', 'count_differential')
        if '_' in head:
            parts = head.split('_')
            if len(parts) == 2 and parts[1] in ['differential', 'fc']:
                # Hybrid mode: need both component heads
                head1, head2 = parts[0], parts[1]
                if head1 in ref_outputs and head2 in ref_outputs:
                    effects[head] = calculate_hybrid_effects(
                        ref_outputs[head1], alt_outputs[head1],
                        ref_outputs[head2], alt_outputs[head2],
                        head1, head2,
                        core_region=core_region
                    )
                    continue
                else:
                    raise ValueError(f"Hybrid mode '{head}' requires both '{head1}' and '{head2}' heads in outputs")

        # Standard single-head modes
        if head not in ref_outputs:
            raise ValueError(f"Head '{head}' not found in model outputs")

        ref = ref_outputs[head]
        alt = alt_outputs[head]

        # Route to appropriate calculation based on head name
        if head == 'regression':
            effects[head] = calculate_regression_effects(ref, alt, core_region=core_region)
        elif head == 'count':
            effects[head] = calculate_count_effects(ref, alt)
        elif head in ['fc', 'differential']:
            effects[head] = calculate_fc_effects(ref, alt)
        else:
            # Generic handling for unknown heads
            effects[head] = calculate_generic_effects(ref, alt)

    return effects


def calculate_regression_effects(ref: torch.Tensor,
                                 alt: torch.Tensor,
                                 core_region: Optional[tuple] = None) -> Dict[str, float]:
    """
    Calculate effects for regression head (1D profile output).

    Metrics:
    - max_difference: delta(max value) - most sensitive to peak changes
    - mean_difference: delta(mean value) - overall signal change
    - correlation: Pearson correlation between profiles
    - js_distance: Jensen-Shannon distance between profiles

    Args:
        ref: Reference profile (shape: [batch, celltype, length] or [batch, length])
        alt: Alternative profile (shape: [batch, celltype, length] or [batch, length])
        core_region: Optional (start, end) indices to slice the spatial dimension
                    before computing metrics. Matches run_analysis_dlbcl.py behavior.

    Returns:
        Dict with effect metrics

    Note:
        Effect sign convention: ref - alt (matches run_analysis_dlbcl.py)
        Positive effect means alt is LOWER than ref (deletion reduced signal)
    """
    # Handle multi-dimensional tensors (batch, celltype, length)
    if ref.ndim == 3:
        # Assume shape [batch=1, celltype, length]
        ref = ref[0, 0, :]  # Take first batch, first celltype
        alt = alt[0, 0, :]
    else:
        ref = ref.squeeze()
        alt = alt.squeeze()

    # Slice to core region if specified
    if core_region is not None:
        start, end = core_region
        ref = ref[start:end]
        alt = alt[start:end]

    ref_np = ref.cpu().numpy()
    alt_np = alt.cpu().numpy()

    # Max difference (primary metric)
    # IMPORTANT: ref - alt to match run_analysis_dlbcl.py sign convention
    max_diff = float(ref_np.max() - alt_np.max())

    # Mean difference
    mean_diff = float(ref_np.mean() - alt_np.mean())

    # Correlation
    if len(ref_np) > 1:
        correlation = float(np.corrcoef(ref_np, alt_np)[0, 1])
    else:
        correlation = 1.0

    # Jensen-Shannon distance
    js_dist = jensen_shannon_distance(ref_np, alt_np)

    return {
        'primary_metric': max_diff,
        'max_difference': max_diff,
        'mean_difference': mean_diff,
        'correlation': correlation,
        'js_distance': js_dist,
        'ref_max': float(ref_np.max()),
        'alt_max': float(alt_np.max()),
    }


def calculate_count_effects(ref: torch.Tensor,
                            alt: torch.Tensor) -> Dict[str, float]:
    """
    Calculate effects for count head (scalar or summed output).

    Metrics:
    - log_fold_change: log2((alt + 1) / (ref + 1)) - primary metric
    - fold_change: alt / ref
    - difference: alt - ref

    Args:
        ref: Reference count (scalar or tensor)
        alt: Alternative count (scalar or tensor)

    Returns:
        Dict with effect metrics
    """
    # Sum if multi-dimensional, otherwise get scalar
    ref_val = ref.sum().item() if ref.ndim > 0 else ref.item()
    alt_val = alt.sum().item() if alt.ndim > 0 else alt.item()

    # Log fold change (primary metric)
    log_fc = float(np.log2((alt_val + 1) / (ref_val + 1)))

    # Fold change
    fc = float(alt_val / (ref_val + 1e-8))

    # Difference
    diff = float(alt_val - ref_val)

    return {
        'primary_metric': log_fc,
        'log_fold_change': log_fc,
        'fold_change': fc,
        'difference': diff,
        'ref_value': ref_val,
        'alt_value': alt_val,
    }


def calculate_fc_effects(ref: torch.Tensor,
                        alt: torch.Tensor) -> Dict[str, float]:
    """
    Calculate effects for fold-change/differential head (scalar output).

    Metrics:
    - difference: alt - ref (primary metric for differential)
    - abs_difference: |alt - ref|

    Args:
        ref: Reference differential value (scalar)
        alt: Alternative differential value (scalar)

    Returns:
        Dict with effect metrics
    """
    ref_val = ref.item() if ref.ndim == 0 else ref.mean().item()
    alt_val = alt.item() if alt.ndim == 0 else alt.mean().item()

    diff = float(alt_val - ref_val)

    return {
        'primary_metric': diff,
        'difference': diff,
        'abs_difference': abs(diff),
        'ref_value': ref_val,
        'alt_value': alt_val,
    }


def calculate_hybrid_effects(ref1: torch.Tensor, alt1: torch.Tensor,
                             ref2: torch.Tensor, alt2: torch.Tensor,
                             head1_name: str, head2_name: str,
                             core_region: Optional[tuple] = None) -> Dict[str, float]:
    """
    Calculate hybrid effects combining two output heads.

    Implements hybrid scoring modes like:
    - regression_differential: |regression_max| * |differential|
    - count_differential: |count| * |differential|

    This matches the behavior in run_analysis_dlbcl.py where deletion effects
    are scored using the product of absolute values from two heads.

    Args:
        ref1: Reference output from first head
        alt1: Alternative output from first head
        ref2: Reference output from second head (typically differential)
        alt2: Alternative output from second head (typically differential)
        head1_name: Name of first head ('regression', 'count', etc.)
        head2_name: Name of second head (typically 'differential')
        core_region: Optional (start, end) indices to slice regression outputs

    Returns:
        Dict with hybrid effect metrics

    Note:
        Effect sign convention: ref - alt (matches run_analysis_dlbcl.py)
        Positive effect means deletion reduced the hybrid score
    """
    # Calculate individual head values
    if head1_name == 'regression':
        # For regression: slice to core region first, then take max
        # Handle multi-dimensional tensors (batch, celltype, length)
        if ref1.ndim == 3:
            ref1 = ref1[0, 0, :]  # Take first batch, first celltype
            alt1 = alt1[0, 0, :]
        else:
            ref1 = ref1.squeeze()
            alt1 = alt1.squeeze()

        if core_region is not None:
            start, end = core_region
            ref1 = ref1[start:end]
            alt1 = alt1[start:end]

        ref1_val = ref1.max().item()
        alt1_val = alt1.max().item()
    elif head1_name == 'count':
        # For count: use scalar or sum
        ref1_val = ref1.sum().item() if ref1.ndim > 0 else ref1.item()
        alt1_val = alt1.sum().item() if alt1.ndim > 0 else alt1.item()
    else:
        # Generic: use mean
        ref1_val = ref1.mean().item()
        alt1_val = alt1.mean().item()

    # Second head (typically differential): use scalar or mean
    if ref2.ndim == 0 or ref2.numel() == 1:
        ref2_val = ref2.item()
        alt2_val = alt2.item()
    else:
        # For multi-element tensors, take first element (batch=1, celltype=1 case)
        ref2_val = ref2.flatten()[0].item()
        alt2_val = alt2.flatten()[0].item()

    # Calculate hybrid scores: |head1| * |head2|
    # This matches run_analysis_dlbcl.py lines 1471-1476, 1514-1518
    ref_hybrid = abs(ref1_val) * abs(ref2_val)
    alt_hybrid = abs(alt1_val) * abs(alt2_val)

    # Effect is the difference in hybrid scores
    # IMPORTANT: ref - alt to match run_analysis_dlbcl.py sign convention
    hybrid_diff = float(ref_hybrid - alt_hybrid)

    return {
        'primary_metric': hybrid_diff,
        'hybrid_score_diff': hybrid_diff,
        'ref_hybrid_score': ref_hybrid,
        'alt_hybrid_score': alt_hybrid,
        f'ref_{head1_name}': ref1_val,
        f'alt_{head1_name}': alt1_val,
        f'ref_{head2_name}': ref2_val,
        f'alt_{head2_name}': alt2_val,
    }


def calculate_generic_effects(ref: torch.Tensor,
                              alt: torch.Tensor) -> Dict[str, float]:
    """
    Generic effect calculation for unknown head types.

    Attempts to determine appropriate metrics based on tensor shape.

    Args:
        ref: Reference output
        alt: Alternative output

    Returns:
        Dict with generic effect metrics
    """
    # Flatten tensors
    ref_flat = ref.flatten().cpu().numpy()
    alt_flat = alt.flatten().cpu().numpy()

    # Calculate basic metrics
    mean_diff = float(alt_flat.mean() - ref_flat.mean())
    max_diff = float(alt_flat.max() - ref_flat.max())

    return {
        'primary_metric': mean_diff,
        'mean_difference': mean_diff,
        'max_difference': max_diff,
    }


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon distance between two distributions.

    JS distance is the square root of JS divergence, providing
    a symmetric measure of similarity between probability distributions.

    Args:
        p: First distribution (will be normalized)
        q: Second distribution (will be normalized)

    Returns:
        JS distance (0 = identical, 2 = completely different)
    """
    # Ensure non-negative and normalize
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    p = p / p.sum()
    q = q / q.sum()

    # Calculate midpoint
    m = 0.5 * (p + q)

    # KL divergences
    kl_p_m = np.sum(p * np.log(p / m))
    kl_q_m = np.sum(q * np.log(q / m))

    # JS divergence
    js_div = 0.5 * kl_p_m + 0.5 * kl_q_m

    # JS distance (square root)
    js_dist = np.sqrt(js_div)

    return float(js_dist)