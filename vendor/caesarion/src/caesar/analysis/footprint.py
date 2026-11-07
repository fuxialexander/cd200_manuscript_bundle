# plot_utils.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from scipy.ndimage import gaussian_filter


def min_max_scale(x):
    a = (x - x.min()) / (x.max() - x.min())
    return a / a.var()


def conv_10(x):
    return np.convolve(min_max_scale(x), np.ones(10) / 10, mode="same")


def plot_footprint_comparison(
    fp1: np.ndarray,
    fp2: np.ndarray,
    labels: list[str] = None,
    regions: tuple = (800, 1200),
) -> plt.Figure:
    """
    Compare footprint profiles between two datasets.

    Parameters:
        fp1: First footprint array (n_motifs x positions)
        fp2: Second footprint array (n_motifs x positions)
        labels: Dataset labels for legend
        regions: Position range to analyze

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, fp in enumerate([fp1, fp2]):
        profile = fp[..., regions[0] : regions[1]].mean(axis=(0, 2))
        sns.lineplot(
            x=np.arange(len(profile)),
            y=profile,
            ax=ax,
            label=labels[i] if labels else f"Dataset {i + 1}",
        )

    ax.set(
        xlabel="Position",
        ylabel="Mean Accessibility",
        title="Footprint Profile Comparison",
    )
    ax.legend()
    return fig


def plot_motif_surface(fp, motif_idx, m, conv_10=conv_10, expand=200):
    """
    Generates a surface plot for motif scores across cutoff and position values.

    Parameters:
    - fp: List of motif score arrays.
    - motif_idx: Index of the motif in the fp array.
    - m: Object containing motif metadata (e.g., motif lengths).
    - conv_10: Function to apply convolution on the motif score arrays.

    Returns:
    - Z_smooth: Smoothed 2D numpy array of scores.
    """

    # Define x (cutoff) and y (position) ranges
    x = np.arange(5, 10, 1)  # Cutoff values
    y = np.arange(0, 2 * expand)  # Position values
    X, Y = np.meshgrid(x, y)

    motif_length = m.motif_mode_lengths[motif_idx * 2]
    Z = []

    for i in range(len(x)):
        if len(fp[i][motif_idx * 2]) > 0:  # Ensure array is not empty
            center_pos = 200 + motif_length // 2
            center_neg = 200 - motif_length // 2

            z_values = conv_10(
                fp[i][motif_idx * 2][center_pos - expand : center_pos + expand]
            ) + conv_10(
                fp[i][motif_idx * 2 + 1][center_neg - expand : center_neg + expand][
                    ::-1
                ]
            )
            Z.append(z_values)
        else:
            Z.append(np.zeros(len(y)))  # Fill with zeros if empty

    Z = np.array(Z).T  # Transpose to match X and Y dimensions

    # Apply Gaussian smoothing
    Z_smooth = gaussian_filter(Z.astype(np.float32), sigma=(1, 0.5))

    # Create a surface plot
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z_smooth, opacity=1)])

    fig.update_layout(
        scene=dict(xaxis_title="Cutoff", yaxis_title="Position", zaxis_title="Score"),
        width=800,
        height=800,
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    )

    fig.show()
    return Z_smooth  # Return the smoothed Z matrix
