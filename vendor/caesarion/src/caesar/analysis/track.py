"""
Module for handling genomic tracks.

Classes
-------
Track: A class to represent and manipulate genomic track data.

Functions
---------
get_iqr_threshold: Get the IQR threshold for a specific track for peak calling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from matplotlib import gridspec

from gcell.dna.genome import ChromSize

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from caesar.io.normalizer import Normalizer


def get_iqr_threshold(
    track, min_length=100, merge_length=100, return_indices=True, iqr_factor=1.2
):
    """
    Get the IQR threshold for a specific track for peak calling.

    Parameters
    ----------
    track : np.ndarray
        The track data.
    min_length : int, optional
        The minimum length of a peak. Defaults to 100.
    merge_length : int, optional
        The maximum distance to merge consecutive peaks. Defaults to 100.
    return_indices : bool, optional
        Whether to return the indices of the peaks. Defaults to True.
    """
    Q1, Q3 = np.percentile(track, [25, 75])
    IQR = Q3 - Q1
    threshold = Q3 + iqr_factor * IQR
    if not return_indices:
        return threshold
    idx = np.where(track > threshold)[0]
    starts = []
    ends = []
    from itertools import groupby
    from operator import itemgetter

    for k, g in groupby(enumerate(idx), lambda ix: ix[0] - ix[1]):
        g = list(map(itemgetter(1), g))
        starts.append(g[0])
        ends.append(g[-1])
    # merge consecutive indices if they are within 100bp

    for i in range(len(starts) - 1, 0, -1):  # Start from len-1, go down to 1 (not 0)
        if starts[i] - ends[i - 1] < merge_length:
            ends[i - 1] = ends[i]
            del starts[i]
            del ends[i]
    for i in range(len(starts) - 1, -1, -1):
        if ends[i] - starts[i] < min_length:
            del starts[i]
            del ends[i]
    return np.array(list(zip(starts, ends)))


class Track:
    """
    A class to represent and manipulate genomic track data.

    This class handles genomic track data, including normalization, convolution,
    and visualization. It supports multiple tracks and provides methods for
    plotting, generating bedGraph and BigWig files.
    """

    def __init__(
        self,
        chrom: str,
        start: int,
        end: int,
        assembly: str,
        tracks: dict[str, np.ndarray | csr_matrix],  # Input tracks are raw
        normalizer: Normalizer | None = None,  # Accept a Normalizer object
        conv_size: int | None = 50,  # Keep conv_size, can be None
        metadata: dict | None = None,  # Optional metadata field
    ):
        """
        Initialize a Track object.

        Parameters
        ----------
        chrom : str
            The chromosome name.
        start : int
            The start position of the track.
        end : int
            The end position of the track.
        assembly : str
            The genome assembly name.
        tracks : dict
            A dictionary of raw track data. Keys are track labels, values are np.ndarray or csr_matrix.
        normalizer : Normalizer, optional
            A Normalizer object to apply to the tracks. Defaults to None (no normalization).
        conv_size : int | None, optional
            The size of the convolution window. If None, no convolution is applied.
            Defaults to 50.
        metadata : dict, optional
            An optional dictionary to store any metadata associated with the track.
        """
        self.chrom = chrom
        self.start = start
        self.end = end
        self.assembly = assembly
        self.metadata = metadata if metadata is not None else {}

        # 1. Apply Normalization if a normalizer is provided
        if normalizer is not None:
            # The normalizer's transform method is expected to handle various
            # data structures (dict of arrays/csr_matrices) appropriately.
            self.tracks = normalizer.transform(tracks)
        else:
            self.tracks = tracks  # Use raw tracks if no normalizer

        self.ids = list(self.tracks.keys())

        # 2. Calculate aggregated track from (potentially) normalized tracks
        # Ensure values are ndarray for sum, handle potential csr_matrix
        track_values_for_agg = []
        for t_val in self.tracks.values():
            if hasattr(t_val, "toarray"):  # For sparse matrices
                track_values_for_agg.append(t_val.toarray())
            else:
                track_values_for_agg.append(t_val)

        if track_values_for_agg:
            # Filter out None or incompatible types before stacking, if any normalizer might return them
            valid_arrays = [
                arr
                for arr in track_values_for_agg
                if isinstance(arr, np.ndarray) and arr.ndim > 0
            ]
            if valid_arrays:
                try:
                    self.tracks_agg = np.sum(np.stack(valid_arrays), axis=0)
                except ValueError as e:  # Handle case where arrays cannot be stacked (e.g. different lengths after bad transform)
                    logging.error(
                        f"Track init: Could not stack tracks for aggregation: {e}. Setting tracks_agg to empty array."
                    )
                    self.tracks_agg = np.array([])
            else:  # No valid arrays to aggregate
                self.tracks_agg = np.array([])
        else:  # No tracks to aggregate
            self.tracks_agg = np.array([])

        # 3. Apply Convolution if conv_size is provided
        self._conv_size = conv_size
        if conv_size is not None and conv_size > 0:
            kernel = np.ones(self.conv_size) / self.conv_size
            self.convoluted_tracks = {}
            for label, y_original in self.tracks.items():
                if (
                    isinstance(y_original, np.ndarray)
                    and y_original.ndim == 1
                    and y_original.size > 0
                ):
                    self.convoluted_tracks[label] = np.convolve(
                        y_original, kernel, mode="same"
                    )
                elif (
                    isinstance(y_original, np.ndarray) and y_original.size == 0
                ):  # Empty track
                    self.convoluted_tracks[label] = y_original  # Keep as empty
                else:  # Non-1D numpy array or csr_matrix - keep as is, log warning
                    logging.warning(
                        f"Track convolution for label '{label}': Data is not a 1D np.ndarray or is empty. Skipping convolution."
                    )
                    self.convoluted_tracks[label] = y_original

            if (
                isinstance(self.tracks_agg, np.ndarray)
                and self.tracks_agg.ndim == 1
                and self.tracks_agg.size > 0
            ):
                self.convoluted_tracks_agg = np.convolve(
                    self.tracks_agg, kernel, mode="same"
                )
            else:  # tracks_agg is empty or not suitable for convolution
                self.convoluted_tracks_agg = self.tracks_agg  # Keep as is
        else:  # No convolution
            self.convoluted_tracks = self.tracks
            self.convoluted_tracks_agg = self.tracks_agg

        self._sanity_check()

    def __repr__(self):
        """
        Return a string representation of the Track object.

        Returns
        -------
        str
            A string representation of the Track object.
        """
        return (
            f"Track(chrom='{self.chrom}', start={self.start}, end={self.end}, "
            f"labels={list(self.tracks.keys())}, conv_size={self.conv_size})"
        )

    def _sanity_check(self):
        """
        Perform a sanity check on the track data.

        Raises
        ------
        ValueError: If the length of any track doesn't match the expected length.
        """
        expected_length = self.end - self.start
        for label, y in self.tracks.items():
            if len(y) != expected_length:
                raise ValueError(
                    f"Length of y for {label} ({len(y)}) does not match end-start ({expected_length})"
                )

    @property
    def conv_size(self):
        """
        Get the convolution window size.

        Returns
        -------
        int
            The convolution window size.
        """
        return self._conv_size

    @conv_size.setter
    def conv_size(self, size):
        """
        Set a new convolution size and recalculate convolved tracks.

        Parameters
        ----------
        size : int
            The new convolution window size.
        """
        self._conv_size = size
        self.convoluted_tracks = {
            label: np.convolve(y, np.ones(self.conv_size) / self.conv_size, mode="same")
            for label, y in self.tracks.items()
        }

    def rename_labels(self, rename_label_dict):
        """
        Rename the labels of the tracks.
        """
        self.convoluted_tracks = {
            rename_label_dict.get(label, label): y for label, y in self.convoluted_tracks.items()
        }
        self.tracks = {
            rename_label_dict.get(label, label): y for label, y in self.tracks.items()
        }
        self.ids = list(self.tracks.keys())
        self.tracks_agg = np.sum(np.stack([y for y in self.tracks.values()]), axis=0)
        self.convoluted_tracks_agg = np.convolve(self.tracks_agg, np.ones(self.conv_size) / self.conv_size, mode="same")

    def plot_tracks(
        self,
        color_dict=None,
        gene_annot=None,
        genes_to_highlight=None,
        beds=None,
        out_file=None,
        center=False,
        track_order=None,
        autoscale=False,
        group_autoscale=None,
    ):
        """
        Plot the convolved tracks and optionally add gene annotations.

        Parameters
        ----------
        color_dict : dict, optional
            A dictionary mapping track labels to colors. Defaults to None.
        gene_annot : Gencode, optional
            A GTF object containing gene annotations. Defaults to None.
        genes_to_highlight : list, optional
            A list of gene names to highlight. Defaults to None.
        beds : list, optional
            A list of BED objects containing regions to plot. Defaults to None.
        out_file : str, optional
            The path to save the output plot. Defaults to None.
        center : bool, optional
            Whether to center the tracks around the mean. Defaults to False.
        track_order : list, optional
            List of track labels specifying the order in which to plot the tracks.
            If not provided, tracks will be plotted in their default order.
        autoscale : bool, optional
            Whether to normalize each track to its maximum value in the displayed region.
            Defaults to False.
        group_autoscale : dict or list, optional
            Groups for autoscaling. If dict, maps group names to lists of track IDs.
            If list of lists, each sublist contains track IDs for a group.
            Each group is normalized to the maximum value across group members.
            Defaults to None.

        Returns
        -------
        tuple
            A tuple containing the matplotlib figure and list of axes.
        """
        # If track_order is provided, validate and use it
        if track_order is not None:
            # Validate that all specified tracks exist
            invalid_tracks = [t for t in track_order if t not in self.convoluted_tracks]
            if invalid_tracks:
                raise ValueError(f"Invalid track(s) specified in track_order: {invalid_tracks}")
            # Use only the tracks specified in track_order
            tracks_to_plot = {k: self.convoluted_tracks[k] for k in track_order}
        else:
            tracks_to_plot = self.convoluted_tracks

        num_tracks = len(tracks_to_plot)
        additional_tracks = 0
        num_subplots = num_tracks
        if isinstance(beds, np.ndarray):
            beds = [beds]
        if beds is not None:
            num_subplots += len(beds)
            additional_tracks += len(beds)

        if gene_annot is not None and (
            self.end - self.start < 1e6 or genes_to_highlight is not None
        ):
            num_subplots += 1
            additional_tracks += 1

        fig = plt.figure(figsize=(8, 1 * num_tracks + 0.2 * additional_tracks))
        gs = gridspec.GridSpec(
            num_subplots, 1, height_ratios=[1] * num_tracks + [0.2] * additional_tracks
        )

        # Adjust the space between categories
        gs.update(hspace=0.5)

        if color_dict is None:
            colors = plt.cm.tab10.colors
            color_dict = {
                label: colors[i % 10]
                for i, label in enumerate(tracks_to_plot.keys())
            }

        # Handle autoscaling
        scaling_factors = {}
        group_names = {}

        if group_autoscale is not None:
            # Parse group_autoscale format
            if isinstance(group_autoscale, dict):
                # Dict format: {group_name: [track_ids]}
                groups = group_autoscale
            elif isinstance(group_autoscale, list):
                # List of lists format: [[track_ids], [track_ids], ...]
                groups = {f"Group_{i+1}": group for i, group in enumerate(group_autoscale)}
            else:
                raise ValueError("group_autoscale must be a dict or list of lists")

            # Calculate scaling factors for each group
            for group_name, track_ids in groups.items():
                # Get tracks in this group that are actually being plotted
                group_tracks = {k: v for k, v in tracks_to_plot.items() if k in track_ids}
                if group_tracks:
                    # Find maximum value across all tracks in the group
                    group_max = max([max(y) for y in group_tracks.values()])
                    if group_max > 0:
                        for track_id in group_tracks.keys():
                            scaling_factors[track_id] = 1.0 / group_max
                            group_names[track_id] = group_name

        elif autoscale:
            # Individual track autoscaling
            for label, y in tracks_to_plot.items():
                track_max = max(y)
                if track_max > 0:
                    scaling_factors[label] = 1.0 / track_max

        # Apply scaling if needed
        if scaling_factors:
            tracks_to_plot = {
                label: y * scaling_factors.get(label, 1.0)
                for label, y in tracks_to_plot.items()
            }

        # Plot convoluted tracks
        max_y = max([max(y) for y in tracks_to_plot.values()])
        if center:
            mean_y = np.stack([y for y in tracks_to_plot.values()]).mean(axis=0)
            if len(mean_y) > 1e5:
                mean_y = mean_y[::100]
        axes = []
        for i, (label, y_conv) in enumerate(tracks_to_plot.items()):
            ax = fig.add_subplot(
                gs[i],
                sharex=axes[0] if axes else None,
                sharey=axes[0] if axes else None,
            )
            x = np.arange(self.start, self.end)
            # get mean per 100bp for faster plotting if the length is too long
            if len(x) > 1e5:
                x = x[::100]
                y_conv = y_conv[::100]

            if center:
                y_conv = y_conv - mean_y

            ax.fill_between(x, y_conv, color=color_dict[label])

            # Add group name if using group_autoscale with dict format
            ylabel = label
            if label in group_names:
                ylabel = f"{group_names[label]}\n{label}"

            ax.set_ylabel(
                ylabel,
                rotation=0,
                color=color_dict[label],
                ha="right",
                fontsize=12,
                va="center",
            )
            ax.set_xlabel("")
            ax.set_xticks([])
            if center:
                ax.set_ylim(bottom=-max_y, top=max_y)
            else:
                ax.set_ylim(bottom=0, top=max_y)
            ax.set_xlim(self.start, self.end)
            # ax.set_yticks([])  # no y ticks
            axes.append(ax)

        y_max_lim = fig.get_axes()[-1].get_ylim()[1]

        # Plot BED regions
        gs.update(hspace=0.2)
        if beds is not None:
            for bed in beds:
                if isinstance(bed, pd.DataFrame):
                    bed = bed.query(
                        "Chromosome==@self.chrom & Start>@self.start & End<@self.end"
                    )[["Start", "End"]].values
                filtered_bed = bed[(bed[:, 0] > self.start) & (bed[:, 1] < self.end)]
                ax_bed = fig.add_subplot(gs[-additional_tracks], sharex=axes[0])
                for start, end in filtered_bed:
                    ax_bed.fill_between(
                        [start, end], 0, y_max_lim, color="lightgray", alpha=1
                    )

                ax_bed.set_yticks([])
                ax_bed.set_ylabel("")
                ax_bed.set_xlabel("")
                ax_bed.set_xticks([])
                # ax_bed.spines['top'].set_visible(False)
                # ax_bed.spines['right'].set_visible(False)
                # ax_bed.spines['bottom'].set_visible(False)
                # ax_bed.spines['left'].set_visible(False)
                additional_tracks -= 1

        if gene_annot is not None and (
            self.end - self.start < 1e6 or genes_to_highlight is not None
        ):
            gene_color_dict = {
                gene: colors[(i + 1) % 10]
                for i, gene in enumerate(gene_annot.gtf.gene_name.unique())
            }
            gene_positions = {}
            ax_gene_pos = fig.add_subplot(gs[-1], sharex=axes[0])
            for gene, df in gene_annot.query_region(
                self.chrom, self.start, self.end
            ).groupby("gene_name"):
                if genes_to_highlight is None or gene in genes_to_highlight:
                    for i, row in df.iterrows():
                        ax_gene_pos.plot(
                            row["Start"],
                            1,
                            marker="^",
                            markersize=5,
                            color=gene_color_dict[gene],
                        )
                        gene_positions[gene] = df["Start"].mean()

            # Sort genes by their positions
            sorted_genes = sorted(gene_positions.items(), key=lambda x: x[1])

            # Calculate jittered positions
            jitter_range = (self.end - self.start) * 0.01  # 1% of the total range
            for i, (gene, position) in enumerate(sorted_genes):
                jitter = (
                    (i / (len(sorted_genes) - 1) - 0.5) * jitter_range
                    if len(sorted_genes) > 1
                    else 0
                )
                jittered_position = position + jitter
                ax_gene_pos.text(
                    jittered_position,
                    0.8,
                    gene,
                    verticalalignment="top",
                    horizontalalignment="center",
                    color=gene_color_dict[gene],
                    rotation=-45,
                    ha="left",
                    fontsize=12,
                )

            ax_gene_pos.set_yticks([])
            ax_gene_pos.set_ylabel("")
            ax_gene_pos.set_xlabel("")
            ax_gene_pos.set_xticks([])
            ax_gene_pos.set_ylim(0.5, 1)
            ax_gene_pos.spines["top"].set_visible(False)
            ax_gene_pos.spines["right"].set_visible(False)
            ax_gene_pos.spines["bottom"].set_visible(False)
            ax_gene_pos.spines["left"].set_visible(False)

        fig.text(
            0.5,
            1,
            f"[{self.assembly}] {self.chrom}:{self.start}-{self.end}",
            ha="center",
        )

        plt.tight_layout()
        if out_file is not None:
            plt.savefig(out_file)
        else:
            plt.show()

        return fig, axes

    def plot_tracks_with_motif_density(
        self,
        motif_zarr_path,
        cutoff_quantile=0.99,
        conv_size=100,
        motif_index=None,
        **kwargs,
    ):
        """
        Plot tracks with motif density.

        Parameters
        ----------
        motif_zarr_path : str
            The path to the motif Zarr file.
        cutoff_quantile : float, optional
            The quantile to cutoff the motif density. Defaults to 0.99.
        conv_size : int, optional
            The size of the convolution window. Defaults to 100.
        motif_index : int, optional
            The index of the motif to plot. Defaults to None.
        cutoff_quantile : float, optional
            The quantile to cutoff the motif density. Defaults to 0.99.
        conv_size : int, optional
            The size of the convolution window. Defaults to 100.
        """
        # Get motif density
        try:
            from caesar.io.zarr_io import DenseZarrIO
        except ImportError:
            "caesar is not installed, this function is not available"
            return
        m = DenseZarrIO(motif_zarr_path)
        motif_density = m.get_track(self.chrom, self.start, self.end)
        for i, quantile in enumerate(
            np.quantile(motif_density, cutoff_quantile, axis=0)
        ):
            motif_density[:, i][motif_density[:, i] < quantile] = quantile
            motif_density[:, i] = motif_density[:, i] - quantile
        if motif_index is not None:
            motif_density = motif_density[:, motif_index]
        else:
            motif_density = motif_density.sum(axis=1)

        # Convolve motif density
        motif_density = np.convolve(
            motif_density, np.ones((conv_size,)) / conv_size, mode="same"
        )

        # Add motif density to tracks
        self.convoluted_tracks["motif_density"] = (
            motif_density / motif_density.max() * self.convoluted_tracks_agg.max()
        )

        # Plot tracks with motif density
        self.plot_tracks(**kwargs)

    def get_local_iqr_peak(
        self, label, min_length=100, merge_length=100, iqr_factor=1.2
    ):
        """
        Get local IQR peaks for a specific track.

        Parameters
        ----------
        label : str
            The label of the track to get the local IQR peaks for, or 'agg' for the aggregated track.
        min_length : int, optional
            The minimum length of a peak. Defaults to 100.
        merge_length : int, optional
            The maximum distance to merge consecutive peaks. Defaults to 100.
        return_indices : bool, optional
            Whether to return the indices of the peaks. Defaults to True.

        Returns
        -------
        np.ndarray: An array of local IQR peaks.
        """
        if label == "agg":
            y = self.convoluted_tracks_agg
        else:
            y = self.convoluted_tracks[label]
        return (
            get_iqr_threshold(
                y,
                min_length=min_length,
                merge_length=merge_length,
                return_indices=True,
                iqr_factor=iqr_factor,
            )
            + self.start
        )

    def generate_bedgraph(self, label, out_file):
        """
        Generate a bedGraph file for a specific track.

        Parameters
        ----------
        label : str
            The label of the track to generate the bedGraph for.
        out_file : str
            The output file path.

        Raises
        ------
        ValueError: If the specified label is not found in the tracks.
        """
        if label not in self.tracks:
            raise ValueError(f"Label {label} not found in tracks")

        with Path(out_file).open("w") as f:
            for i, v in enumerate(self.convoluted_tracks[label]):
                f.write(f"{self.chrom}\t{i + self.start}\t{i + self.start + 1}\t{v}\n")

    def generate_bigwig(self, label, out_file):
        """
        Generate a BigWig file for a specific track.

        Parameters
        ----------
        label : str
            The label of the track to generate the BigWig for.
        out_file : str
            The output file path.

        Raises
        ------
        ValueError: If the specified label is not found in the tracks.
        """
        if label not in self.tracks:
            raise ValueError(f"Label {label} not found in tracks")

        # Create a new BigWig file
        bw = pyBigWig.open(out_file, "w")

        # load chrom_size
        chrom_sizes = ChromSize(self.assembly, "../../data").dict

        # Add chromosome sizes (you may need to adjust this based on your assembly)
        bw.addHeader(list(chrom_sizes.items()))

        # Prepare the data
        starts = list(range(self.start, self.end))
        values = self.convoluted_tracks[label]
        mask = values > 0
        filtered_starts = np.array(starts)[mask]
        filtered_ends = filtered_starts + 1
        filtered_values = values[mask]

        # Convert to lists explicitly
        chroms = [str(self.chrom) for _ in range(len(filtered_starts))]
        starts_list = [int(x) for x in filtered_starts]
        ends_list = [int(x) for x in filtered_ends]
        values_list = [float(x) for x in filtered_values]

        # Add entries to the BigWig file
        bw.addEntries(chroms, starts_list, ends=ends_list, values=values_list)

        # Close the file
        bw.close()

    def liftover(self, to_assembly):
        """
        Liftover the track to a different assembly.
        """
        # has to be same species
        if self.assembly[0:2] != to_assembly[0:2]:
            raise ValueError(
                f"Cross species liftover is not supported, {self.assembly} != {to_assembly}"
            )
        import pyliftover

        liftover = pyliftover.LiftOver(self.assembly, to_assembly)
        new_start = liftover.convert_coordinate(self.chrom, self.start)[0][1]
        new_end = liftover.convert_coordinate(self.chrom, self.end)[0][1]
        if new_end - new_start != self.end - self.start:
            raise ValueError(
                f"The length of the track is not the same after liftover, {new_end - new_start} != {self.end - self.start}"
            )
        self.start = new_start
        self.end = new_end
        self.assembly = to_assembly

    def plot_tracks_with_genebody(
        self,
        color_dict=None,
        gene_annot=None,
        genes_to_highlight=None,
        beds=None,
        out_file=None,
        center=False,
        protein_coding_only=False,
        isoforms=False,
        gene_track_height=1,
        track_order=None,
        autoscale=False,
        group_autoscale=None,
    ):
        """
        Plot the tracks with detailed gene structure visualization.

        Parameters
        ----------
        color_dict : dict, optional
            A dictionary mapping track labels to colors. Defaults to None.
        gene_annot : Gencode, optional
            A GTF object containing gene annotations. Defaults to None.
        genes_to_highlight : list, optional
            A list of gene names to highlight. Defaults to None.
        beds : list, optional
            A list of BED objects containing regions to plot. Defaults to None.
        out_file : str, optional
            The path to save the output plot. Defaults to None.
        center : bool, optional
            Whether to center the y-axis around the mean. Defaults to False.
        protein_coding_only : bool, optional
            Whether to show only protein-coding genes. Defaults to False.
        isoforms : bool, optional
            Whether to show individual transcript isoforms. Defaults to False.
        gene_track_height : float, optional
            Height of the gene track relative to signal tracks. Defaults to 1.
        track_order : list, optional
            List of track labels specifying the order in which to plot the tracks.
            If not provided, tracks will be plotted in their default order.
        autoscale : bool, optional
            Whether to normalize each track to its maximum value in the displayed region.
            Defaults to False.
        group_autoscale : dict or list, optional
            Groups for autoscaling. If dict, maps group names to lists of track IDs.
            If list of lists, each sublist contains track IDs for a group.
            Each group is normalized to the maximum value across group members.
            Defaults to None.

        Returns
        -------
        tuple
            A tuple containing the matplotlib figure and list of axes.
        """
        # If track_order is provided, validate and use it
        if track_order is not None:
            # Validate that all specified tracks exist
            invalid_tracks = [t for t in track_order if t not in self.convoluted_tracks]
            if invalid_tracks:
                raise ValueError(f"Invalid track(s) specified in track_order: {invalid_tracks}")
            # Use only the tracks specified in track_order
            tracks_to_plot = {k: self.convoluted_tracks[k] for k in track_order}
        else:
            tracks_to_plot = self.convoluted_tracks

        num_tracks = len(tracks_to_plot)
        additional_tracks = 0
        num_subplots = num_tracks
        if isinstance(beds, np.ndarray):
            beds = [beds]
        if beds is not None:
            num_subplots += len(beds)
            additional_tracks += len(beds)

        # Calculate how much space to allocate for gene visualization
        if gene_annot is not None:
            num_subplots += 1
            additional_tracks += 1

        fig = plt.figure(
            figsize=(8, 1 * num_tracks + gene_track_height * additional_tracks)
        )

        # Create heights for GridSpec - track plots get height 1, gene plot gets more
        height_ratios = [1] * num_tracks
        if beds is not None:
            height_ratios.extend([0.2] * len(beds))
        if gene_annot is not None:
            height_ratios.append(gene_track_height)

        gs = gridspec.GridSpec(num_subplots, 1, height_ratios=height_ratios)

        # Adjust the space between categories
        gs.update(hspace=0.5)

        if color_dict is None:
            colors = plt.cm.tab10.colors
            color_dict = {
                label: colors[i % 10]
                for i, label in enumerate(tracks_to_plot.keys())
            }

        # Handle autoscaling
        scaling_factors = {}
        group_names = {}

        if group_autoscale is not None:
            # Parse group_autoscale format
            if isinstance(group_autoscale, dict):
                # Dict format: {group_name: [track_ids]}
                groups = group_autoscale
            elif isinstance(group_autoscale, list):
                # List of lists format: [[track_ids], [track_ids], ...]
                groups = {f"Group_{i+1}": group for i, group in enumerate(group_autoscale)}
            else:
                raise ValueError("group_autoscale must be a dict or list of lists")

            # Calculate scaling factors for each group
            for group_name, track_ids in groups.items():
                # Get tracks in this group that are actually being plotted
                group_tracks = {k: v for k, v in tracks_to_plot.items() if k in track_ids}
                if group_tracks:
                    # Find maximum value across all tracks in the group
                    group_max = max([max(y) for y in group_tracks.values()])
                    if group_max > 0:
                        for track_id in group_tracks.keys():
                            scaling_factors[track_id] = 1.0 / group_max
                            group_names[track_id] = group_name

        elif autoscale:
            # Individual track autoscaling
            for label, y in tracks_to_plot.items():
                track_max = max(y)
                if track_max > 0:
                    scaling_factors[label] = 1.0 / track_max

        # Apply scaling if needed
        if scaling_factors:
            tracks_to_plot = {
                label: y * scaling_factors.get(label, 1.0)
                for label, y in tracks_to_plot.items()
            }

        # Plot convoluted tracks
        max_y = max([max(y) for y in tracks_to_plot.values()])
        if center:
            mean_y = np.stack([y for y in tracks_to_plot.values()]).mean(axis=0)
            if len(mean_y) > 1e5:
                mean_y = mean_y[::100]

        axes = []
        for i, (label, y_conv) in enumerate(tracks_to_plot.items()):
            ax = fig.add_subplot(
                gs[i],
                sharex=axes[0] if axes else None,
                sharey=axes[0] if axes else None,
            )
            x = np.arange(self.start, self.end)
            # get mean per 100bp for faster plotting if the length is too long
            if len(x) > 1e5:
                x = x[::100]
                y_conv = y_conv[::100]

            if center:
                y_conv = y_conv - mean_y

            ax.fill_between(x, y_conv, color=color_dict[label])

            # Add group name if using group_autoscale with dict format
            ylabel = label
            if label in group_names:
                ylabel = f"{group_names[label]}\n{label}"

            ax.set_ylabel(
                ylabel,
                rotation=0,
                color=color_dict[label],
                ha="right",
                fontsize=12,
                va="center",
            )
            ax.set_xlabel("")
            ax.set_xticks([])
            if center:
                ax.set_ylim(bottom=-max_y, top=max_y)
            else:
                ax.set_ylim(bottom=0, top=max_y)
            ax.set_xlim(self.start, self.end)
            axes.append(ax)

        y_max_lim = fig.get_axes()[-1].get_ylim()[1]

        # Plot BED regions
        gs.update(hspace=0.2)
        if beds is not None:
            for bed in beds:
                if isinstance(bed, pd.DataFrame):
                    bed = bed.query(
                        "Chromosome==@self.chrom & Start>@self.start & End<@self.end"
                    )[["Start", "End"]].values
                filtered_bed = bed[(bed[:, 0] > self.start) & (bed[:, 1] < self.end)]
                ax_bed = fig.add_subplot(gs[-additional_tracks], sharex=axes[0])
                for start, end in filtered_bed:
                    ax_bed.fill_between(
                        [start, end], 0, y_max_lim, color="lightgray", alpha=1
                    )

                ax_bed.set_yticks([])
                ax_bed.set_ylabel("")
                ax_bed.set_xlabel("")
                ax_bed.set_xticks([])
                additional_tracks -= 1

        # Plot gene structures if annotation is provided
        if gene_annot is not None:
            gtf = gene_annot.original_gtf
            ax_gene = fig.add_subplot(gs[-1], sharex=axes[0])

            # Filter GTF for the region and feature type
            region_gtf = gtf[
                (gtf["Chromosome"] == self.chrom)
                & (gtf["End"] >= self.start)
                & (gtf["Start"] <= self.end)
            ].copy()

            # Filter for protein-coding genes if requested
            if protein_coding_only:
                if "gene_type" in region_gtf.columns:
                    region_gtf = region_gtf[region_gtf["gene_type"] == "protein_coding"]
                elif "gene_biotype" in region_gtf.columns:
                    region_gtf = region_gtf[
                        region_gtf["gene_biotype"] == "protein_coding"
                    ]

            # Filter for specific genes if a list is provided
            if genes_to_highlight is not None:
                region_gtf = region_gtf[
                    region_gtf["gene_name"].isin(genes_to_highlight)
                ]

            if not region_gtf.empty:
                # Get unique genes
                gene_rows = region_gtf[region_gtf["Feature"] == "gene"]

                # Create a list of gene/transcript info
                feature_info = []

                if isoforms:
                    # For each gene, get all its transcripts
                    for _, gene_row in gene_rows.iterrows():
                        gene_name = gene_row["gene_name"]
                        gene_data = region_gtf[
                            region_gtf["gene_id"] == gene_row["gene_id"]
                        ]

                        # Get all transcripts for this gene
                        transcripts = gene_data[gene_data["Feature"] == "transcript"]

                        for _, transcript in transcripts.iterrows():
                            feature_info.append(
                                {
                                    "name": transcript.get(
                                        "transcript_name", transcript["transcript_id"]
                                    ),
                                    "gene_name": gene_name,
                                    "start": max(self.start, transcript["Start"]),
                                    "end": min(self.end, transcript["End"]),
                                    "strand": transcript["Strand"],
                                    "data": gene_data[
                                        gene_data["transcript_id"]
                                        == transcript["transcript_id"]
                                    ],
                                }
                            )
                else:
                    # Just use gene-level information
                    for _, gene_row in gene_rows.iterrows():
                        gene_name = gene_row["gene_name"]
                        gene_data = region_gtf[
                            region_gtf["gene_id"] == gene_row["gene_id"]
                        ]
                        feature_info.append(
                            {
                                "name": gene_name,
                                "gene_name": gene_name,
                                "start": max(self.start, gene_row["Start"]),
                                "end": min(self.end, gene_row["End"]),
                                "strand": gene_row["Strand"],
                                "data": gene_data,
                            }
                        )

                # Calculate optimal track assignments to minimize vertical space
                tracks = []
                track_assignments = {}

                # Helper function to check if a feature overlaps with any feature in a track
                def overlaps_with_track(feature, track, padding=0.1):
                    feature_start = feature["start"]
                    feature_end = feature["end"]
                    feature_name = feature["name"]

                    # Estimate text width based on feature name length
                    name_width = len(feature_name) * (self.end - self.start) * 0.015

                    # Add padding for feature name labels
                    label_padding = (self.end - self.start) * padding

                    for existing in track:
                        existing_name = existing["name"]
                        existing_name_width = (
                            len(existing_name) * (self.end - self.start) * 0.015
                        )

                        # Check feature body overlap
                        if (
                            feature_start <= existing["end"]
                            and feature_end >= existing["start"]
                        ):
                            return True

                        # Check label overlap if feature is outside left boundary
                        if (
                            feature_start < self.start
                            and existing["start"] < self.start
                        ):
                            # Both features have labels on right side
                            return True

                        # Check if feature starts outside left boundary but existing feature is within view
                        if (
                            feature_start < self.start
                            and existing["start"] >= self.start
                        ):
                            # Feature label will be on right, existing label on left
                            if (
                                min(self.end, feature_end) + name_width
                                > existing["start"]
                                - existing_name_width
                                - label_padding
                            ):
                                return True

                        # Check if existing feature starts outside left boundary but current feature is within view
                        if (
                            existing["start"] < self.start
                            and feature_start >= self.start
                        ):
                            # Existing label on right, feature label on left
                            if (
                                min(self.end, existing["end"]) + existing_name_width
                                > feature_start - name_width - label_padding
                            ):
                                return True

                        # Check label overlap if both features are inside visible range (labels on left)
                        if (
                            feature_start >= self.start
                            and existing["start"] >= self.start
                        ):
                            # Both have labels on left
                            left_extent1 = feature_start - name_width - label_padding
                            left_extent2 = (
                                existing["start"] - existing_name_width - label_padding
                            )

                            # Check if either left extent overlaps with the other feature's body
                            if (
                                left_extent1 <= existing["end"]
                                and feature_start >= existing["start"]
                            ) or (
                                left_extent2 <= feature_end
                                and existing["start"] >= feature_start
                            ):
                                return True

                            # Check if the left extents themselves overlap
                            if (
                                left_extent1 <= existing["start"]
                                and feature_start >= existing["start"]
                            ) and abs(feature_start - existing["start"]) < max(
                                name_width, existing_name_width
                            ) + label_padding:
                                return True

                        # Check overlap for features with labels on right side
                        if feature_end <= self.end and existing["end"] <= self.end:
                            # Check if there's overlap in the right label zones
                            right_extent1 = feature_end + name_width + label_padding
                            right_extent2 = (
                                existing["end"] + existing_name_width + label_padding
                            )

                            if (
                                feature_end <= existing["end"]
                                and right_extent1 >= existing["end"]
                            ) or (
                                existing["end"] <= feature_end
                                and right_extent2 >= feature_end
                            ):
                                return True

                    return False

                # Assign each feature to the first available track that doesn't cause overlap
                for feature in feature_info:
                    assigned = False

                    for i, track in enumerate(tracks):
                        if not overlaps_with_track(feature, track):
                            track.append(feature)
                            track_assignments[feature["name"]] = i
                            assigned = True
                            break

                    if not assigned:
                        # Create a new track
                        tracks.append([feature])
                        track_assignments[feature["name"]] = len(tracks) - 1

                num_gene_tracks = len(tracks)

                # Track height parameters
                track_height = 0.2
                exon_height = 0.4 * track_height
                cds_height = 0.8 * track_height

                # Top-justify genes
                # Use a smaller spacing for tracks to compact them at the top
                track_spacing = 0.5

                # Set y-axis limits to create a small fixed height regardless of gene count
                max_tracks_to_show = num_gene_tracks if num_gene_tracks > 0 else 1
                ax_gene.set_ylim(-0.2, max_tracks_to_show * track_spacing)

                if num_gene_tracks > 0:
                    # Plot each feature
                    for feature in feature_info:
                        feature_name = feature["name"]
                        gene_name = feature["gene_name"]
                        feature_data = feature["data"]
                        feature_start = feature["start"]
                        feature_end = feature["end"]
                        strand = feature["strand"]

                        # Calculate y position from top (0) down
                        y_pos = track_assignments[feature_name] * track_spacing

                        # Plot feature body as a line
                        ax_gene.plot(
                            [
                                max(self.start, feature_start),
                                min(self.end, feature_end),
                            ],
                            [y_pos, y_pos],
                            "k-",
                            linewidth=1,
                        )

                        # Plot exons as boxes
                        exons = feature_data[feature_data["Feature"] == "exon"]
                        for _, exon in exons.iterrows():
                            # Only plot parts of exons within the range
                            exon_start = max(self.start, exon["Start"])
                            exon_end = min(self.end, exon["End"])
                            if exon_start < exon_end:  # Only if visible
                                rect = plt.Rectangle(
                                    (exon_start, y_pos - exon_height / 2),
                                    exon_end - exon_start,
                                    exon_height,
                                    facecolor="grey",
                                    alpha=1,
                                    linewidth=1,
                                    edgecolor="grey",
                                )
                                ax_gene.add_patch(rect)

                        # Plot CDS as wider bars
                        cds = feature_data[feature_data["Feature"] == "CDS"]
                        for _, cds_region in cds.iterrows():
                            # Only plot parts of CDS within the range
                            cds_start = max(self.start, cds_region["Start"])
                            cds_end = min(self.end, cds_region["End"])
                            if cds_start < cds_end:  # Only if visible
                                rect = plt.Rectangle(
                                    (cds_start, y_pos - cds_height / 2),
                                    cds_end - cds_start,
                                    cds_height,
                                    facecolor="grey",
                                    alpha=1,
                                    linewidth=1,
                                    edgecolor="grey",
                                )
                                ax_gene.add_patch(rect)

                        # Add strand direction markers
                        num_arrows = min(
                            8,
                            max(
                                3,
                                int(
                                    (feature_end - feature_start)
                                    / ((self.end - self.start) / 40)
                                ),
                            ),
                        )
                        visible_feature_start = max(self.start, feature_start)
                        visible_feature_end = min(self.end, feature_end)
                        arrow_positions = np.linspace(
                            visible_feature_start, visible_feature_end, num_arrows
                        )
                        arrow_size = (
                            self.end - self.start
                        ) * 0.005  # Smaller arrows relative to viewport

                        for pos in arrow_positions[:-1]:
                            if strand == "+":
                                ax_gene.plot(
                                    [pos, pos + arrow_size],
                                    [y_pos + track_height / 8, y_pos],
                                    "k-",
                                    linewidth=1,
                                )
                                ax_gene.plot(
                                    [pos, pos + arrow_size],
                                    [y_pos - track_height / 8, y_pos],
                                    "k-",

                                    linewidth=1,
                                )
                            else:
                                ax_gene.plot(
                                    [pos, pos - arrow_size],
                                    [y_pos + track_height / 8, y_pos],
                                    "k-",
                                    linewidth=1,
                                )
                                ax_gene.plot(
                                    [pos, pos - arrow_size],
                                    [y_pos - track_height / 8, y_pos],
                                    "k-",
                                    linewidth=1,
                                )

                        # Add feature name based on position
                        if feature_start < self.start:
                            # Feature starts before visible region, put label on right side
                            ax_gene.text(
                                min(self.end, feature_end)
                                + (self.end - self.start) * 0.01,
                                y_pos,
                                feature_name,
                                ha="left",
                                va="center",
                                fontsize=10,
                            )
                        else:
                            # Feature starts within visible region, put label on left side
                            ax_gene.text(
                                max(self.start, feature_start)
                                - (self.end - self.start) * 0.01,
                                y_pos,
                                feature_name,
                                ha="right",
                                va="center",
                                fontsize=10,
                            )

                # Format x-axis to show Mb
                import matplotlib.ticker as ticker

                def mb_formatter(x, pos):
                    return f"{x / 1e6:.2f} Mb"

                ax_gene.xaxis.set_major_formatter(ticker.FuncFormatter(mb_formatter))
                # ax_gene.set_ylabel("Genes", fontsize=10, rotation=0, ha='right', va='center')
                # remove y-axis label, ticks, and spines
                ax_gene.set_ylabel("")
                ax_gene.set_yticks([])
                ax_gene.spines["left"].set_visible(False)
                ax_gene.spines["top"].set_visible(False)
                ax_gene.spines["right"].set_visible(False)
                ax_gene.spines["bottom"].set_visible(False)

            else:  # No genes found
                ax_gene.set_yticks([])
                ax_gene.set_frame_on(False)

        fig.text(
            0.5,
            1,
            f"[{self.assembly}] {self.chrom}:{self.start}-{self.end}",
            ha="center",
        )

        plt.tight_layout()
        if out_file is not None:
            plt.savefig(out_file, transparent=True, bbox_inches="tight")
        else:
            plt.show()

        return fig, axes
