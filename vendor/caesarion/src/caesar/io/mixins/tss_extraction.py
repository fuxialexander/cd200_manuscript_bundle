"""TSS extraction mixin for genomic IO classes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from gcell.rna.gencode import Gencode


class TSSExtractionMixin:
    """
    Mixin for computing and writing TSS tracks in parallel.
    """

    def compute_tss_track(self, gene_name: str, gencode: Gencode, **track_kwargs):
        """
        Computes the TSS track for a given gene.

        Parameters:
        - gene_name: Name of the gene
        - gencode: Gencode reference containing gene data
        - **track_kwargs: Additional arguments passed to get_track_obj()

        Returns:
        - Tuple (gene_name, tracks) where tracks is the computed TSS track or None if error
        """
        try:
            gene = gencode.get_gene(gene_name)
            tracks = gene.get_tss_track_obj(
                track=self, upstream=1000, downstream=1000, **track_kwargs
            )
            return (gene_name, tracks)
        except Exception as e:
            print(f"Error processing gene {gene_name}: {e}")
            return (gene_name, None)

    def write_tf_tss_tracks_parallel(
        self, tf_list, gencode, overwrite=False, n_jobs=64, **track_kwargs
    ):
        """
        Writes TF TSS tracks in parallel.

        Parameters:
        - tf_list: List of transcription factors to process
        - gencode: Gencode reference containing gene data
        - overwrite: Whether to overwrite existing dataset
        - n_jobs: Number of parallel jobs
        - **track_kwargs: Additional arguments passed to get_track_obj()
        """
        if self.mode != "a":
            raise ValueError("Cannot write in read-only mode")

        # Create a list of dictionaries containing track_kwargs for each TF
        track_kwargs_list = [track_kwargs] * len(tf_list)

        with ProcessPoolExecutor(n_jobs) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self.compute_tss_track,
                        tf_list,
                        [gencode] * len(tf_list),
                        track_kwargs_list,
                    ),
                    total=len(tf_list),
                )
            )

        # Aggregating results
        tss_tracks = []
        tf_gene_list = []
        for gene_name, tracks in results:
            if tracks is not None:
                tf_gene_list.append(gene_name)
                tss_tracks.append(tracks)

        # Stack and store in dataset
        tss_tracks = np.vstack(tss_tracks)
        self.dataset.create_array(
            "tf_tss_tracks", data=tss_tracks, overwrite=overwrite
        )
        self.dataset["tf_tss_tracks"].attrs["tf_list"] = tf_gene_list
