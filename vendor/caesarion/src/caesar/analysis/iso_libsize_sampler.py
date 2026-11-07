"""Iso-libsize metacell sampler using geometric sketching and KNN expansion.

This module provides functionality to create balanced metacells with:
1. Even coverage across cell state space (via geometric sketching in PCA/UMAP)
2. Consistent ATAC sequencing depth per metacell (iso-libsize)

The algorithm:
1. Select seed cells evenly distributed in embedding space (geometric sketching)
2. For each seed, expand to KNN neighbors until target library size is reached
3. Avoid overlapping cells between metacells
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans

if TYPE_CHECKING:
    import anndata
    from caesar.io.zarr_io.multiome import MultiomeSparseZarrIO

logger = logging.getLogger(__name__)


@dataclass
class Metacell:
    """A metacell consisting of a group of cells."""

    seed_barcode: str
    cell_barcodes: list[str]
    cell_indices: np.ndarray
    total_libsize: int
    n_cells: int

    def __repr__(self):
        return f"Metacell(seed={self.seed_barcode}, n_cells={self.n_cells}, libsize={self.total_libsize:,})"


class IsoLibsizeSampler:
    """Sample metacells with uniform library size using geometric sketching.

    This sampler creates metacells that:
    - Are evenly distributed across the cell embedding space
    - Have approximately equal total ATAC library size
    - Do not share cells between metacells

    Example:
        >>> from caesar.io.zarr_io.multiome import MultiomeSparseZarrIO
        >>> from caesar.analysis.iso_libsize_sampler import IsoLibsizeSampler
        >>>
        >>> io = MultiomeSparseZarrIO('/path/to/multiome.zarr')
        >>> sampler = IsoLibsizeSampler(io, target_libsize=5_000_000)
        >>> metacells = sampler.sample(n_metacells=100)
        >>>
        >>> # Get aggregated ATAC signal for metacells
        >>> for mc in metacells:
        >>>     signal = io.get_track_for_cells(mc.cell_indices, 'chr1', 0, 1000000)
    """

    def __init__(
        self,
        multiome_io: MultiomeSparseZarrIO,
        target_libsize: int = 5_000_000,
        embedding_key: str = "X_pca",
        n_neighbors: int = 50,
        use_shared_cells_only: bool = True,
        random_state: int = 42,
    ):
        """Initialize the iso-libsize sampler.

        Args:
            multiome_io: MultiomeSparseZarrIO object with ATAC and RNA data
            target_libsize: Target total ATAC library size per metacell
            embedding_key: Key for cell embeddings in RNA adata.obsm (X_pca, X_umap, etc.)
            n_neighbors: Number of neighbors to consider for KNN expansion
            use_shared_cells_only: If True, only use cells shared between ATAC and RNA
            random_state: Random seed for reproducibility
        """
        self.io = multiome_io
        self.target_libsize = target_libsize
        self.embedding_key = embedding_key
        self.n_neighbors = n_neighbors
        self.use_shared_cells_only = use_shared_cells_only
        self.random_state = random_state

        # Load data
        self._load_data()

    def _load_data(self):
        """Load embeddings, library sizes, and build KNN index."""
        logger.info("Loading multiome data for sampling...")

        # Get RNA data with embeddings
        self.rna_adata = self.io.rna_data

        # Determine which cells to use
        if self.use_shared_cells_only:
            self.cell_barcodes = np.array(self.io.shared_barcodes)
            # Get ATAC indices for shared cells
            self.atac_indices = self.io.mapping['atac_idx'].values
        else:
            self.cell_barcodes = np.array(self.io.barcodes)
            self.atac_indices = np.arange(len(self.cell_barcodes))

        # Get library sizes for these cells (ATAC)
        atac_libsizes = np.array([
            self.io.per_cell_libsize.get(bc, 0) for bc in self.cell_barcodes
        ])
        self.libsizes = atac_libsizes

        # Get embeddings from RNA
        if self.embedding_key not in self.rna_adata.obsm:
            raise ValueError(
                f"Embedding '{self.embedding_key}' not found in RNA data. "
                f"Available: {list(self.rna_adata.obsm.keys())}"
            )

        # Get embeddings for shared cells
        if self.use_shared_cells_only:
            rna_indices = self.io.mapping['rna_idx'].values
            self.embeddings = self.rna_adata.obsm[self.embedding_key][rna_indices]
        else:
            # Need to match barcodes
            rna_bc_to_idx = {bc: i for i, bc in enumerate(self.rna_adata.obs_names)}
            valid_mask = np.array([bc in rna_bc_to_idx for bc in self.cell_barcodes])
            self.cell_barcodes = self.cell_barcodes[valid_mask]
            self.atac_indices = self.atac_indices[valid_mask]
            self.libsizes = self.libsizes[valid_mask]
            rna_idx = np.array([rna_bc_to_idx[bc] for bc in self.cell_barcodes])
            self.embeddings = self.rna_adata.obsm[self.embedding_key][rna_idx]

        logger.info(f"Loaded {len(self.cell_barcodes)} cells with embeddings")
        logger.info(f"Total ATAC library size: {self.libsizes.sum():,}")
        logger.info(f"Mean library size per cell: {self.libsizes.mean():,.0f}")

        # Build KNN index
        logger.info(f"Building KNN index with k={self.n_neighbors}...")
        self.kd_tree = cKDTree(self.embeddings)

    def _geometric_sketch(self, n_seeds: int) -> np.ndarray:
        """Select seed cells using geometric sketching (k-means based).

        Uses MiniBatchKMeans to find cluster centers, then selects the cell
        closest to each center as a seed.

        Args:
            n_seeds: Number of seed cells to select

        Returns:
            Array of seed cell indices
        """
        logger.info(f"Selecting {n_seeds} seed cells via geometric sketching...")

        # Use k-means to find representative points
        kmeans = MiniBatchKMeans(
            n_clusters=n_seeds,
            random_state=self.random_state,
            batch_size=min(1000, len(self.embeddings)),
            n_init=3,
        )
        kmeans.fit(self.embeddings)

        # Find closest cell to each cluster center
        centers = kmeans.cluster_centers_
        _, seed_indices = self.kd_tree.query(centers, k=1)

        return seed_indices.flatten()

    def _expand_to_target_libsize(
        self,
        seed_idx: int,
        assigned: np.ndarray,
        max_expansion: int = 1000,
        tolerance: float = 0.05,
    ) -> tuple[np.ndarray, int]:
        """Expand from seed cell to neighbors until target library size.

        Uses adaptive KNN querying to ensure we find enough unassigned neighbors
        to reach the target library size.

        Args:
            seed_idx: Index of the seed cell
            assigned: Boolean array of already-assigned cells
            max_expansion: Maximum number of cells to include
            tolerance: Allow stopping within this fraction of target (e.g., 0.05 = 5%)

        Returns:
            Tuple of (cell indices in metacell, total library size)
        """
        n_available = len(self.embeddings) - assigned.sum()
        if n_available < 1:
            return np.array([seed_idx]), self.libsizes[seed_idx]

        # Start with a reasonable k, expand if needed
        min_target = self.target_libsize * (1 - tolerance)

        # Estimate how many cells we might need based on mean libsize
        mean_libsize = self.libsizes[~assigned].mean() if (~assigned).any() else self.libsizes.mean()
        estimated_cells_needed = int(self.target_libsize / mean_libsize * 1.5)  # 1.5x buffer

        # Query enough neighbors, but cap at available
        query_k = min(max(estimated_cells_needed, self.n_neighbors * 20), n_available)

        _, neighbor_indices = self.kd_tree.query(
            self.embeddings[seed_idx],
            k=query_k
        )

        # Ensure neighbor_indices is 1D
        if neighbor_indices.ndim == 0:
            neighbor_indices = np.array([neighbor_indices])

        # Iteratively add neighbors until we hit target
        metacell_indices = []
        total_libsize = 0

        for idx in neighbor_indices:
            if assigned[idx]:
                continue

            metacell_indices.append(idx)
            total_libsize += self.libsizes[idx]

            # Stop if we've reached or exceeded target
            if total_libsize >= self.target_libsize:
                break

            if len(metacell_indices) >= max_expansion:
                break

        # If we haven't reached minimum target and there are more unassigned cells,
        # do a broader search
        if total_libsize < min_target and len(metacell_indices) < n_available:
            # Get ALL remaining unassigned cells sorted by distance
            all_distances, all_indices = self.kd_tree.query(
                self.embeddings[seed_idx],
                k=n_available
            )

            for idx in all_indices:
                if idx in metacell_indices or assigned[idx]:
                    continue

                metacell_indices.append(idx)
                total_libsize += self.libsizes[idx]

                if total_libsize >= self.target_libsize:
                    break

                if len(metacell_indices) >= max_expansion:
                    break

        return np.array(metacell_indices), total_libsize

    def sample(
        self,
        n_metacells: int | None = None,
        min_libsize_fraction: float = 0.95,
        max_cells_per_metacell: int = 2000,
        tolerance: float = 0.05,
    ) -> list[Metacell]:
        """Sample metacells with approximately equal library size.

        Args:
            n_metacells: Number of metacells to create. If None, automatically
                        determined based on total library size and target.
            min_libsize_fraction: Minimum fraction of target_libsize to accept
                                 a metacell (default 0.95 = 95% of target)
            max_cells_per_metacell: Maximum cells per metacell
            tolerance: Tolerance for reaching target (passed to expansion)

        Returns:
            List of Metacell objects
        """
        # Determine number of metacells
        total_libsize = self.libsizes.sum()
        if n_metacells is None:
            n_metacells = int(total_libsize / self.target_libsize)
            logger.info(
                f"Auto-determined n_metacells={n_metacells} "
                f"(total={total_libsize:,}, target={self.target_libsize:,})"
            )

        # Get seed cells via geometric sketching
        seed_indices = self._geometric_sketch(n_metacells)

        # Track assigned cells
        assigned = np.zeros(len(self.cell_barcodes), dtype=bool)

        # Create metacells
        metacells = []
        min_libsize = int(self.target_libsize * min_libsize_fraction)

        for i, seed_idx in enumerate(seed_indices):
            if assigned[seed_idx]:
                # Find nearest unassigned cell as new seed
                distances, candidates = self.kd_tree.query(
                    self.embeddings[seed_idx], k=100
                )
                for cand in candidates:
                    if not assigned[cand]:
                        seed_idx = cand
                        break
                else:
                    logger.warning(f"Skipping seed {i}: no unassigned neighbors")
                    continue

            # Expand to target library size
            mc_indices, mc_libsize = self._expand_to_target_libsize(
                seed_idx, assigned, max_cells_per_metacell, tolerance
            )

            if mc_libsize < min_libsize:
                logger.debug(
                    f"Skipping metacell {i}: libsize {mc_libsize:,} < {min_libsize:,}"
                )
                continue

            # Mark cells as assigned
            assigned[mc_indices] = True

            # Create metacell object
            mc = Metacell(
                seed_barcode=self.cell_barcodes[seed_idx],
                cell_barcodes=[self.cell_barcodes[j] for j in mc_indices],
                cell_indices=self.atac_indices[mc_indices],
                total_libsize=mc_libsize,
                n_cells=len(mc_indices),
            )
            metacells.append(mc)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Created {len(metacells)} metacells, "
                    f"{assigned.sum()}/{len(assigned)} cells assigned"
                )

        logger.info(
            f"Created {len(metacells)} metacells with "
            f"mean libsize={np.mean([m.total_libsize for m in metacells]):,.0f}"
        )

        return metacells

    def sample_with_labels(
        self,
        n_metacells_per_label: int | dict[str, int] = 10,
        label_key: str = "labels",
        **kwargs,
    ) -> dict[str, list[Metacell]]:
        """Sample metacells separately for each cell label/cluster.

        Args:
            n_metacells_per_label: Number of metacells per label, or dict
                                  mapping label -> count
            label_key: Column in RNA adata.obs containing labels
            **kwargs: Additional arguments passed to sample()

        Returns:
            Dict mapping label -> list of metacells
        """
        # Get labels for shared cells
        if self.use_shared_cells_only:
            rna_indices = self.io.mapping['rna_idx'].values
            labels = self.rna_adata.obs[label_key].iloc[rna_indices].values
        else:
            labels = self.rna_adata.obs[label_key].loc[self.cell_barcodes].values

        unique_labels = np.unique(labels)
        logger.info(f"Sampling metacells for {len(unique_labels)} labels")

        result = {}
        for label in unique_labels:
            mask = labels == label
            n_cells = mask.sum()

            # Determine n_metacells for this label
            if isinstance(n_metacells_per_label, dict):
                n_mc = n_metacells_per_label.get(label, 10)
            else:
                n_mc = n_metacells_per_label

            # Adjust if not enough cells
            estimated_cells_needed = n_mc * (self.target_libsize / self.libsizes[mask].mean())
            if estimated_cells_needed > n_cells:
                n_mc = max(1, int(n_cells * self.libsizes[mask].mean() / self.target_libsize))
                logger.warning(
                    f"Label '{label}': reduced n_metacells to {n_mc} due to limited cells"
                )

            # Create sub-sampler for this label
            label_sampler = IsoLibsizeSamplerSubset(
                parent=self,
                mask=mask,
                random_state=self.random_state,
            )

            result[label] = label_sampler.sample(n_metacells=n_mc, **kwargs)

        return result

    def get_metacell_track(
        self,
        metacell: Metacell,
        chr_name: str,
        start: int,
        end: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """Get aggregated ATAC track for a metacell.

        Args:
            metacell: Metacell object
            chr_name: Chromosome name
            start: Start position
            end: End position
            normalize: If True, return RPM-normalized values

        Returns:
            Numpy array of ATAC signal
        """
        from caesar.io.query import IndexQuery

        query = IndexQuery(metacell.cell_indices)
        track = self.io.get_track(chr_name=chr_name, start=start, end=end, query=query)
        data = track.tracks_agg

        if normalize:
            # Normalize to reads per million
            data = data * 1e6 / metacell.total_libsize

        return data

    def get_all_metacell_tracks(
        self,
        metacells: list[Metacell],
        chr_name: str,
        start: int,
        end: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """Get aggregated ATAC tracks for all metacells.

        Args:
            metacells: List of Metacell objects
            chr_name: Chromosome name
            start: Start position
            end: End position
            normalize: If True, return RPM-normalized values

        Returns:
            2D numpy array of shape (n_metacells, end-start)
        """
        tracks = []
        for mc in metacells:
            track = self.get_metacell_track(mc, chr_name, start, end, normalize)
            tracks.append(track)
        return np.vstack(tracks)

    def get_metacell_dataframe(self, metacells: list[Metacell]) -> pd.DataFrame:
        """Convert metacells to a DataFrame.

        Args:
            metacells: List of Metacell objects

        Returns:
            DataFrame with metacell statistics
        """
        data = []
        for i, mc in enumerate(metacells):
            data.append({
                'metacell_id': i,
                'seed_barcode': mc.seed_barcode,
                'n_cells': mc.n_cells,
                'total_libsize': mc.total_libsize,
                'mean_libsize_per_cell': mc.total_libsize / mc.n_cells,
            })
        return pd.DataFrame(data)


class IsoLibsizeSamplerSubset:
    """Helper class for sampling within a subset of cells."""

    def __init__(
        self,
        parent: IsoLibsizeSampler,
        mask: np.ndarray,
        random_state: int = 42,
    ):
        self.parent = parent
        self.mask = mask
        self.random_state = random_state

        # Get subset data
        self.cell_barcodes = parent.cell_barcodes[mask]
        self.atac_indices = parent.atac_indices[mask]
        self.libsizes = parent.libsizes[mask]
        self.embeddings = parent.embeddings[mask]

        # Build local KNN index
        self.kd_tree = cKDTree(self.embeddings)

    def sample(self, n_metacells: int, **kwargs) -> list[Metacell]:
        """Sample metacells from the subset."""
        # Create temporary sampler-like object
        temp = type('TempSampler', (), {
            'embeddings': self.embeddings,
            'libsizes': self.libsizes,
            'cell_barcodes': self.cell_barcodes,
            'atac_indices': self.atac_indices,
            'kd_tree': self.kd_tree,
            'target_libsize': self.parent.target_libsize,
            'n_neighbors': self.parent.n_neighbors,
            'random_state': self.random_state,
        })()

        # Use parent's methods with local data
        seed_indices = self.parent._geometric_sketch.__func__(temp, n_metacells)

        assigned = np.zeros(len(self.cell_barcodes), dtype=bool)
        metacells = []
        min_libsize = int(self.parent.target_libsize * kwargs.get('min_libsize_fraction', 0.5))
        max_cells = kwargs.get('max_cells_per_metacell', 1000)

        for seed_idx in seed_indices:
            if assigned[seed_idx]:
                _, candidates = self.kd_tree.query(self.embeddings[seed_idx], k=100)
                for cand in candidates:
                    if not assigned[cand]:
                        seed_idx = cand
                        break
                else:
                    continue

            mc_indices, mc_libsize = self.parent._expand_to_target_libsize.__func__(
                temp, seed_idx, assigned, max_cells
            )

            if mc_libsize < min_libsize:
                continue

            assigned[mc_indices] = True

            mc = Metacell(
                seed_barcode=self.cell_barcodes[seed_idx],
                cell_barcodes=[self.cell_barcodes[j] for j in mc_indices],
                cell_indices=self.atac_indices[mc_indices],
                total_libsize=mc_libsize,
                n_cells=len(mc_indices),
            )
            metacells.append(mc)

        return metacells


def sample_iso_libsize_metacells(
    multiome_path: str,
    target_libsize: int = 5_000_000,
    n_metacells: int | None = None,
    embedding_key: str = "X_pca",
    output_path: str | None = None,
    **kwargs,
) -> list[Metacell]:
    """Convenience function to sample iso-libsize metacells.

    Args:
        multiome_path: Path to multiome zarr
        target_libsize: Target ATAC library size per metacell
        n_metacells: Number of metacells (auto if None)
        embedding_key: Embedding to use for sketching
        output_path: Optional path to save metacell assignments
        **kwargs: Additional arguments for IsoLibsizeSampler

    Returns:
        List of Metacell objects
    """
    from caesar.io.zarr_io.multiome import MultiomeSparseZarrIO

    io = MultiomeSparseZarrIO(multiome_path)
    sampler = IsoLibsizeSampler(
        io,
        target_libsize=target_libsize,
        embedding_key=embedding_key,
        **kwargs,
    )

    metacells = sampler.sample(n_metacells=n_metacells)

    if output_path:
        df = sampler.get_metacell_dataframe(metacells)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved metacell info to {output_path}")

        # Also save cell assignments
        assignments = []
        for i, mc in enumerate(metacells):
            for bc in mc.cell_barcodes:
                assignments.append({'metacell_id': i, 'barcode': bc})
        pd.DataFrame(assignments).to_csv(
            output_path.replace('.csv', '_assignments.csv'),
            index=False
        )

    return metacells
