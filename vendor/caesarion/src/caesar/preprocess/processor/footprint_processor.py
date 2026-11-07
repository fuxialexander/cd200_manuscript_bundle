"""footprint processor with simplified configuration access."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from caesar.io.zarr_io import MotifDenseZarrIO
from caesar.io.zarr_io.motif_hits import MotifHitZarrIO
from caesar.preprocess.core.base import AnnotationProcessor
from caesar.preprocess.core.exceptions import ValidationError
from caesar.preprocess.core.mixins import CompressorMixin

if TYPE_CHECKING:
    from caesar.preprocess.core.config import CaesarConfig


class FootprintProcessor(AnnotationProcessor, CompressorMixin):
    """footprint processor with direct configuration access.

    This processor detects TF footprints and saves them to zarr stores
    using a simplified configuration pattern that provides type-safe
    direct access to configuration fields.

    Note: Currently only zarr format motif databases are supported.
    Other formats (JASPAR, CIS-BP, HOCOMOCO) are not yet implemented.
    """

    def __init__(self, config: CaesarConfig):
        """Initialize FootprintProcessor.

        Args:
            config: Structured CaesarConfig instance
        """
        # Initialize base class with full config object
        super().__init__(config)

        # Store full config for direct access
        self.config = config

        # Direct access to configuration - clean and simple
        self.footprint_config = config.footprints

        # Extract footprint parameters with type safety
        self.mode = self.footprint_config.mode
        self.motif_zarr = self.footprint_config.motif_zarr
        self.cutoff_range = self.footprint_config.cutoff_range
        self.extend = self.footprint_config.extend
        self.n_processes = self.footprint_config.n_processes

        # ML feature extraction parameters
        self.core_extend = self.footprint_config.core_extend
        self.context_extend = self.footprint_config.context_extend
        self.n_bins = self.footprint_config.n_bins
        self.shift_offsets = (
            np.array(self.footprint_config.shift_offsets, dtype=np.int32)
            if self.footprint_config.shift_offsets is not None
            else np.array([10, 20, 30], dtype=np.int32)
        )
        self.aggregate_by_motif = self.footprint_config.aggregate_by_motif
        self.check_convergence_every = self.footprint_config.check_convergence_every
        self.convergence_threshold = self.footprint_config.convergence_threshold
        
        # Override defaults if specified in config (CLI will populate config)
        self.n_processes = self.footprint_config.n_processes if self.footprint_config.n_processes is not None else 2

        # Custom motif database support
        self.motif_database_type = self.footprint_config.motif_database_type
        self.motif_scan_on_demand = self.footprint_config.motif_scan_on_demand
        self.motif_scan_config = self.footprint_config.motif_scan_config or {}

        # Visualization options
        self.visualize = self.footprint_config.visualize
        self.visualization_output_dir = self.footprint_config.visualization_output_dir
        self.motifs_to_visualize = self.footprint_config.motifs_to_visualize

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate footprint configuration."""
        # Don't validate input/target path here - it may be set later by workflow
        # Just validate footprint-specific configuration

        # Skip validation if footprinting is disabled
        if not self.footprint_config.enabled:
            return

        # Validate motif zarr is specified (but don't check existence yet)
        # Unless on-demand scanning is enabled
        if not self.motif_zarr and not self.motif_scan_on_demand:
            raise ValidationError(
                "Motif zarr path required for footprint detection, "
                "or enable motif_scan_on_demand"
            )

        # Validate motif database type - only zarr is currently supported
        if self.motif_database_type != "zarr":
            raise ValidationError(
                f"Invalid motif_database_type: {self.motif_database_type}. "
                "Currently only 'zarr' format is supported."
            )

        # Validate mode
        if self.mode not in ["footprint", "ml_features"]:
            raise ValidationError(
                f"Invalid mode: {self.mode}. Must be 'footprint' or 'ml_features'"
            )

        # Validate cutoff range
        if len(self.cutoff_range) != 2:
            raise ValidationError(
                "cutoff_range must have exactly 2 values [start, end]"
            )

        if self.cutoff_range[0] >= self.cutoff_range[1]:
            raise ValidationError("cutoff_range[0] must be less than cutoff_range[1]")

        # Validate other parameters
        if self.extend <= 0:
            raise ValidationError("extend must be positive")

        if self.n_processes < 1:
            raise ValidationError("n_processes must be at least 1")

        # Validate ML feature parameters if mode is ml_features
        if self.mode == "ml_features":
            if self.core_extend <= 0:
                raise ValidationError("core_extend must be positive")
            if self.context_extend <= 0:
                raise ValidationError("context_extend must be positive")
            if self.context_extend < self.core_extend:
                raise ValidationError("context_extend must be >= core_extend")
            if self.n_bins <= 0:
                raise ValidationError("n_bins must be positive")
            if self.check_convergence_every < 1:
                raise ValidationError("check_convergence_every must be at least 1")
            if self.convergence_threshold <= 0:
                raise ValidationError("convergence_threshold must be positive")

    @classmethod
    def get_command_name(cls) -> str:
        """Get the CLI command name for this processor."""
        return "footprint"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add footprint-specific CLI arguments."""
        # Call parent to add common args (config, n_jobs, etc.)
        from caesar.preprocess.core.base import BaseProcessor
        super(FootprintProcessor, cls).add_cli_args(parser)

        # Input/Output
        parser.add_argument(
            "-i", "--input", required=False,
            help="Input Zarr store or BPCells directory (optional, matches data.input)"
        )
        parser.add_argument(
            "-o", "--output", help="Output file path (optional, relative to output.dir)"
        )
        parser.add_argument(
            "--motif-zarr", help="Path to motif hits Zarr database"
        )
        
        # Parameters
        parser.add_argument(
            "--footprint-mode", choices=["footprint", "ml_features"],
            help="Processing mode: 'footprint' or 'ml_features'"
        )
        parser.add_argument(
            "--cutoff-range", nargs=2, type=int,
            help="Motif score cutoff range [start, end]"
        )
        parser.add_argument(
            "--extend", type=int, help="Footprint extension ±bp"
        )
        parser.add_argument(
            "--n-processes", type=int, help="Number of parallel processes (default: 2)"
        )
        
        # ML feature specific
        parser.add_argument(
            "--core-extend", type=int, help="Core window size (±bp) for ML features"
        )
        parser.add_argument(
            "--context-extend", type=int, help="Context window size (±bp) for ML features"
        )
        parser.add_argument(
            "--n-bins", type=int, help="Number of bins for shape reconstruction"
        )
        parser.add_argument(
            "--shift-offsets", nargs="+", type=int,
            help="Co-accessibility offset positions (e.g., 10 20 30)"
        )
    
    def _get_relevant_config_sections(self) -> list[str]:
        """Get footprint-specific configuration sections."""
        sections = ["data.input", "footprints", "output.dir"]
        if self.motif_scan_on_demand:
            sections.append("non_redundant_motif")
        return sections

    def process(self) -> dict[str, Any]:
        """Detect footprints and save to zarr.

        Returns:
            Processing results
        """
        self._start_processing()

        # Check if footprinting is disabled
        skipped = self._check_enabled("footprints")
        if skipped:
            return skipped

        # Get target path for in-place modification
        input_path = self.validate_input_path(self.get_target_path())

        # Open input data store (supports Zarr and BPCells)
        io = self._open_data_store(input_path)

        # Load or create motif database
        m = self._load_motif_database(input_path)

        # Route to appropriate processing method based on mode
        results = {}
        output_data = None  # Initialize for visualization
        cutoff = None  # Will be set based on mode

        try:
            if self.mode == "ml_features":
                # ML feature extraction mode
                if not hasattr(io, "get_genome_wide_ml_features_prefix_sum"):
                    raise ValueError(
                        f"{io.__class__.__name__} does not support ML feature extraction. "
                        "Requires IO class with get_genome_wide_ml_features_prefix_sum method."
                    )

                # Use single cutoff for ML features (first value from range)
                ml_cutoff = float(self.cutoff_range[0])

                if hasattr(io, "ids"):
                    # For CelltypeDenseZarrIO with multiple tracks
                    from caesar.io.aggregator import ConcatAggregator
                    from caesar.io.normalizer import NoNormalizer
                    from caesar.io.query import DefaultQuery

                    self.logger.info(
                        f"Computing ML features for all {len(io.ids)} clusters at once..."
                    )

                    ml_features = io.get_genome_wide_ml_features_prefix_sum(
                        m,
                        n_processes=self.n_processes,
                        cutoff=ml_cutoff,
                        core_extend=self.core_extend,
                        context_extend=self.context_extend,
                        n_bins=self.n_bins,
                        shift_offsets=self.shift_offsets,
                        aggregate_by_motif=self.aggregate_by_motif,
                        check_convergence_every=self.check_convergence_every,
                        convergence_threshold=self.convergence_threshold,
                        query=DefaultQuery(),
                        aggregator=ConcatAggregator(),
                        normalizer=NoNormalizer(),
                    )

                    # ml_features shape: (n_cells, n_motifs, n_features)
                    self.logger.info(f"Computed ML features shape: {ml_features.shape}")

                    # Save ML features for all clusters
                    self._save_all_ml_features_to_zarr(
                        io, ml_features, ml_cutoff, self.core_extend, self.context_extend, self.n_bins
                    )

                    results["all_clusters"] = {
                        "n_clusters": len(io.ids),
                        "n_motifs": ml_features.shape[1],
                        "n_features": ml_features.shape[2],
                        "cutoff": ml_cutoff,
                        "core_extend": self.core_extend,
                        "context_extend": self.context_extend,
                        "n_bins": self.n_bins,
                        "shape": ml_features.shape,
                    }
                    output_data = ml_features

                else:
                    # For single track
                    track_id = "default"
                    self.logger.info(f"Computing ML features for {track_id}...")

                    ml_features = io.get_genome_wide_ml_features_prefix_sum(
                        m,
                        n_processes=self.n_processes,
                        cutoff=ml_cutoff,
                        core_extend=self.core_extend,
                        context_extend=self.context_extend,
                        n_bins=self.n_bins,
                        shift_offsets=self.shift_offsets,
                        aggregate_by_motif=self.aggregate_by_motif,
                        check_convergence_every=self.check_convergence_every,
                        convergence_threshold=self.convergence_threshold,
                    )

                    # Save ML features to zarr
                    self._save_ml_features_to_zarr(
                        io,
                        track_id,
                        ml_features,
                        ml_cutoff,
                        self.core_extend,
                        self.context_extend,
                        self.n_bins,
                    )

                    results[track_id] = {
                        "n_motifs": ml_features.shape[1],
                        "n_features": ml_features.shape[2],
                        "cutoff": ml_cutoff,
                        "core_extend": self.core_extend,
                        "context_extend": self.context_extend,
                        "n_bins": self.n_bins,
                        "shape": ml_features.shape,
                    }
                    output_data = ml_features

            else:
                # Traditional footprint mode
                if not hasattr(io, "get_genome_wide_footprint_all_motifs"):
                    raise ValueError(
                        f"{io.__class__.__name__} does not support footprint detection. "
                        "Requires CelltypeDenseZarrIO or compatible class."
                    )

                # Process cutoff range (for footprint mode)
                cutoff = np.arange(self.cutoff_range[0], self.cutoff_range[1])

                if hasattr(io, "ids"):
                    # For CelltypeDenseZarrIO with multiple tracks
                    from caesar.io.aggregator import ConcatAggregator
                    from caesar.io.normalizer import NoNormalizer
                    from caesar.io.query import DefaultQuery

                    self.logger.info(
                        f"Computing footprints for all {len(io.ids)} clusters at once..."
                    )

                    footprints = io.get_genome_wide_footprint_all_motifs(
                        m,
                        n_processes=self.n_processes,
                        cutoff=cutoff,
                        extend=self.extend,
                        query=DefaultQuery(),
                        aggregator=ConcatAggregator(),
                        normalizer=NoNormalizer(),
                    )

                    # footprints shape: (n_motifs, 2*extend, n_clusters) or (n_cutoffs, n_motifs, 2*extend, n_clusters)
                    self.logger.info(f"Computed footprints shape: {footprints.shape}")

                    # Save footprints for all clusters
                    self._save_all_footprints_to_zarr(io, footprints, cutoff, self.extend)

                    results["all_clusters"] = {
                        "n_clusters": len(io.ids),
                        "n_motifs": footprints.shape[1]
                        if footprints.ndim == 4
                        else footprints.shape[0],
                        "cutoff_range": cutoff.tolist(),
                        "extend": self.extend,
                        "shape": footprints.shape,
                    }
                    output_data = footprints

                else:
                    # For single track
                    track_id = "default"
                    self.logger.info(f"Computing footprints for {track_id}...")

                    footprints = io.get_genome_wide_footprint_all_motifs(
                        m, n_processes=self.n_processes, cutoff=cutoff, extend=self.extend
                    )

                    # Save footprints to zarr
                    self._save_footprints_to_zarr(
                        io, track_id, footprints, cutoff, self.extend
                    )

                    results[track_id] = {
                        "n_motifs": footprints.shape[-1]
                        if len(footprints.shape) > 1
                        else 1,
                        "cutoff_range": cutoff.tolist(),
                        "extend": self.extend,
                        "shape": footprints.shape,
                    }
                    output_data = footprints

        except Exception as e:
            self.logger.error(f"Failed to compute {'ML features' if self.mode == 'ml_features' else 'footprints'}: {e}")
            results["error"] = str(e)

        # Generate visualizations if requested (only for footprint mode)
        if self.visualize and output_data is not None and self.mode == "footprint":
            self._generate_visualizations(io, m, output_data, cutoff)

        runtime = self._end_processing()

        return {
            "output_path": str(input_path),
            "tracks_processed": len(results),
            "results": results,
            "motif_zarr": str(self.motif_zarr) if self.motif_zarr else "on-demand",
            "runtime": runtime,
        }

    def _open_data_store(self, store_path: str | Path):
        """Open genomic data store (Zarr or BPCells).

        Args:
            store_path: Path to the store

        Returns:
            Appropriate IO object
        """
        from caesar.io.genomic_io import GenomicIO
        
        store_type = GenomicIO.detect_store_type(store_path)
        
        if store_type == "bpcells_celltype_dense":
            from caesar.io.bpcell_io import CelltypeDenseBPCellsIO
            return CelltypeDenseBPCellsIO(str(store_path), mode="a")
        elif store_type == "bpcells_sparse":
            from caesar.io.bpcell_io import SingleSparseBPCellsIO
            return SingleSparseBPCellsIO(str(store_path), mode="a")
            
        # Fallback to Zarr
        from caesar.io.zarr_io import open_zarr
        return open_zarr(store_path, mode="a")

    def _get_zarr_root(self, io) -> Any:
        """Get or create a Zarr root for saving results.

        For Zarr-backed IO, returns the existing Zarr dataset/root.
        For others (like BPCells), opens a 'footprint/' subfolder as a Zarr store
        to keep footprint data organized separately.

        Args:
            io: IO object

        Returns:
            Zarr group object
        """
        import zarr
        import os
        
        # Check for Zarr group in dataset (zarr v3 uses zarr.Group, not zarr.hierarchy.Group)
        if hasattr(io, "dataset") and isinstance(io.dataset, zarr.Group):
            return io.dataset
            
        # Check for explicit zarr_root
        if hasattr(io, "zarr_root"):
            return io.zarr_root
            
        # For BPCells or other non-Zarr backends, create a footprint/ subfolder
        # This keeps footprint data organized separately from BPCells files
        footprint_path = os.path.join(str(io.path), "footprint")
        os.makedirs(footprint_path, exist_ok=True)
        return zarr.open(footprint_path, mode="a")

    def _save_footprints_to_zarr(
        self, io, track_id: str, footprints: np.ndarray, cutoff: np.ndarray, extend: int
    ) -> None:
        """Save footprint data to zarr.

        Args:
            io: Zarr IO object
            track_id: Track/cluster ID
            footprints: Footprint array
            cutoff: Cutoff values used
            extend: Extension used
        """
        compressor = self.compressor

        # Get the zarr group for this track
        zarr_root = self._get_zarr_root(io)

        # Get the track-specific group (e.g., cluster_0, cluster_1, etc.)
        track_group = zarr_root.require_group(track_id)

        # Determine the appropriate dtype based on data range
        max_val = np.abs(footprints).max() if footprints.size > 0 else 0
        if max_val <= 65504:  # float16 max value
            dtype = np.float16
        else:
            dtype = np.float32
            self.logger.warning(
                f"Footprint values for {track_id} exceed float16 range (max: {max_val:.1f}), using float32"
            )

        # Save footprint data directly in the track group
        n_motifs = footprints.shape[-1] if len(footprints.shape) > 1 else 1
        footprint_data = footprints.astype(dtype)
        footprint_arr = track_group.create_array(
            "footprint",
            shape=footprint_data.shape,
            dtype=dtype,
            chunks=(1, n_motifs, 2 * extend)
            if footprints.ndim == 3
            else (n_motifs, 2 * extend),
            compressors=compressor,
            overwrite=True,
        )
        footprint_arr[:] = footprint_data

        # Save metadata as attributes of the footprint dataset
        footprint_dataset = track_group["footprint"]
        footprint_dataset.attrs["cutoff"] = cutoff.tolist()
        footprint_dataset.attrs["extend"] = extend
        footprint_dataset.attrs["n_motifs"] = n_motifs
        footprint_dataset.attrs["shape"] = list(footprints.shape)

        self.logger.info(
            f"Saved footprints for {track_id}: shape={footprints.shape} at {track_id}/footprint"
        )

    def _save_all_footprints_to_zarr(
        self, io, footprints: np.ndarray, cutoff: np.ndarray, extend: int
    ) -> None:
        """Save footprint data for all clusters to zarr at once.

        Args:
            io: Zarr IO object
            footprints: Footprint array with shape (n_motifs, 2*extend, n_clusters)
                       or (n_cutoffs, n_motifs, 2*extend, n_clusters)
            cutoff: Cutoff values used
            extend: Extension used
        """
        compressor = self.compressor

        # Get the zarr root
        zarr_root = self._get_zarr_root(io)

        # Save footprints for each cluster
        track_ids = io.ids if hasattr(io, "ids") else ["default"]

        for i, track_id in enumerate(track_ids):
            track_group = zarr_root.require_group(track_id)

            # Extract footprints for this cluster
            if footprints.ndim == 3:
                # Shape: (n_motifs, 2*extend, n_clusters)
                cluster_footprints = footprints[..., i]
            else:
                # Shape: (n_cutoffs, n_motifs, 2*extend, n_clusters)
                cluster_footprints = footprints[..., i]

            # Save footprint data
            n_motifs = (
                cluster_footprints.shape[1]
                if cluster_footprints.ndim == 3
                else cluster_footprints.shape[0]
            )

            # Determine the appropriate dtype based on data range
            max_val = (
                np.abs(cluster_footprints).max() if cluster_footprints.size > 0 else 0
            )
            if max_val <= 65504:  # float16 max value
                dtype = np.float16
            else:
                dtype = np.float32
                self.logger.warning(
                    f"Footprint values for {track_id} exceed float16 range (max: {max_val:.1f}), using float32"
                )

            footprint_data = cluster_footprints.astype(dtype)
            footprint_arr = track_group.create_array(
                "footprint",
                shape=footprint_data.shape,
                dtype=dtype,
                chunks=(1, n_motifs, 2 * extend)
                if cluster_footprints.ndim == 3
                else (n_motifs, 2 * extend),
                compressors=compressor,
                overwrite=True,
            )
            footprint_arr[:] = footprint_data

            # Save metadata
            footprint_dataset = track_group["footprint"]
            footprint_dataset.attrs["cutoff"] = cutoff.tolist()
            footprint_dataset.attrs["extend"] = extend
            footprint_dataset.attrs["n_motifs"] = n_motifs
            footprint_dataset.attrs["shape"] = list(cluster_footprints.shape)

        self.logger.info(f"Saved footprints for all {len(track_ids)} clusters")

    def _save_ml_features_to_zarr(
        self,
        io,
        track_id: str,
        ml_features: np.ndarray,
        cutoff: float,
        core_extend: int,
        context_extend: int,
        n_bins: int,
    ) -> None:
        """Save ML feature data to zarr.

        Args:
            io: Zarr IO object
            track_id: Track/cluster ID
            ml_features: ML feature array with shape (n_cells, n_motifs, n_features)
            cutoff: Cutoff value used
            core_extend: Core window size used
            context_extend: Context window size used
            n_bins: Number of bins used
        """
        compressor = self.compressor

        # Get the zarr group for this track
        zarr_root = self._get_zarr_root(io)

        # Get the track-specific group (e.g., cluster_0, cluster_1, etc.)
        track_group = zarr_root.require_group(track_id)

        # ML features are already float32, but check if we can use float16
        max_val = np.abs(ml_features).max() if ml_features.size > 0 else 0
        if max_val <= 65504:  # float16 max value
            dtype = np.float16
        else:
            dtype = np.float32
            self.logger.warning(
                f"ML feature values for {track_id} exceed float16 range (max: {max_val:.1f}), using float32"
            )

        # Save ML feature data directly in the track group
        n_cells, n_motifs, n_features = ml_features.shape
        ml_data = ml_features.astype(dtype)
        ml_arr = track_group.create_array(
            "ml_features",
            shape=ml_data.shape,
            dtype=dtype,
            chunks=(1, n_motifs, n_features),  # Chunk by cell type
            compressors=compressor,
            overwrite=True,
        )
        ml_arr[:] = ml_data

        # Save metadata as attributes of the ML features dataset
        ml_features_dataset = track_group["ml_features"]
        ml_features_dataset.attrs["cutoff"] = float(cutoff)
        ml_features_dataset.attrs["core_extend"] = int(core_extend)
        ml_features_dataset.attrs["context_extend"] = int(context_extend)
        ml_features_dataset.attrs["n_bins"] = int(n_bins)
        ml_features_dataset.attrs["n_cells"] = int(n_cells)
        ml_features_dataset.attrs["n_motifs"] = int(n_motifs)
        ml_features_dataset.attrs["n_features"] = int(n_features)
        ml_features_dataset.attrs["shape"] = list(ml_features.shape)
        ml_features_dataset.attrs["feature_layout"] = [
            "core_plus",
            "core_minus",
            "occupancy",
            "strand_asymmetry",
            f"shape_plus_{n_bins}bins",
            f"shape_minus_{n_bins}bins",
            f"coaccess_{len(self.shift_offsets)}offsets",
        ]

        self.logger.info(
            f"Saved ML features for {track_id}: shape={ml_features.shape} at {track_id}/ml_features"
        )

    def _save_all_ml_features_to_zarr(
        self,
        io,
        ml_features: np.ndarray,
        cutoff: float,
        core_extend: int,
        context_extend: int,
        n_bins: int,
    ) -> None:
        """Save ML feature data for all clusters to zarr at once.

        Args:
            io: Zarr IO object
            ml_features: ML feature array with shape (n_cells, n_motifs, n_features)
            cutoff: Cutoff value used
            core_extend: Core window size used
            context_extend: Context window size used
            n_bins: Number of bins used
        """
        compressor = self.compressor

        # Get the zarr root
        zarr_root = self._get_zarr_root(io)

        # Save ML features for each cluster
        track_ids = io.ids if hasattr(io, "ids") else ["default"]
        n_cells, n_motifs, n_features = ml_features.shape

        # Verify we have the right number of clusters
        if len(track_ids) != n_cells:
            self.logger.warning(
                f"Mismatch: {len(track_ids)} track IDs but {n_cells} cells in ML features array"
            )

        for i, track_id in enumerate(track_ids):
            track_group = zarr_root.require_group(track_id)

            # Extract ML features for this cluster (cell type)
            # Shape: (n_cells, n_motifs, n_features)
            cluster_ml_features = ml_features[i : i + 1, :, :]  # Keep cell dimension

            # Determine the appropriate dtype based on data range
            max_val = (
                np.abs(cluster_ml_features).max() if cluster_ml_features.size > 0 else 0
            )
            if max_val <= 65504:  # float16 max value
                dtype = np.float16
            else:
                dtype = np.float32
                self.logger.warning(
                    f"ML feature values for {track_id} exceed float16 range (max: {max_val:.1f}), using float32"
                )

            ml_data = cluster_ml_features.astype(dtype)
            ml_arr = track_group.create_array(
                "ml_features",
                shape=ml_data.shape,
                dtype=dtype,
                chunks=(1, n_motifs, n_features),
                compressors=compressor,
                overwrite=True,
            )
            ml_arr[:] = ml_data

            # Save metadata
            ml_features_dataset = track_group["ml_features"]
            ml_features_dataset.attrs["cutoff"] = float(cutoff)
            ml_features_dataset.attrs["core_extend"] = int(core_extend)
            ml_features_dataset.attrs["context_extend"] = int(context_extend)
            ml_features_dataset.attrs["n_bins"] = int(n_bins)
            ml_features_dataset.attrs["n_cells"] = 1  # One cell type per track
            ml_features_dataset.attrs["n_motifs"] = int(n_motifs)
            ml_features_dataset.attrs["n_features"] = int(n_features)
            ml_features_dataset.attrs["shape"] = list(cluster_ml_features.shape)
            ml_features_dataset.attrs["feature_layout"] = [
                "core_plus",
                "core_minus",
                "occupancy",
                "strand_asymmetry",
                f"shape_plus_{n_bins}bins",
                f"shape_minus_{n_bins}bins",
                f"coaccess_{len(self.shift_offsets)}offsets",
            ]

        self.logger.info(f"Saved ML features for all {len(track_ids)} clusters")

    def _load_motif_database(
        self, input_zarr_path: Path
    ) -> MotifDenseZarrIO | MotifHitZarrIO:
        """Load motif database from various sources.

        Args:
            input_zarr_path: Path to input zarr (used for on-demand scanning)

        Returns:
            MotifDenseZarrIO object
        """
        # Check if motif zarr exists
        if self.motif_zarr:
            motif_zarr_path = Path(self.motif_zarr)
            if motif_zarr_path.exists():
                self.logger.info(f"Loading motif database from {motif_zarr_path}")
                return self._load_motif_zarr(motif_zarr_path)
            elif not self.motif_scan_on_demand:
                raise ValidationError(f"Motif zarr not found: {motif_zarr_path}")

        # If on-demand scanning is enabled
        if self.motif_scan_on_demand:
            self.logger.info("Running on-demand motif scanning...")
            motif_zarr_path = self._run_motif_scanning(input_zarr_path)
            return self._load_motif_zarr(motif_zarr_path)

        raise ValidationError("No motif database available")

    def _load_motif_zarr(
        self, motif_zarr_path: Path
    ) -> MotifDenseZarrIO | MotifHitZarrIO:
        """Load motif zarr with auto-detection of format.

        Args:
            motif_zarr_path: Path to motif zarr

        Returns:
            MotifDenseZarrIO or MotifHitZarrIO object based on detected format
        """
        import zarr

        # Open zarr store to inspect structure
        zarr_store = zarr.open(str(motif_zarr_path), mode="r")

        # Auto-detect format based on group structure
        if self._is_motif_hits_format(zarr_store):
            self.logger.info(f"Detected MotifHitZarrIO format in {motif_zarr_path}")
            return MotifHitZarrIO(str(motif_zarr_path), mode="r")
        else:
            self.logger.info(f"Detected MotifDenseZarrIO format in {motif_zarr_path}")
            return MotifDenseZarrIO(str(motif_zarr_path), mode="r")

    def _is_motif_hits_format(self, zarr_store) -> bool:
        """Check if zarr store uses MotifHitZarrIO format.

        Args:
            zarr_store: Opened zarr store

        Returns:
            True if MotifHitZarrIO format, False if MotifDenseZarrIO format
        """
        try:
            # Check for MotifHitZarrIO signature groups
            has_motif_hit_data = (
                "motif_hit_data" in zarr_store
                and "regions" in zarr_store["motif_hit_data"]
            )
            has_peak_regions = (
                "peak_regions" in zarr_store and "regions" in zarr_store["peak_regions"]
            )
            has_metadata = (
                "metadata" in zarr_store and "motif_names" in zarr_store["metadata"]
            )

            # All three must be present for MotifHitZarrIO
            if has_motif_hit_data and has_peak_regions and has_metadata:
                return True

            # Check for MotifDenseZarrIO signature
            has_motif_match_data = "motif_match_data" in zarr_store

            if has_motif_match_data:
                return False

            # If neither signature is clear, check for chromosome-level data structure
            # MotifDenseZarrIO typically has chromosome arrays directly at root level
            chroms = [key for key in zarr_store.keys() if key.startswith("chr")]
            if chroms:
                # Check if these are direct data arrays (MotifDenseZarrIO style)
                sample_chrom = chroms[0]
                if (
                    hasattr(zarr_store[sample_chrom], "shape")
                    and len(zarr_store[sample_chrom].shape) >= 2
                ):
                    return False  # MotifDenseZarrIO

            # Default to False (MotifDenseZarrIO) if unclear
            return False

        except Exception as e:
            self.logger.warning(f"Error detecting motif zarr format: {e}")
            return False  # Default to MotifDenseZarrIO

    def _run_motif_scanning(self, input_zarr_path: Path) -> Path:
        """Run motif scanning on demand using MotifProcessor.

        Args:
            input_zarr_path: Path to input zarr file

        Returns:
            Path to generated motif zarr
        """
        from caesar.preprocess.processor.motif_processor import (
            NonRedundantMotifProcessor,
        )

        # Create a sub-config for motif scanning
        motif_config = self.config.__class__()
        motif_config.__dict__.update(self.config.__dict__)

        # Update with scan config
        if "motif_scan_results" in self.motif_scan_config:
            motif_config.non_redundant_motif.motif_scan_results = (
                self.motif_scan_config["motif_scan_results"]
            )
        else:
            raise ValidationError(
                "motif_scan_results required in motif_scan_config for on-demand scanning"
            )

        # Set other parameters from scan config
        for key, value in self.motif_scan_config.items():
            if hasattr(motif_config.non_redundant_motif, key):
                setattr(motif_config.non_redundant_motif, key, value)

        # Run motif processor
        motif_processor = NonRedundantMotifProcessor(motif_config)
        result = motif_processor.process()

        return Path(result["output_path"])

    def _generate_visualizations(
        self, io, motif_db, footprints: np.ndarray, cutoff: np.ndarray
    ) -> None:
        """Generate footprint profile visualizations.

        Args:
            io: Zarr IO object
            motif_db: Motif database object
            footprints: Computed footprints array
            cutoff: Cutoff values used
        """
        # Determine output directory
        if self.visualization_output_dir:
            output_dir = Path(self.visualization_output_dir)
        else:
            output_dir = Path(self.output_dir) / "footprint_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating visualizations in {output_dir}")

        # Get motif information
        if hasattr(motif_db, "dataset") and "motif_clusters" in motif_db.dataset.attrs:
            motif_names = motif_db.dataset.attrs["motif_clusters"]
        else:
            # Generate generic names
            n_motifs = (
                footprints.shape[1] if footprints.ndim == 4 else footprints.shape[0]
            )
            motif_names = [f"motif_{i}" for i in range(n_motifs)]

        # Determine which motifs to visualize
        if self.motifs_to_visualize:
            # Filter to requested motifs
            motif_indices = []
            for motif in self.motifs_to_visualize:
                if motif in motif_names:
                    motif_indices.append(motif_names.index(motif))
                elif motif.isdigit() and int(motif) < len(motif_names):
                    motif_indices.append(int(motif))
                else:
                    self.logger.warning(f"Motif {motif} not found")
        else:
            # Visualize top 10 motifs by footprint depth
            if footprints.ndim == 4:
                # Shape: (n_cutoffs, n_motifs, 2*extend, n_clusters)
                # Average across cutoffs and clusters
                mean_depths = np.abs(footprints).mean(axis=(0, 3))
            else:
                # Shape: (n_motifs, 2*extend) or (n_motifs, 2*extend, n_clusters)
                mean_depths = np.abs(footprints).mean(
                    axis=tuple(range(1, footprints.ndim))
                )

            # Get center region depths
            center_start = self.extend - 50
            center_end = self.extend + 50
            center_depths = mean_depths[:, center_start:center_end].mean(axis=1)

            # Get top 10 motifs
            motif_indices = np.argsort(center_depths)[-10:][::-1]

        # Generate plots
        for idx in motif_indices:
            motif_name = motif_names[idx] if idx < len(motif_names) else f"motif_{idx}"
            self._plot_single_motif_footprint(
                footprints, idx, motif_name, cutoff, output_dir
            )

        # Generate summary plot
        self._plot_footprint_summary(footprints, motif_indices, motif_names, output_dir)

    def _plot_single_motif_footprint(
        self,
        footprints: np.ndarray,
        motif_idx: int,
        motif_name: str,
        cutoff: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Plot footprint profile for a single motif.

        Args:
            footprints: Footprint array
            motif_idx: Index of motif to plot
            motif_name: Name of motif
            cutoff: Cutoff values
            output_dir: Output directory
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Extract footprint for this motif
        if footprints.ndim == 4:
            # Shape: (n_cutoffs, n_motifs, 2*extend, n_clusters)
            # Average across clusters
            motif_fp = footprints[:, motif_idx, :, :].mean(axis=2)
        elif footprints.ndim == 3:
            # Shape: (n_motifs, 2*extend, n_clusters)
            motif_fp = footprints[motif_idx, :, :].mean(axis=1)
        else:
            # Shape: (n_motifs, 2*extend)
            motif_fp = footprints[motif_idx, :]

        # Plot 1: Footprint profile across cutoffs
        ax1 = axes[0]
        x_pos = np.arange(-self.extend, self.extend)

        if motif_fp.ndim == 2:
            # Multiple cutoffs
            for i, c in enumerate(cutoff):
                ax1.plot(x_pos, motif_fp[i, :], label=f"Cutoff {c}", alpha=0.8)
        else:
            # Single profile
            ax1.plot(x_pos, motif_fp, label="Footprint", color="darkblue", linewidth=2)

        ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Distance from motif center (bp)")
        ax1.set_ylabel("Accessibility")
        ax1.set_title(f"Footprint Profile: {motif_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Heatmap if multiple cutoffs
        ax2 = axes[1]
        if motif_fp.ndim == 2:
            im = ax2.imshow(
                motif_fp,
                aspect="auto",
                cmap="RdBu_r",
                extent=[-self.extend, self.extend, cutoff[-1], cutoff[0]],
            )
            ax2.set_xlabel("Distance from motif center (bp)")
            ax2.set_ylabel("Cutoff value")
            ax2.set_title("Footprint Heatmap")
            plt.colorbar(im, ax=ax2, label="Accessibility")
        else:
            # Single profile - show smoothed version
            from scipy.ndimage import gaussian_filter1d

            smoothed = gaussian_filter1d(motif_fp, sigma=5)
            ax2.plot(x_pos, motif_fp, alpha=0.5, label="Raw")
            ax2.plot(x_pos, smoothed, label="Smoothed", color="darkred", linewidth=2)
            ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax2.set_xlabel("Distance from motif center (bp)")
            ax2.set_ylabel("Accessibility")
            ax2.set_title("Smoothed Profile")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / f"footprint_{motif_name.replace('/', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.debug(f"Saved footprint plot: {output_file}")

    def _plot_footprint_summary(
        self,
        footprints: np.ndarray,
        motif_indices: list[int],
        motif_names: list[str],
        output_dir: Path,
    ) -> None:
        """Generate summary plot of multiple footprints.

        Args:
            footprints: Footprint array
            motif_indices: Indices of motifs to include
            motif_names: Names of all motifs
            output_dir: Output directory
        """
        n_motifs = len(motif_indices)
        fig, axes = plt.subplots(2, min(5, n_motifs), figsize=(min(5, n_motifs) * 3, 6))

        if n_motifs == 1:
            axes = axes.reshape(-1, 1)
        elif n_motifs < 5:
            # Pad with empty axes
            axes = np.pad(axes, ((0, 0), (0, 5 - n_motifs)), mode="constant")

        for i, idx in enumerate(motif_indices[:10]):
            if i >= 5:
                row = 1
                col = i - 5
            else:
                row = 0
                col = i

            ax = axes[row, col] if n_motifs > 1 else axes[row, 0]

            # Extract footprint
            if footprints.ndim == 4:
                # Average across cutoffs and clusters
                motif_fp = footprints[:, idx, :, :].mean(axis=(0, 2))
            elif footprints.ndim == 3:
                motif_fp = footprints[idx, :, :].mean(axis=1)
            else:
                motif_fp = footprints[idx, :]

            # Plot
            x_pos = np.arange(-self.extend, self.extend)
            ax.plot(x_pos, motif_fp, color="darkblue", linewidth=1.5)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(
                motif_names[idx] if idx < len(motif_names) else f"Motif {idx}",
                fontsize=10,
            )
            ax.set_xlabel("Distance (bp)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Accessibility", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

        # Remove empty axes
        for i in range(n_motifs, 10):
            if i >= 5:
                row = 1
                col = i - 5
            else:
                row = 0
                col = i
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].set_visible(False)

        plt.suptitle("Top Footprint Profiles", fontsize=14)
        plt.tight_layout()

        output_file = output_dir / "footprint_summary.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved footprint summary: {output_file}")
