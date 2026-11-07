"""library size processor with simplified configuration access."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from caesar.preprocess.core.base import AnnotationProcessor
from caesar.preprocess.core.exceptions import ValidationError

if TYPE_CHECKING:
    import argparse
    from caesar.preprocess.core.config import CaesarConfig, LibrarySizeConfig


class LibrarySizeProcessor(AnnotationProcessor):
    """library size processor with direct configuration access.

    This processor calculates and stores library size metrics accounting for
    blacklist regions, using a simplified configuration pattern that provides
    type-safe direct access to configuration fields.

    Library size metrics calculated:
    - library_size_total: Total signal across all regions
    - library_size_blacklist: Signal in blacklisted regions
    - library_size_nonblacklist: Signal in non-blacklisted regions (default)
    """

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add library-size-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Input/Output
        parser.add_argument(
            "-i", "--input", required=False,
            help="Input celltype dense Zarr or BPCells store"
        )

        # Library size parameters
        parser.add_argument(
            "--blacklist",
            help="Blacklist file for filtering regions"
        )
        parser.add_argument(
            "--overwrite", action="store_true",
            help="Overwrite existing library size annotations"
        )

    def __init__(self, config: CaesarConfig):
        """Initialize LibrarySizeProcessor.

        Args:
            config: Full CaesarConfig object
        """
        # Initialize base class
        super().__init__(config)

        # Store full config for direct access
        self.config = config

        # Direct access to configuration - clean and simple
        self.libsize_config: LibrarySizeConfig = config.library_size
        self.n_jobs = self.libsize_config.n_jobs
        self.overwrite = self.libsize_config.overwrite

        # Get blacklist file from assembly if not specified
        self.blacklist_file = self._get_blacklist_file()

        # Validate configuration
        self.validate_config()

    def _get_blacklist_file(self) -> str | None:
        """Get blacklist file path.

        Returns blacklist file from config or assembly default.
        """
        # Check if blacklist_file is specified at root level
        if self.config.blacklist_file:
            return self.config.blacklist_file

        # Use assembly default if available
        if self.config.assembly and self.config.assembly != "auto":
            # This would typically look up the default blacklist for the assembly
            # For now, return None to use the default behavior
            return None

        return None

    def validate_config(self) -> None:
        """Validate library size annotation configuration."""
        # Validate n_jobs
        if self.n_jobs < 1:
            raise ValidationError("n_jobs must be at least 1")

        # Validate blacklist file if specified
        if self.blacklist_file:
            blacklist_path = Path(self.blacklist_file)
            if not blacklist_path.exists():
                raise ValidationError(f"Blacklist file not found: {blacklist_path}")

    def _detect_store_type(self, input_path: Path) -> str:
        """Detect if store is Zarr or BPCells (dense formats only).

        Uses the unified backend-specific classmethods for detection.

        Args:
            input_path: Path to the data store

        Returns:
            Store type: 'zarr_dense', 'bpcells_celltype_dense', 'bpcells_signal_dense'
        """
        from caesar.io.zarr_io.base import ZarrIO
        from caesar.io.bpcell_io import BPCellsIO

        # Try Zarr detection
        zarr_type = ZarrIO.detect_store_type(input_path)
        if zarr_type in ["celltype_dense", "dense", "signal_dense"]:
            self.logger.info(f"Detected Zarr {zarr_type} at {input_path}")
            return "zarr_dense"

        # Try BPCells detection
        bpcells_type = BPCellsIO.detect_store_type(input_path)
        if bpcells_type in ["bpcells_celltype_dense", "bpcells_signal_dense"]:
            self.logger.info(f"Detected BPCells {bpcells_type} at {input_path}")
            return bpcells_type

        raise ValueError(
            f"Cannot determine store type for {input_path}: "
            "no recognized dense store structure found. "
            "LibrarySizeProcessor only supports dense stores (Zarr or BPCells celltype_dense/signal_dense)."
        )

    def process(self) -> dict[str, Any]:
        """Process library size annotation.

        Supports both Zarr and BPCells dense stores.

        Returns:
            Processing results with library size metrics
        """
        self._start_processing()

        # Check if library size annotation is disabled
        skipped = self._check_enabled("library_size")
        if skipped:
            return skipped

        # Get target path for in-place modification
        store_path = self.validate_input_path(self.get_target_path())

        # Detect store type
        store_type = self._detect_store_type(store_path)

        # Open appropriate IO class
        if store_type == "zarr_dense":
            from caesar.io.zarr_io import open_zarr
            io = open_zarr(store_path, mode="a")
        elif store_type == "bpcells_celltype_dense":
            from caesar.io.bpcell_io import CelltypeDenseBPCellsIO
            io = CelltypeDenseBPCellsIO(str(store_path), mode="a")
        elif store_type == "bpcells_signal_dense":
            from caesar.io.bpcell_io import SignalDenseBPCellsIO
            io = SignalDenseBPCellsIO(str(store_path), mode="a")
        else:
            raise ValueError(f"Unsupported store type: {store_type}")

        # Perform annotation
        results = self.annotate(io)

        runtime = self._end_processing()
        results["runtime"] = runtime
        results["input_path"] = str(store_path)
        results["output_path"] = str(store_path)  # Library size annotates in place

        return results

    def annotate(self, io) -> dict[str, Any]:
        """Add library size annotations to zarr or BPCells stores.

        Args:
            io: IO object (DenseZarrIO, CelltypeDenseZarrIO, CelltypeDenseBPCellsIO, or SignalDenseBPCellsIO)

        Returns:
            Library size calculation results
        """
        # Check if mixin is available
        if not hasattr(io, "annotate_library_sizes"):
            raise AttributeError(
                f"{io.__class__.__name__} does not have LibrarySizeAnnotationMixin. "
                "Please update your Caesar installation."
            )

        # Perform annotation
        self.logger.info("Calculating library sizes with blacklist annotation...")
        results = io.annotate_library_sizes(
            blacklist_file=self.blacklist_file,
            n_jobs=self.n_jobs,
            overwrite=self.overwrite,
        )

        # Log and prepare summary
        summary = {"n_tracks": len(results), "tracks": {}}

        for track_id, metrics in results.items():
            blacklist_pct = (
                (metrics["blacklist"] / metrics["total"] * 100)
                if metrics["total"] > 0
                else 0
            )

            self.logger.info(
                f"  {track_id}: total={metrics['total']:.0f}, "
                f"blacklist={metrics['blacklist']:.0f} ({blacklist_pct:.1f}%), "
                f"non-blacklist={metrics['nonblacklist']:.0f}"
            )

            summary["tracks"][track_id] = {
                "total": float(metrics["total"]),
                "blacklist": float(metrics["blacklist"]),
                "nonblacklist": float(metrics["nonblacklist"]),
                "blacklist_percentage": float(blacklist_pct),
            }

        return summary
