"""normalization processor with simplified configuration access."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from caesar.io.zarr_io import CelltypeDenseZarrIO
from caesar.io.bpcell_io import CelltypeDenseBPCellsIO
from caesar.io.interface import CelltypeDenseIO
from caesar.preprocess.core.base import AnnotationProcessor
from caesar.preprocess.core.exceptions import ValidationError
from gcell.dna.genome import ChromSize

if TYPE_CHECKING:
    import argparse
    from caesar.preprocess.core.config import CaesarConfig


class NormalizationProcessor(AnnotationProcessor):
    """normalization processor with direct configuration access.

    This processor performs quantile normalization on genomic signals
    using a simplified configuration pattern that provides type-safe
    direct access to configuration fields.
    """

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add normalization-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Input/Output
        parser.add_argument(
            "-i", "--input", required=False,
            help="Input celltype dense Zarr or BPCells store"
        )

        # Normalization parameters
        parser.add_argument(
            "--method", choices=["quantile"],
            help="Normalization method (default: quantile)"
        )
        parser.add_argument(
            "--reference", choices=["peaks", "genome-wide"],
            help="Reference for normalization: 'peaks' or 'genome-wide' (default: peaks)"
        )
        parser.add_argument(
            "--peak-file",
            help="Peak file for peak-based normalization (required if reference=peaks)"
        )
        parser.add_argument(
            "--overwrite", action="store_true",
            help="Overwrite existing normalization parameters"
        )
        parser.add_argument(
            "--write-output", action="store_true",
            help="Write normalized data to a new file (Zarr only)"
        )

    def __init__(self, config: CaesarConfig):
        """Initialize NormalizationProcessor.

        Args:
            config: Full CaesarConfig object
        """
        # Initialize base class
        super().__init__(config)

        # Store full config for direct access
        self.config = config

        # Direct access to configuration - clean and simple
        self.enabled = config.normalization.enabled
        self.method = config.normalization.method
        self.reference = config.normalization.reference
        self.peak_file = config.normalization.peak_file
        self.write_output = config.normalization.write_output
        self.conv_size = 20  # Default value, as it's not in NormalizationConfig yet

        # Resources
        self.n_jobs = config.resources.n_jobs

        # Assembly
        self.assembly = config.assembly

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate normalization configuration."""
        # Call parent validation first
        super().validate_config()

        # Skip validation if normalization is disabled
        if not self.enabled:
            self.logger.warning("Normalization is disabled in configuration")
            return

        if self.method not in ["quantile"]:
            raise ValidationError(
                f"Invalid normalization method: {self.method}. Only 'quantile' is supported"
            )

        if self.reference not in ["peaks", "genome-wide"]:
            raise ValidationError(
                f"Invalid reference: {self.reference}. Must be 'peaks' or 'genome-wide'"
            )

        # Note: peak_file validation is deferred to process() to allow auto-discovery
        # from zarr store when running in workflows where peaks are called in a previous step

        # Validate peak file if specified at config time
        if self.peak_file:
            peak_path = Path(self.peak_file)
            if not peak_path.exists():
                raise ValidationError(f"Peak file not found: {peak_path}")

    def _auto_discover_peak_file(self, input_path: Path) -> str | None:
        """Auto-discover peak file from zarr store.

        Looks for peaks in the zarr attributes or common file locations.

        Args:
            input_path: Path to the input zarr/bpcells store

        Returns:
            Path to peak file if found, None otherwise
        """
        import zarr

        # Try to read peaks_file from zarr attributes
        try:
            if input_path.suffix == ".zarr" or (input_path / ".zattrs").exists():
                root = zarr.open(str(input_path), mode="r")
                if "peaks_file" in root.attrs:
                    peaks_path = root.attrs["peaks_file"]
                    if Path(peaks_path).exists():
                        self.logger.info(f"Auto-discovered peak file from zarr attrs: {peaks_path}")
                        return peaks_path
        except Exception:
            pass

        # Look for peaks.bed in common locations
        candidates = [
            input_path / "peaks.bed",
            input_path.parent / "peaks.bed",
            input_path.parent / "peaks" / "peaks.bed",
        ]

        for candidate in candidates:
            if candidate.exists():
                self.logger.info(f"Auto-discovered peak file: {candidate}")
                return str(candidate)

        return None

    def _detect_backend(self, input_path: Path) -> tuple[str, CelltypeDenseIO]:
        """Detect backend type and return appropriate IO object.

        Args:
            input_path: Path to the input data store

        Returns:
            Tuple of (backend_type, io_object) where backend_type is 'zarr' or 'bpcells'
        """
        from caesar.io.genomic_io import GenomicIO

        # Ensure input_path is a Path object, not None
        if input_path is None:
            raise ValueError("input_path cannot be None for backend detection")
        
        # Convert to Path if it's a string
        if isinstance(input_path, str):
            input_path = Path(input_path)

        store_type = GenomicIO.detect_store_type(input_path)
        if store_type is None:
            # Fallback: try to detect by file structure
            if input_path.suffix == ".zarr" or (input_path.is_dir() and (input_path / ".zattrs").exists()):
                store_type = "zarr"
            elif input_path.is_dir() and (input_path / "attrs.json").exists():
                store_type = "bpcells"
            else:
                raise ValueError(
                    f"Cannot determine backend type for {input_path}. "
                    "Expected Zarr (.zarr or .zattrs) or BPCells (attrs.json) format."
                )

        # Determine backend from store_type
        if store_type in ["celltype_dense", "dense", "signal_dense"]:
            backend_type = "zarr"
            io_obj = CelltypeDenseZarrIO(str(input_path), mode="a")
        elif store_type in ["bpcells_celltype_dense", "bpcells_signal_dense"]:
            backend_type = "bpcells"
            io_obj = CelltypeDenseBPCellsIO(str(input_path), mode="a")
        else:
            # Try both backends
            try:
                io_obj = CelltypeDenseZarrIO(str(input_path), mode="a")
                backend_type = "zarr"
                self.logger.info(f"Detected Zarr backend for {input_path}")
            except Exception:
                try:
                    io_obj = CelltypeDenseBPCellsIO(str(input_path), mode="a")
                    backend_type = "bpcells"
                    self.logger.info(f"Detected BPCells backend for {input_path}")
                except Exception as e:
                    raise ValueError(
                        f"Cannot load {input_path} as either Zarr or BPCells format: {e}"
                    )

        return backend_type, io_obj

    def process(self) -> dict[str, Any]:
        """Perform normalization on the input data.

        Supports both Zarr and BPCells backends:
        - For BPCells: Calculates and stores normalization params only (preserves raw counts)
        - For Zarr: Stores params in attrs; optionally writes normalized data if write_output=True

        Returns:
            Dictionary containing normalization results
        """
        self._start_processing()

        # Check if normalization is enabled
        skipped = self._check_enabled("normalization")
        if skipped:
            return skipped

        # Get target path
        target_path = self.get_target_path()
        if target_path is None:
            raise ValueError(
                "No target path specified for normalization. "
                "Set normalization.input in config or ensure previous step sets current_output."
            )
        input_path = self.validate_input_path(target_path)

        # Detect backend and load appropriate IO object
        backend_type, io_obj = self._detect_backend(input_path)
        self.logger.info(f"Processing normalization with {backend_type} backend")

        # For BPCells, always skip writing normalized data (preserve raw counts)
        # For Zarr, respect write_output config
        if backend_type == "bpcells":
            output_path = None
            self.logger.info(
                "BPCells backend detected: normalization params will be stored, "
                "but raw counts will be preserved (no normalized file written)"
            )
        elif self.write_output:
            output_path = self.get_output_path("normalized.zarr")
            self.logger.info(f"Zarr backend with write_output=True: will write normalized data to {output_path}")
        else:
            output_path = None
            self.logger.info(
                "Zarr backend with write_output=False: normalization params will be stored in attrs only"
            )

        # Auto-discover peak file if reference='peaks' but peak_file not provided
        if self.reference == "peaks" and not self.peak_file:
            discovered_peak_file = self._auto_discover_peak_file(input_path)
            if discovered_peak_file:
                self.peak_file = discovered_peak_file
            else:
                raise ValidationError(
                    "peak_file is required when reference='peaks'. "
                    "No peak file was provided and auto-discovery failed. "
                    "Either specify peak_file in config or run peak calling first."
                )

        try:
            if self.method == "quantile":
                if self.reference == "genome-wide":
                    # For genome-wide normalization, create a temporary peak file
                    # that covers the entire genome
                    self.logger.info("Performing genome-wide quantile normalization")

                    # Get chromosome sizes from the IO object or config
                    assembly_for_chrom = (
                        self.assembly if self.assembly != "auto" else io_obj.assembly
                    )

                    # Get chromosome sizes
                    chrom_size = ChromSize(assembly=assembly_for_chrom)

                    # Create genome-wide "peaks" using shared filtering logic
                    from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

                    # Use shared chromosome filtering
                    chr_filter = ChromosomeFilter({})
                    valid_chroms = [
                        chrom
                        for chrom in chrom_size.dict.keys()
                        if chr_filter.is_valid_chromosome(chrom)
                    ]

                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".bed", delete=False
                    ) as tmp:
                        for chrom in valid_chroms:
                            size = chrom_size.dict[chrom]
                            # Write entire chromosome as a "peak"
                            tmp.write(f"{chrom}\t0\t{size}\n")
                        genome_wide_peaks = tmp.name

                    try:
                        norm_results = io_obj.quantile_normalization_peak_based(
                            peak_file=genome_wide_peaks,
                            n_jobs=self.n_jobs,
                            conv_size=self.conv_size,
                            output_zarr_path=str(output_path) if output_path else None,
                        )
                    finally:
                        # Clean up temp file
                        genome_wide_path = Path(genome_wide_peaks)
                        if genome_wide_path.exists():
                            genome_wide_path.unlink()
                else:
                    # Peak-based normalization
                    norm_results = io_obj.quantile_normalization_peak_based(
                        peak_file=str(self.peak_file),
                        n_jobs=self.n_jobs,
                        conv_size=self.conv_size,
                        output_zarr_path=str(output_path) if output_path else None,
                    )
            else:
                raise ValidationError(f"Unknown normalization method: {self.method}")

            runtime = self._end_processing()

            result = {
                "method": self.method,
                "reference": self.reference,
                "backend": backend_type,
                "normalization_params": norm_results,
                "runtime": runtime,
            }

            # For AnnotationProcessor, always return the input path as output_path
            # since we modify the store in-place (even if we don't write a new normalized file)
            # This ensures current_output is preserved for the next workflow step
            result["output_path"] = str(input_path)
            
            if not output_path:
                # Add note when normalization params are stored but no new file is written
                result["note"] = (
                    "Normalization params stored in attrs. "
                    "Use QuantileNormalizer with get_track() to apply normalization."
                )

            return result

        except Exception as e:
            self.logger.error(f"Error during normalization: {e}")
            raise
