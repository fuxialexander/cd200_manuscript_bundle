"""Simplified BAM processor using unified configuration.

This demonstrates how processors are simplified by using the full
CaesarConfig object instead of complex parameter extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from caesar.preprocess.core.base import PreprocessingProcessor
from caesar.preprocess.core.config import CaesarConfig, validate_bam_config
from caesar.preprocess.core.exceptions import ProcessingError

if TYPE_CHECKING:
    import argparse


class BAMProcessor(PreprocessingProcessor):
    """Process BAM files to coverage tracks with simplified configuration.

    This processor handles aligned BAM files and creates dense coverage
    arrays. Supports both single-end and paired-end sequencing data
    with optional Tn5 shifting.
    """

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add BAM-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # BAM-specific arguments
        parser.add_argument(
            "-i", "--input", nargs="+", required=True, help="Input BAM/CRAM files"
        )
        parser.add_argument(
            "-o", "--output", default="coverage.zarr", help="Output zarr path (default: coverage.zarr)"
        )
        parser.add_argument(
            "--single-end", action="store_true", help="Treat as single-end data"
        )
        # Signal mode: mutually exclusive --tn5 and --chip
        signal_group = parser.add_mutually_exclusive_group()
        signal_group.add_argument(
            "--tn5", action="store_true", default=True,
            help="Use Tn5 mode for ATAC-seq/CUT&Tag: extract cut sites with Tn5 bias correction (+4/-5 bp shift) (default)"
        )
        signal_group.add_argument(
            "--chip", action="store_true",
            help="Use ChIP-seq mode: use entire read/fragment span as signal, no Tn5 bias correction"
        )
        # Keep deprecated --tn5-shift for backwards compatibility
        parser.add_argument(
            "--tn5-shift", action="store_true", dest="tn5_shift_deprecated",
            help="(Deprecated) Use --tn5 instead. Apply Tn5 shift for ATAC-seq"
        )
        parser.add_argument(
            "--backend", choices=["zarr", "bpcells"], default="zarr",
            help="Backend to use: 'zarr' for Zarr format, 'bpcells' for BPCells format (default: zarr)"
        )
        parser.add_argument(
            "--assembly", type=str, default="auto",
            help="Genome assembly (e.g., 'hg38', 'mm10', 'auto'). Default: 'auto' (detect from BAM)"
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return ("Process BAM/CRAM files to coverage arrays. "
                "Supports --tn5 mode (ATAC-seq/CUT&Tag: cut sites with Tn5 bias correction) "
                "and --chip mode (ChIP-seq: full read/fragment span as signal).")

    def __init__(self, config: CaesarConfig):
        """Initialize BAM processor with unified configuration.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        # Direct access to BAM configuration
        self.bam_config = config.bam

        # Get BAM files
        self.bam_files = self._get_bam_files()

        # Validate configuration
        self.validate_config()

    @property
    def signal_mode(self) -> str:
        """Get the signal mode (tn5 or chip).

        Returns:
            Signal mode string: "tn5" for ATAC-seq/CUT&Tag, "chip" for ChIP-seq
        """
        return self.bam_config.signal_mode

    @property
    def use_tn5_shift(self) -> bool:
        """Check if Tn5 shifting should be applied.

        This returns True for tn5 mode (ATAC-seq/CUT&Tag) and False for chip mode.

        Returns:
            True if Tn5 shifting should be applied
        """
        return self.signal_mode == "tn5"

    def get_input_files(self) -> list[Path]:
        """Get BAM files from configuration.

        Overrides base class method to extract BAM-specific files.
        """
        files = []
        if self.config.data.bam:
            for item in self.config.data.bam:
                files.append(Path(item.path))
        return files

    def _get_bam_files(self) -> list[Path]:
        """Get BAM files from configuration.

        Deprecated: Use get_input_files() instead.
        """
        return self.get_input_files()

    def validate_config(self) -> None:
        """Validate BAM processor configuration."""
        # Validate BAM-specific config
        validate_bam_config(self.bam_config)

        # Check files exist
        if not self.bam_files:
            raise ProcessingError("No BAM files specified")

        for f in self.bam_files:
            if not f.exists():
                raise ProcessingError(f"BAM file not found: {f}")

    def process(self) -> dict[str, Any]:
        """Process BAM files to coverage tracks.

        Returns:
            Processing results dictionary
        """
        self._start_processing()

        # Determine output path based on backend
        backend = self.bam_config.backend
        if backend == "bpcells":
            # BPCells uses directory-based output
            if len(self.bam_files) == 1:
                sample_id = self._get_sample_id(0)
                output_path = self.get_output_path(f"{sample_id}.bpcells")
            else:
                # For multiple BAMs, process all in one call to single output directory
                output_path = self.get_output_path("celltype_dense_bpcells")
        else:  # zarr backend
            if len(self.bam_files) == 1:
                sample_id = self._get_sample_id(0)
                output_path = self.get_output_path(f"{sample_id}.zarr")
            else:
                output_path = self.get_output_path("celltype_dense.zarr")

        # For BPCells backend with multiple BAMs, process all in one call
        if backend == "bpcells" and len(self.bam_files) > 1:
            return self._process_multiple_bams_bpcells(output_path)

        # Single BAM or Zarr backend - process individually
        results = []

        for i, bam_file in enumerate(self.bam_files):
            sample_info = self._extract_sample_info(i)

            # Determine assembly if auto
            if self.assembly == "auto":
                assembly = self._determine_assembly_from_bam(bam_file)
            else:
                assembly = self.assembly

            # Single BAM or Zarr backend - use output_path directly
            bam_output_path = output_path

            # Process BAM
            result = self._process_single_bam(
                bam_file, bam_output_path, sample_info, assembly
            )
            results.append(result)

        runtime = self._end_processing()

        # Return consolidated results
        if len(results) == 1:
            result = results[0]
            result["runtime"] = runtime
            return result
        else:
            return {
                "output_path": str(output_path),
                "bam_files": [str(f) for f in self.bam_files],
                "results": results,
                "runtime": runtime,
                "backend": backend,
            }

    def _get_sample_id(self, index: int) -> str:
        """Get sample ID for a BAM file."""
        if self.config.data.bam and index < len(self.config.data.bam):
            return self.config.data.bam[index].id
        return f"sample_{index}"

    def _extract_sample_info(self, index: int) -> dict[str, str]:
        """Extract sample information from config."""
        sample_id = "unknown"
        celltype = "unknown"

        if self.config.data.bam and index < len(self.config.data.bam):
            bam_info = self.config.data.bam[index]
            sample_id = bam_info.id
            celltype = bam_info.celltype

        return {"id": sample_id, "celltype": celltype}

    def _determine_assembly_from_bam(self, bam_file: Path) -> str:
        """Determine genome assembly from BAM file.

        Args:
            bam_file: Path to BAM file

        Returns:
            Assembly name
        """
        import pysam

        try:
            with pysam.AlignmentFile(str(bam_file), "rb") as bamfile:
                # Check header for assembly info
                header_dict = dict(bamfile.header)

                # Look for assembly in @PG lines
                if "PG" in header_dict:
                    for pg in header_dict["PG"]:
                        if "VN" in pg:
                            version = pg["VN"].lower()
                            if "hg38" in version:
                                return "hg38"
                            elif "hg19" in version:
                                return "hg19"
                            elif "mm10" in version:
                                return "mm10"
                            elif "mm9" in version:
                                return "mm9"

                # Check reference sequence names
                refs = bamfile.references[:10]  # Check first 10

                # Human assemblies typically have chr1, chr2, etc.
                if any("chr" in ref for ref in refs):
                    # Check chromosome sizes to distinguish human vs mouse
                    # This is a simplified heuristic
                    chr1_length = bamfile.lengths[bamfile.references.index("chr1")]
                    if chr1_length > 240_000_000:  # Human chr1 is ~249M
                        return "hg38"
                    else:  # Mouse chr1 is ~195M
                        return "mm10"

            self.logger.warning(
                f"Could not determine assembly from {bam_file}, defaulting to hg38"
            )
            return "hg38"

        except Exception as e:
            self.logger.warning(f"Assembly detection failed: {e}, defaulting to hg38")
            return "hg38"

    def _process_multiple_bams_bpcells(
        self,
        output_path: Path,
    ) -> dict[str, Any]:
        """Process multiple BAM files to BPCells format in a single call.

        This uses the multi-BAM function which processes all files together
        with unique cell prefixes (bulk.FILENAME).
        """
        # Determine assembly (use first BAM if auto)
        if self.assembly == "auto":
            assembly = self._determine_assembly_from_bam(self.bam_files[0])
        else:
            assembly = self.assembly

        # Extract sample info for all BAMs
        sample_infos = [self._extract_sample_info(i) for i in range(len(self.bam_files))]

        # Get chromosome filter config
        chr_filter_config = None
        if hasattr(self.config, "chromosome_filter") and self.config.chromosome_filter:
            chr_filter_config = self.config.chromosome_filter

        # Process all BAM files together
        from caesar.preprocess.utils.bam_utils import process_bam_multi_to_bpcells

        library_sizes = process_bam_multi_to_bpcells(
            bam_files=[str(f) for f in self.bam_files],
            output_dir=str(output_path),
            sample_ids=[info["id"] for info in sample_infos],
            celltypes=[info["celltype"] for info in sample_infos],
            assembly=assembly,
            signal_mode=self.signal_mode,
            threads=self.bam_config.n_jobs if self.bam_config.n_jobs > 0 else 0,
            chr_filter_config=chr_filter_config,
        )

        # Save attributes to attrs.json
        self._save_bpcells_attributes(output_path, assembly)

        # Create results list
        results = []
        for i, (bam_file, sample_info) in enumerate(zip(self.bam_files, sample_infos)):
            group_name = f"{sample_info['celltype']}.{sample_info['id']}"
            results.append({
                "id": sample_info["id"],
                "celltype": sample_info["celltype"],
                "output_path": str(output_path),  # All share the same output directory
                "n_fragments": library_sizes.get(group_name, 0),
                "assembly": assembly,
                "backend": "bpcells",
            })

        runtime = self._end_processing()

        return {
            "output_path": str(output_path),
            "bam_files": [str(f) for f in self.bam_files],
            "results": results,
            "runtime": runtime,
            "backend": "bpcells",
        }

    def _process_single_bam(
        self,
        bam_file: Path,
        output_path: Path,
        sample_info: dict[str, str],
        assembly: str,
    ) -> dict[str, Any]:
        """Process a single BAM file.

        Args:
            bam_file: Path to BAM file
            output_path: Output path (zarr file or bpcells directory)
            sample_info: Sample metadata
            assembly: Genome assembly

        Returns:
            Processing results
        """
        # Create chromosome filter config dict
        chr_filter_config = {
            "max_len": self.config.chromosome_filter.max_length,
            "exclude_patterns": self.config.chromosome_filter.exclude_patterns,
            "exclude_exact": self.config.chromosome_filter.exclude_exact,
            "include_patterns": self.config.chromosome_filter.include_patterns,
            "include_exact": self.config.chromosome_filter.include_exact,
        }

        # Route to appropriate backend
        backend = self.bam_config.backend
        self.logger.info(f"Processing BAM file with {backend} backend: {bam_file}")

        if backend == "bpcells":
            from caesar.preprocess.utils.bam_utils import process_bam_to_bpcells

            # For BPCells, output_path is a directory
            library_size = process_bam_to_bpcells(
                bam_file=str(bam_file),
                output_dir=str(output_path),
                sample_id=sample_info["id"],
                celltype=sample_info["celltype"],
                assembly=assembly,
                signal_mode=self.signal_mode,
                threads=self.bam_config.n_jobs if self.bam_config.n_jobs > 0 else 0,
                chr_filter_config=chr_filter_config,
            )

            # Save attributes to attrs.json
            self._save_bpcells_attributes(output_path, assembly)
        else:  # backend == "zarr"
            from caesar.preprocess.utils.bam_utils import process_bam_to_zarr

            # For Zarr, output_path is a zarr file
            library_size = process_bam_to_zarr(
                bam_file=str(bam_file),
                output_zarr=str(output_path),
                sample_id=sample_info["id"],
                celltype=sample_info["celltype"],
                assembly=assembly,
                single_end=self.bam_config.single_end,
                signal_mode=self.signal_mode,
                chunk_size=self.bam_config.chunk_size,
                n_jobs=self.bam_config.n_jobs,
                chr_filter_config=chr_filter_config,
            )

        return {
            "id": sample_info["id"],
            "celltype": sample_info["celltype"],
            "output_path": str(output_path),
            "n_fragments": library_size,
            "assembly": assembly,
            "backend": backend,
        }

    def _save_bpcells_attributes(self, output_path: Path, assembly: str) -> None:
        """Save BPCells attributes file with assembly information.

        Args:
            output_path: Path to BPCells output directory
            assembly: Genome assembly name
        """
        import json

        # Determine shift parameters based on signal mode
        if self.signal_mode == "tn5":
            shift_start = 4
            shift_end = -5
        else:  # chip mode - no shift
            shift_start = 0
            shift_end = 0

        attrs = {
            "assembly": assembly,
            "store_type": "bpcells_celltype_dense",
            "class": "CelltypeDenseBPCellsIO",
            "source": "bam_files" if len(self.bam_files) == 1 else "bam_files_multi",
            "bam_files": [str(f) for f in self.bam_files],
            "signal_mode": self.signal_mode,
            "shift_start": shift_start,
            "shift_end": shift_end,
        }

        # Try to get group names from the BPCells output
        try:
            from bpcells.experimental import PrecalculatedInsertionMatrix
            matrix = PrecalculatedInsertionMatrix(str(output_path))
            attrs["group_names"] = matrix.group_names
        except Exception as e:
            self.logger.debug(f"Could not load group names from matrix: {e}")

        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        attrs_path = output_path / "attrs.json"
        try:
            with open(attrs_path, "w") as f:
                json.dump(attrs, f, indent=2)
            self.logger.info(f"Saved attributes to {attrs_path} (assembly: {assembly})")
        except Exception as e:
            self.logger.error(f"Failed to save attributes to {attrs_path}: {e}")
            raise
