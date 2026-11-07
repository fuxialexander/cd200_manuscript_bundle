"""BigWig processor for converting BigWig files to Zarr format.

This processor handles BigWig file conversion to Zarr format,
supporting both single-file (SignalDense) and multi-file (CelltypeDense) outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from caesar.preprocess.core.base import PreprocessingProcessor
from caesar.preprocess.core.config import CaesarConfig
from caesar.preprocess.core.exceptions import ProcessingError

if TYPE_CHECKING:
    import argparse


class BigWigProcessor(PreprocessingProcessor):
    """Convert BigWig files to Zarr format.

    This processor reads BigWig files and converts them to Zarr format:
    - Single file: Creates SignalDenseZarrIO compatible output
    - Multiple files: Creates CelltypeDenseZarrIO compatible output

    BigWig files contain continuous signal data (float values), so this
    processor only supports Zarr output (not BPCells, which is designed
    for integer count data).
    """

    @classmethod
    def get_command_name(cls) -> str:
        """Return 'bigwig' as the command name."""
        return "bigwig"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add BigWig-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # BigWig-specific arguments
        parser.add_argument(
            "-i",
            "--input",
            nargs="+",
            required=True,
            help="Input BigWig file(s). Multiple files create CelltypeDense output.",
        )
        parser.add_argument(
            "-o",
            "--output",
            default="output.zarr",
            help="Output zarr path (default: output.zarr)",
        )
        parser.add_argument(
            "--assembly",
            type=str,
            default="hg38",
            help="Genome assembly (e.g., 'hg38', 'mm10'). Default: hg38",
        )
        parser.add_argument(
            "--sample-ids",
            nargs="+",
            help="Sample IDs for each input file. If not provided, derived from filenames.",
        )
        parser.add_argument(
            "--celltypes",
            nargs="+",
            help="Cell type names for each input file. If not provided, uses 'signal'.",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=2_000_000,
            help="Chunk size for zarr arrays (default: 2000000)",
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return (
            "Convert BigWig files to Zarr format. "
            "Single file creates SignalDense output, multiple files create CelltypeDense output."
        )

    def __init__(self, config: CaesarConfig):
        """Initialize BigWig processor with full configuration.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Get BigWig files from config
        self.bigwig_files = self._get_bigwig_files()

        # Validate configuration
        self.validate_config()

    def get_input_files(self) -> list[Path]:
        """Get BigWig files from configuration."""
        files = []
        if self.config.data.bigwig:
            for item in self.config.data.bigwig:
                files.append(Path(item.path))
        return files

    def _get_bigwig_files(self) -> list[Path]:
        """Get BigWig files from configuration."""
        return self.get_input_files()

    def validate_config(self) -> None:
        """Validate BigWig processor configuration."""
        # Check files exist
        if not self.bigwig_files:
            raise ProcessingError("No BigWig files specified")

        for f in self.bigwig_files:
            if not f.exists():
                raise ProcessingError(f"BigWig file not found: {f}")

    def _get_sample_id(self, index: int) -> str:
        """Get sample ID for a BigWig file."""
        if self.config.data.bigwig and index < len(self.config.data.bigwig):
            return self.config.data.bigwig[index].id
        # Derive from filename
        return self.bigwig_files[index].stem

    def _get_celltype(self, index: int) -> str:
        """Get cell type for a BigWig file."""
        if self.config.data.bigwig and index < len(self.config.data.bigwig):
            return self.config.data.bigwig[index].celltype
        return "signal"

    def process(self) -> dict[str, Any]:
        """Process BigWig files to Zarr format.

        Returns:
            Processing results dictionary
        """
        self._start_processing()

        # Determine output path
        if len(self.bigwig_files) == 1:
            sample_id = self._get_sample_id(0)
            output_path = self.get_output_path(f"{sample_id}.zarr")
        else:
            output_path = self.get_output_path("celltype_dense.zarr")

        # Get chromosome filter config
        chr_filter_config = None
        if hasattr(self.config, "chromosome_filter") and self.config.chromosome_filter:
            chr_filter_config = {
                "max_len": self.config.chromosome_filter.max_length,
                "exclude_patterns": self.config.chromosome_filter.exclude_patterns,
                "exclude_exact": self.config.chromosome_filter.exclude_exact,
                "include_patterns": self.config.chromosome_filter.include_patterns,
                "include_exact": self.config.chromosome_filter.include_exact,
            }

        # Get processing parameters
        chunk_size = getattr(self.config.bigwig, "chunk_size", 2_000_000)
        n_jobs = getattr(self.config.bigwig, "n_jobs", 4)

        if len(self.bigwig_files) == 1:
            # Single file - create SignalDense output
            result = self._process_single_bigwig(
                self.bigwig_files[0],
                output_path,
                self._get_sample_id(0),
                chunk_size,
                n_jobs,
                chr_filter_config,
            )
        else:
            # Multiple files - create CelltypeDense output
            result = self._process_multiple_bigwigs(
                output_path,
                chunk_size,
                n_jobs,
                chr_filter_config,
            )

        runtime = self._end_processing()
        result["runtime"] = runtime

        return result

    def _process_single_bigwig(
        self,
        bigwig_file: Path,
        output_path: Path,
        sample_id: str,
        chunk_size: int,
        n_jobs: int,
        chr_filter_config: dict | None,
    ) -> dict[str, Any]:
        """Process a single BigWig file to SignalDense Zarr.

        Args:
            bigwig_file: Path to BigWig file
            output_path: Output zarr path
            sample_id: Sample ID
            chunk_size: Chunk size for zarr arrays
            n_jobs: Number of parallel jobs
            chr_filter_config: Chromosome filter configuration

        Returns:
            Processing results
        """
        from caesar.preprocess.utils.bigwig_utils import (
            process_bigwig_to_signal_dense_zarr,
        )

        self.logger.info(f"Processing single BigWig file: {bigwig_file}")

        total_signal = process_bigwig_to_signal_dense_zarr(
            bigwig_file=str(bigwig_file),
            output_zarr=str(output_path),
            sample_id=sample_id,
            assembly=self.assembly,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            chr_filter_config=chr_filter_config,
        )

        return {
            "output_path": str(output_path),
            "bigwig_file": str(bigwig_file),
            "sample_id": sample_id,
            "total_signal": total_signal,
            "assembly": self.assembly,
            "output_type": "signal_dense",
        }

    def _process_multiple_bigwigs(
        self,
        output_path: Path,
        chunk_size: int,
        n_jobs: int,
        chr_filter_config: dict | None,
    ) -> dict[str, Any]:
        """Process multiple BigWig files to CelltypeDense Zarr.

        Args:
            output_path: Output zarr path
            chunk_size: Chunk size for zarr arrays
            n_jobs: Number of parallel jobs
            chr_filter_config: Chromosome filter configuration

        Returns:
            Processing results
        """
        from caesar.preprocess.utils.bigwig_utils import (
            process_bigwig_to_celltype_dense_zarr,
        )

        self.logger.info(f"Processing {len(self.bigwig_files)} BigWig files")

        # Get sample IDs and celltypes
        sample_ids = [self._get_sample_id(i) for i in range(len(self.bigwig_files))]
        celltypes = [self._get_celltype(i) for i in range(len(self.bigwig_files))]

        library_sizes = process_bigwig_to_celltype_dense_zarr(
            bigwig_files=[str(f) for f in self.bigwig_files],
            output_zarr=str(output_path),
            sample_ids=sample_ids,
            celltypes=celltypes,
            assembly=self.assembly,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            chr_filter_config=chr_filter_config,
        )

        # Create results for each file
        results = []
        for i, (bigwig_file, sample_id, celltype) in enumerate(
            zip(self.bigwig_files, sample_ids, celltypes)
        ):
            group_name = f"{celltype}.{sample_id}"
            results.append(
                {
                    "bigwig_file": str(bigwig_file),
                    "sample_id": sample_id,
                    "celltype": celltype,
                    "total_signal": library_sizes.get(group_name, 0),
                }
            )

        return {
            "output_path": str(output_path),
            "bigwig_files": [str(f) for f in self.bigwig_files],
            "results": results,
            "assembly": self.assembly,
            "output_type": "celltype_dense",
        }
