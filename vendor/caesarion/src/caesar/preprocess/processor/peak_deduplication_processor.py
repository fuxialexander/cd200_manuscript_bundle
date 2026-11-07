"""Peak deduplication processor using mmseqs clustering.

This processor removes redundant peak sequences by clustering similar sequences
using mmseqs2, keeping only representative sequences from each cluster.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from caesar.preprocess.core.base import BaseProcessor
from caesar.preprocess.core.exceptions import ValidationError

if TYPE_CHECKING:
    from caesar.preprocess.core.config import CaesarConfig

logger = logging.getLogger(__name__)


class PeakDeduplicationProcessor(BaseProcessor):
    """Processor for deduplicating peaks using mmseqs clustering.

    This processor:
    1. Takes a BED file with peaks
    2. Extracts sequences and runs mmseqs clustering
    3. Keeps only representative sequences from each cluster
    4. Outputs a deduplicated BED file
    """

    def __init__(self, config: CaesarConfig):
        """Initialize PeakDeduplicationProcessor.

        Args:
            config: Structured CaesarConfig instance
        """
        super().__init__(config)

        # Direct access to configuration
        self.dedup_config = config.peak_deduplication
        self.input_bed = Path(self.dedup_config.input_bed)
        self.output_bed = Path(self.dedup_config.output_bed)
        self.genome_fasta = Path(self.dedup_config.genome_fasta)
        self.min_seq_id = self.dedup_config.min_seq_id
        self.coverage = self.dedup_config.coverage
        self.script_dir = (
            Path(self.dedup_config.script_dir)
            if self.dedup_config.script_dir
            else None
        )

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate deduplication configuration."""
        if not self.input_bed.exists():
            raise ValidationError(f"Input BED file not found: {self.input_bed}")

        if not self.genome_fasta.exists():
            raise ValidationError(f"Genome FASTA file not found: {self.genome_fasta}")

        if not 0 < self.min_seq_id <= 1:
            raise ValidationError(
                f"min_seq_id must be between 0 and 1, got: {self.min_seq_id}"
            )

        if not 0 < self.coverage <= 1:
            raise ValidationError(
                f"coverage must be between 0 and 1, got: {self.coverage}"
            )

    def process(self) -> dict[str, Any]:
        """Run peak deduplication.

        Returns:
            Processing results dictionary
        """
        self._start_processing()

        logger.info("Loading input BED file...")
        all_peaks = pd.read_csv(self.input_bed, sep="\t")

        # Determine script directory
        if self.script_dir is None:
            # Try to find scripts in the peak_io directory
            peak_io_dir = Path(__file__).parent.parent.parent / "io" / "peak_io"
            if (peak_io_dir / "mmseqs_cluster_analysis.py").exists():
                script_dir = peak_io_dir
            else:
                # Fallback: use current directory or raise error
                script_dir = Path.cwd()
                logger.warning(
                    f"mmseqs scripts not found in default location. "
                    f"Looking in: {script_dir}"
                )
        else:
            script_dir = self.script_dir

        # Create temporary output prefix
        output_prefix = str(self.output_bed.with_suffix(""))

        # Save BED file without header for mmseqs
        bed_file = f"{output_prefix}.bed"
        all_peaks[["Chromosome", "Start", "End"]].to_csv(
            bed_file, index=False, sep="\t", header=False
        )

        # Save BED file with header for merging
        bed_with_header = f"{output_prefix}.with_header.bed"
        all_peaks.to_csv(bed_with_header, index=False, sep="\t", header=True)

        # Run mmseqs clustering
        cluster_file = f"{output_prefix}_cluster.tsv"
        dedup_script = script_dir / "mmseqs_cluster_analysis.py"

        if not dedup_script.exists():
            logger.warning(
                f"mmseqs script not found at {dedup_script}, skipping deduplication"
            )
            logger.info("Returning original peaks without deduplication")
            self.output_bed.parent.mkdir(parents=True, exist_ok=True)
            all_peaks.to_csv(self.output_bed, sep="\t", index=False)
            runtime = self._end_processing()
            return {
                "output_path": str(self.output_bed),
                "input_peaks": len(all_peaks),
                "output_peaks": len(all_peaks),
                "deduplicated": False,
                "runtime": runtime,
            }

        cmd = [
            sys.executable,
            str(dedup_script),
            "--bed",
            bed_file,
            "--genome",
            str(self.genome_fasta),
            "-o",
            output_prefix,
            "--min-seq-id",
            str(self.min_seq_id),
            "--coverage",
            str(self.coverage),
        ]

        logger.info(f"Running mmseqs clustering: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Error running mmseqs clustering: {result.stderr}")
            logger.info("Returning original peaks without deduplication")
            self.output_bed.parent.mkdir(parents=True, exist_ok=True)
            all_peaks.to_csv(self.output_bed, sep="\t", index=False)
            runtime = self._end_processing()
            return {
                "output_path": str(self.output_bed),
                "input_peaks": len(all_peaks),
                "output_peaks": len(all_peaks),
                "deduplicated": False,
                "error": result.stderr,
                "runtime": runtime,
            }

        # Merge clustering results with original peaks
        merge_script = script_dir / "create_deduplicated_peaks.py"
        output_file = f"{output_prefix}.deduplicated.bed"

        if not merge_script.exists():
            logger.warning(
                f"merge script not found at {merge_script}, skipping merge"
            )
            logger.info("Returning original peaks without deduplication")
            self.output_bed.parent.mkdir(parents=True, exist_ok=True)
            all_peaks.to_csv(self.output_bed, sep="\t", index=False)
            runtime = self._end_processing()
            return {
                "output_path": str(self.output_bed),
                "input_peaks": len(all_peaks),
                "output_peaks": len(all_peaks),
                "deduplicated": False,
                "runtime": runtime,
            }

        cmd = [
            sys.executable,
            str(merge_script),
            "--original-bed",
            bed_with_header,
            "--cluster-file",
            cluster_file,
            "--output",
            output_file,
        ]

        logger.info(f"Running merge script: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Error merging clustering results: {result.stderr}")
            logger.info("Returning original peaks without deduplication")
            self.output_bed.parent.mkdir(parents=True, exist_ok=True)
            all_peaks.to_csv(self.output_bed, sep="\t", index=False)
            runtime = self._end_processing()
            return {
                "output_path": str(self.output_bed),
                "input_peaks": len(all_peaks),
                "output_peaks": len(all_peaks),
                "deduplicated": False,
                "error": result.stderr,
                "runtime": runtime,
            }

        # Load deduplicated peaks
        deduplicated = pd.read_csv(output_file, sep="\t")
        logger.info(
            f"Deduplicated from {len(all_peaks)} to {len(deduplicated)} peaks "
            f"({len(deduplicated)/len(all_peaks)*100:.1f}% retained)"
        )

        # Save final output
        self.output_bed.parent.mkdir(parents=True, exist_ok=True)
        deduplicated.to_csv(self.output_bed, sep="\t", index=False)

        runtime = self._end_processing()

        return {
            "output_path": str(self.output_bed),
            "input_peaks": len(all_peaks),
            "output_peaks": len(deduplicated),
            "deduplicated": True,
            "reduction": len(all_peaks) - len(deduplicated),
            "reduction_percent": (1 - len(deduplicated) / len(all_peaks)) * 100,
            "runtime": runtime,
        }

