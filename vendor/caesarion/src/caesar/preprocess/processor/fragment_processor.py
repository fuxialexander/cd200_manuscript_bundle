"""Fragment processor that handles both single-cell and bulk modes.

This processor uses the actual Caesar preprocessing functions from
sc_accessibility_simple.py and fragment_to_zarr.py to process fragments.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from zarr.core.dtype import VariableLengthUTF8

from caesar.io.metadata_io import (
    load_and_merge_metadata,
    load_cell_metadata,
    validate_cell_labels,
)
from caesar.preprocess.core.base import PreprocessingProcessor
from caesar.preprocess.core.config import CaesarConfig, validate_fragment_config
from caesar.preprocess.core.exceptions import ProcessingError
from caesar.preprocess.core.mixins import CompressorMixin
from caesar.preprocess.processor.metadata_filter_processor import (
    MetadataFilterProcessor,
)
from caesar.preprocess.utils.fragment_info import extract_fragment_info, extract_id_from_path
from gcell.dna.genome import ChromSize


class FragmentProcessor(PreprocessingProcessor, CompressorMixin):
    """Process fragment files for both single-cell and bulk modes.

    This processor automatically detects the processing mode (single-cell vs bulk)
    based on the presence of barcodes in the fragment file, or uses the mode
    specified in the configuration.

    Metadata Handling for Multiple Fragments:
    ----------------------------------------
    The processor supports unified metadata handling compatible with BPCells workflow:

    1. **Unified Merged Metadata** (Recommended for shared metadata):
       - Single shared metadata file with `sample_id` column (or use `--sample-id-col`)
       - Automatically merges metadata from all fragments
       - Handles non-unique barcodes across fragments via `sample_id` disambiguation
       - Requires `sample_id` column when using single metadata with multiple fragments
       - Uses `load_and_merge_metadata()` for consistent behavior with BPCells

    2. **Per-Fragment Metadata** (Backward compatible):
       - Each fragment file can have its own metadata file specified via:
         fragments[i].metadata and fragments[i].metadata_column
       - Loaded independently for each fragment
       - Works when barcodes are unique across fragments

    3. **Global Metadata** (via MetadataFilterProcessor):
       - Specified at data.metadata level
       - Used when metadata_filter is enabled in configuration
       - Provides filtered barcodes and cell type labels
       - Acts as fallback if other metadata not specified

    Priority Order:
    1. Merged metadata (if available and metadata_column specified)
    2. Per-fragment metadata (if fragment.metadata specified)
    3. MetadataFilterProcessor metadata (if enabled and available)
    4. Uniform celltype assignment (default fallback)

    Example Configuration (Unified Metadata):
    ```yaml
    data:
      fragments:
        - path: "sample1.fragments.tsv.gz"
          id: "sample1"
          metadata: "combined_metadata.csv"  # Shared metadata
          metadata_column: "cell_type"
        - path: "sample2.fragments.tsv.gz"
          id: "sample2"
          metadata: "combined_metadata.csv"  # Same shared metadata
          metadata_column: "cell_type"
    ```
    Metadata file should have: `barcode,cell_type,sample_id` columns

    Example Configuration (Per-Fragment Metadata):
    ```yaml
    data:
      fragments:
        - path: "sample1.fragments.tsv.gz"
          id: "sample1"
          metadata: "sample1_cells.csv"  # Per-fragment metadata
          metadata_column: "cell_type"
        - path: "sample2.fragments.tsv.gz"
          id: "sample2"
          metadata: "sample2_cells.csv"  # Different metadata
          metadata_column: "cell_type"
    ```
    """

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add fragment-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Fragment-specific arguments
        parser.add_argument(
            "-i", "--input", nargs="+", required=True, help="Input fragment files"
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            default="./output",
            help="Output directory (default: ./output)",
        )
        parser.add_argument(
            "--mode",
            choices=["auto", "single-cell", "bulk"],
            default="auto",
            help="Processing mode (default: auto-detect)",
        )
        parser.add_argument(
            "--assembly",
            choices=["auto", "hg38", "hg19", "mm10", "mm9"],
            default="auto",
            help="Genome assembly (default: auto-detect)",
        )
        parser.add_argument(
            "--min-cells", type=int, help="Minimum cells for single-cell processing"
        )
        parser.add_argument(
            "--min-fragments", type=int, help="Minimum fragments per cell"
        )
        parser.add_argument(
            "--metadata", nargs="+",
            help="Metadata CSV files (one per input, or single file for all inputs)"
        )
        parser.add_argument(
            "--barcode-col", default="barcode",
            help="Column name for cell barcodes in metadata (default: barcode)"
        )
        parser.add_argument(
            "--sample-id-col", default="sample_id",
            help="Column name for sample IDs in metadata (default: sample_id). "
                 "Required when using single metadata file with non-unique barcodes across fragments."
        )
        # CUT&RUN-specific arguments
        parser.add_argument(
            "--assay-type",
            choices=["auto", "atac", "cutandrun", "cutnrun", "chip"],
            default="auto",
            help="Assay type for fragment processing (default: auto-detect). "
                 "Use 'cutandrun' or 'cutnrun' for CUT&RUN experiments to enable "
                 "fragment size validation and spike-in handling."
        )
        parser.add_argument(
            "--antibody-target",
            help="Antibody target for CUT&RUN (e.g., 'H3K27ac', 'CTCF'). "
                 "Stored in zarr metadata for downstream analysis."
        )
        parser.add_argument(
            "--spike-in-chroms", nargs="+",
            help="Spike-in chromosome names to exclude (e.g., 'chrEcoli' 'chrYeast'). "
                 "Fragments mapping to these chromosomes are filtered out."
        )
        parser.add_argument(
            "--spike-in-factor-file",
            help="JSON file with spike-in normalization factors per sample. "
                 "Format: {\"sample_id\": factor, ...}. Factors are stored but not applied."
        )
        parser.add_argument(
            "--no-validate-fragment-sizes", action="store_true",
            help="Disable fragment size distribution validation for the assay type."
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return (
            "Process fragment files (auto-detects single-cell vs bulk). "
            "Creates sparse matrices for single-cell data or dense coverage arrays for bulk data."
        )

    def __init__(self, config: CaesarConfig):
        """Initialize fragment processor with full configuration.

        Args:
            config: Full CaesarConfig object
        """
        # Initialize base classes
        PreprocessingProcessor.__init__(self, config)
        CompressorMixin.__init__(self)

        # Store full config
        self.config = config

        # Direct access to configuration
        self.assembly = config.assembly
        self.processing_type = config.workflow.type or "auto"  # Use workflow type
        self.fragment_config = config.fragments

        # Get fragment files
        self.fragment_files = self._get_fragment_files()

        # Validate configuration
        validate_fragment_config(self.fragment_config)
        self._validate_files()

        # Initialize metadata filter processor if metadata filtering is enabled
        self.metadata_filter_processor = None
        self.filtered_barcodes = None
        # Only initialize metadata filter if explicitly requested via use_metadata_filter
        # or if metadata_filter has a dataset or metadata_path specified
        if self.fragment_config.use_metadata_filter or (
            hasattr(config, "metadata_filter")
            and config.metadata_filter
            and (
                config.metadata_filter.dataset is not None
                or config.metadata_filter.metadata_path is not None
            )
        ):
            self.metadata_filter_processor = MetadataFilterProcessor(config)

        # Check if dual output is requested
        self.dual_output = getattr(self.fragment_config, "dual_output", False)
        self.output_formats = getattr(
            self.fragment_config, "output_formats", ["sparse"]
        )  # Default to sparse only

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Metadata column names (for unified metadata handling)
        self.barcode_col = self.fragment_config.barcode_col
        self.sample_id_col = self.fragment_config.sample_id_col

        # Load and merge metadata if available (unified approach)
        # Metadata is optional - sparse outputs can be created without it
        self.merged_metadata = None
        if self.config.data.fragments:
            # Only try to load metadata if at least one fragment has metadata specified
            has_any_metadata = any(
                hasattr(frag, "metadata") and frag.metadata
                for frag in self.config.data.fragments
            )
            if has_any_metadata:
                self.merged_metadata = load_and_merge_metadata(
                    fragments=self.config.data.fragments,
                    barcode_col=self.barcode_col,
                    sample_id_col=self.sample_id_col,
                    logger=self.logger,
                )
                if self.merged_metadata is not None:
                    self.logger.info(
                        f"Loaded merged metadata: {len(self.merged_metadata)} cells"
                    )
                else:
                    self.logger.info("No metadata loaded (metadata files not found or empty)")
            else:
                self.logger.info("No metadata specified - will use uniform celltype assignment")

        # CUT&RUN-specific settings
        self.assay_type = self.fragment_config.assay_type
        self.spike_in_chroms = self.fragment_config.spike_in_chroms or []
        self.validate_fragment_sizes = self.fragment_config.validate_fragment_sizes
        self.expected_fragment_size_range = self.fragment_config.expected_fragment_size_range

        # Load spike-in normalization factors if provided
        self.spike_in_factors = self._load_spike_in_factors()

        # Log CUT&RUN settings if applicable
        if self.assay_type in ("cutandrun", "cutnrun"):
            self.logger.info(f"CUT&RUN mode enabled (assay_type={self.assay_type})")
            if self.spike_in_chroms:
                self.logger.info(f"Spike-in chromosomes to filter: {self.spike_in_chroms}")
            if self.spike_in_factors:
                self.logger.info(f"Loaded spike-in factors for {len(self.spike_in_factors)} samples")

    def validate_config(self) -> None:
        """Validate fragment processor configuration.

        This method is called by the base class to ensure configuration is valid.
        """
        # Validation is already done in __init__ via validate_fragment_config
        # and _validate_files, but we need this method to satisfy the abstract base class
        pass

    def get_input_files(self) -> list[Path]:
        """Get fragment files from configuration.

        Overrides base class method to extract fragment-specific files.
        """
        files = []
        if self.config.data.fragments:
            for item in self.config.data.fragments:
                files.append(Path(item.path))
        return files

    def _get_fragment_files(self) -> list[Path]:
        """Get fragment files from configuration.

        Deprecated: Use get_input_files() instead.
        """
        return self.get_input_files()

    def _validate_files(self) -> None:
        """Validate that fragment files exist."""
        if not self.fragment_files:
            raise ProcessingError("No fragment files specified")

        for f in self.fragment_files:
            if not f.exists():
                raise ProcessingError(f"Fragment file not found: {f}")

    def _load_spike_in_factors(self) -> dict[str, float]:
        """Load spike-in normalization factors from JSON file.

        Returns:
            Dictionary mapping sample_id to normalization factor.
            Empty dict if no file specified or file not found.
        """
        factor_file = self.fragment_config.spike_in_factor_file
        if not factor_file:
            return {}

        factor_path = Path(factor_file)
        if not factor_path.exists():
            self.logger.warning(f"Spike-in factor file not found: {factor_file}")
            return {}

        try:
            with open(factor_path) as f:
                factors = json.load(f)
            # Validate that all values are numeric
            validated_factors = {}
            for sample_id, factor in factors.items():
                if isinstance(factor, (int, float)) and factor > 0:
                    validated_factors[str(sample_id)] = float(factor)
                else:
                    self.logger.warning(
                        f"Invalid spike-in factor for {sample_id}: {factor} (must be positive number)"
                    )
            return validated_factors
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse spike-in factor file: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading spike-in factors: {e}")
            return {}

    def _get_fragment_info(self, fragment_file: Path) -> dict[str, Any]:
        """Get CUT&RUN-specific info from fragment file configuration.

        Args:
            fragment_file: Path to fragment file

        Returns:
            Dictionary with antibody_target, spike_in_factor, and assay_type
        """
        info = {
            "antibody_target": None,
            "spike_in_factor": None,
            "assay_type": self.assay_type,
        }

        # Find matching fragment config
        if self.config.data.fragments:
            for frag in self.config.data.fragments:
                if Path(frag.path) == fragment_file:
                    info["antibody_target"] = getattr(frag, "antibody_target", None)
                    info["spike_in_factor"] = getattr(frag, "spike_in_factor", None)
                    info["assay_type"] = getattr(frag, "assay_type", None) or self.assay_type

                    # Also check global spike-in factors
                    frag_id = getattr(frag, "id", None)
                    if frag_id and frag_id in self.spike_in_factors:
                        info["spike_in_factor"] = self.spike_in_factors[frag_id]
                    break

        return info

    def _validate_fragment_sizes(
        self, fragment_file: Path, sample_size: int = 10000
    ) -> dict[str, Any]:
        """Validate fragment size distribution for the assay type.

        CUT&RUN fragments are typically smaller than ATAC-seq fragments:
        - CUT&RUN: Most fragments are <150bp (nucleosome-free) or ~150bp (mono-nucleosome)
        - ATAC-seq: Nucleosome-free (<100bp) + mono-nucleosome (~200bp) + multi-nucleosome peaks

        Args:
            fragment_file: Path to fragment file
            sample_size: Number of fragments to sample for validation

        Returns:
            Dictionary with validation results:
            - sizes: List of sampled fragment sizes
            - median_size: Median fragment size
            - expected_assay: Detected assay type based on fragment sizes
            - warning: Warning message if sizes don't match expected assay
        """
        if not self.validate_fragment_sizes:
            return {"skipped": True}

        # Expected fragment size ranges by assay type
        expected_ranges = {
            "atac": (50, 600),  # ATAC-seq: nucleosome-free + multi-nucleosome
            "cutandrun": (20, 300),  # CUT&RUN: smaller fragments
            "cutnrun": (20, 300),  # Same as CUT&RUN
            "chip": (100, 500),  # ChIP-seq: variable depending on sonication
        }

        # Override with user-specified range
        if self.expected_fragment_size_range:
            current_assay = self.assay_type if self.assay_type != "auto" else "custom"
            expected_ranges[current_assay] = self.expected_fragment_size_range

        sizes = []
        try:
            opener = gzip.open if str(fragment_file).endswith(".gz") else open
            with opener(fragment_file, "rt") as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    if line.startswith("#"):
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        try:
                            start = int(parts[1])
                            end = int(parts[2])
                            frag_size = end - start
                            if 0 < frag_size < 10000:  # Sanity check
                                sizes.append(frag_size)
                        except ValueError:
                            continue

            if not sizes:
                return {"error": "No valid fragment sizes found"}

            median_size = np.median(sizes)
            sizes_array = np.array(sizes)

            # Detect likely assay type based on fragment sizes
            # CUT&RUN: majority <200bp, ATAC: bimodal with peaks at ~100 and ~200bp
            pct_small = np.sum(sizes_array < 150) / len(sizes_array) * 100
            pct_nucleosome = np.sum((sizes_array >= 150) & (sizes_array < 300)) / len(sizes_array) * 100

            if pct_small > 70:
                detected_assay = "cutandrun"
            elif pct_nucleosome > 40:
                detected_assay = "atac"
            else:
                detected_assay = "unknown"

            result = {
                "sizes": sizes[:100],  # Return first 100 for debugging
                "median_size": float(median_size),
                "pct_small_fragments": float(pct_small),
                "pct_nucleosome": float(pct_nucleosome),
                "detected_assay": detected_assay,
                "warning": None,
            }

            # Check for mismatch
            if self.assay_type not in ("auto", detected_assay):
                result["warning"] = (
                    f"Fragment size distribution suggests {detected_assay} assay, "
                    f"but assay_type is set to '{self.assay_type}'. "
                    f"Median size: {median_size:.0f}bp, {pct_small:.1f}% < 150bp"
                )
                self.logger.warning(result["warning"])

            return result

        except Exception as e:
            self.logger.warning(f"Fragment size validation failed: {e}")
            return {"error": str(e)}

    def _get_valid_chromosomes_filtered(self, chrom_size, config_dict: dict) -> list[str]:
        """Get valid chromosomes excluding spike-in chromosomes.

        Args:
            chrom_size: ChromSize object
            config_dict: Configuration dictionary for chromosome filtering

        Returns:
            List of valid chromosome names (excluding spike-in chromosomes)
        """
        from caesar.preprocess.utils.fragment_utils import get_valid_chromosomes

        valid_chroms = get_valid_chromosomes(chrom_size, config_dict)

        # Filter out spike-in chromosomes
        if self.spike_in_chroms:
            filtered_chroms = [c for c in valid_chroms if c not in self.spike_in_chroms]
            if len(filtered_chroms) < len(valid_chroms):
                n_filtered = len(valid_chroms) - len(filtered_chroms)
                self.logger.info(
                    f"Filtered {n_filtered} spike-in chromosome(s): "
                    f"{[c for c in valid_chroms if c in self.spike_in_chroms]}"
                )
            return filtered_chroms

        return valid_chroms

    def auto_detect_mode(self, fragment_file: Path) -> str:
        """Auto-detect processing mode based on fragment file content.

        Detection algorithm:
        1. Reads first 100 non-comment lines
        2. Checks 4th column for barcode-like patterns
        3. Validates barcode format (supports multiple formats):
           a) With '-' separator (e.g., "TTGTGCGAGTAAGTCC-1")
              - Barcode part (before '-') is >= 10bp and contains only ATCGN characters
              - Handles prefixed barcodes (e.g., "sample1_TTGTGCGAGTAAGTCC-1")
           b) Pure DNA barcodes without suffix (e.g., "TGCACATTACGGCCTCTGCAAT")
              - 16-30bp sequences containing only ATCGN characters
              - Must see multiple unique barcodes to confirm single-cell
        4. Returns 'single-cell' if valid barcodes found, otherwise 'bulk'

        Args:
            fragment_file: Path to fragment file

        Returns:
            'single-cell' or 'bulk'
        """
        # Check first few lines for barcode presence
        try:
            opener = gzip.open if str(fragment_file).endswith(".gz") else open
            unique_barcodes = set()  # Track unique barcodes for pure DNA format
            with opener(fragment_file, "rt") as f:
                for i, line in enumerate(f):
                    if i >= 100:  # Check first 100 lines
                        break

                    if line.startswith("#"):
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        # Check if 4th column looks like a barcode
                        potential_barcode = parts[3]

                        # Format 1: 10x barcodes with '-' suffix (e.g., "TTGTGCGAGTAAGTCC-1")
                        # May have prefixes like "sample1_TTGTGCGAGTAAGTCC-1"
                        if "-" in potential_barcode:
                            # Split by '-' to get barcode part and suffix
                            barcode_with_prefix = potential_barcode.split("-")[0]

                            # Handle prefixed barcodes (e.g., "sample1_TTGTGCGAGTAAGTCC")
                            # Extract the actual barcode part (last segment after underscore)
                            if "_" in barcode_with_prefix:
                                # Take the last segment after the last underscore
                                barcode_part = barcode_with_prefix.split("_")[-1]
                            else:
                                barcode_part = barcode_with_prefix

                            # Validate barcode format: >= 10bp and contains only ATCGN
                            if len(barcode_part) >= 10 and all(
                                c in "ATCGN" for c in barcode_part.upper()
                            ):
                                self.logger.info(
                                    f"Detected barcodes with '-' suffix in {fragment_file}, using single-cell mode"
                                )
                                return "single-cell"
                        else:
                            # Format 2: Pure DNA barcodes without suffix (e.g., "TGCACATTACGGCCTCTGCAAT")
                            # Single-cell barcodes vary: 16-32bp for most technologies
                            # (10X: 16bp, SHARE-seq: 24bp, sci-ATAC: 32bp)
                            barcode_upper = potential_barcode.upper()
                            if (
                                14 <= len(potential_barcode) <= 40
                                and all(c in "ATCGN" for c in barcode_upper)
                            ):
                                unique_barcodes.add(potential_barcode)
                                # If we see multiple unique DNA barcodes, it's likely single-cell
                                if len(unique_barcodes) >= 10:
                                    self.logger.info(
                                        f"Detected {len(unique_barcodes)} unique DNA barcodes in {fragment_file}, using single-cell mode"
                                    )
                                    return "single-cell"

            # Check if we found enough unique barcodes (even if < 10 in first 100 lines)
            if len(unique_barcodes) >= 5:
                self.logger.info(
                    f"Detected {len(unique_barcodes)} unique DNA barcodes in {fragment_file}, using single-cell mode"
                )
                return "single-cell"

            self.logger.info(
                f"No barcodes detected in {fragment_file}, using bulk mode"
            )
            return "bulk"

        except Exception as e:
            self.logger.warning(f"Error detecting mode: {e}, defaulting to bulk")
            return "bulk"

    def determine_assembly(self, fragment_file: Path) -> str:
        """Determine genome assembly for the fragment file.

        Args:
            fragment_file: Path to fragment file

        Returns:
            Assembly name (e.g., 'hg38', 'mm10')
        """
        if self.assembly != "auto":
            return self.assembly

        # Import the actual assembly detection function
        try:
            from caesar.preprocess.utils.determine_assembly import (
                determine_genome_assembly,
            )

            self.logger.info("Auto-detecting genome assembly...")
            assembly, fragment_count = determine_genome_assembly(
                str(fragment_file), root_path=str(fragment_file.parent)
            )
            self.logger.info(
                f"Detected assembly: {assembly} (heterochromatin fragments: {fragment_count})"
            )
            return assembly
        except Exception as e:
            self.logger.warning(f"Assembly detection failed: {e}, defaulting to hg38")
            return "hg38"

    def process(self) -> dict[str, Any]:
        """Process fragment files based on detected or configured mode.

        Returns:
            Processing results dictionary
        """
        self._start_processing()

        # Apply metadata filtering if configured
        if self.metadata_filter_processor:
            self.logger.info("Applying metadata filtering...")
            filter_results = self.metadata_filter_processor.process()
            self.filtered_barcodes = filter_results["barcode_dict"]
            self.logger.info(
                f"Metadata filtering complete: {filter_results['n_cells_filtered']} cells retained"
            )

        results = []

        for i, fragment_file in enumerate(self.fragment_files):
            # Detect processing mode
            if self.processing_type == "auto":
                mode = self.auto_detect_mode(fragment_file)
            else:
                mode = self.processing_type

            # Determine assembly
            assembly = self.determine_assembly(fragment_file)

            # Get sample info
            sample_info = self._extract_sample_info(fragment_file, i)

            # Process based on mode
            if mode == "single-cell":
                # Process with potential dual output support
                if self.dual_output or "dense" in self.output_formats:
                    result = self._process_single_cell_dual(
                        fragment_file, sample_info, assembly
                    )
                else:
                    result = self._process_single_cell(
                        fragment_file, sample_info, assembly
                    )
            else:
                result = self._process_bulk(fragment_file, sample_info, assembly)

            results.append(result)

        runtime = self._end_processing()

        # Return results
        if len(results) == 1:
            # Single output - return it directly with runtime
            result = results[0]
            result["runtime"] = runtime
            return result
        else:
            # Multiple outputs - create MultiSparseZarr if all are single-cell
            if all(r["mode"] == "single-cell" for r in results):
                from caesar.io.zarr_io import MultiSparseZarrIO

                # MultiSparseZarrIO provides a virtual consolidated view of multiple zarr files
                # It doesn't create a physical multi_sparse.zarr directory - it's just an interface
                # The actual data is stored in multi_sparse/{sample_id}.zarr
                sample_paths = [r["output_path"] for r in results]
                sparse_dir = self.output_dir / "multi_sparse"
                multi_sparse_io = MultiSparseZarrIO(
                    str(sparse_dir), mode="r", pattern="*.zarr"
                )

                return {
                    "output_path": str(sparse_dir),  # Return the actual directory containing the zarr files
                    "mode": "single-cell",
                    "assembly": results[0]["assembly"],
                    "n_samples": len(results),
                    "n_cells": sum(r["n_cells"] for r in results),
                    "n_fragments": sum(r["n_fragments"] for r in results),
                    "sample_results": results,
                    "zarr_io": multi_sparse_io,
                    "runtime": runtime,
                }
            else:
                # Mixed or all bulk - return primary output
                return {
                    "output_path": results[0]["output_path"],
                    "fragment_files": [str(f) for f in self.fragment_files],
                    "results": results,
                    "runtime": runtime,
                }

    def _extract_sample_info(self, fragment_file: Path, index: int = None) -> dict[str, Any]:
        """Extract sample information from config or filename.

        Uses unified extract_fragment_info() function for consistent behavior
        with BPCellsProcessor. Matches fragments by path (not index) for robustness.

        Args:
            fragment_file: Path to fragment file
            index: Optional index (kept for backward compatibility, but matching is by path)

        Returns:
            Dictionary with id, celltype, metadata_path, metadata_column
        """
        # Check if verbose logging is enabled (INFO level or lower)
        verbose = logging.getLogger().isEnabledFor(logging.INFO)

        # Use unified extraction function (matches by path, not index)
        frag_info = extract_fragment_info(
            frag_path=fragment_file,
            fragments_config=self.config.data.fragments if self.config.data.fragments else None,
            verbose=verbose,
        )

        # Return with metadata_path key for backward compatibility
        return {
            "id": frag_info["id"],
            "celltype": frag_info["celltype"],
            "metadata_path": frag_info["metadata"],
            "metadata_column": frag_info["metadata_column"],
        }

    def _process_single_cell(
        self, fragment_file: Path, sample_info: dict[str, str], assembly: str
    ) -> dict[str, Any]:
        """Process fragment file in single-cell mode.

        Args:
            fragment_file: Path to fragment file
            sample_info: Sample metadata
            assembly: Genome assembly

        Returns:
            Processing results
        """
        from caesar.preprocess.utils.fragment_utils import (
            create_genome_tiles,
            create_tabix_index,
            get_unique_valid_barcodes,
            initialize_zarr_store,
            process_chromosome,
        )
        from gcell.dna.genome import read_peaks

        # Use configured output name if specified AND only processing a single fragment
        # For multiple fragments, always use multi_sparse/{sample_id}.zarr to avoid conflicts
        num_fragments = len(self.fragment_files)
        has_output_config = hasattr(self.fragment_config, "output") and self.fragment_config.output

        self.logger.info(
            f"Fragment output path decision: num_fragments={num_fragments}, "
            f"has_output_config={has_output_config}, "
            f"output_config_value={getattr(self.fragment_config, 'output', None)}"
        )

        if num_fragments == 1 and has_output_config:
            output_name = self.fragment_config.output
            output_zarr = self.get_output_path(output_name)
            self.logger.info(f"Using single-fragment output path: {output_zarr}")
        else:
            # Always use multi_sparse/{sample_id}.zarr structure for multiple fragments
            # This prevents overwriting when processing multiple fragments
            sparse_dir = self.output_dir / "multi_sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            output_zarr = sparse_dir / f"{sample_info['id']}.zarr"
            self.logger.info(f"Using multi-fragment output path: {output_zarr}")

        self.logger.info(f"Processing single-cell fragments: {fragment_file}")

        # Get CUT&RUN-specific info for this fragment
        cutandrun_info = self._get_fragment_info(fragment_file)

        # Validate fragment size distribution if enabled
        if self.validate_fragment_sizes:
            size_validation = self._validate_fragment_sizes(fragment_file)
            if size_validation.get("warning"):
                self.logger.warning(size_validation["warning"])

        # Create tabix index if needed
        # NOTE: Tabix indexing is required for FragmentProcessor (Zarr backend) because
        # process_chromosome() uses pysam.TabixFile for efficient chromosome-by-chromosome
        # fragment reading. BPCellsProcessor doesn't need tabix as it uses BPCells'
        # native C++ fragment reader (import_10x_fragments) which handles indexing internally.
        create_tabix_index(str(fragment_file))

        # Get valid barcodes - apply metadata filtering if available
        if self.filtered_barcodes:
            # Read all barcodes from fragment file
            all_barcodes = get_unique_valid_barcodes(
                fragment_file,
                min_fragments=0,  # Don't filter here
                max_fragments=float("inf"),
                min_cells=0,
            )
            # Intersect with filtered barcodes
            barcode_dict = {}
            for barcode in all_barcodes:
                if barcode in self.filtered_barcodes:
                    barcode_dict[barcode] = self.filtered_barcodes[barcode]
            self.logger.info(
                f"Applied metadata filter: {len(all_barcodes)} -> {len(barcode_dict)} barcodes"
            )
        else:
            # Standard barcode filtering
            barcode_dict = get_unique_valid_barcodes(
                fragment_file,
                min_fragments=self.fragment_config.min_fragments,
                max_fragments=self.fragment_config.max_fragments,
                min_cells=self.fragment_config.min_cells,
            )
        n_cells = len(barcode_dict)

        # Get chromosome info with spike-in filtering
        chrom_size = ChromSize(assembly)
        valid_chroms = self._get_valid_chromosomes_filtered(chrom_size, self.config.__dict__)

        # Create zarr store
        # Convert overwrite to skip_exist (inverse logic)
        skip_exist = not self.fragment_config.overwrite
        zarr_root = initialize_zarr_store(
            str(output_zarr), skip_exist=skip_exist
        )

        # Store essential attributes
        # Use extract_id_from_path for consistent ID extraction (same as BPCellsProcessor)
        data_key = extract_id_from_path(fragment_file, verbose=False)
        zarr_root.attrs.update(
            {
                "tile_size": self.fragment_config.tile_size,
                "max_fragments_per_cell": self.fragment_config.max_fragments_per_cell,
                "chunk_size": self.fragment_config.chunk_size,
                "assembly": assembly,
                "data_key": data_key,
            }
        )

        # Store barcodes and metadata (including CUT&RUN-specific info)
        self._store_metadata(
            zarr_root,
            barcode_dict,
            sample_info["id"],
            assembly,
            metadata_path=sample_info.get("metadata_path"),
            metadata_column=sample_info.get("metadata_column"),
            cutandrun_info=cutandrun_info,
        )

        # Create tiles if needed
        tiles_df = None
        if self.fragment_config.tile_size:
            tiles_df = create_genome_tiles(chrom_size, self.fragment_config.tile_size)
            self.logger.info(
                f"Created {len(tiles_df)} tiles of size {self.fragment_config.tile_size}bp"
            )

        # Process peaks if provided
        peaks_df = None
        if self.fragment_config.peak_file:
            peaks_df = read_peaks(self.fragment_config.peak_file)

        # Process fragments by chromosome
        total_fragments = 0
        for chrom in valid_chroms:
            self.logger.info(f"Processing chromosome {chrom}")
            process_chromosome(
                chrom=chrom,
                zarr_root=zarr_root,
                fragment_file=str(fragment_file),
                barcode_dict=barcode_dict,
                chrom_length=chrom_size.dict,
                max_fragments_per_cell=self.fragment_config.max_fragments_per_cell,
                chunk_size=self.fragment_config.chunk_size,
                tiles=tiles_df,
                peaks=peaks_df,
                skip_exist=skip_exist,
                n_jobs=max(1, self.config.resources.n_jobs // 4),
            )
            # Estimate fragments (actual count is stored in zarr)
            total_fragments += n_cells * 1000

        return {
            "id": sample_info["id"],
            "celltype": sample_info["celltype"],
            "output_path": str(output_zarr),
            "mode": "single-cell",
            "n_cells": n_cells,
            "n_fragments": total_fragments,
            "assembly": assembly,
        }

    def _process_bulk(
        self, fragment_file: Path, sample_info: dict[str, str], assembly: str
    ) -> dict[str, Any]:
        """Process fragment file in bulk mode.

        Args:
            fragment_file: Path to fragment file
            sample_info: Sample metadata
            assembly: Genome assembly

        Returns:
            Processing results
        """
        from caesar.preprocess.utils.fragment_utils import run_fragment_to_zarr

        # Use consistent output path for all bulk samples
        output_zarr = self.get_output_path("celltype_dense.zarr")

        self.logger.info(f"Processing bulk fragments: {fragment_file}")

        # Create args object for run_fragment_to_zarr
        args = argparse.Namespace(
            id=sample_info["id"],
            celltype=sample_info["celltype"],
            fragment_file=str(fragment_file),
            assembly=assembly,
            output_zarr=str(output_zarr),
            chunk_size=self.fragment_config.chunk_size,
            single_end=False,  # Default for fragment files
            chr_filter_config=self.config.__dict__,  # Pass full config for chromosome filtering
        )

        # Run the fragment to zarr conversion
        run_fragment_to_zarr(args)

        # Get library size from zarr attributes
        zarr_root = zarr.open(str(output_zarr), mode="r")
        # Data is stored under celltype.sample groups in fragment_to_zarr
        group_name = f"{sample_info['celltype']}.{sample_info['id']}"
        if group_name in zarr_root:
            group = zarr_root[group_name]
            library_size = group.attrs.get("library_size", 0)
        else:
            # Fallback to root attributes
            library_size = zarr_root.attrs.get("library_size", 0)

        return {
            "id": sample_info["id"],
            "celltype": sample_info["celltype"],
            "output_path": str(output_zarr),
            "mode": "bulk",
            "n_fragments": library_size,
            "assembly": assembly,
        }

    def _store_metadata(
        self,
        zarr_root,
        barcode_dict: dict[str, int],
        sample_id: str,
        assembly: str,
        metadata_path: str = None,
        metadata_column: str = None,
        cutandrun_info: dict[str, Any] = None,
    ) -> None:
        """Store metadata in zarr store.

        Uses unified metadata handling: if merged_metadata is available, filter by sample_id.
        Otherwise falls back to per-fragment metadata loading.

        Args:
            zarr_root: Zarr root group
            barcode_dict: Dictionary mapping barcodes to indices
            sample_id: Sample identifier
            assembly: Genome assembly
            metadata_path: Path to metadata file
            metadata_column: Column name for cell type labels
            cutandrun_info: CUT&RUN-specific metadata (antibody_target, spike_in_factor, assay_type)
        """
        # Store barcodes
        barcodes = sorted(barcode_dict.keys())
        # Use VariableLengthUTF8 dtype to handle variable-length barcodes
        # This prevents truncation of long barcodes (e.g., with prefixes)
        barcodes_data = np.array(barcodes, dtype=object)
        barcodes_arr = zarr_root.create_array(
            "barcodes",
            shape=barcodes_data.shape,
            dtype=VariableLengthUTF8(),
            chunks=(1000,),
            compressors=self.compressor,
            overwrite=True,
        )
        barcodes_arr[:] = barcodes_data

        # Store labels (cell types)
        # Priority: 1) Merged metadata (if available and metadata_column specified),
        #           2) Per-fragment metadata, 3) MetadataFilterProcessor, 4) Uniform
        # Note: metadata_column is required to use merged_metadata (we need to know which column contains labels)
        if self.merged_metadata is not None and metadata_column:
            # Use merged metadata - filter by sample_id for this fragment
            try:
                # Filter metadata for this fragment's sample_id
                frag_metadata = self.merged_metadata.copy()
                if self.sample_id_col in frag_metadata.columns:
                    frag_metadata = frag_metadata[frag_metadata[self.sample_id_col] == sample_id].copy()
                    self.logger.info(
                        f"Filtered merged metadata for sample_id='{sample_id}': "
                        f"{len(frag_metadata)} rows"
                    )

                # Create barcode to label mapping
                if metadata_column in frag_metadata.columns:
                    barcode_to_label = dict(zip(
                        frag_metadata[self.barcode_col],
                        frag_metadata[metadata_column]
                    ))
                    labels = np.array(
                        [barcode_to_label.get(bc, "unknown") for bc in barcodes],
                        dtype="<U50",
                    )
                    labels = validate_cell_labels(labels, max_length=50)
                    self.logger.info(
                        f"Loaded labels from merged metadata for {sample_id}"
                    )
                else:
                    raise ValueError(
                        f"Column '{metadata_column}' not found in merged metadata. "
                        f"Available columns: {list(frag_metadata.columns)}"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Failed to use merged metadata for {sample_id}: {e}. "
                    f"Falling back to per-fragment metadata."
                )
                # Fall through to per-fragment metadata loading
                if metadata_path and metadata_column:
                    try:
                        labels = load_cell_metadata(
                            metadata_path=metadata_path,
                            barcodes=barcodes,
                            label_column=metadata_column,
                            barcode_column=self.barcode_col,
                            default_label="unknown",
                        )
                        labels = validate_cell_labels(labels, max_length=50)
                        self.logger.info(f"Loaded per-cell metadata from {metadata_path}")
                    except Exception as e2:
                        self.logger.warning(
                            f"Failed to load per-fragment metadata: {e2}. "
                            f"Using uniform celltype assignment."
                        )
                        celltype = self._get_celltype(sample_id)
                        labels = np.full(len(barcodes), celltype, dtype="<U50")
                else:
                    celltype = self._get_celltype(sample_id)
                    labels = np.full(len(barcodes), celltype, dtype="<U50")
        elif metadata_path and metadata_column:
            # Load per-cell metadata from file (per-fragment, backward compatibility)
            try:
                labels = load_cell_metadata(
                    metadata_path=metadata_path,
                    barcodes=barcodes,
                    label_column=metadata_column,
                    barcode_column=self.barcode_col,
                    default_label="unknown",
                )
                # Validate labels
                labels = validate_cell_labels(labels, max_length=50)
                self.logger.info(f"Loaded per-cell metadata from {metadata_path}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load metadata from {metadata_path}: {e}"
                )
                self.logger.warning("Falling back to uniform celltype assignment")
                celltype = self._get_celltype(sample_id)
                labels = np.full(len(barcodes), celltype, dtype="<U50")
        elif (
            self.metadata_filter_processor
            and self.metadata_filter_processor.filtered_metadata is not None
        ):
            # Use metadata from metadata filter processor
            filtered_df = self.metadata_filter_processor.filtered_metadata
            barcode_col = self.metadata_filter_processor._get_barcode_column(
                filtered_df
            )

            # Create a mapping from barcode to label
            label_column = metadata_column or "celltype"
            if label_column in filtered_df.columns:
                barcode_to_label = dict(zip(barcode_col, filtered_df[label_column]))
                labels = np.array(
                    [barcode_to_label.get(bc, "unknown") for bc in barcodes],
                    dtype="<U50",
                )
                self.logger.info(
                    "Using cell type labels from metadata filter processor"
                )
            else:
                celltype = self._get_celltype(sample_id)
                labels = np.full(len(barcodes), celltype, dtype="<U50")
        else:
            # Use uniform celltype for all cells
            celltype = self._get_celltype(sample_id)
            labels = np.full(len(barcodes), celltype, dtype="<U50")

        labels_arr = zarr_root.create_array(
            "labels",
            shape=labels.shape,
            dtype=labels.dtype,
            compressors=self.compressor,
            overwrite=True,
        )
        labels_arr[:] = labels

        # Store metadata
        zarr_root.attrs["sample_id"] = sample_id
        zarr_root.attrs["assembly"] = assembly
        zarr_root.attrs["n_cells"] = len(barcodes)
        zarr_root.attrs["processing_date"] = str(pd.Timestamp.now())

        # Store metadata filtering info if used
        if self.metadata_filter_processor:
            zarr_root.attrs["metadata_filtered"] = True
            if hasattr(self.config.metadata_filter, "dataset"):
                zarr_root.attrs["metadata_filter_dataset"] = (
                    self.config.metadata_filter.dataset
                )

        # Store CUT&RUN-specific metadata
        if cutandrun_info:
            assay_type = cutandrun_info.get("assay_type")
            if assay_type and assay_type != "auto":
                zarr_root.attrs["assay_type"] = assay_type

            antibody_target = cutandrun_info.get("antibody_target")
            if antibody_target:
                zarr_root.attrs["antibody_target"] = antibody_target

            spike_in_factor = cutandrun_info.get("spike_in_factor")
            if spike_in_factor is not None:
                zarr_root.attrs["spike_in_factor"] = float(spike_in_factor)
                # Note: spike-in factor is stored but NOT applied to data
                # Users can apply normalization downstream using this factor
                zarr_root.attrs["spike_in_applied"] = False

            # Store spike-in chromosomes that were filtered
            if self.spike_in_chroms:
                zarr_root.attrs["spike_in_chroms_filtered"] = self.spike_in_chroms

    def _get_celltype(self, sample_id: str) -> str:
        """Get celltype for sample."""
        # Check if celltype specified in config
        if self.config.data.fragments:
            for item in self.config.data.fragments:
                # Handle optional id field - use getattr with None default
                item_id = getattr(item, "id", None)
                if item_id == sample_id:
                    # Handle optional celltype field
                    return getattr(item, "celltype", None) or "unknown"
        return "unknown"

    def get_output_path(self, filename: str) -> Path:
        """Get output path for a given filename."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
