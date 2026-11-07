"""BPCells processor for Caesar preprocessing pipeline.

This processor handles conversion of fragment files to BPCells format,
supporting single-cell, bulk, and multiome workflows with efficient
sparse matrix storage.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from caesar.preprocess.core.base import BaseProcessor
from caesar.io.metadata_io import (
    apply_cluster_mapping,
    load_and_merge_metadata,
)
from gcell.dna.genome import ChromSize

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig

# Conditional import for BPCells
try:
    import bpcells.cpp
    from bpcells.experimental.fragments import (
        build_cell_groups,
        import_10x_fragments,
        precalculate_insertion_counts,
        precalculate_insertion_counts_binned,
    )
    from bpcells.experimental.matrix import DirMatrix

    HAS_BPCELLS = True
except ImportError:
    HAS_BPCELLS = False

# Conditional import for AnnData
try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False


def _process_single_fragment_for_anndata(args):
    """Process a single fragment for AnnData output (used in parallel processing).

    This function must be at module level to be picklable by ProcessPoolExecutor.

    Note:
        This function processes ALL chromosomes together in a single call to
        `precalculate_insertion_counts_binned`. The BPCells backend creates a single
        matrix where columns represent bins across all chromosomes concatenated together.
        Chromosomes are NOT processed separately - they are all included in one matrix
        with shape (n_cells, total_bins_across_all_chromosomes).
    """
    import sys
    import shutil
    from pathlib import Path

    # Import required modules inside function for pickling
    import bpcells.cpp
    from bpcells.experimental.fragments import (
        build_cell_groups,
        precalculate_insertion_counts_binned,
    )

    sys.stdout.flush()
    sys.stderr.flush()

    try:
        frag_path, temp_output_base, chrom_sizes, bin_size, min_fragments, max_fragments, threads = args

        frag_cells = bpcells.cpp.cell_names_fragments_dir(frag_path)
        frag_name = Path(frag_path).name

        # Build cell groups (each cell is its own group)
        # Use actual cell barcodes as group IDs so they're preserved in AnnData obs index
        group_ids_list = list(frag_cells)  # Each cell is its own group, use barcode as group name
        group_order = group_ids_list.copy()

        cell_groups = build_cell_groups(
            fragments=[frag_path],
            cell_ids={frag_path: list(frag_cells)},
            group_ids={frag_path: group_ids_list},
            group_order=group_order,
            min_library_size=min_fragments,
            max_library_size=max_fragments,
        )

        filtered_cell_count = len(cell_groups)

        # Create output directory for this fragment
        frag_output = Path(temp_output_base) / frag_name
        if frag_output.exists():
            shutil.rmtree(frag_output)

        # Run binned precalculation
        # NOTE: This processes ALL chromosomes together in a single call.
        # The resulting matrix has columns representing bins from all chromosomes
        # concatenated together (not processed separately per chromosome).
        mat = precalculate_insertion_counts_binned(
            fragments=[frag_path],
            output_dir=str(frag_output),
            cell_groups=cell_groups,
            chrom_sizes=chrom_sizes,  # All chromosomes processed together
            bin_size=bin_size,
            threads=threads,
            group_names=None,  # Auto-extract from cell_groups
            preserve_chrom_order=True,  # Ensure consistent chromosome ordering for concatenation
        )

        return str(frag_output), filtered_cell_count, None
    except Exception as e:
        import traceback
        return None, 0, f"{str(e)}\n{traceback.format_exc()}"


class BPCellsProcessor(BaseProcessor):
    """Process fragment files into BPCells format.

    This processor converts fragment files into efficient BPCells storage format,
    supporting various output types:
    - SingleSparse: Per-cell sparse matrices (all fragments combined)
    - MultiSparse: Separate single_sparse outputs per fragment (for MultiSparseBPCellsIO)
    - CelltypeDense: Celltype-aggregated dense arrays
    - SignalDense: Pseudobulk dense signal

    Key features:
    - Efficient C++ backend for fast processing
    - Memory-mapped storage for large datasets
    - Support for normalization factors (stored, not applied)
    - Compatible with Caesar's BPCells IO classes

    IMPORTANT - Backend Compatibility:
    BPCells format is now FULLY integrated with Caesar's core preprocessing pipeline:

    SUPPORTED downstream operations:
    - ✅ Peak calling via PeakCallingProcessor (celltype_dense or signal_dense)
    - ✅ Library size annotation via LibrarySizeProcessor (celltype_dense or signal_dense)
    - ✅ Clustering via ClusteringProcessor (anndata output)
    - ✅ Reading/querying via BPCells IO classes
    - ✅ Footprinting via IO class methods (celltype_dense/signal_dense)
      Example: bpcells_io.get_genome_wide_footprint_all_motifs(motif_io)

    All core processors auto-detect BPCells vs Zarr stores and route accordingly.

    NOT YET SUPPORTED with BPCells output:

    - PseudoBulkProcessor (use celltype_dense output instead)
      Reason: BPCells already provides celltype_dense output which is pseudobulk data.
      Use that output type directly instead of running a separate pseudobulk step.

    - NormalizationProcessor (BPCells stores factors only)
      Reason: BPCells maintains raw counts for efficiency and only stores normalization
      factors (if provided) without applying them. The processor's in-place normalization
      approach conflicts with BPCells' raw count preservation model.

    For full preprocessing pipelines:
    - Use `caesar workflow` with `output.backend: "bpcells"` for automatic routing
    - See workflows/bpcells-single-cell-complete.yaml for a complete example
    - See docs/bpcells_backend_support.md for details
    """

    @classmethod
    def get_command_name(cls) -> str:
        """Override to return 'bpcells' instead of 'b-p-cells'."""
        return "bpcells"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add BPCells-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Input arguments
        parser.add_argument(
            "-i", "--input", nargs="+", required=True,
            help="Input fragment files (10x format) or BPCells fragment directories"
        )
        parser.add_argument(
            "--input-format", choices=["auto", "10x", "bpcells"], default="auto",
            help="Input format: '10x' for fragment files, 'bpcells' for BPCells directories, "
                 "'auto' to detect automatically (default: auto)"
        )

        # Output arguments
        parser.add_argument(
            "-o", "--output-dir", default="./output",
            help="Output directory (default: ./output)"
        )
        parser.add_argument(
            "--output-type", choices=["single_sparse", "multi_sparse", "celltype_dense", "signal_dense", "anndata"],
            default="single_sparse",
            help=(
                "BPCells output type:\n"
                "  single_sparse: Per-cell sparse matrices, all fragments combined (default)\n"
                "  multi_sparse: Separate single_sparse per fragment (for MultiSparseBPCellsIO)\n"
                "  celltype_dense: Celltype-aggregated pseudobulk (requires metadata)\n"
                "  signal_dense: Single pseudobulk signal (metadata optional)\n"
                "  anndata: Binned genome-wide AnnData object (cells x bins) with annotations"
            )
        )

        # Metadata arguments
        parser.add_argument(
            "--metadata", nargs="+",
            help="Metadata CSV files (one per input, or single file for all inputs). "
                 "Required for celltype_dense output."
        )
        parser.add_argument(
            "--barcode-col", default="barcode",
            help="Column name for cell barcodes in metadata (default: barcode)"
        )
        parser.add_argument(
            "--annotation-col", default="celltype",
            help="Column name for cell annotations in metadata (default: celltype). "
                 "Used for grouping in celltype_dense output."
        )
        parser.add_argument(
            "--sample-id-col", default="sample_id",
            help="Column name for sample IDs in metadata (default: sample_id). "
                 "Required when using single metadata file with non-unique barcodes across fragments. "
                 "Values should match fragment file prefixes or fragment IDs."
        )

        # Cluster mapping and filtering
        parser.add_argument(
            "--cluster-mapping", metavar="FILE",
            help="JSON file mapping fine-grained clusters to broad groups. "
                 "Format: {\"cluster1\": \"group1\", \"cluster2\": \"group1\", ...}"
        )
        parser.add_argument(
            "--target-groups", nargs="+",
            help="Filter to only process specific cell types/groups. "
                 "Only cells in these groups will be included."
        )

        # Processing arguments
        parser.add_argument(
            "--assembly", choices=["auto", "hg38", "hg19", "mm10", "mm9"], default="auto",
            help="Genome assembly (default: auto-detect)"
        )
        parser.add_argument(
            "--min-fragments", type=int, default=1000,
            help="Minimum fragments per cell to include (default: 1000). "
                 "Cells with fewer fragments will be filtered out."
        )
        parser.add_argument(
            "--max-fragments", type=int, default=50000,
            help="Maximum fragments per cell to include (default: 50000). "
                 "Cells with more fragments will be filtered out."
        )
        parser.add_argument(
            "--bin-size", type=int, default=500,
            help="Bin size in base pairs for binned matrices (default: 500). "
                 "Used for anndata output type."
        )
        parser.add_argument(
            "--threads", type=int,
            help="Number of threads for parallel processing (default: from config)"
        )
        parser.add_argument(
            "--group-by", "--groupby", dest="group_by", default="celltype",
            help="Column name for grouping cells (default: celltype). "
                 "Used for celltype_dense output."
        )
        parser.add_argument(
            "--overwrite", action="store_true",
            help="Overwrite existing output directory"
        )
        parser.add_argument(
            "--cleanup-temp-dirs", action="store_true",
            help="Remove temporary BPCells fragment directories after processing (default: keep them for reuse)"
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return (
            "Process fragment files or BPCells directories into efficient BPCells format.\n\n"
            "Output types:\n"
            "  • single_sparse: Per-cell sparse matrices (default)\n"
            "  • celltype_dense: Celltype-aggregated pseudobulk (requires metadata)\n"
            "  • signal_dense: Single pseudobulk signal (metadata optional)\n"
            "  • anndata: Binned genome-wide AnnData object (cells x bins)\n\n"
            "Supports both 10x fragment files and pre-converted BPCells directories.\n"
            "Use celltype_dense output to create pseudobulk from metadata annotations.\n"
            "Use anndata output for binned genome-wide coverage matrices."
        )

    def __init__(self, config: CaesarConfig):
        """Initialize BPCells processor.

        Args:
            config: Full Caesar configuration
        """
        super().__init__(config)

        if not HAS_BPCELLS:
            raise ImportError(
                "BPCells library not found. Please install with: pip install bpcells"
            )

        # Extract BPCells-specific config
        # The workflow engine now selects the appropriate config based on step name
        # and passes it via _bpcells_config_override attribute
        # This makes the workflow explicit and removes complex selection logic

        # Check if workflow engine provided a config override
        if hasattr(config, "_bpcells_config_override") and config._bpcells_config_override is not None:
            self.bpcells_config = config._bpcells_config_override
            config_source = "workflow_selected"
            self.logger.info(f"Using workflow-selected BPCells config (output_type={self.bpcells_config.output_type})")
        else:
            # Fallback: try to auto-detect (for backward compatibility and CLI usage)
            # Priority: bpcells_anndata > bpcells (legacy) > defaults
            bpcells_anndata_config = getattr(config, "bpcells_anndata", None)
            bpcells_config = getattr(config, "bpcells", None)

            if bpcells_anndata_config is not None:
                self.bpcells_config = bpcells_anndata_config
                config_source = "bpcells_anndata"
            elif bpcells_config is not None:
                self.bpcells_config = bpcells_config
                config_source = "bpcells"
            else:
                from caesar.preprocess.core.config import BPCellsConfig
                self.bpcells_config = BPCellsConfig()
                config_source = "default"

            self.logger.info(f"Using config section: {config_source} (auto-detected)")

        # Set processor parameters
        self.output_type = self.bpcells_config.output_type
        self.shift_end = self.bpcells_config.shift_end
        self.threads = self.bpcells_config.threads or config.resources.n_jobs
        self.min_fragments = self.bpcells_config.min_fragments
        self.max_fragments = self.bpcells_config.max_fragments
        self.bin_size = self.bpcells_config.bin_size
        self.group_by = self.bpcells_config.group_by
        self.save_metadata = self.bpcells_config.save_metadata
        self.normalization_factor_file = self.bpcells_config.normalization_factor_file
        # Metadata column flexibility
        self.barcode_col = self.bpcells_config.barcode_col
        # When using clustering labels, override annotation_col to use "labels" column
        if self.bpcells_config.use_clustering_labels:
            self.annotation_col = "labels"  # Clustering output uses "labels" column
        else:
            self.annotation_col = self.bpcells_config.annotation_col
        self.sample_id_col = self.bpcells_config.sample_id_col
        # Cluster mapping and filtering
        self.cluster_mapping_file = self.bpcells_config.cluster_mapping_file
        self.target_groups = self.bpcells_config.target_groups
        self.input_format = self.bpcells_config.input_format
        self.cleanup_temp_dirs = self.bpcells_config.cleanup_temp_dirs

        # Determine output path
        self.output_dir = Path(config.output.dir)
        if self.output_type == "anndata":
            # For anndata, output is a directory containing .h5ad file and attrs.json
            self.output_path = self.output_dir / "anndata"
        elif self.output_type == "fragments":
            # For fragments, output is the bpcells_fragments directory
            self.output_path = self.output_dir / "bpcells_fragments"
        else:
            self.output_path = self.output_dir / f"{self.output_type}_bpcells"

        # Cache resolved assembly (resolved from "auto" if needed)
        self._resolved_assembly = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_input_format(self, path: str) -> str:
        """Get input format for a path, respecting auto-detection setting.

        Args:
            path: Path to check

        Returns:
            Detected or configured input format
        """
        if self.input_format == "auto":
            return self._detect_input_format(path)
        return self.input_format

    def _detect_input_format(self, path: str) -> str:
        """Detect if input is 10x fragment file or BPCells directory.

        Args:
            path: Path to check

        Returns:
            "10x" if fragment file, "bpcells" if BPCells directory, "unknown" otherwise
        """
        path_obj = Path(path)

        # Check if it's a file (10x fragment file)
        if path_obj.is_file():
            # Check extension - use suffixes to handle .tsv.gz properly
            # path_obj.suffix only returns last suffix (.gz), so check suffixes or name
            suffixes = path_obj.suffixes  # Returns ['.tsv', '.gz'] for .tsv.gz
            full_suffix = "".join(suffixes)  # Join to get '.tsv.gz'
            if full_suffix in [".tsv", ".tsv.gz", ".bed", ".bed.gz"] or path_obj.suffix in [".tsv", ".bed"]:
                return "10x"

        # Check if it's a directory (BPCells format)
        if path_obj.is_dir():
            # Check for BPCells fragments marker files (single sample)
            frag_marker_files = ["cell_names.txt", "cell_counts.bin", "chr_names.txt", "cell_names"]
            if any((path_obj / marker).exists() for marker in frag_marker_files):
                return "bpcells"

            # Check for BPCells PrecalculatedInsertionMatrix format (single/multi sparse)
            matrix_marker_files = ["val_data", "version", "idxptr_data"]
            if all((path_obj / marker).exists() for marker in matrix_marker_files):
                return "bpcells"

            # Check for multi-sample BPCells fragments directory
            # This is a directory containing subdirectories, each with BPCells fragments
            subdirs = [d for d in path_obj.iterdir() if d.is_dir()]
            if subdirs:
                # Check if any subdirectory has BPCells fragments markers
                for subdir in subdirs[:5]:  # Check first 5 subdirs
                    if (subdir / "version").exists() and (subdir / "chr_names").exists():
                        return "bpcells"

        return "unknown"

    def _resolve_assembly(self) -> str:
        """Resolve assembly from 'auto' to actual assembly name.

        If assembly is 'auto', detects the actual assembly from fragment files.
        Otherwise returns the configured assembly.

        Returns:
            Resolved assembly name (e.g., 'hg38', 'mm10')
        """
        if self._resolved_assembly is not None:
            return self._resolved_assembly

        assembly = self.config.assembly
        if assembly != "auto":
            self._resolved_assembly = assembly
            return assembly

        # Auto-detect assembly from first fragment file
        fragment_paths = self._get_fragment_paths()
        if not fragment_paths:
            self.logger.warning("Cannot determine assembly: no fragment files specified, defaulting to hg38")
            self._resolved_assembly = "hg38"
            return "hg38"

        # Use the first fragment file for detection
        first_fragment = fragment_paths[0]

        # Import the actual assembly detection function
        try:
            from caesar.preprocess.utils.determine_assembly import (
                determine_genome_assembly,
            )

            self.logger.info("Auto-detecting genome assembly...")
            # Determine root path (parent directory of fragment file)
            fragment_path_obj = Path(first_fragment)
            root_path = str(fragment_path_obj.parent.parent) if fragment_path_obj.parent.name == "raw" else str(fragment_path_obj.parent)

            assembly, fragment_count = determine_genome_assembly(
                str(first_fragment), root_path=root_path
            )
            self.logger.info(
                f"Detected assembly: {assembly} (heterochromatin fragments: {fragment_count})"
            )
            self._resolved_assembly = assembly
            return assembly
        except Exception as e:
            self.logger.warning(f"Assembly detection failed: {e}, defaulting to hg38")
            self._resolved_assembly = "hg38"
            return "hg38"

    def _get_fragment_paths(self) -> list[str]:
        """Get fragment paths, handling both 10x files and BPCells directories.

        Returns:
            List of fragment file paths or BPCells directory paths
        """
        fragment_paths = []

        if not self.config.data.fragments:
            self.logger.error("No fragments configured in config.data.fragments")
            return fragment_paths

        for frag in self.config.data.fragments:
            frag_path = frag.path
            if not frag_path or not str(frag_path).strip():
                self.logger.error(f"Empty fragment path found in config: {frag}")
                raise ValueError(f"Empty fragment path in config.data.fragments: {frag}")

            # Auto-detect format if needed
            if self.input_format == "auto":
                detected_format = self._detect_input_format(frag_path)
                if detected_format == "unknown":
                    self.logger.warning(
                        f"Could not detect format for {frag_path}, assuming 10x fragment file"
                    )
                    detected_format = "10x"
            else:
                detected_format = self.input_format

            if detected_format == "bpcells":
                # BPCells directory - use directly
                if not Path(frag_path).exists():
                    raise FileNotFoundError(f"BPCells directory not found: {frag_path}")
                fragment_paths.append(frag_path)
                self.logger.info(f"Using BPCells directory: {frag_path}")
            else:
                # 10x fragment file - will need conversion
                if not Path(frag_path).exists():
                    raise FileNotFoundError(f"Fragment file not found: {frag_path}")
                fragment_paths.append(frag_path)

        return fragment_paths

    def validate_config(self) -> None:
        """Validate processor configuration."""
        # Check output type
        valid_types = ["single_sparse", "multi_sparse", "celltype_dense", "signal_dense", "anndata", "fragments"]
        if self.output_type not in valid_types:
            raise ValueError(
                f"Invalid output_type: {self.output_type}. Must be one of {valid_types}"
            )

        # Check metadata requirements for celltype_dense
        if self.output_type == "celltype_dense" and self.group_by == "celltype":
            # This will be validated more thoroughly in validate_inputs
            pass

        # Validate numeric parameters
        if self.min_fragments < 0:
            raise ValueError("min_fragments must be non-negative")

        if self.max_fragments <= self.min_fragments:
            raise ValueError(f"max_fragments ({self.max_fragments}) must be greater than min_fragments ({self.min_fragments})")

        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be at least 1")

    def validate_inputs(self) -> None:
        """Validate input configuration with detailed error messages."""
        if not self.config.data.fragments:
            raise ValueError(
                "No fragment files specified. Please provide at least one fragment file "
                "or BPCells directory in config.data.fragments"
            )

        # Check fragment files/directories exist and validate format
        fragment_paths = []
        for frag in self.config.data.fragments:
            frag_path = frag.path
            if not Path(frag_path).exists():
                raise FileNotFoundError(
                    f"Fragment file/directory not found: {frag_path}. "
                    f"Please check the path is correct."
                )

            # Validate format if not auto
            if self.input_format != "auto":
                detected = self._detect_input_format(frag_path)
                if detected != self.input_format and detected != "unknown":
                    self.logger.warning(
                        f"Input format mismatch: specified '{self.input_format}' but "
                        f"detected '{detected}' for {frag_path}"
                    )

            fragment_paths.append(frag_path)

        # Validate output type
        valid_types = ["single_sparse", "multi_sparse", "celltype_dense", "signal_dense", "anndata", "fragments"]
        if self.output_type not in valid_types:
            raise ValueError(
                f"Invalid output_type: '{self.output_type}'. "
                f"Must be one of: {valid_types}"
            )

        # Check metadata requirements with clear messages
        if self.output_type == "celltype_dense":
            # Check if using clustering labels (which provides metadata)
            if self.bpcells_config.use_clustering_labels:
                # Metadata will come from clustering step output
                # Check for both anndata.h5ad (preferred) and metadata.csv (fallback)
                anndata_path = self.output_dir / "anndata" / "anndata.h5ad"
                clustering_metadata_path = self.output_dir / "anndata" / "metadata.csv"
                if not anndata_path.exists() and not clustering_metadata_path.exists():
                    raise ValueError(
                        f"CelltypeDense output with use_clustering_labels=True requires "
                        f"clustering output (either anndata.h5ad or metadata.csv) at:\n"
                        f"  - {anndata_path} (preferred)\n"
                        f"  - {clustering_metadata_path} (fallback)\n"
                        f"  - Make sure clustering step has run before pseudobulk step\n"
                        f"  - Clustering step should output anndata.h5ad or metadata.csv in anndata/ directory"
                    )
            else:
                # Need metadata from fragment files for celltype grouping
                # OR metacell_assignments.csv when using iso-libsize metacells
                has_metadata = any(getattr(frag, "metadata", None) for frag in self.config.data.fragments)

                # Check for metacell assignments when annotation_col is "metacell_id"
                has_metacell_assignments = False
                if self.annotation_col == "metacell_id":
                    # Check if any fragment path has metacell_assignments.csv
                    for frag in self.config.data.fragments:
                        frag_path = Path(frag.path)
                        if (frag_path / "metacell_assignments.csv").exists():
                            has_metacell_assignments = True
                            break
                    # Also check output directory
                    if not has_metacell_assignments:
                        if (self.output_dir / "metacell_assignments.csv").exists():
                            has_metacell_assignments = True

                if not has_metadata and not has_metacell_assignments:
                    raise ValueError(
                        f"CelltypeDense output requires metadata with '{self.annotation_col}' "
                        f"column for grouping.\n"
                        f"  - Metadata is REQUIRED for celltype_dense output\n"
                        f"  - Metadata is OPTIONAL for single_sparse and signal_dense outputs\n"
                        f"  - Please specify metadata files in config.data.fragments[].metadata\n"
                        f"  - OR set use_clustering_labels=true to use clustering output metadata\n"
                        f"  - OR run isolibsize-metacell first to generate metacell_assignments.csv"
                    )
        # For single_sparse, multi_sparse, and signal_dense, metadata is optional
        # They can work without metadata - will just use all cells from fragments

            # Validate metadata if provided
            metadata = self._load_and_merge_metadata()
            if metadata is not None:
                if self.barcode_col not in metadata.columns:
                    raise ValueError(
                        f"Barcode column '{self.barcode_col}' not found in metadata. "
                        f"Available columns: {list(metadata.columns)}"
                    )
                if self.annotation_col not in metadata.columns and self.group_by == "celltype":
                    raise ValueError(
                        f"Annotation column '{self.annotation_col}' not found in metadata. "
                        f"Available columns: {list(metadata.columns)}. "
                        f"This column is required for celltype grouping."
                    )

    def process(self) -> dict[str, Any]:
        """Process fragment files to BPCells format.

        Returns:
            Processing results with output paths
        """
        self.validate_inputs()

        # Skip if output exists and not overwriting
        if self.output_path.exists() and not self.bpcells_config.overwrite:
            self.logger.info(f"Output exists at {self.output_path}, skipping")
            # For anndata output, return the .h5ad file path, not the directory
            if self.output_type == "anndata":
                output_path_to_return = str(self.output_path / "anndata.h5ad")
                return {
                    "output_path": output_path_to_return,
                    "output_type": self.output_type,
                    "skipped": True,
                }
            elif self.output_type == "fragments":
                # For fragments, check if all fragment directories exist
                fragment_paths = self._get_fragment_paths()
                all_exist = True
                imported_paths = []
                missing_samples = []
                for frag_path in fragment_paths:
                    sample_id = self._get_frag_id_from_config(frag_path)
                    frag_bpcells_dir = self.output_dir / "bpcells_fragments" / sample_id
                    self.logger.debug(f"Checking fragment directory: {frag_bpcells_dir} (exists: {frag_bpcells_dir.exists()})")
                    if frag_bpcells_dir.exists():
                        is_valid = self._is_valid_bpcells_dir(frag_bpcells_dir)
                        self.logger.debug(f"  Valid BPCells directory: {is_valid}")
                        if is_valid:
                            imported_paths.append(str(frag_bpcells_dir))
                        else:
                            all_exist = False
                            missing_samples.append((sample_id, str(frag_path), "invalid_directory"))
                    else:
                        all_exist = False
                        missing_samples.append((sample_id, str(frag_path), "missing_directory"))
                if all_exist:
                    return {
                        "output_path": str(self.output_path),
                        "output_type": "fragments",
                        "imported_fragments": imported_paths,
                        "n_fragments": len(imported_paths),
                        "skipped": True,
                    }
                # If not all fragments exist, warn and quit
                missing_info = ", ".join([f"{sid} (from {path}, reason: {reason})" for sid, path, reason in missing_samples])
                self.logger.error(
                    f"Output directory exists but fragment directories are missing or invalid. "
                    f"Missing/invalid samples: {missing_info}. "
                    f"Please remove the output directory '{self.output_path}' or set overwrite=True to reprocess."
                )
                raise RuntimeError(
                    f"Cannot proceed: fragment directories are missing or invalid but output directory exists. "
                    f"Missing/invalid samples: {missing_info}. "
                    f"Output directory: {self.output_path}"
                )
            else:
                output_path_to_return = str(self.output_path)
                return {
                    "output_path": output_path_to_return,
                    "output_type": self.output_type,
                    "skipped": True,
                }

        # Clean up existing output if overwriting
        if self.output_path.exists():
            self.logger.info(f"Removing existing output at {self.output_path}")
            shutil.rmtree(self.output_path)

        # Ensure parent directory exists, but not the output directory itself
        # BPCells expects the output directory to not exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Process based on output type
        if self.output_type == "single_sparse":
            result = self._process_single_sparse()
        elif self.output_type == "multi_sparse":
            result = self._process_multi_sparse()
        elif self.output_type == "celltype_dense":
            result = self._process_celltype_dense()
        elif self.output_type == "signal_dense":
            result = self._process_signal_dense()
        elif self.output_type == "anndata":
            result = self._process_anndata()
        elif self.output_type == "fragments":
            result = self._process_fragments()
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")

        # Save attributes
        self._save_attributes()

        # Handle normalization factors if specified
        if self.normalization_factor_file:
            self._save_normalization_factors()

        return result

    def _process_single_sparse(self) -> dict[str, Any]:
        """Process fragments to SingleSparse BPCells format.

        Combines all fragments into one single_sparse output with all cells.
        """
        self.logger.info("Processing fragments to SingleSparse BPCells format")

        # Get fragment paths (handles both 10x files and BPCells directories)
        fragment_paths = self._get_fragment_paths()

        self.logger.info(f"Processing {len(fragment_paths)} fragment file(s) - will combine into single output")

        # Determine input format and convert/collect BPCells directories
        fragments_for_build = []
        temp_dirs_to_cleanup = []

        for frag_idx, frag_path in enumerate(fragment_paths):
            frag_format = self._get_input_format(frag_path)

            # Get sample_id from config or path (always returns something)
            sample_id = self._get_frag_id_from_config(frag_path)

            # Get BPCells directory (use existing or convert)
            temp_bpcells = self._get_bpcells_dir_or_convert(
                frag_path=frag_path,
                category="fragments",
                identifier=sample_id,
                check_existing=True,
                existing_category="fragments",
                log_message=f"Converting 10x fragment file {frag_idx + 1}/{len(fragment_paths)} to BPCells format: {frag_path}",
            )
            fragments_for_build.append(str(temp_bpcells))
            # Track for cleanup if we converted it (not if it was already existing)
            if self._get_input_format(frag_path) != "bpcells":
                temp_dirs_to_cleanup.append(temp_bpcells)

        # Collect all cell barcodes from all fragments
        # Use dict-based API for multiple fragments to ensure correct cell-to-fragment mapping
        all_cells = []
        prefixed_to_original = {}  # Map: prefixed_barcode -> (original_barcode, sample_id) for metadata

        # Per-fragment data structures for dict-based API (when multiple fragments)
        frag_cell_ids = {}  # frag_path_normalized -> list of original cell IDs
        frag_group_ids = {}  # frag_path_normalized -> list of group IDs (prefixed or original)
        cell_to_frag = {}  # original_cell -> frag_path_normalized (for tracking which fragment each cell belongs to)

        # Normalize all fragment paths once at the start
        fragments_normalized = [str(os.path.abspath(os.path.expanduser(f))) for f in fragments_for_build]
        frag_path_to_normalized = dict(zip(fragments_for_build, fragments_normalized))

        # Try to get sample_id information from metadata or fragment configs
        all_metadata = self._load_and_merge_metadata()
        has_sample_id_in_meta = all_metadata is not None and self.sample_id_col in all_metadata.columns if all_metadata is not None else False

        for frag_idx, frag_path in enumerate(fragments_for_build):
            frag_cells = bpcells.cpp.cell_names_fragments_dir(frag_path)
            frag_path_normalized = frag_path_to_normalized[frag_path]

            # Determine sample_id for this fragment
            sample_id = None
            if has_sample_id_in_meta:
                sample_id = self._get_frag_id_from_config(frag_path)

            # Collect cells per fragment for dict-based API
            frag_original_cells = []
            frag_prefixed_cells = []

            for cell in frag_cells:
                frag_original_cells.append(cell)  # Original barcode (matches fragment cell names)
                cell_to_frag[cell] = frag_path_normalized  # Track which fragment this cell belongs to

                if sample_id:
                    # Use prefixed barcode for group_ids to avoid collisions in final output
                    prefixed_barcode = f"{sample_id}_{cell}"
                    frag_prefixed_cells.append(prefixed_barcode)
                    all_cells.append(prefixed_barcode)
                    prefixed_to_original[prefixed_barcode] = (cell, sample_id)
                else:
                    # No sample_id available - use original barcode
                    frag_prefixed_cells.append(cell)
                    all_cells.append(cell)

            # Store per-fragment data
            frag_cell_ids[frag_path_normalized] = frag_original_cells
            frag_group_ids[frag_path_normalized] = frag_prefixed_cells

            self.logger.info(
                f"Found {len(frag_cells)} cells in {Path(frag_path).name}"
                + (f" (sample_id: {sample_id})" if sample_id else "")
            )

        # For build_cell_groups: cell_ids must match fragment cell names (original barcodes)
        # group_ids can be prefixed barcodes (unique labels)
        # When using multiple fragments, use dict-based API for correct mapping
        # Build original_cells_for_build (needed for both single and multiple fragments)
        original_cells_for_build = []
        for cell in all_cells:
            if cell in prefixed_to_original:
                original_cells_for_build.append(prefixed_to_original[cell][0])
            else:
                original_cells_for_build.append(cell)

        if len(all_cells) == 0:
            raise ValueError("No cells found in fragment files.")

        self.logger.info(f"Processing {len(all_cells)} cells from {len(fragments_for_build)} fragment(s)")

        # Build cell groups (each cell is its own group)
        # For single_sparse: each cell is its own group, so group_ids = unique labels for each cell
        group_ids = all_cells  # Prefixed barcodes (if available) or original barcodes
        group_order = sorted(set(all_cells))

        if len(fragments_for_build) > 1:
            # Multiple fragments: use dict-based API for correct cell-to-fragment mapping
            # Build dicts per fragment by grouping cells
            frag_cell_ids_dict = defaultdict(list)
            frag_group_ids_dict = defaultdict(list)

            # Group cells by fragment (using cell_to_frag from collection phase)
            for orig_cell, group_cell in zip(original_cells_for_build, group_ids):
                if orig_cell in cell_to_frag:
                    frag_path_norm = cell_to_frag[orig_cell]
                    frag_cell_ids_dict[frag_path_norm].append(orig_cell)
                    frag_group_ids_dict[frag_path_norm].append(group_cell)

            # Convert to regular dicts
            frag_cell_ids_dict = dict(frag_cell_ids_dict)
            frag_group_ids_dict = dict(frag_group_ids_dict)

            cell_groups = build_cell_groups(
                fragments_normalized,
                frag_cell_ids_dict,
                frag_group_ids_dict,
                group_order,
                min_library_size=self.min_fragments,
                max_library_size=self.max_fragments,
            )
        else:
            # Single fragment: use list-based API (backward compatible)
            cell_groups = build_cell_groups(
                fragments_for_build[0],  # Single path
                original_cells_for_build,  # Original barcodes (must match fragment cell names)
                group_ids,  # Group labels: prefixed barcodes (unique) or original barcodes
                group_order,  # Order of unique groups
                min_library_size=self.min_fragments,
                max_library_size=self.max_fragments,
            )

        # Get chromosome sizes (filtered by chromosomes in fragments, use first fragment)
        chrom_sizes = self._get_chrom_sizes(fragment_path=fragments_for_build[0])

        # Get filtered cells directly from cell_groups (before precalculate_insertion_counts)
        # For single_sparse, each cell is its own group, so the categorical values are the filtered cell barcodes
        filtered_cells = list(cell_groups)  # Convert categorical to list to get filtered cell barcodes
        n_filtered = len(filtered_cells)
        n_total = len(all_cells)
        if n_filtered < n_total:
            self.logger.info(
                f"Library size filtering: {n_total} -> {n_filtered} cells "
                f"({n_total - n_filtered} cells filtered out)"
            )

        # Filter chromosome sizes for pseudobulk (removes patches, chrM, chrY, etc.)
        chrom_sizes_filtered = self._filter_chrom_sizes_for_pseudobulk(chrom_sizes)

        # Precalculate insertion counts
        # IMPORTANT: When cell_groups is pd.Categorical, group_names is ignored and
        # cell_groups.categories is used instead. So we don't need to pass group_names.
        # The categories are already set to group_order (sorted unique prefixed barcodes).
        self.logger.info("Precalculating insertion counts for single cells (combining all fragments)")
        precalculate_insertion_counts(
            fragments=fragments_for_build,  # List of fragment paths
            output_dir=str(self.output_path),
            cell_groups=cell_groups,  # pd.Categorical with prefixed barcodes as categories
            chrom_sizes=chrom_sizes_filtered,
            threads=self.threads,
            # group_names is ignored when cell_groups is pd.Categorical
            # cell_groups.categories (group_order) will be used automatically
        )

        # Clean up temp directories if we created them (only if cleanup is enabled)
        if self.cleanup_temp_dirs:
            for temp_dir in temp_dirs_to_cleanup:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"Cleaned up temp directory: {temp_dir}")
        else:
            self.logger.info(
                f"Keeping {len(temp_dirs_to_cleanup)} temporary BPCells fragment directory(ies) for reuse. "
                f"Use --cleanup-temp-dirs to remove them."
            )

        # Library sizes are already saved by precalculate_insertion_counts in library_size.json
        # The C++ function creates a properly filtered library_size.json file
        library_size_path = self.output_path / "library_size.json"
        if library_size_path.exists():
            self.logger.debug("Library sizes already saved by precalculate_insertion_counts")
        else:
            self.logger.debug("No library_size.json found (library sizes may not be available)")

        # Save metadata if requested
        if self.save_metadata:
            # Store mapping for metadata saving
            self._prefixed_to_original = prefixed_to_original
            # Use filtered_cells instead of all_cells to ensure metadata matches the matrix
            self._save_single_sparse_metadata(filtered_cells)

        return {
            "output_path": str(self.output_path),
            "output_type": "single_sparse",
            "n_cells": len(filtered_cells),
            "assembly": self._resolve_assembly(),
        }

    def _process_multi_sparse(self) -> dict[str, Any]:
        """Process fragments to MultiSparse BPCells format.

        Creates separate single_sparse outputs for each fragment file,
        organized in a parent directory structure compatible with MultiSparseBPCellsIO.
        """
        self.logger.info("Processing fragments to MultiSparse BPCells format")

        # Get fragment paths (handles both 10x files and BPCells directories)
        fragment_paths = self._get_fragment_paths()

        if len(fragment_paths) == 1:
            self.logger.warning(
                "Only one fragment file provided for multi_sparse output. "
                "Consider using single_sparse instead, or provide multiple fragments."
            )

        # Ensure parent output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        sample_results = []
        total_cells = 0

        # Process each fragment separately
        for frag_idx, frag_path in enumerate(fragment_paths):
            # Get sample ID from config or path (always returns something)
            sample_id = self._get_frag_id_from_config(frag_path)

            # Create subdirectory for this sample (MultiSparseBPCellsIO expects pattern: single_sparse_*)
            sample_output = self.output_path / f"single_sparse_{sample_id}"

            self.logger.info(f"Processing fragment {frag_idx + 1}/{len(fragment_paths)}: {frag_path} -> {sample_output}")

            # Get BPCells directory (use existing or convert)
            # Reuse bpcells_fragments if available to avoid duplicate conversions
            temp_bpcells = self._get_bpcells_dir_or_convert(
                frag_path=frag_path,
                category="fragments",
                identifier=sample_id,
                check_existing=True,
                existing_category="fragments",
            )

            # Get all cell barcodes
            all_cells = bpcells.cpp.cell_names_fragments_dir(str(temp_bpcells))

            if len(all_cells) == 0:
                self.logger.warning(
                    f"No cells found for {sample_id}. Skipping this fragment."
                )
                # Clean up temp directory if we created it (only if cleanup is enabled)
                if self.cleanup_temp_dirs and self._get_input_format(frag_path) != "bpcells":
                    shutil.rmtree(temp_bpcells)
                continue

            self.logger.info(f"Processing {len(all_cells)} cells for {sample_id}")

            # Build cell groups (each cell is its own group)
            cell_groups = build_cell_groups(
                str(temp_bpcells),
                all_cells,
                all_cells,  # Each cell is its own group
                all_cells,  # Order same as cells
                min_library_size=self.min_fragments,
                max_library_size=self.max_fragments,
            )

            # Get filtered cells directly from cell_groups (before precalculate_insertion_counts)
            # For single_sparse, each cell is its own group, so the categorical values are the filtered cell barcodes
            filtered_cells = list(cell_groups)  # Convert categorical to list to get filtered cell barcodes
            n_filtered = len(filtered_cells)
            n_total = len(all_cells)
            if n_filtered < n_total:
                self.logger.info(
                    f"Library size filtering for {sample_id}: {n_total} -> {n_filtered} cells "
                    f"({n_total - n_filtered} cells filtered out)"
                )

            # Get chromosome sizes (filtered by chromosomes in fragments)
            chrom_sizes = self._get_chrom_sizes(fragment_path=str(temp_bpcells))
            # Filter chromosome sizes for pseudobulk (removes patches, chrM, chrY, etc.)
            chrom_sizes_filtered = self._filter_chrom_sizes_for_pseudobulk(chrom_sizes)

            # Precalculate insertion counts
            # IMPORTANT: When cell_groups is pd.Categorical, group_names is ignored and
            # cell_groups.categories is used instead. However, we pass filtered_cells to be safe
            # in case the implementation changes or cell_groups is not Categorical.
            self.logger.info(f"Precalculating insertion counts for {sample_id}")
            precalculate_insertion_counts(
                str(temp_bpcells),
                str(sample_output),
                cell_groups,
                chrom_sizes_filtered,
                threads=self.threads,
                group_names=filtered_cells,  # Use filtered cells, not all_cells, to ensure consistency
            )

            # Clean up temp directory only if we created it (not if using existing BPCells)
            # and only if cleanup is enabled
            if self.cleanup_temp_dirs and self._get_input_format(frag_path) != "bpcells":
                shutil.rmtree(temp_bpcells)
                self.logger.debug(f"Cleaned up temp directory: {temp_bpcells}")
            elif not self.cleanup_temp_dirs and self._get_input_format(frag_path) != "bpcells":
                self.logger.info(
                    f"Keeping temporary BPCells fragment directory for reuse: {temp_bpcells}. "
                    f"Use --cleanup-temp-dirs to remove it."
                )

            # Save metadata if requested
            if self.save_metadata:
                # Create basic metadata using filtered cells
                metadata = pd.DataFrame(
                    {
                        "barcodes": filtered_cells,
                        "cell_idx": range(len(filtered_cells)),
                    }
                )

                # Add any additional metadata if available (filter by sample_id if present)
                all_metadata = self._load_and_merge_metadata()
                if all_metadata is not None:
                    if self.barcode_col in all_metadata.columns:
                        # Filter metadata for this sample if sample_id_col exists
                        frag_metadata = all_metadata.copy()
                        if self.sample_id_col in frag_metadata.columns:
                            frag_metadata = frag_metadata[frag_metadata[self.sample_id_col] == sample_id].copy()

                        metadata = metadata.merge(
                            frag_metadata,
                            left_on="barcodes",
                            right_on=self.barcode_col,
                            how="left",
                        )
                    else:
                        self.logger.warning(
                            f"Barcode column '{self.barcode_col}' not found in metadata, "
                            "skipping metadata merge"
                        )

                # Save files using filtered cells
                metadata.to_csv(sample_output / "metadata.csv", index=False)
                (sample_output / "barcodes.tsv").write_text("\n".join(filtered_cells) + "\n")

                # Library sizes are already saved by precalculate_insertion_counts in library_size.json
                # The C++ function creates a properly filtered library_size.json file
                library_size_path = sample_output / "library_size.json"
                if library_size_path.exists():
                    self.logger.debug(f"Library sizes already saved by precalculate_insertion_counts for {sample_id}")
                else:
                    self.logger.debug(f"No library_size.json found for {sample_id} (library sizes may not be available)")

                self.logger.info(f"Saved metadata for {sample_id} to {sample_output}")

            # Save attributes for this sample
            attrs = {
                "class": "SingleSparseBPCellsIO",
                "assembly": self._resolve_assembly(),
                "created_by": "caesar_bpcells_processor",
                "data_key": sample_id,
            }
            attrs_path = sample_output / "attrs.json"
            with open(attrs_path, "w") as f:
                json.dump(attrs, f, indent=4)

            sample_results.append({
                "sample_id": sample_id,
                "path": str(sample_output),
                "n_cells": len(filtered_cells),
            })
            total_cells += len(filtered_cells)

        if len(sample_results) == 0:
            raise ValueError(
                "No fragments were successfully processed. "
                "Check fragment file validity."
            )

        self.logger.info(
            f"MultiSparse processing complete: {len(sample_results)} samples, "
            f"{total_cells} total cells"
        )

        return {
            "output_path": str(self.output_path),
            "output_type": "multi_sparse",
            "n_samples": len(sample_results),
            "n_cells": total_cells,
            "samples": sample_results,
            "assembly": self._resolve_assembly(),
        }

    def _process_celltype_dense(self) -> dict[str, Any]:
        """Process fragments to CelltypeDense BPCells format."""
        self.logger.info("Processing fragments to CelltypeDense BPCells format")

        # Load metadata and merge with fragments
        all_metadata = self._load_and_merge_metadata()

        if all_metadata is None or all_metadata.empty:
            raise ValueError(
                f"No metadata found for CelltypeDense processing. "
                f"Metadata is required for celltype_dense output type."
            )

        # Validate barcode column exists
        if self.barcode_col not in all_metadata.columns:
            raise ValueError(
                f"Barcode column '{self.barcode_col}' not found in metadata. "
                f"Available columns: {list(all_metadata.columns)}"
            )

        # Determine grouping column
        if self.group_by == "celltype":
            group_col = self.annotation_col
        elif self.group_by == "sample":
            group_col = "sample"
        else:
            group_col = self.group_by

        # Apply cluster mapping if provided
        if self.cluster_mapping_file:
            self.logger.info(f"Applying cluster mapping from {self.cluster_mapping_file}")
            all_metadata = apply_cluster_mapping(
                metadata=all_metadata,
                mapping_file=self.cluster_mapping_file,
                source_col=group_col,
                target_col="mapped_labels",
                logger=self.logger,
            )
            group_col = "mapped_labels"

        # Validate grouping column exists
        if group_col not in all_metadata.columns:
            raise ValueError(
                f"Grouping column '{group_col}' not found in metadata. "
                f"Available columns: {list(all_metadata.columns)}. "
                f"Check annotation_col ({self.annotation_col}) or cluster mapping."
            )

        # Filter by target groups if specified
        if self.target_groups is not None:
            self.logger.info(f"Filtering for target groups: {self.target_groups}")
            n_before = len(all_metadata)
            all_metadata = all_metadata[all_metadata[group_col].isin(self.target_groups)].copy()
            n_after = len(all_metadata)
            self.logger.info(
                f"Filtered metadata: {n_before} -> {n_after} cells "
                f"({n_before - n_after} removed)"
            )
            if n_after == 0:
                raise ValueError(
                    f"No cells found in target groups: {self.target_groups}. "
                    f"Available groups: {sorted(all_metadata[group_col].unique())}"
                )

        # Get unique groups
        unique_groups = sorted(all_metadata[group_col].dropna().unique())
        self.logger.info(f"Found {len(unique_groups)} unique groups: {unique_groups}")

        # Get fragment paths and determine format
        fragment_paths = self._get_fragment_paths()

        # Expand any parent directories containing BPCells subdirectories
        expanded_paths = []
        for path in fragment_paths:
            path_obj = Path(path)
            if path_obj.is_dir():
                # Check if this directory itself is a BPCells directory (has version file)
                if (path_obj / "version").exists():
                    # It's a BPCells directory, use it directly
                    expanded_paths.append(path)
                else:
                    # Check if this is a parent directory with BPCells subdirectories
                    subdirs = [d for d in path_obj.iterdir() if d.is_dir()]
                    bpcells_subdirs = []
                    for subdir in subdirs:
                        # Check if subdirectory is a BPCells directory
                        if (subdir / "version").exists():
                            bpcells_subdirs.append(str(subdir))

                    if bpcells_subdirs:
                        # Expand to all BPCells subdirectories
                        expanded_paths.extend(bpcells_subdirs)
                        self.logger.info(f"Expanded parent directory {path} to {len(bpcells_subdirs)} BPCells subdirectories")
                    else:
                        # Not a BPCells directory and no BPCells subdirs - might be 10x or error
                        expanded_paths.append(path)
            else:
                expanded_paths.append(path)

        fragment_paths = expanded_paths
        first_path = fragment_paths[0]
        first_format = self._detect_input_format(first_path) if self.input_format == "auto" else self.input_format

        # When using clustering labels, we MUST use the saved BPCells fragments from import step
        # Don't try to re-import from tsv.gz files
        if self.bpcells_config.use_clustering_labels:
            # Find saved BPCells fragments from import step
            # They should be in output_dir/bpcells_fragments/{sample_id}/
            saved_fragments = []
            saved_frag_to_original = {}  # Map saved fragment path to original fragment config

            for frag in self.config.data.fragments:
                sample_id = self._get_frag_id_from_config(frag.path)
                saved_frag_path = self.output_dir / "bpcells_fragments" / sample_id
                if saved_frag_path.exists() and self._is_valid_bpcells_dir(saved_frag_path):
                    saved_frag_str = str(saved_frag_path)
                    saved_fragments.append(saved_frag_str)
                    saved_frag_to_original[saved_frag_str] = frag.path  # Store original path for ID lookup
                    self.logger.info(f"Using saved BPCells fragments from import step: {saved_frag_path}")
                else:
                    raise ValueError(
                        f"use_clustering_labels=True but saved BPCells fragments not found at {saved_frag_path}. "
                        f"Make sure the import_fragments step has run successfully."
                    )

            if not saved_fragments:
                raise ValueError(
                    "use_clustering_labels=True but no saved BPCells fragments found. "
                    "Make sure the import_fragments step has run successfully."
                )

            fragments_for_build = saved_fragments
            chrom_sizes_path = saved_fragments[0]

            # Store mapping for later use in cell matching
            self._saved_frag_to_original = saved_frag_to_original
        # Determine if we're using existing BPCells or need to convert
        elif first_format == "bpcells" or (self.input_format == "bpcells" and Path(first_path).is_dir()):
            # Use BPCells directories directly (can be multiple)
            # Check for version file to confirm BPCells directories
            bpcells_paths = [p for p in fragment_paths if Path(p).is_dir() and (Path(p) / "version").exists()]
            if not bpcells_paths:
                # Fallback: if input_format is explicitly "bpcells", trust the paths
                if self.input_format == "bpcells":
                    bpcells_paths = fragment_paths
                    self.logger.warning("Could not verify BPCells directories (no version files found), but proceeding with input_format='bpcells'")
                else:
                    raise ValueError(f"No valid BPCells directories found in fragment paths. Checked: {fragment_paths}")
            self.logger.info(f"Using {len(bpcells_paths)} existing BPCells directory(ies)")
            # For build_cell_groups and precalculate_insertion_counts, always pass a list
            # (matches reference script behavior)
            fragments_for_build = bpcells_paths
            # For chrom_sizes, use first path
            chrom_sizes_path = bpcells_paths[0]
        else:
            # Get sample_id for reuse check
            sample_id = self._get_frag_id_from_config(first_path)

            # Try to reuse existing bpcells_fragments first
            # Note: We check if existing fragments have all needed cells
            # If keeper_cells is used, we'll need to check if existing dir has all cells
            keeper_cells = all_metadata[self.barcode_col].tolist()

            # Check for existing fragments directory
            existing_fragments = self.output_dir / f"bpcells_fragments" / sample_id
            if existing_fragments.exists() and self._is_valid_bpcells_dir(existing_fragments):
                # Check if existing directory has all needed cells
                existing_cells = set(bpcells.cpp.cell_names_fragments_dir(str(existing_fragments)))
                needed_cells = set(keeper_cells)
                if needed_cells.issubset(existing_cells):
                    # Existing directory has all needed cells, reuse it
                    self.logger.info(
                        f"Reusing existing bpcells_fragments for {sample_id} "
                        f"(contains {len(existing_cells)} cells, need {len(needed_cells)})"
                    )
                    temp_bpcells = existing_fragments
                else:
                    # Need to create new with filtered cells
                    self.logger.info(
                        f"Existing bpcells_fragments missing {len(needed_cells - existing_cells)} cells, "
                        f"creating filtered conversion"
                    )
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    identifier = f"{timestamp}_{self.output_path.name}"
                    temp_bpcells = self._convert_10x_to_bpcells(
                        frag_path=first_path,
                        output_base_dir=self.output_dir,
                        category="fragments",
                        identifier=identifier,
                        keeper_cells=keeper_cells,
                    )
            else:
                # No existing directory, create new (with or without filtering)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                identifier = f"{timestamp}_{self.output_path.name}"
                temp_bpcells = self._convert_10x_to_bpcells(
                    frag_path=first_path,
                    output_base_dir=self.output_dir,
                    category="fragments",
                    identifier=identifier,
                    keeper_cells=keeper_cells if keeper_cells else None,
                )
            # Always pass as list for consistency with reference script
            fragments_for_build = [str(temp_bpcells)]
            chrom_sizes_path = str(temp_bpcells)

        # Build cell groups
        # Use dict-based API for multiple fragments to ensure correct cell-to-fragment mapping
        if len(fragments_for_build) > 1:
            # Multiple fragments: use dict-based API
            # Normalize all fragment paths first to ensure consistent matching
            fragments_normalized = [str(os.path.abspath(os.path.expanduser(f))) for f in fragments_for_build]
            frag_cell_ids_dict = {}
            frag_group_ids_dict = {}

            if self.sample_id_col in all_metadata.columns:
                # Match barcodes using sample_id when available
                for frag_path, frag_path_normalized in zip(fragments_for_build, fragments_normalized):
                    # For saved fragments, get original path to determine frag_id
                    if hasattr(self, '_saved_frag_to_original') and frag_path in self._saved_frag_to_original:
                        original_path = self._saved_frag_to_original[frag_path]
                        frag_id = self._get_frag_id_from_config(original_path)
                    else:
                        frag_id = self._get_frag_id_from_config(frag_path)

                    # Get cells from this fragment
                    try:
                        frag_cells = bpcells.cpp.cell_names_fragments_dir(frag_path)
                    except Exception as e:
                        self.logger.warning(f"Could not get cells from {frag_path}: {e}. Skipping.")
                        # Still add empty entry to dict to avoid "Missing cell_ids" error
                        frag_cell_ids_dict[frag_path_normalized] = []
                        frag_group_ids_dict[frag_path_normalized] = []
                        continue

                    # Filter metadata for this fragment's sample_id
                    frag_metadata = all_metadata[all_metadata[self.sample_id_col] == frag_id].copy()

                    # Match barcodes
                    barcode_to_group = dict(zip(frag_metadata[self.barcode_col], frag_metadata[group_col]))

                    # Collect matched cells and groups for this fragment
                    frag_matched_cells = [cell for cell in frag_cells if cell in barcode_to_group]
                    frag_matched_groups = [barcode_to_group[cell] for cell in frag_matched_cells]

                    # Always add entry to dict (even if empty) to avoid "Missing cell_ids" error
                    frag_cell_ids_dict[frag_path_normalized] = frag_matched_cells
                    frag_group_ids_dict[frag_path_normalized] = frag_matched_groups

                # Check if we matched any cells
                total_matched = sum(len(cells) for cells in frag_cell_ids_dict.values())
                if total_matched == 0:
                    raise ValueError(
                        f"No cells matched between metadata and fragments. "
                        f"Check that sample_id in metadata matches fragment file prefixes/ids. "
                        f"Fragment IDs: {[Path(p).stem.split('.')[0] if not Path(p).is_dir() else Path(p).name for p in fragments_for_build]}"
                    )

                self.logger.info(
                    f"Matched {total_matched} cells using sample_id for disambiguation across {len(frag_cell_ids_dict)} fragments"
                )
            else:
                # No sample_id - match all cells from all fragments
                # This assumes barcodes are unique or metadata has all cells
                barcode_to_group = dict(zip(all_metadata[self.barcode_col], all_metadata[group_col]))

                for frag_path, frag_path_normalized in zip(fragments_for_build, fragments_normalized):
                    try:
                        frag_cells = bpcells.cpp.cell_names_fragments_dir(frag_path)
                    except Exception as e:
                        self.logger.warning(f"Could not get cells from {frag_path}: {e}. Skipping.")
                        # Still add empty entry to dict to avoid "Missing cell_ids" error
                        frag_cell_ids_dict[frag_path_normalized] = []
                        frag_group_ids_dict[frag_path_normalized] = []
                        continue

                    frag_matched_cells = [cell for cell in frag_cells if cell in barcode_to_group]
                    frag_matched_groups = [barcode_to_group[cell] for cell in frag_matched_cells]

                    # Always add entry to dict (even if empty) to avoid "Missing cell_ids" error
                    frag_cell_ids_dict[frag_path_normalized] = frag_matched_cells
                    frag_group_ids_dict[frag_path_normalized] = frag_matched_groups

            # Use dict-based API for multiple fragments
            # Use normalized paths to match dict keys
            cell_groups_categorical = build_cell_groups(
                fragments_normalized,  # Use normalized paths
                frag_cell_ids_dict,
                frag_group_ids_dict,
                unique_groups,
                min_library_size=self.min_fragments,
                max_library_size=self.max_fragments,
            )
        else:
            # Single fragment: use list-based API (backward compatible)
            if self.sample_id_col in all_metadata.columns:
                # Match with sample_id (though only one fragment)
                frag_id = self._get_frag_id_from_config(fragments_for_build[0])
                frag_metadata = all_metadata[all_metadata[self.sample_id_col] == frag_id].copy()
                cell_barcodes = frag_metadata[self.barcode_col].tolist()
                cell_groups = frag_metadata[group_col].tolist()
            else:
                # No sample_id - simple matching
                cell_barcodes = all_metadata[self.barcode_col].tolist()
                cell_groups = all_metadata[group_col].tolist()

            # Use list-based API for single fragment
            cell_groups_categorical = build_cell_groups(
                fragments_for_build[0],  # Single path
                cell_barcodes,
                cell_groups,
                unique_groups,
                min_library_size=self.min_fragments,
                max_library_size=self.max_fragments,
            )

        # Get chromosome sizes (filtered by chromosomes in fragments)
        chrom_sizes = self._get_chrom_sizes(fragment_path=chrom_sizes_path)
        # Filter chromosome sizes for pseudobulk (removes patches, chrM, chrY, etc.)
        chrom_sizes_filtered = self._filter_chrom_sizes_for_pseudobulk(chrom_sizes)

        # Precalculate insertion counts by celltype
        self.logger.info(
            f"Precalculating insertion counts for {len(unique_groups)} groups"
        )
        # Use raw celltype names without prefix to match IO class expectations
        # Match reference script: use keyword arguments for clarity
        precalculate_insertion_counts(
            fragments=fragments_for_build,
            output_dir=str(self.output_path),
            cell_groups=cell_groups_categorical,
            chrom_sizes=chrom_sizes_filtered,
            threads=self.threads,
            group_names=unique_groups,  # Use raw names, not "CellType-" prefix
            preserve_chrom_order=True,  # Ensure consistent chromosome ordering for concatenation
        )

        # Clean up temp directory only if we created it (not if using existing BPCells or saved fragments)
        # temp_bpcells is only defined in the else branch (10x conversion path)
        # Skip cleanup if using clustering labels (saved fragments) or existing BPCells
        if not self.bpcells_config.use_clustering_labels and first_format != "bpcells":
            # Check if temp_bpcells was defined (only in else branch)
            if 'temp_bpcells' in locals():
                if self.cleanup_temp_dirs:
                    shutil.rmtree(temp_bpcells)
                    self.logger.debug(f"Cleaned up temp directory: {temp_bpcells}")
                else:
                    self.logger.info(
                        f"Keeping temporary BPCells fragment directory for reuse: {temp_bpcells}. "
                        f"Use --cleanup-temp-dirs to remove it."
                    )

        # Save group metadata (like reference script)
        group_metadata = pd.DataFrame({
            "group": unique_groups,
        })
        # Try to get library sizes from the matrix
        try:
            from bpcells.experimental import PrecalculatedInsertionMatrix
            matrix = PrecalculatedInsertionMatrix(str(self.output_path))
            group_metadata["library_size"] = matrix.library_size
        except Exception as e:
            self.logger.warning(f"Could not load library sizes for group metadata: {e}")

        group_metadata_path = self.output_path / "group_metadata.csv"
        group_metadata.to_csv(group_metadata_path, index=False)
        self.logger.info(f"Saved group metadata to {group_metadata_path}")

        return {
            "output_path": str(self.output_path),
            "output_type": "celltype_dense",
            "n_celltypes": len(unique_groups),
            "celltypes": unique_groups,
            "assembly": self._resolve_assembly(),
        }

    def _process_signal_dense(self) -> dict[str, Any]:
        """Process fragments to SignalDense BPCells format (pseudobulk)."""
        self.logger.info("Processing fragments to SignalDense BPCells format")

        # Get fragment paths and determine format
        fragment_paths = self._get_fragment_paths()
        first_path = fragment_paths[0]
        first_format = self._get_input_format(first_path)

        # Get sample_id for reuse check
        sample_id = self._get_frag_id_from_config(first_path)

        # Get BPCells directory (use existing or convert)
        # Reuse bpcells_fragments if available to avoid duplicate conversions
        temp_bpcells = self._get_bpcells_dir_or_convert(
            frag_path=first_path,
            category="fragments",
            identifier=sample_id,
            check_existing=True,
            existing_category="fragments",
        )

        # Get all cells
        all_cells = bpcells.cpp.cell_names_fragments_dir(str(temp_bpcells))
        self.logger.info(f"Creating pseudobulk from {len(all_cells)} cells")

        # Create single pseudobulk group
        pseudobulk_groups = ["pseudobulk"] * len(all_cells)

        cell_groups_categorical = build_cell_groups(
            str(temp_bpcells),
            all_cells,
            pseudobulk_groups,
            ["pseudobulk"],
            min_library_size=self.min_fragments,
            max_library_size=self.max_fragments,
        )

        # Get chromosome sizes (filtered by chromosomes in fragments)
        chrom_sizes = self._get_chrom_sizes(fragment_path=str(temp_bpcells))
        # Filter chromosome sizes for pseudobulk (removes patches, chrM, chrY, etc.)
        chrom_sizes_filtered = self._filter_chrom_sizes_for_pseudobulk(chrom_sizes)

        # Precalculate insertion counts
        self.logger.info("Precalculating pseudobulk insertion counts")
        precalculate_insertion_counts(
            str(temp_bpcells),
            str(self.output_path),
            cell_groups_categorical,
            chrom_sizes_filtered,
            threads=self.threads,
            group_names=["pseudobulk"],
            preserve_chrom_order=True,  # Ensure consistent chromosome ordering for concatenation
        )

        # Clean up temp directory only if we created it (not if using existing BPCells)
        # and only if cleanup is enabled
        if self.cleanup_temp_dirs and first_format != "bpcells":
            shutil.rmtree(temp_bpcells)
            self.logger.debug(f"Cleaned up temp directory: {temp_bpcells}")
        elif not self.cleanup_temp_dirs and first_format != "bpcells":
            self.logger.info(
                f"Keeping temporary BPCells fragment directory for reuse: {temp_bpcells}. "
                f"Use --cleanup-temp-dirs to remove it."
            )

        return {
            "output_path": str(self.output_path),
            "output_type": "signal_dense",
            "n_cells": len(all_cells),
            "assembly": self._resolve_assembly(),
        }

    def _process_anndata(self) -> dict[str, Any]:
        """Process fragments to AnnData format with binned genome-wide coverage.

        Creates a binned genome-wide insertion count matrix where each column
        represents a genomic bin. Supports processing multiple fragments in parallel
        and concatenating them into a single AnnData object.

        Note:
            Chromosome processing: All chromosomes are processed together in a single
            call to `precalculate_insertion_counts_binned`. The BPCells backend creates
            a single matrix where columns represent bins across all chromosomes
            concatenated together (not processed separately). The matrix has shape
            (n_cells, total_bins_across_all_chromosomes). Chromosome filtering (e.g.,
            removing chrM, chrY, patches) is applied AFTER matrix creation during tile
            filtering, matching the Zarr backend approach.
        """
        if not HAS_ANNDATA:
            raise ImportError(
                "anndata library not found. Please install with: pip install anndata"
            )

        self.logger.info("Processing fragments to AnnData format (binned genome-wide)")

        # Get fragment paths (handles both 10x files and BPCells directories)
        fragment_paths = self._get_fragment_paths()

        if not fragment_paths:
            raise ValueError(
                "No fragment paths found. Please ensure fragment files are specified in config.data.fragments"
            )

        self.logger.info(f"Processing {len(fragment_paths)} fragment file(s) - will combine into single AnnData")

        # Determine input format and convert/collect BPCells directories
        fragments_for_build = []
        temp_dirs_to_cleanup = []

        for frag_idx, frag_path in enumerate(fragment_paths):
            if not frag_path or not frag_path.strip():
                raise ValueError(f"Empty fragment path at index {frag_idx}")

            frag_format = self._get_input_format(frag_path)
            sample_id = self._get_frag_id_from_config(frag_path)

            # Get BPCells directory (use existing or convert)
            temp_bpcells = self._get_bpcells_dir_or_convert(
                frag_path=frag_path,
                category="fragments",
                identifier=sample_id,
                check_existing=True,
                existing_category="fragments",
                log_message=f"Converting 10x fragment file {frag_idx + 1}/{len(fragment_paths)} to BPCells format: {frag_path}",
            )

            if not temp_bpcells or not str(temp_bpcells).strip():
                raise ValueError(f"Failed to get BPCells directory for fragment {frag_path}")

            fragments_for_build.append(str(temp_bpcells))
            if frag_format != "bpcells":
                temp_dirs_to_cleanup.append(temp_bpcells)

        if not fragments_for_build:
            raise ValueError("No BPCells fragments were successfully prepared for AnnData generation")

        # Get chromosome sizes (unfiltered - includes all chromosomes from fragments)
        # We use unfiltered chrom_sizes to create the matrix, then filter tiles afterward
        # This matches the Zarr backend approach
        # NOTE: precalculate_insertion_counts_binned processes ALL chromosomes together
        # in a single call - it does NOT process chromosomes separately. The resulting
        # matrix has columns representing bins from all chromosomes concatenated.
        chrom_sizes_unfiltered = self._get_chrom_sizes(fragment_path=fragments_for_build[0])
        self.logger.info(
            f"Using unfiltered chrom_sizes for binning: {len(chrom_sizes_unfiltered)} chromosomes "
            f"(includes patches, chrM, chrY, etc. - will be filtered in tiles)"
        )

        # Create temporary output directory for intermediate matrices
        import tempfile
        temp_output_base = tempfile.mkdtemp(prefix="bpcells_anndata_")
        frag_outputs = []
        frag_filtered_cell_counts = {}

        # Process each fragment independently in parallel
        self.logger.info(f"Processing {len(fragments_for_build)} fragments in parallel with bin_size={self.bin_size}bp")

        # Use ProcessPoolExecutor to process fragments in parallel
        # Note: _process_single_fragment_for_anndata is at module level to be picklable
        frag_errors = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_frag = {
                executor.submit(
                    _process_single_fragment_for_anndata,
                    (frag_path, temp_output_base, chrom_sizes_unfiltered, self.bin_size,
                     self.min_fragments, self.max_fragments, self.threads)
                ): frag_path
                for frag_path in fragments_for_build
            }

            # Collect results as they complete
            for future in as_completed(future_to_frag):
                frag_path = future_to_frag[future]
                frag_name = Path(frag_path).name
                try:
                    frag_output, filtered_cell_count, error = future.result()
                    if error:
                        self.logger.error(f"Error processing {frag_name}: {error}")
                        frag_errors.append((frag_path, error))
                    else:
                        self.logger.info(f"Completed {frag_name}: {filtered_cell_count:,} filtered cells")
                        frag_outputs.append(frag_output)
                        frag_filtered_cell_counts[frag_output] = filtered_cell_count
                except Exception as e:
                    self.logger.error(f"Exception processing {frag_name}: {e}")
                    import traceback
                    frag_errors.append((frag_path, f"{str(e)}\n{traceback.format_exc()}"))

        if frag_errors:
            self.logger.error(f"{len(frag_errors)} fragments failed to process")
            raise RuntimeError(f"Failed to process {len(frag_errors)} fragments. Check logs for details.")

        if not frag_outputs:
            raise ValueError("No fragments were successfully processed")

        self.logger.info(f"Successfully processed {len(frag_outputs)} fragments")

        # Concatenate DirMatrix objects using from_vstack
        self.logger.info("Concatenating fragment matrices...")
        self.logger.info(f"Creating DirMatrix objects for {len(frag_outputs)} fragments...")
        dir_matrices = []
        for i, frag_output in enumerate(frag_outputs):
            try:
                self.logger.debug(f"Loading DirMatrix {i+1}/{len(frag_outputs)}: {frag_output}")
                mat = DirMatrix(frag_output)
                self.logger.debug(f"  Shape: {mat.shape}")
                dir_matrices.append(mat)
            except Exception as e:
                self.logger.error(f"Failed to load DirMatrix from {frag_output}: {e}")
                raise

        if len(dir_matrices) == 0:
            raise ValueError("No valid DirMatrix objects to concatenate")

        combined_output_dir = Path(temp_output_base) / "combined_matrix"
        if combined_output_dir.exists():
            shutil.rmtree(combined_output_dir)

        self.logger.info(f"Concatenating {len(dir_matrices)} DirMatrix objects using from_vstack...")
        try:
            combined_mat = DirMatrix.from_vstack(dir_matrices, str(combined_output_dir))
            self.logger.info(f"Combined matrix shape: {combined_mat.shape}")
        except Exception as e:
            self.logger.error(f"Failed to concatenate DirMatrix objects: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

        # Load metadata to get chrom_sizes and bin_size
        import json
        metadata_path = Path(frag_outputs[0]) / "binning_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            chrom_sizes_for_adata = metadata.get('chrom_sizes', chrom_sizes_unfiltered)
            bin_size_for_adata = metadata.get('bin_size', self.bin_size)
        else:
            chrom_sizes_for_adata = chrom_sizes_unfiltered
            bin_size_for_adata = self.bin_size

        # Get blacklist from config (same as Zarr backend)
        # Priority:
        # 1. Explicit blacklist_file path (if specified)
        # 2. clustering.blacklist_filtering setting (if available)
        # 3. Default to None (use assembly-specific blacklist)
        blacklist = getattr(self.config, "blacklist_file", None)

        # If blacklist_file is not explicitly set, check clustering config
        if blacklist is None:
            # Try to get from clustering config if available
            if hasattr(self.config, "clustering") and hasattr(self.config.clustering, "blacklist_filtering"):
                if self.config.clustering.blacklist_filtering:
                    # Use assembly-specific blacklist (blacklist=None triggers default in filter_tiles)
                    blacklist = None
                    self.logger.info("Blacklist filtering enabled via clustering.blacklist_filtering=True, using assembly-specific blacklist")
                else:
                    # Explicitly skip blacklist filtering
                    blacklist = False
                    self.logger.info("Blacklist filtering disabled via clustering.blacklist_filtering=False")
            else:
                # Default: use assembly-specific blacklist (blacklist=None triggers default in filter_tiles)
                blacklist = None
                self.logger.info("Using default blacklist filtering (assembly-specific blacklist)")
        elif blacklist is not False:
            # Explicit blacklist file path provided
            self.logger.info(f"Using blacklist file from config: {blacklist}")

        # Create bin annotations (var) using PyRanges tiling
        self.logger.info("Creating bin annotations (var) using PyRanges tiling...")
        from caesar.preprocess.core.tile_filtering import filter_tiles, create_column_indices

        # Resolve assembly from 'auto' if needed
        resolved_assembly = self._resolve_assembly()

        # Get annotation directory from gcell settings
        try:
            from gcell import _settings
            annotation_dir = _settings.get_setting("annotation_dir")
        except (ImportError, AttributeError):
            annotation_dir = None

        if annotation_dir is None:
            raise ValueError(
                "Annotation directory not configured. Please set gcell annotation_dir setting."
            )

        chrom_size_obj = ChromSize(assembly=resolved_assembly, annotation_dir=annotation_dir)

        # Create ALL tiles first (no filtering) - this matches the matrix columns
        # The matrix was created with unfiltered chrom_sizes, so all_tiles should match
        all_tiles_df = chrom_size_obj.tiling_region(tile_size=bin_size_for_adata, tile_overlap=0)

        # Filter to only chromosomes we have in the matrix (from unfiltered chrom_sizes)
        all_tiles_df = all_tiles_df[all_tiles_df['Chromosome'].isin(chrom_sizes_for_adata.keys())].copy()
        all_tiles_df = all_tiles_df.reset_index(drop=True)

        # Apply chromosome and blacklist filtering to get final tiles (var)
        # This matches the Zarr backend approach: filter tiles AFTER matrix creation
        tiles_df = filter_tiles(
            all_tiles_df.copy(),
            blacklist=blacklist,
            assembly=resolved_assembly,
        )

        # Filter to only include chromosomes we have
        tiles_df = tiles_df[tiles_df['Chromosome'].isin(chrom_sizes_for_adata.keys())]

        # Reset index to ensure continuous indices [0, 1, 2, ...]
        # This is important for peak ordering to work correctly
        tiles_df = tiles_df.reset_index(drop=True)

        # Create column indices using shared utility (vectorized, no loop)
        # This returns indices to select from the matrix, not a boolean mask
        self.logger.info(
            f"Creating column indices: {len(all_tiles_df)} total bins -> {len(tiles_df)} after filtering"
        )

        column_indices = create_column_indices(all_tiles_df, tiles_df)

        n_kept = len(column_indices)
        n_removed = len(all_tiles_df) - n_kept
        self.logger.info(
            f"Column selection: keeping {n_kept:,} columns, removing {n_removed:,} filtered columns"
        )

        # Rename columns to match expected format
        var_df = tiles_df.rename(columns={
            'Chromosome': 'chrom',
            'Start': 'start',
            'End': 'end'
        })[['chrom', 'start', 'end']].copy()

        # Create cell/group annotations (obs)
        self.logger.info("Creating cell/group annotations (obs)...")
        all_group_names = []
        for frag_output in frag_outputs:
            frag_group_names_path = Path(frag_output) / "group_names.json"
            if frag_group_names_path.exists():
                with open(frag_group_names_path, 'r') as f:
                    frag_group_names = json.load(f)
                    all_group_names.extend(frag_group_names)
            else:
                # Fallback: use indices
                frag_mat = DirMatrix(frag_output)
                num_groups = frag_mat.shape[0]
                all_group_names.extend([f"group_{i}" for i in range(num_groups)])

        # Ensure we have the right number of group names
        if len(all_group_names) < combined_mat.shape[0]:
            for i in range(len(all_group_names), combined_mat.shape[0]):
                all_group_names.append(f"group_{i}")
        elif len(all_group_names) > combined_mat.shape[0]:
            all_group_names = all_group_names[:combined_mat.shape[0]]

        obs_df = pd.DataFrame({
            'group': all_group_names
        })
        obs_df.index = obs_df['group'].values

        # Get matrix dimensions
        rows, cols = combined_mat.shape
        self.logger.info(f"Combined matrix dimensions: {rows:,} rows x {cols:,} columns")

        # Test loading a small subset first to verify the matrix is accessible
        self.logger.info("Testing matrix accessibility with small subset...")
        try:
            test_subset = combined_mat[0:min(100, rows), 0:min(1000, cols)]
            subset_sum = test_subset.sum()
            self.logger.info(f"Test subset (100x1000): shape {test_subset.shape}, nnz={test_subset.nnz:,}, sum={subset_sum:,}")
            if subset_sum == 0:
                self.logger.warning("Test subset sum is zero - matrix may be empty")
            del test_subset  # Free memory
        except Exception as e:
            self.logger.error(f"Failed to load test subset: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

        # Load matrix in chunks and convert to AnnData
        # Use BPCells utility to calculate safe chunk size and load in chunks
        self.logger.info("Loading combined matrix in chunks and converting to AnnData...")
        self.logger.info(f"Matrix dimensions: {rows:,} rows x {cols:,} columns")

        # Calculate safe chunk size using BPCells utility (avoids uint32_t overflow)
        chunk_size = DirMatrix.calculate_safe_chunk_size(rows)
        self.logger.info(f"Calculated safe chunk size: {chunk_size:,} columns (to avoid uint32_t overflow)")

        # Use BPCells chunked loading method
        try:
            chunk_matrices = combined_mat.load_chunked_columns(
                start_col=0,
                end_col=cols,
                chunk_size=chunk_size,
                format='csr'
            )
            self.logger.info(f"Loaded {len(chunk_matrices)} chunks successfully")
        except Exception as e:
            self.logger.error(f"Failed to load matrix in chunks: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

        self.logger.info(f"Concatenating {len(chunk_matrices)} chunks horizontally...")
        import scipy.sparse
        try:
            combined_sparse = scipy.sparse.hstack(chunk_matrices, format='csr')
            self.logger.info(f"Loaded full matrix: shape {combined_sparse.shape}, nnz={combined_sparse.nnz:,}")
        except Exception as e:
            self.logger.error(f"Failed to concatenate chunks: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

        # Select matrix columns to match filtered tiles (same as Zarr backend)
        # Use index selection rather than boolean mask for efficiency
        if n_removed > 0:
            self.logger.info(f"Selecting matrix columns to keep {n_kept:,} filtered bins...")
            combined_sparse = combined_sparse[:, column_indices]
            self.logger.info(
                f"Selected matrix: shape {combined_sparse.shape}, nnz={combined_sparse.nnz:,}"
            )

        # Create AnnData object
        adata = ad.AnnData(X=combined_sparse)
        adata.var = var_df
        adata.obs = obs_df

        self.logger.info(f"Created AnnData object: shape {adata.shape} (cells x bins)")

        # Save AnnData to anndata folder
        adata_path = self.output_path / "anndata.h5ad"
        adata_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing file or directory if it exists (to avoid conflicts)
        # This is critical because anndata.write() will fail if the path exists as a directory
        if adata_path.exists():
            if adata_path.is_dir():
                self.logger.warning(f"Removing existing directory at {adata_path} to create file")
                shutil.rmtree(adata_path)
            elif adata_path.is_file():
                self.logger.info(f"Removing existing file at {adata_path}")
                adata_path.unlink()

        # Double-check: ensure the path doesn't exist before writing
        if adata_path.exists():
            self.logger.error(f"Path still exists after cleanup: {adata_path}")
            raise RuntimeError(f"Cannot write to {adata_path}: path still exists after cleanup")

        self.logger.info(f"Saving AnnData to {adata_path}...")
        # Convert to absolute path string to avoid any path resolution issues
        adata_path_str = str(adata_path.absolute())
        self.logger.debug(f"Absolute path: {adata_path_str}")

        try:
            adata.write(adata_path_str)
            # Verify the file was created correctly
            if not Path(adata_path_str).exists():
                raise RuntimeError(f"anndata.write() did not create file at {adata_path_str}")
            if Path(adata_path_str).is_dir():
                raise RuntimeError(f"anndata.write() created a directory instead of a file at {adata_path_str}")
            self.logger.info(f"Saved AnnData to {adata_path_str}")
        except Exception as e:
            # Check if the path was created as a directory (which would indicate a problem)
            if adata_path.exists() and adata_path.is_dir():
                self.logger.error(f"anndata.write() created a directory instead of a file at {adata_path}")
                self.logger.error(f"This usually means the path already existed as a directory before writing")
            raise

        # Clean up temp directories
        if self.cleanup_temp_dirs:
            shutil.rmtree(temp_output_base)
            for temp_dir in temp_dirs_to_cleanup:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        else:
            self.logger.info(f"Keeping temporary directories for reuse. Use --cleanup-temp-dirs to remove them.")

        return {
            "output_path": str(adata_path),
            "output_type": "anndata",
            "n_cells": combined_mat.shape[0],
            "n_bins": combined_mat.shape[1],
            "bin_size": bin_size_for_adata,
            "assembly": self._resolve_assembly(),
        }

    def _process_fragments(self) -> dict[str, Any]:
        """Process fragments to BPCells fragment format only (no matrix building).

        This method imports 10x fragment files to BPCells format and saves them
        for use in subsequent workflow steps. It does not build any matrices.

        Returns:
            Processing results with paths to saved BPCells fragment directories
        """
        self.logger.info("Importing fragments to BPCells format (no matrix building)")

        # Get fragment paths (handles both 10x files and BPCells directories)
        fragment_paths = self._get_fragment_paths()

        self.logger.info(f"Importing {len(fragment_paths)} fragment file(s) to BPCells format")

        # Import each fragment and save to output directory
        imported_paths = []
        for frag_idx, frag_path in enumerate(fragment_paths):
            frag_format = self._get_input_format(frag_path)

            if frag_format == "bpcells":
                # Already in BPCells format - just use it
                self.logger.info(f"Fragment {frag_idx + 1} is already in BPCells format: {frag_path}")
                imported_paths.append(frag_path)
                continue

            # Get sample_id from config or path
            sample_id = self._get_frag_id_from_config(frag_path)

            # Import to BPCells format
            bpcells_dir = self._convert_10x_to_bpcells(
                frag_path=frag_path,
                output_base_dir=self.output_dir,
                category="fragments",
                identifier=sample_id,
                keeper_cells=None,  # Import all cells at this stage
                log_message=f"Importing fragment {frag_idx + 1}/{len(fragment_paths)} to BPCells format: {frag_path}",
            )

            imported_paths.append(str(bpcells_dir))
            self.logger.info(f"Imported fragment {frag_idx + 1} to: {bpcells_dir}")

        # Save attributes
        self._save_attributes()

        return {
            "output_path": str(self.output_path),
            "output_type": "fragments",
            "imported_fragments": imported_paths,
            "n_fragments": len(imported_paths),
        }

    def _convert_10x_to_bpcells(
        self,
        frag_path: str,
        output_base_dir: Path,
        category: str,
        identifier: str,
        keeper_cells: list[str] | None = None,
        log_message: str | None = None,
    ) -> Path:
        """Convert a 10x fragment file to BPCells format.

        Args:
            frag_path: Path to 10x fragment file
            output_base_dir: Base directory for temporary BPCells output
            category: Category name for organizing temp directories (e.g., "fragments", "multi_sparse")
            identifier: Unique identifier within the category (e.g., sample_id, frag_idx)
            keeper_cells: Optional list of cell barcodes to keep
            log_message: Optional custom log message (default: auto-generated)

        Returns:
            Path to the converted BPCells directory
        """
        # Create temp directory path with nested structure: bpcells_{category}/{identifier}
        temp_bpcells = output_base_dir / f"bpcells_{category}" / identifier

        # Check if directory already exists and is valid
        if temp_bpcells.exists() and self._is_valid_bpcells_dir(temp_bpcells):
            # Report number of cells in existing directory
            n_cells = len(bpcells.cpp.cell_names_fragments_dir(str(temp_bpcells)))
            self.logger.info(
                f"BPCells directory already exists: {temp_bpcells} "
                f"(contains {n_cells} cells)"
            )
            return temp_bpcells

        # Clean up temp directory if it exists but is invalid
        if temp_bpcells.exists():
            self.logger.warning(f"Removing invalid BPCells directory: {temp_bpcells}")
            shutil.rmtree(temp_bpcells)

        # Log conversion
        if log_message:
            self.logger.info(log_message)
        else:
            self.logger.info(f"Converting 10x fragment file to BPCells format: {frag_path}")

        # Convert to BPCells format
        # Convert keeper_cells to list if it's a numpy array (bpcells expects list or None)
        keeper_cells_list = None
        if keeper_cells is not None:
            import numpy as np
            if isinstance(keeper_cells, np.ndarray):
                keeper_cells_list = keeper_cells.tolist()
            elif isinstance(keeper_cells, list):
                keeper_cells_list = keeper_cells
            else:
                keeper_cells_list = list(keeper_cells)

        # Import fragment file to BPCells format
        # NOTE: BPCells uses its native C++ fragment reader which handles indexing
        # internally - no tabix index needed (unlike FragmentProcessor which uses
        # pysam.TabixFile for chromosome-by-chromosome reading)
        import_10x_fragments(
            input=frag_path,
            output=str(temp_bpcells),
            shift_end=self.shift_end,
            keeper_cells=keeper_cells_list,
        )

        # Report number of cells after import
        n_cells = len(bpcells.cpp.cell_names_fragments_dir(str(temp_bpcells)))
        self.logger.info(f"Imported {n_cells} cells from {frag_path}")

        return temp_bpcells

    def _is_valid_bpcells_dir(self, bpcells_path: Path) -> bool:
        """Check if a directory is a valid BPCells fragment directory.

        Args:
            bpcells_path: Path to check

        Returns:
            True if the directory is a valid BPCells directory, False otherwise
        """
        if not bpcells_path.is_dir():
            return False

        # Check for essential BPCells marker files
        marker_files = ["cell_names", "chr_names"]
        # Check for at least one of the count files
        count_files = ["cell_data"]

        has_markers = all((bpcells_path / marker).exists() for marker in marker_files)
        has_counts = any((bpcells_path / count_file).exists() for count_file in count_files)

        return has_markers and has_counts

    def _get_bpcells_dir_or_convert(
        self,
        frag_path: str,
        category: str,
        identifier: str,
        check_existing: bool = False,
        existing_category: str | None = None,
        keeper_cells: list[str] | None = None,
        log_message: str | None = None,
    ) -> Path:
        """Get BPCells directory, using existing if available or converting if needed.

        Args:
            frag_path: Path to fragment file or BPCells directory
            category: Category for temp directory organization
            identifier: Identifier within the category
            check_existing: Whether to check for existing BPCells directory in bpcells_{existing_category}
            existing_category: Category to check for existing directories (if check_existing is True)
            keeper_cells: Optional list of cell barcodes to keep during conversion
            log_message: Optional custom log message

        Returns:
            Path to BPCells directory
        """
        frag_format = self._get_input_format(frag_path)

        if frag_format == "bpcells":
            # Use BPCells directory directly
            bpcells_path = Path(frag_path)
            # Report number of cells in existing directory
            n_cells = len(bpcells.cpp.cell_names_fragments_dir(str(bpcells_path)))
            self.logger.info(
                f"Using existing BPCells directory: {frag_path} "
                f"(contains {n_cells} cells)"
            )
            return bpcells_path

        # Check for existing converted directory if requested
        # For anndata output, use output_dir directly (not output_path.parent) to avoid putting bpcells_fragments under anndata/
        if self.output_type == "anndata":
            output_base_dir = self.output_dir
        else:
            if not hasattr(self, 'output_path') or self.output_path is None:
                raise ValueError(
                    f"output_path not set for output_type={self.output_type}. "
                    f"This should be set during processor initialization."
                )
            output_base_dir = self.output_path.parent

        if not output_base_dir or not str(output_base_dir).strip():
            raise ValueError(
                f"Empty output_base_dir calculated: output_type={self.output_type}, "
                f"output_dir={self.output_dir}, output_path={getattr(self, 'output_path', None)}"
            )

        if check_existing and existing_category:
            existing_bpcells = output_base_dir / f"bpcells_{existing_category}" / identifier
            if existing_bpcells.exists() and self._is_valid_bpcells_dir(existing_bpcells):
                # Report number of cells in existing directory
                existing_bpcells_str = str(existing_bpcells)
                if not existing_bpcells_str.strip():
                    raise ValueError(f"Empty path generated for existing_bpcells: {existing_bpcells}")
                n_cells = len(bpcells.cpp.cell_names_fragments_dir(existing_bpcells_str))
                self.logger.info(
                    f"Using existing BPCells directory for {identifier}: {existing_bpcells} "
                    f"(contains {n_cells} cells)"
                )
                return existing_bpcells

        # Convert 10x fragment file to BPCells format
        # For anndata output, use output_dir directly (not output_path.parent) to avoid putting bpcells_fragments under anndata/
        # output_base_dir already set above
        return self._convert_10x_to_bpcells(
            frag_path=frag_path,
            output_base_dir=output_base_dir,
            category=category,
            identifier=identifier,
            keeper_cells=keeper_cells,
            log_message=log_message,
        )

    def _get_frag_id_from_config(self, frag_path: str) -> str:
        """Extract fragment ID from config or path.

        Uses unified extract_fragment_info() function for consistent behavior
        with FragmentProcessor.

        Args:
            frag_path: Path to fragment file/directory

        Returns:
            Fragment ID (from config if available, otherwise from path prefix)
        """
        from caesar.preprocess.utils.fragment_info import extract_fragment_info

        # Check if verbose logging is enabled (INFO level or lower)
        verbose = logging.getLogger().isEnabledFor(logging.INFO)

        frag_info = extract_fragment_info(
            frag_path=frag_path,
            fragments_config=self.config.data.fragments if self.config.data.fragments else None,
            verbose=verbose,
        )
        return frag_info["id"]

    def _get_chrom_sizes(self, fragment_path: str | None = None) -> dict[str, int]:
        """Get chromosome sizes for the assembly using gcell ChromSize class.

        Args:
            fragment_path: Path to fragment file/directory to get chromosomes from.
                          If None, uses first fragment file from config.

        Returns:
            Dictionary mapping chromosome names to sizes, filtered by chromosomes
            present in fragment files.
        """
        # Resolve assembly from 'auto' if needed
        assembly = self._resolve_assembly()

        # Get annotation directory from gcell settings or config
        try:
            from gcell import _settings
            annotation_dir = _settings.get_setting("annotation_dir")
        except (ImportError, AttributeError):
            annotation_dir = None

        if annotation_dir is None:
            # Fallback: use a default location or raise error
            raise ValueError(
                "Annotation directory not configured. Please set gcell annotation_dir setting."
            )

        # Get chromosomes from fragment files if fragment_path provided
        chr_names = None
        if fragment_path:
            if not fragment_path.strip():
                raise ValueError(f"Empty fragment_path provided to _get_chrom_sizes: '{fragment_path}'")
            try:
                chr_names = bpcells.cpp.chr_names_fragments_dir(fragment_path)
            except Exception as e:
                self.logger.warning(f"Could not get chromosomes from {fragment_path}: {e}")

        # Use ChromSize to get sizes (downloads/caches automatically)
        try:
            chrom_size_obj = ChromSize(assembly=assembly, annotation_dir=annotation_dir)
        except Exception as e:
            self.logger.error(f"Failed to load chromosome sizes: {e}")
            raise

        # Filter to only chromosomes present in fragments if available
        if chr_names:
            chrom_sizes = chrom_size_obj.get_dict(chr_included=chr_names)
            self.logger.info(
                f"Filtered chromosome sizes: {len(chrom_sizes)} chromosomes "
                f"from {len(chr_names)} in fragments"
            )
        else:
            chrom_sizes = chrom_size_obj.get_dict()
            self.logger.info(f"Using all chromosome sizes: {len(chrom_sizes)} chromosomes")

        # Filter out None values (chromosomes not in reference genome)
        chrom_sizes = {k: v for k, v in chrom_sizes.items() if v is not None}
        if len(chrom_sizes) < len([k for k in chrom_sizes.keys()]):
            self.logger.warning(
                f"Filtered out {len([k for k, v in chrom_sizes.items() if v is None])} "
                f"chromosomes with None sizes"
            )

        return chrom_sizes

    def _filter_chrom_sizes_for_pseudobulk(self, chrom_sizes: dict[str, int]) -> dict[str, int]:
        """Filter chromosome sizes for pseudobulk generation using ChromosomeFilter.

        This removes non-standard chromosomes (patches, chrM, chrY, etc.) to ensure
        consistency with tile filtering used in downstream analysis.

        IMPORTANT: This also enforces canonical chromosome ordering (chr1, chr2, ..., chrX)
        to ensure consistent chrom_offsets across samples when using BPCells backend.
        This is critical for concatenating celltype_dense matrices from different samples.

        Note: This should be used for precalculate_insertion_counts (pseudobulk),
        but NOT for precalculate_insertion_counts_binned (AnnData) which needs
        to keep aligned with original fragment chromosome order due to C++ implementation.

        Args:
            chrom_sizes: Dictionary of chromosome names to sizes

        Returns:
            Filtered dictionary with only canonical chromosomes in canonical order
        """
        from caesar.preprocess.core.chromosome_filter import ChromosomeFilter
        chr_filter = ChromosomeFilter({})  # Use default filtering rules

        # Get canonical chromosome order from assembly
        assembly = self._resolve_assembly()
        canonical_order = chr_filter.get_standard_chromosomes(assembly)

        # Filter to only chromosomes present in chrom_sizes and in canonical order
        # This ensures consistent ordering across different fragment files
        n_before_filter = len(chrom_sizes)
        filtered_sizes = {}
        for chrom in canonical_order:
            if chrom in chrom_sizes:
                filtered_sizes[chrom] = chrom_sizes[chrom]

        n_filtered = n_before_filter - len(filtered_sizes)
        if n_filtered > 0:
            self.logger.info(
                f"Filtered out {n_filtered} non-standard chromosomes "
                f"(patches, chrM, chrY, etc.), keeping {len(filtered_sizes)} canonical chromosomes "
                f"in canonical order"
            )

        return filtered_sizes

    def _load_and_merge_metadata(self) -> pd.DataFrame | None:
        """Load and merge metadata from fragment files using centralized function.

        Automatically adds sample_id column when needed to handle non-unique barcodes
        across samples. The sample_id is always added if fragment.id is available,
        but only used for matching when multiple fragments are present.

        If use_clustering_labels is True, loads metadata from the clustering step
        output (anndata/metadata.csv) instead of fragment-specific metadata files.
        """
        # Check if we should use clustering labels
        if self.bpcells_config.use_clustering_labels:
            # Load metadata from clustering output
            # Try to load from AnnData file first (has barcodes in index)
            anndata_path = self.output_dir / "anndata" / "anndata.h5ad"
            clustering_metadata_path = self.output_dir / "anndata" / "metadata.csv"

            if anndata_path.exists():
                # Load from AnnData file - barcodes are in obs.index
                try:
                    import scanpy as sc
                    self.logger.info(f"Loading clustering metadata from AnnData file: {anndata_path}")
                    adata = sc.read_h5ad(anndata_path)
                    metadata = adata.obs.copy()
                    # Barcodes are in the index - reset to make them a column
                    metadata = metadata.reset_index()
                    # The first column should now be the barcodes (from index)
                    # Rename it to barcode_col
                    first_col = metadata.columns[0]

                    # Check if index is numeric (old format) - if so, get barcodes from saved fragments
                    if metadata[first_col].dtype in ['int64', 'int32'] or all(str(x).isdigit() for x in metadata[first_col].head(10)):
                        self.logger.warning(
                            f"AnnData obs index is numeric (old format). "
                            f"Getting actual barcodes from saved BPCells fragments..."
                        )
                        # Get barcodes from saved fragments (order should match)
                        all_barcodes = []
                        for frag in self.config.data.fragments:
                            sample_id = self._get_frag_id_from_config(frag.path)
                            saved_frag_path = self.output_dir / "bpcells_fragments" / sample_id
                            if saved_frag_path.exists() and self._is_valid_bpcells_dir(saved_frag_path):
                                frag_barcodes = list(bpcells.cpp.cell_names_fragments_dir(str(saved_frag_path)))
                                all_barcodes.extend(frag_barcodes)

                        # Match by position (assuming order is preserved)
                        if len(all_barcodes) >= len(metadata):
                            metadata[first_col] = all_barcodes[:len(metadata)]
                            self.logger.info(f"Matched {len(metadata)} cells by position from saved fragments")
                        else:
                            raise ValueError(
                                f"Not enough barcodes in saved fragments ({len(all_barcodes)}) "
                                f"for AnnData cells ({len(metadata)}). "
                                f"Please regenerate AnnData with the fixed code."
                            )

                    if first_col != self.barcode_col:
                        metadata = metadata.rename(columns={first_col: self.barcode_col})
                    self.logger.info(f"Loaded {len(metadata)} cells from AnnData file")
                except Exception as e:
                    self.logger.warning(f"Could not load from AnnData file: {e}. Trying metadata.csv...")
                    # Fall back to CSV
                    if clustering_metadata_path.exists():
                        metadata = pd.read_csv(clustering_metadata_path, index_col=0)
                        # If index is numeric, we need to get barcodes from AnnData
                        if metadata.index.dtype in ['int64', 'int32']:
                            raise ValueError(
                                f"Clustering metadata.csv has numeric indices instead of barcodes. "
                                f"Please ensure AnnData file exists at {anndata_path} or that "
                                f"ClusteringProcessor saves barcodes in metadata.csv."
                            )
                        metadata = metadata.reset_index()
                        first_col = metadata.columns[0]
                        if first_col != self.barcode_col:
                            metadata = metadata.rename(columns={first_col: self.barcode_col})
            elif clustering_metadata_path.exists():
                # Fall back to CSV if AnnData doesn't exist
                self.logger.warning(
                    f"AnnData file not found at {anndata_path}, using metadata.csv. "
                    "Barcodes may not be correctly loaded."
                )
                metadata = pd.read_csv(clustering_metadata_path, index_col=0)
                metadata = metadata.reset_index()
                first_col = metadata.columns[0]
                if first_col != self.barcode_col:
                    metadata = metadata.rename(columns={first_col: self.barcode_col})
            else:
                self.logger.warning(
                    f"use_clustering_labels=True but neither AnnData file nor metadata.csv found. "
                    f"Checked: {anndata_path} and {clustering_metadata_path}. "
                    "Make sure clustering step has run before pseudobulk."
                )
                return None

            return metadata

        # Otherwise, use regular fragment metadata loading
        metadata = load_and_merge_metadata(
            fragments=self.config.data.fragments,
            barcode_col=self.barcode_col,
            sample_id_col=self.sample_id_col,
            logger=self.logger,
        )

        # Check if we need to load metacell assignments
        # This is needed when annotation_col is "metacell_id" and we're not using clustering labels
        if self.annotation_col == "metacell_id":
            if metadata is not None:
                metadata = self._merge_metacell_assignments(metadata)
            else:
                # Try to create metadata directly from metacell_assignments.csv
                metadata = self._load_metacell_only_metadata()

        return metadata

    def _load_metacell_only_metadata(self) -> pd.DataFrame | None:
        """Load metadata when only metacell_assignments.csv is available.

        This is used when no fragment metadata is provided but we have
        metacell assignments from the isolibsize-metacell processor.
        """
        # Search for metacell_assignments.csv
        search_paths = []

        # Check fragment paths (input BPCells stores)
        for frag in self.config.data.fragments:
            frag_path = Path(frag.path)
            search_paths.append(frag_path / "metacell_assignments.csv")

        # Check output directory
        search_paths.extend([
            self.output_dir / "metacell_assignments.csv",
            self.output_dir / "bpcells_single_sparse" / "metacell_assignments.csv",
        ])

        # Find the metacell assignments file
        metacell_csv = None
        for path in search_paths:
            if path.exists():
                metacell_csv = path
                break

        if metacell_csv is None:
            self.logger.warning(
                f"annotation_col='metacell_id' but metacell_assignments.csv not found. "
                f"Searched in: {search_paths[:3]}..."
            )
            return None

        self.logger.info(f"Loading metadata from metacell assignments: {metacell_csv}")

        # Load metacell assignments as the base metadata
        metadata = pd.read_csv(metacell_csv)

        # Standardize column names
        if "barcode" in metadata.columns and self.barcode_col not in metadata.columns:
            metadata = metadata.rename(columns={"barcode": self.barcode_col})

        return metadata

    def _merge_metacell_assignments(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Merge metacell assignments into metadata.

        Searches for metacell_assignments.csv in several locations:
        1. BPCells store path
        2. Output directory
        3. Saved BPCells fragments path
        """
        # Search locations for metacell_assignments.csv
        search_paths = []

        # Check saved fragments location
        saved_frag_dir = self.output_dir / "bpcells_fragments"
        if saved_frag_dir.exists():
            # Check each sample's directory
            for sample_dir in saved_frag_dir.iterdir():
                if sample_dir.is_dir():
                    search_paths.append(sample_dir / "metacell_assignments.csv")
            # Also check at the fragments level
            search_paths.append(saved_frag_dir / "metacell_assignments.csv")

        # Check output directory
        search_paths.extend([
            self.output_dir / "metacell_assignments.csv",
            self.output_dir / "bpcells_single_sparse" / "metacell_assignments.csv",
            self.output_dir / "single_sparse" / "metacell_assignments.csv",
        ])

        # Find the metacell assignments file
        metacell_csv = None
        for path in search_paths:
            if path.exists():
                metacell_csv = path
                break

        if metacell_csv is None:
            self.logger.warning(
                f"annotation_col='metacell_id' but metacell_assignments.csv not found. "
                f"Searched in: {search_paths[:3]}... "
                f"Run isolibsize-metacell processor first or specify a different annotation_col."
            )
            return metadata

        self.logger.info(f"Loading metacell assignments from {metacell_csv}")

        # Load and merge metacell assignments
        metacell_df = pd.read_csv(metacell_csv)

        # Standardize column names
        if "barcode" in metacell_df.columns:
            metacell_df = metacell_df.rename(columns={"barcode": self.barcode_col})

        if self.barcode_col not in metacell_df.columns:
            self.logger.warning(
                f"Barcode column '{self.barcode_col}' not found in metacell_assignments.csv. "
                f"Available columns: {list(metacell_df.columns)}"
            )
            return metadata

        # Merge with metadata
        n_before = len(metadata)
        merged = metadata.merge(
            metacell_df[[self.barcode_col, "metacell_id"]],
            on=self.barcode_col,
            how="left"
        )

        # Count cells with assignments
        n_assigned = merged["metacell_id"].notna().sum()
        self.logger.info(
            f"Merged metacell assignments: {n_assigned}/{n_before} cells have assignments"
        )

        if n_assigned == 0:
            self.logger.warning(
                "No cells matched between metadata and metacell_assignments.csv. "
                "Check that barcodes match between the files."
            )

        return merged

    def _save_attributes(self) -> None:
        """Save BPCells attributes file."""
        attrs = {
            "class": f"{self.output_type.title().replace('_', '')}BPCellsIO",
            "assembly": self._resolve_assembly(),
            "created_by": "caesar_bpcells_processor",
        }

        if self.output_type == "celltype_dense":
            # Load celltypes from the matrix
            try:
                from bpcells.experimental import PrecalculatedInsertionMatrix

                matrix = PrecalculatedInsertionMatrix(str(self.output_path))
                attrs["celltypes"] = matrix.group_names
            except Exception as e:
                self.logger.warning(f"Could not load celltypes for attrs.json: {e}")

        # Ensure output directory exists (should exist after precalculate_insertion_counts, but check to be safe)
        if not self.output_path.exists():
            self.logger.warning(
                f"Output directory {self.output_path} does not exist. "
                "Creating it now. This may indicate an issue with the processing step."
            )
            self.output_path.mkdir(parents=True, exist_ok=True)

        attrs_path = self.output_path / "attrs.json"
        try:
            with open(attrs_path, "w") as f:
                json.dump(attrs, f, indent=4)
            self.logger.info(f"Saved attributes to {attrs_path} (class: {attrs['class']}, assembly: {attrs['assembly']})")
        except Exception as e:
            self.logger.error(
                f"Failed to save attributes to {attrs_path}: {e}. "
                "The output directory may be missing or permissions may be insufficient."
            )
            raise

    def _save_single_sparse_metadata(self, cell_barcodes: list[str]) -> None:
        """Save metadata for single sparse format.

        The cell_barcodes should match the group_names passed to precalculate_insertion_counts,
        which will be used as self.ids by SingleSparseBPCellsIO. This ensures alignment.
        """
        # Create basic metadata
        metadata = pd.DataFrame(
            {
                "barcodes": cell_barcodes,  # These should match group_names from precalculate_insertion_counts
                "cell_idx": range(len(cell_barcodes)),
            }
        )

        # If we used prefixed barcodes, add original_barcode and sample_id columns for tracking
        if hasattr(self, '_prefixed_to_original') and self._prefixed_to_original:
            original_barcodes = []
            sample_ids = []
            for prefixed_bc in cell_barcodes:
                if prefixed_bc in self._prefixed_to_original:
                    orig_bc, sid = self._prefixed_to_original[prefixed_bc]
                    original_barcodes.append(orig_bc)
                    sample_ids.append(sid)
                else:
                    # Not a prefixed barcode (shouldn't happen, but handle gracefully)
                    original_barcodes.append(prefixed_bc)
                    sample_ids.append(None)
            metadata["original_barcode"] = original_barcodes
            metadata["sample_id"] = sample_ids

        # Add any additional metadata if available
        all_metadata = self._load_and_merge_metadata()
        if all_metadata is not None:
            # Merge on barcodes (handle flexible column names)
            # If we have prefixed barcodes, try to match on original_barcode first
            merge_col = "original_barcode" if "original_barcode" in metadata.columns else "barcodes"
            if self.barcode_col in all_metadata.columns:
                # If merging on original_barcode, we need to handle sample_id matching too
                if merge_col == "original_barcode" and self.sample_id_col in all_metadata.columns:
                    # Merge on both original_barcode and sample_id for accurate matching
                    metadata = metadata.merge(
                        all_metadata,
                        left_on=["original_barcode", "sample_id"],
                        right_on=[self.barcode_col, self.sample_id_col],
                        how="left",
                    )
                else:
                    metadata = metadata.merge(
                        all_metadata,
                        left_on=merge_col,
                        right_on=self.barcode_col,
                        how="left",
                    )
            else:
                self.logger.warning(
                    f"Barcode column '{self.barcode_col}' not found in metadata, "
                    "skipping metadata merge"
                )

        # Save files
        metadata.to_csv(self.output_path / "metadata.csv", index=False)

        # Save barcodes file
        with open(self.output_path / "barcodes.tsv", "w") as f:
            for bc in cell_barcodes:
                f.write(f"{bc}\n")

        # Note: Library size filtering is handled in _process_single_sparse before this method is called

    def _save_normalization_factors(self) -> None:
        """Save normalization factors if specified.

        For BPCells, normalization factors are stored but not applied to the data.
        This maintains the raw count nature of the data while allowing downstream
        normalization when needed.
        """
        if not Path(self.normalization_factor_file).exists():
            self.logger.warning(
                f"Normalization factor file not found: {self.normalization_factor_file}"
            )
            return

        # Copy normalization factors to output directory
        import shutil

        dest = self.output_path / "normalization_factors.json"
        shutil.copy(self.normalization_factor_file, dest)
        self.logger.info(f"Saved normalization factors to {dest}")
