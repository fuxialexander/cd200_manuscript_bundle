"""Multiome processor for paired ATAC + RNA data.

This processor handles the integration of ATAC-seq and RNA-seq data from the same cells,
ensuring barcode matching and creating unified storage structures.

Supports both Zarr and BPCells backends for ATAC storage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scanpy as sc

from caesar.io import SingleSparseZarrIO
from caesar.preprocess.core.base import PreprocessingProcessor
from caesar.preprocess.core.exceptions import ProcessingError, ValidationError
from caesar.preprocess.processor.fragment_processor import FragmentProcessor
from caesar.preprocess.processor.multiome_rna_processor import MultiomeRNAProcessor

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig


class MultiomeProcessor(PreprocessingProcessor):
    """Process paired ATAC + RNA data into multiome storage.

    This processor:
    1. Processes ATAC fragments (delegates to FragmentProcessor or BPCellsProcessor)
    2. Processes RNA data (delegates to MultiomeRNAProcessor)
    3. Matches barcodes between modalities
    4. Keeps only shared barcodes in both datasets
    5. Creates unified storage structure with aligned cell indices
    6. Optionally computes cross-modality metrics

    Supports both Zarr and BPCells backends:
    - Zarr: Creates multiome_sparse.zarr with embedded rna.h5ad
    - BPCells: Creates multiome_sparse_bpcells/ with rna.h5ad and attrs.json

    Key Design: Only shared barcodes between ATAC and RNA are kept,
    ensuring consistent cell indices across both modalities.
    """

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add multiome-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Multiome-specific arguments
        parser.add_argument(
            "--atac", nargs="+", required=True, help="ATAC fragment files"
        )
        parser.add_argument(
            "--rna",
            nargs="+",
            required=True,
            help="RNA expression files (h5ad/h5/mtx formats)",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            default="./output",
            help="Output directory (default: ./output)",
        )
        parser.add_argument(
            "--backend",
            choices=["zarr", "bpcells"],
            default="zarr",
            help="Storage backend for ATAC data (default: zarr)",
        )
        parser.add_argument(
            "--min-shared-cells",
            type=int,
            default=100,
            help="Minimum number of shared cells required (default: 100)",
        )
        parser.add_argument(
            "--barcode-suffix",
            help="Suffix to remove from barcodes for matching (e.g., '-1')",
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return (
            "Process multiome data (paired ATAC + RNA). "
            "Creates integrated storage with matched barcodes. "
            "Supports both Zarr and BPCells backends."
        )

    def __init__(self, config: CaesarConfig):
        """Initialize multiome processor with full configuration.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Get multiome configuration
        self.multiome_config = getattr(config, "multiome", None)
        if not self.multiome_config:
            # Create default config if not present
            from caesar.preprocess.core.config import MultiomeConfig

            self.multiome_config = MultiomeConfig()

        # Extract config parameters
        self.min_shared_cells = getattr(self.multiome_config, "min_shared_cells", 100)
        self.barcode_suffix = getattr(self.multiome_config, "barcode_suffix", None)
        self.backend = getattr(self.multiome_config, "backend", "zarr")

        # Initialize RNA processor (same for both backends)
        self.rna_processor = MultiomeRNAProcessor(config)

        # Initialize ATAC processor and paths based on backend
        if self.backend == "bpcells":
            self._init_bpcells_backend(config)
        else:
            self._init_zarr_backend(config)

        # RNA goes directly into multiome directory
        self.rna_output_path = self.multiome_output_path / "rna.h5ad"

        # Log paths for debugging
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Backend: {self.backend}")
        self.logger.info(f"Multiome output path: {self.multiome_output_path}")
        self.logger.info(f"RNA output path: {self.rna_output_path}")

    def _init_zarr_backend(self, config: "CaesarConfig") -> None:
        """Initialize Zarr backend for ATAC processing."""
        self.fragment_processor = FragmentProcessor(config)
        self.bpcells_processor = None
        self.multiome_output_path = self.output_dir / "multiome_sparse.zarr"

    def _init_bpcells_backend(self, config: "CaesarConfig") -> None:
        """Initialize BPCells backend for ATAC processing."""
        # Import BPCellsProcessor only when needed
        try:
            from caesar.preprocess.processor.bpcells_processor import BPCellsProcessor
        except ImportError as e:
            raise ImportError(
                "BPCells backend requires the bpcells package. "
                "Install with: pip install bpcells"
            ) from e

        self.fragment_processor = None
        self.bpcells_processor = BPCellsProcessor(config)
        self.multiome_output_path = self.output_dir / "multiome_sparse_bpcells"

    def validate_config(self) -> None:
        """Validate multiome-specific configuration."""
        # Validate fragment files
        if not self.config.data.fragments:
            raise ValidationError("No ATAC fragment files specified")

        # Validate RNA files
        if not self.config.data.rna:
            raise ValidationError("No RNA expression files specified")

        # Validate sub-processor configs based on backend
        if self.backend == "bpcells":
            self.bpcells_processor.validate_inputs()
        else:
            self.fragment_processor.validate_config()
        self.rna_processor.validate_config()

        # Check minimum shared cells threshold
        if self.min_shared_cells < 0:
            raise ValidationError("min_shared_cells must be >= 0")

    def process(self) -> dict[str, Any]:
        """Process multiome data.

        Returns:
            Processing results dictionary with keys:
            - success: bool
            - n_shared_cells: int
            - atac_cells: int
            - rna_cells: int
            - output_path: str
            - runtime: float
            - backend: str
        """
        self._start_processing()

        try:
            # Step 1: Process ATAC data
            self.logger.info(f"Processing ATAC fragments with {self.backend} backend...")
            atac_result = self._process_atac()

            # Step 2: Process RNA data
            self.logger.info("Processing RNA expression data...")
            rna_result = self._process_rna()

            # Step 3: Match barcodes between modalities
            self.logger.info("Matching barcodes between ATAC and RNA...")
            shared_barcodes = self._match_barcodes()

            # Step 4: Filter to shared cells and create unified storage
            self.logger.info(
                f"Creating multiome {self.backend} storage with {len(shared_barcodes)} shared cells..."
            )
            self._finalize_multiome(shared_barcodes)

            runtime = self._end_processing()

            return {
                "success": True,
                "n_shared_cells": len(shared_barcodes),
                "atac_cells": atac_result.get("n_cells", 0),
                "rna_cells": rna_result.get("n_cells", 0),
                "output_path": str(self.multiome_output_path),
                "rna_path": str(self.rna_output_path),
                "backend": self.backend,
                "runtime": runtime,
            }

        except Exception as e:
            self.logger.error(f"Multiome processing failed: {e}")
            raise ProcessingError(f"Multiome processing failed: {e}") from e

    def _process_atac(self) -> dict[str, Any]:
        """Process ATAC fragments directly to multiome output path.

        Returns:
            Processing results from FragmentProcessor or BPCellsProcessor
        """
        if self.backend == "bpcells":
            return self._process_atac_bpcells()
        else:
            return self._process_atac_zarr()

    def _process_atac_zarr(self) -> dict[str, Any]:
        """Process ATAC fragments to Zarr format."""
        # Check if multiome ATAC data already exists
        if self.multiome_output_path.exists():
            self.logger.info(
                f"Found existing multiome ATAC data at: {self.multiome_output_path}"
            )
            io = SingleSparseZarrIO(str(self.multiome_output_path), mode="r")
            return {"n_cells": len(io.barcodes), "reused_existing": True}

        # Process ATAC data directly to multiome output path
        self.logger.info(f"Processing ATAC data to: {self.multiome_output_path}")

        # Temporarily override output path
        original_output = self.fragment_processor.output_dir
        self.fragment_processor.output_dir = self.multiome_output_path.parent

        self.logger.info(
            f"Fragment processor output dir set to: {self.fragment_processor.output_dir}"
        )

        # Configure for single-cell sparse output with multiome name
        self.fragment_processor.fragment_config.output = self.multiome_output_path.name

        self.logger.info(
            f"Fragment processor output file: {self.fragment_processor.fragment_config.output}"
        )

        try:
            result = self.fragment_processor.process()

            # Check if the file was created
            if not self.multiome_output_path.exists():
                raise ProcessingError(
                    f"ATAC processing completed but output not found at {self.multiome_output_path}"
                )

            # Get cell count from zarr
            io = SingleSparseZarrIO(str(self.multiome_output_path), mode="r")
            result["n_cells"] = len(io.barcodes)

            return result
        finally:
            # Restore original output path
            self.fragment_processor.output_dir = original_output

    def _process_atac_bpcells(self) -> dict[str, Any]:
        """Process ATAC fragments to BPCells format."""
        from caesar.io.bpcell_io import SingleSparseBPCellsIO

        # Check if multiome ATAC data already exists
        if self.multiome_output_path.exists():
            self.logger.info(
                f"Found existing multiome ATAC data at: {self.multiome_output_path}"
            )
            try:
                io = SingleSparseBPCellsIO(str(self.multiome_output_path), mode="r")
                return {"n_cells": len(io.ids), "reused_existing": True}
            except Exception as e:
                self.logger.warning(f"Could not read existing BPCells: {e}, reprocessing...")

        # Process ATAC data directly to multiome output path
        self.logger.info(f"Processing ATAC data to BPCells: {self.multiome_output_path}")

        # Configure BPCells processor for single_sparse output
        self.bpcells_processor.output_path = self.multiome_output_path
        self.bpcells_processor.output_type = "single_sparse"

        try:
            result = self.bpcells_processor.process()

            # Check if the file was created
            if not self.multiome_output_path.exists():
                raise ProcessingError(
                    f"ATAC processing completed but output not found at {self.multiome_output_path}"
                )

            # Get cell count from BPCells
            io = SingleSparseBPCellsIO(str(self.multiome_output_path), mode="r")
            result["n_cells"] = len(io.ids)

            return result
        except Exception as e:
            raise ProcessingError(f"BPCells ATAC processing failed: {e}") from e

    def _process_rna(self) -> dict[str, Any]:
        """Process RNA data directly to multiome directory.

        Returns:
            Processing results from MultiomeRNAProcessor
        """
        # Ensure multiome directory exists (ATAC processing should have created it)
        self.multiome_output_path.mkdir(parents=True, exist_ok=True)

        # Check if RNA has already been processed
        if self.rna_output_path.exists():
            self.logger.info("RNA data already processed, loading existing file")
            adata = sc.read_h5ad(self.rna_output_path)
            return {"n_cells": adata.n_obs, "already_processed": True}

        # Configure RNA processor to write directly to multiome directory
        # Save original output directory
        original_output_dir = self.rna_processor.output_dir

        # Set RNA processor to write to multiome directory
        self.rna_processor.output_dir = self.multiome_output_path

        try:
            # Process RNA data
            result = self.rna_processor.process()

            # Count cells in output
            if self.rna_output_path.exists():
                adata = sc.read_h5ad(self.rna_output_path)
                result["n_cells"] = adata.n_obs

            return result
        finally:
            # Restore original output directory
            self.rna_processor.output_dir = original_output_dir

    def _match_barcodes(self) -> list[str]:
        """Match barcodes between ATAC and RNA modalities.

        Returns:
            Ordered list of shared barcodes

        Raises:
            ProcessingError: If too few shared cells found
        """
        # Load ATAC barcodes based on backend
        if self.backend == "bpcells":
            from caesar.io.bpcell_io import SingleSparseBPCellsIO
            atac_io = SingleSparseBPCellsIO(str(self.multiome_output_path), mode="r")
            atac_barcodes = set(atac_io.ids)
            atac_barcode_list = list(atac_io.ids)
        else:
            atac_io = SingleSparseZarrIO(str(self.multiome_output_path), mode="r")
            atac_barcodes = set(atac_io.barcodes)
            atac_barcode_list = list(atac_io.barcodes)

        # Load RNA barcodes
        rna_adata = sc.read_h5ad(self.rna_output_path)
        rna_barcodes = set(rna_adata.obs.index)

        self.logger.info(
            f"ATAC cells: {len(atac_barcodes)}, RNA cells: {len(rna_barcodes)}"
        )

        # Handle barcode suffix if specified
        if self.barcode_suffix:
            # Create mapping for suffix removal
            atac_cleaned = {bc.rstrip(self.barcode_suffix): bc for bc in atac_barcodes}
            rna_cleaned = {bc.rstrip(self.barcode_suffix): bc for bc in rna_barcodes}

            # Find shared cleaned barcodes
            shared_cleaned = set(atac_cleaned.keys()) & set(rna_cleaned.keys())

            # Map back to original barcodes
            shared_barcodes = set()
            for cleaned_bc in shared_cleaned:
                if cleaned_bc in atac_cleaned and cleaned_bc in rna_cleaned:
                    # Use ATAC barcode as canonical
                    shared_barcodes.add(atac_cleaned[cleaned_bc])
        else:
            # Direct matching
            shared_barcodes = atac_barcodes & rna_barcodes

        # Maintain order from ATAC barcodes
        shared_barcode_list = [bc for bc in atac_barcode_list if bc in shared_barcodes]

        self.logger.info(
            f"Found {len(shared_barcode_list)} shared barcodes "
            f"({len(shared_barcode_list) / len(atac_barcodes) * 100:.1f}% of ATAC, "
            f"{len(shared_barcode_list) / len(rna_barcodes) * 100:.1f}% of RNA)"
        )

        # Check minimum threshold
        if len(shared_barcode_list) < self.min_shared_cells:
            raise ProcessingError(
                f"Too few shared cells: {len(shared_barcode_list)} < {self.min_shared_cells}. "
                f"Check barcode formats or adjust min_shared_cells threshold."
            )

        return shared_barcode_list

    def _finalize_multiome(self, shared_barcodes: list[str]) -> None:
        """Finalize multiome storage with RNA data and barcode mapping.

        Since ATAC was processed directly to multiome_output_path, we just need to:
        1. Add rna.h5ad with full RNA data (already done)
        2. Add barcode_mapping.csv for shared cell correspondence
        3. Update storage attributes with multiome metadata

        Args:
            shared_barcodes: Ordered list of shared barcodes
        """
        if self.backend == "bpcells":
            self._finalize_multiome_bpcells(shared_barcodes)
        else:
            self._finalize_multiome_zarr(shared_barcodes)

    def _finalize_multiome_zarr(self, shared_barcodes: list[str]) -> None:
        """Finalize multiome Zarr storage."""
        import zarr

        # Get barcode mappings for both modalities
        self.logger.info("Creating barcode index mappings...")

        # ATAC barcode mapping
        atac_io = SingleSparseZarrIO(str(self.multiome_output_path), mode="r")
        atac_barcodes = list(atac_io.barcodes)
        atac_bc_to_idx = {bc: i for i, bc in enumerate(atac_barcodes)}

        # RNA barcode mapping
        rna_adata = sc.read_h5ad(self.rna_output_path)
        rna_barcodes = list(rna_adata.obs.index)
        rna_bc_to_idx = {bc: i for i, bc in enumerate(rna_barcodes)}

        # Create mapping arrays for shared cells
        n_shared = len(shared_barcodes)
        atac_indices = np.array(
            [atac_bc_to_idx.get(bc, -1) for bc in shared_barcodes], dtype=np.int32
        )
        rna_indices = np.array(
            [rna_bc_to_idx.get(bc, -1) for bc in shared_barcodes], dtype=np.int32
        )

        # Create and save mapping structure
        mapping_data = pd.DataFrame(
            {
                "shared_idx": np.arange(n_shared),
                "barcode": shared_barcodes,
                "atac_idx": atac_indices,
                "rna_idx": rna_indices,
            }
        )
        mapping_path = self.multiome_output_path / "barcode_mapping.csv"
        mapping_data.to_csv(mapping_path, index=False)
        self.logger.info(f"Saved barcode mapping to {mapping_path}")

        # RNA data is already in the multiome directory
        self.logger.info(f"RNA data already at {self.rna_output_path}")

        # Update zarr attributes with multiome metadata
        zarr_root = zarr.open(str(self.multiome_output_path), mode="a")
        zarr_root.attrs.update(
            {
                "multiome": True,
                "n_shared_cells": n_shared,
                "atac_n_cells": len(atac_barcodes),
                "rna_n_cells": len(rna_barcodes),
                "has_rna": True,
                "rna_file": "rna.h5ad",
                "mapping_file": "barcode_mapping.csv",
            }
        )
        self.logger.info("Updated zarr attributes with multiome metadata")

    def _finalize_multiome_bpcells(self, shared_barcodes: list[str]) -> None:
        """Finalize multiome BPCells storage."""
        from caesar.io.bpcell_io import SingleSparseBPCellsIO

        # Get barcode mappings for both modalities
        self.logger.info("Creating barcode index mappings...")

        # ATAC barcode mapping
        atac_io = SingleSparseBPCellsIO(str(self.multiome_output_path), mode="r")
        atac_barcodes = list(atac_io.ids)
        atac_bc_to_idx = {bc: i for i, bc in enumerate(atac_barcodes)}

        # RNA barcode mapping
        rna_adata = sc.read_h5ad(self.rna_output_path)
        rna_barcodes = list(rna_adata.obs.index)
        rna_bc_to_idx = {bc: i for i, bc in enumerate(rna_barcodes)}

        # Create mapping arrays for shared cells
        n_shared = len(shared_barcodes)
        atac_indices = np.array(
            [atac_bc_to_idx.get(bc, -1) for bc in shared_barcodes], dtype=np.int32
        )
        rna_indices = np.array(
            [rna_bc_to_idx.get(bc, -1) for bc in shared_barcodes], dtype=np.int32
        )

        # Create and save mapping structure
        mapping_data = pd.DataFrame(
            {
                "shared_idx": np.arange(n_shared),
                "barcode": shared_barcodes,
                "atac_idx": atac_indices,
                "rna_idx": rna_indices,
            }
        )
        mapping_path = self.multiome_output_path / "barcode_mapping.csv"
        mapping_data.to_csv(mapping_path, index=False)
        self.logger.info(f"Saved barcode mapping to {mapping_path}")

        # RNA data is already in the multiome directory
        self.logger.info(f"RNA data already at {self.rna_output_path}")

        # Update attrs.json with multiome metadata
        attrs_path = self.multiome_output_path / "attrs.json"

        # Load existing attrs or create new
        if attrs_path.exists():
            with open(attrs_path) as f:
                attrs = json.load(f)
        else:
            attrs = {}

        # Update with multiome metadata
        attrs.update(
            {
                "multiome": True,
                "n_shared_cells": n_shared,
                "atac_n_cells": len(atac_barcodes),
                "rna_n_cells": len(rna_barcodes),
                "has_rna": True,
                "rna_file": "rna.h5ad",
                "mapping_file": "barcode_mapping.csv",
            }
        )

        # Write updated attrs
        with open(attrs_path, "w") as f:
            json.dump(attrs, f, indent=2)

        self.logger.info("Updated BPCells attrs.json with multiome metadata")

    def get_input_files(self) -> list[Path]:
        """Get all input files (ATAC + RNA).

        Returns:
            List of input file paths
        """
        files = []

        # Add fragment files based on backend
        if self.backend == "bpcells" and self.bpcells_processor:
            # BPCells processor has fragment paths in config
            if self.config.data.fragments:
                for frag in self.config.data.fragments:
                    if isinstance(frag, str):
                        files.append(Path(frag))
                    elif isinstance(frag, dict):
                        files.append(Path(frag.get("path", frag.get("input", ""))))
        elif self.fragment_processor:
            files.extend(self.fragment_processor.get_input_files())

        # Add RNA files
        if self.config.data.rna:
            for rna_file in self.config.data.rna:
                if isinstance(rna_file, str):
                    files.append(Path(rna_file))
                elif isinstance(rna_file, dict):
                    files.append(Path(rna_file["path"]))

        return files
