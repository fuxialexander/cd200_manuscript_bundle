"""Iso-libsize metacell processor.

This processor creates metacells with uniform ATAC library size using
geometric sketching in embedding space. It can be used as an alternative
to clustering-based pseudobulk aggregation.

Works with both:
- Multiome data: Uses RNA embeddings (X_pca) from the linked h5ad
- Standard single-cell ATAC: Uses LSI embeddings (X_svd) from tile-based AnnData
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from caesar.preprocess.core.base import TransformationProcessor
from caesar.preprocess.core.exceptions import ValidationError

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig, IsoLibsizeMetacellConfig

logger = logging.getLogger(__name__)


class IsoLibsizeMetacellProcessor(TransformationProcessor):
    """Generate iso-libsize metacells from single-cell data.

    This processor uses the IsoLibsizeSampler to create balanced metacells
    with approximately equal ATAC sequencing depth. The metacell assignments
    are saved to the zarr metadata for use by downstream pseudobulk processors.

    Key features:
    - Even coverage across cell state space (via geometric sketching)
    - Consistent ATAC sequencing depth per metacell (iso-libsize)
    - Compatible with both multiome and standard ATAC data

    Example usage:
        ```bash
        caesar isolibsize-metacell -i multiome_sparse.zarr \
            --target-libsize 5000000 \
            --n-metacells 100
        ```

    Or in a workflow YAML:
        ```yaml
        isolibsize_metacell:
          enabled: true
          target_libsize: 5000000
          embedding_key: "X_pca"
        ```
    """

    @classmethod
    def get_command_name(cls) -> str:
        """Return the CLI command name."""
        return "isolibsize-metacell"

    @classmethod
    def get_description(cls) -> str:
        """Return the CLI description."""
        return "Generate iso-libsize metacells using geometric sketching"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add processor-specific CLI arguments."""
        super().add_cli_args(parser)

        # Input/output
        parser.add_argument(
            "-i",
            "--input",
            required=False,
            help="Input sparse zarr (multiome_sparse.zarr or single_sparse.zarr)",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            help="Output directory for CSV files (default: same as input zarr)",
        )
        parser.add_argument(
            "--adata-path",
            help="Path to AnnData with embeddings (for non-multiome data)",
        )

        # Sampling parameters
        parser.add_argument(
            "--target-libsize",
            type=int,
            default=5_000_000,
            help="Target ATAC library size per metacell (default: 5000000)",
        )
        parser.add_argument(
            "--n-metacells",
            type=int,
            default=None,
            help="Number of metacells (auto if not specified)",
        )
        parser.add_argument(
            "--embedding-key",
            default="X_pca",
            help="Embedding key in adata.obsm (default: X_pca)",
        )
        parser.add_argument(
            "--n-neighbors",
            type=int,
            default=50,
            help="KNN neighbors for expansion (default: 50)",
        )
        parser.add_argument(
            "--min-libsize-fraction",
            type=float,
            default=0.95,
            help="Minimum fraction of target libsize to accept (default: 0.95)",
        )
        parser.add_argument(
            "--max-cells-per-metacell",
            type=int,
            default=2000,
            help="Maximum cells per metacell (default: 2000)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed (default: 42)",
        )
        parser.add_argument(
            "--metacell-column",
            default="metacell_id",
            help="Column name for metacell assignments (default: metacell_id)",
        )
        # Advanced options (also available in YAML config)
        parser.add_argument(
            "--use-shared-cells-only",
            action="store_true",
            default=None,
            help="For multiome: only use cells with both ATAC and RNA (default: True)",
        )
        parser.add_argument(
            "--tolerance",
            type=float,
            default=0.05,
            help="Tolerance for target libsize matching (default: 0.05)",
        )
        parser.add_argument(
            "--groupby",
            default=None,
            help="Alias for --metacell-column (for consistency with pseudobulk)",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing metacell assignments",
        )

    def __init__(self, config: CaesarConfig):
        """Initialize the processor.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        self.config = config
        self.iso_config: IsoLibsizeMetacellConfig = config.isolibsize_metacell

        # Extract configuration
        self.target_libsize = self.iso_config.target_libsize
        self.n_metacells = self.iso_config.n_metacells
        self.embedding_key = self.iso_config.embedding_key
        self.n_neighbors = self.iso_config.n_neighbors
        self.min_libsize_fraction = self.iso_config.min_libsize_fraction
        self.max_cells_per_metacell = self.iso_config.max_cells_per_metacell
        self.seed = self.iso_config.seed
        self.use_shared_cells_only = self.iso_config.use_shared_cells_only
        self.tolerance = self.iso_config.tolerance
        self.metacell_column = self.iso_config.metacell_column
        self.adata_path = self.iso_config.adata_path
        self.overwrite = self.iso_config.overwrite

        self.validate_config()

    def validate_config(self) -> None:
        """Validate configuration."""
        if self.target_libsize <= 0:
            raise ValidationError("target_libsize must be positive")
        if self.n_neighbors <= 0:
            raise ValidationError("n_neighbors must be positive")
        if not 0 < self.min_libsize_fraction <= 1:
            raise ValidationError("min_libsize_fraction must be in (0, 1]")
        if self.max_cells_per_metacell <= 0:
            raise ValidationError("max_cells_per_metacell must be positive")

    def _detect_store_type(self, input_path: Path) -> str:
        """Detect the type of sparse store.

        Returns:
            Store type: 'multiome_sparse', 'single_sparse', 'multi_sparse',
                       'bpcells_single_sparse', 'bpcells_multi_sparse',
                       'bpcells_fragments'
        """
        # First try zarr detection
        from caesar.io.zarr_io.base import ZarrIO

        store_type = ZarrIO.detect_store_type(input_path)

        if store_type == "multiome_sparse":
            self.logger.info(f"Detected multiome sparse zarr at {input_path}")
            return "multiome_sparse"
        elif store_type in ["single_sparse", "multi_sparse"]:
            self.logger.info(f"Detected {store_type} zarr at {input_path}")
            return store_type

        # Try BPCells detection
        try:
            from caesar.io.bpcell_io import BPCellsIO

            bpcells_type = BPCellsIO.detect_store_type(input_path)
            # BPCells returns "bpcells_single_sparse" or "bpcells_multi_sparse"
            if bpcells_type in ["bpcells_single_sparse", "bpcells_multi_sparse"]:
                self.logger.info(f"Detected {bpcells_type} at {input_path}")
                return bpcells_type
        except Exception:
            pass

        # Try BPCells fragments format detection
        if input_path.is_dir():
            # Check for BPCells fragments marker files
            frag_markers = ["cell_names", "version", "chr_names"]
            if all((input_path / marker).exists() for marker in frag_markers):
                # Check version to distinguish from sparse matrix
                version_file = input_path / "version"
                with open(version_file) as f:
                    version_str = f.read().strip()
                if "fragments" in version_str.lower():
                    self.logger.info(f"Detected BPCells fragments at {input_path}")
                    return "bpcells_fragments"

            # Check if parent is a multi-sample fragments directory
            if (input_path / "attrs.json").exists():
                # Could be a multi-sample fragments directory
                subdirs = [d for d in input_path.iterdir() if d.is_dir()]
                if subdirs and all((d / "version").exists() for d in subdirs[:3]):
                    self.logger.info(f"Detected multi-sample BPCells fragments at {input_path}")
                    return "bpcells_fragments_multi"

        raise ValueError(
            f"Cannot determine store type for {input_path}. "
            "IsoLibsizeMetacellProcessor requires sparse zarr or BPCells stores."
        )

    def _load_multiome_io(self, input_path: Path):
        """Load MultiomeSparseZarrIO."""
        from caesar.io.zarr_io.multiome import MultiomeSparseZarrIO

        return MultiomeSparseZarrIO(str(input_path), mode="a")

    def _load_sparse_io(self, input_path: Path):
        """Load SingleSparseZarrIO or MultiSparseZarrIO."""
        from caesar.io.zarr_io import open_zarr

        return open_zarr(input_path, mode="a")

    def _load_bpcells_io(self, input_path: Path, store_type: str):
        """Load BPCells sparse IO."""
        from caesar.io.bpcell_io import SingleSparseBPCellsIO, MultiSparseBPCellsIO

        if store_type == "bpcells_single_sparse":
            return SingleSparseBPCellsIO(str(input_path), mode="a")
        else:  # bpcells_multi_sparse
            return MultiSparseBPCellsIO(str(input_path), mode="a")

    def _load_bpcells_fragments_data(self, input_path: Path, store_type: str) -> dict:
        """Load data from BPCells fragments format.

        Returns a dict with 'barcodes', 'libsizes', and 'path' info.
        """
        import json
        import bpcells.cpp

        if store_type == "bpcells_fragments_multi":
            # Multiple sample directories
            all_barcodes = []
            all_libsizes = []
            subdirs = sorted([d for d in input_path.iterdir() if d.is_dir() and (d / "version").exists()])

            for subdir in subdirs:
                # Load cell names
                cell_names = list(bpcells.cpp.cell_names_fragments_dir(str(subdir)))
                all_barcodes.extend(cell_names)

                # Load library sizes
                libsize_file = subdir / "library_size.json"
                if libsize_file.exists():
                    with open(libsize_file) as f:
                        ls_data = json.load(f)
                    if isinstance(ls_data, dict) and "library_sizes" in ls_data:
                        all_libsizes.extend(ls_data["library_sizes"])
                    elif isinstance(ls_data, list):
                        all_libsizes.extend(ls_data)
                else:
                    # No library size data, use zeros
                    all_libsizes.extend([0] * len(cell_names))

            return {
                "barcodes": np.array(all_barcodes),
                "libsizes": np.array(all_libsizes),
                "path": input_path,
                "sample_dirs": subdirs,
            }
        else:
            # Single sample directory
            cell_names = list(bpcells.cpp.cell_names_fragments_dir(str(input_path)))

            # Load library sizes
            libsize_file = input_path / "library_size.json"
            if libsize_file.exists():
                with open(libsize_file) as f:
                    ls_data = json.load(f)
                if isinstance(ls_data, dict) and "library_sizes" in ls_data:
                    libsizes = ls_data["library_sizes"]
                elif isinstance(ls_data, list):
                    libsizes = ls_data
                else:
                    libsizes = [0] * len(cell_names)
            else:
                libsizes = [0] * len(cell_names)

            return {
                "barcodes": np.array(cell_names),
                "libsizes": np.array(libsizes),
                "path": input_path,
            }

    def _load_embeddings_adata(self, io, store_type: str):
        """Load AnnData with embeddings.

        For multiome: Uses the linked RNA h5ad
        For standard ATAC: Uses external adata_path or auto-discovers

        Auto-detects appropriate embedding key based on data type:
        - Multiome: prefers X_pca (RNA embeddings)
        - ATAC-only: prefers X_svd (LSI embeddings)
        """
        import scanpy as sc

        if store_type == "multiome_sparse":
            # Use RNA embeddings from multiome
            adata = io.rna_data

            # Check for embeddings with auto-detection
            self.embedding_key = self._resolve_embedding_key(
                adata, store_type, preferred_keys=["X_pca", "X_svd"]
            )

            return adata

        else:
            # Standard ATAC (zarr or BPCells): need external AnnData
            if self.adata_path:
                adata_file = Path(self.adata_path)
            else:
                # Try to auto-discover in output directory
                # Handle both zarr and BPCells path structures
                if hasattr(io, "path"):
                    io_path = Path(io.path)
                    output_dir = io_path.parent
                elif hasattr(io, "root") and hasattr(io.root, "store"):
                    output_dir = Path(io.root.store.root).parent
                elif hasattr(io, "_path"):
                    output_dir = Path(io._path).parent
                else:
                    # Fallback to config output dir
                    output_dir = Path(self.config.output.dir)

                candidates = [
                    output_dir / "anndata" / "anndata.h5ad",
                    output_dir / "anndata.h5ad",
                    output_dir / "clustering.h5ad",
                    # BPCells-specific paths
                    output_dir / "bpcells_anndata.h5ad",
                    output_dir.parent / "anndata" / "anndata.h5ad",
                ]

                # For BPCells, also check in the store directory itself
                if store_type.startswith("bpcells_") and hasattr(io, "path"):
                    io_path = Path(io.path)
                    candidates.insert(0, io_path / "anndata.h5ad")
                    candidates.insert(1, io_path.parent / "anndata.h5ad")

                adata_file = None
                for cand in candidates:
                    if cand.exists():
                        adata_file = cand
                        break

                if adata_file is None:
                    raise ValueError(
                        "Could not find AnnData with embeddings. "
                        f"Searched in: {output_dir}. "
                        "For non-multiome data, specify --adata-path or run clustering first."
                    )

            self.logger.info(f"Loading embeddings from {adata_file}")
            adata = sc.read_h5ad(adata_file)

            # Check for embeddings with auto-detection
            # For BPCells, strip the prefix for comparison
            base_store_type = store_type.replace("bpcells_", "") if store_type.startswith("bpcells_") else store_type
            self.embedding_key = self._resolve_embedding_key(
                adata, base_store_type, preferred_keys=["X_svd", "X_pca"]
            )

            return adata

    def _resolve_embedding_key(
        self, adata, store_type: str, preferred_keys: list[str]
    ) -> str:
        """Resolve the embedding key with smart fallback.

        Args:
            adata: AnnData object
            store_type: Type of store (multiome_sparse, single_sparse, etc.)
            preferred_keys: Ordered list of preferred embedding keys to try

        Returns:
            Resolved embedding key
        """
        available_keys = list(adata.obsm.keys())

        # If configured key exists, use it
        if self.embedding_key in available_keys:
            return self.embedding_key

        # Try preferred keys in order
        for key in preferred_keys:
            if key in available_keys:
                self.logger.info(
                    f"Embedding '{self.embedding_key}' not found, "
                    f"using '{key}' instead for {store_type} data"
                )
                return key

        # No suitable embedding found
        raise ValueError(
            f"Embedding '{self.embedding_key}' not found in AnnData. "
            f"Available: {available_keys}. "
            f"For multiome data, ensure RNA processing ran with compute_embedding=True. "
            f"For ATAC-only data, run clustering first to generate X_svd embeddings."
        )

    def _discover_adata_for_bpcells(self, input_path: Path) -> Path:
        """Discover AnnData file path for BPCells backend.

        This method uses a comprehensive search strategy to find the anndata file
        containing embeddings (X_svd, X_pca) when using BPCells backend.

        The BPCells workflow stores fragments and sparse matrices separately from
        the anndata file that contains clustering results and embeddings. This
        method handles the typical directory structures:

        Standard BPCells output structure:
            output_dir/
            ├── {assembly}/
            │   ├── bpcells_fragments/     <- input_path (fragments)
            │   ├── bpcells_sparse/        <- sparse matrix
            │   ├── celltype_dense_bpcells/
            │   └── anndata/
            │       └── anndata.h5ad       <- embeddings here

        Args:
            input_path: Path to the BPCells fragments or sparse store

        Returns:
            Path to the discovered anndata file

        Raises:
            ValueError: If no anndata file with embeddings can be found
        """
        if self.adata_path:
            adata_file = Path(self.adata_path)
            if not adata_file.exists():
                raise ValueError(f"Specified --adata-path does not exist: {adata_file}")
            return adata_file

        # Build comprehensive candidate list based on BPCells directory structure
        # input_path is typically: .../output/{assembly}/bpcells_fragments
        assembly_dir = input_path.parent  # .../output/{assembly}
        output_dir = assembly_dir.parent  # .../output

        candidates = [
            # Most common: anndata subdir in assembly directory (BPCells workflow)
            assembly_dir / "anndata" / "anndata.h5ad",
            # Direct in assembly directory
            assembly_dir / "anndata.h5ad",
            assembly_dir / "clustering.h5ad",
            # In the fragments directory itself
            input_path / "anndata.h5ad",
            # Parent output directory
            output_dir / "anndata" / "anndata.h5ad",
            output_dir / "anndata.h5ad",
            # Legacy/alternative naming
            assembly_dir / "bpcells_anndata.h5ad",
            output_dir / "bpcells_anndata.h5ad",
            # Workflow output directory (config-specified)
            Path(self.config.output.dir) / "anndata" / "anndata.h5ad",
            Path(self.config.output.dir) / "anndata.h5ad",
        ]

        # Log search paths for debugging
        self.logger.debug(f"Searching for AnnData in {len(candidates)} candidate paths")

        for cand in candidates:
            if cand.exists():
                self.logger.info(f"Auto-discovered AnnData with embeddings: {cand}")
                return cand

        # Build helpful error message with searched paths
        searched_dirs = sorted(set(str(c.parent) for c in candidates))
        raise ValueError(
            f"Could not find AnnData with embeddings for BPCells backend. "
            f"Searched in: {', '.join(searched_dirs)}. "
            f"For BPCells workflows, ensure clustering ran first to generate anndata/anndata.h5ad. "
            f"Alternatively, specify --adata-path explicitly."
        )

    def _is_cli_invocation(self) -> bool:
        """Check if this processor was invoked directly via CLI.

        When a user runs `caesar isolibsize-metacell`, they clearly want to run
        the processor, so we should not check the enabled flag.
        """
        # Check if we have explicit input path from CLI (indicates direct invocation)
        input_path = self.get_input_path()
        return input_path is not None

    def _has_existing_metacells(self, input_path: Path, io, store_type: str) -> bool:
        """Check if metacell assignments already exist.

        Args:
            input_path: Path to the input store
            io: IO object (zarr or BPCells)
            store_type: Type of store

        Returns:
            True if metacells already exist, False otherwise
        """
        import json

        # Check for BPCells stores
        if store_type.startswith("bpcells_"):
            # Check for metacell_metadata.json
            metacell_json = input_path / "metacell_metadata.json"
            if metacell_json.exists():
                return True
            # Check for attrs.json with has_metacells flag
            attrs_path = input_path / "attrs.json"
            if attrs_path.exists():
                try:
                    with open(attrs_path) as f:
                        attrs = json.load(f)
                    if attrs.get("has_metacells", False):
                        return True
                except Exception:
                    pass
            return False

        # Check for zarr stores
        if hasattr(io, "dataset"):
            root = io.dataset
        elif hasattr(io, "root"):
            root = io.root
        else:
            return False

        # Check zarr attributes
        if root.attrs.get("has_metacells", False):
            return True

        # Check for metacells group
        if "metacells" in root:
            return True

        return False

    def process(self) -> dict[str, Any]:
        """Run iso-libsize metacell sampling.

        Returns:
            Processing results with metacell statistics
        """
        self._start_processing()

        # Check if enabled (skip check if run via CLI directly - user's intent is clear)
        # When run via workflow, enabled=False means skip
        if not self.iso_config.enabled and not self._is_cli_invocation():
            self.logger.info("IsoLibsizeMetacell is disabled, skipping")
            return {
                "status": "skipped",
                "reason": "isolibsize_metacell.enabled is False",
            }

        # Get input path
        input_path = self.validate_input_path(self.get_target_path())

        # Detect store type
        store_type = self._detect_store_type(input_path)

        # Handle BPCells fragments separately (different data structure)
        if store_type in ["bpcells_fragments", "bpcells_fragments_multi"]:
            return self._process_bpcells_fragments(input_path, store_type)

        # Load appropriate IO for sparse stores
        if store_type == "multiome_sparse":
            io = self._load_multiome_io(input_path)
        elif store_type.startswith("bpcells_"):
            io = self._load_bpcells_io(input_path, store_type)
        else:
            io = self._load_sparse_io(input_path)

        # Check if metacells already exist and overwrite is False
        if self._has_existing_metacells(input_path, io, store_type) and not self.overwrite:
            self.logger.info(
                "Metacell assignments already exist and overwrite=False, skipping"
            )
            runtime = self._end_processing()
            return {
                "status": "skipped",
                "input_path": str(input_path),
                "store_type": store_type,
                "message": "Metacell assignments already exist",
                "runtime": runtime,
            }

        # Load embeddings
        adata = self._load_embeddings_adata(io, store_type)

        # Run sampling
        metacells = self._run_sampling(io, adata, store_type)

        # Save metacell assignments to zarr metadata
        self._save_metacell_assignments(io, metacells, store_type)

        runtime = self._end_processing()

        # Generate summary statistics
        libsizes = [mc.total_libsize for mc in metacells]
        n_cells_list = [mc.n_cells for mc in metacells]

        return {
            "status": "success",
            "input_path": str(input_path),
            "store_type": store_type,
            "n_metacells": len(metacells),
            "target_libsize": self.target_libsize,
            "mean_libsize": float(np.mean(libsizes)),
            "std_libsize": float(np.std(libsizes)),
            "cv_libsize": float(np.std(libsizes) / np.mean(libsizes) * 100),
            "mean_cells_per_metacell": float(np.mean(n_cells_list)),
            "total_cells_assigned": sum(n_cells_list),
            "runtime": runtime,
        }

    def _process_bpcells_fragments(self, input_path: Path, store_type: str) -> dict[str, Any]:
        """Process BPCells fragments format.

        BPCells fragments don't have a sparse IO interface, so we handle them differently.
        """
        import scanpy as sc

        # Check if metacells already exist and overwrite is False
        if self._has_existing_metacells(input_path, None, store_type) and not self.overwrite:
            self.logger.info(
                "Metacell assignments already exist and overwrite=False, skipping"
            )
            runtime = self._end_processing()
            return {
                "status": "skipped",
                "input_path": str(input_path),
                "store_type": store_type,
                "message": "Metacell assignments already exist",
                "runtime": runtime,
            }

        # Load fragment data (barcodes and library sizes)
        frag_data = self._load_bpcells_fragments_data(input_path, store_type)
        cell_barcodes = frag_data["barcodes"]
        libsizes = frag_data["libsizes"]

        self.logger.info(f"Loaded {len(cell_barcodes)} cells from BPCells fragments")

        # Load embeddings from AnnData
        adata_file = self._discover_adata_for_bpcells(input_path)

        self.logger.info(f"Loading embeddings from {adata_file}")
        adata = sc.read_h5ad(adata_file)

        # Resolve embedding key
        self.embedding_key = self._resolve_embedding_key(
            adata, "single_sparse", preferred_keys=["X_svd", "X_pca"]
        )

        # Run sampling with fragment data
        metacells = self._run_sampling_with_data(
            cell_barcodes, libsizes, adata
        )

        # Save metacell assignments
        self._save_metacell_assignments_bpcells_fragments(
            input_path, metacells, cell_barcodes
        )

        runtime = self._end_processing()

        # Generate summary statistics
        mc_libsizes = [mc.total_libsize for mc in metacells]
        n_cells_list = [mc.n_cells for mc in metacells]

        return {
            "status": "success",
            "input_path": str(input_path),
            "store_type": store_type,
            "n_metacells": len(metacells),
            "target_libsize": self.target_libsize,
            "mean_libsize": float(np.mean(mc_libsizes)),
            "std_libsize": float(np.std(mc_libsizes)),
            "cv_libsize": float(np.std(mc_libsizes) / np.mean(mc_libsizes) * 100),
            "mean_cells_per_metacell": float(np.mean(n_cells_list)),
            "total_cells_assigned": sum(n_cells_list),
            "runtime": runtime,
        }

    def _run_sampling_with_data(
        self, cell_barcodes: np.ndarray, libsizes: np.ndarray, adata
    ) -> list:
        """Run sampling with pre-loaded data arrays."""
        from caesar.analysis.iso_libsize_sampler import Metacell
        from scipy.spatial import cKDTree
        from sklearn.cluster import MiniBatchKMeans

        # Match barcodes between fragments and adata
        adata_bc_to_idx = {bc: i for i, bc in enumerate(adata.obs_names)}
        valid_mask = np.array([bc in adata_bc_to_idx for bc in cell_barcodes])

        if valid_mask.sum() == 0:
            raise ValueError(
                "No matching barcodes between fragments and AnnData. "
                "Ensure the AnnData was generated from the same data."
            )

        self.logger.info(
            f"Matched {valid_mask.sum()}/{len(cell_barcodes)} barcodes"
        )

        cell_barcodes_matched = cell_barcodes[valid_mask]
        atac_indices = np.where(valid_mask)[0]
        libsizes_matched = libsizes[valid_mask]

        adata_indices = np.array([adata_bc_to_idx[bc] for bc in cell_barcodes_matched])
        embeddings = adata.obsm[self.embedding_key][adata_indices]

        # Build KNN tree
        kd_tree = cKDTree(embeddings)

        # Geometric sketch for seeds
        n_metacells = self.n_metacells
        if n_metacells is None:
            n_metacells = int(libsizes_matched.sum() / self.target_libsize)
            self.logger.info(f"Auto n_metacells={n_metacells}")

        kmeans = MiniBatchKMeans(
            n_clusters=n_metacells,
            random_state=self.seed,
            batch_size=min(1000, len(embeddings)),
            n_init=3,
        )
        kmeans.fit(embeddings)
        centers = kmeans.cluster_centers_
        _, seed_indices = kd_tree.query(centers, k=1)
        seed_indices = seed_indices.flatten()

        assigned = np.zeros(len(cell_barcodes_matched), dtype=bool)
        metacells = []
        min_libsize = int(self.target_libsize * self.min_libsize_fraction)

        for seed_idx in seed_indices:
            if assigned[seed_idx]:
                # Find alternative seed
                _, candidates = kd_tree.query(embeddings[seed_idx], k=100)
                for cand in candidates:
                    if not assigned[cand]:
                        seed_idx = cand
                        break
                else:
                    continue

            # Expand to target
            mc_indices, mc_libsize = self._expand_metacell(
                seed_idx, assigned, embeddings, libsizes_matched,
                kd_tree, self.max_cells_per_metacell, self.tolerance
            )

            if mc_libsize < min_libsize:
                continue

            assigned[mc_indices] = True
            mc = Metacell(
                seed_barcode=cell_barcodes_matched[seed_idx],
                cell_barcodes=[cell_barcodes_matched[j] for j in mc_indices],
                cell_indices=atac_indices[mc_indices],
                total_libsize=mc_libsize,
                n_cells=len(mc_indices),
            )
            metacells.append(mc)

        self.logger.info(f"Created {len(metacells)} metacells")
        return metacells

    def _expand_metacell(
        self, seed_idx, assigned, embeddings, libsizes, kd_tree, max_expansion, tolerance
    ):
        """Expand a metacell from seed to target library size."""
        n_available = len(embeddings) - assigned.sum()
        if n_available < 1:
            return np.array([seed_idx]), libsizes[seed_idx]

        min_target = self.target_libsize * (1 - tolerance)
        mean_libsize = libsizes[~assigned].mean()
        estimated_cells = int(self.target_libsize / mean_libsize * 1.5)
        query_k = min(max(estimated_cells, self.n_neighbors * 20), n_available)

        _, neighbor_indices = kd_tree.query(embeddings[seed_idx], k=query_k)
        if neighbor_indices.ndim == 0:
            neighbor_indices = np.array([neighbor_indices])

        metacell_indices = []
        total_libsize = 0

        for idx in neighbor_indices:
            if assigned[idx]:
                continue
            metacell_indices.append(idx)
            total_libsize += libsizes[idx]
            if total_libsize >= self.target_libsize:
                break
            if len(metacell_indices) >= max_expansion:
                break

        # Broader search if needed
        if total_libsize < min_target and len(metacell_indices) < n_available:
            _, all_indices = kd_tree.query(embeddings[seed_idx], k=n_available)
            for idx in all_indices:
                if idx in metacell_indices or assigned[idx]:
                    continue
                metacell_indices.append(idx)
                total_libsize += libsizes[idx]
                if total_libsize >= self.target_libsize:
                    break
                if len(metacell_indices) >= max_expansion:
                    break

        return np.array(metacell_indices), total_libsize

    def _save_metacell_assignments_bpcells_fragments(
        self, input_path: Path, metacells: list, all_barcodes: np.ndarray
    ) -> None:
        """Save metacell assignments for BPCells fragments format."""
        import json

        # Build assignment mapping
        assignments = {}
        barcode_to_metacell_idx = {}
        for i, mc in enumerate(metacells):
            metacell_id = f"metacell_{i}"
            for bc in mc.cell_barcodes:
                assignments[bc] = metacell_id
                barcode_to_metacell_idx[bc] = i

        # Create DataFrame
        assignment_df = pd.DataFrame([
            {"barcode": bc, self.metacell_column: mc_id}
            for bc, mc_id in assignments.items()
        ])

        self.logger.info(f"Saving {len(assignment_df)} metacell assignments")

        # Get store path
        store_path = input_path

        # Compute statistics
        metacell_indices = np.array([
            barcode_to_metacell_idx.get(bc, -1) for bc in all_barcodes
        ], dtype=np.int32)
        n_cells_arr = np.array([mc.n_cells for mc in metacells], dtype=np.int32)
        libsize_arr = np.array([mc.total_libsize for mc in metacells], dtype=np.int64)
        cv = float(np.std(libsize_arr) / np.mean(libsize_arr))

        # Update attrs.json if it exists
        attrs_path = store_path / "attrs.json"
        attrs = {}
        if attrs_path.exists():
            try:
                with open(attrs_path) as f:
                    attrs = json.load(f)
            except Exception:
                pass

        attrs["metacell_libsize_cv"] = cv
        attrs["has_metacells"] = True
        attrs["n_metacells"] = len(metacells)
        attrs["metacell_target_libsize"] = self.target_libsize
        attrs["metacell_column"] = self.metacell_column

        with open(attrs_path, "w") as f:
            json.dump(attrs, f, indent=2)
        self.logger.info(f"Updated attrs.json with metacell info")

        # Save metacell metadata as JSON
        metacell_metadata = {
            "cell_metacell_idx": metacell_indices.tolist(),
            "n_cells": n_cells_arr.tolist(),
            "libsizes": libsize_arr.tolist(),
            "cv": cv,
            "n_metacells": len(metacells),
            "target_libsize": self.target_libsize,
            "metacell_column": self.metacell_column,
        }
        metacell_json_path = store_path / "metacell_metadata.json"
        with open(metacell_json_path, "w") as f:
            json.dump(metacell_metadata, f, indent=2)
        self.logger.info(f"Saved metacell metadata to {metacell_json_path}")

        # Save CSV files
        csv_path = store_path / "metacell_assignments.csv"
        assignment_df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved metacell assignments to {csv_path}")

        stats_df = pd.DataFrame([
            {
                "metacell_id": f"metacell_{i}",
                "n_cells": mc.n_cells,
                "total_libsize": mc.total_libsize,
                "seed_barcode": mc.seed_barcode,
            }
            for i, mc in enumerate(metacells)
        ])
        stats_csv = store_path / "metacell_stats.csv"
        stats_df.to_csv(stats_csv, index=False)
        self.logger.info(f"Saved metacell stats to {stats_csv}")

    def _run_sampling(self, io, adata, store_type: str) -> list:
        """Run the iso-libsize sampling algorithm.

        This adapts the IsoLibsizeSampler to work with both multiome and standard ATAC.
        """
        from caesar.analysis.iso_libsize_sampler import IsoLibsizeSampler, Metacell

        if store_type == "multiome_sparse":
            # Multiome: use the standard sampler
            sampler = IsoLibsizeSampler(
                multiome_io=io,
                target_libsize=self.target_libsize,
                embedding_key=self.embedding_key,
                n_neighbors=self.n_neighbors,
                use_shared_cells_only=self.use_shared_cells_only,
                random_state=self.seed,
            )
        else:
            # Standard ATAC: need to create compatible sampler
            sampler = self._create_atac_sampler(io, adata)

        # Run sampling
        metacells = sampler.sample(
            n_metacells=self.n_metacells,
            min_libsize_fraction=self.min_libsize_fraction,
            max_cells_per_metacell=self.max_cells_per_metacell,
            tolerance=self.tolerance,
        )

        self.logger.info(f"Created {len(metacells)} metacells")
        self.logger.info(
            f"Library size CV: {np.std([m.total_libsize for m in metacells]) / np.mean([m.total_libsize for m in metacells]) * 100:.2f}%"
        )

        return metacells

    def _get_cell_barcodes(self, io) -> np.ndarray:
        """Get cell barcodes from IO object (handles both zarr and BPCells)."""
        if hasattr(io, "barcodes"):
            # Zarr sparse IO
            return np.array(io.barcodes)
        elif hasattr(io, "ids"):
            # BPCells IO
            return np.array(io.ids)
        else:
            raise ValueError("Could not find cell barcodes in IO object")

    def _get_per_cell_libsizes(self, io, cell_barcodes: np.ndarray) -> np.ndarray:
        """Get per-cell library sizes from IO object (handles both zarr and BPCells)."""
        libsizes = np.zeros(len(cell_barcodes))

        # Try BPCells format first (_per_cell_libsize is a pd.Series)
        if hasattr(io, "_per_cell_libsize") and io._per_cell_libsize is not None:
            per_cell_libsize = io._per_cell_libsize
            if isinstance(per_cell_libsize, pd.Series):
                libsizes = per_cell_libsize.values
            elif isinstance(per_cell_libsize, dict):
                libsizes = np.array([per_cell_libsize.get(bc, 0) for bc in cell_barcodes])
            else:
                libsizes = np.array(per_cell_libsize)
            return libsizes

        # Try zarr format (per_cell_libsize dict or root array)
        if hasattr(io, "per_cell_libsize") and io.per_cell_libsize:
            libsizes = np.array([io.per_cell_libsize.get(bc, 0) for bc in cell_barcodes])
        elif hasattr(io, "root") and "per_cell_libsize" in io.root:
            libsizes = io.root["per_cell_libsize"][:]
        elif hasattr(io, "dataset") and "per_cell_libsize" in io.dataset:
            libsizes = io.dataset["per_cell_libsize"][:]

        return libsizes

    def _create_atac_sampler(self, io, adata):
        """Create a sampler for standard ATAC data (non-multiome)."""
        from caesar.analysis.iso_libsize_sampler import Metacell
        from scipy.spatial import cKDTree
        from sklearn.cluster import MiniBatchKMeans

        # Get cell barcodes and library sizes (handles both zarr and BPCells)
        cell_barcodes = self._get_cell_barcodes(io)
        libsizes = self._get_per_cell_libsizes(io, cell_barcodes)

        # Match barcodes between zarr and adata
        adata_bc_to_idx = {bc: i for i, bc in enumerate(adata.obs_names)}
        valid_mask = np.array([bc in adata_bc_to_idx for bc in cell_barcodes])

        if valid_mask.sum() == 0:
            raise ValueError(
                "No matching barcodes between zarr and AnnData. "
                "Ensure the AnnData was generated from the same data."
            )

        self.logger.info(
            f"Matched {valid_mask.sum()}/{len(cell_barcodes)} barcodes between zarr and AnnData"
        )

        cell_barcodes_matched = cell_barcodes[valid_mask]
        atac_indices = np.where(valid_mask)[0]
        libsizes_matched = libsizes[valid_mask]

        adata_indices = np.array([adata_bc_to_idx[bc] for bc in cell_barcodes_matched])
        embeddings = adata.obsm[self.embedding_key][adata_indices]

        # Build KNN tree
        kd_tree = cKDTree(embeddings)

        # Create a simple sampler object
        class ATACSampler:
            def __init__(sampler_self):
                sampler_self.embeddings = embeddings
                sampler_self.libsizes = libsizes_matched
                sampler_self.cell_barcodes = cell_barcodes_matched
                sampler_self.atac_indices = atac_indices
                sampler_self.kd_tree = kd_tree
                sampler_self.target_libsize = self.target_libsize
                sampler_self.n_neighbors = self.n_neighbors
                sampler_self.random_state = self.seed

            def sample(sampler_self, n_metacells, min_libsize_fraction, max_cells_per_metacell, tolerance):
                # Geometric sketch for seeds
                if n_metacells is None:
                    n_metacells = int(sampler_self.libsizes.sum() / sampler_self.target_libsize)
                    logger.info(f"Auto n_metacells={n_metacells}")

                kmeans = MiniBatchKMeans(
                    n_clusters=n_metacells,
                    random_state=sampler_self.random_state,
                    batch_size=min(1000, len(sampler_self.embeddings)),
                    n_init=3,
                )
                kmeans.fit(sampler_self.embeddings)
                centers = kmeans.cluster_centers_
                _, seed_indices = sampler_self.kd_tree.query(centers, k=1)
                seed_indices = seed_indices.flatten()

                assigned = np.zeros(len(sampler_self.cell_barcodes), dtype=bool)
                metacells = []
                min_libsize = int(sampler_self.target_libsize * min_libsize_fraction)

                for seed_idx in seed_indices:
                    if assigned[seed_idx]:
                        _, candidates = sampler_self.kd_tree.query(
                            sampler_self.embeddings[seed_idx], k=100
                        )
                        for cand in candidates:
                            if not assigned[cand]:
                                seed_idx = cand
                                break
                        else:
                            continue

                    # Expand to target
                    mc_indices, mc_libsize = sampler_self._expand(
                        seed_idx, assigned, max_cells_per_metacell, tolerance
                    )

                    if mc_libsize < min_libsize:
                        continue

                    assigned[mc_indices] = True
                    mc = Metacell(
                        seed_barcode=sampler_self.cell_barcodes[seed_idx],
                        cell_barcodes=[sampler_self.cell_barcodes[j] for j in mc_indices],
                        cell_indices=sampler_self.atac_indices[mc_indices],
                        total_libsize=mc_libsize,
                        n_cells=len(mc_indices),
                    )
                    metacells.append(mc)

                return metacells

            def _expand(sampler_self, seed_idx, assigned, max_expansion, tolerance):
                n_available = len(sampler_self.embeddings) - assigned.sum()
                if n_available < 1:
                    return np.array([seed_idx]), sampler_self.libsizes[seed_idx]

                min_target = sampler_self.target_libsize * (1 - tolerance)
                mean_libsize = sampler_self.libsizes[~assigned].mean()
                estimated_cells = int(sampler_self.target_libsize / mean_libsize * 1.5)
                query_k = min(max(estimated_cells, sampler_self.n_neighbors * 20), n_available)

                _, neighbor_indices = sampler_self.kd_tree.query(
                    sampler_self.embeddings[seed_idx], k=query_k
                )
                if neighbor_indices.ndim == 0:
                    neighbor_indices = np.array([neighbor_indices])

                metacell_indices = []
                total_libsize = 0

                for idx in neighbor_indices:
                    if assigned[idx]:
                        continue
                    metacell_indices.append(idx)
                    total_libsize += sampler_self.libsizes[idx]
                    if total_libsize >= sampler_self.target_libsize:
                        break
                    if len(metacell_indices) >= max_expansion:
                        break

                # Broader search if needed
                if total_libsize < min_target and len(metacell_indices) < n_available:
                    _, all_indices = sampler_self.kd_tree.query(
                        sampler_self.embeddings[seed_idx], k=n_available
                    )
                    for idx in all_indices:
                        if idx in metacell_indices or assigned[idx]:
                            continue
                        metacell_indices.append(idx)
                        total_libsize += sampler_self.libsizes[idx]
                        if total_libsize >= sampler_self.target_libsize:
                            break
                        if len(metacell_indices) >= max_expansion:
                            break

                return np.array(metacell_indices), total_libsize

        return ATACSampler()

    def _save_metacell_assignments(self, io, metacells: list, store_type: str) -> None:
        """Save metacell assignments to store metadata.

        Creates a mapping of cell barcode -> metacell_id that can be used
        by downstream pseudobulk processors.

        For zarr: saves to zarr groups and attributes
        For BPCells: saves to JSON and CSV files
        """
        import json

        # Build assignment mapping
        assignments = {}
        barcode_to_metacell_idx = {}
        for i, mc in enumerate(metacells):
            metacell_id = f"metacell_{i}"
            for bc in mc.cell_barcodes:
                assignments[bc] = metacell_id
                barcode_to_metacell_idx[bc] = i

        # Create DataFrame
        assignment_df = pd.DataFrame([
            {"barcode": bc, self.metacell_column: mc_id}
            for bc, mc_id in assignments.items()
        ])

        self.logger.info(f"Saving {len(assignment_df)} metacell assignments")

        # Get all cell barcodes from IO
        all_barcodes = self._get_cell_barcodes(io)
        metacell_indices = np.array([
            barcode_to_metacell_idx.get(bc, -1) for bc in all_barcodes
        ], dtype=np.int32)

        # Compute statistics
        n_cells_arr = np.array([mc.n_cells for mc in metacells], dtype=np.int32)
        libsize_arr = np.array([mc.total_libsize for mc in metacells], dtype=np.int64)
        cv = float(np.std(libsize_arr) / np.mean(libsize_arr))

        # Get store path
        store_path = Path(io.path) if hasattr(io, "path") else Path(self.config.output.dir)

        if store_type.startswith("bpcells_"):
            # BPCells format: save to JSON and CSV files
            self._save_metacell_bpcells(
                store_path, assignment_df, metacells, metacell_indices,
                n_cells_arr, libsize_arr, cv, io
            )
        else:
            # Zarr format: save to zarr groups
            self._save_metacell_zarr(
                io, assignment_df, metacells, metacell_indices,
                n_cells_arr, libsize_arr, cv, store_path
            )

    def _save_metacell_zarr(
        self, io, assignment_df, metacells, metacell_indices,
        n_cells_arr, libsize_arr, cv, store_path
    ) -> None:
        """Save metacell assignments to zarr format."""
        import zarr

        # Get zarr root
        if hasattr(io, "dataset"):
            root = io.dataset
        elif hasattr(io, "root"):
            root = io.root
        else:
            root = zarr.open(io.path, mode="a")

        # Create metacell group
        if "metacells" in root:
            del root["metacells"]
        mc_group = root.create_group("metacells")

        # Save metacell index per cell (numeric array - zarr v3 compatible)
        cell_mc_arr = mc_group.create_array(
            "cell_metacell_idx",
            shape=metacell_indices.shape,
            dtype=metacell_indices.dtype,
            chunks=(min(10000, len(metacell_indices)),),
        )
        cell_mc_arr[:] = metacell_indices

        # Save metacell statistics as numeric arrays (zarr v3 compatible)
        n_cells_array = mc_group.create_array(
            "n_cells",
            shape=n_cells_arr.shape,
            dtype=n_cells_arr.dtype,
        )
        n_cells_array[:] = n_cells_arr

        libsize_array = mc_group.create_array(
            "libsizes",
            shape=libsize_arr.shape,
            dtype=libsize_arr.dtype,
        )
        libsize_array[:] = libsize_arr

        # Update zarr attributes
        root.attrs["metacell_libsize_cv"] = cv
        root.attrs["has_metacells"] = True
        root.attrs["n_metacells"] = len(metacells)
        root.attrs["metacell_target_libsize"] = self.target_libsize
        root.attrs["metacell_column"] = self.metacell_column

        # Get zarr path for CSV files
        if hasattr(root, "store") and hasattr(root.store, "root"):
            zarr_path = Path(str(root.store.root))
        else:
            zarr_path = store_path

        # Save CSV files
        self._save_metacell_csv_files(zarr_path, assignment_df, metacells)

    def _save_metacell_bpcells(
        self, store_path, assignment_df, metacells, metacell_indices,
        n_cells_arr, libsize_arr, cv, io
    ) -> None:
        """Save metacell assignments to BPCells format."""
        import json

        # Update attrs.json
        attrs_path = store_path / "attrs.json"
        attrs = {}
        if attrs_path.exists():
            try:
                with open(attrs_path) as f:
                    attrs = json.load(f)
            except Exception:
                pass

        attrs["metacell_libsize_cv"] = cv
        attrs["has_metacells"] = True
        attrs["n_metacells"] = len(metacells)
        attrs["metacell_target_libsize"] = self.target_libsize
        attrs["metacell_column"] = self.metacell_column

        with open(attrs_path, "w") as f:
            json.dump(attrs, f, indent=2)
        self.logger.info(f"Updated BPCells attrs.json with metacell info")

        # Save metacell metadata as JSON for fast lookup
        metacell_metadata = {
            "cell_metacell_idx": metacell_indices.tolist(),
            "n_cells": n_cells_arr.tolist(),
            "libsizes": libsize_arr.tolist(),
            "cv": cv,
            "n_metacells": len(metacells),
            "target_libsize": self.target_libsize,
            "metacell_column": self.metacell_column,
        }
        metacell_json_path = store_path / "metacell_metadata.json"
        with open(metacell_json_path, "w") as f:
            json.dump(metacell_metadata, f, indent=2)
        self.logger.info(f"Saved metacell metadata to {metacell_json_path}")

        # Save CSV files (same as zarr format)
        self._save_metacell_csv_files(store_path, assignment_df, metacells)

    def _save_metacell_csv_files(self, store_path: Path, assignment_df, metacells) -> None:
        """Save metacell assignment and stats CSV files."""
        # Save full assignments as CSV
        csv_path = store_path / "metacell_assignments.csv"
        assignment_df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved metacell assignments to {csv_path}")

        # Save stats CSV
        stats_df = pd.DataFrame([
            {
                "metacell_id": f"metacell_{i}",
                "n_cells": mc.n_cells,
                "total_libsize": mc.total_libsize,
                "seed_barcode": mc.seed_barcode,
            }
            for i, mc in enumerate(metacells)
        ])
        stats_csv = store_path / "metacell_stats.csv"
        stats_df.to_csv(stats_csv, index=False)
        self.logger.info(f"Saved metacell stats to {stats_csv}")

    def get_target_path(self) -> Path:
        """Get the target path for processing.

        Uses the standard input path resolution:
        1. Workflow-provided input_path attribute
        2. Config data.input field
        3. Processor-specific input field (isolibsize_metacell.input)
        4. Default multiome_sparse.zarr in output directory
        """
        # First check workflow/CLI-provided input
        input_path = self.get_input_path()
        if input_path:
            return input_path

        # Then check processor-specific input
        if self.iso_config.input:
            # Build path relative to output dir
            output_dir = Path(self.config.output.dir)
            return output_dir / self.iso_config.input

        # Default
        return Path(self.config.output.dir) / "multiome_sparse.zarr"
