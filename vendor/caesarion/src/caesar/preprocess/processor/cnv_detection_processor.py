"""CNV detection processor for single-cell ATAC data.

Detects copy number variations from tile count matrices using log2 ratio
analysis with GC correction and segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec

from caesar.preprocess.core.base import AnnotationProcessor

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig


class CNVDetectionProcessor(AnnotationProcessor):
    """Detect CNVs from single-cell ATAC tile counts.

    This processor analyzes tile count matrices to identify copy number
    variations (gains and losses) in single cells. It supports input from:
    - Zarr sparse stores (with existing tile counts)
    - BPCells fragment directories (regenerates tile counts)

    The algorithm uses:
    1. GC correction (binned median or linear regression)
    2. Library size normalization
    3. Log2 ratio vs pseudo-bulk reference
    4. Rolling median smoothing
    5. Joint segmentation across cells
    6. CNV state assignment (-1=loss, 0=neutral, 1=gain)
    """

    @classmethod
    def get_command_name(cls) -> str:
        """Return CLI command name."""
        return "cnv-detection"

    @classmethod
    def get_description(cls) -> str:
        """Return CLI description."""
        return "Detect copy number variations from ATAC-seq tile counts"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CNV detection-specific CLI arguments."""
        super().add_cli_args(parser)

        # Input/Output
        parser.add_argument(
            "-i",
            "--input",
            required=False,
            help="Input sparse Zarr store or BPCells fragment directory",
        )
        parser.add_argument("-o", "--output", help="Output path (optional, modifies input in-place)")

        # Tile parameters
        parser.add_argument(
            "--tile-size",
            type=int,
            help="Tile size in base pairs (default: 500)",
        )
        parser.add_argument(
            "--regenerate-tiles",
            action="store_true",
            help="Force regenerate tile counts from fragments",
        )

        # GC correction
        parser.add_argument(
            "--no-gc-correction",
            action="store_true",
            help="Disable GC bias correction",
        )
        parser.add_argument(
            "--gc-method",
            choices=["median_bin", "linear"],
            help="GC correction method (default: median_bin)",
        )
        parser.add_argument(
            "--assembly",
            help="Genome assembly for GC content (default: auto-detect from store)",
        )

        # Smoothing
        parser.add_argument(
            "--smooth-method",
            choices=["rolling", "gaussian"],
            help="Smoothing method (default: rolling)",
        )
        parser.add_argument(
            "--window-size",
            type=int,
            help="Smoothing window size in bins (default: 21)",
        )

        # Segmentation
        parser.add_argument(
            "--per-cell-segment",
            action="store_true",
            help="Use per-cell segmentation instead of joint (slower but may be more accurate)",
        )
        parser.add_argument(
            "--min-segment-size",
            type=int,
            help="Minimum segment size in bins (default: 10)",
        )

        # CNV thresholds
        parser.add_argument(
            "--loss-threshold",
            type=float,
            help="Log2 ratio threshold for calling loss (default: -0.3)",
        )
        parser.add_argument(
            "--gain-threshold",
            type=float,
            help="Log2 ratio threshold for calling gain (default: 0.3)",
        )

        # Reference
        parser.add_argument(
            "--no-iterative",
            action="store_true",
            help="Disable iterative diploid detection",
        )
        parser.add_argument(
            "--diploid-threshold",
            type=float,
            help="Max %% altered bins to consider cell diploid (default: 10.0)",
        )

        # Output
        parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CNV results")

    def __init__(self, config: CaesarConfig):
        """Initialize CNV detection processor.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        # Direct access to CNV configuration
        self.cnv_config = config.cnv_detection

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate CNV detection configuration."""
        if not self.cnv_config.enabled:
            return

        if self.cnv_config.tile_size <= 0:
            raise ValueError("tile_size must be greater than 0")

        if self.cnv_config.window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        if self.cnv_config.min_segment_size <= 0:
            raise ValueError("min_segment_size must be greater than 0")

    def _detect_input_type(self, input_path: Path) -> str:
        """Detect input type.

        Returns:
            Input type: 'zarr_sparse', 'zarr_dense', 'bpcells_fragments', 'bpcells_sparse'
        """
        # Check for Zarr sparse (has tiles/count_sparse)
        if (input_path / "tiles" / "count_sparse").exists():
            self.logger.info(f"Detected Zarr sparse store with tiles at {input_path}")
            return "zarr_sparse"

        # Check for CelltypeDense (has subdirectories with chrs/)
        # Structure: celltype_dense.zarr/cluster_0/chrs/, cluster_1/chrs/, etc.
        subdirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if subdirs:
            first_subdir = subdirs[0]
            if (first_subdir / "chrs").exists():
                self.logger.info(f"Detected CelltypeDense Zarr store at {input_path}")
                return "zarr_celltype_dense"

        # Check for Zarr dense (has data array)
        if (input_path / "data").exists() or (input_path / ".zarray").exists():
            self.logger.info(f"Detected Zarr dense store at {input_path}")
            return "zarr_dense"

        # Check for BPCells fragments
        has_chr_ptr = (input_path / "chr_ptr").exists()
        has_chr_names = (input_path / "chr_names").exists()

        if has_chr_ptr and has_chr_names:
            # Check if it's already processed (has metadata)
            if (input_path / "metadata.csv").exists():
                self.logger.info(f"Detected BPCells sparse store at {input_path}")
                return "bpcells_sparse"
            else:
                self.logger.info(f"Detected BPCells fragments at {input_path}")
                return "bpcells_fragments"

        # Check for Zarr chrs directory (sparse zarr without tiles yet)
        if (input_path / "chrs").exists():
            self.logger.info(f"Detected Zarr sparse store (no tiles) at {input_path}")
            return "zarr_sparse_no_tiles"

        raise ValueError(f"Cannot determine input type for {input_path}")

    def _load_tile_counts_zarr(
        self, store_path: Path, max_bins: int = 50000, max_cells: int = 2000
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load tile counts from Zarr sparse store.

        For memory efficiency, we aggregate (sum) adjacent bins when total exceeds max_bins.
        This preserves all count information while reducing dimensionality.
        Summing increases signal-to-noise ratio compared to sampling.

        Returns:
            (counts_df, bin_info_df) - counts matrix and bin information
        """
        from scipy.sparse import coo_matrix, vstack as sparse_vstack

        root = zarr.open(str(store_path), mode="r")

        # Get tile size
        tile_size = root.attrs.get("tile_size", 500)

        # Load tile counts per chromosome and concatenate
        tile_dir = root["tiles"]
        regions_dir = tile_dir["regions"]
        counts_dir = tile_dir["count_sparse"]

        # Get barcodes
        barcodes = root["barcodes"][:]
        if isinstance(barcodes[0], bytes):
            barcodes = [b.decode() for b in barcodes]

        n_cells = len(barcodes)

        # Subsample cells if too many
        if n_cells > max_cells:
            self.logger.info(f"Subsampling from {n_cells} to {max_cells} cells")
            np.random.seed(42)
            cell_indices = np.sort(np.random.choice(n_cells, max_cells, replace=False))
            barcodes = [barcodes[i] for i in cell_indices]
            n_cells = max_cells
        else:
            cell_indices = np.arange(n_cells)

        # Get chromosomes in order (chr1, chr2, ..., chr10, chr11, ...)
        def chrom_sort_key(x: str) -> tuple:
            chrom = x.replace("chr", "")
            if chrom.isdigit():
                return (0, int(chrom))
            else:
                return (1, chrom)

        chroms = sorted(regions_dir.keys(), key=chrom_sort_key)

        # First pass: count total bins to determine aggregation factor
        total_bins = 0
        chrom_bin_counts = {}
        for chrom in chroms:
            n_bins = len(regions_dir[chrom])
            chrom_bin_counts[chrom] = n_bins
            total_bins += n_bins

        # Determine bin aggregation factor (how many bins to sum together)
        if total_bins > max_bins:
            bin_agg_factor = (total_bins + max_bins - 1) // max_bins  # Ceiling division
            new_bin_size = tile_size * bin_agg_factor
            self.logger.info(
                f"Total bins ({total_bins}) exceeds max ({max_bins}), "
                f"aggregating {bin_agg_factor} bins → effective bin size {new_bin_size}bp"
            )
        else:
            bin_agg_factor = 1
            new_bin_size = tile_size

        # Second pass: load data with bin aggregation
        all_sparse_matrices = []
        all_bin_info = []

        for chrom in chroms:
            # Load regions (start, end pairs)
            regions = regions_dir[chrom][:]
            n_bins_orig = len(regions)

            # Calculate new number of bins after aggregation
            n_bins_new = (n_bins_orig + bin_agg_factor - 1) // bin_agg_factor

            # Load sparse counts - stored as COO format (row_idx, col_idx, value)
            counts_data = counts_dir[chrom][:]

            # Check data format
            if counts_data.ndim == 2 and counts_data.shape[1] == 3:
                # COO format: (cell_idx, tile_idx, count)
                row_indices = counts_data[:, 0].astype(np.int32)
                col_indices = counts_data[:, 1].astype(np.int32)
                values = counts_data[:, 2].astype(np.float32)

                # Filter by subsampled cells
                cell_mask = np.isin(row_indices, cell_indices)
                row_indices = row_indices[cell_mask]
                col_indices = col_indices[cell_mask]
                values = values[cell_mask]

                # Remap cell indices
                cell_old_to_new = {int(old): new for new, old in enumerate(cell_indices)}
                row_indices = np.array([cell_old_to_new[int(r)] for r in row_indices], dtype=np.int32)

                # Aggregate bins by integer division (bins 0,1,2 -> 0; bins 3,4,5 -> 1; etc.)
                if bin_agg_factor > 1:
                    col_indices = col_indices // bin_agg_factor

                # Create sparse matrix (bins x cells) - COO automatically sums duplicates
                # when converted to CSR, which gives us the bin aggregation for free!
                sparse_mat = coo_matrix(
                    (values, (col_indices, row_indices)),
                    shape=(n_bins_new, n_cells),
                ).tocsr()
            else:
                raise ValueError(f"Unknown tile count format for {chrom}: shape={counts_data.shape}")

            all_sparse_matrices.append(sparse_mat)

            # Build bin info for aggregated bins
            for i in range(n_bins_new):
                start_bin_idx = i * bin_agg_factor
                end_bin_idx = min((i + 1) * bin_agg_factor - 1, n_bins_orig - 1)
                start_pos = int(regions[start_bin_idx, 0])
                end_pos = int(regions[end_bin_idx, 1])
                all_bin_info.append(
                    {
                        "bin_id": f"{chrom}_{start_pos}_{end_pos}",
                        "seqnames": chrom,
                        "start": start_pos,
                        "end": end_pos,
                    }
                )

            self.logger.debug(f"Loaded {chrom}: {n_bins_new} bins (aggregated from {n_bins_orig})")

        # Stack all chromosomes (sparse)
        combined_sparse = sparse_vstack(all_sparse_matrices)

        # Convert to dense for CNV detection (now manageable size)
        counts_matrix = combined_sparse.toarray()

        bin_info_df = pd.DataFrame(all_bin_info)
        bin_info_df = bin_info_df.set_index("bin_id")

        counts_df = pd.DataFrame(counts_matrix, index=bin_info_df.index, columns=barcodes)

        self.logger.info(
            f"Loaded {counts_df.shape[0]} bins x {counts_df.shape[1]} cells from Zarr "
            f"(aggregated {bin_agg_factor}x: {tile_size}bp → {new_bin_size}bp effective resolution)"
        )

        return counts_df, bin_info_df

    def _load_tile_counts_celltype_dense(
        self, store_path: Path, target_bin_size: int = 50000
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load tile counts from CelltypeDense Zarr store.

        CelltypeDense stores have coverage data per cell type at base-pair resolution.
        We bin the coverage into larger tiles for CNV detection.

        Args:
            store_path: Path to celltype_dense.zarr
            target_bin_size: Target bin size in bp for CNV detection (default 50kb)

        Returns:
            (counts_df, bin_info_df) - counts matrix and bin information
        """
        root = zarr.open(str(store_path), mode="r")

        # Get cell type IDs (subdirectories)
        celltype_ids = [
            d.name for d in store_path.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (d / "chrs").exists()
        ]
        celltype_ids = sorted(celltype_ids)

        if not celltype_ids:
            raise ValueError(f"No cell type directories found in {store_path}")

        self.logger.info(f"Found {len(celltype_ids)} cell types: {celltype_ids[:5]}...")

        # Get assembly and chromosome info from first cell type
        first_ct = root[celltype_ids[0]]
        assembly = root.attrs.get("assembly", "hg38")

        # Get chromosomes from first cell type
        chrs_group = first_ct["chrs"]
        chroms = sorted(chrs_group.keys(), key=lambda x: (len(x), x))

        # Standard chromosome order
        def chrom_sort_key(x: str) -> tuple:
            chrom = x.replace("chr", "")
            if chrom.isdigit():
                return (0, int(chrom))
            return (1, chrom)

        chroms = sorted(chroms, key=chrom_sort_key)

        # Filter to standard chromosomes
        standard_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
        chroms = [c for c in chroms if c in standard_chroms]

        self.logger.info(f"Processing {len(chroms)} chromosomes")

        # Collect binned counts for all cell types
        all_bin_info = []
        all_counts = {ct: [] for ct in celltype_ids}

        for chrom in chroms:
            # Get chromosome size from first cell type
            chrom_data = chrs_group[chrom]
            chrom_size = len(chrom_data)

            # Calculate number of bins
            n_bins = (chrom_size + target_bin_size - 1) // target_bin_size

            # Build bin info
            for i in range(n_bins):
                start = i * target_bin_size
                end = min((i + 1) * target_bin_size, chrom_size)
                all_bin_info.append({
                    "bin_id": f"{chrom}_{start}_{end}",
                    "seqnames": chrom,
                    "start": start,
                    "end": end,
                })

            # Load and bin coverage for each cell type
            for ct_id in celltype_ids:
                ct_chrs = root[ct_id]["chrs"]
                if chrom not in ct_chrs:
                    # Missing chromosome - fill with zeros
                    all_counts[ct_id].extend([0] * n_bins)
                    continue

                # Load coverage data
                coverage = ct_chrs[chrom][:]

                # Bin the coverage by summing
                binned = np.zeros(n_bins)
                for i in range(n_bins):
                    start = i * target_bin_size
                    end = min((i + 1) * target_bin_size, len(coverage))
                    binned[i] = coverage[start:end].sum()

                all_counts[ct_id].extend(binned.tolist())

            self.logger.debug(f"Processed {chrom}: {n_bins} bins")

        # Build DataFrames
        bin_info_df = pd.DataFrame(all_bin_info)
        bin_info_df = bin_info_df.set_index("bin_id")

        # Build counts matrix (bins x cell_types)
        counts_matrix = np.column_stack([all_counts[ct] for ct in celltype_ids])
        counts_df = pd.DataFrame(
            counts_matrix,
            index=bin_info_df.index,
            columns=celltype_ids,
        )

        self.logger.info(
            f"Loaded {counts_df.shape[0]} bins x {counts_df.shape[1]} cell types "
            f"from CelltypeDense (bin size: {target_bin_size}bp)"
        )

        return counts_df, bin_info_df

    def _load_tile_counts_bpcells(self, store_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load/regenerate tile counts from BPCells fragments.

        Returns:
            (counts_df, bin_info_df) - counts matrix and bin information
        """
        try:
            import bpcells
        except ImportError:
            raise ImportError("bpcells is required for BPCells fragment processing")

        # Get assembly for chromosome sizes
        assembly = self.cnv_config.assembly
        if assembly == "auto":
            assembly = self.config.assembly
        if assembly == "auto":
            # Try to detect from store
            version_file = store_path / "version"
            if version_file.exists():
                # Try to read assembly from nearby config or metadata
                assembly = "hg38"  # Default fallback
                self.logger.warning(f"Could not auto-detect assembly, using {assembly}")

        # Get chromosome sizes
        from gcell.dna.genome import Genome

        genome = Genome(assembly, load_genome_seq=False)
        chrom_sizes = genome.chrom_sizes

        tile_size = self.cnv_config.tile_size

        # Open BPCells fragments
        fragments = bpcells.open_fragments_dir(str(store_path))

        # Get cell names
        cell_names = fragments.cell_names()

        # Generate tile counts using BPCells
        self.logger.info(f"Generating tile counts with tile_size={tile_size}...")

        # Use BPCells binned counting
        counts_matrix = bpcells.precalculate_insertion_counts_binned(fragments, bin_size=tile_size)

        # Build bin info from chromosome sizes
        bin_info = []
        chrom_offsets = {}
        offset = 0

        # Standard chromosome order
        standard_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

        for chrom in standard_chroms:
            if chrom not in chrom_sizes:
                continue

            chrom_size = chrom_sizes[chrom]
            n_bins = (chrom_size + tile_size - 1) // tile_size
            chrom_offsets[chrom] = offset

            for i in range(n_bins):
                start = i * tile_size
                end = min(start + tile_size, chrom_size)
                bin_info.append(
                    {
                        "bin_id": f"{chrom}_{start}_{end}",
                        "seqnames": chrom,
                        "start": start,
                        "end": end,
                    }
                )

            offset += n_bins

        bin_info_df = pd.DataFrame(bin_info)
        bin_info_df = bin_info_df.set_index("bin_id")

        # Convert counts matrix to dense
        if hasattr(counts_matrix, "toarray"):
            counts_dense = counts_matrix.toarray()
        else:
            counts_dense = np.array(counts_matrix)

        # Transpose if needed (BPCells returns cells x bins)
        if counts_dense.shape[0] == len(cell_names):
            counts_dense = counts_dense.T

        # Align with bin_info
        if counts_dense.shape[0] != len(bin_info_df):
            # Truncate or pad as needed
            n_bins = len(bin_info_df)
            if counts_dense.shape[0] > n_bins:
                counts_dense = counts_dense[:n_bins, :]
            else:
                # Pad with zeros
                pad = np.zeros((n_bins - counts_dense.shape[0], counts_dense.shape[1]))
                counts_dense = np.vstack([counts_dense, pad])

        counts_df = pd.DataFrame(counts_dense, index=bin_info_df.index, columns=cell_names)

        self.logger.info(f"Generated {counts_df.shape[0]} bins x {counts_df.shape[1]} cells from BPCells")

        return counts_df, bin_info_df

    def _compute_gc_content(self, bin_info_df: pd.DataFrame, assembly: str) -> np.ndarray:
        """Compute GC content for each bin.

        Args:
            bin_info_df: Bin information with seqnames, start, end columns
            assembly: Genome assembly name

        Returns:
            Array of GC content values (0-1) for each bin
        """
        try:
            from gcell.dna.genome import Genome

            genome = Genome(assembly, load_genome_seq=True)
        except Exception as e:
            self.logger.warning(f"Could not load genome sequence: {e}")
            self.logger.warning("Using uniform GC content (0.5) as fallback")
            return np.full(len(bin_info_df), 0.5)

        gc_content = np.zeros(len(bin_info_df))

        for i, (idx, row) in enumerate(bin_info_df.iterrows()):
            try:
                seq = genome.get_sequence(row["seqnames"], row["start"], row["end"])
                seq_str = str(seq.seq).upper()
                if len(seq_str) > 0:
                    gc = (seq_str.count("G") + seq_str.count("C")) / len(seq_str)
                    gc_content[i] = gc
                else:
                    gc_content[i] = 0.5
            except Exception:
                gc_content[i] = 0.5

            # Progress logging
            if (i + 1) % 10000 == 0:
                self.logger.info(f"Computed GC for {i + 1}/{len(bin_info_df)} bins")

        return gc_content

    def _store_cnv_results(self, store_path: Path, result: dict[str, Any]) -> None:
        """Store CNV detection results in the Zarr store.

        Args:
            store_path: Path to Zarr store
            result: CNV detection result dictionary
        """
        root = zarr.open(str(store_path), mode="a")

        output_key = self.cnv_config.output_key
        compressor = BloscCodec(cname="zstd", clevel=5)

        # Create CNV group
        if output_key in root:
            if self.cnv_config.overwrite:
                del root[output_key]
            else:
                self.logger.warning(f"CNV results already exist at {output_key}, skipping")
                return

        cnv_group = root.create_group(output_key)

        # Store CNV states matrix (zarr v3 compatible)
        cnv_states = result["cnv_states"].values
        cnv_group.create_array(
            "states",
            data=cnv_states,
            chunks=(min(1000, cnv_states.shape[0]), min(100, cnv_states.shape[1])),
            compressors=compressor,
            dtype=np.int8,
        )

        # Store log2 ratios
        log2_ratio = result["log2_ratio"].values
        cnv_group.create_array(
            "log2_ratio",
            data=log2_ratio.astype(np.float32),
            chunks=(min(1000, log2_ratio.shape[0]), min(100, log2_ratio.shape[1])),
            compressors=compressor,
        )

        # Store bin IDs and cell IDs
        bin_ids = np.array(result["cnv_states"].index.tolist(), dtype="S")
        cell_ids = np.array(result["cnv_states"].columns.tolist(), dtype="S")

        bin_arr = cnv_group.create_array(
            "bin_ids",
            shape=bin_ids.shape,
            dtype=bin_ids.dtype,
            compressors=compressor,
        )
        bin_arr[:] = bin_ids

        cell_arr = cnv_group.create_array(
            "cell_ids",
            shape=cell_ids.shape,
            dtype=cell_ids.dtype,
            compressors=compressor,
        )
        cell_arr[:] = cell_ids

        # Store summary statistics
        summary = result["summary"]
        summary_group = cnv_group.create_group("summary")
        for col in summary.columns:
            if summary[col].dtype == object:
                col_data = np.array(summary[col].tolist(), dtype="S")
            else:
                col_data = summary[col].values
            col_arr = summary_group.create_array(
                col,
                shape=col_data.shape,
                dtype=col_data.dtype,
                compressors=compressor,
            )
            col_arr[:] = col_data

        # Store segments
        segments = result["segments"]
        seg_group = cnv_group.create_group("segments")
        for chrom, seg_list in segments.items():
            seg_array = np.array(seg_list, dtype=np.int32)
            seg_arr = seg_group.create_array(
                chrom,
                shape=seg_array.shape,
                dtype=np.int32,
            )
            seg_arr[:] = seg_array

        # Store metadata
        cnv_group.attrs["tile_size"] = self.cnv_config.tile_size
        cnv_group.attrs["gc_correction"] = self.cnv_config.gc_correction
        cnv_group.attrs["gc_method"] = self.cnv_config.gc_method
        cnv_group.attrs["loss_threshold"] = self.cnv_config.loss_threshold
        cnv_group.attrs["gain_threshold"] = self.cnv_config.gain_threshold
        cnv_group.attrs["window_size"] = self.cnv_config.window_size
        cnv_group.attrs["min_segment_size"] = self.cnv_config.min_segment_size

        self.logger.info(f"Stored CNV results in {store_path}/{output_key}")

    def process(self) -> dict[str, Any]:
        """Run CNV detection.

        Returns:
            Dictionary with processing results
        """
        self._start_processing()

        # Check if disabled
        skipped = self._check_enabled("cnv_detection")
        if skipped:
            return skipped

        # Get input path
        input_path = self.get_target_path()
        if input_path is None:
            input_path = self.cnv_config.input
        if input_path is None:
            raise ValueError("No input path specified for CNV detection")

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        # Detect input type
        input_type = self._detect_input_type(input_path)

        # Load tile counts based on input type
        if input_type in ("zarr_sparse", "zarr_sparse_no_tiles"):
            counts_df, bin_info_df = self._load_tile_counts_zarr(input_path)
        elif input_type == "zarr_celltype_dense":
            counts_df, bin_info_df = self._load_tile_counts_celltype_dense(input_path)
        elif input_type in ("bpcells_fragments", "bpcells_sparse"):
            counts_df, bin_info_df = self._load_tile_counts_bpcells(input_path)
        else:
            raise ValueError(f"Unsupported input type for CNV detection: {input_type}")

        # Get assembly for GC content
        assembly = self.cnv_config.assembly
        if assembly == "auto":
            assembly = self.config.assembly
        if assembly == "auto":
            # Try to read from store
            try:
                root = zarr.open(str(input_path), mode="r")
                assembly = root.attrs.get("assembly", "hg38")
            except Exception:
                assembly = "hg38"
                self.logger.warning(f"Could not detect assembly, using {assembly}")

        # Compute GC content
        if self.cnv_config.gc_correction:
            self.logger.info(f"Computing GC content for {assembly}...")
            gc_content = self._compute_gc_content(bin_info_df, assembly)
            bin_info_df["GC"] = gc_content
        else:
            bin_info_df["GC"] = 0.5  # Placeholder

        # Import CNV detection functions
        from caesar.analysis.cnv_detection import detect_cnv, detect_cnv_iterative

        # Run CNV detection
        self.logger.info("Running CNV detection...")

        if self.cnv_config.iterative:
            result = detect_cnv_iterative(
                counts_df,
                bin_info_df,
                max_iterations=2,
                diploid_threshold=self.cnv_config.diploid_threshold,
                gc_correction=self.cnv_config.gc_correction,
                gc_method=self.cnv_config.gc_method,
                smooth_method=self.cnv_config.smooth_method,
                window_size=self.cnv_config.window_size,
                min_segment_size=self.cnv_config.min_segment_size,
                loss_threshold=self.cnv_config.loss_threshold,
                gain_threshold=self.cnv_config.gain_threshold,
                joint_segment=self.cnv_config.joint_segment,
            )
        else:
            result = detect_cnv(
                counts_df,
                bin_info_df,
                gc_correction=self.cnv_config.gc_correction,
                gc_method=self.cnv_config.gc_method,
                smooth_method=self.cnv_config.smooth_method,
                window_size=self.cnv_config.window_size,
                min_segment_size=self.cnv_config.min_segment_size,
                loss_threshold=self.cnv_config.loss_threshold,
                gain_threshold=self.cnv_config.gain_threshold,
                joint_segment=self.cnv_config.joint_segment,
            )

        # Store results
        output_path = self.config.output.dir if hasattr(self.config.output, "dir") else None
        if output_path:
            output_path = Path(output_path)
        else:
            output_path = input_path

        # For Zarr stores, store in-place
        if input_type.startswith("zarr"):
            self._store_cnv_results(input_path, result)

        # Also save summary CSV
        summary_path = output_path / "cnv_summary.csv" if output_path.is_dir() else output_path.with_suffix(".cnv_summary.csv")
        result["summary"].to_csv(summary_path, index=False)
        self.logger.info(f"Saved CNV summary to {summary_path}")

        runtime = self._end_processing()

        # Compute statistics
        summary = result["summary"]
        mean_altered = summary["pct_altered"].mean()
        n_cells_with_cnv = (summary["pct_altered"] > 5).sum()

        return {
            "output_path": str(input_path),
            "status": "success",
            "runtime": runtime,
            "n_cells": len(summary),
            "n_bins": len(bin_info_df),
            "mean_pct_altered": float(mean_altered),
            "n_cells_with_cnv": int(n_cells_with_cnv),
            "summary_path": str(summary_path),
        }
