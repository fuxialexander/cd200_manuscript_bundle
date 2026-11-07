"""Utility functions for fragment and BAM processing.

This module contains shared functions for processing genomic fragments
from various sources (fragment files, BAM files, etc.).
"""

from __future__ import annotations

# Standard library imports
import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import pysam
import zarr
from intervaltree import IntervalTree
from scipy.sparse import csr_matrix, lil_matrix
from zarr.codecs import BloscCodec
from zarr.core.dtype import VariableLengthUTF8
from tqdm import tqdm

from caesar.io.aggregator import ByKeyAggregator, SumAggregator
from caesar.io.query import KeyQuery, MetadataQuery, MultiKeyQuery

# Local imports
from caesar.io.zarr_io import (
    CelltypeDenseZarrIO,
    MotifDenseZarrIO,
    MultiSparseZarrIO,
    SingleSparseZarrIO,
)
from caesar.preprocess.utils.determine_assembly import determine_genome_assembly
from gcell.dna.genome import ChromSize, read_blacklist, read_peaks

# Constants
COMPRESSOR = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")
EXCLUDED_CHROMS = {"chrY", "chrM", "chrMT"}

# Type aliases
BarcodeDict = dict[str, int]
FragmentData = list[tuple[int, int, str, int]]

compressor = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")


def get_counts_in_range(
    start: int, end: int, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return unique positions and their counts within a genomic range.

    Args:
        start: Start position of the range
        end: End position of the range (exclusive)
        positions: 1D array of positions (already collected for this chromosome)

    Returns:
        Tuple of (unique_positions_in_range, counts)
    """
    # Boolean mask for positions within [start, end)
    mask = (positions >= start) & (positions < end)
    sub_positions = positions[mask]
    return np.unique(sub_positions, return_counts=True)


def create_chromosome_coverage_arrays(
    zarr_root: zarr.Group,
    group_name: str,
    chrom: str,
    chrom_size: int,
    chunk_size: int,
    compressor: BloscCodec | None = None,
) -> None:
    """Create the main chromosome arrays with proper chunking.

    Args:
        zarr_root: Zarr root group
        group_name: Name of the group (e.g., "celltype.id")
        chrom: Chromosome name
        chrom_size: Size of the chromosome
        chunk_size: Chunk size for internal chunking
        compressor: Optional compressor, defaults to BloscCodec
    """
    if compressor is None:
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    chrom_group_path = f"{group_name}/chrs/{chrom}"

    # Check if array already exists
    if chrom_group_path not in zarr_root:
        # Create main coverage array
        zarr_root.create_array(
            chrom_group_path,
            shape=(chrom_size,),
            chunks=(chunk_size,),  # Use chunk_size for internal chunking
            dtype=np.uint16,
            compressors=compressor,
            fill_value=0,
        )


def process_chromosome_coverage(
    zarr_root: zarr.Group,
    group_name: str,
    chrom: str,
    positions: np.ndarray,
    chrom_size: int,
    chunk_size: int = 2_000_000,
    compressor: BloscCodec | None = None,
) -> None:
    """
    Process coverage for a single chromosome in chunks and store in Zarr.

    Args:
        zarr_root: Zarr root group
        group_name: Name of the group
        chrom: Chromosome name
        positions: Array of positions for this chromosome
        chrom_size: Size of the chromosome
        chunk_size: Size of chunks for processing
        compressor: Optional compressor
    """
    # Create the chromosome array if it doesn't exist
    create_chromosome_coverage_arrays(
        zarr_root, group_name, chrom, chrom_size, chunk_size, compressor
    )

    # Number of chunks needed for this chromosome
    n_chunks = (chrom_size // chunk_size) + 1

    # Process each chunk
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, chrom_size)

        if chunk_start >= chrom_size:
            break  # No more coverage needed if chunk_start beyond chromosome length

        # Count positions falling into [chunk_start, chunk_end)
        uniq_positions, counts = get_counts_in_range(chunk_start, chunk_end, positions)

        # Create a dense array for this chunk
        length_of_chunk = chunk_end - chunk_start
        chunk_coverage = np.zeros(length_of_chunk, dtype=np.uint16)

        # Adjust the unique positions to be zero-based within this chunk
        chunk_coverage[uniq_positions - chunk_start] = counts

        # Write directly to the chromosome array
        chrom_group_path = f"{group_name}/chrs/{chrom}"
        zarr_root[chrom_group_path][chunk_start:chunk_end] = chunk_coverage


def apply_tn5_shift(
    positions: np.ndarray,
    strands: np.ndarray | None = None,
    single_end: bool = False,
    tn5_shift: bool = True,
) -> np.ndarray:
    """
    Apply position shifting based on assay type and strand information.

    For Tn5 transposase (ATAC-seq), the actual cut site is:
    - +4 bp for + strand
    - -4 bp for - strand

    For DNase-seq, the cut site is:
    - 0 bp for + strand (no shift)
    - +1 bp for - strand

    Args:
        positions: Array of positions
        strands: Array of strand information ('+' or '-' or boolean)
        single_end: Whether data is single-end (requires strand info)
        tn5_shift: Whether to apply Tn5 shifting (True) or DNase shifting (False)

    Returns:
        Shifted positions array
    """
    shifted_positions = positions.copy()

    if single_end:
        if strands is None:
            raise ValueError("Strand information required for single-end shifting")

        # Handle boolean or string strand representation
        if strands.dtype == bool:
            # True = + strand, False = - strand
            if tn5_shift:
                shifted_positions[strands] += 4  # + strand
                shifted_positions[~strands] -= 4  # - strand
            else:  # DNase-seq
                # + strand: no shift (0)
                shifted_positions[~strands] += 1  # - strand: +1
        else:
            # String representation
            plus_mask = strands == "+"
            minus_mask = strands == "-"
            if tn5_shift:
                shifted_positions[plus_mask] += 4
                shifted_positions[minus_mask] -= 4
            else:  # DNase-seq
                # + strand: no shift (0)
                shifted_positions[minus_mask] += 1  # - strand: +1

    else:  # paired-end
        if tn5_shift:
            # For paired-end Tn5, apply alternating shifts
            # First position (start of fragment) gets +4
            # Second position (end of fragment) gets -4
            shifted_positions[::2] += 4  # Start positions
            shifted_positions[1::2] -= 4  # End positions
        else:  # DNase-seq paired-end
            # For paired-end DNase, apply 0/+1 shifts
            # Start positions: no shift (0)
            shifted_positions[1::2] += 1  # End positions: +1

    return shifted_positions


def filter_positions_by_blacklist(
    positions: np.ndarray, blacklist_df: pd.DataFrame, chrom: str
) -> np.ndarray:
    """
    Filter positions based on blacklist regions.

    Args:
        positions: Array of positions
        blacklist_df: DataFrame with columns ["Chromosome", "Start", "End"]
        chrom: Chromosome name

    Returns:
        Filtered positions array
    """

    chrom_blacklist = blacklist_df[blacklist_df["Chromosome"] == chrom]
    if chrom_blacklist.empty:
        return positions

    mask = np.zeros(positions.shape, dtype=bool)
    for _, region in chrom_blacklist.iterrows():
        mask |= (positions >= region["Start"]) & (positions < region["End"])

    return positions[~mask]


def run_fragment_to_zarr(args):
    """
    Run the fragment to zarr conversion

    Args:
        args: Namespace object with the following attributes:
            id (str): Sample or replicate ID
            celltype (str): Cell type or group identifier
            fragment_file (str): Path to fragment file
            assembly (str): Genome assembly
            output_zarr (str): Output path
            chunk_size (int): Chunk size
            single_end (bool): Process as single-end
    """
    # Read chromosome sizes into a dict
    chrom_sizes = ChromSize(args.assembly).dict

    # Load blacklist
    blacklist = read_blacklist(args.assembly)

    # Open the tabix file
    tbx = pysam.TabixFile(args.fragment_file)

    # Check number of columns
    try:
        first_rec = next(tbx.fetch(parser=pysam.asTuple()))
        num_columns = len(first_rec)
        logging.info(f"Fragment file has {num_columns} columns.")
        if args.single_end and num_columns < 6:
            raise ValueError(
                "Single-end mode requires at least 6 columns for strand information."
            )
    except StopIteration:
        logging.info("Fragment file is empty.")
        return  # No data to process
    finally:
        # We need to close and reopen to reset the iterator
        tbx.close()
        tbx = pysam.TabixFile(args.fragment_file)

    # Create (or open) a Zarr store
    zarr_root = zarr.open(args.output_zarr, mode="a")

    # Always use celltype.id group structure for consistency
    group_name = f"{args.celltype}.{args.id}"

    # Create the group and set attributes
    group = zarr_root.require_group(group_name)
    group.attrs["id"] = str(args.id)
    group.attrs["celltype"] = str(args.celltype)
    group.attrs["assembly"] = str(args.assembly)
    group.attrs["chunk_size"] = int(args.chunk_size)  # Store chunk size in metadata

    # Also set/update root attributes for CelltypeDenseZarrIO compatibility
    if "assembly" not in zarr_root.attrs:
        zarr_root.attrs["assembly"] = args.assembly

    # Update the list of celltype IDs
    existing_ids = list(zarr_root.attrs.get("celltype_ids", []))
    if group_name not in existing_ids:
        existing_ids.append(group_name)
        zarr_root.attrs["celltype_ids"] = existing_ids

    # Initialize library size
    library_size = 0

    # For each chromosome in the chromosome sizes (and in the tabix contigs),
    # fetch lines from tabix and accumulate positions. Then chunk and store.
    chroms_in_tabix = set(tbx.contigs)

    # Use chromosome filtering from config if available
    if hasattr(args, "chr_filter_config"):
        from caesar.preprocess.core.chromosome_filter import get_chromosomes_to_process

        chroms_list = get_chromosomes_to_process(
            args.chr_filter_config, list(chrom_sizes.keys()), chroms_in_tabix
        )
    else:
        # Fallback to shared filtering logic for backward compatibility
        from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

        chr_filter = ChromosomeFilter({})
        chroms_list = [
            chrom
            for chrom in chrom_sizes
            if (chrom in chroms_in_tabix) and chr_filter.is_valid_chromosome(chrom)
        ]
    for chrom in tqdm(chroms_list, desc="Processing chromosomes"):
        positions = []
        # Fetch fragment records for this chromosome
        for rec in tbx.fetch(chrom):
            fields = rec.strip().split("\t")
            start = int(fields[1])
            end = int(fields[2])

            if args.single_end:
                strand = fields[5]
                if strand == "+":
                    positions.append(start)
                    library_size += 1
                elif strand == "-":
                    positions.append(end)
                    library_size += 1
            else:  # Paired-end
                positions.extend([start, end])
                # Update library size
                library_size += 2

        if not positions:
            continue  # No data for this chromosome

        # Convert to a NumPy array for faster slicing
        positions = np.array(positions, dtype=np.int32)

        # Apply blacklist filtering
        positions = filter_positions_by_blacklist(positions, blacklist, chrom)

        if positions.size == 0:
            continue

        # Process and write this chromosome to Zarr
        process_chromosome_coverage(
            zarr_root=zarr_root,
            group_name=group_name,
            chrom=chrom,
            positions=positions,
            chrom_size=chrom_sizes[chrom],
            chunk_size=args.chunk_size,
        )

    # Set the library size as a group attribute (convert to Python int for JSON serialization)
    group.attrs["library_size"] = int(library_size)

    print(f"Done. Dense coverage Zarr written to: {args.output_zarr}")


def create_tabix_index(fragment_file: str | Path) -> None:
    """
    Create tabix index for fragment file if it doesn't exist.

    This function is used by FragmentProcessor (Zarr backend) to enable efficient
    chromosome-by-chromosome fragment reading via pysam.TabixFile.

    NOTE: BPCellsProcessor does NOT use this function because BPCells' native
    import_10x_fragments() handles fragment reading and indexing internally using
    its C++ backend, which is more efficient than tabix for BPCells workflows.

    Args:
        fragment_file: Path to the fragment file (.tsv.gz)

    Raises:
        pysam.SamtoolsError: If tabix indexing fails
    """
    index_path = Path(str(fragment_file) + ".tbi")
    if not index_path.exists():
        logging.info("Creating tabix index...")
        try:
            pysam.tabix_index(str(fragment_file), preset="bed", force=True)
        except pysam.SamtoolsError as e:
            logging.error(f"Failed to create tabix index: {e}")
            raise


def get_valid_chromosomes(
    chrom_size: ChromSize, chr_filter_config: dict | None = None
) -> list[str]:
    """
    Get list of valid chromosomes for processing, excluding Y and mitochondrial chromosomes.

    Args:
        chrom_size: ChromSize object containing chromosome information
        chr_filter_config: Optional chromosome filter configuration

    Returns:
        List of chromosome names meeting the criteria:
        - Length of chromosome name <= 5 characters
        - Not in excluded list (chrY, chrM, chrMT)
    """
    if chr_filter_config is not None:
        from caesar.preprocess.core.chromosome_filter import get_chromosomes_to_process

        return get_chromosomes_to_process(
            chr_filter_config, list(chrom_size.dict.keys()), None
        )

    # Default filtering
    return sorted(
        [
            chrom
            for chrom in chrom_size.dict
            if len(chrom) <= 5 and chrom not in EXCLUDED_CHROMS
        ]
    )


def get_unique_valid_barcodes(
    fragment_file: str | Path,
    min_fragments: int = 1000,
    max_fragments: int = 50000,
    min_cells: int = 100,
) -> BarcodeDict:
    """
    Extract unique barcodes from fragment file and create barcode dictionary.
    Filters barcodes based on minimum and maximum fragment counts.

    Args:
        fragment_file: Path to the fragment file (.tsv.gz)
        min_fragments: Minimum number of fragments required per barcode
        max_fragments: Maximum number of fragments allowed per barcode

    Returns:
        Dictionary mapping barcodes to their numerical indices (0-based)

    Raises:
        pysam.TabixError: If file cannot be opened or is not properly indexed
    """
    parent_dir = Path(fragment_file).parent
    fname_split = Path(fragment_file).stem.split(".")[0]
    barcode_file = parent_dir / f"{fname_split}.barcodes.txt"
    if barcode_file.exists():
        # Check if barcodes.txt was created with the same filtering parameters
        # by checking file modification time vs fragment file
        fragment_mtime = Path(fragment_file).stat().st_mtime
        barcode_mtime = barcode_file.stat().st_mtime

        # If barcodes.txt is older than fragment file, or if we want to recompute,
        # we should regenerate it. For now, always use it if it exists (backward compatible)
        # but log a warning if it seems stale
        if barcode_mtime < fragment_mtime:
            logging.warning(
                f"barcodes.txt ({barcode_file}) is older than fragment file. "
                f"Consider deleting it to regenerate with current filtering parameters."
            )

        barcodes = np.loadtxt(barcode_file, dtype=str)
        # Remove duplicates and sort to ensure unique barcodes
        barcodes = np.sort(np.unique(barcodes))
        # Convert to Python strings for consistent matching
        logging.info(f"Loaded {len(barcodes)} unique barcodes from {barcode_file}")
        return {str(bc): i for i, bc in enumerate(barcodes)}
    else:
        try:
            tbx = pysam.TabixFile(str(fragment_file), threads=16)

            # Count fragments per barcode
            barcode_counts = {}
            for row in tbx.fetch():
                barcode = row.split("\t")[3]
                if barcode not in barcode_counts:
                    barcode_counts[barcode] = 0
                barcode_counts[barcode] += 1
            tbx.close()
            # Filter barcodes based on fragment counts
            valid_barcodes = []
            for barcode, count in barcode_counts.items():
                if count > min_fragments and count < max_fragments:
                    valid_barcodes.append(barcode)
            valid_barcodes = np.sort(valid_barcodes)
            # Remove duplicates to ensure unique barcodes
            valid_barcodes = np.unique(valid_barcodes)
            # save to .barcode.txt
            parent_dir = Path(fragment_file).parent
            fname_split = Path(fragment_file).stem.split(".")[0]
            barcode_file = parent_dir / f"{fname_split}.barcodes.txt"
            np.savetxt(barcode_file, valid_barcodes, fmt="%s")
            # Sort barcodes for deterministic indexing
            print("Number of valid barcodes: ", len(valid_barcodes))
            if len(valid_barcodes) <= min_cells:
                raise ValueError(f"Less than {min_cells} valid barcodes found, exiting")
            else:
                print("Number of valid barcodes: ", len(valid_barcodes))

            # Convert to Python strings for consistent matching
            return {str(bc): i for i, bc in enumerate(valid_barcodes)}

        except Exception as e:
            logging.error(f"Failed to read barcodes from fragment file: {e}")
            raise


def initialize_zarr_store(zarr_path: str | Path, skip_exist: bool = True) -> zarr.Group:
    """
    Initialize zarr store for data storage.

    Args:
        zarr_path: Path to zarr store
        skip_exist: If True, don't overwrite existing store

    Returns:
        Initialized zarr group

    Raises:
        zarr.errors.PathNotFoundError: If parent directory doesn't exist
    """
    try:
        # Use zarr.open_group for v3 format
        mode = "w" if not skip_exist else "a"
        return zarr.open_group(str(zarr_path), mode=mode, zarr_format=3)
    except zarr.errors.PathNotFoundError as e:
        logging.error(f"Failed to initialize zarr store: {e}")
        raise


def create_genome_tiles(chrom_size: ChromSize, tile_size: int = 500) -> pd.DataFrame:
    """
    Create genome-wide tiles with chromosome filtering only.

    **Used by: Fragment Processor (preprocessing stage)**

    This function applies chromosome filtering only (removes patches, chrM, chrY, etc.)
    and does NOT apply blacklist filtering. Blacklist filtering is applied later during
    AnnData generation or peak count matrix generation (in ClusteringMixin).

    **Filtering Strategy:**
    - **Preprocessing (Fragment Processor)**: Chromosome filtering only
      - Removes non-standard chromosomes (patches, chrM, chrY) for efficiency
      - Stores chromosome-filtered tiles in Zarr
    - **Analysis (Clustering/AnnData)**: Blacklist filtering on stored tiles
      - Reads stored chromosome-filtered tiles from Zarr
      - Applies blacklist filtering on top
      - Computes matrix with blacklist-filtered tiles

    **Note:** This is different from BPCells backend, which filters after matrix creation.

    Args:
        chrom_size: ChromSize object containing chromosome sizes
        tile_size: Size of tiles in base pairs

    Returns:
        DataFrame with columns ['Chromosome', 'Start', 'End'], filtered by chromosomes only
    """
    from caesar.preprocess.core.tile_filtering import filter_tiles_by_chromosomes

    tiles = chrom_size.tiling_region(tile_size)
    return filter_tiles_by_chromosomes(tiles)


def store_region_and_counts(
    zarr_root: zarr.Group,
    group_path: str,
    regions: pd.DataFrame,
    counts: np.ndarray | csr_matrix,
    chrom: str,
    chunk_idx: int | None = None,
    start_idx: int | None = None,
) -> None:
    """
    Store regions and their counts in specified zarr group.

    Args:
        zarr_root: Zarr root group
        group_path: Path to group (e.g. 'tiles' or 'peaks')
        regions: DataFrame with ['Chromosome', 'Start', 'End'] columns
        counts: Count matrix (dense array or sparse matrix)
        chrom: Chromosome name
        chunk_idx: If provided, store as chunk data
        start_idx: Starting index for the regions in this chunk

    Raises:
        KeyError: If required columns are missing from regions DataFrame
        ValueError: If data dimensions don't match
        zarr.errors.ZarrError: If zarr operations fail
    """
    try:
        # Validate input data
        required_cols = ["Chromosome", "Start", "End"]
        if not all(col in regions.columns for col in required_cols):
            raise KeyError(f"Regions DataFrame must contain columns: {required_cols}")

        # Create the main group if it doesn't exist
        if group_path not in zarr_root:
            zarr_root.create_group(group_path)

        # Create regions group if it doesn't exist
        if "regions" not in zarr_root[group_path]:
            zarr_root[group_path].create_group("regions")

        # Store regions as Start, End array if not already stored
        regions_group = zarr_root[group_path]["regions"]
        if chrom not in regions_group:
            chrom_regions = regions.query("Chromosome == @chrom")[["Start", "End"]]
            if len(chrom_regions) == 0:
                logging.warning(f"No regions found for chromosome {chrom}")
                return

            regions_data = np.asarray(chrom_regions.values, dtype="i4")
            regions_arr = regions_group.create_array(
                chrom,
                shape=regions_data.shape,
                dtype="i4",
                chunks=(1000, 2),
                compressors=COMPRESSOR,
                overwrite=True,
            )
            regions_arr[:] = regions_data
            # Store number of peaks/tiles as attribute (convert to Python int for JSON serialization)
            regions_group[chrom].attrs["n_regions"] = int(len(chrom_regions))

        # Convert to sparse if not already
        count_sparse = counts if isinstance(counts, csr_matrix) else csr_matrix(counts)

        if count_sparse.sum() > 0:
            row, col = count_sparse.nonzero()
            data = count_sparse.data
            sparse_data = np.vstack((row, col, data)).T

            if chunk_idx is not None:
                # Create and store chunked data
                chunk_path = f"count_sparse_chunk/{chrom}"
                if chunk_path not in zarr_root[group_path]:
                    chunk_group = zarr_root[group_path].create_group(chunk_path)
                    chunk_group.attrs["n_chunks"] = 0

                chunk_group = zarr_root[group_path][chunk_path]
                # Convert to Python int for JSON serialization
                chunk_group.attrs["n_chunks"] = int(max(
                    chunk_group.attrs["n_chunks"], chunk_idx + 1
                ))

                chunk_dataset = chunk_group.create_array(
                    f"chunk_{chunk_idx}",
                    shape=sparse_data.shape,
                    chunks=(sparse_data.shape[0], sparse_data.shape[1]),
                    dtype="u4",
                    compressors=COMPRESSOR,
                    overwrite=True,
                )
                chunk_dataset[:] = sparse_data
                if start_idx is not None:
                    # Convert to Python int for JSON serialization
                    chunk_dataset.attrs["start_idx"] = int(start_idx)
            else:
                # Create and store combined data
                if "count_sparse" not in zarr_root[group_path]:
                    zarr_root[group_path].create_group("count_sparse")

                sparse_arr = zarr_root[group_path]["count_sparse"].create_array(
                    chrom,
                    shape=sparse_data.shape,
                    chunks=(10000, sparse_data.shape[1]),
                    dtype="u4",
                    compressors=COMPRESSOR,
                    overwrite=True,
                )
                sparse_arr[:] = sparse_data
        else:
            if chunk_idx is not None:
                logging.info(
                    f"Warning: count matrix for {chrom} chunk {chunk_idx} is all zeros"
                )
            else:
                logging.info(f"Warning: count matrix for {chrom} is all zeros")

    except Exception as e:
        logging.error(f"Failed to store regions and counts: {e}")
        raise


def process_fragments(
    fragment_file: str | Path,
    zarr_path: str | Path,
    detected_assembly: str | None = None,
    peak_file: str | Path | None = None,
    tile_size: int = 500,
    min_fragments: int = 1000,
    max_fragments: int = 50000,
    max_fragments_per_cell: int = 500,
    chunk_size: int = 2_000_000,
    n_jobs: int = 64,
    skip_exist: bool = True,
    min_cells: int = 100,
    verbose: bool = False,
    chr_filter_config: dict | None = None,
) -> None:
    """
    Process fragment file and peaks into SingleSparseZarrIO format.

    This function performs the following steps:
    1. Sets up logging and creates tabix index
    2. Determines genome assembly and initializes resources
    3. Creates genome tiles or reads peaks
    4. Processes fragments by chromosome and chunk
    5. Stores metadata and finalizes processing

    Args:
        fragment_file: Path to fragment file (.tsv.gz)
        zarr_path: Output zarr path
        peak_file: Optional path to peak file (.bed/.narrowPeak)
        max_fragments_per_cell: Max fragments to store per cell
        chunk_size: Size of genomic chunks to process
        n_jobs: Number of parallel jobs
        skip_exist: Skip processing if output exists
        verbose: If True, show INFO level logging messages
        tile_size: Size of genome tiles in base pairs

    Raises:
        ValueError: If input parameters are invalid
        FileNotFoundError: If input files don't exist
        RuntimeError: If processing fails
    """
    try:
        create_tabix_index(fragment_file)

        # Determine genome assembly and initialize resources
        if detected_assembly is None:
            detected_assembly, _ = determine_genome_assembly(
                fragment_file=fragment_file,
                root_path=".",
            )
        chrom_size = ChromSize(assembly=detected_assembly)

        # Create tiles
        tiles = create_genome_tiles(chrom_size, tile_size)

        # Read peaks if provided
        peaks = None
        if peak_file:
            try:
                peaks = read_peaks(peak_file).rename(
                    columns={"chromosome": "Chromosome", "start": "Start", "end": "End"}
                )
            except Exception as e:
                logging.error(f"Failed to read peak file: {e}")
                raise

        logging.info(f"Initializing zarr store for {zarr_path}")
        zarr_root = initialize_zarr_store(zarr_path, skip_exist)

        logging.info(f"Getting unique barcodes for {fragment_file}")
        barcode_dict = get_unique_valid_barcodes(
            fragment_file, min_fragments, max_fragments, min_cells
        )

        logging.info("Storing essential attributes")
        # Store essential attributes early
        zarr_root.attrs.update(
            {
                "tile_size": tile_size,
                "max_fragments_per_cell": max_fragments_per_cell,
                "chunk_size": chunk_size,
                "assembly": detected_assembly,
                "data_key": Path(fragment_file).stem,
            }
        )

        logging.info("Processing chromosomes...")
        # Process chromosomes
        for chrom in get_valid_chromosomes(chrom_size, chr_filter_config):
            process_chromosome(
                chrom=chrom,
                zarr_root=zarr_root,
                fragment_file=fragment_file,
                barcode_dict=barcode_dict,
                chrom_length=chrom_size.dict,
                max_fragments_per_cell=max_fragments_per_cell,
                chunk_size=chunk_size,
                tiles=tiles,
                peaks=peaks,
                skip_exist=skip_exist,
                n_jobs=n_jobs // 4,
            )

        logging.info("Storing remaining metadata...")
        # Store remaining metadata
        barcodes_data = np.array(list(barcode_dict.keys()), dtype=object)
        barcodes_arr = zarr_root.create_array(
            "barcodes",
            shape=barcodes_data.shape,
            dtype=VariableLengthUTF8(),
            chunks=(1000,),
            compressors=COMPRESSOR,
            overwrite=True,
        )
        barcodes_arr[:] = barcodes_data
        zarr_root.attrs["completed"] = True
        zarr_root.attrs["completion_date"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        logging.info("Processing completed successfully")

    except Exception as e:
        logging.error(f"Failed to process fragments: {e}")
        raise RuntimeError(f"Fragment processing failed: {str(e)}") from e


def process_chromosome(
    chrom: str,
    zarr_root: zarr.Group,
    fragment_file: str | Path,
    barcode_dict: BarcodeDict,
    chrom_length: dict[str, int],
    max_fragments_per_cell: int,
    chunk_size: int,
    tiles: pd.DataFrame,
    peaks: pd.DataFrame | None,
    skip_exist: bool,
    n_jobs: int,
) -> None:
    """
    Process a single chromosome's data.

    This function:
    1. Creates necessary arrays for the chromosome
    2. Processes fragments in chunks
    3. Stores tile and peak counts
    4. Updates library size information

    Args:
        chrom: Chromosome name
        zarr_root: Zarr root group
        fragment_file: Path to fragment file
        barcode_dict: Dictionary mapping barcodes to indices
        chrom_length: Dictionary of chromosome lengths
        max_fragments_per_cell: Maximum fragments per cell
        chunk_size: Size of chunks to process
        tiles: DataFrame of genome tiles
        peaks: Optional DataFrame of peaks
        skip_exist: Skip if chunk exists
        n_jobs: Number of parallel jobs

    Raises:
        KeyError: If chromosome not found in chrom_length
        ValueError: If chunk processing fails
        RuntimeError: If parallel processing fails
    """
    try:
        if chrom not in chrom_length:
            raise KeyError(f"Chromosome {chrom} not found in chromosome lengths")

        logging.info(f"Processing chromosome {chrom}...")

        # Create the main chromosome arrays if they don't exist
        if not skip_exist or f"chrs/{chrom}" not in zarr_root:
            create_chromosome_arrays(
                zarr_root=zarr_root,
                chrom=chrom,
                num_cells=len(barcode_dict),
                chrom_length=chrom_length[chrom],
                chunk_size=chunk_size,
            )

        # Open tabix-indexed fragment file
        tbx = pysam.TabixFile(str(fragment_file))

        chunk_args = prepare_chunk_arguments(
            chrom,
            chrom_length,
            chunk_size,
            zarr_root,
            tbx,
            barcode_dict,
            max_fragments_per_cell,
            tiles,
            peaks,
            skip_exist,
        )

        tbx.close()

        # Process chunks in parallel
        failed_chunks = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = list(executor.map(process_chunk, chunk_args))
            for i, future in enumerate(futures):
                try:
                    if future:
                        process_chunk_result(
                            zarr_root,
                            future,
                            tiles,
                            peaks,
                        )
                except Exception as e:
                    logging.error(f"Failed to process chunk {i} of {chrom}: {e}")
                    failed_chunks.append(i)

        if failed_chunks:
            raise RuntimeError(f"Failed to process chunks: {failed_chunks}")

        # Finalize chromosome processing
        finalize_chromosome_processing(
            zarr_root=zarr_root,
            chrom=chrom,
            tiles=tiles,
            peaks=peaks,
            n_cells=len(barcode_dict),
        )

    except Exception as e:
        logging.error(f"Failed to process chromosome {chrom}: {e}")
        raise


def create_chromosome_arrays(
    zarr_root: zarr.Group,
    chrom: str,
    num_cells: int,
    chrom_length: int,
    chunk_size: int,
) -> None:
    """Create the main chromosome arrays with proper chunking"""
    n_chunks = (chrom_length // chunk_size) + 1
    max_fragments_per_cell = zarr_root.attrs["max_fragments_per_cell"]

    # Create main fragment array group
    if f"chrs/{chrom}" not in zarr_root:
        zarr_root.create_group(f"chrs/{chrom}")

    # Create library size group
    if f"library_size/{chrom}" not in zarr_root:
        zarr_root.create_group(f"library_size/{chrom}")

    # Pre-create arrays for each chunk
    for chunk_idx in range(n_chunks):
        # Fragment data array
        if f"chrs/{chrom}/chunk_{chunk_idx}" not in zarr_root:
            zarr_root.create_array(
                f"chrs/{chrom}/chunk_{chunk_idx}",
                shape=(num_cells, max_fragments_per_cell),
                chunks=(min(1000, num_cells), max_fragments_per_cell),
                dtype="u4",
                compressors=COMPRESSOR,
                fill_value=0,
            )

        # End mask array
        if f"chrs/{chrom}/chunk_{chunk_idx}_end_mask" not in zarr_root:
            zarr_root.create_array(
                f"chrs/{chrom}/chunk_{chunk_idx}_end_mask",
                shape=(num_cells, max_fragments_per_cell),
                chunks=(min(1000, num_cells), max_fragments_per_cell),
                dtype="i2",
                compressors=COMPRESSOR,
                fill_value=0,
            )

        # Library size array
        if f"library_size/{chrom}/chunk_{chunk_idx}" not in zarr_root:
            zarr_root.create_array(
                f"library_size/{chrom}/chunk_{chunk_idx}",
                shape=(num_cells,),
                chunks=(min(1000, num_cells),),
                dtype="f4",
                compressors=COMPRESSOR,
                fill_value=0,
            )


def prepare_chunk_arguments(
    chrom,
    chrom_length,
    chunk_size,
    zarr_root,
    tbx,
    barcode_dict,
    max_fragments_per_cell,
    tiles,
    peaks,
    skip_exist,
):
    """
    Prepare arguments for processing each chunk of a chromosome.

    Args:
        chrom: Chromosome name
        chrom_length: Dictionary of chromosome lengths
        chunk_size: Size of each chunk
        zarr_root: Zarr root group
        tbx: Tabix file handle
        barcode_dict: Dictionary mapping barcodes to indices
        max_fragments_per_cell: Maximum fragments per cell
        tiles: DataFrame containing tile regions
        peaks: DataFrame containing peak regions or None
        skip_exist: Skip if chunk exists

    Returns:
        List of arguments for each chunk
    """
    n_chunks = (chrom_length[chrom] // chunk_size) + 1
    chunk_args = []

    for chunk_idx in tqdm(range(n_chunks), desc=f"Processing {chrom}", unit="chunk"):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, chrom_length[chrom])

        if (
            Path(f"{zarr_root.store.root}/chrs/{chrom}/chunk_{str(chunk_idx)}").exists()
            and skip_exist
        ):
            logging.info(f"chunk {chunk_idx} of {chrom} already exists")
            continue

        create_zarr_groups(zarr_root, chrom)
        fragment_data = fetch_fragment_data(tbx, chrom, chunk_start, chunk_end)

        # Filter tiles for this chunk
        filtered_tiles = (
            tiles.query("Chromosome == @chrom")
            .reset_index(drop=True)
            .reset_index()
            .query("Start >= @chunk_start & End <= @chunk_end")
            if tiles is not None
            else None
        )

        # Filter peaks for this chunk
        filtered_peaks = (
            peaks.query("Chromosome == @chrom")
            .reset_index(drop=True)
            .reset_index()
            .query("Start >= @chunk_start & End <= @chunk_end")
            if peaks is not None
            else None
        )

        chunk_args.append(
            (
                chunk_start,
                chunk_end,
                chrom,
                chunk_idx,
                zarr_root,
                barcode_dict,
                max_fragments_per_cell,
                filtered_tiles,
                filtered_peaks,
                fragment_data,
            )
        )

    return chunk_args


def create_zarr_groups(zarr_root: zarr.Group, chrom: str) -> None:
    """
    Create necessary zarr groups for chromosome data.

    Args:
        zarr_root: Zarr root group
        chrom: Chromosome name
    """
    # Create main chromosome group for fragments and end masks
    if not Path(f"{zarr_root.store.root}/chrs/{chrom}").exists():
        zarr_root.create_group(f"chrs/{chrom}")

    # Create library size group for per-chunk library sizes
    if not Path(f"{zarr_root.store.root}/library_size/{chrom}").exists():
        zarr_root.create_group(f"library_size/{chrom}")


def fetch_fragment_data(
    tbx: pysam.TabixFile,
    chrom: str,
    chunk_start: int,
    chunk_end: int,
) -> list:
    """
    Fetch fragment data for a specific chunk from tabix file.

    Args:
        tbx: Tabix file handle
        chrom: Chromosome name
        chunk_start: Start position of chunk
        chunk_end: End position of chunk

    Returns:
        List of fragment data tuples
    """
    fragment_data = []
    try:
        for row in tbx.fetch(chrom, chunk_start, chunk_end):
            fields = row.split("\t")
            start, end = int(fields[1]), int(fields[2])
            if start >= chunk_start and start < chunk_end:
                fragment_data.append((start, end, fields[3], 1))
    except ValueError:
        logging.warning(f"No fragments found for {chrom}:{chunk_start}-{chunk_end}")

    return fragment_data


def process_chunk(args):
    """Process a single chunk of data"""
    (
        chunk_start,
        chunk_end,
        chrom,
        chunk_idx,
        zarr_root,
        barcode_dict,
        max_fragments_per_cell,
        tiles,
        peaks,
        fragment_data,
    ) = args

    logging.info(f"Processing chunk {chunk_idx} of {chrom}...")

    # Initialize count tables
    tile_count = lil_matrix(
        (len(barcode_dict), len(tiles) if tiles is not None else 0), dtype=int
    )
    peak_count = lil_matrix(
        (len(barcode_dict), len(peaks) if peaks is not None else 0), dtype=int
    )

    # Create interval trees for both tiles and peaks
    tile_interval = None
    peak_interval = None
    tile_start_idx = 0
    peak_start_idx = 0

    if tiles is not None:
        tile_start_end_index = tiles[["Start", "End", "index"]].values.tolist()
        tile_interval = IntervalTree.from_tuples(tile_start_end_index)
        tile_start_idx = tiles["index"].min()

    if peaks is not None:
        peak_start_end_index = peaks[["Start", "End", "index"]].values.tolist()
        peak_interval = IntervalTree.from_tuples(peak_start_end_index)
        peak_start_idx = peaks["index"].min()

    # Reconstruct DataFrame from passed data
    fragment_chunk = pd.DataFrame(
        fragment_data, columns=["start", "end", "barcodes", "count"]
    )
    fragment_chunk["chromosome"] = chrom

    try:
        barcode_array, tile_count, peak_count, FRiP, library_size, end_mask = (
            get_barcode_array_chunk(
                chrom,
                fragment_chunk,
                barcode_dict,
                chunk_idx=chunk_idx,
                seed=42,
                max_fragment_per_cell=max_fragments_per_cell,
                tile_interval=tile_interval,
                tile_count=tile_count,
                tile_start_idx=tile_start_idx,
                peak_interval=peak_interval,
                peak_count=peak_count,
                peak_start_idx=peak_start_idx,
            )
        )
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_idx} of {chrom}: {e}")
        raise e

    try:
        # Store fragment data
        zarr_root[f"chrs/{chrom}/chunk_{str(chunk_idx)}"] = barcode_array
        # Convert FRiP to Python float for JSON serialization
        zarr_root[f"chrs/{chrom}/chunk_{str(chunk_idx)}"].attrs["FRiP"] = float(FRiP)

        # Store end mask with correct naming
        zarr_root[f"chrs/{chrom}/chunk_{str(chunk_idx)}_end_mask"] = end_mask

        # Store per-chunk library size
        zarr_root[f"library_size/{chrom}/chunk_{str(chunk_idx)}"] = library_size

    except Exception as e:
        logging.error(f"Error storing data for chunk {chunk_idx} of {chrom}: {e}")
        raise e

    return (
        chrom,
        chunk_idx,
        tile_count,
        tile_start_idx,
        peak_count,
        peak_start_idx,
        FRiP,
        library_size,
    )


def update_peak_counts(
    insertions: list,
    peak_interval: IntervalTree,
    peak_count: np.ndarray,
    barcode_idx: int,
    peak_start_idx: int,
) -> tuple[int, int]:
    """
    Update peak counts for a given set of insertions.

    Args:
        insertions: List of insertion positions
        peak_interval: IntervalTree containing peak regions
        peak_count: Array to store peak counts
        barcode_idx: Index of current barcode
        peak_start_idx: Starting index for peaks in this chunk

    Returns:
        Tuple of (insertions in peaks, insertions not in peaks)
    """
    in_peak = 0
    not_in_peak = 0

    for insertion in insertions:
        intersect = peak_interval[insertion]
        if intersect:
            peak_idx = int(next(iter(intersect)).data)
            peak_count[barcode_idx, peak_idx - peak_start_idx] += 1
            in_peak += 1
        else:
            not_in_peak += 1

    return in_peak, not_in_peak


def get_barcode_array_chunk(
    chromosome: str,
    fragment: pd.DataFrame,
    barcodes_dict: dict,
    chunk_idx: int = 0,
    seed: int = 42,
    max_fragment_per_cell: int = 500,
    tile_interval: IntervalTree | None = None,
    tile_count: lil_matrix | None = None,
    tile_start_idx: int = 0,
    peak_interval: IntervalTree | None = None,
    peak_count: lil_matrix | None = None,
    peak_start_idx: int = 0,
) -> tuple[np.ndarray, lil_matrix, lil_matrix, float, np.ndarray, np.ndarray]:
    """
    Generates a barcode array for a specific chunk of genomic data.

    Args:
        chromosome: The chromosome identifier
        fragment: DataFrame with fragment data
        barcodes_dict: Dictionary mapping barcodes to indices
        chunk_idx: Index of the current chunk
        seed: Random seed for reproducibility
        max_fragment_per_cell: Maximum fragments allowed per cell
        tile_interval: IntervalTree containing tile regions
        tile_count: Array to store tile counts
        tile_start_idx: Starting index for tiles in this chunk
        peak_interval: IntervalTree containing peak regions
        peak_count: Array to store peak counts
        peak_start_idx: Starting index for peaks in this chunk

    Returns:
        Tuple containing:
        - barcode array (fragments positions)
        - tile count table
        - peak count table
        - FRiP score
        - library size
        - end mask array
    """
    barcodes = list(barcodes_dict.keys())
    # Ensure barcodes are Python strings (not numpy.str_) for proper matching
    barcodes = [str(bc) for bc in barcodes]
    library_size = np.zeros(len(barcodes), dtype=int)

    if len(fragment) == 0:
        logging.info(f"Warning: chunk {chunk_idx} has no fragments")
        return (
            np.zeros((len(barcodes), max_fragment_per_cell), dtype="u4"),
            tile_count,
            peak_count,
            0,
            library_size,
            np.zeros((len(barcodes), max_fragment_per_cell), dtype="i2"),
        )

    # Process fragment data
    # Ensure fragment barcodes are strings and match exactly
    fragment["barcodes"] = fragment["barcodes"].astype(str)
    fragment["barcodes"] = pd.Categorical(
        fragment["barcodes"].values, categories=barcodes
    )
    fragment["barcodes"] = fragment["barcodes"].cat.codes
    fragment = fragment.query("barcodes!=-1")

    # Create mask data
    fragment_mask = fragment.copy()
    start_offset = fragment_mask["end"] - fragment_mask["start"]
    end_offset = fragment_mask["start"] - fragment_mask["end"]
    fragment_mask["start"] = start_offset
    fragment_mask["end"] = end_offset

    # Melt the dataframes
    fragment_pos = fragment.melt(
        id_vars=["barcodes"],
        value_vars=["start", "end"],
        var_name="coord",
        value_name="pos",
    )
    fragment_mask = fragment_mask.melt(
        id_vars=["barcodes"],
        value_vars=["start", "end"],
        var_name="coord",
        value_name="mask",
    )

    # Initialize arrays
    chunk_barcode_array = np.zeros((len(barcodes), max_fragment_per_cell), dtype="u4")
    chunk_barcode_mask = np.zeros((len(barcodes), max_fragment_per_cell), dtype="i2")

    # Group by barcode
    pos_dict = {k: g["pos"].tolist() for k, g in fragment_pos.groupby("barcodes")}
    mask_dict = {k: g["mask"].tolist() for k, g in fragment_mask.groupby("barcodes")}

    # Track statistics
    total_in_peak = 0
    total_not_in_peak = 0
    n_barcodes_over_limit = 0

    # Process each barcode
    for barcode_i, insertions_i in pos_dict.items():
        masks_i = mask_dict[barcode_i]
        library_size[barcode_i] += len(insertions_i)

        # Update tile counts
        if tile_interval is not None:
            for insertion in insertions_i:
                intersect = tile_interval[insertion]
                if intersect:
                    tile_idx = int(next(iter(intersect)).data)
                    tile_count[barcode_i, tile_idx - tile_start_idx] += 1

        # Update peak counts and track FRiP
        if peak_interval is not None:
            in_peak, not_in_peak = update_peak_counts(
                insertions_i, peak_interval, peak_count, barcode_i, peak_start_idx
            )
            total_in_peak += in_peak
            total_not_in_peak += not_in_peak

        if len(insertions_i) >= max_fragment_per_cell:
            n_barcodes_over_limit += 1

        array_value, mask_value = process_barcode_insertions(
            insertions_i, masks_i, max_fragment_per_cell, seed
        )
        chunk_barcode_array[barcode_i] = array_value
        chunk_barcode_mask[barcode_i] = mask_value

    # Calculate FRiP
    total_fragments = total_in_peak + total_not_in_peak
    FRiP = total_in_peak / total_fragments if total_fragments > 0 else 0

    logging.info(
        f"{chromosome} chunk {chunk_idx}: Total fragments: {total_fragments}; "
        f"FRiP: {FRiP * 100:.2f}%; {n_barcodes_over_limit} in {len(pos_dict)} "
        f"barcodes have more than {max_fragment_per_cell} fragments;"
    )

    return (
        chunk_barcode_array,
        tile_count,
        peak_count,
        FRiP,
        library_size,
        chunk_barcode_mask,
    )


def process_barcode_insertions(
    insertions: list[int],
    masks: list[int],
    max_fragments_per_cell: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process insertions for a barcode, applying sampling if needed.

    Args:
        insertions: List of insertion positions
        masks: List of corresponding masks (0 for start, 1 for end)
        max_fragments_per_cell: Maximum number of fragments to keep
        seed: Random seed for sampling

    Returns:
        Tuple of (processed_insertions, processed_masks) as numpy arrays

    Raises:
        ValueError: If insertions and masks have different lengths
    """
    if len(insertions) != len(masks):
        raise ValueError("Insertions and masks must have the same length")

    np.random.seed(seed)

    if len(insertions) >= max_fragments_per_cell:
        # Get indices for random sampling
        sample_indices = np.random.choice(
            len(insertions), max_fragments_per_cell, replace=False
        )
        # Sort the indices to maintain order
        sample_indices.sort()
        # Apply same sampling to both arrays
        array_value = np.array([insertions[i] for i in sample_indices])
        mask_value = np.array([masks[i] for i in sample_indices])
    else:
        # Pad both arrays
        array_value = np.pad(
            insertions,
            (0, max_fragments_per_cell - len(insertions)),
            "constant",
            constant_values=0,
        )
        mask_value = np.pad(
            masks,
            (0, max_fragments_per_cell - len(masks)),
            "constant",
            constant_values=0,
        )

    return array_value, mask_value


def process_chunk_result(
    zarr_root: zarr.Group,
    result: tuple,
    tiles: pd.DataFrame | None,
    peaks: pd.DataFrame | None,
) -> None:
    """
    Process the result of a chunk and store it in the zarr store.

    Args:
        zarr_root: Zarr root group
        result: Tuple containing (chrom, chunk_idx, chunk_tile_count, tile_start_idx,
               chunk_peak_count, peak_start_idx, FRiP, library_size)
        tiles: Optional DataFrame of genome tiles
        peaks: Optional DataFrame of peaks

    Raises:
        ValueError: If result tuple has incorrect format
    """
    try:
        (
            chrom,
            chunk_idx,
            chunk_tile_count,
            tile_start_idx,
            chunk_peak_count,
            peak_start_idx,
            FRiP,
            library_size,
        ) = result
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid chunk result format: {e}")
        raise ValueError("Invalid chunk result format") from e

    # Update per-cell library size
    if "per_cell_libsize" not in zarr_root:
        libsize_data = np.asarray(library_size, dtype="i4")
        libsize_arr = zarr_root.create_array(
            "per_cell_libsize",
            shape=libsize_data.shape,
            dtype="i4",
            compressors=COMPRESSOR,
        )
        libsize_arr[:] = libsize_data
    else:
        current_libsize = zarr_root["per_cell_libsize"][:]
        zarr_root["per_cell_libsize"][:] = current_libsize + library_size

    # Store tile counts for this chunk
    if tiles is not None and chunk_tile_count is not None:
        store_region_and_counts(
            zarr_root=zarr_root,
            group_path="tiles",
            regions=tiles,
            counts=chunk_tile_count.tocsr(),
            chrom=chrom,
            chunk_idx=chunk_idx,
            start_idx=tile_start_idx,
        )

    # Store peak counts for this chunk
    if peaks is not None and chunk_peak_count is not None:
        store_region_and_counts(
            zarr_root=zarr_root,
            group_path="peaks",
            regions=peaks,
            counts=chunk_peak_count.tocsr(),
            chrom=chrom,
            chunk_idx=chunk_idx,
            start_idx=peak_start_idx,
        )

    # Store FRiP score (convert to Python float for JSON serialization)
    if chunk_idx is not None:
        chunk_path = f"chrs/{chrom}/chunk_{chunk_idx}"
        if chunk_path in zarr_root:
            zarr_root[chunk_path].attrs["FRiP"] = float(FRiP)


def concatenate_sparse_chunks(
    zarr_root: zarr.Group,
    chrom: str,
    group_path: str,
    n_cells: int,
) -> csr_matrix | None:
    """
    Concatenate sparse chunk matrices into a single matrix.

    Args:
        zarr_root: Zarr root group
        chrom: Chromosome name
        group_path: Path to group (e.g. 'tiles' or 'peaks')
        n_cells: Number of cells

    Returns:
        Combined sparse matrix or None if no data
    """
    all_rows = []
    all_cols = []
    all_data = []

    chunk_group = zarr_root[f"{group_path}/count_sparse_chunk"][chrom]
    n_chunks = chunk_group.attrs["n_chunks"]

    for chunk_idx in range(n_chunks):
        try:
            start_idx = chunk_group[f"chunk_{chunk_idx}"].attrs["start_idx"]
        except KeyError:
            continue

        if f"chunk_{chunk_idx}" in chunk_group:
            sparse_data = chunk_group[f"chunk_{chunk_idx}"][:]
            if len(sparse_data) > 0:
                all_rows.append(sparse_data[:, 0])
                all_cols.append(sparse_data[:, 1] + start_idx)
                all_data.append(sparse_data[:, 2])

    if not all_data:
        return None

    # Concatenate all arrays
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.concatenate(all_data)

    # Create single CSR matrix
    return csr_matrix(
        (data, (rows, cols)),
        shape=(n_cells, zarr_root[f"{group_path}/regions"][chrom].attrs["n_regions"]),
    )


def finalize_chromosome_processing(
    zarr_root: zarr.Group,
    chrom: str,
    tiles: pd.DataFrame | None,
    peaks: pd.DataFrame | None,
    n_cells: int,
) -> None:
    """
    Finalize chromosome processing by concatenating chunks and cleaning up.

    Args:
        zarr_root: Zarr root group
        chrom: Chromosome name
        tiles: Optional DataFrame of genome tiles
        peaks: Optional DataFrame of peaks
        n_cells: Number of cells
    """
    # Process tiles
    if tiles is not None:
        tile_count_total = concatenate_sparse_chunks(zarr_root, chrom, "tiles", n_cells)
        if tile_count_total is not None:
            store_region_and_counts(
                zarr_root=zarr_root,
                group_path="tiles",
                regions=tiles,
                counts=tile_count_total,
                chrom=chrom,
            )
        gc.collect()

    # Process peaks
    if peaks is not None:
        peak_count_total = concatenate_sparse_chunks(zarr_root, chrom, "peaks", n_cells)
        if peak_count_total is not None:
            store_region_and_counts(
                zarr_root=zarr_root,
                group_path="peaks",
                regions=peaks,
                counts=peak_count_total,
                chrom=chrom,
            )
        gc.collect()


def process_fragments_batch(
    fragment_files: list[str | Path],
    output_dir: str | Path,
    detected_assembly: str | None = None,
    peak_file: str | Path | None = None,
    tile_size: int = 500,
    min_fragments: int = 1000,
    max_fragments: int = 50000,
    max_fragments_per_cell: int = 500,
    chunk_size: int = 2_000_000,
    n_jobs: int = 64,
    skip_exist: bool = True,
    min_cells: int = 100,
    verbose: bool = False,
) -> None:
    """
    Process multiple fragment files into SingleSparseZarrIO format.
    """
    logging.info("Starting sparse zarr generation from fragment files...")

    # Create output directory structure
    output_dir = Path(output_dir)
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Process each fragment file
    for fragment_file in fragment_files:
        fragment_file = Path(fragment_file)
        # Get data_key from filename (first segment before dot)
        filestem = fragment_file.name.split(".")[0]
        zarr_path = sparse_dir / f"{filestem}.zarr"
        try:
            process_fragments(
                fragment_file=fragment_file,
                zarr_path=zarr_path,
                detected_assembly=detected_assembly,
                peak_file=peak_file,
                tile_size=tile_size,
                min_fragments=min_fragments,
                max_fragments=max_fragments,
                max_fragments_per_cell=max_fragments_per_cell,
                chunk_size=chunk_size,
                n_jobs=n_jobs,
                skip_exist=skip_exist,
                min_cells=min_cells,
                verbose=verbose,
            )
        except Exception as e:
            logging.warning(f"Error processing fragment file {fragment_file}: {e}")
            continue

    logging.info("Completed sparse zarr generation.")


def process_clustering(
    zarr_path: str | Path,
    n_components: int = 30,
    device: str = "auto",
    tile_size: int = 500,
    verbose: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Process clustering and create pseudobulks from SingleSparseZarrIO or MultiSparseZarrIO.

    Args:
        zarr_path: Path to input zarr store or directory containing multiple zarr stores
        n_components: Number of SVD components for clustering
        device: Device to use for torch computations ('cuda' or 'cpu')
        tile_size: Size of genome tiles in base pairs
        verbose: If True, show INFO level logging messages
        overwrite: If True, overwrite existing data
    """
    logging.info("Starting clustering...")

    # Initialize zarr store - auto-detect type based on directory structure
    zarr_path = Path(zarr_path)
    if (zarr_path / "chrs").exists():
        # Single zarr store
        z = SingleSparseZarrIO(zarr_path, mode="a")
        is_multi = False
    else:
        # Directory containing multiple zarr stores
        z = MultiSparseZarrIO(zarr_path, mode="a")
        is_multi = True

    # Check assembly information
    if not hasattr(z, "assembly"):
        detected_assembly, _ = determine_genome_assembly(
            fragment_file=None, root_path=".", zarr_path=zarr_path
        )
        if not is_multi:
            z.dataset.attrs["assembly"] = detected_assembly
        else:
            for zarr_store in z.zarr_dict.values():
                zarr_store.dataset.attrs["assembly"] = detected_assembly

    try:
        # Create genome tiles, perform clustering, and save results
        logging.info(
            f"Creating/loading genome tiles with size {tile_size} and performing clustering..."
        )
        adata = z.create_and_cluster_tiles_adata(
            n_components=n_components,
            device=device,
            overwrite=overwrite,
        )

        # Create a DataFrame with cluster assignments that includes data_key
        cluster_assignments = []

        if not is_multi:
            # For single zarr store
            data_key = z.data_key
            for barcode, label in zip(adata.obs.barcodes, adata.obs["labels"]):
                cluster_assignments.append(
                    {"cluster_names": label, "data_key": data_key, "barcodes": barcode}
                )
        else:
            # For multiple zarr stores, use data_key from adata.obs
            cluster_assignments = adata.obs[["labels", "data_key", "barcodes"]].rename(
                columns={"labels": "cluster_names"}
            )
        # Convert to DataFrame and save
        output_dir = zarr_path.parent

        cluster_df = pd.DataFrame(cluster_assignments)
        cluster_df.to_csv(output_dir / "cluster_metadata.csv", index=False)

        # add adata.X_umap to 'umap1' and 'umap2'
        adata.obs["umap1"] = adata.obsm["X_umap"][:, 0]
        adata.obs["umap2"] = adata.obsm["X_umap"][:, 1]
        # Add additional metadata columns if available
        if "umap1" in adata.obs.columns and "umap2" in adata.obs.columns:
            umap_df = adata.obs[["barcodes", "data_key", "umap1", "umap2"]].copy()
            cluster_df = cluster_df.merge(
                umap_df, on=["barcodes", "data_key"], how="left"
            )
        # plot umap scatter plot with leiden color
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            data=cluster_df, x="umap1", y="umap2", hue="cluster_names", palette="Set2"
        )
        plt.savefig(output_dir / "umap_leiden.png")

        # Save cluster assignments to CSV
        cluster_df.to_csv(output_dir / "cluster_metadata.with_umap.csv", index=False)
        # Write cluster labels to individual zarr stores
        if not is_multi:
            # For single zarr store
            all_barcodes = z.metadata.barcodes.values
            cluster_labels = np.array(
                [
                    cluster_df[cluster_df["barcodes"] == bc]["cluster_names"].iloc[0]
                    if bc in cluster_df["barcodes"].values
                    else "cluster_unassigned"
                    for bc in all_barcodes
                ]
            )
            labels_arr = z.dataset.create_array(
                "labels",
                shape=cluster_labels.shape,
                dtype=VariableLengthUTF8(),
                chunks=(1000,),
                compressors=compressor,
                overwrite=True,
            )
            labels_arr[:] = cluster_labels
        else:
            # For multiple zarr stores
            for data_key, zarr_store in z.zarr_dict.items():
                all_barcodes = zarr_store.metadata.barcodes.values
                data_key_clusters = cluster_df[cluster_df["data_key"] == data_key]

                cluster_labels = np.array(
                    [
                        data_key_clusters[data_key_clusters["barcodes"] == bc][
                            "cluster_names"
                        ].iloc[0]
                        if bc in data_key_clusters["barcodes"].values
                        else "cluster_unassigned"
                        for bc in all_barcodes
                    ]
                )
                labels_arr = zarr_store.dataset.create_array(
                    "labels",
                    shape=cluster_labels.shape,
                    dtype=VariableLengthUTF8(),
                    chunks=(1000,),
                    compressors=compressor,
                    overwrite=True,
                )
                labels_arr[:] = cluster_labels

        logging.info("Clustering complete!")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise


def process_pseudobulk_chunk(
    z: SingleSparseZarrIO | MultiSparseZarrIO,
    root: zarr.Group,
    cluster_queries: MultiKeyQuery,
    chrom: str,
    start: int,
    end: int,
) -> dict[str, int]:
    """
    Process a single chunk for pseudobulk creation for all clusters.
    Works with both SingleSparseZarrIO and MultiSparseZarrIO.

    Args:
        z: SingleSparseZarrIO or MultiSparseZarrIO instance
        root: Zarr root group for output
        cluster_queries: A MultiKeyQuery object with cluster names as keys.
        chrom: Chromosome name
        start: Start position
        end: End position

    Returns:
        Dictionary mapping cluster names to their chunk library sizes

    Raises:
        ValueError: If chunk processing fails
        zarr.errors.ZarrError: If zarr operations fail
    """
    try:
        chunk_libsizes = {}

        # The new query system handles both single and multi zarr transparently
        tracks = z.get_track(
            chr_name=chrom,
            start=start,
            end=end,
            query=cluster_queries,
            aggregator=ByKeyAggregator(),
            output_format="raw_dict",
        )

        # Store tracks and calculate library sizes
        for cluster_name, signal in tracks.items():
            signal_path = f"{cluster_name}/chrs/{chrom}"
            if signal_path in root:
                root[signal_path][start:end] = signal
            chunk_libsizes[cluster_name] = int(signal.sum())

        return chunk_libsizes

    except Exception as e:
        logging.error(
            f"Failed to process chunk for clusters at {chrom}:{start}-{end}: {e}"
        )
        raise


def process_pseudobulk(
    zarr_path: str | Path,
    dense_dir: str | Path,
    n_jobs: int = 1,
    min_libsize: int = 3_000_000,
    verbose: bool = False,
    cluster_metadata: str | Path | pd.DataFrame | None = None,
) -> dict[str, dict[str, int]]:
    """
    Process pseudobulks for all clusters in parallel across chunks.
    Can use either provided cluster metadata or read from zarr stores directly.
    Only processes clusters with library size larger than 3M and skips the 'unassigned' cluster.

    Args:
        zarr_path: Path to zarr file/directory
        dense_dir: Directory to store dense zarr files
        n_jobs: Number of parallel jobs
        verbose: If True, show detailed progress
        min_libsize: Minimum library size per pseudobulk
        cluster_metadata: Optional path to cluster metadata CSV or DataFrame with columns:
                        ["cluster_names", "data_key", "barcodes"]

    Returns:
        Dictionary mapping cluster IDs to their library sizes
    """
    logging.info("Starting pseudobulk processing...")

    # Auto-detect zarr type and initialize
    zarr_path = Path(zarr_path)
    if (zarr_path / "chrs").exists():
        # Single zarr store
        z = SingleSparseZarrIO(zarr_path, mode="r")
    else:
        # Directory containing multiple zarr stores
        z = MultiSparseZarrIO(zarr_path, mode="r")

    # Set up dense directory path
    if dense_dir is None:
        dense_dir = zarr_path.parent / "dense"
    dense_dir = Path(dense_dir)
    dense_dir.mkdir(exist_ok=True)
    dense_path = dense_dir / "celltype_dense.zarr"

    # Initialize root group
    root = zarr.open_group(str(dense_path), mode="a")

    try:
        # Get cluster queries based on input
        if cluster_metadata is not None:
            # Use provided cluster metadata
            if isinstance(cluster_metadata, str | Path):
                cluster_df = pd.read_csv(cluster_metadata)
            else:
                cluster_df = cluster_metadata

            # Validate columns
            required_cols = ["cluster_names", "data_key", "barcodes"]
            if not all(col in cluster_df.columns for col in required_cols):
                raise ValueError(
                    f"Cluster metadata must contain columns: {required_cols}"
                )

            queries = {}
            for cluster_name in cluster_df["cluster_names"].unique():
                if cluster_name == "unassigned":
                    logging.info("Skipping 'unassigned' cluster")
                    continue
                queries[cluster_name] = MetadataQuery(f"labels == '{cluster_name}'")
            cluster_queries = MultiKeyQuery(queries)

        else:
            # Use zarr store's cluster information
            labels = z.metadata.labels.unique()
            queries = {
                label: MetadataQuery(f"labels == '{label}'")
                for label in labels
                if label != "unassigned"
            }
            cluster_queries = MultiKeyQuery(queries)

        logging.info(f"Processing clusters: {list(cluster_queries.mapping.keys())}")

        # Initialize arrays for each cluster and chromosome
        for cluster_name in cluster_queries.mapping:
            if cluster_name not in root:
                root.create_group(cluster_name)
            if "chrs" not in root[cluster_name]:
                root[cluster_name].create_group("chrs")

            for chrom in z.chroms:
                if chrom not in root[cluster_name]["chrs"]:
                    root[cluster_name]["chrs"].create_array(
                        chrom,
                        shape=(z.chrom_sizes[chrom],),
                        chunks=(z.chunk_size,),
                        dtype="u4",
                        compressors=compressor,
                    )

        # Process chunks in parallel
        futures = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for chrom in z.chroms:
                chrom_length = z.chrom_sizes[chrom]
                for start in range(0, chrom_length, z.chunk_size):
                    end = min(start + z.chunk_size, chrom_length)
                    futures.append(
                        executor.submit(
                            process_pseudobulk_chunk,
                            z=z,
                            root=root,
                            cluster_queries=cluster_queries,
                            chrom=chrom,
                            start=start,
                            end=end,
                        )
                    )

            # Collect results
            failed_chunks = []
            libsizes = {cluster_name: 0 for cluster_name in cluster_queries.mapping}

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing chunks"
            ):
                try:
                    chunk_libsizes = future.result()
                    for cluster_name, libsize in chunk_libsizes.items():
                        libsizes[cluster_name] += libsize
                except Exception as e:
                    logging.error(f"Failed to process chunk: {e}")
                    failed_chunks.append(len(failed_chunks))

            if failed_chunks:
                raise RuntimeError(f"Failed to process chunks: {failed_chunks}")

        # Filter clusters with library size less than 3M
        filtered_libsizes = {}
        for cluster_name, libsize in libsizes.items():
            if libsize >= min_libsize:
                filtered_libsizes[cluster_name] = libsize
            else:
                logging.info(
                    f"Skipping cluster {cluster_name} with library size {libsize} < 3M"
                )
                # Remove the cluster from root if it was created
                if cluster_name in root:
                    del root[cluster_name]

        # Store metadata only for clusters that meet the library size threshold
        # Convert values to Python native types for JSON serialization
        for cluster_name, libsize in filtered_libsizes.items():
            root[cluster_name].attrs["library_size"] = int(libsize)
            root[cluster_name].attrs["celltype"] = str(cluster_name)
            root[cluster_name].attrs["assembly"] = str(z.assembly)
            root[cluster_name].attrs["completed"] = True

        # Store global metadata
        root.attrs.update(
            {
                "n_clusters": int(len(filtered_libsizes)),
                "cluster_ids": sorted(list(filtered_libsizes.keys())),
                "library_size": sum(filtered_libsizes.values()),
                "assembly": z.assembly,
                "completed": True,
            }
        )

        logging.info(
            f"Completed pseudobulk processing for {len(filtered_libsizes)} clusters."
        )

        return filtered_libsizes

    except Exception as e:
        logging.warning(f"Error during pseudobulk processing: {e}")


def normalize_pseudobulk(
    dense_path: str | Path,
    peak_file: str | Path = None,
    n_jobs: int = 1,
    output_zarr_path: str | Path = None,
) -> dict:
    """
    Normalize pseudobulk signals using quantile normalization.

    Args:
        dense_path: Path to the dense zarr directory containing pseudobulks
        peak_file: Path to peak file for normalization
        blacklist_file: Path to blacklist file, defaults to assembly-specific blacklist
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary containing normalization parameters
    """
    logging.info("Starting signal normalization...")

    # Initialize CelltypeDenseZarrIO
    cdz = CelltypeDenseZarrIO(path=str(dense_path), mode="a")

    # Calculate normalization parameters using existing function
    norm_results = cdz.quantile_normalization_peak_based(
        peak_file=str(peak_file) if peak_file else None,
        n_jobs=n_jobs,
        conv_size=20,
        output_zarr_path=str(output_zarr_path) if output_zarr_path else None,
    )
    logging.info("Completed normalization.")
    return norm_results


def process_callpeak(output_dense: str | Path) -> dict:
    """
    Process calling peaks for each cluster.

    Args:
        zarr_path: Path to input zarr store

    Returns:
        Dictionary mapping cluster IDs to their peaks

    Raises:
        RuntimeError: If peak calling fails
    """
    logging.info("Starting peak calling...")
    try:
        cdz = CelltypeDenseZarrIO(output_dense, mode="a")
        peaks_dict = cdz.call_peaks(
            track_id=cdz.ids,
            normalize_factor=cdz.libsize,
            aggregator="by_key",
        )

        for cluster_id in cdz.ids:
            try:
                cdz.write_peaks(
                    id=cluster_id,
                    peaks=peaks_dict[cluster_id],
                    name="cluster_peaks",
                    overwrite=True,
                )
            except Exception as e:
                logging.error(f"Error processing peaks for cluster {cluster_id}: {e}")

        logging.info("Completed peak calling.")
        return peaks_dict

    except Exception as e:
        logging.warning(f"Error initializing CelltypeDenseZarrIO: {e}")


def process_footprint(output_dense: str | Path, motif_zarr: str | Path) -> None:
    """
    Process footprint detection using motif information.

    Args:
        zarr_path: Path to input zarr store
        motif_zarr: Path to motif zarr store

    Raises:
        RuntimeError: If footprint detection fails
    """
    if not motif_zarr or not Path(motif_zarr).exists():
        logging.info("Skipping footprint detection - no motif zarr provided or found")
        return

    logging.info("Running footprint detection...")
    try:
        cdz = CelltypeDenseZarrIO(output_dense, mode="a")
        m = MotifDenseZarrIO(str(motif_zarr), dtype="float32", mode="r")

        for cluster_id in cdz.ids:
            try:
                cdz.get_genome_wide_footprint_all_motifs(
                    m,
                    cutoff=np.arange(5, 6, 1),
                    extend=400,
                    query=KeyQuery([cluster_id]),
                    aggregator=SumAggregator(),
                )
            except Exception as e:
                logging.error(f"Error during footprint detection for {cluster_id}: {e}")

        logging.info("Footprint detection completed successfully")

    except Exception as e:
        logging.error(f"Error during footprint detection: {e}")
        raise RuntimeError("Footprint detection failed") from e


def archived_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process single-cell accessibility data"
    )
    parser.add_argument(
        "--fragment-files", nargs="+", required=True, help="One or more fragment files"
    )
    parser.add_argument(
        "--peak-file",
        help="Path to peak file for normalization (optional)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for zarr stores"
    )

    parser.add_argument("--cluster-assignments", help="Path to cluster assignments CSV")

    parser.add_argument("--detected-assembly", help="Detected assembly")
    parser.add_argument("--motif-zarr", help="Path to motif zarr")
    parser.add_argument("--min-fragments", type=int, default=1000)
    parser.add_argument("--max-fragments", type=int, default=50000)
    parser.add_argument(
        "--min-cells",
        type=int,
        default=100,
        help="Minimum number of cells per fragment file",
    )
    parser.add_argument(
        "--min-libsize-per-pseudobulk",
        type=int,
        default=3000000,
        help="Minimum library size per pseudobulk",
    )
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument(
        "--tile-size", type=int, default=500, help="Size of genome tiles in base pairs"
    )
    parser.add_argument(
        "--skip-exist", action="store_true", default=False, help="Skip existing chunks"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Show detailed progress"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=30,
        help="Number of SVD components for clustering",
    )
    parser.add_argument(
        "--device", default="auto", help="Device for clustering ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--skip-fragments",
        action="store_true",
        default=False,
        help="Skip fragment processing",
    )

    parser.add_argument(
        "--max-fragments-per-cell",
        type=int,
        default=500,
        help="Maximum fragments per cell",
    )
    parser.add_argument(
        "--dense-dir",
        help="Directory to store dense pseudobulk zarr files",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing dense zarr store",
    )
    args = parser.parse_args()

    # Configure logging based on verbosity
    # setup_logging(args.verbose)

    # First create the zarr store from fragments
    if not args.skip_fragments:
        process_fragments_batch(
            fragment_files=args.fragment_files,
            output_dir=args.output_dir,
            detected_assembly=args.detected_assembly,
            peak_file=args.peak_file,
            tile_size=args.tile_size,
            min_fragments=args.min_fragments,
            max_fragments=args.max_fragments,
            max_fragments_per_cell=args.max_fragments_per_cell,
            chunk_size=args.chunk_size,
            n_jobs=args.n_jobs,
            skip_exist=args.skip_exist,
            min_cells=args.min_cells,
            verbose=args.verbose,
        )

    # Then do clustering
    process_clustering(
        zarr_path=Path(args.output_dir) / "sparse",
        n_components=args.n_components,
        device=args.device,
        tile_size=args.tile_size,
        verbose=args.verbose,
        overwrite=False,
    )

    # Then do pseudobulk
    process_pseudobulk(
        zarr_path=Path(args.output_dir) / "sparse",
        dense_dir=Path(args.output_dir) / "dense",
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        min_libsize=args.min_libsize,
        cluster_metadata=(
            Path(args.output_dir) / "cluster_metadata.with_umap.csv"
            if (Path(args.output_dir) / "cluster_metadata.with_umap.csv").exists()
            else None
        ),
    )

    # Finally, call peaks and run footprint detection
    process_callpeak(
        output_dense=Path(args.output_dir) / "dense/celltype_dense.zarr",
    )

    # If peak file is provided, normalize the signals
    if args.peak_file is not None:
        output_zarr_path = Path(args.output_dir) / "dense/celltype_dense_norm.zarr"
        normalize_pseudobulk(
            dense_path=Path(args.output_dir) / "dense/celltype_dense.zarr",
            peak_file=args.peak_file,
            n_jobs=args.n_jobs,
            output_zarr_path=output_zarr_path,
        )

    if args.motif_zarr:
        logging.info("Starting footprint detection...")
        process_footprint(
            output_dense=Path(args.output_dir) / "dense/celltype_dense.zarr",
            motif_zarr=args.motif_zarr,
        )
        logging.info("Completed footprint detection.")
