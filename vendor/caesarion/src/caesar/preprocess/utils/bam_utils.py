"""BAM file processing utilities.

This module contains functions for processing aligned BAM files
and extracting genomic positions with optional Tn5 shifting.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pysam
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from caesar.preprocess.utils.fragment_utils import (
    apply_tn5_shift,
    filter_positions_by_blacklist,
    process_chromosome_coverage,
)
from gcell.dna.genome import ChromSize, read_blacklist

try:
    import bpcells
    from bpcells.experimental import precalculate_insertion_counts_bam
    BPCells_AVAILABLE = True
except ImportError:
    BPCells_AVAILABLE = False


def ensure_bam_indexed(bam_file: str | Path) -> None:
    """Ensure BAM file has an index, creating one if it doesn't exist.

    Args:
        bam_file: Path to BAM file

    Raises:
        pysam.SamtoolsError: If index creation fails
    """
    bam_path = Path(bam_file)
    index_path = bam_path.with_suffix(bam_path.suffix + ".bai")

    # Check if index exists
    if not index_path.exists():
        logging.info(f"BAM index not found. Creating index for {bam_path}...")
        try:
            pysam.index(str(bam_path))
            logging.info(f"Successfully created index: {index_path}")
        except pysam.SamtoolsError as e:
            logging.error(f"Failed to create BAM index: {e}")
            raise
    else:
        logging.debug(f"BAM index found: {index_path}")


def extract_positions_from_bam(
    bam_file: str | Path,
    chrom: str,
    chunk_start: int,
    chunk_end: int,
    single_end: bool = False,
    signal_mode: str = "tn5",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract positions from BAM file based on signal mode.

    Args:
        bam_file: Path to BAM file
        chrom: Chromosome name
        chunk_start: Start position of chunk
        chunk_end: End position of chunk
        single_end: Whether data is single-end
        signal_mode: Signal extraction mode:
            - "tn5": Extract cut sites with Tn5 bias correction (ATAC-seq/CUT&Tag)
            - "chip": Use entire read/fragment span as coverage (ChIP-seq)

    Returns:
        Tuple of (positions, strands) where strands is None for paired-end or chip mode
    """
    positions = []
    strands = [] if (single_end and signal_mode == "tn5") else None

    with pysam.AlignmentFile(str(bam_file), "rb") as bamfile:
        for read in bamfile.fetch(chrom, chunk_start, chunk_end):
            # Skip unmapped reads
            if read.is_unmapped:
                continue

            # Skip secondary and supplementary alignments
            if read.is_secondary or read.is_supplementary:
                continue

            if signal_mode == "chip":
                # ChIP mode: use entire read/fragment span
                if single_end:
                    # For single-end ChIP, add all positions covered by the read
                    read_start = read.reference_start
                    read_end = read.reference_end
                    if read_start is not None and read_end is not None:
                        # Add both start and end to represent the span
                        # These will be used to create coverage over the entire read
                        positions.extend(range(read_start, read_end))
                else:
                    # For paired-end ChIP, use the fragment span
                    if read.is_proper_pair and read.is_read1:
                        frag_start = read.reference_start
                        if read.template_length > 0:
                            frag_end = read.reference_start + abs(read.template_length)
                        else:
                            # Reverse orientation
                            frag_end = read.next_reference_start + abs(read.template_length) if read.next_reference_start else read.reference_end
                        if frag_start is not None and frag_end is not None:
                            # Add all positions in the fragment span
                            positions.extend(range(frag_start, frag_end))
            else:
                # Tn5 mode: extract cut sites (original behavior)
                if single_end:
                    # For single-end, use read start/end based on strand
                    if read.is_reverse:  # - strand
                        positions.append(read.reference_end)
                        strands.append(False)  # False for - strand
                    else:  # + strand
                        positions.append(read.reference_start)
                        strands.append(True)  # True for + strand
                else:
                    # For paired-end, extract both ends
                    if read.is_proper_pair and read.is_read1:
                        # Only process read1 to avoid double counting
                        # Add both start and end of the fragment
                        positions.append(read.reference_start)
                        # For the mate, calculate the end position correctly
                        if read.template_length > 0:
                            # Forward orientation: template_length includes the full fragment
                            positions.append(
                                read.reference_start + abs(read.template_length) - 1
                            )
                        else:
                            # Reverse orientation: use mate's start position
                            positions.append(read.next_reference_start)

    positions = np.array(positions, dtype=np.int32)

    if single_end and strands is not None and signal_mode == "tn5":
        strands = np.array(strands, dtype=bool)

    # Apply position shifting only for Tn5 mode
    if signal_mode == "tn5":
        positions = apply_tn5_shift(
            positions, strands, single_end=single_end, tn5_shift=True
        )

    return positions, strands


def process_bam_chunk(args: tuple) -> tuple[str, int, int, np.ndarray]:
    """Process a single chunk of BAM file.

    Args:
        args: Tuple containing (bam_file, chrom, chunk_start, chunk_end, single_end, signal_mode)

    Returns:
        Tuple of (chrom, chunk_start, chunk_end, positions)
    """
    bam_file, chrom, chunk_start, chunk_end, single_end, signal_mode = args

    positions, _ = extract_positions_from_bam(
        bam_file, chrom, chunk_start, chunk_end, single_end, signal_mode
    )

    # Filter positions to be within chunk boundaries
    mask = (positions >= chunk_start) & (positions < chunk_end)
    positions = positions[mask]

    return chrom, chunk_start, chunk_end, positions


def get_bam_chromosomes(bam_file: str | Path) -> dict[str, int]:
    """Get chromosome names and sizes from BAM file.

    Args:
        bam_file: Path to BAM file

    Returns:
        Dictionary mapping chromosome names to sizes
    """
    chrom_sizes = {}

    with pysam.AlignmentFile(str(bam_file), "rb") as bamfile:
        for ref, length in zip(bamfile.references, bamfile.lengths):
            chrom_sizes[ref] = length

    return chrom_sizes


def process_bam_to_zarr(
    bam_file: str,
    output_zarr: str,
    sample_id: str,
    celltype: str,
    assembly: str,
    single_end: bool = False,
    signal_mode: str = "tn5",
    chunk_size: int = 2_000_000,
    n_jobs: int = 4,
    chr_filter_config: dict | None = None,
) -> int:
    """Process BAM file to zarr format with parallel processing.

    Args:
        bam_file: Path to BAM file
        output_zarr: Output zarr path
        sample_id: Sample ID
        celltype: Cell type
        assembly: Genome assembly
        single_end: Whether data is single-end
        signal_mode: Signal extraction mode:
            - "tn5": Extract cut sites with Tn5 bias correction (ATAC-seq/CUT&Tag)
            - "chip": Use entire read/fragment span as coverage (ChIP-seq)
        chunk_size: Chunk size for processing
        n_jobs: Number of parallel jobs
        chr_filter_config: Chromosome filter configuration

    Returns:
        Total library size (number of reads/fragments)
    """
    logging.info(f"Processing BAM file: {bam_file}")

    # Ensure BAM file is indexed (required for fetch operations)
    ensure_bam_indexed(bam_file)

    # Get chromosome sizes from BAM
    bam_chroms = get_bam_chromosomes(bam_file)

    # Get reference chromosome sizes
    chrom_sizes_ref = ChromSize(assembly).dict

    # Load blacklist
    blacklist = read_blacklist(assembly)

    # Create or open zarr store
    zarr_root = zarr.open(output_zarr, mode="a")

    # Always use celltype.id group structure for consistency
    group_name = f"{celltype}.{sample_id}"

    # Create group and set attributes
    group = zarr_root.require_group(group_name)
    group.attrs["id"] = sample_id
    group.attrs["celltype"] = celltype
    group.attrs["assembly"] = assembly
    group.attrs["chunk_size"] = chunk_size

    # Also set/update root attributes for CelltypeDenseZarrIO compatibility
    if "assembly" not in zarr_root.attrs:
        zarr_root.attrs["assembly"] = assembly

    # Update the list of celltype IDs
    existing_ids = list(zarr_root.attrs.get("celltype_ids", []))
    if group_name not in existing_ids:
        existing_ids.append(group_name)
        zarr_root.attrs["celltype_ids"] = existing_ids

    # Initialize library size
    library_size = 0

    # Determine chromosomes to process
    if chr_filter_config and hasattr(chr_filter_config, "get"):
        from caesar.preprocess.core.chromosome_filter import get_chromosomes_to_process

        chroms_list = get_chromosomes_to_process(
            chr_filter_config, list(chrom_sizes_ref.keys()), set(bam_chroms.keys())
        )
    else:
        # Default filtering using shared logic
        from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

        chr_filter = ChromosomeFilter({})
        chroms_list = [
            chrom
            for chrom in chrom_sizes_ref
            if (chrom in bam_chroms) and chr_filter.is_valid_chromosome(chrom)
        ]

    # Prepare chunk arguments for parallel processing
    chunk_args = []
    for chrom in chroms_list:
        chrom_length = bam_chroms.get(chrom, chrom_sizes_ref.get(chrom, 0))
        if chrom_length == 0:
            continue

        for chunk_start in range(0, chrom_length, chunk_size):
            chunk_end = min(chunk_start + chunk_size, chrom_length)
            chunk_args.append(
                (bam_file, chrom, chunk_start, chunk_end, single_end, signal_mode)
            )

    # Process chunks in parallel
    all_chrom_positions = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_bam_chunk, args) for args in chunk_args]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing BAM chunks"
        ):
            try:
                chrom, chunk_start, chunk_end, positions = future.result()

                if chrom not in all_chrom_positions:
                    all_chrom_positions[chrom] = []

                all_chrom_positions[chrom].append(positions)

            except Exception as e:
                logging.error(f"Error processing chunk: {e}")
                continue

    # Process each chromosome
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    for chrom in tqdm(chroms_list, desc="Writing chromosomes to zarr"):
        if chrom not in all_chrom_positions:
            continue

        # Concatenate all positions for this chromosome
        positions = np.concatenate(all_chrom_positions[chrom])

        if positions.size == 0:
            continue

        # Apply blacklist filtering
        positions = filter_positions_by_blacklist(positions, blacklist, chrom)

        if positions.size == 0:
            continue

        # Update library size
        if single_end:
            library_size += positions.size
        else:
            library_size += positions.size // 2  # Paired-end counts as fragments

        # Process and write chromosome coverage
        process_chromosome_coverage(
            zarr_root=zarr_root,
            group_name=group_name,
            chrom=chrom,
            positions=positions,
            chrom_size=chrom_sizes_ref[chrom],
            chunk_size=chunk_size,
            compressor=compressor,
        )

    # Set library size as group attribute
    group.attrs["library_size"] = library_size

    logging.info(f"Done. Dense coverage zarr written to: {output_zarr}")
    logging.info(f"Total library size: {library_size}")

    return library_size


def process_bam_to_bpcells(
    bam_file: str,
    output_dir: str,
    sample_id: str,
    celltype: str,
    assembly: str,
    signal_mode: str = "tn5",
    threads: int = 0,
    chr_filter_config: dict | None = None,
) -> int:
    """Process BAM file to BPCells format using precalculate_insertion_counts_bam.

    This function processes bulk BAM files (without CB tags) directly to BPCells
    celltype_dense format. For bulk data, a dummy "bulk" barcode is automatically
    assigned.

    Args:
        bam_file: Path to BAM file (must be coordinate-sorted)
        output_dir: Output directory path where BPCells matrix will be saved
        sample_id: Sample ID
        celltype: Cell type label
        assembly: Genome assembly (e.g., "hg38", "mm10")
        signal_mode: Signal extraction mode:
            - "tn5": Extract cut sites with Tn5 bias correction (ATAC-seq/CUT&Tag)
            - "chip": Use entire read/fragment span as coverage (ChIP-seq)
        threads: Number of parallel threads (0 = use all available CPUs)
        chr_filter_config: Chromosome filter configuration (optional)

    Returns:
        Total library size (number of fragments)

    Raises:
        ImportError: If BPCells is not installed
        RuntimeError: If BAM file cannot be processed
    """
    if not BPCells_AVAILABLE:
        raise ImportError(
            "BPCells is required for BPCells backend. Install with: pip install bpcells"
        )

    logging.info(f"Processing BAM file to BPCells format: {bam_file}")

    # Get chromosome sizes from assembly
    chrom_sizes_ref = ChromSize(assembly).dict

    # Get chromosomes from BAM file
    bam_chroms = get_bam_chromosomes(bam_file)

    # Determine chromosomes to process (apply filtering if provided)
    if chr_filter_config and hasattr(chr_filter_config, "get"):
        from caesar.preprocess.core.chromosome_filter import get_chromosomes_to_process

        chroms_list = get_chromosomes_to_process(
            chr_filter_config, list(chrom_sizes_ref.keys()), set(bam_chroms.keys())
        )
    else:
        # Default filtering using shared logic
        from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

        chr_filter = ChromosomeFilter({})
        chroms_list = [
            chrom
            for chrom in chrom_sizes_ref
            if (chrom in bam_chroms) and chr_filter.is_valid_chromosome(chrom)
        ]

    # Filter chrom_sizes to only include chromosomes we'll process
    chrom_sizes = {chrom: chrom_sizes_ref[chrom] for chrom in chroms_list}

    # Extract file prefix for bulk barcode naming
    bam_basename = os.path.splitext(os.path.basename(bam_file))[0]
    # Remove common suffixes
    for suffix in ['.sorted', '.dedup', '.filtered', '.bam']:
        if bam_basename.endswith(suffix):
            bam_basename = bam_basename[:-len(suffix)]
    bulk_barcode = f"bulk.{bam_basename}"

    # For bulk BAM files (no CB tags), use bulk.FILE_PREFIX barcode
    # All fragments will be assigned to group 0
    cell_groups = {bulk_barcode: 0}

    # Determine group name - use bulk_barcode as default (can be overridden via group_names dict)
    # The barcode is 'bulk.FILE_PREFIX', so we use it directly as group name
    group_name = bulk_barcode  # Default: use barcode as group name

    # Set shift parameters based on signal mode
    if signal_mode == "tn5":
        # Tn5 mode: ATAC-seq/CUT&Tag - apply +4bp start, -5bp end shift
        shift_start = 4
        shift_end = -5
    else:
        # ChIP mode: no shift
        # NOTE: BPCells backend currently counts insertion sites, not full coverage.
        # For ChIP-seq, no shift is needed
        shift_start = 0
        shift_end = 0

    # Convert signal_mode string to integer (0 = tn5, 1 = chip)
    signal_mode_int = 1 if signal_mode == "chip" else 0

    # Process BAM file using BPCells
    logging.info(f"Calling precalculate_insertion_counts_bam with:")
    logging.info(f"  BAM file: {bam_file}")
    logging.info(f"  Output dir: {output_dir}")
    logging.info(f"  Chromosomes: {len(chrom_sizes)}")
    logging.info(f"  Group name: {group_name}")
    logging.info(f"  Signal mode: {signal_mode} (signal_mode_int={signal_mode_int}, shift_start={shift_start}, shift_end={shift_end})")

    matrix = precalculate_insertion_counts_bam(
        bam_file=bam_file,
        output_dir=output_dir,
        cell_groups=cell_groups,
        chrom_sizes=chrom_sizes,
        shift_start=shift_start,
        shift_end=shift_end,
        threads=threads,
        group_names={bulk_barcode: group_name},  # Map barcode to group name
        signal_mode=signal_mode_int,  # 0 = tn5 (insertion), 1 = chip (coverage)
    )

    # Get library size from the matrix metadata
    # The library_size.json file should have been created by precalculate_insertion_counts_bam
    import json

    library_size_path = os.path.join(output_dir, "library_size.json")
    if os.path.exists(library_size_path):
        with open(library_size_path, "r") as f:
            library_size_data = json.load(f)
            # library_size.json is a dict mapping group names to library sizes
            library_size = library_size_data.get(group_name, 0)
    else:
        # Fallback: try to get from matrix if available
        try:
            library_size = int(matrix.libsize) if hasattr(matrix, "libsize") else 0
        except Exception:
            logging.warning("Could not determine library size from BPCells output")
            library_size = 0

    logging.info(f"Done. BPCells matrix written to: {output_dir}")
    logging.info(f"Total library size: {library_size}")

    return library_size


def process_bam_multi_to_bpcells(
    bam_files: List[str],
    output_dir: str,
    sample_ids: List[str],
    celltypes: List[str],
    assembly: str,
    signal_mode: str = "tn5",
    threads: int = 0,
    chr_filter_config: dict | None = None,
) -> Dict[str, int]:
    """Process multiple BAM files to BPCells format in a single call.

    This function processes multiple bulk BAM files (without CB tags) directly to BPCells
    celltype_dense format in a single matrix. Each BAM file gets a unique cell prefix
    (e.g., "bulk.FILENAME") to distinguish cells.

    Args:
        bam_files: List of paths to BAM files (must be coordinate-sorted)
        output_dir: Output directory path where BPCells matrix will be saved
        sample_ids: List of sample IDs, one per BAM file
        celltypes: List of cell type labels, one per BAM file
        assembly: Genome assembly (e.g., "hg38", "mm10")
        signal_mode: Signal extraction mode:
            - "tn5": Extract cut sites with Tn5 bias correction (ATAC-seq/CUT&Tag)
            - "chip": Use entire read/fragment span as coverage (ChIP-seq)
        threads: Number of parallel threads (0 = use all available CPUs)
        chr_filter_config: Chromosome filter configuration (optional)

    Returns:
        Dictionary mapping group names to library sizes

    Raises:
        ImportError: If BPCells is not installed
        RuntimeError: If BAM files cannot be processed
    """
    if not BPCells_AVAILABLE:
        raise ImportError(
            "BPCells is required for BPCells backend. Install with: pip install bpcells"
        )

    if len(bam_files) != len(sample_ids) or len(bam_files) != len(celltypes):
        raise ValueError("bam_files, sample_ids, and celltypes must have the same length")

    logging.info(f"Processing {len(bam_files)} BAM files to BPCells format in single call")

    # Get chromosome sizes from assembly
    chrom_sizes_ref = ChromSize(assembly).dict

    # Get chromosomes from first BAM file (all should have same chromosomes)
    bam_chroms = get_bam_chromosomes(bam_files[0])

    # Determine chromosomes to process (apply filtering if provided)
    if chr_filter_config and hasattr(chr_filter_config, "get"):
        from caesar.preprocess.core.chromosome_filter import get_chromosomes_to_process

        chroms_list = get_chromosomes_to_process(
            chr_filter_config, list(chrom_sizes_ref.keys()), set(bam_chroms.keys())
        )
    else:
        # Default filtering using shared logic
        from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

        chr_filter = ChromosomeFilter({})
        chroms_list = [
            chrom
            for chrom in chrom_sizes_ref
            if (chrom in bam_chroms) and chr_filter.is_valid_chromosome(chrom)
        ]

    # Filter chrom_sizes to only include chromosomes we'll process
    chrom_sizes = {chrom: chrom_sizes_ref[chrom] for chrom in chroms_list}

    # Set shift parameters based on signal mode
    if signal_mode == "tn5":
        # Tn5 mode: ATAC-seq/CUT&Tag - apply +4bp start, -5bp end shift
        shift_start = 4
        shift_end = -5
    else:
        # ChIP mode: no shift
        shift_start = 0
        shift_end = 0

    # Convert signal_mode string to integer (0 = tn5, 1 = chip)
    signal_mode_int = 1 if signal_mode == "chip" else 0

    # Extract file prefixes for bulk barcodes
    bam_barcodes = []
    for bam_file in bam_files:
        bam_basename = os.path.splitext(os.path.basename(bam_file))[0]
        # Remove common suffixes
        for suffix in ['.sorted', '.dedup', '.filtered', '.bam']:
            if bam_basename.endswith(suffix):
                bam_basename = bam_basename[:-len(suffix)]
        bam_barcodes.append(f"bulk.{bam_basename}")

    # Determine group names: use celltype.sample_id format, mapped from barcodes
    # Format: {'bulk.FILE_PREFIX': 'celltype.sample_id'}
    group_names_dict = {
        barcode: f"{celltype}.{sample_id}"
        for barcode, celltype, sample_id in zip(bam_barcodes, celltypes, sample_ids)
    }

    # Process all BAM files using multi-BAM function
    logging.info(f"Calling precalculate_insertion_counts_bam_multi with:")
    logging.info(f"  BAM files: {len(bam_files)}")
    logging.info(f"  Output dir: {output_dir}")
    logging.info(f"  Chromosomes: {len(chrom_sizes)}")
    logging.info(f"  Group names mapping: {group_names_dict}")
    logging.info(f"  Signal mode: {signal_mode} (signal_mode_int={signal_mode_int}, shift_start={shift_start}, shift_end={shift_end})")

    from bpcells.experimental import precalculate_insertion_counts_bam_multi

    matrix = precalculate_insertion_counts_bam_multi(
        bam_files=bam_files,
        output_dir=output_dir,
        chrom_sizes=chrom_sizes,
        shift_start=shift_start,
        shift_end=shift_end,
        threads=threads,
        group_names=group_names_dict,  # Pass dict mapping barcodes to group names
        signal_mode=signal_mode_int,  # 0 = tn5 (insertion), 1 = chip (coverage)
    )

    # Get library sizes from the matrix metadata
    import json

    library_sizes = {}
    library_size_path = os.path.join(output_dir, "library_size.json")
    if os.path.exists(library_size_path):
        with open(library_size_path, "r") as f:
            library_size_data = json.load(f)
            # library_size.json format: {"group_name": size, ...}
            library_sizes = library_size_data

    logging.info(f"Done. BPCells matrix written to: {output_dir}")
    logging.info(f"Library sizes: {library_sizes}")

    return library_sizes
