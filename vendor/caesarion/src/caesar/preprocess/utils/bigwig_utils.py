"""BigWig file processing utilities.

This module contains functions for processing BigWig files
and converting them to Zarr format.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyBigWig
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from gcell.dna.genome import ChromSize


def get_bigwig_chromosomes(bigwig_file: str | Path) -> dict[str, int]:
    """Get chromosome names and sizes from BigWig file.

    Args:
        bigwig_file: Path to BigWig file

    Returns:
        Dictionary mapping chromosome names to sizes
    """
    bw = pyBigWig.open(str(bigwig_file))
    chroms = dict(bw.chroms())
    bw.close()
    return chroms


def read_bigwig_region(
    bigwig_file: str | Path,
    chrom: str,
    start: int,
    end: int,
) -> np.ndarray:
    """Read values from a BigWig file for a specific region.

    Args:
        bigwig_file: Path to BigWig file
        chrom: Chromosome name
        start: Start position
        end: End position

    Returns:
        Numpy array of values (NaN replaced with 0)
    """
    bw = pyBigWig.open(str(bigwig_file))
    try:
        values = bw.values(chrom, start, end)
        # Convert to numpy and replace None/NaN with 0
        values = np.array(values, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0)
    finally:
        bw.close()
    return values


def process_bigwig_chromosome(
    bigwig_file: str,
    chrom: str,
    chrom_size: int,
    chunk_size: int = 2_000_000,
) -> tuple[str, np.ndarray, float]:
    """Process a single chromosome from BigWig file.

    Args:
        bigwig_file: Path to BigWig file
        chrom: Chromosome name
        chrom_size: Size of the chromosome
        chunk_size: Size of chunks for reading (memory optimization)

    Returns:
        Tuple of (chromosome name, coverage array, total signal)
    """
    bw = pyBigWig.open(str(bigwig_file))
    try:
        # Read the entire chromosome in chunks to avoid memory issues
        coverage = np.zeros(chrom_size, dtype=np.float32)

        for start in range(0, chrom_size, chunk_size):
            end = min(start + chunk_size, chrom_size)
            try:
                values = bw.values(chrom, start, end)
                if values is not None:
                    chunk_data = np.array(values, dtype=np.float32)
                    chunk_data = np.nan_to_num(chunk_data, nan=0.0)
                    coverage[start:end] = chunk_data
            except Exception as e:
                logging.warning(f"Error reading {chrom}:{start}-{end}: {e}")
                continue

        total_signal = float(np.sum(coverage))
    finally:
        bw.close()

    return chrom, coverage, total_signal


def process_bigwig_to_signal_dense_zarr(
    bigwig_file: str,
    output_zarr: str,
    sample_id: str,
    assembly: str,
    chunk_size: int = 2_000_000,
    n_jobs: int = 4,
    chr_filter_config: dict | None = None,
) -> float:
    """Process a single BigWig file to SignalDense Zarr format.

    This creates a Zarr store compatible with SignalDenseZarrIO.

    Args:
        bigwig_file: Path to BigWig file
        output_zarr: Output zarr path
        sample_id: Sample ID
        assembly: Genome assembly
        chunk_size: Chunk size for zarr arrays
        n_jobs: Number of parallel jobs
        chr_filter_config: Chromosome filter configuration

    Returns:
        Total signal (sum of all values)
    """
    logging.info(f"Processing BigWig file: {bigwig_file}")

    # Get chromosome sizes from BigWig
    bw_chroms = get_bigwig_chromosomes(bigwig_file)

    # Get reference chromosome sizes
    chrom_sizes_ref = ChromSize(assembly).dict

    # Create zarr store
    zarr_root = zarr.open(output_zarr, mode="w")

    # Set root attributes
    zarr_root.attrs["assembly"] = assembly
    zarr_root.attrs["sample_id"] = sample_id
    zarr_root.attrs["source"] = "bigwig"
    zarr_root.attrs["source_file"] = str(bigwig_file)
    zarr_root.attrs["store_type"] = "signal_dense"
    zarr_root.attrs["class"] = "SignalDenseZarrIO"

    # Create chrs group
    chrs_group = zarr_root.require_group("chrs")

    # Determine chromosomes to process
    if chr_filter_config:
        from caesar.preprocess.core.chromosome_filter import get_chromosomes_to_process

        chroms_list = get_chromosomes_to_process(
            chr_filter_config, list(chrom_sizes_ref.keys()), set(bw_chroms.keys())
        )
    else:
        from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

        chr_filter = ChromosomeFilter({})
        chroms_list = [
            chrom
            for chrom in chrom_sizes_ref
            if (chrom in bw_chroms) and chr_filter.is_valid_chromosome(chrom)
        ]

    # Prepare arguments for parallel processing
    process_args = []
    for chrom in chroms_list:
        chrom_size = bw_chroms.get(chrom, chrom_sizes_ref.get(chrom, 0))
        if chrom_size == 0:
            continue
        process_args.append((bigwig_file, chrom, chrom_size, chunk_size))

    # Process chromosomes in parallel
    total_signal = 0.0
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(process_bigwig_chromosome, *args): args[1]
            for args in process_args
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing chromosomes"
        ):
            try:
                chrom, coverage, chrom_signal = future.result()
                total_signal += chrom_signal

                # Write chromosome data to zarr (zarr v3 compatible)
                chrs_group.create_array(
                    chrom,
                    data=coverage,
                    chunks=(chunk_size,),
                    compressors=compressor,
                    overwrite=True,
                )

            except Exception as e:
                chrom_name = futures[future]
                logging.error(f"Error processing chromosome {chrom_name}: {e}")
                continue

    # Set library size attribute
    zarr_root.attrs["library_size"] = total_signal

    logging.info(f"Processed BigWig to {output_zarr}, total signal: {total_signal:.2f}")
    return total_signal


def process_bigwig_to_celltype_dense_zarr(
    bigwig_files: list[str],
    output_zarr: str,
    sample_ids: list[str],
    celltypes: list[str],
    assembly: str,
    chunk_size: int = 2_000_000,
    n_jobs: int = 4,
    chr_filter_config: dict | None = None,
) -> dict[str, float]:
    """Process multiple BigWig files to CelltypeDense Zarr format.

    This creates a Zarr store compatible with CelltypeDenseZarrIO.

    Args:
        bigwig_files: List of paths to BigWig files
        output_zarr: Output zarr path
        sample_ids: List of sample IDs (one per file)
        celltypes: List of cell type names (one per file)
        assembly: Genome assembly
        chunk_size: Chunk size for zarr arrays
        n_jobs: Number of parallel jobs
        chr_filter_config: Chromosome filter configuration

    Returns:
        Dictionary mapping group names to total signal values
    """
    if len(bigwig_files) != len(sample_ids) or len(bigwig_files) != len(celltypes):
        raise ValueError(
            "bigwig_files, sample_ids, and celltypes must have the same length"
        )

    logging.info(f"Processing {len(bigwig_files)} BigWig files")

    # Get chromosome sizes from first BigWig
    bw_chroms = get_bigwig_chromosomes(bigwig_files[0])

    # Get reference chromosome sizes
    chrom_sizes_ref = ChromSize(assembly).dict

    # Create zarr store
    zarr_root = zarr.open(output_zarr, mode="w")

    # Set root attributes
    zarr_root.attrs["assembly"] = assembly
    zarr_root.attrs["source"] = "bigwig"
    zarr_root.attrs["store_type"] = "celltype_dense"
    zarr_root.attrs["class"] = "CelltypeDenseZarrIO"

    # Determine chromosomes to process
    if chr_filter_config:
        from caesar.preprocess.core.chromosome_filter import get_chromosomes_to_process

        chroms_list = get_chromosomes_to_process(
            chr_filter_config, list(chrom_sizes_ref.keys()), set(bw_chroms.keys())
        )
    else:
        from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

        chr_filter = ChromosomeFilter({})
        chroms_list = [
            chrom
            for chrom in chrom_sizes_ref
            if (chrom in bw_chroms) and chr_filter.is_valid_chromosome(chrom)
        ]

    # Track group names and signals
    group_names = []
    library_sizes = {}
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    # Process each BigWig file
    for bigwig_file, sample_id, celltype in zip(bigwig_files, sample_ids, celltypes):
        group_name = f"{celltype}.{sample_id}"
        group_names.append(group_name)
        logging.info(f"Processing {bigwig_file} as {group_name}")

        # Get chromosomes from this file
        file_chroms = get_bigwig_chromosomes(bigwig_file)

        # Create group
        group = zarr_root.require_group(group_name)
        group.attrs["id"] = sample_id
        group.attrs["celltype"] = celltype
        group.attrs["source_file"] = str(bigwig_file)

        # Create chrs subgroup
        chrs_group = group.require_group("chrs")

        # Prepare arguments for parallel processing
        process_args = []
        for chrom in chroms_list:
            if chrom not in file_chroms:
                continue
            chrom_size = file_chroms.get(chrom, chrom_sizes_ref.get(chrom, 0))
            if chrom_size == 0:
                continue
            process_args.append((bigwig_file, chrom, chrom_size, chunk_size))

        # Process chromosomes in parallel
        total_signal = 0.0

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(process_bigwig_chromosome, *args): args[1]
                for args in process_args
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {group_name}",
            ):
                try:
                    chrom, coverage, chrom_signal = future.result()
                    total_signal += chrom_signal

                    # Write chromosome data to zarr (zarr v3 compatible)
                    chrs_group.create_array(
                        chrom,
                        data=coverage,
                        chunks=(chunk_size,),
                        compressors=compressor,
                        overwrite=True,
                    )

                except Exception as e:
                    chrom_name = futures[future]
                    logging.error(f"Error processing chromosome {chrom_name}: {e}")
                    continue

        # Set group library size
        group.attrs["library_size"] = total_signal
        library_sizes[group_name] = total_signal

    # Set root-level attributes
    zarr_root.attrs["celltype_ids"] = group_names

    logging.info(f"Processed {len(bigwig_files)} BigWig files to {output_zarr}")
    return library_sizes
