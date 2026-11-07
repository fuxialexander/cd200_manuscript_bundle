#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging

import numpy as np
import pysam
import zarr
from tqdm import tqdm

from caesar.preprocess.utils.fragment_utils import (
    filter_positions_by_blacklist,
    process_chromosome_coverage,
)
from gcell.dna.genome import ChromSize, read_blacklist


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
    group.attrs["id"] = args.id
    group.attrs["celltype"] = args.celltype
    group.attrs["assembly"] = args.assembly
    group.attrs["chunk_size"] = args.chunk_size  # Store chunk size in metadata

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
        # Fallback to hardcoded filtering for backward compatibility
        chroms_list = [
            chrom
            for chrom in chrom_sizes
            if (chrom in chroms_in_tabix)
            and (len(chrom) <= 5)
            and (chrom != "chrM")
            and ("Un" not in chrom)
            and ("random" not in chrom)
            and (chrom != "chrY")
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

    # Set the library size as a group attribute
    group.attrs["library_size"] = library_size

    print(f"Done. Dense coverage Zarr written to: {args.output_zarr}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert a fragment file to a dense Zarr coverage store."
    )
    parser.add_argument("--id", type=str, required=True, help="Sample or replicate ID")
    parser.add_argument(
        "--celltype", type=str, required=True, help="Cell type or group identifier"
    )
    parser.add_argument(
        "--fragment-file", type=str, required=True, help="Path to fragment file"
    )
    parser.add_argument(
        "--assembly",
        type=str,
        required=True,
        help="Genome assembly (e.g., hg38, mm10)",
    )
    parser.add_argument("--output-zarr", type=str, required=True, help="Output path")
    parser.add_argument(
        "--chunk-size", type=int, default=2_000_000, help="Chunk size for Zarr arrays"
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--single-end",
        action="store_true",
        help="Process as single-end. Requires strand in 6th column.",
    )
    mode.add_argument(
        "--paired-end",
        action="store_false",
        dest="single_end",
        help="Process as paired-end (default).",
    )
    parser.set_defaults(single_end=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_fragment_to_zarr(args)


if __name__ == "__main__":
    main()
