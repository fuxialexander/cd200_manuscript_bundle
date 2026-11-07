#!/usr/bin/env python3
"""Convert VCF variants to one-hot encoded zarr format with liftover support."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pysam
import zarr
from pyliftover import LiftOver
from zarr.codecs import BloscCodec
from tqdm import tqdm


def load_variants_from_vcf(vcf_path, source_assembly="hg19", target_assembly="hg38"):
    """Load variants from VCF file and optionally perform liftover."""
    print(f"Loading variants from VCF: {vcf_path}")

    # Initialize liftover if needed
    lo = None
    if source_assembly != target_assembly:
        print(f"Initializing liftover: {source_assembly} -> {target_assembly}")
        lo = LiftOver(source_assembly, target_assembly)

    variants_by_chrom = defaultdict(list)
    total_variants = 0
    failed_liftover = 0

    # Open VCF file
    vcf = pysam.VariantFile(vcf_path)

    for record in tqdm(vcf, desc="Reading variants"):
        chrom = record.chrom
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"

        pos = record.pos - 1  # Convert to 0-based indexing
        ref = record.ref
        alt = record.alts[0] if record.alts else None

        if alt is None:
            continue

        # Only handle SNVs for now (single nucleotide variants)
        if len(ref) != 1 or len(alt) != 1:
            continue

        # Perform liftover if needed
        if lo is not None:
            lifted = lo.convert_coordinate(chrom, pos)
            if lifted:
                # Use the first successful lift
                new_chrom, new_pos = lifted[0][:2]
                chrom = new_chrom
                pos = new_pos
            else:
                failed_liftover += 1
                continue

        # Store just the position
        variants_by_chrom[chrom].append(pos)
        total_variants += 1

    vcf.close()

    print(
        f"Loaded {total_variants:,} variants across {len(variants_by_chrom)} chromosomes"
    )
    if lo is not None:
        print(f"Failed liftover: {failed_liftover:,} variants")

    # Sort variants by position within each chromosome
    for chrom in variants_by_chrom:
        variants_by_chrom[chrom].sort()
        print(f"  {chrom}: {len(variants_by_chrom[chrom]):,} variants")

    return dict(variants_by_chrom)


def get_chromosome_lengths(assembly="hg38"):
    """Get standard chromosome lengths for the assembly."""
    # Standard chromosome lengths for hg38
    hg38_lengths = {
        "chr1": 248956422,
        "chr2": 242193529,
        "chr3": 198295559,
        "chr4": 190214555,
        "chr5": 181538259,
        "chr6": 170805979,
        "chr7": 159345973,
        "chr8": 145138636,
        "chr9": 138394717,
        "chr10": 133797422,
        "chr11": 135086622,
        "chr12": 133275309,
        "chr13": 114364328,
        "chr14": 107043718,
        "chr15": 101991189,
        "chr16": 90338345,
        "chr17": 83257441,
        "chr18": 80373285,
        "chr19": 58617616,
        "chr20": 64444167,
        "chr21": 46709983,
        "chr22": 50818468,
        "chrX": 156040895,
        "chrY": 57227415,
    }

    # Also support without 'chr' prefix
    no_chr_lengths = {k.replace("chr", ""): v for k, v in hg38_lengths.items()}

    if assembly == "hg38":
        return {**hg38_lengths, **no_chr_lengths}
    else:
        raise ValueError(f"Chromosome lengths not available for assembly: {assembly}")


def encode_variants_for_chromosome(args):
    """Process variants for a single chromosome."""
    chrom, variants, chrom_length, zarr_path = args

    # Create binary array: 1 where variant exists, 0 otherwise
    variant_mask = np.zeros(chrom_length, dtype=np.uint8)

    variant_count = 0
    for pos in variants:
        if pos >= chrom_length:
            continue  # Skip variants beyond chromosome length

        # Set position to 1 if variant exists
        variant_mask[pos] = 1
        variant_count += 1

    # Write to zarr
    z = zarr.open_group(zarr_path, mode="r+")
    z[f"chrs/{chrom}"][:] = variant_mask

    return variant_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert VCF variants to one-hot zarr with liftover support"
    )
    parser.add_argument(
        "--input",
        default="/home/xf2217/Projects/pcawg/snv_mnv/BRCA_samples_merged.vcf.gz",
        help="Input VCF file (can be gzipped)",
    )
    parser.add_argument(
        "--output",
        default="../get_data/variants_hg38.zarr",
        help="Output zarr directory",
    )
    parser.add_argument(
        "--source-assembly",
        default="hg19",
        help="Source assembly of VCF coordinates (default: hg19)",
    )
    parser.add_argument(
        "--target-assembly",
        default="hg38",
        help="Target assembly for output (default: hg38)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2000000, help="Chunk size for zarr storage"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    args = parser.parse_args()

    # Expand paths
    vcf_path = Path(args.input).expanduser()
    zarr_path = Path(args.output).expanduser()
    chunk_size = args.chunk_size

    # Load variants from VCF
    variants_by_chrom = load_variants_from_vcf(
        vcf_path, args.source_assembly, args.target_assembly
    )

    # Get chromosome lengths for target assembly
    chrom_lengths = get_chromosome_lengths(args.target_assembly)

    # Filter to chromosomes that have both variants and known lengths
    processed_chroms = {}
    for chrom in variants_by_chrom:
        # Try both with and without 'chr' prefix
        chrom_key = chrom
        if chrom_key not in chrom_lengths:
            chrom_key = (
                f"chr{chrom}"
                if not chrom.startswith("chr")
                else chrom.replace("chr", "")
            )

        if chrom_key in chrom_lengths:
            processed_chroms[chrom] = chrom_lengths[chrom_key]
        else:
            print(f"Warning: No length found for chromosome {chrom}, skipping")

    print(f"\nProcessing {len(processed_chroms)} chromosomes:")
    for chrom in sorted(processed_chroms.keys()):
        variant_count = len(variants_by_chrom[chrom])
        chrom_len = processed_chroms[chrom]
        print(f"  {chrom}: {variant_count:,} variants, {chrom_len:,} bp")

    # Create zarr structure
    print(f"\nCreating zarr structure at: {zarr_path}")
    # Use zarr.open_group for v3 compatibility (LocalStore is used internally)
    root = zarr.open_group(zarr_path, mode="w", zarr_format=2)
    root.attrs["assembly"] = args.target_assembly
    root.attrs["source_assembly"] = args.source_assembly
    root.attrs["source_vcf"] = vcf_path.name

    # Create chromosome groups
    chrs_group = root.create_group("chrs")

    # Create zarr arrays for each chromosome
    for chrom, chrom_length in processed_chroms.items():
        print(f"Creating zarr array for {chrom}: {chrom_length:,} bp")

        chrs_group.create_array(
            chrom,
            shape=(chrom_length,),
            chunks=(chunk_size,),
            dtype="u1",
            fill_value=0,
            compressors=BloscCodec(
                cname="zstd", clevel=3, shuffle="bitshuffle"
            ),
        )

    # Prepare tasks for parallel processing
    tasks = [
        (chrom, variants_by_chrom[chrom], processed_chroms[chrom], zarr_path)
        for chrom in processed_chroms
    ]

    # Process chromosomes in parallel
    print(
        f"\nProcessing {len(tasks)} chromosomes with {args.workers or 'all'} workers..."
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(encode_variants_for_chromosome, task): task
            for task in tasks
        }

        # Process completed tasks with progress bar
        total_variants_processed = 0
        with tqdm(total=len(tasks), desc="Processing chromosomes") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    variant_count = future.result()
                    chrom = task[0]
                    total_variants_processed += variant_count
                    pbar.set_postfix(
                        {
                            "last_completed": chrom,
                            "variants": f"{total_variants_processed:,}",
                        }
                    )
                    pbar.update(1)
                except Exception as e:
                    chrom = task[0]
                    print(f"\nError processing {chrom}: {e}")

    print(f"\nConversion complete! Processed {total_variants_processed:,} variants")

    # Verify the structure
    z = zarr.open_group(zarr_path, mode="r")
    print(f"\nCreated zarr with assembly: {z.attrs['assembly']}")
    print(f"Source assembly: {z.attrs['source_assembly']}")
    print(f"Source VCF: {z.attrs['source_vcf']}")
    print("Chromosomes:")
    for chrom in sorted(z["chrs"].keys()):
        arr = z[f"chrs/{chrom}"]
        # Count non-zero positions (positions with variants)
        variant_positions = np.sum(arr[:])
        print(
            f"  {chrom}: shape={arr.shape}, chunks={arr.chunks}, "
            f"dtype={arr.dtype}, variants={variant_positions:,}"
        )


if __name__ == "__main__":
    main()
