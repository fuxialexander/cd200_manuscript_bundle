#!/usr/bin/env python3
"""Convert genome FASTA to one-hot encoded zarr format with parallel processing and auto-download.

This script can either process existing FASTA files or automatically download genome assemblies
from UCSC Genome Browser. It supports both regular and hard-masked (RepeatMasker + TRF) versions.

Usage Examples:
    # Auto-download and convert hg38 (regular version)
    python fasta_to_onehot_zarr.py --assembly hg38

    # Auto-download and convert mm10 hard-masked version
    python fasta_to_onehot_zarr.py --assembly mm10 --masked

    # Convert existing FASTA file
    python fasta_to_onehot_zarr.py --input ~/genomes/hg38.fa.gz

    # Custom download and output locations
    python fasta_to_onehot_zarr.py --assembly hg19 --download-dir ./ref_genomes --output hg19_custom.zarr

    # Parallel processing with custom chunk size
    python fasta_to_onehot_zarr.py --assembly dm6 --workers 8 --chunk-size 1000000

Supported Assemblies:
    - hg38, hg19 (Human)
    - mm10, mm39 (Mouse)
    - dm6 (Drosophila)
    - ce11 (C. elegans)
    - danRer11 (Zebrafish)
    - rn6 (Rat)

Output Format:
    The zarr structure contains:
    - Root attributes: assembly name
    - chrs/ group: one-hot encoded chromosomes (shape: length x 4)
    - Encoding: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1], N/masked=[0,0,0,0]
    - Only chromosomes with names ≤5 characters are included
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

SUPPORTED_ASSEMBLIES_NAMES = [
    "hg38",
    "hg19",
    "mm10",
    "mm39",
    "dm6",
    "ce11",
    "danRer11",
    "rn6",
    "gorGor6",
    "panTro6",
    "canFam6",
    "felCat9",
    "rn7",
    "galGal6",
    "sacCer3",
]

BASE_URL = "http://hgdownload.cse.ucsc.edu/goldenpath/{assembly}/bigZips/{assembly}.fa.gz"
MASKED_URL = "http://hgdownload.cse.ucsc.edu/goldenpath/{assembly}/bigZips/{assembly}.fa.masked.gz"

def generate_url_dict(assemblies):
    normal = {asm: BASE_URL.format(assembly=asm) for asm in assemblies}
    masked = {asm: MASKED_URL.format(assembly=asm) for asm in assemblies}
    return normal, masked

SUPPORTED_ASSEMBLIES, SUPPORTED_MASKED_ASSEMBLIES = generate_url_dict(SUPPORTED_ASSEMBLIES_NAMES)

def validate_assembly(assembly):
    """Validate that the assembly is supported."""
    if assembly not in SUPPORTED_ASSEMBLIES:
        print(f"Error: Assembly '{assembly}' is not supported.")
        print(f"Supported assemblies: {', '.join(sorted(SUPPORTED_ASSEMBLIES.keys()))}")
        sys.exit(1)


def download_genome(assembly, masked=False, output_dir="."):
    """Download genome FASTA from UCSC."""
    validate_assembly(assembly)

    # Choose URL based on masked option
    url_dict = SUPPORTED_MASKED_ASSEMBLIES if masked else SUPPORTED_ASSEMBLIES

    # Check if masked version is available
    if masked and assembly not in SUPPORTED_MASKED_ASSEMBLIES:
        print(f"Error: Masked version not available for assembly '{assembly}'")
        print(
            f"Available masked assemblies: {', '.join(sorted(SUPPORTED_MASKED_ASSEMBLIES.keys()))}"
        )
        sys.exit(1)

    url = url_dict[assembly]
    suffix = ".fa.masked.gz" if masked else ".fa.gz"
    filename = f"{assembly}{suffix}"
    output_path = Path(output_dir) / filename

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if output_path.exists():
        print(f"Genome file already exists: {output_path}")
        return str(output_path)

    print(f"Downloading {assembly} genome {'(masked)' if masked else ''}...")
    print(f"Source: {url}")
    print(f"Output: {output_path}")

    try:
        # Download with progress bar
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("Content-Length", 0))

            with (
                output_path.open("wb") as f,
                tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as pbar,
            ):
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    except Exception as e:
        print(f"Error downloading genome: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        sys.exit(1)

    print(f"Download complete: {output_path}")
    return str(output_path)


def load_fasta_to_dict(fasta_path):
    """Load gzipped FASTA file directly into a dictionary of chromosome sequences."""
    print(f"Loading FASTA file into memory: {fasta_path}")

    chrom_sequences = {}
    current_chrom = None
    current_seq = []
    total_bases = 0

    # Open gzipped file
    with gzip.open(fasta_path, "rt") as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading FASTA lines")):
            line = line.strip()

            if line.startswith(">"):
                # Save previous chromosome if exists
                if current_chrom is not None:
                    seq_str = "".join(current_seq)
                    chrom_sequences[current_chrom] = seq_str
                    total_bases += len(seq_str)
                    print(f"Loaded {current_chrom}: {len(seq_str):,} bp")

                # Start new chromosome
                current_chrom = line[1:].split()[0]  # Take first part after '>'
                current_seq = []
            else:
                # Add sequence line
                if current_chrom is not None:
                    current_seq.append(line.upper())

    # Don't forget the last chromosome
    if current_chrom is not None:
        seq_str = "".join(current_seq)
        chrom_sequences[current_chrom] = seq_str
        total_bases += len(seq_str)
        print(f"Loaded {current_chrom}: {len(seq_str):,} bp")

    print(f"Total bases loaded: {total_bases:,}")
    print(f"Total chromosomes: {len(chrom_sequences)}")

    return chrom_sequences


def encode_sequence_chunk(seq_chunk):
    """Convert DNA sequence to one-hot encoding.

    Encoding:
        A = [1,0,0,0]
        C = [0,1,0,0]
        G = [0,0,1,0]
        T = [0,0,0,1]
        N/masked/other = [0,0,0,0]
    """
    # Mapping: A=0, C=1, G=2, T=3
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    # Convert to one-hot: shape (seq_len, 4)
    one_hot = np.zeros((len(seq_chunk), 4), dtype=np.uint8)

    for i, base in enumerate(seq_chunk):
        if base in mapping:
            one_hot[i, mapping[base]] = 1
        # N, masked bases, and other characters remain [0,0,0,0]

    return one_hot


def process_chromosome(args):
    """Process a single chromosome."""
    chrom_seq, chrom, zarr_path = args

    # Encode entire chromosome sequence
    one_hot = encode_sequence_chunk(chrom_seq)

    # Write to zarr
    z = zarr.open_group(zarr_path, mode="r+")
    z[f"chrs/{chrom}"][:] = one_hot

    return len(chrom_seq)


def main():
    parser = argparse.ArgumentParser(
        description="Convert genome FASTA to one-hot encoded zarr format with auto-download support"
    )

    # Input options - either provide file or assembly for auto-download
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", help="Input genome FASTA file (.fa.gz or .fa.masked.gz)"
    )
    input_group.add_argument(
        "--assembly",
        help="Genome assembly name for auto-download (e.g., hg38, hg19, mm10, mm39, dm6, ce11, danRer11, rn6)",
    )

    parser.add_argument(
        "--masked",
        action="store_true",
        help="Download hard-masked version (RepeatMasker + TRF masked). Only used with --assembly.",
    )
    parser.add_argument(
        "--download-dir",
        default="./genomes",
        help="Directory to download genomes to (default: ./genomes)",
    )
    parser.add_argument(
        "--output",
        help="Output zarr directory (default: auto-generated from assembly name)",
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

    # Handle input: either provided file or auto-download
    if args.input:
        # User provided input file
        fasta_path = Path(args.input).expanduser()
        assembly = None
        if not fasta_path.exists():
            print(f"Error: Input file not found: {fasta_path}")
            sys.exit(1)

        # Try to infer assembly from filename if not provided elsewhere
        filename = fasta_path.name
        for asm in SUPPORTED_ASSEMBLIES:
            if asm in filename:
                assembly = asm
                break

        if not assembly:
            print(
                "Warning: Could not infer assembly from filename. Using 'unknown' in output."
            )
            assembly = "unknown"

    else:
        # Auto-download mode
        assembly = args.assembly
        validate_assembly(assembly)

        # Show warning for --masked with assemblies that don't support it
        if args.masked and assembly not in SUPPORTED_MASKED_ASSEMBLIES:
            print(f"Warning: --masked requested but not available for {assembly}")
            print(
                f"Available masked assemblies: {', '.join(sorted(SUPPORTED_MASKED_ASSEMBLIES.keys()))}"
            )
            print("Proceeding with regular (unmasked) version.")
            args.masked = False

        # Download the genome
        fasta_path = Path(download_genome(assembly, args.masked, args.download_dir))

    # Set up output path
    if args.output:
        zarr_path = Path(args.output).expanduser()
    else:
        # Auto-generate output name
        masked_suffix = "_masked" if (args.assembly and args.masked) else ""
        zarr_path = Path(f"{assembly}{masked_suffix}.zarr")

    print(f"\nInput FASTA: {fasta_path}")
    print(f"Output zarr: {zarr_path}")
    print(f"Assembly: {assembly}")

    chunk_size = args.chunk_size

    # Load all chromosome sequences into memory
    chrom_sequences = load_fasta_to_dict(fasta_path)

    # Filter chromosomes by name length (≤ 5 characters)
    filtered_chroms = {
        chrom: seq for chrom, seq in chrom_sequences.items() if len(chrom) <= 5
    }

    print("\nFiltered chromosomes (name length ≤ 5):")
    print(f"  Keeping: {len(filtered_chroms)} chromosomes")
    print(f"  Excluding: {len(chrom_sequences) - len(filtered_chroms)} chromosomes")

    for chrom in sorted(filtered_chroms.keys()):
        print(f"  Keeping {chrom}: {len(filtered_chroms[chrom]):,} bp")

    # Create zarr structure
    print(f"\nCreating zarr structure at: {zarr_path}")
    # Use zarr.open_group for v3 compatibility (LocalStore is used internally)
    root = zarr.open_group(zarr_path, mode="w", zarr_format=2)
    root.attrs["assembly"] = assembly

    # Create chromosome groups
    chrs_group = root.create_group("chrs")

    # Create zarr arrays for each filtered chromosome
    for chrom, seq in filtered_chroms.items():
        chrom_len = len(seq)
        print(f"Creating zarr array for {chrom}: {chrom_len:,} bp")

        chrs_group.create_array(
            chrom,
            shape=(chrom_len, 4),
            chunks=(chunk_size, 4),
            dtype="u1",
            fill_value=0,
            compressors=BloscCodec(
                cname="zstd", clevel=3, shuffle="bitshuffle"
            ),
        )

    # Prepare tasks for parallel processing (one task per filtered chromosome)
    tasks = [
        (filtered_chroms[chrom], chrom, str(zarr_path)) for chrom in filtered_chroms
    ]

    # Process chromosomes in parallel
    print(
        f"\nProcessing {len(tasks)} chromosomes with {args.workers or 'all'} workers..."
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_chromosome, task): task for task in tasks}

        # Process completed tasks with progress bar
        with tqdm(total=len(tasks), desc="Processing chromosomes") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()  # Just check for completion
                    chrom = task[1]
                    pbar.set_postfix({"last_completed": chrom})
                    pbar.update(1)
                except Exception as e:
                    chrom = task[1]
                    print(f"\nError processing {chrom}: {e}")

    print("\nConversion complete!")

    # Verify the structure
    z = zarr.open_group(str(zarr_path), mode="r")
    print(f"\nCreated zarr with assembly: {z.attrs['assembly']}")
    print("Chromosomes:")
    for chrom in sorted(z["chrs"].keys()):
        arr = z[f"chrs/{chrom}"]
        print(f"  {chrom}: shape={arr.shape}, chunks={arr.chunks}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
