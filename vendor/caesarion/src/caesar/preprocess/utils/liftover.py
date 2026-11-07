from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import current_process

import requests


def check_and_download_tools(directory, assembly_from, assembly_to):
    """Ensure liftover tools and chain files are available, download if necessary."""
    liftover_url = "http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver"
    chain_file_url = f"http://hgdownload.soe.ucsc.edu/goldenPath/{assembly_from}/liftOver/{assembly_from}To{assembly_to.capitalize()}.over.chain.gz"

    liftover_path = os.path.join(directory, "liftOver")
    chain_file_path = os.path.join(
        directory, f"{assembly_from}To{assembly_to.capitalize()}.over.chain.gz"
    )

    if not os.path.exists(liftover_path):
        print(f"{current_process().name}: Downloading liftOver tool...")
        r = requests.get(liftover_url)
        with open(liftover_path, "wb") as f:
            f.write(r.content)
        os.chmod(liftover_path, 0o755)

    if not os.path.exists(chain_file_path):
        print(
            f"{current_process().name}: Downloading {assembly_from} to {assembly_to} chain file..."
        )
        r = requests.get(chain_file_url)
        with open(chain_file_path, "wb") as f:
            f.write(r.content)


def is_gzipped(file_path):
    """Check if the file is gzipped."""
    return file_path.endswith(".gz")


def decompress_gz(file_path):
    """Decompress a gzipped file."""
    if is_gzipped(file_path):
        # skip decompression if the file is already decompressed
        if not os.path.exists(file_path.rstrip(".gz") + ".decompressed"):
            with gzip.open(file_path, "rb") as f_in:
                with open(file_path.rstrip(".gz") + ".decompressed", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(
                f"Decompressed file already exists for: {os.path.basename(file_path)}. Skipping decompression."
            )
        return file_path.rstrip(".gz") + ".decompressed"
    return file_path


def chunk_file(file_path, chunks_dir, n_chunks):
    """Split a file into N chunks."""
    # if the chunk file already exists, skip the chunking process
    data_key = os.path.basename(file_path).split(".")[0]
    if os.path.exists(
        os.path.join(
            chunks_dir, f"{os.path.basename(file_path).split('.')[0]}_chunk0.bed"
        )
    ):
        print(
            f"Chunk files already exist for: {os.path.basename(file_path)}. Skipping chunking."
        )
        return [
            os.path.join(chunks_dir, f)
            for f in os.listdir(chunks_dir)
            if f.endswith(".bed") and f.startswith(data_key + "_chunk")
        ]
    os.makedirs(chunks_dir, exist_ok=True)
    with open(file_path) as f:
        lines = f.readlines()
    chunk_size = len(lines) // n_chunks + (len(lines) % n_chunks > 0)
    for i in range(n_chunks):
        chunk_path = os.path.join(
            chunks_dir, f"{os.path.basename(file_path).split('.')[0]}_chunk{i}.bed"
        )
        with open(chunk_path, "w") as f:
            f.writelines(lines[i * chunk_size : (i + 1) * chunk_size])
    return [
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
        if f.endswith(".bed") and f.startswith(data_key + "_chunk")
    ]


def perform_liftover_chunk(
    chunk_file, annotation_dir, assembly_from, assembly_to, output_dir, unlifted_dir
):
    """
    Perform the liftover operation on a chunk file.

    Parameters:
    - chunk_file: Path to the chunk file to be lifted over.
    - annotation_dir: Directory where the liftover binary and chain files are stored.
    - assembly_from: Source genome assembly version.
    - assembly_to: Target genome assembly version.
    - output_dir: Directory to store the lifted over chunk files.
    - unlifted_dir: Directory to store the chunks that couldn't be lifted over.
    """
    # Construct paths to the liftover binary and the chain file
    liftover_path = os.path.join(annotation_dir, "liftOver")
    chain_file_path = os.path.join(
        annotation_dir, f"{assembly_from}To{assembly_to.capitalize()}.over.chain.gz"
    )

    # Ensure output and unlifted directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(unlifted_dir, exist_ok=True)

    # Naming output and unlifted files based on the chunk file name
    output_file_path = os.path.join(
        output_dir, os.path.basename(chunk_file).replace(".bed", "_lifted.bed")
    )
    unlifted_file_path = os.path.join(
        unlifted_dir, os.path.basename(chunk_file).replace(".bed", "_unlifted.bed")
    )
    # if the output file already exists, skip the liftover process
    if os.path.exists(output_file_path):
        print(
            f"Output file already exists for chunk: {os.path.basename(chunk_file)}. Skipping liftover."
        )
        return
    # Constructing the command for the liftover process
    command = [
        liftover_path,
        chunk_file,
        chain_file_path,
        output_file_path,
        unlifted_file_path,
    ]

    # Execute the liftover command
    try:
        subprocess.run(command, check=True)
        print(f"Liftover completed for chunk: {os.path.basename(chunk_file)}")
    except subprocess.CalledProcessError as e:
        print(
            f"Error during liftover for chunk: {os.path.basename(chunk_file)}. Error: {e}"
        )


def merge_and_sort_files(files, output_file):
    """Merge and sort BED files."""
    # Use 'sort' command for merging and sorting, ensure genomic order if necessary
    with open(output_file, "wb") as merged:
        for f in sorted(files):
            with open(f, "rb") as infile:
                shutil.copyfileobj(infile, merged)
    # Optionally, use 'sort -k1,1 -k2,2n output_file > sorted_output_file' for genomic sorting


def process_file_in_parallel(
    file_path,
    annotation_dir,
    assembly_from,
    assembly_to,
    output_dir,
    unlifted_dir,
    n_jobs,
):
    """
    Decompress (if necessary), chunk, perform parallel liftover, and merge & sort the input file.
    """
    # Step 1: Decompress the file if it is gzipped

    decompressed_file = decompress_gz(file_path)

    # Step 2: Chunk the decompressed file
    chunks_dir = os.path.join(os.path.dirname(decompressed_file), "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    chunk_files = chunk_file(decompressed_file, chunks_dir, n_jobs)
    # Step 3: Perform liftover in parallel on each chunk
    liftover_args = [
        (chunk, annotation_dir, assembly_from, assembly_to, output_dir, unlifted_dir)
        for chunk in chunk_files
    ]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        executor.map(perform_liftover_chunk, *zip(*liftover_args))

    # Step 4: Merge and sort the output files from each chunk
    merged_output_file = os.path.join(
        output_dir,
        f"{os.path.basename(decompressed_file).split('.')[0]}_merged_sorted.bed",
    )
    output_files = [
        os.path.join(output_dir, os.path.basename(chunk.replace(".bed", "_lifted.bed")))
        for chunk in chunk_files
    ]  # Assuming liftover outputs are named accordingly
    # if the merged output file already exists, skip the merging and sorting process
    if os.path.exists(merged_output_file):
        print(
            f"Merged output file already exists for: {os.path.basename(decompressed_file)}. Skipping merging and sorting."
        )
        return
    merge_and_sort_files(output_files, merged_output_file)

    # Cleanup: Remove temporary chunk files and directories
    # shutil.rmtree(chunks_dir)
    # if file_path != decompressed_file:  # If a decompression happened, remove the decompressed file
    #     os.remove(decompressed_file)


def perform_liftover(
    bed_file, annotation_dir, assembly_from, assembly_to, output_dir, unlifted_dir
):
    """Perform the liftover operation on the input BED file with parallel processing."""
    chain_file_path = os.path.join(
        annotation_dir, f"{assembly_from}To{assembly_to.capitalize()}.over.chain.gz"
    )
    output_file = os.path.join(output_dir, os.path.basename(bed_file).split(".")[0])
    unlifted_file = os.path.join(
        unlifted_dir, os.path.basename(bed_file).replace(".bed", "_unlifted.bed")
    )

    command = f"{annotation_dir}/liftOver {bed_file} {chain_file_path} {output_file} {unlifted_file}"
    os.system(command)  # Consider subprocess.run() for better security in production


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Perform liftover on BED files.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input directory containing BED files to liftover",
    )
    parser.add_argument(
        "--from",
        dest="assembly_from",
        type=str,
        required=True,
        help="Source genome assembly (e.g. hg19)",
    )
    parser.add_argument(
        "--to",
        dest="assembly_to",
        type=str,
        required=True,
        help="Target genome assembly (e.g. hg38)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="Number of parallel jobs to run"
    )
    return parser.parse_args()


def main(args):
    assembly_from = args.assembly_from
    assembly_to = args.assembly_to
    root_path = args.input_path

    # Create necessary directories
    fragments_to_liftover_dir = os.path.join(root_path, "fragments_to_liftover")
    fragments_dir = os.path.join(root_path, "fragments")
    liftover_dir = os.path.join(root_path, "liftover_fragments")
    unlifted_dir = os.path.join(root_path, "unlifted_fragments")
    os.makedirs(fragments_to_liftover_dir, exist_ok=True)
    os.makedirs(fragments_dir, exist_ok=True)
    os.makedirs(liftover_dir, exist_ok=True)
    os.makedirs(unlifted_dir, exist_ok=True)

    check_and_download_tools(root_path, assembly_from, assembly_to)

    n_jobs = args.jobs
    bed_files = [
        os.path.join(fragments_to_liftover_dir, f)
        for f in os.listdir(fragments_to_liftover_dir)
        if f.endswith(".bed")
        or f.endswith(".bed.gz")
        or f.endswith(".gz")
        or f.endswith(".tsv.gz")
        or f.endswith(".tsv")
    ]

    for bed_file in bed_files:
        process_file_in_parallel(
            bed_file,
            root_path,
            assembly_from,
            assembly_to,
            liftover_dir,
            unlifted_dir,
            n_jobs,
        )
    # move merged and sorted files to the fragments directory and remove _merged_sorted.bed suffix
    for f in os.listdir(liftover_dir):
        if f.endswith("_merged_sorted.bed"):
            shutil.copyfile(
                os.path.join(liftover_dir, f),
                os.path.join(fragments_dir, f.replace("_merged_sorted.bed", "")),
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
