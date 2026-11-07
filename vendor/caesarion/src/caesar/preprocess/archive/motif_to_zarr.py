"""
Convert MOODS-scanned motif BED file to Zarr format
"""

from __future__ import annotations

import argparse
import gc
import io
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from caesar.io.chromsize import ChromSize


def get_motif_df(motif_bed_path, chrom, start, end):
    """Call tabix to get the motif df for a given chrom, start, end"""
    cmd = f"tabix {motif_bed_path} {chrom}:{start}-{end}"
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if stderr:
        raise Exception(stderr)
    motif_df = pd.read_csv(
        io.StringIO(stdout.decode("utf-8")),
        sep="\t",
        header=None,
        names=[
            "chrom",
            "start",
            "end",
            "cluster",
            "score",
            "strand",
            "seed_motif",
            "n_motifs",
        ],
    )
    return motif_df


def process_chunk(
    motif_bed_path,
    chrom,
    chunk_start,
    chunk_end,
    cluster_to_idx,
    zc,
    chrom_size,
    stranded=False,
):
    motif_df = get_motif_df(motif_bed_path, chrom, chunk_start, chunk_end)

    # Calculate the actual chunk size (important for the last chunk)
    actual_chunk_size = min(chunk_end, chrom_size.chrom_sizes[chrom]) - chunk_start

    # If stranded, create array with double width to store + and - strands separately
    n_cols = len(cluster_to_idx) * 2 if stranded else len(cluster_to_idx)
    chunk_arr = np.zeros((actual_chunk_size, n_cols), dtype="float32")
    chunk = motif_df[
        (motif_df["start"] >= chunk_start) & (motif_df["end"] <= chunk_end)
    ]

    for _, row in chunk.iterrows():
        start_offset = max(0, row["start"] - chunk_start)
        end_offset = min(actual_chunk_size, row["end"] - chunk_start)
        if stranded:
            # For stranded data, put + strand in even columns, - strand in odd columns
            col_idx = cluster_to_idx[row["cluster"]] * 2
            if row["strand"] == "-":
                col_idx += 1
            chunk_arr[start_offset:end_offset, col_idx] = row["score"]
        else:
            chunk_arr[start_offset:end_offset, cluster_to_idx[row["cluster"]]] = row[
                "score"
            ]

    zc[chunk_start : chunk_start + actual_chunk_size, :] = chunk_arr

    del motif_df, chunk, chunk_arr
    gc.collect()
    return 1


def process_chromosome(
    motif_bed_path, chrom, z, cluster_to_idx, chrom_size, stranded=False, n_jobs=16
):
    chrom_length = chrom_size.chrom_sizes[chrom]
    n_cols = len(cluster_to_idx) * 2 if stranded else len(cluster_to_idx)
    zc = z.create_array(
        chrom,
        shape=(chrom_length, n_cols),
        chunks=(100000, n_cols),
        overwrite=True,
        dtype="float32",
        compressor=BloscCodec(cname="zstd", clevel=4),
    )
    chunk_starts = list(range(0, chrom_length, 100000))

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                process_chunk,
                motif_bed_path,
                chrom,
                chunk_start,
                min(chunk_start + 100000, chrom_length),
                cluster_to_idx,
                zc,
                chrom_size,
                stranded,
            )
            for chunk_start in chunk_starts
        ]

        for chunk_start, future in zip(
            chunk_starts,
            tqdm(futures, total=len(chunk_starts), desc=f"Processing {chrom}"),
        ):
            future.result()
            gc.collect()
    del zc
    gc.collect()


def main(motif_bed_path, output_zarr_path, stranded=False, n_jobs=16):
    # Get initial motif clusters
    initial_motif_df = get_motif_df(motif_bed_path, "chr1", 1, 500000)
    motif_clusters = sorted(initial_motif_df["cluster"].unique())
    cluster_to_idx = {cluster: i for i, cluster in enumerate(motif_clusters)}
    # Initialize zarr
    z = zarr.open_group(output_zarr_path, mode="w")
    z.create_group("chrs")
    z = z["chrs"]

    # Get chromosome sizes
    chrom_size = ChromSize("hg38", os.path.dirname(motif_bed_path))

    # Process chromosomes
    for chrom in tqdm(sorted(chrom_size.chrom_sizes.keys())[::-1], desc="Chromosomes"):
        try:
            process_chromosome(
                motif_bed_path, chrom, z, cluster_to_idx, chrom_size, stranded, n_jobs
            )
        except Exception as e:
            print(f"Error processing {chrom}: {e}")
            if chrom in z:
                del z[chrom]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert motif BED file to Zarr format"
    )
    parser.add_argument("--motif_bed_path", help="Path to the input motif BED file")
    parser.add_argument("--output_zarr_path", help="Path to the output Zarr file")
    parser.add_argument(
        "--stranded",
        action="store_true",
        help="If set, store + and - strand motifs separately",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=16, help="Number of jobs to run in parallel"
    )
    args = parser.parse_args()

    main(
        args.motif_bed_path,
        args.output_zarr_path,
        stranded=args.stranded,
        n_jobs=args.n_jobs,
    )
