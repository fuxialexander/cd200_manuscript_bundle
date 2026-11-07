# %%
from __future__ import annotations

import argparse
import gzip

import numpy as np
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from caesar.config.config import load_config
from caesar.io.chromsize import ChromSize


def get_counts_in_range(start, end, arr):
    return np.unique(arr[(arr < end) & (arr >= start)], return_counts=True)


def process_chromosome(
    id,
    chrom_dict: dict,
    chrom: str,
    n_chunks: int,
    zarr_root: zarr.Group,
    chunk_size=2_000_000,
):
    """
    Process a single chromosome file and store in Zarr.
    """
    print(f"Processing {chrom}...")

    for chunk_idx in tqdm(range(n_chunks)):
        chunk_start = chunk_idx * chunk_size
        chunk_end = (chunk_idx + 1) * chunk_size
        arr = get_counts_in_range(chunk_start, chunk_end, chrom_dict[chrom])
        arr_dense = np.zeros(chunk_size)
        arr_dense[arr[0] - chunk_start] = arr[1]
        zarr_root.create_array(
            f"{id}/chrs/{chrom}/chunk_{chunk_idx}",
            dtype="uint16",
            overwrite=True,
            chunks=(chunk_size),
            data=arr_dense,
            compressor=BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),
        )
        zarr_root[f"{id}"].attrs["id"] = id
        zarr_root[f"{id}"].attrs["celltype"] = id.split(".")[0]
        zarr_root[f"{id}"].attrs["n_sample"] = id.split(".")[-1]


def main(args):
    # Load the config file
    config = load_config(args.config)
    print(config)
    # override config settings if specified in args
    if args.assembly:
        config.assembly = args.assembly
    if args.root_path:
        config.root_path = args.root_path
    if args.data_key:
        config.data_key = args.data_key
    if args.n_jobs:
        config.n_jobs = args.n_jobs

    print("Using assembly:", config.assembly)
    chrom_length = ChromSize(config=config).dict

    chunk_size = config.chunk_size

    # Initialize a zarr
    # Use zarr.open_group for v3 compatibility
    zarr_root = zarr.open_group(f"{config.data_key}_dense.zarr", mode="a", zarr_format=2)
    zarr_root.attrs["assembly"] = config.assembly

    id = args.id
    celltype = args.celltype
    fragment_file = args.fragment_file
    chrom_dict = {}
    current_chrom = None
    with gzip.open(fragment_file, "rt") as file:
        for line in tqdm(file):
            fields = line.strip().split("\t")
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])

            if chrom != current_chrom:
                if current_chrom is not None and current_chrom in chrom_dict:
                    chr_len = chrom_length[current_chrom]
                    n_chunks = chr_len // chunk_size + 1
                    chrom_dict[current_chrom] = np.array(chrom_dict[current_chrom])
                    process_chromosome(
                        f"{celltype}.{config.data_key}.{id}.max",
                        chrom_dict,
                        current_chrom,
                        n_chunks,
                        zarr_root,
                    )
                    chrom_dict = {}
                current_chrom = chrom

            if "_" not in chrom and "Y" not in chrom and "M" not in chrom:
                if chrom not in chrom_dict:
                    chrom_dict[chrom] = []
                chrom_dict[chrom].extend([start, end])

    # Process the last chromosome
    if current_chrom is not None and current_chrom in chrom_dict:
        chr_len = chrom_length[current_chrom]
        n_chunks = chr_len // chunk_size + 1
        process_chromosome(
            f"{celltype}.{config.data_key}.{id}.max",
            chrom_dict,
            current_chrom,
            n_chunks,
            zarr_root,
        )

    # tagalign = pd.read_csv(fragment_file, sep='\t', header=None, names=[
    #                         'Chromosome', 'Start', 'End', 'N', 'V', 'Strand'], compression='gzip')

    # chrom_dict = {}
    # for chrom in tqdm(chrom_length.keys()):
    #     if '_' not in chrom and 'Y' not in chrom and 'M' not in chrom:
    #         chr_len = chrom_length[chrom]
    #         n_chunks = chr_len // chunk_size + 1
    #         df_chr = tagalign.query(
    #             'Chromosome==@chrom').drop_duplicates()
    #         chrom_dict[chrom] = df_chr[[
    #             'Start', 'End']].values.flatten()
    #         process_chromosome(f'{celltype}.{config.data_key}.{id}.max', chrom_dict, chrom, n_chunks, zarr_root)


def setup_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the config file")
    parser.add_argument("-i", "--id", type=str, help="Cell type ID")
    parser.add_argument("-ct", "--celltype", type=str, help="Cell type")
    parser.add_argument(
        "-f", "--fragment_file", type=str, help="Path to the fragment file"
    )
    # skip splitting fragment file into chromosomes
    parser.add_argument(
        "-sk",
        "--skip",
        action="store_true",
        help="Skip splitting fragment file into chromosomes",
    )
    parser.add_argument(
        "-se", "--skip_exist", action="store_true", help="Skip existing chunks"
    )
    parser.add_argument(
        "-n", "--n_jobs", type=int, help="Number of jobs to run in parallel"
    )
    parser.add_argument("-g", "--assembly", type=str, help="Genome assembly")
    parser.add_argument(
        "-r", "--root_path", type=str, help="Path to the raw data directory"
    )
    parser.add_argument("-d", "--data_key", type=str, help="Data key")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_args()
    main(args)
