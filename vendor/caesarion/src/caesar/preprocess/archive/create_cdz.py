from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import zarr
from zarr.codecs import BloscCodec
from tqdm import tqdm

from caesar.config.config import Config
from caesar.io.chromsize import ChromSize
from caesar.io.zarr_io import CelltypeDenseZarrIO, MultiSparseZarrIO


class CreateCDZ:
    def __init__(self, config):
        self.config = config
        self.zio = MultiSparseZarrIO(
            self.config.root_path + "/output/",
            "*.zarr",
            quick_look=False,
            mode="a",
            peaks_name=self.config.specific_peak_names[0],
        )
        self.chrom_size = ChromSize(
            self.config.assembly, f"{self.config.codebase}/data/"
        )
        self.root = zarr.open_group(
            f"{self.config.output_dir}/{self.config.data_key}_dense.zarr", mode="a"
        )
        # write metadata
        self.root.attrs["assembly"] = self.config.assembly

    def test_load_cdz(self, mode="a"):
        """Test the generated CDZ can be loaded"""
        self.cdz = CelltypeDenseZarrIO(
            f"{self.config.output_dir}/{self.config.data_key}_dense.zarr", mode=mode
        )
        print(self.cdz)

    def create_queries(self, n_insertions=None, n_cells=None, query_df=None):
        queries = {}
        if query_df is None:
            # use labels in self.zio.metadata
            for label in tqdm(self.zio.metadata.labels.unique()):
                if self.config.integrated:
                    try:
                        query_metadata = self.zio.metadata.loc[
                            self.zio.metadata["labels"] == label
                        ]
                        query_names, query = self.subsample_query(
                            n_cells, n_insertions, label, query_metadata
                        )
                        queries[query_names] = query
                    except Exception:
                        continue
                else:
                    for dk in self.zio.metadata.loc[
                        self.zio.metadata["labels"] == label
                    ].data_key.unique():
                        try:
                            query_metadata = self.zio.metadata.loc[
                                (self.zio.metadata["labels"] == label)
                                & (self.zio.metadata["data_key"] == dk)
                            ]
                            query_names, query = self.subsample_query(
                                n_cells,
                                n_insertions,
                                label,
                                query_metadata,
                                sample_id=dk,
                            )
                            queries[query_names] = query
                        except Exception:
                            continue
        else:
            # use labels in query_df and integrate with self.zio.metadata
            for labels in tqdm(query_df.labels.unique()):
                query_df_subset = query_df.loc[query_df["labels"] == labels]
                print(f"Processing {labels} with {len(query_df_subset)} cells")
                # depending on user input, we can either sample a fixed number of cells or target a fixed number of reads, this can be done for each labels, or each label-data_key pair
                if self.config.integrated:
                    try:
                        query_metadata = pd.merge(
                            self.zio.metadata,
                            query_df_subset,
                            on=["cell_idx", "data_key"],
                            how="inner",
                            suffixes=("", "_y"),
                        )
                        # query_metadata = self.zio.metadata.loc[self.zio.metadata['cell_idx'].isin(
                        #     query_df_subset['cell_idx'].values)]
                        query_names, query = self.subsample_query(
                            n_cells, n_insertions, labels, query_metadata
                        )
                        print(query)
                        queries[query_names] = query
                    except Exception:
                        continue
                else:
                    subquery = {}
                    for dk, metadata_subset_dk in query_df_subset.groupby("data_key"):
                        try:
                            query_metadata = self.zio.metadata.loc[
                                (self.zio.metadata["data_key"] == dk)
                                & (
                                    self.zio.metadata["cell_idx"].isin(
                                        metadata_subset_dk["cell_idx"].values
                                    )
                                )
                            ]
                            query_names, query = self.subsample_query(
                                n_cells,
                                n_insertions,
                                labels,
                                query_metadata,
                                sample_id=dk,
                            )
                            subquery.update({f"{dk}": query[dk]})

                        except Exception:
                            continue
                    queries[query_names] = subquery

        return queries

    def subsample_query(
        self, n_cells, n_insertions, labels, query_metadata, sample_id=None
    ):
        """
        Generate a query based on the provided parameters.

        Args:
            n_cells (int): The number of cells to sample.
            n_insertions (int): The target number of reads.
            labels (str): The cell type labels.
            query_metadata (pandas.DataFrame): The metadata for the query. It should contain the columns 'data_key' and 'cell_idx', corresponding to all cells that satisfy certain query in upstream steps.

        Returns:
            tuple: A tuple containing the query name and the query dictionary.
        """
        data_key = self.construct_data_key(sample_id)
        if n_insertions is not None:
            # Sample all rows and reset the index
            query_metadata = query_metadata.sample(frac=1).reset_index(drop=True)

            # Calculate the cumulative sum of per_cell_libsize
            query_metadata["cumsum_libsize"] = query_metadata[
                "per_cell_libsize"
            ].cumsum()

            # Filter the metadata based on the n_insertions
            query_metadata = query_metadata.loc[
                query_metadata["cumsum_libsize"] <= n_insertions
            ]
            # Generate the query name based on n_insertions
            query_name = f"{labels}.{data_key}.L{str(n_insertions // 1000000)}M"

        elif n_cells is not None:
            # Sample n_cells rows from the metadata
            query_metadata = query_metadata.sample(n_cells)

            # Generate the query name based on n_cells
            query_name = f"{labels}.{data_key}.{str(n_cells)}"

        else:
            raise ValueError("Either n_cells or n_insertions must be provided.")

        # Generate the query dictionary
        query_dict = (
            query_metadata[["data_key", "cell_idx"]]
            .groupby("data_key")
            .agg(list)
            .to_dict()["cell_idx"]
        )

        return query_name, query_dict

    def construct_data_key(self, sample_id):
        if self.config.integrated:
            data_key = self.config.data_key
        elif sample_id is not None:
            if self.config.data_key in sample_id:
                data_key = sample_id
            else:
                data_key = self.config.data_key + "." + sample_id
        else:
            raise ValueError("Either integrated or sample_id must be provided.")
        return data_key

    def process_chunk(self, id, chrom, start, end, queries, i):
        query = queries.get(id, None)
        if query is not None or not os.path.exists(
            f"{self.root.store.path}/{id}/chrs/{chrom}/chunk_{i}"
        ):
            chunk = self.zio.get_track(query, chrom, start, end)
            if len(chunk.shape) > 1:
                chunk = chunk.sum(0)
            self.root.create_array(
                f"{id}/chrs/{chrom}/chunk_{i}",
                data=chunk,
                chunks=(chunk.shape[0]),
                dtype="uint16",
                overwrite=True,
                compressor=BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),
            )
            self.root[f"{id}"].attrs["id"] = id
            self.root[f"{id}"].attrs["celltype"] = id.split(".")[0]
            self.root[f"{id}"].attrs["n_cells"] = id.split(".")[-1]
            self.root[f"{id}"].attrs["libsize"] = chunk.sum()

    def parallel_process(self, chunk_size, queries):
        chromosomes = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = []
            for chrom in chromosomes:
                for i, start in enumerate(
                    range(0, self.chrom_size.chrom_sizes[chrom], chunk_size)
                ):
                    end = min(start + chunk_size, self.chrom_size.chrom_sizes[chrom])
                    for key in queries:
                        if os.path.exists(
                            f"{self.root.store.path}/{key}/chrs/{chrom}/chunk_{i}"
                        ):
                            continue
                        futures.append(
                            executor.submit(
                                self.process_chunk, key, chrom, start, end, queries, i
                            )
                        )

            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()  # Catch and display exceptions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process genomic data with configurable n_cells depths."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--n_cells", type=str, help="List of n_cells values to process."
    )
    parser.add_argument("--output_dir", help="Override the output directory path.")
    parser.add_argument("--metadata", help="Path to the custom metadata CSV file.")
    parser.add_argument(
        "--n_insertions", type=int, help="Target number of reads for sampling."
    )
    parser.add_argument("--skip", action="store_true", help="Skip the processing step.")
    # Add other configuration overrides as needed
    return parser.parse_args()


def main(args):
    config = Config(args.config)
    config.update_from_args(args)

    processor = CreateCDZ(config)

    if not args.skip:
        metadata = None
        if args.metadata:
            metadata = pd.read_csv(args.metadata)

        queries = {}

        n_insertions = processor.config.n_insertions
        if n_insertions is not None:
            queries.update(
                processor.create_queries(n_insertions=n_insertions, query_df=metadata)
            )
        if processor.config.n_cells is not None:
            n_cells = [int(i) for i in processor.config.n_cells.split(",")]
            [
                queries.update(processor.create_queries(n_cells=i, query_df=metadata))
                for i in n_cells
            ]
        queries = {k: v for k, v in queries.items() if v is not None}
        processor.parallel_process(config.chunk_size, queries)

        # read through all chunks and get the total libsize
        for id in tqdm(processor.root):
            libsize_id = 0
            for chrom in processor.root[id]["chrs"]:
                for chunk in processor.root[id]["chrs"][chrom]:
                    libsize_id += processor.root[id]["chrs"][chrom][chunk][:].sum()
            processor.root[id].attrs["libsize"] = libsize_id

    processor.test_load_cdz()


if __name__ == "__main__":
    args = parse_args()
    main(args)
