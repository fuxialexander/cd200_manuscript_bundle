from __future__ import annotations

import argparse

import numpy as np

np.bool = np.bool_
import pandas as pd
from pyranges import PyRanges as pr
from scipy.sparse import csr_matrix
from tqdm import tqdm

from caesar.config.config import Config
from caesar.io.gencode import Gencode
from caesar.io.zarr_io import CelltypeDenseZarrIO


def count_to_log_tpm(count):
    return np.log10(count / count.sum() * 1e6 + 1)


class AddExpressionToCDZ:
    def __init__(self, config, metadata_filter):
        self.config = config
        self.cdz = CelltypeDenseZarrIO(
            f"{config.output_dir}/{config.data_key}_dense.zarr", mode="a"
        )
        self.gencode = Gencode(gtf_dir=config.annotation_dir)
        self.metadata_filter = metadata_filter

    def get_pseudobulk_from_adata(self, adata):
        pseudobulk = {}
        for celltype in adata.obs[self.config.labels_column].unique():
            total_count = np.array(
                adata[adata.obs[self.config.labels_column] == celltype].X.sum(0)
            ).flatten()
            tpm = count_to_log_tpm(total_count)
            pseudobulk[celltype] = tpm
        return pd.DataFrame(pseudobulk, index=adata.var_names)

    def get_pseudobulk(self, sample):
        path = f"{self.config.root_path}/rna/{sample}.filtered_feature_bc_matrix.h5"
        adata = sc.read_10x_h5(path)
        metadata = self.metadata_filter(f"{self.config.root_path}/metadata/{sample}")[0]
        adata = adata[adata.obs.index.isin(metadata.barcode)]
        adata.obs = metadata.set_index("barcode").loc[adata.obs.index]
        return self.get_pseudobulk_from_adata(adata)

    def add_expression_to_cdz(self):
        samples = np.unique([id.split(".")[2] for id in self.cdz.ids])
        samples_pseudobulk = {
            sample: self.get_pseudobulk(sample) for sample in tqdm(samples)
        }
        for id in tqdm(self.cdz.ids):
            self.process_expression(id, samples_pseudobulk[samples])

    def process_expression(self, id, pseudobulk):
        id.split(".")[2]
        celltype = id.split(".")[0]
        exp = (
            pseudobulk[celltype]
            .reset_index()
            .rename(columns={"index": "gene_name", celltype: "TPM"})
        )
        exp = pd.merge(
            self.gencode.gtf,
            exp[["gene_name", "TPM"]].drop_duplicates(),
            left_on="gene_name",
            right_on="gene_name",
        )
        exp["Strand"] = exp.Strand.map({"+": 0, "-": 1})
        # category to int
        exp["Strand_exp"] = exp["Strand"].astype(int)
        self.write_expression(exp, id)

    def write_expression(self, exp, id, peak_name=None):
        if peak_name is None:
            peak_name = f"{self.config.merged_peak_name}"
        peaks = self.cdz.get_peaks(id, name=f"{peak_name}_open")
        overlap = (
            pr(peaks.reset_index(drop=True).reset_index())
            .join(pr(exp).extend(300), suffix="_exp", how="left")
            .df[
                [
                    "index",
                    "Chromosome",
                    "Start",
                    "End",
                    "gene_name",
                    "Strand_exp",
                    "TPM",
                ]
            ]
            .drop_duplicates()
        )
        row_col_data = (
            overlap.query('gene_name!="-1" & TPM>=0')
            .groupby(["index", "Strand_exp"])
            .TPM.max()
            .fillna(0)
            .reset_index()
            .values
        )
        row, col, data = row_col_data[:, 0], row_col_data[:, 1], row_col_data[:, 2]
        row = row.astype(np.int64)
        col = col.astype(np.int64)
        data = data.astype(np.float32)
        exp_array = csr_matrix((data, (row, col)), shape=(len(peaks), 2)).todense()
        peaks = peaks.reset_index(drop=True)
        peaks["TSS"] = 0
        peaks["Expression_positive"] = exp_array[:, 0]
        peaks["Expression_negative"] = exp_array[:, 1]
        peaks["TSS"][np.unique(row)] = 1
        peaks["aTPM"] = np.log10(peaks.Count / peaks.Count.sum() * 1e5 + 1)
        peaks["aTPM"] = peaks["aTPM"] / peaks["aTPM"].max()
        self.cdz.write_peaks(
            id,
            peaks,
            f"{peak_name}_open_exp",
            [
                "Start",
                "End",
                "Expression_positive",
                "Expression_negative",
                "aTPM",
                "Count",
                "TSS",
            ],
            overwrite=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Add expression data to CDZ.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data."
    )
    return parser.parse_args()


def main(args):
    from caesar.preprocess.metadata_filters import (
        filter_cell_metadata_vijay_hematopoiesis as filter_cell_metadata,
    )

    config = Config(args.config)
    config.update_from_args(args)

    processor = AddExpressionToCDZ(config, filter_cell_metadata)
    if (
        processor.config.merged_peak_name + "_open_exp"
        not in processor.cdz.dataset[processor.cdz.ids[0]]
    ):
        processor.add_expression_to_cdz()


if __name__ == "__main__":
    args = parse_args()
    main(args)
