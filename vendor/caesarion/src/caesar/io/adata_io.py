from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from pyranges import PyRanges as pr
from scipy.sparse import csr_matrix
from tqdm import tqdm

from caesar.config.config import Config
from caesar.io.zarr_io import MultiSparseZarrIO
from gcell.rna.gencode import Gencode

sc.settings.n_jobs = -1


# Suppress the pkg_resources deprecation warning from sorted_nearest
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="sorted_nearest",
)


def prepare_adata(adata, min_counts=None, min_cells=None, n_top_genes=None):
    adata_bak = adata.copy()
    adata.obs["counts"] = adata.X.sum(1)
    adata.obs["tissue"] = adata.obs["data_key"].apply(lambda x: x.split("_")[-1])
    if min_counts is not None:
        sc.pp.filter_genes(adata, min_counts=min_counts)
    elif min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e5)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.obs = adata.obs.reset_index(drop=True)
    adata_v = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata_v, max_value=10)
    # regress out depth
    sc.pp.regress_out(adata_v, ["counts"], n_jobs=1)
    sc.tl.pca(adata_v, svd_solver="arpack")
    sc.pp.neighbors(adata_v, n_neighbors=20)
    sc.tl.leiden(adata_v, resolution=0.5)
    sc.tl.umap(adata_v)
    return adata_bak, adata, adata_v


def add_gene_activity_matrix(adata, gencode):
    """
    Add gene activity matrix to adata by summing up counts of peaks that overlap with gene body.
    """
    genebody = gencode.get_genebodies()
    genebody = pr(genebody)
    peaks = pr(adata.var.reset_index(drop=True).reset_index())
    overlap = (
        genebody.extend(5000)
        .join(peaks)
        .df[["gene_name", "index"]]
        .groupby("gene_name")["index"]
        .apply(lambda x: x.unique().astype(int).tolist())
    )
    overlap = overlap[overlap.apply(lambda x: len(x) > 0)]
    # print(overlap)
    X = adata.X.tocsc()

    gene_activity = [np.array(X[:, x].sum(1)) for gene_name, x in tqdm(overlap.items())]
    gene_activity = csr_matrix(np.hstack(gene_activity))
    gene_activity = ad.AnnData(
        gene_activity, obs=adata.obs, var=pd.DataFrame(index=overlap.index.values)
    )
    return gene_activity


def add_gene_activity_matrix_use_encode_re2g(
    adata,
    re2g_file="/manitou/pmg/users/xf2217/bmms/caesar/dataset_scripts/t_exhaustion/encode_re2g.bed.merged.cut.bed.sorted.merged.bed",
):
    """
    Add gene activity matrix to adata by summing up counts of peaks defined by encode_re2g.
    Format of encode_re2g:
    chr1	762066	763367	LINC00115,LINC01128,LOC100288069
    Chromosome, Start, End, Genes
    First overlap encode_re2g with peaks, then sum up counts of peaks for each gene
    """
    # read encode_re2g
    encode_re2g = pd.read_csv(
        re2g_file, sep="\t", header=None, names=["Chromosome", "Start", "End", "Genes"]
    )
    # explode genes
    encode_re2g = encode_re2g.assign(Genes=encode_re2g.Genes.str.split(",")).explode(
        "Genes"
    )
    # convert to pyranges
    encode_re2g = pr(encode_re2g)
    # convert adata.var to pyranges
    peaks = pr(adata.var.reset_index(drop=True).reset_index())
    # overlap encode_re2g with peaks
    overlap = (
        encode_re2g.join(peaks)
        .df[["Genes", "index"]]
        .groupby("Genes")["index"]
        .apply(lambda x: x.unique().astype(int).tolist())
    )
    overlap = overlap[overlap.apply(lambda x: len(x) > 0)]
    # for each gene, calculate the sum of counts of peaks
    X = adata.X.tocsc()
    gene_activity = [np.array(X[:, x].sum(1)) for gene_name, x in tqdm(overlap.items())]
    gene_activity = csr_matrix(np.hstack(gene_activity))
    gene_activity = ad.AnnData(
        gene_activity, obs=adata.obs, var=pd.DataFrame(index=overlap.index.values)
    )
    return gene_activity


def get_neighbor_genes(adata, gencode, extend=250000):
    peaks = pr(adata.var.reset_index(drop=True).reset_index())
    genebody = gencode.get_genebodies()
    genebody = pr(genebody)
    overlap = peaks.join(genebody.extend(extend)).df[["index", "gene_name"]]
    overlap = overlap.rename(columns={"index": "peak_id"})
    overlap["peak_id"] = overlap["peak_id"].astype(int)
    neighbor_genes = overlap.groupby("peak_id")["gene_name"].apply(
        lambda x: x.unique().tolist()
    )
    neighbor_peaks = overlap.groupby("gene_name")["peak_id"].apply(
        lambda x: x.unique().tolist()
    )

    # to dict
    neighbor_peaks = neighbor_peaks.to_dict()
    neighbor_genes = neighbor_genes.to_dict()
    return neighbor_genes, neighbor_peaks


def prepare_gene_activity(adata, gencode):
    gene_activity = add_gene_activity_matrix(adata, gencode)
    gene_activity.obs["counts"] = gene_activity.X.sum(1)
    gene_activity.var["libsize"] = np.array(gene_activity.X.sum(0)).flatten()
    gene_activity.obs["tissue"] = gene_activity.obs["data_key"].apply(
        lambda x: x.split("_")[-1]
    )
    gene_activity_bak = gene_activity.copy()
    sc.pp.filter_cells(gene_activity, min_counts=5000)
    sc.pp.filter_genes(gene_activity, min_counts=500)
    sc.pp.normalize_total(gene_activity, target_sum=1e5)
    sc.pp.log1p(gene_activity)
    sc.pp.highly_variable_genes(gene_activity, n_top_genes=5000)
    gene_activity_v = gene_activity[:, gene_activity.var.highly_variable]
    sc.pp.scale(gene_activity_v, max_value=10)
    sc.tl.pca(gene_activity_v, svd_solver="arpack")
    sc.pp.neighbors(gene_activity_v, n_neighbors=20)
    sc.tl.leiden(gene_activity_v, resolution=0.5)
    sc.tl.umap(gene_activity_v)
    return gene_activity_bak, gene_activity, gene_activity_v


def get_correlated_peaks(gene_peaks, gene_activity_g):
    # get correlation
    corr = []
    for i in tqdm(range(gene_peaks.shape[1])):
        corr.append(np.corrcoef(gene_peaks[:, i].flatten(), gene_activity_g)[0, 1])
    corr = np.array(corr)
    return corr


def parallel_correlation_analysis(
    genes,
    neighbor_peaks,
    adata_X,
    gene_activity_X,
    gene_idx_dict,
    peaks,
    output=None,
    n_jobs=96,
):
    """
    Perform parallel correlation analysis for a list of genes.

    :param genes: List of genes to analyze.
    :param neighbor_peaks: Dictionary of neighboring peaks for each gene.
    :param adata: AnnData object.
    :param gene_activity: Gene activity data.
    :return: Dictionary of correlation results for each gene.
    """
    results = {}
    df = []
    with ProcessPoolExecutor(n_jobs) as executor:
        for gene in tqdm(genes):
            if gene not in neighbor_peaks:
                continue
            gene_peaks = neighbor_peaks[gene]
            peak_df = peaks.iloc[neighbor_peaks[gene]]
            gene_peaks = adata_X[:, gene_peaks]  # a cell by peak matrix
            gene_activity_g = gene_activity_X[:, gene_idx_dict[gene]]  # a cell vector
            futures = executor.submit(get_correlated_peaks, gene_peaks, gene_activity_g)
            results[gene] = futures.result()
            peak_df.loc[:, "corr"] = results[gene]
            peak_df.loc[:, "gene"] = gene
            df.append(peak_df)
    df = pd.concat(df)
    if output is not None:
        # write to csv
        df.to_csv(output, index=True)
    return df


def get_averaged_X_over_knn(X, connectivities):
    """
    Get averaged X over k nearest neighbors.
    """
    averaged_X = []
    for i in tqdm(range(X.shape[0])):
        neighbors = connectivities[i].nonzero()[1]
        averaged_X.append(X[neighbors, :].mean(0))
    averaged_X = np.vstack(averaged_X)
    averaged_X = np.nan_to_num(averaged_X, 0)
    return averaged_X


def main(args):
    config = Config(args.config)
    zarr_dir = os.path.join(config.root_path, "output")
    output_dir = os.path.join(config.root_path, "analysis")
    zio = MultiSparseZarrIO(zarr_dir, quick_look=False, mode="r")
    if not os.path.exists(os.path.join(output_dir, "adata_bak.h5ad")):
        adata = zio.create_adata()
        adata_bak, adata, adata_v = prepare_adata(adata)
        # save to h5df

        # create if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # if not exist
        adata_bak.write(os.path.join(output_dir, "adata_bak.h5ad"))
        adata.write(os.path.join(output_dir, "adata.h5ad"))
        adata_v.write(os.path.join(output_dir, "adata_v.h5ad"))
    else:
        adata_bak = ad.read_h5ad(os.path.join(output_dir, "adata_bak.h5ad"))
        adata = ad.read_h5ad(os.path.join(output_dir, "adata.h5ad"))
        adata_v = ad.read_h5ad(os.path.join(output_dir, "adata_v.h5ad"))
    # get gene activity
    gencode = Gencode(
        assembly=config.assembly, version=44, gtf_dir=config.annotation_dir
    )
    neighbor_genes, neighbor_peaks = get_neighbor_genes(adata_v, gencode)
    gene_activity_bak, gene_activity, gene_activity_v = prepare_gene_activity(
        adata, gencode
    )
    # save to h5df
    gene_activity_bak.write(os.path.join(output_dir, "gene_activity_bak.h5ad"))
    gene_activity.write(os.path.join(output_dir, "gene_activity.h5ad"))
    gene_activity_v.write(os.path.join(output_dir, "gene_activity_v.h5ad"))
    if args.compute_correlation:
        # convert to dense matrix
        adata_X = adata[gene_activity_v.obs.cell_idx_to_adata.values, :].X.todense()
        adata_X = np.array(adata_X)
        gene_activity_v_X = gene_activity_v.X
        gene_idx_dict = (
            gene_activity_v.var.reset_index()
            .reset_index()
            .set_index("index")
            .to_dict()["level_0"]
        )
        # get averaged X over k nearest neighbors
        knn = adata_v.obsp["connectivities"][gene_activity_v.obs.index.values][
            :, gene_activity_v.obs.index.values
        ]
        gene_activity_v_X_avg = get_averaged_X_over_knn(gene_activity_v_X, knn)
        adata_X_avg = get_averaged_X_over_knn(adata_X, knn)
        peaks = adata_v.var.reset_index()
        parallel_correlation_analysis(
            gene_activity_v.var.index.values,
            neighbor_peaks,
            adata_X_avg,
            gene_activity_v_X_avg,
            gene_idx_dict,
            peaks,
            output=os.path.join(
                output_dir, "peak_gene_correlation_results.csv", n_jobs=config.n_jobs
            ),
        )


def setup_argparse():
    import argparse

    parser = argparse.ArgumentParser(description="Caesar analysis")
    parser.add_argument("--config", "-c", type=str, help="Path to config file")
    parser.add_argument(
        "--compute_correlation",
        action="store_true",
        help="Whether to compute correlation",
    )
    return parser


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    main(args)
