from __future__ import annotations

import pandas as pd


def filter_cell_metadata_joung_tfatlas(
    metadata_path: str, depth_threshold: int = 0
) -> pd.DataFrame:
    # Use gzip
    cell_metadata = pd.read_csv(metadata_path, sep=",")
    # set column name [0] -> Barcodes
    columns = cell_metadata.columns.values
    columns[0] = "Barcodes"
    cell_metadata.columns = columns
    filtered_metadata = cell_metadata
    # Get a dict of filtered barcodes and their indices, this will be the index of the barcode array in zarr
    barcodes = filtered_metadata.Barcodes.values
    barcodes_dict = {k: i for i, k in enumerate(barcodes)}
    return filtered_metadata, barcodes_dict


def filter_cell_metadata_t_exhaustion(
    metadata_path: str, depth_threshold: int = 0
) -> pd.DataFrame:
    cell_metadata = pd.read_csv(metadata_path, sep="\t", compression="gzip")
    filtered_metadata = cell_metadata
    # Get a dict of filtered barcodes and their indices, this will be the index of the barcode array in zarr
    barcodes = filtered_metadata.Barcodes.values
    barcodes_dict = {k: i for i, k in enumerate(barcodes)}
    return filtered_metadata, barcodes_dict


def filter_cell_metadata_linnarsson_brain(
    metadata_path: str, depth_threshold: int = 0
) -> pd.DataFrame:
    # Use gzip
    cell_metadata = pd.read_csv(metadata_path, sep=",", header=0, index_col=0)
    filtered_metadata = cell_metadata
    # Get a dict of filtered barcodes and their indices, this will be the index of the barcode array in zarr, split by ':' and get the first part
    barcodes = filtered_metadata.index.str.split(":").str[1]
    # append '-1' to the end of the barcode
    barcodes = barcodes + "-1"
    barcodes = barcodes.values
    barcodes_dict = {k: i for i, k in enumerate(barcodes)}
    return filtered_metadata, barcodes_dict


def filter_cell_metadata_yanhu_bmms(
    metadata_path: str, depth_threshold: int = 500
) -> pd.DataFrame:
    cell_metadata = pd.read_csv(metadata_path, sep="\t")
    filtered_metadata = cell_metadata.sort_values(
        ["cistopic.louvain", "depth"], ascending=[True, False]
    ).query(f"depth > {depth_threshold}")
    # Get a dict of filtered barcodes and their indices, this will be the index of the barcode array in zarr
    barcodes = filtered_metadata.index.values
    barcodes_dict = {k: i for i, k in enumerate(barcodes)}
    return filtered_metadata, barcodes_dict


def filter_cell_metadata_shendure_fetal(
    metadata_path: str, depth_threshold: int = 0
) -> pd.DataFrame:
    cell_metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        names=["barcode", "logUMI", "tsse", "sample_id", "cell type", "life stage"],
    )
    filtered_metadata = cell_metadata
    # Get a dict of filtered barcodes and their indices, this will be the index of the barcode array in zarr
    barcodes = filtered_metadata.barcode.values
    barcodes_dict = {k: i for i, k in enumerate(barcodes)}
    print(f"Number of cells: {len(barcodes)}")
    return filtered_metadata, barcodes_dict


def filter_cell_metadata_archr_cellline(
    metadata_path: str, depth_threshold: int = 500
) -> pd.DataFrame:
    cell_metadata = pd.read_csv(metadata_path, sep=",")
    cell_metadata.columns = ["barcode"] + list(cell_metadata.columns[1:])
    cell_metadata["barcode"] = cell_metadata["barcode"].str.split("#").str[1]
    filtered_metadata = (
        cell_metadata.rename({"nFrags": "depth"}, axis=1)
        .sort_values(["barcode", "depth"], ascending=[True, False])
        .query(f"depth > {depth_threshold}")
        .query('barcode!="."')
    )
    # Get a dict of filtered barcodes and their indices, this will be the index of the barcode array in zarr
    barcodes = filtered_metadata.barcode.values
    barcodes_dict = {k: i for i, k in enumerate(barcodes)}
    return filtered_metadata, barcodes_dict


def filter_cell_metadata_htan(
    metadata_path: str, depth_threshold: int = 0
) -> pd.DataFrame:
    # Use gzip
    cell_metadata = pd.read_csv(metadata_path, sep="\t")
    filtered_metadata = cell_metadata
    # Get a dict of filtered barcodes and their indices, this will be the index of the barcode array in zarr
    barcodes = filtered_metadata.Barcode.values
    barcodes_dict = {k: i for i, k in enumerate(barcodes)}
    return filtered_metadata, barcodes_dict


def filter_cell_metadata_vijay_hematopoiesis(
    metadata_path: str, depth_threshold: int = 0
) -> pd.DataFrame:
    cell_metadata = pd.read_csv(metadata_path).rename({"Unnamed: 0": "barcode"}, axis=1)
    cell_metadata["STD.CellType"] = cell_metadata["STD.CellType"].map(
        lambda x: x if x != "Refined.HSC" else "HSC"
    )
    filtered_metadata = cell_metadata
    filtered_metadata["barcode"] = (
        filtered_metadata.barcode.str.split("-").str[0] + "-1"
    )
    # Get a dict of filtered barcodes and their indices, this will be the index of the barcode array in zarr
    barcodes = filtered_metadata.barcode.values

    barcodes_dict = {k: i for i, k in enumerate(barcodes)}
    return filtered_metadata, barcodes_dict
