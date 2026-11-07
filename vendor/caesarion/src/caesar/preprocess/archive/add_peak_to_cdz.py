from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from caesar.io.zarr_io import CelltypeDenseZarrIO, MultiSparseZarrIO

if TYPE_CHECKING:
    from caesar.config.config import Config


def parse_narrowpeaks_centered(path, target_length=500):
    peaks = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "Chromosome",
            "Start",
            "End",
            "Name",
            "Score",
            "Strand",
            "SignalValue",
            "PValue",
            "QValue",
            "Summit",
        ],
    )
    peaks = peaks[["Chromosome", "Start", "End", "Summit"]]
    peaks["Start"] = peaks["Start"] + peaks["Summit"] - target_length // 2
    peaks["End"] = peaks["Start"] + target_length
    return peaks


def parse_narrowpeaks(path):
    peaks = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "Chromosome",
            "Start",
            "End",
            "Name",
            "Score",
            "Strand",
            "SignalValue",
            "PValue",
            "QValue",
            "Summit",
        ],
    )
    peaks = peaks[["Chromosome", "Start", "End", "Summit"]]
    return peaks


def parse_bed3(path):
    peaks = pd.read_csv(
        path, sep="\t", header=None, names=["Chromosome", "Start", "End"]
    )
    peaks = peaks[["Chromosome", "Start", "End"]]
    return peaks


class AddPeakToCDZ:
    def __init__(self, config):
        self.config = config
        print(config)
        self.specific_peak_names = config.specific_peak_names
        if hasattr(config, "merged_peak_name"):
            self.merged_peak_name = config.merged_peak_name
        else:
            self.merged_peak_name = f"{'.'.join(self.specific_peak_names)}_tissue"
        if config.data_type == "sc":
            self.zio = MultiSparseZarrIO(
                self.config.root_path + "/output/",
                "*.zarr",
                quick_look=False,
                mode="r",
                peaks_name=self.specific_peak_names,
            )

        self.cdz = CelltypeDenseZarrIO(
            f"{self.config.output_dir}/{self.config.data_key}_dense.zarr", mode="a"
        )

    def get_data_key_specific_peaks(self, id, specific_peak_name):
        if len(id.split(".")) == 4:
            dk = id.split(".")[1] + "." + id.split(".")[2]
        elif len(id.split(".")) == 3:
            dk = id.split(".")[1]
        if self.config.data_type == "sc":
            print(self.zio.zarr_dict)
            peaks = self.zio.zarr_dict[dk].get_peaks(specific_peak_name)
        elif self.config.data_type == "encode_bulk":
            path = os.path.join(
                self.config.root_path,
                specific_peak_name,
                dk.split(".")[1] + "_peaks.narrowPeak*",
            )
            # find file start with path
            path = glob(path)[0]
            print(path)
            peaks = parse_narrowpeaks(path)
        return peaks

    def write_data_key_specific_peaks(self, specific_peak_name):
        for id in list(self.cdz.ids):
            print(id)
            peaks = self.get_data_key_specific_peaks(id, specific_peak_name)
            self.cdz.write_peaks(
                id=id,
                peaks=peaks,
                name=specific_peak_name,
                overwrite=True,
                columns_to_include=["Start", "End"],
            )

    def write_merged_peaks(self, specific_peak_name):
        if self.config.data_type == "encode_bulk":
            for id in list(self.cdz.ids):
                peaks = self.get_data_key_specific_peaks(id, specific_peak_name)
                self.cdz.write_peaks(
                    id=id,
                    peaks=peaks,
                    name=f"{specific_peak_name}_tissue",
                    overwrite=True,
                    columns_to_include=["Start", "End"],
                )
        else:
            merged_peaks = self.zio.get_merged_peaks(specific_peak_name).df
            for id in list(self.cdz.ids):
                self.cdz.write_peaks(
                    id=id,
                    peaks=merged_peaks,
                    name=f"{specific_peak_name}_tissue",
                    overwrite=True,
                )

    def process_celltype(self, celltype, merged_peak_name):
        peaks = self.cdz.get_peak_counts(celltype, merged_peak_name)
        peaks["Count"] = peaks["Count"].astype("int64")
        peaks = peaks.query("Count > 0")
        peaks["aTPM"] = np.log10(peaks.Count / peaks.Count.sum() * 1e5 + 1)
        peaks["aTPM"] = peaks["aTPM"]
        self.cdz.write_peaks(
            celltype,
            peaks,
            f"{merged_peak_name}_open",
            overwrite=True,
            columns_to_include=["Start", "End", "Count", "aTPM"],
        )

    def filter_peaks(self, specific_peak_name):
        merged_peak_name = f"{specific_peak_name}_tissue"
        with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = [
                executor.submit(self.process_celltype, celltype, merged_peak_name)
                for celltype in tqdm(self.cdz.ids)
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()  # Catch and display exceptions


@hydra.main(config_path="../config/", config_name="default", version_base="1.3")
def main(config: Config):
    print(OmegaConf.to_yaml(config))
    processor = AddPeakToCDZ(config)
    if "peaks_tissue_open" not in processor.cdz.dataset[processor.cdz.ids[0]]:
        # call peaks using cdz method
        peaks = processor.cdz.call_peaks(n_jobs=config.n_jobs, min_length=100)
        # add peaks to cdz
        for id in processor.cdz.ids:
            processor.cdz.write_peaks(
                id=id, peaks=peaks, name=config.specific_peak_names, overwrite=True
            )
            processor.cdz.write_peaks(
                id=id,
                peaks=peaks,
                name=config.specific_peak_names + "_tissue",
                overwrite=True,
            )
        processor.filter_peaks(config.specific_peak_names)
    # if hasattr(config, 'specific_peak_name'):
    #     print(f"Adding specific peak {config.specific_peak_name} to CDZ.")
    #     processor.write_data_key_specific_peaks(config.specific_peak_name)
    #     processor.write_merged_peaks(config.specific_peak_name)
    #     processor.filter_peaks(config.specific_peak_name)
    #     processor.cdz.libsize
    #     print("Finished processing.")
    #     return
    # for specific_peak_name in processor.specific_peak_names:
    #     if specific_peak_name not in processor.cdz.dataset[processor.cdz.ids[0]] or config.overwrite:
    #         processor.write_data_key_specific_peaks(specific_peak_name)
    # for specific_peak_name in processor.specific_peak_names:
    #     merged_peak_name = f"{specific_peak_name}_tissue"
    #     if merged_peak_name not in processor.cdz.dataset[processor.cdz.ids[0]] or config.overwrite:
    #         processor.write_merged_peaks(specific_peak_name)
    #     if merged_peak_name + '_open' not in processor.cdz.dataset[processor.cdz.ids[0]] or config.overwrite:
    #         processor.filter_peaks(specific_peak_name)
    # save peak bed file with aTPM
    for id in processor.cdz.ids:
        peak_to_save = "peaks_tissue_open"
        peaks = processor.cdz.get_peaks(id, peak_to_save)
        peaks = peaks.query("Count > 0")
        peaks = peaks[["Chromosome", "Start", "End", "aTPM"]]
        peaks.to_csv(f"{id}.bed", sep="\t", index=False, header=False)
    processor.cdz.libsize
    print("Finished processing.")


if __name__ == "__main__":
    main()
