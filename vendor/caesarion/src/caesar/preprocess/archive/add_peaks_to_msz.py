from __future__ import annotations

import argparse
import os
from glob import glob

from caesar.config.config import Config
from caesar.io.zarr_io import MultiSparseZarrIO


class AddPeaksToMSZ:
    def __init__(self, config):
        self.config = config
        self.zio = MultiSparseZarrIO(
            self.config.root_path + "/output/", "*.zarr", quick_look=False, mode="a"
        )

    def write_peaks(self, peak_files_dir, peak_name):
        files = glob(f"{peak_files_dir}/*")
        peak_dict = {
            f"{self.config.data_key}.{os.path.basename(f).split('.')[0]}": f
            for f in files
        }
        self.zio.write_peaks(peak_dict, name=peak_name)
        print(f"Peaks written for {peak_name}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Process and store peak data.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--peak_files_dir",
        type=str,
        required=True,
        help="Directory containing peak files.",
    )
    parser.add_argument(
        "--peak_name",
        type=str,
        required=True,
        help="Name to store the peak data under.",
    )
    # Add other configuration overrides as needed
    return parser.parse_args()


def main(args):
    config = Config(args.config)
    config.update_from_args(args)

    processor = AddPeaksToMSZ(config)
    processor.write_peaks(args.peak_files_dir, args.peak_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)

# python add_peaks_to_msz.py --config path/to/config.yaml --peak_files_dir /path/to/peaks_p0.01 --peak_name peaks_p0.01
