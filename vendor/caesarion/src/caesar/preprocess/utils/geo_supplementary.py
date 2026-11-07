from __future__ import annotations

import argparse
import os
import re

import GEOparse
from tqdm import tqdm


def download_geo_series_supplementary(gse_id, download_path, exclude_pattern):
    """
    Download supplementary files associated with a GEO Series ID,
    excluding the files that match the exclude_pattern.
    """
    gse = GEOparse.get_GEO(
        geo=gse_id, destdir=download_path, include_data=True, annotate_gpl=False
    )
    file_list = gse.metadata["supplementary_file"]

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for file in tqdm(file_list):
        if re.search(exclude_pattern, file):
            continue
        else:
            print(f"Downloading: {file}")
            os.system(f"wget -P {download_path} {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download GEO Series supplementary files."
    )
    parser.add_argument("gse_id", type=str, help="GEO Series ID (e.g., 'GSEXXXXXX')")
    parser.add_argument(
        "download_path", type=str, help="Path to save the downloaded files"
    )
    parser.add_argument(
        "--exclude_pattern",
        type=str,
        default="rds",
        help="Pattern to exclude files (default: 'rds')",
    )

    args = parser.parse_args()

    download_geo_series_supplementary(
        args.gse_id, args.download_path, args.exclude_pattern
    )

# Example: python geo_supplementary.py GSE162690 /pmglocal/xf2217/geo/ArchR/ --exclude_pattern rds
