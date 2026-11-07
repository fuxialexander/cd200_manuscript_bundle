#!/usr/bin/env python3

from __future__ import annotations

import argparse

# Standard library imports
import logging
from pathlib import Path

# Third-party imports
import pandas as pd

# Local imports
from .sc_accessibility_cpeaks import (
    normalize_pseudobulk,
    process_pseudobulk_with_clusters,
    setup_logging,
)


def main():
    """Main function to process pseudobulk data from existing sparse zarr files."""
    parser = argparse.ArgumentParser(
        description="Generate pseudobulk dense zarr from existing sparse zarr using new metadata"
    )
    parser.add_argument(
        "--sparse-dir",
        required=True,
        help="Directory containing sparse zarr files",
    )
    parser.add_argument(
        "--cluster-assignments",
        required=True,
        help="Path to cluster assignments CSV with columns: cluster_name, data_key, barcode",
    )
    parser.add_argument(
        "--peak-file",
        help="Path to peak file for normalization (optional)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for dense zarr files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=64,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed progress",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate cluster assignments file
    try:
        cluster_df = pd.read_csv(args.cluster_assignments)
        required_cols = ["cluster_names", "data_key", "barcodes"]
        if not all(col in cluster_df.columns for col in required_cols):
            raise ValueError(
                f"Cluster assignments must contain columns: {required_cols}"
            )
    except Exception as e:
        logging.error(f"Error reading cluster assignments file: {e}")
        raise

    try:
        # Process pseudobulk with cluster assignments
        dense_path = Path(args.output_dir) / "celltype_dense.zarr"
        if not dense_path.exists() or args.overwrite:
            logging.info("Processing pseudobulk with cluster assignments...")
            process_pseudobulk_with_clusters(
                sparse_zarr=args.sparse_dir,
                cluster_assignments=cluster_df,
                peak_file=args.peak_file,
                blacklist_file=None,
                dense_dir=args.output_dir,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
            )

        # Normalize the pseudobulk signals
        logging.info("Normalizing pseudobulk signals...")
        output_zarr_path = Path(args.output_dir) / "pseudobulk_dense_norm.zarr"
        normalize_pseudobulk(
            dense_path=dense_path,
            peak_file=args.peak_file,
            n_jobs=args.n_jobs,
            output_zarr_path=output_zarr_path,
        )

        logging.info("Processing completed successfully!")
        logging.info(f"Output files are in: {args.output_dir}")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
