"""
Utilities for loading and extracting sequence data.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from caesar.io.zarr_io.dense import SequenceDenseZarrIO
from caesar.utils import is_torch_available, requires_torch

logger = logging.getLogger(__name__)


NARROWPEAK_COLUMNS = [
    "chrom",
    "start",
    "end",
    "name",
    "score",
    "strand",
    "signalValue",
    "pValue",
    "qValue",
    "peak",
]


def _open_text_maybe_gzip(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return Path(path).open()


def _bed_skiprows_and_column_count(path) -> tuple[list[int], int]:
    skiprows: list[int] = []
    num_columns = 0
    with _open_text_maybe_gzip(path) as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith(("track", "browser"))
            ):
                skiprows.append(line_number)
                continue
            num_columns = len(stripped.split("\t"))
            break
    if num_columns < 3:
        raise ValueError(f"Expected at least 3 BED columns in {path!s}.")
    return skiprows, num_columns


def _bed_column_names(num_columns: int) -> list[str]:
    if num_columns <= len(NARROWPEAK_COLUMNS):
        return NARROWPEAK_COLUMNS[:num_columns]
    extras = [f"extra_{index}" for index in range(len(NARROWPEAK_COLUMNS), num_columns)]
    return [*NARROWPEAK_COLUMNS, *extras]


def read_narrowpeak(narrowpeak_file):
    """
    Read a narrowPeak file and return regions as a pandas DataFrame.

    Args:
        narrowpeak_file: Path to narrowPeak file (.bed or .bed.gz)

    Returns:
        DataFrame with columns: chrom, start, end, name, score, strand, signalValue, pValue, qValue, peak
    """
    logger.info(f"Reading narrowPeak file: {narrowpeak_file}")

    # Check if this is a ReMap file and use pyranges for reading
    if "remap" in str(narrowpeak_file).lower():
        import pyranges as pr

        logger.info("Detected ReMap file, using pyranges read_bed")
        # Use pyranges to read the bed file (handles gzip automatically)
        gr = pr.read_bed(narrowpeak_file)
        df = gr.df
        # Ensure we have the basic columns needed
        if "Name" in df.columns:
            df = df.rename(columns={"Name": "name"})
        if "Score" in df.columns:
            df = df.rename(columns={"Score": "score"})
        # Convert to lowercase column names to match expected format
        df.columns = df.columns.str.lower()
        # rename chromosome to chrom
        df = df.rename(columns={"chromosome": "chrom"})
    else:
        # Original logic for non-ReMap files
        # detect number of columns in the file
        skiprows, num_columns = _bed_skiprows_and_column_count(narrowpeak_file)
        columns = _bed_column_names(num_columns)

        with _open_text_maybe_gzip(narrowpeak_file) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=columns,
                skiprows=skiprows,
            )

    df["length"] = df["end"] - df["start"]
    # df = df.query("length > 200")
    df["center"] = (df["start"] + df["end"]) / 2
    df["start"] = (df["center"] - 1000).astype(int)
    df["end"] = (df["center"] + 1000).astype(int)
    # if the score column is integer:
    # if "score" in df.columns and df["score"].dtype == int and df.score.max() >= 3:
    #     df = df.query(

    #         "score>3"
    #     )  # when use remap data, the score means number of cell context this appears
    logger.info(
        f"Loaded {len(df)} regions from {len(df['chrom'].unique())} chromosomes"
    )
    chrom_counts = df["chrom"].value_counts().sort_index()
    for chrom, count in chrom_counts.items():
        logger.info(f"  {chrom}: {count} regions")
    if len(df) > 1000000:
        return df.sample(n=1000000)
    else:
        return df


def extract_sequences_from_regions(
    seq_io: SequenceDenseZarrIO, 
    regions_df: pd.DataFrame, 
    extend_bp: int = 0,
    extend_right_only: bool = False,
):
    """
    Extract sequences from specified regions using efficient batch processing.

    Args:
        seq_io: SequenceDenseZarrIO object
        regions_df: DataFrame with chrom, start, end columns
        extend_bp: Number of base pairs to extend (on both sides if extend_right_only=False, 
                   on right side only if extend_right_only=True)
        extend_right_only: If True, only extend on the right side (end position)

    Returns:
        Tuple of (sequences, region_info) where sequences is a list of numpy arrays
        and region_info is a list of dictionaries with region metadata
    """
    logger.info(f"Extracting sequences from {len(regions_df)} regions")
    if extend_bp > 0:
        if extend_right_only:
            logger.info(f"Extending regions by {extend_bp} bp on right side only")
        else:
            logger.info(f"Extending regions by {extend_bp} bp on both sides")

    sequences = []
    region_info = []

    # Process all regions (filtering should be done by caller if needed)
    logger.info(f"Processing {len(regions_df)} regions")

    for chrom, chrom_regions in regions_df.groupby("chrom"):
        logger.info(f"Processing {len(chrom_regions)} regions on {chrom}")

        # Pre-process all regions to compute extended boundaries
        valid_region_indices = []

        for idx, (orig_idx, row) in enumerate(chrom_regions.iterrows()):
            # Store original boundaries for saving
            original_start = row["start"]
            original_end = row["end"]
            
            # Apply extension for scanning
            if extend_right_only:
                # Only extend on the right side
                start = original_start
                end = min(seq_io.chrom_sizes[chrom], original_end + extend_bp)
            else:
                # Extend on both sides
                start = max(0, original_start - extend_bp)
                end = min(seq_io.chrom_sizes[chrom], original_end + extend_bp)

            if end <= start:
                logger.warning(
                    f"Skipping empty region after extension: {chrom}:{original_start}-{original_end}"
                )
                continue

            valid_region_indices.append((orig_idx, row, start, end, original_start, original_end))

        if not valid_region_indices:
            logger.warning(f"No valid regions found for chromosome {chrom}")
            continue

        # Load coalesced chunk-aware blocks instead of one min-to-max span.
        # The old whole-span read is efficient for clustered regions but can
        # over-read hundreds of Mb for sparse peak subsets.
        chunk_size = getattr(seq_io, "chunk_size", None) or 100_000
        max_gap_bp = int(chunk_size)
        max_block_span_bp = max(1_000_000, int(chunk_size) * 4)
        sorted_regions = sorted(valid_region_indices, key=lambda item: (item[2], item[3]))
        blocks = []
        current_block = None

        for region in sorted_regions:
            start, end = region[2], region[3]
            if current_block is None:
                current_block = {
                    "start": start,
                    "end": end,
                    "regions": [region],
                }
                continue

            proposed_end = max(current_block["end"], end)
            proposed_span = proposed_end - current_block["start"]
            close_enough = start <= current_block["end"] + max_gap_bp
            if close_enough and proposed_span <= max_block_span_bp:
                current_block["end"] = proposed_end
                current_block["regions"].append(region)
            else:
                blocks.append(current_block)
                current_block = {
                    "start": start,
                    "end": end,
                    "regions": [region],
                }

        if current_block is not None:
            blocks.append(current_block)

        logger.debug(
            "Coalesced %d %s regions into %d sequence blocks",
            len(valid_region_indices),
            chrom,
            len(blocks),
        )

        extracted_for_chrom = []
        for block in blocks:
            block_start = int(block["start"])
            block_end = int(block["end"])
            try:
                chrom_sequence = seq_io.get_track(
                    chr_name=chrom,
                    start=block_start,
                    end=block_end,
                    output_format="raw_array",
                ).astype(np.float32)
                logger.debug(
                    f"Loaded chromosome {chrom} block {block_start}-{block_end} ({chrom_sequence.shape[0]:,} bp)"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load chromosome {chrom} block {block_start}-{block_end}: {e}"
                )
                continue

            for orig_idx, row, start, end, original_start, original_end in block["regions"]:
                try:
                    relative_start = start - block_start
                    relative_end = end - block_start
                    track = chrom_sequence[relative_start:relative_end]
                    extracted_for_chrom.append(
                        (
                            orig_idx,
                            track,
                            {
                                "chrom": chrom,
                                "start": start,
                                "end": end,
                                "original_start": original_start,
                                "original_end": original_end,
                                "length": end - start,
                                "original_idx": orig_idx,
                            },
                        )
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract sequence for region {chrom}:{start}-{end}: {e}"
                    )
                    continue

            del chrom_sequence

        # Preserve caller region order within and across coalesced blocks.
        extracted_for_chrom.sort(key=lambda item: item[0])
        for _, track, info in extracted_for_chrom:
            sequences.append(track)
            region_info.append(info)

    if not sequences:
        raise ValueError("No sequences could be extracted from the regions")

    ordered = sorted(
        zip(sequences, region_info, strict=False),
        key=lambda item: item[1]["original_idx"],
    )
    sequences = [item[0] for item in ordered]
    region_info = [item[1] for item in ordered]

    sequences = [seq.astype(np.float32) for seq in sequences]
    logger.info(f"Extracted {len(sequences)} sequences")
    if len(sequences) > 0:
        lengths = [s.shape[0] for s in sequences]
        logger.info(
            f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, median={np.median(lengths):.0f}"
        )

    return sequences, region_info


@requires_torch
def reverse_complement_batch(batch_tensor):
    """Computes the reverse complement for a DNA tensor.

    Parameters
    ----------
    batch_tensor : torch.Tensor
        Batch of DNA sequences as tensors with shape (batch, length, 4).

    Returns
    -------
    torch.Tensor
        Reverse complement of the input sequences.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    import torch

    seq_rev = torch.flip(batch_tensor, dims=[1])
    rc_seq = torch.flip(seq_rev, dims=[2])
    return rc_seq
