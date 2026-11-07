"""Non-redundant motif curator processor for creating ML training datasets.

This processor samples genomic positions based on non-redundant motif scores to create
balanced datasets for machine learning applications. It extracts DNA sequences for
selected regions and stores training data in zarr format.
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec
from tqdm import tqdm

from caesar.io.zarr_io import SequenceDenseZarrIO
from caesar.preprocess.core.base import BaseProcessor
from caesar.preprocess.core.exceptions import ProcessingError, ValidationError
from gcell.dna.genome import ChromSize

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig


class NonRedundantMotifCuratorProcessor(BaseProcessor):
    """Create curated datasets from non-redundant motif scans for ML training.

    This processor:
    1. Samples genomic positions based on motif scores
    2. Creates balanced datasets (motif vs background)
    3. Extracts DNA sequences for selected regions
    4. Stores training data in zarr format
    """

    @classmethod
    def get_command_name(cls) -> str:
        """Override to return 'motif-curator' instead of long name."""
        return "motif-curator"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add motif curator-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Curator-specific arguments
        parser.add_argument(
            "--motif-zarr", required=True,
            help="Path to motif scores zarr from NonRedundantMotifProcessor"
        )
        parser.add_argument(
            "--genome", required=True,
            help="Path to genome sequence zarr (SequenceDenseZarrIO format)"
        )
        parser.add_argument(
            "--n-positive", type=int, default=100000,
            help="Number of positive samples to extract (default: 100000)"
        )
        parser.add_argument(
            "--n-negative", type=int, default=100000,
            help="Number of negative samples to extract (default: 100000)"
        )
        parser.add_argument(
            "--window-size", type=int, default=200,
            help="Window size around motif center, must be even (default: 200)"
        )
        parser.add_argument(
            "--sampling-strategy", choices=["balanced", "weighted"], default="balanced",
            help="Sampling strategy: balanced or weighted (default: balanced)"
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return ("Create curated ML datasets from motif scores. "
                "Samples genomic positions based on motif scores and extracts DNA sequences for training data.")

    def __init__(self, config: CaesarConfig):
        """Initialize motif curator processor.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        # Curator-specific configuration
        self.curator_config = config.non_redundant_motif_curation
        self.motif_zarr_path = Path(self.curator_config.motif_zarr)
        self.genome_fasta_path = Path(self.curator_config.genome_fasta)
        self.n_positive_samples = self.curator_config.n_positive_samples
        self.n_negative_samples = self.curator_config.n_negative_samples
        self.window_size = self.curator_config.window_size
        self.sampling_strategy = self.curator_config.sampling_strategy
        self.score_ranges = self.curator_config.score_ranges
        self.samples_per_motif_per_range = (
            self.curator_config.samples_per_motif_per_range
        )

        # Output path
        self.output_path = self.get_output_path("nr_motif_curated.zarr")

        # Validate configuration
        self.validate_config()

        # Initialize chromsize
        self.chrom_size = ChromSize(self.assembly)

        # Will be populated during processing
        self.genome_zarr = None
        self.motif_zarr = None
        self.selected_positions = None

    def validate_config(self) -> None:
        """Validate curator processor configuration."""
        # Check motif zarr exists
        if not self.motif_zarr_path.exists():
            raise ValidationError(f"Motif zarr not found: {self.motif_zarr_path}")

        # Check genome zarr exists
        if not self.genome_fasta_path.exists():
            raise ValidationError(f"Genome zarr not found: {self.genome_fasta_path}")

        # Validate window size
        if self.window_size < 100:
            raise ValidationError("Window size must be at least 100")

        if self.window_size % 2 != 0:
            raise ValidationError("Window size must be even")

        # Validate sample counts
        if self.n_positive_samples < 1:
            raise ValidationError("n_positive_samples must be at least 1")

        if self.n_negative_samples < 1:
            raise ValidationError("n_negative_samples must be at least 1")

        # Validate sampling strategy
        valid_strategies = ["balanced", "weighted"]
        if self.sampling_strategy not in valid_strategies:
            raise ValidationError(
                f"Invalid sampling strategy: {self.sampling_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        # Validate score ranges
        if not self.score_ranges:
            raise ValidationError("Score ranges cannot be empty")

        for i, (lower, upper) in enumerate(self.score_ranges):
            if lower >= upper:
                raise ValidationError(
                    f"Invalid score range at index {i}: lower ({lower}) must be less than upper ({upper})"
                )

    def process(self) -> dict[str, Any]:
        """Process motif data to create curated training dataset.

        Returns:
            Processing results including output path and statistics
        """
        self._start_processing()

        try:
            # Open input zarr stores
            self._open_input_stores()

            # Sample positive positions based on motif scores
            self._sample_positive_positions()

            # Sample negative positions (background)
            self._sample_negative_positions()

            # Create output zarr
            self._create_output_zarr()

            # Process all chromosomes to extract sequences and motif scores
            self._process_all_chromosomes()

            # Write metadata
            self._write_metadata()

            runtime = self._end_processing()

            return {
                "output_path": str(self.output_path),
                "n_positive_samples": len(
                    self.selected_positions[self.selected_positions["is_positive"]]
                ),
                "n_negative_samples": len(
                    self.selected_positions[~self.selected_positions["is_positive"]]
                ),
                "n_chromosomes": len(self.selected_positions["chrom"].unique()),
                "window_size": self.window_size,
                "runtime": runtime,
            }

        except Exception as e:
            self.logger.error(f"Error curating motif data: {e}")
            raise ProcessingError(f"Motif curation failed: {e}") from e

    def _open_input_stores(self) -> None:
        """Open input zarr stores."""
        self.logger.info("Opening input zarr stores...")

        # Open genome sequence zarr
        self.genome_zarr = SequenceDenseZarrIO(str(self.genome_fasta_path), mode="r")

        # Open motif score zarr
        self.motif_zarr = zarr.open_group(str(self.motif_zarr_path), mode="r")

        # Get motif metadata
        self.n_motifs = self.motif_zarr.attrs.get("n_motifs", 0)
        self.stranded = self.motif_zarr.attrs.get("stranded", True)

        if self.n_motifs == 0:
            raise ProcessingError("No motifs found in motif zarr")

        self.logger.info(f"Found {self.n_motifs} motifs (stranded: {self.stranded})")

    def _sample_positions_for_motif(
        self,
        data: np.ndarray,
        chrom: str,
        motif_idx: int,
        lower: float,
        upper: float,
        num_samples: int,
    ) -> pd.DataFrame:
        """Sample positions for a specific motif in a score range.

        Args:
            data: Motif score array
            chrom: Chromosome name
            motif_idx: Motif index
            lower: Lower score threshold
            upper: Upper score threshold
            num_samples: Number of samples to take

        Returns:
            DataFrame with sampled positions
        """
        try:
            # Find positions within score range
            if self.stranded:
                # For stranded data, check both strands
                pos_strand_idx = motif_idx * 2
                neg_strand_idx = motif_idx * 2 + 1

                pos_positions = np.where(
                    (data[:, pos_strand_idx] >= lower)
                    & (data[:, pos_strand_idx] < upper)
                )[0]
                neg_positions = np.where(
                    (data[:, neg_strand_idx] >= lower)
                    & (data[:, neg_strand_idx] < upper)
                )[0]

                # Combine positions from both strands
                all_positions = np.concatenate([pos_positions, neg_positions])
                strands = np.concatenate(
                    [np.ones(len(pos_positions)), -np.ones(len(neg_positions))]
                )
            else:
                # For unstranded data
                all_positions = np.where(
                    (data[:, motif_idx] >= lower) & (data[:, motif_idx] < upper)
                )[0]
                strands = np.zeros(len(all_positions))

            if len(all_positions) == 0:
                return pd.DataFrame(
                    columns=[
                        "pos",
                        "motif_idx",
                        "chrom",
                        "strand",
                        "lower_threshold",
                        "upper_threshold",
                    ]
                )

            # Sample positions
            if len(all_positions) <= num_samples:
                selected_indices = np.arange(len(all_positions))
            else:
                selected_indices = np.random.choice(
                    len(all_positions), num_samples, replace=False
                )

            # Create result DataFrame
            result = pd.DataFrame(
                {
                    "pos": all_positions[selected_indices],
                    "motif_idx": motif_idx,
                    "chrom": chrom,
                    "strand": strands[selected_indices].astype(int),
                    "lower_threshold": lower,
                    "upper_threshold": upper,
                }
            )

            return result

        except Exception:
            return pd.DataFrame(
                columns=[
                    "pos",
                    "motif_idx",
                    "chrom",
                    "strand",
                    "lower_threshold",
                    "upper_threshold",
                ]
            )

    def _process_chunk(self, args: tuple) -> pd.DataFrame:
        """Process a chunk of chromosome data.

        Args:
            args: Tuple of (chrom, start, end, score_ranges)

        Returns:
            DataFrame with sampled positions
        """
        chrom, start, end, score_ranges = args

        # Re-open zarr in subprocess
        motif_zarr = zarr.open_group(str(self.motif_zarr_path), mode="r")
        data = motif_zarr[f"chrs/{chrom}"][start:end]

        results = []
        for lower, upper in score_ranges:
            for motif_idx in range(self.n_motifs):
                result = self._sample_positions_for_motif(
                    data,
                    chrom,
                    motif_idx,
                    lower,
                    upper,
                    self.samples_per_motif_per_range,
                )
                if not result.empty:
                    # Adjust positions to absolute coordinates
                    result["pos"] += start
                    results.append(result)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def _sample_positive_positions(self) -> None:
        """Sample positive positions based on motif scores."""
        self.logger.info("Sampling positive positions from motif scores...")

        all_results = []
        chunk_size = 2_000_000

        # Process each chromosome
        from caesar.preprocess.core.chromosome_filter import ChromosomeFilter

        chr_filter = ChromosomeFilter({})
        for chrom in tqdm(
            self.chrom_size.chrom_sizes.keys(), desc="Sampling chromosomes"
        ):
            if not chr_filter.is_valid_chromosome(chrom):
                continue

            # Skip if chromosome not in motif zarr
            if f"chrs/{chrom}" not in self.motif_zarr:
                continue

            chrom_length = self.chrom_size.chrom_sizes[chrom]

            # Create tasks for parallel processing
            tasks = []
            for start in range(0, chrom_length, chunk_size):
                end = min(start + chunk_size, chrom_length)
                tasks.append((chrom, start, end, self.score_ranges))

            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self._process_chunk, task) for task in tasks]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing {chrom}",
                    leave=False,
                ):
                    result = future.result()
                    if not result.empty:
                        all_results.append(result)

            gc.collect()

        # Combine results from all chromosomes
        if all_results:
            self.selected_positions = pd.concat(all_results, ignore_index=True)
            self.selected_positions["is_positive"] = True

            # Sample down to requested number if needed
            if len(self.selected_positions) > self.n_positive_samples:
                if self.sampling_strategy == "balanced":
                    # Sample equally from each motif
                    sampled = (
                        self.selected_positions.groupby("motif_idx")
                        .apply(
                            lambda x: x.sample(
                                min(len(x), self.n_positive_samples // self.n_motifs),
                                replace=False,
                            )
                        )
                        .reset_index(drop=True)
                    )
                else:  # weighted
                    # Sample proportionally to motif frequency
                    sampled = self.selected_positions.sample(
                        self.n_positive_samples, replace=False
                    ).reset_index(drop=True)

                self.selected_positions = sampled

            self.logger.info(
                f"Sampled {len(self.selected_positions)} positive positions"
            )
        else:
            raise ProcessingError("No positive positions found matching score criteria")

    def _sample_negative_positions(self) -> None:
        """Sample negative positions (background regions)."""
        self.logger.info("Sampling negative positions...")

        negative_positions = []
        half_window = self.window_size // 2

        # For each positive position, sample a nearby negative position
        for _, row in tqdm(
            self.selected_positions.iterrows(),
            total=len(self.selected_positions),
            desc="Sampling negatives",
        ):
            chrom = row["chrom"]
            pos = row["pos"]
            chrom_size = self.chrom_size.chrom_sizes[chrom]

            # Sample a random position within the same chromosome
            # Avoid positions too close to chromosome boundaries
            min_pos = half_window
            max_pos = chrom_size - half_window

            if max_pos <= min_pos:
                continue

            # Sample nearby position (within 10kb) but not overlapping
            offset = np.random.randint(1000, 10000)
            neg_pos = pos - offset if np.random.random() < 0.5 else pos + offset

            # Ensure within bounds
            neg_pos = max(min_pos, min(neg_pos, max_pos))

            negative_positions.append(
                {
                    "pos": neg_pos,
                    "motif_idx": -1,  # Indicates background
                    "chrom": chrom,
                    "strand": 0,
                    "lower_threshold": -1,
                    "upper_threshold": -1,
                    "is_positive": False,
                }
            )

        # Convert to DataFrame and sample to requested number
        negative_df = pd.DataFrame(negative_positions)

        if len(negative_df) > self.n_negative_samples:
            negative_df = negative_df.sample(
                self.n_negative_samples, replace=False
            ).reset_index(drop=True)

        # Combine positive and negative positions
        self.selected_positions = (
            pd.concat([self.selected_positions, negative_df], ignore_index=True)
            .sample(frac=1)
            .reset_index(drop=True)
        )  # Shuffle

        self.logger.info(
            f"Total positions: {len(self.selected_positions)} "
            f"({len(self.selected_positions[self.selected_positions['is_positive']])} positive, "
            f"{len(self.selected_positions[~self.selected_positions['is_positive']])} negative)"
        )

    def _create_output_zarr(self) -> None:
        """Create the output zarr structure."""
        self.logger.info(f"Creating output zarr at {self.output_path}")

        # Create zarr group (use zarr_format=2 for backward compatibility)
        self.z = zarr.open_group(str(self.output_path), mode="w", zarr_format=2)
        self.z.create_group("sequence")
        self.z.create_group("motif")
        self.z.create_group("metadata")

    def _retrieve_track(self, args: tuple) -> tuple[np.ndarray, np.ndarray] | None:
        """Retrieve sequence and motif tracks for a position.

        Args:
            args: Tuple of (chrom, pos, window_size, motif_zarr_path, genome_zarr_path)

        Returns:
            Tuple of (sequence_array, motif_array) or None if failed
        """
        chrom, pos, window_size, motif_zarr_path, genome_zarr_path = args
        half_window = window_size // 2

        try:
            # Re-open zarr stores in subprocess
            genome_zarr = SequenceDenseZarrIO(str(genome_zarr_path), mode="r")
            motif_zarr = zarr.open_group(str(motif_zarr_path), mode="r")

            # Get sequence track
            sequence_track = genome_zarr.get_track(
                chrom, pos - half_window, pos + half_window, output_format="raw_array"
            )

            # Get motif track
            motif_track = motif_zarr[f"chrs/{chrom}"][
                pos - half_window : pos + half_window
            ]

            # Validate tracks
            if sequence_track.sum() > 0 and motif_track.sum() > 0:
                return sequence_track, motif_track
            else:
                return None

        except Exception as e:
            logging.warning(f"Error retrieving track at {chrom}:{pos}: {e}")
            return None

    def _process_all_chromosomes(self) -> None:
        """Process all chromosomes to extract sequences and motif scores."""
        self.logger.info("Extracting sequences and motif scores...")

        # Group positions by chromosome
        for chrom, chrom_positions in self.selected_positions.groupby("chrom"):
            self._process_chromosome(chrom, chrom_positions)

    def _process_chromosome(self, chrom: str, positions: pd.DataFrame) -> None:
        """Process a single chromosome.

        Args:
            chrom: Chromosome name
            positions: DataFrame with positions for this chromosome
        """
        self.logger.info(f"Processing {chrom} with {len(positions)} positions")

        # Prepare tasks for parallel processing
        tasks = [
            (
                row["chrom"],
                int(row["pos"]),
                self.window_size,
                self.motif_zarr_path,
                self.genome_fasta_path,
            )
            for _, row in positions.iterrows()
        ]

        # Process positions in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self._retrieve_track, task) for task in tasks]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Retrieving tracks for {chrom}",
                leave=False,
            ):
                result = future.result()
                if result is not None:
                    results.append(result)

        if not results:
            self.logger.warning(f"No valid tracks retrieved for {chrom}")
            return

        # Determine dimensions
        n_cols = self.n_motifs * 2 if self.stranded else self.n_motifs

        # Stack results into arrays
        sequence_array = np.stack([r[0] for r in results], axis=0)
        motif_array = np.stack([r[1] for r in results], axis=0)

        # Create datasets with compression
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

        seq_data = sequence_array.astype(np.int8)
        seq_arr = self.z["sequence"].create_array(
            chrom,
            shape=seq_data.shape,
            dtype=np.int8,
            chunks=(1, self.window_size, 4),
            compressors=compressor,
            overwrite=True,
        )
        seq_arr[:] = seq_data

        motif_data = motif_array.astype(np.float16)
        motif_arr = self.z["motif"].create_array(
            chrom,
            shape=motif_data.shape,
            dtype=np.float16,
            chunks=(1, self.window_size, n_cols),
            compressors=compressor,
            overwrite=True,
        )
        motif_arr[:] = motif_data

        # Store metadata for positions
        valid_positions = positions.iloc[: len(results)]
        meta_data = valid_positions[["pos", "motif_idx", "strand", "is_positive"]].values.astype(np.int32)
        meta_arr = self.z["metadata"].create_array(
            f"{chrom}_positions",
            shape=meta_data.shape,
            dtype=np.int32,
            overwrite=True,
        )
        meta_arr[:] = meta_data

        self.logger.info(f"Saved {len(results)} tracks for {chrom}")
        gc.collect()

    def _write_metadata(self) -> None:
        """Write metadata to the zarr store."""
        self.logger.info("Writing metadata...")

        # Write global attributes
        self.z.attrs["assembly"] = self.assembly
        self.z.attrs["window_size"] = self.window_size
        self.z.attrs["n_positive_samples"] = len(
            self.selected_positions[self.selected_positions["is_positive"]]
        )
        self.z.attrs["n_negative_samples"] = len(
            self.selected_positions[~self.selected_positions["is_positive"]]
        )
        self.z.attrs["n_motifs"] = self.n_motifs
        self.z.attrs["stranded"] = self.stranded
        self.z.attrs["sampling_strategy"] = self.sampling_strategy
        self.z.attrs["score_ranges"] = self.score_ranges
        self.z.attrs["source_motif_zarr"] = str(self.motif_zarr_path)
        self.z.attrs["source_genome_zarr"] = str(self.genome_fasta_path)

        # Write position summary
        position_summary = self.selected_positions.groupby("chrom").agg(
            {"is_positive": ["sum", "count"]}
        )
        position_summary.columns = ["n_positive", "n_total"]
        position_summary["n_negative"] = (
            position_summary["n_total"] - position_summary["n_positive"]
        )

        self.z.attrs["position_summary"] = position_summary.to_dict()

        self.logger.info(f"Metadata written to {self.output_path}")
