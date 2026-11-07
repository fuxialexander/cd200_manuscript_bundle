"""Non-redundant motif processor for converting motif scan results to dense zarr arrays.

This processor reads Jeff Vierstra's non-redundant motif scanning results (BED.gz format)
and creates dense arrays of motif scores across the genome. It supports:
- Tabix-indexed BED.gz files for efficient region queries
- Memory-efficient chunked processing
- Stranded motif storage
- Parallel processing across chromosomes
"""

from __future__ import annotations

import gc
import io
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec
from tqdm import tqdm

from caesar.preprocess.core.base import BaseProcessor
from caesar.preprocess.core.exceptions import ProcessingError, ValidationError
from gcell.dna.genome import ChromSize

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig


class NonRedundantMotifProcessor(BaseProcessor):
    """Process Jeff Vierstra's non-redundant motif scan output to dense zarr arrays.

    This processor creates dense arrays where each column represents a motif cluster
    and values represent motif scores at each genomic position.
    """

    @classmethod
    def get_command_name(cls) -> str:
        """Override to return 'motif' instead of 'non-redundant-motif'."""
        return "motif"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add motif processor-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Motif-specific arguments
        parser.add_argument(
            "--motif-scan-results", required=True,
            help="Path to tabix-indexed BED.gz file with motif scan results"
        )
        parser.add_argument(
            "--motif-database", default="vierstra_v2.0",
            help="Version/name of non-redundant motif set (default: vierstra_v2.0)"
        )
        parser.add_argument(
            "--unstranded", action="store_true",
            help="Combine both strands (default: store strands separately)"
        )
        parser.add_argument(
            "--score-threshold", type=float, default=5.0,
            help="Minimum motif score to include (default: 5.0)"
        )
        parser.add_argument(
            "--chunk-size", type=int, default=10_000_000,
            help="Chunk size for processing (default: 10000000)"
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help."""
        return ("Process non-redundant motif scan results to dense zarr arrays. "
                "Reads tabix-indexed BED.gz files with motif scores and creates dense arrays across genome.")

    def __init__(self, config: CaesarConfig):
        """Initialize non-redundant motif processor.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        # Motif-specific configuration
        self.motif_config = config.non_redundant_motif
        self.motif_scan_results = Path(self.motif_config.motif_scan_results)
        self.motif_database = self.motif_config.motif_database
        self.stranded = self.motif_config.stranded
        self.score_threshold = self.motif_config.score_threshold
        self.chunk_size = self.motif_config.chunk_size

        # Output path
        self.output_path = self.get_output_path(f"nr_motifs_{self.motif_database}.zarr")

        # Validate configuration
        self.validate_config()

        # Initialize chromsize
        self.chrom_size = ChromSize(self.assembly)

        # Will be populated during processing
        self.cluster_to_idx = {}
        self.n_motifs = 0

    def validate_config(self) -> None:
        """Validate motif processor configuration."""
        # Check motif scan file exists and is indexed
        if not self.motif_scan_results.exists():
            raise ValidationError(
                f"Motif scan results not found: {self.motif_scan_results}"
            )

        # Check for tabix index
        index_file = Path(str(self.motif_scan_results) + ".tbi")
        if not index_file.exists():
            raise ValidationError(
                f"Tabix index not found for {self.motif_scan_results}. "
                "Please run: tabix -p bed {self.motif_scan_results}"
            )

        # Validate chunk size
        if self.chunk_size < 10000:
            raise ValidationError("Chunk size must be at least 10000")

        # Validate score threshold
        if self.score_threshold < 0:
            raise ValidationError("Score threshold must be non-negative")

    def process(self) -> dict[str, Any]:
        """Process motif scan results to dense zarr arrays.

        Returns:
            Processing results including output path and statistics
        """
        self._start_processing()

        try:
            # Initialize motif clusters by sampling the file
            self._initialize_motif_clusters()

            # Create output zarr
            self._create_output_zarr()

            # Process each chromosome
            self._process_all_chromosomes()

            # Write metadata
            self._write_metadata()

            runtime = self._end_processing()

            return {
                "output_path": str(self.output_path),
                "n_motifs": self.n_motifs,
                "n_chromosomes": len(self.chrom_size.chrom_sizes),
                "stranded": self.stranded,
                "database": self.motif_database,
                "runtime": runtime,
            }

        except Exception as e:
            self.logger.error(f"Error processing motif data: {e}")
            raise ProcessingError(f"Motif processing failed: {e}") from e

    def _initialize_motif_clusters(self) -> None:
        """Initialize motif cluster mapping by sampling the file."""
        self.logger.info("Initializing motif clusters...")

        # Sample first 500kb of chr1 to get motif clusters
        initial_df = self._get_motif_df("chr1", 1, 500000)

        if initial_df.empty:
            raise ProcessingError("No motifs found in initial sampling")

        # Get unique clusters and create mapping
        motif_clusters = sorted(initial_df["cluster"].unique())
        self.cluster_to_idx = {cluster: i for i, cluster in enumerate(motif_clusters)}
        self.n_motifs = len(motif_clusters)

        self.logger.info(f"Found {self.n_motifs} unique motif clusters")

    def _get_motif_df(self, chrom: str, start: int, end: int) -> pd.DataFrame:
        """Call tabix to get motif data for a given region.

        Args:
            chrom: Chromosome name
            start: Start position (1-based)
            end: End position (1-based)

        Returns:
            DataFrame with motif data
        """
        cmd = f"tabix {self.motif_scan_results} {chrom}:{start}-{end}"

        try:
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if stderr and process.returncode != 0:
                raise ProcessingError(f"Tabix error: {stderr.decode('utf-8')}")

            # Jeff Vierstra's format: chrom, start, end, cluster, score, strand, seed_motif, n_motifs
            motif_df = pd.read_csv(
                io.StringIO(stdout.decode("utf-8")),
                sep="\t",
                header=None,
                names=[
                    "chrom",
                    "start",
                    "end",
                    "cluster",
                    "score",
                    "strand",
                    "seed_motif",
                    "n_motifs",
                ],
            )

            # Filter by score threshold if specified
            if self.score_threshold > 0:
                motif_df = motif_df[motif_df["score"] >= self.score_threshold]

            return motif_df

        except Exception as e:
            self.logger.warning(f"Error querying region {chrom}:{start}-{end}: {e}")
            return pd.DataFrame()

    def _create_output_zarr(self) -> None:
        """Create the output zarr structure."""
        self.logger.info(f"Creating output zarr at {self.output_path}")

        # Create zarr group (use zarr_format=2 for backward compatibility)
        self.z = zarr.open_group(str(self.output_path), mode="w", zarr_format=2)
        self.z.create_group("chrs")

    def _process_all_chromosomes(self) -> None:
        """Process all chromosomes in parallel."""
        chromosomes = sorted(self.chrom_size.chrom_sizes.keys())

        # Process largest chromosomes first for better load balancing
        chromosomes.sort(key=lambda x: self.chrom_size.chrom_sizes[x], reverse=True)

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(self._process_chromosome, chrom): chrom
                for chrom in chromosomes
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing chromosomes"
            ):
                chrom = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error processing {chrom}: {e}")
                    # Clean up failed chromosome
                    if chrom in self.z["chrs"]:
                        del self.z["chrs"][chrom]

    def _process_chromosome(self, chrom: str) -> None:
        """Process a single chromosome.

        Args:
            chrom: Chromosome to process
        """
        chrom_length = self.chrom_size.chrom_sizes[chrom]

        # Determine number of columns based on stranded option
        n_cols = self.n_motifs * 2 if self.stranded else self.n_motifs

        # Create chromosome dataset with chunking
        compressor = BloscCodec(cname="zstd", clevel=4)
        zc = self.z["chrs"].create_array(
            chrom,
            shape=(chrom_length, n_cols),
            chunks=(min(100000, chrom_length), n_cols),
            dtype="float32",
            compressors=compressor,
            overwrite=True,
        )

        # Process in chunks
        chunk_starts = list(range(0, chrom_length, self.chunk_size))

        for chunk_start in tqdm(chunk_starts, desc=f"Processing {chrom}", leave=False):
            chunk_end = min(chunk_start + self.chunk_size, chrom_length)
            self._process_chunk(chrom, chunk_start, chunk_end, zc)
            gc.collect()

    def _process_chunk(
        self, chrom: str, chunk_start: int, chunk_end: int, zc: zarr.Array
    ) -> None:
        """Process a chunk of a chromosome.

        Args:
            chrom: Chromosome name
            chunk_start: 0-based start position
            chunk_end: 0-based end position (exclusive)
            zc: Zarr array for the chromosome
        """
        # Get motifs in this region (convert to 1-based for tabix)
        motif_df = self._get_motif_df(chrom, chunk_start + 1, chunk_end)

        # Calculate actual chunk size
        actual_chunk_size = chunk_end - chunk_start

        # Initialize chunk array
        n_cols = self.n_motifs * 2 if self.stranded else self.n_motifs
        chunk_arr = np.zeros((actual_chunk_size, n_cols), dtype="float32")

        # Filter to chunk boundaries
        chunk = motif_df[
            (motif_df["start"] >= chunk_start) & (motif_df["end"] <= chunk_end)
        ]

        # Fill array with motif scores
        for _, row in chunk.iterrows():
            start_offset = max(0, row["start"] - chunk_start)
            end_offset = min(actual_chunk_size, row["end"] - chunk_start)

            if row["cluster"] not in self.cluster_to_idx:
                continue

            if self.stranded:
                # For stranded data: + strand in even columns, - strand in odd columns
                col_idx = self.cluster_to_idx[row["cluster"]] * 2
                if row["strand"] == "-":
                    col_idx += 1
                chunk_arr[start_offset:end_offset, col_idx] = row["score"]
            else:
                # For unstranded data: combine both strands
                col_idx = self.cluster_to_idx[row["cluster"]]
                chunk_arr[start_offset:end_offset, col_idx] = max(
                    chunk_arr[start_offset:end_offset, col_idx], row["score"]
                )

        # Write chunk to zarr
        zc[chunk_start:chunk_end, :] = chunk_arr

        # Clean up
        del motif_df, chunk, chunk_arr

    def _write_metadata(self) -> None:
        """Write metadata to the zarr store."""
        self.logger.info("Writing metadata...")

        # Write global attributes
        self.z.attrs["assembly"] = self.assembly
        self.z.attrs["motif_database"] = self.motif_database
        self.z.attrs["stranded"] = self.stranded
        self.z.attrs["n_motifs"] = self.n_motifs
        self.z.attrs["score_threshold"] = self.score_threshold
        self.z.attrs["source_file"] = str(self.motif_scan_results)

        # Write motif cluster mapping
        clusters = sorted(
            self.cluster_to_idx.keys(), key=lambda x: self.cluster_to_idx[x]
        )
        self.z.attrs["motif_clusters"] = clusters

        # If stranded, indicate column layout
        if self.stranded:
            self.z.attrs["column_layout"] = "alternating_strands"
            self.z.attrs["column_description"] = (
                "Even columns (0, 2, 4, ...) contain + strand scores, "
                "odd columns (1, 3, 5, ...) contain - strand scores"
            )

        self.logger.info(f"Metadata written to {self.output_path}")
