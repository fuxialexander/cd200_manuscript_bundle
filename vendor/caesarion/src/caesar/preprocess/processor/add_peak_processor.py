"""Add peak processor for importing external peak files into zarr stores.

This processor reads narrowPeak/BED format peak files and adds them to existing
zarr stores (both sparse MSZ and dense CDZ formats). It supports:
- Batch import from directories with pattern matching
- Mapping peaks to samples/cell types based on filename patterns
- Merging overlapping peaks across samples
- Calculating peak statistics (aTPM normalization)
- Storing additional columns from peak files
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from caesar.io import read_peaks
from caesar.io.zarr_io import CelltypeDenseZarrIO, MultiSparseZarrIO, open_zarr
from caesar.preprocess.core.base import BaseProcessor
from caesar.preprocess.core.exceptions import ProcessingError, ValidationError

if TYPE_CHECKING:
    from caesar.preprocess.core.config import CaesarConfig


class AddPeakProcessor(BaseProcessor):
    """Process and add external peak files to existing zarr stores.

    This processor handles importing peak files (narrowPeak/BED format) into
    existing zarr stores. It can work with both MultiSparseZarr (MSZ) and
    CelltypeDenseZarr (CDZ) formats.
    """

    def __init__(self, config: CaesarConfig):
        """Initialize add peak processor.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        # Add peak specific configuration
        self.peak_config = config.add_peak
        self.peak_files = self._get_peak_files()
        self.mapping_pattern = self.peak_config.mapping_pattern
        self.merge_strategy = self.peak_config.merge_strategy
        self.min_peak_count = self.peak_config.min_peak_count
        self.normalize = self.peak_config.normalize
        self.peak_name = self.peak_config.peak_name
        self.columns_to_include = self.peak_config.columns_to_include

        # Set input zarr from config.data.input
        if config.data.input:
            self.input_zarr = Path(config.data.input)
        else:
            self.input_zarr = None

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate add peak configuration."""
        # Check input zarr exists
        if not self.input_zarr:
            raise ValidationError("No input zarr specified")
        if not self.input_zarr.exists():
            raise ValidationError(f"Input zarr not found: {self.input_zarr}")

        # Check peak files
        if not self.peak_files:
            raise ValidationError("No peak files specified")

        # Validate merge strategy
        if self.merge_strategy not in ["union", "intersection"]:
            raise ValidationError(
                f"Invalid merge strategy: {self.merge_strategy}. Must be 'union' or 'intersection'"
            )

        # Validate mapping pattern has placeholders
        if self.mapping_pattern and "{" not in self.mapping_pattern:
            raise ValidationError(
                "Mapping pattern must contain placeholders like {celltype} or {sample}"
            )

    def _get_peak_files(self) -> list[Path]:
        """Get peak files from configuration."""
        files = []
        if self.config.add_peak.peak_files:
            for pattern in self.config.add_peak.peak_files:
                # Handle glob patterns
                if "*" in pattern:
                    matched_files = list(Path(pattern).parent.glob(Path(pattern).name))
                    files.extend(matched_files)
                else:
                    files.append(Path(pattern))

        # Filter to only existing files
        files = [f for f in files if f.exists()]

        if not files:
            self.logger.warning("No peak files found matching the specified patterns")

        return files

    def _parse_mapping_pattern(self, filename: str) -> dict[str, str]:
        """Parse filename using mapping pattern to extract metadata.

        Args:
            filename: Name of the peak file (without path)

        Returns:
            Dict with extracted metadata (e.g., {'celltype': 'CD4'})
        """
        if not self.mapping_pattern:
            # Default: use filename without extension as identifier
            return {"id": Path(filename).stem}

        # Convert pattern placeholders to regex groups
        # e.g., "{celltype}_peaks.narrowPeak" -> "(?P<celltype>.+)_peaks\.narrowPeak"
        pattern = self.mapping_pattern
        pattern = pattern.replace(".", r"\.")  # Escape dots
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>.+)", pattern)

        match = re.match(pattern, filename)
        if not match:
            self.logger.warning(
                f"Filename '{filename}' doesn't match pattern '{self.mapping_pattern}'"
            )
            return {"id": Path(filename).stem}

        return match.groupdict()

    def _read_peak_file(self, file_path: Path) -> pd.DataFrame:
        """Read a peak file and extract relevant columns.

        Args:
            file_path: Path to peak file

        Returns:
            DataFrame with peak data
        """
        # Read full peak file data based on format
        if file_path.suffix == ".narrowPeak":
            # Read all narrowPeak columns
            peaks_df = pd.read_csv(
                file_path,
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
        elif file_path.suffix == ".bed":
            # Read BED file - number of columns varies
            df = pd.read_csv(file_path, sep="\t", header=None)
            n_cols = df.shape[1]

            # Standard BED column names
            col_names = ["Chromosome", "Start", "End"]
            if n_cols >= 4:
                col_names.append("Name")
            if n_cols >= 5:
                col_names.append("Score")
            if n_cols >= 6:
                col_names.append("Strand")
            # Add generic names for any additional columns
            for i in range(6, n_cols):
                col_names.append(f"Col{i}")

            df.columns = col_names[:n_cols]
            peaks_df = df
        else:
            # Fallback to basic read_peaks function
            peaks_df = read_peaks(str(file_path))

        return peaks_df

    def _merge_peaks(self, peak_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge peaks from multiple samples based on merge strategy.

        Args:
            peak_dict: Dict mapping sample IDs to peak DataFrames

        Returns:
            Merged peak DataFrame
        """
        if not peak_dict:
            raise ProcessingError("No peaks to merge")

        # Combine all peaks
        all_peaks = []
        for sample_id, peaks_df in peak_dict.items():
            peaks_copy = peaks_df[["Chromosome", "Start", "End"]].copy()
            peaks_copy["sample"] = sample_id
            all_peaks.append(peaks_copy)

        combined_df = pd.concat(all_peaks, ignore_index=True)

        if self.merge_strategy == "union":
            # Merge overlapping peaks across all samples
            merged_peaks = self._merge_overlapping_peaks(combined_df)
        else:  # intersection
            # Keep only peaks present in all samples
            merged_peaks = self._find_intersection_peaks(combined_df, peak_dict)

        return merged_peaks

    def _merge_overlapping_peaks(self, peaks_df: pd.DataFrame) -> pd.DataFrame:
        """Merge overlapping peaks using union strategy.

        Args:
            peaks_df: DataFrame with all peaks from all samples

        Returns:
            DataFrame with merged non-overlapping peaks
        """
        merged_peaks = []

        for chrom in peaks_df["Chromosome"].unique():
            chrom_peaks = peaks_df[peaks_df["Chromosome"] == chrom].copy()
            chrom_peaks = chrom_peaks.sort_values("Start")

            if len(chrom_peaks) == 0:
                continue

            # Merge overlapping intervals
            intervals = chrom_peaks[["Start", "End"]].values
            merged_intervals = []

            current_start, current_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= current_end:
                    # Overlapping - extend current interval
                    current_end = max(current_end, end)
                else:
                    # Non-overlapping - save current and start new
                    merged_intervals.append([current_start, current_end])
                    current_start, current_end = start, end

            # Add last interval
            merged_intervals.append([current_start, current_end])

            # Create DataFrame for this chromosome
            for start, end in merged_intervals:
                merged_peaks.append({"Chromosome": chrom, "Start": start, "End": end})

        return pd.DataFrame(merged_peaks)

    def _find_intersection_peaks(
        self, combined_df: pd.DataFrame, peak_dict: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Find peaks present in all samples (intersection).

        Args:
            combined_df: DataFrame with all peaks from all samples
            peak_dict: Original dict mapping sample IDs to peak DataFrames

        Returns:
            DataFrame with intersection peaks
        """
        n_samples = len(peak_dict)
        intersection_peaks = []

        for chrom in combined_df["Chromosome"].unique():
            chrom_peaks = combined_df[combined_df["Chromosome"] == chrom]

            # Count occurrences of each unique peak
            peak_counts = (
                chrom_peaks.groupby(["Start", "End"]).size().reset_index(name="count")
            )

            # Keep only peaks present in all samples
            common_peaks = peak_counts[peak_counts["count"] == n_samples]

            for _, row in common_peaks.iterrows():
                intersection_peaks.append(
                    {"Chromosome": chrom, "Start": row["Start"], "End": row["End"]}
                )

        return pd.DataFrame(intersection_peaks)

    def _calculate_atpm(self, peaks_df: pd.DataFrame, counts: np.ndarray) -> np.ndarray:
        """Calculate aTPM (log10 TPM) normalization for peak counts.

        Args:
            peaks_df: DataFrame with peak coordinates
            counts: Array of peak counts

        Returns:
            Array of aTPM values
        """
        # Filter to peaks with counts > 0
        mask = counts > 0
        if not mask.any():
            return np.zeros_like(counts, dtype=float)

        # Calculate aTPM: log10(count / total_count * 1e5 + 1)
        total_count = counts[mask].sum()
        atpm = np.zeros_like(counts, dtype=float)
        atpm[mask] = np.log10(counts[mask] / total_count * 1e5 + 1)

        return atpm

    def _process_sparse_zarr(self, zarr_io: MultiSparseZarrIO) -> dict[str, Any]:
        """Process peaks for MultiSparseZarr format.

        Args:
            zarr_io: MultiSparseZarr IO object

        Returns:
            Processing results
        """
        results = {"n_samples": 0, "n_peaks_added": {}}

        # Group peak files by sample/celltype
        peak_dict = {}
        for peak_file in self.peak_files:
            metadata = self._parse_mapping_pattern(peak_file.name)
            sample_id = metadata.get("celltype", metadata.get("sample", metadata["id"]))

            # Read peak file
            peaks_df = self._read_peak_file(peak_file)
            peak_dict[sample_id] = peaks_df

            self.logger.info(
                f"Read {len(peaks_df)} peaks from {peak_file.name} for sample {sample_id}"
            )

        # Write peaks to zarr
        if self.merge_strategy == "union":
            # Write merged peaks for all samples
            merged_peaks = self._merge_peaks(peak_dict)
            self.logger.info(f"Merged to {len(merged_peaks)} unique peaks")

            for zarr_key in zarr_io.zarr_dict:
                zarr_io.zarr_dict[zarr_key].write_peaks(
                    merged_peaks, name=self.peak_name, overwrite=True
                )
                results["n_peaks_added"][zarr_key] = len(merged_peaks)
        else:
            # Write sample-specific peaks
            zarr_io.write_peaks(peak_dict, name=self.peak_name)
            for sample_id, peaks_df in peak_dict.items():
                results["n_peaks_added"][sample_id] = len(peaks_df)

        results["n_samples"] = len(peak_dict)
        return results

    def _process_dense_zarr(self, zarr_io: CelltypeDenseZarrIO) -> dict[str, Any]:
        """Process peaks for CelltypeDenseZarr format.

        Args:
            zarr_io: CelltypeDenseZarr IO object

        Returns:
            Processing results
        """
        results = {"n_celltypes": len(zarr_io.ids), "n_peaks_added": {}}

        # Read all peak files
        peak_dict = {}
        for peak_file in self.peak_files:
            metadata = self._parse_mapping_pattern(peak_file.name)
            celltype = metadata.get("celltype", metadata.get("id"))

            # Check if celltype exists in zarr
            if celltype not in zarr_io.ids:
                self.logger.warning(
                    f"Celltype '{celltype}' from {peak_file.name} not found in zarr"
                )
                continue

            peaks_df = self._read_peak_file(peak_file)
            peak_dict[celltype] = peaks_df

        if not peak_dict:
            raise ProcessingError("No matching celltypes found in zarr")

        # Process based on merge strategy
        if self.merge_strategy == "union":
            # Merge peaks across all celltypes
            merged_peaks = self._merge_peaks(peak_dict)
            self.logger.info(f"Merged to {len(merged_peaks)} unique peaks")

            # Write merged peaks to all celltypes
            for celltype in zarr_io.ids:
                zarr_io.write_peaks(
                    id=celltype,
                    peaks=merged_peaks,
                    name=self.peak_name,
                    columns_to_include=self.columns_to_include,
                    overwrite=True,
                )
                results["n_peaks_added"][celltype] = len(merged_peaks)

            # If normalize is enabled, calculate peak statistics
            if self.normalize:
                self._calculate_peak_statistics(zarr_io, self.peak_name)

        else:
            # Write celltype-specific peaks
            for celltype, peaks_df in peak_dict.items():
                # Determine columns to write
                available_cols = [
                    col for col in self.columns_to_include if col in peaks_df.columns
                ]
                if not available_cols:
                    available_cols = ["Start", "End"]

                zarr_io.write_peaks(
                    id=celltype,
                    peaks=peaks_df,
                    name=self.peak_name,
                    columns_to_include=available_cols,
                    overwrite=True,
                )
                results["n_peaks_added"][celltype] = len(peaks_df)

        return results

    def _calculate_peak_statistics(
        self, zarr_io: CelltypeDenseZarrIO, peak_name: str
    ) -> None:
        """Calculate peak statistics including counts and aTPM.

        Args:
            zarr_io: CelltypeDenseZarr IO object
            peak_name: Name of peaks to calculate statistics for
        """
        self.logger.info("Calculating peak statistics with aTPM normalization")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(
                    self._process_celltype_peaks, zarr_io, celltype, peak_name
                )
                for celltype in zarr_io.ids
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing celltypes"
            ):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error processing celltype: {e}")

    def _process_celltype_peaks(
        self, zarr_io: CelltypeDenseZarrIO, celltype: str, peak_name: str
    ) -> None:
        """Process peaks for a single celltype to add statistics.

        Args:
            zarr_io: CelltypeDenseZarr IO object
            celltype: Celltype ID
            peak_name: Name of peaks to process
        """
        # Get peak counts
        peaks_with_counts = zarr_io.get_peak_counts(celltype, peak_name)
        peaks_with_counts["Count"] = peaks_with_counts["Count"].astype("int64")

        # Filter by minimum count
        filtered_peaks = peaks_with_counts.query(f"Count >= {self.min_peak_count}")

        if len(filtered_peaks) > 0:
            # Calculate aTPM
            filtered_peaks["aTPM"] = self._calculate_atpm(
                filtered_peaks, filtered_peaks["Count"].values
            )

            # Determine columns to include
            cols_to_include = ["Start", "End", "Count", "aTPM"]
            # Add any additional columns from original that are numeric
            for col in self.columns_to_include:
                if col in filtered_peaks.columns and col not in cols_to_include:
                    cols_to_include.append(col)

            # Write filtered peaks with statistics
            zarr_io.write_peaks(
                celltype,
                filtered_peaks,
                f"{peak_name}_filtered",
                overwrite=True,
                columns_to_include=cols_to_include,
            )

    def process(self) -> dict[str, Any]:
        """Process peak files and add to zarr store.

        Returns:
            Processing results
        """
        self._start_processing()

        try:
            # Open zarr store
            zarr_io = open_zarr(str(self.input_zarr), mode="a")

            # Determine zarr type and process accordingly
            if isinstance(zarr_io, MultiSparseZarrIO):
                self.logger.info("Processing peaks for MultiSparseZarr format")
                results = self._process_sparse_zarr(zarr_io)
            elif isinstance(zarr_io, CelltypeDenseZarrIO):
                self.logger.info("Processing peaks for CelltypeDenseZarr format")
                results = self._process_dense_zarr(zarr_io)
            else:
                raise ProcessingError(
                    f"Unsupported zarr type: {type(zarr_io).__name__}. "
                    "Only MultiSparseZarr and CelltypeDenseZarr are supported."
                )

            # Add common metadata
            results.update(
                {
                    "peak_name": self.peak_name,
                    "merge_strategy": self.merge_strategy,
                    "n_peak_files": len(self.peak_files),
                    "columns_included": self.columns_to_include,
                }
            )

            runtime = self._end_processing()
            results["runtime"] = runtime

            return results

        except Exception as e:
            self.logger.error(f"Error processing peaks: {e}")
            raise ProcessingError(f"Failed to process peaks: {e}")
