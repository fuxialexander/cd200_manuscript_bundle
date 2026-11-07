"""Peak calling processor with summit extraction, background generation, and classification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyranges as pr

from caesar.io.zarr_io import CelltypeDenseZarrIO
from caesar.preprocess.core.base import AnnotationProcessor
from caesar.preprocess.core.exceptions import ValidationError

if TYPE_CHECKING:
    import argparse
    from caesar.preprocess.core.config import CaesarConfig

try:
    from gcell.dna.genome import Genome
except ImportError:
    Genome = None


from caesar.preprocess.utils.peak_utils import sanitize_column_name


class PeakCallingProcessor(AnnotationProcessor):
    """Peak calling processor with summit extraction, background generation, and classification.

    This processor calls peaks on pseudobulk data and optionally:
    - Extracts high-quality summits
    - Generates background regions
    - Classifies peaks (for_count, for_profile, background)
    """

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add peak-calling-specific CLI arguments."""
        # Call parent to add common args
        super().add_cli_args(parser)

        # Input/Output
        parser.add_argument(
            "-i", "--input", required=False,
            help="Input celltype dense Zarr store"
        )

        # Peak calling parameters
        parser.add_argument(
            "--qvalue", type=float,
            help="Q-value threshold for peak calling (default: 0.05)"
        )
        parser.add_argument(
            "--method", choices=["caesar"],
            help="Peak calling method (default: caesar)"
        )
        parser.add_argument(
            "--overwrite", action="store_true",
            help="Overwrite existing peak calls"
        )

        # Summit extraction options
        parser.add_argument(
            "--no-extract-summits", action="store_true",
            help="Disable summit extraction"
        )
        parser.add_argument(
            "--summit-threshold", type=float,
            help="Minimum value threshold for summits (default: 0)"
        )

        # Background generation options
        parser.add_argument(
            "--generate-background", action="store_true",
            help="Generate background regions"
        )
        parser.add_argument(
            "--background-reference-celltype",
            help="Reference celltype for background scoring"
        )

        # Peak classification options
        parser.add_argument(
            "--classify-peaks", action="store_true",
            help="Classify peaks into categories"
        )

    def __init__(self, config: CaesarConfig):
        """Initialize PeakCallingProcessor.

        Args:
            config: Structured CaesarConfig instance
        """
        super().__init__(config)

        # Direct access to configuration
        self.method = config.peaks.method
        self.qvalue = config.peaks.qvalue
        self.normalize_factor = getattr(config.peaks, "normalize_factor", None)  # Default to None (no normalizer)

        # Summit extraction options
        self.extract_summits = config.peaks.extract_summits
        self.summit_threshold = config.peaks.summit_threshold
        self.summit_return_maxval = config.peaks.summit_return_maxval
        self.summit_return_sum = config.peaks.summit_return_sum
        self.summit_recenter = config.peaks.summit_recenter

        # Background generation options
        self.generate_background = config.peaks.generate_background
        self.background_tile_size = config.peaks.background_tile_size
        self.background_extend_bp = config.peaks.background_extend_bp
        self.background_min_sum_value = config.peaks.background_min_sum_value
        self.background_max_quantile = config.peaks.background_max_quantile
        self.background_n_regions = config.peaks.background_n_regions
        self.background_reference_celltype = config.peaks.background_reference_celltype

        # Peak classification options
        self.classify_peaks = config.peaks.classify_peaks
        self.classification_min_sum_value = config.peaks.classification_min_sum_value
        self.classification_min_max_value = config.peaks.classification_min_max_value

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate peak calling configuration."""
        if self.method not in ["caesar"]:
            raise ValidationError(
                f"Invalid peak calling method: {self.method}. Only 'caesar' is supported"
            )

        if not 0 < self.qvalue <= 1:
            raise ValidationError(f"qvalue must be between 0 and 1, got: {self.qvalue}")

        if self.generate_background and self.background_reference_celltype is None:
            raise ValidationError(
                "background_reference_celltype must be specified when generate_background=True"
            )

    def _detect_store_type(self, input_path: Path) -> str:
        """Detect if store is Zarr or BPCells (dense formats only)."""
        from caesar.io.zarr_io.base import ZarrIO
        from caesar.io.bpcell_io import BPCellsIO

        zarr_type = ZarrIO.detect_store_type(input_path)
        if zarr_type in ["celltype_dense", "dense"]:
            self.logger.info(f"Detected Zarr {zarr_type} at {input_path}")
            return "zarr_dense"

        bpcells_type = BPCellsIO.detect_store_type(input_path)
        if bpcells_type == "bpcells_celltype_dense":
            self.logger.info(f"Detected BPCells celltype_dense at {input_path}")
            return "bpcells_celltype_dense"

        raise ValidationError(
            f"Cannot determine store type for {input_path}: "
            "no recognized dense store structure found. "
            "PeakCallingProcessor only supports celltype_dense stores (Zarr or BPCells)."
        )

    def process(self) -> dict[str, Any]:
        """Call peaks and optionally extract summits, generate background, and classify.

        Returns:
            Processing results
        """
        self._start_processing()

        skipped = self._check_enabled("peaks")
        if skipped:
            return skipped

        store_path = self.validate_input_path(self.get_target_path())
        store_type = self._detect_store_type(store_path)

        if store_type == "zarr_dense":
            from caesar.io.zarr_io import open_zarr
            io = open_zarr(store_path, mode="a")
            if not isinstance(io, CelltypeDenseZarrIO):
                raise ValidationError(
                    "Caesar peak calling requires CelltypeDenseZarrIO input"
                )
        elif store_type == "bpcells_celltype_dense":
            from caesar.io.bpcell_io import CelltypeDenseBPCellsIO
            io = CelltypeDenseBPCellsIO(str(store_path), mode="a")
        else:
            raise ValidationError(f"Unsupported store type: {store_type}")

        self.logger.info(f"Peak calling using assembly: {io.assembly}")

        # Only use normalizer if explicitly requested
        track_kwargs = {}
        if self.normalize_factor == "library_size":
            track_kwargs["normalize_factor"] = io.libsize
        # Otherwise, no normalizer (default behavior)

        # Call peaks
        peaks_dict = io.call_peaks(
            track_id=io.ids, aggregator="by_key", **track_kwargs
        )

        n_peaks_total = 0
        for track_id in io.ids:
            try:
                io.write_peaks(
                    id=track_id,
                    peaks=peaks_dict[track_id],
                    name="cluster_peaks",
                    overwrite=True,
                )
                n_peaks = len(peaks_dict[track_id])
                n_peaks_total += n_peaks
                self.logger.info(f"Called {n_peaks} peaks for {track_id}")
            except Exception as e:
                self.logger.error(f"Error processing peaks for {track_id}: {e}")

        # Merge peaks from all IDs to create aggregated peaks
        if "agg" not in peaks_dict:
            self.logger.info("Merging peaks from all celltypes to create aggregated peaks...")
            peaks_dict["agg"] = self._merge_peaks_union(peaks_dict)
            self.logger.info(f"Merged {len(peaks_dict['agg'])} aggregated peaks from {len(io.ids)} celltypes")

        # Write union peaks to all IDs
        self.logger.info("Writing union peaks to all celltypes...")
        io.write_peaks(
            id=None,  # Write to all celltypes
            peaks=peaks_dict["agg"],
            name="union_peaks",
            overwrite=True,
        )
        self.logger.info(f"Wrote {len(peaks_dict['agg'])} union peaks to all {len(io.ids)} celltypes")

        results = {
            "output_path": str(store_path),
            "method": "caesar",
            "peaks_called": n_peaks_total,
            "groups_processed": len(io.ids),
            "union_peaks": len(peaks_dict["agg"]),
        }

        # Extract summits if requested
        if self.extract_summits:
            self.logger.info("Extracting summits...")
            summits_dict = self._extract_summits(io, peaks_dict["agg"])
            results["summits_extracted"] = {
                ct: len(summits) for ct, summits in summits_dict.items()
            }

            # If background or classification requested, use summits
            peaks_for_downstream = summits_dict
        else:
            peaks_for_downstream = {"agg": peaks_dict["agg"]}

        # Generate background if requested
        if self.generate_background:
            self.logger.info("Generating background regions...")
            background = self._generate_background(
                io, peaks_for_downstream.get("agg", peaks_dict["agg"])
            )
            results["background_regions"] = len(background)
            peaks_for_downstream["background"] = background

        # Classify peaks if requested
        if self.classify_peaks:
            self.logger.info("Classifying peaks...")
            final_peaks = self._classify_peaks(peaks_for_downstream)
            results["classified_peaks"] = len(final_peaks)
            results["peak_types"] = (
                final_peaks["peak_type"].value_counts().to_dict() if "peak_type" in final_peaks.columns else {}
            )

        runtime = self._end_processing()
        results["runtime"] = runtime

        return results

    def _extract_summits(
        self, io: Any, peaks: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Extract high-quality summits for each celltype.

        Args:
            io: CelltypeDenseIO object
            peaks: Peak regions DataFrame

        Returns:
            Dictionary mapping celltype -> summits DataFrame
        """
        summits_dict = {}

        for celltype in io.ids:
            self.logger.info(f"  Extracting summits for {celltype}...")
            celltype_io = io.subset([celltype])
            summits = celltype_io.get_high_quality_summits(
                peaks,
                threshold=self.summit_threshold,
                return_dataframe=True,
                return_maxval=self.summit_return_maxval,
                return_sum=self.summit_return_sum,
                recenter=self.summit_recenter,
            )
            summits_dict[celltype] = summits

        return summits_dict

    def _generate_background(
        self, io: Any, summits_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate background regions.

        Args:
            io: CelltypeDenseIO object
            summits_df: Summits DataFrame

        Returns:
            DataFrame with background regions
        """
        # Load genome if available for blacklist filtering
        genome = None
        if Genome is not None and self.config.assembly:
            try:
                genome = Genome(self.config.assembly)
            except Exception as e:
                self.logger.warning(f"Could not load genome for blacklist: {e}")

        # Generate background tiles
        background_tiles = self._generate_background_tiles(
            summits_df, self.background_tile_size, self.background_extend_bp
        ).query(f"End-Start=={self.background_tile_size}")

        # Remove blacklisted regions
        if (
            genome is not None
            and hasattr(genome, "modeling_blacklist")
            and genome.modeling_blacklist is not None
        ):
            self.logger.info("  Removing blacklisted regions...")
            region_to_avoid = pr.PyRanges(genome.modeling_blacklist)
            background_tiles = (
                pr.PyRanges(background_tiles)
                .overlap(region_to_avoid, invert=True)
                .df
            )
            background_tiles = background_tiles.query(
                f"End-Start=={self.background_tile_size}"
            )

        # Get signal values for background regions
        self.logger.info("  Getting signal values for background regions...")
        background_celltypes = [
            ct for ct in io.ids if self.background_reference_celltype in ct
        ]
        if not background_celltypes:
            background_celltypes = [io.ids[0]]

        background_io = io.subset([background_celltypes[0]])
        background_tiles_count = background_io.get_high_quality_summits(
            background_tiles,
            threshold=0,
            return_dataframe=True,
            return_maxval=True,
            return_sum=True,
            recenter=False,
        )

        # Filter background regions
        background_tiles_count = background_tiles_count.query(
            f"SumValue>{self.background_min_sum_value}"
        ).drop_duplicates()
        background_tiles_count = background_tiles_count[
            background_tiles_count.SumValue
            < background_tiles_count.SumValue.quantile(self.background_max_quantile)
        ].sort_values("SumValue", ascending=False)

        # Remove background regions that overlap with peaks
        self.logger.info("  Removing overlapping background regions...")
        background_pr = pr.PyRanges(background_tiles_count)
        for_count_pr = pr.PyRanges(summits_df)
        background_pr = background_pr.overlap(
            for_count_pr.extend(self.background_extend_bp), invert=True
        )
        background = background_pr.df

        # Sample background regions
        if len(background) > self.background_n_regions:
            background = background.sample(self.background_n_regions)

        self.logger.info(f"  Generated {len(background)} background regions")
        return background

    def _merge_peaks_union(
        self, peaks_dict: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge peaks from multiple celltypes using union strategy.

        Uses PyRanges merge (same approach as sparse.py) to combine overlapping peaks.

        Args:
            peaks_dict: Dictionary mapping celltype IDs to peak DataFrames

        Returns:
            Merged peak DataFrame with overlapping peaks merged
        """
        # Handle edge cases: empty dict, None, or non-dict types
        if peaks_dict is None or not isinstance(peaks_dict, dict) or len(peaks_dict) == 0:
            return pd.DataFrame(columns=["Chromosome", "Start", "End"])

        # Create PyRanges objects from each celltype's peaks
        peaks_pr_list = []
        for celltype_id, peaks_df in peaks_dict.items():
            # Skip non-DataFrame values and empty DataFrames
            if not isinstance(peaks_df, pd.DataFrame):
                continue
            if peaks_df.empty:
                continue
            if not all(col in peaks_df.columns for col in ["Chromosome", "Start", "End"]):
                continue
            peaks_pr = pr.PyRanges(peaks_df[["Chromosome", "Start", "End"]])
            peaks_pr_list.append(peaks_pr)

        if not peaks_pr_list:
            return pd.DataFrame(columns=["Chromosome", "Start", "End"])

        # Concatenate and merge using PyRanges (same pattern as sparse.py)
        merged_pr = pr.concat(peaks_pr_list).merge()
        merged_df = merged_pr.df[["Chromosome", "Start", "End"]]

        return merged_df

    def _generate_background_tiles(
        self, summits_df: pd.DataFrame, tile_size: int, extend_bp: int
    ) -> pd.DataFrame:
        """Generate background tiles avoiding peak regions.

        Args:
            summits_df: DataFrame with columns ['Chromosome', 'Start', 'End']
            tile_size: Size of background tiles
            extend_bp: Extension for overlap checking

        Returns:
            DataFrame with background tiles
        """
        peaks_pr = pr.PyRanges(
            chromosomes=summits_df["Chromosome"],
            starts=summits_df["Start"],
            ends=summits_df["End"],
        )

        extended_peaks = peaks_pr.extend(extend_bp)
        self.logger.info(
            f"Extended {len(peaks_pr)} peaks by ±{extend_bp}bp for overlap checking"
        )

        chrom_info = summits_df.groupby("Chromosome").agg(
            {"Start": "min", "End": "max"}
        ).reset_index()

        all_tiles = []
        for _, chrom_row in chrom_info.iterrows():
            chrom = chrom_row["Chromosome"]
            chrom_start = 0
            chrom_end = chrom_row["End"] + 100000

            for start in range(chrom_start, chrom_end, tile_size):
                end = min(start + tile_size, chrom_end)
                all_tiles.append(
                    {
                        "Chromosome": chrom,
                        "Start": start,
                        "End": end,
                        "Name": f"{chrom}:{start}-{end}",
                    }
                )

        self.logger.info(f"Generated {len(all_tiles)} initial tiles")

        tiles_df = pd.DataFrame(all_tiles)
        tiles_pr = pr.PyRanges(
            chromosomes=tiles_df["Chromosome"],
            starts=tiles_df["Start"],
            ends=tiles_df["End"],
        )

        non_overlapping_tiles = tiles_pr.subtract(extended_peaks)
        self.logger.info(
            f"After removing overlaps: {len(non_overlapping_tiles)} tiles remain"
        )

        background_df = non_overlapping_tiles.df
        background_df["Score"] = background_df["End"] - background_df["Start"]

        return background_df[["Chromosome", "Start", "End", "Score"]]

    def _classify_peaks(
        self, peaks_dict: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Classify peaks and combine with background.

        Args:
            peaks_dict: Dictionary of peak DataFrames (including 'background' if available)

        Returns:
            Combined DataFrame with peak_type column
        """
        # Get final summits (use first celltype's summits or aggregated peaks)
        if "agg" in peaks_dict:
            final_summits = peaks_dict["agg"]
        else:
            final_summits = list(peaks_dict.values())[0]

        # Create final summits with filtering
        final_summits = self._create_final_summits(
            final_summits, self.classification_min_sum_value
        )

        # Combine with background if available
        background = peaks_dict.get("background", pd.DataFrame())
        all_peaks = self._combine_peaks_and_background(
            final_summits, background, self.classification_min_max_value
        )

        return all_peaks

    def _create_final_summits(
        self, combined_summits: pd.DataFrame, min_sum_value: float
    ) -> pd.DataFrame:
        """Create final summits DataFrame with selected columns.

        Args:
            combined_summits: Combined summits DataFrame
            min_sum_value: Minimum SumValue for filtering

        Returns:
            Final summits DataFrame
        """
        # Find Chromosome, Start, End columns
        chrom_cols = [c for c in combined_summits.columns if "Chromosome" in c]
        if not chrom_cols:
            # Assume standard BED format
            if "Chromosome" in combined_summits.columns:
                final_summits = combined_summits.copy()
            else:
                raise ValueError("Could not find Chromosome column")
        else:
            ref_chrom = chrom_cols[0]
            ref_start = ref_chrom.replace("Chromosome", "Start")
            ref_end = ref_chrom.replace("Chromosome", "End")

            columns_to_keep = [ref_chrom, ref_start, ref_end]
            for col in combined_summits.columns:
                if any(
                    x in col
                    for x in ["MaxValue", "SumValue", "logSumValue", "Diff_", "Summit"]
                ):
                    columns_to_keep.append(col)

            final_summits = combined_summits[columns_to_keep].copy()
            final_summits = final_summits.rename(
                columns={
                    ref_chrom: "Chromosome",
                    ref_start: "Start",
                    ref_end: "End",
                }
            )

        # Filter by minimum signal
        sum_value_cols = [
            c for c in final_summits.columns if "SumValue_" in c and "log" not in c
        ]
        if sum_value_cols:
            filter_expr = " | ".join(
                [f"{col} > {min_sum_value}" for col in sum_value_cols]
            )
            before = len(final_summits)
            final_summits = final_summits.query(filter_expr)
            self.logger.info(
                f"  Filtered from {before} to {len(final_summits)} peaks "
                f"(min SumValue > {min_sum_value})"
            )

        return final_summits

    def _combine_peaks_and_background(
        self,
        final_summits: pd.DataFrame,
        background: pd.DataFrame,
        min_max_value: float,
    ) -> pd.DataFrame:
        """Combine peaks and background, add peak_type column.

        Args:
            final_summits: Final summits DataFrame
            background: Background regions DataFrame
            min_max_value: Minimum MaxValue for 'for_profile' classification

        Returns:
            Combined DataFrame with peak_type column
        """
        final_summits = final_summits.copy()
        final_summits["peak_type"] = "for_count"

        if len(background) > 0:
            background = background.copy()
            background["peak_type"] = "background"

            for col in final_summits.columns:
                if col not in background.columns:
                    background[col] = 0

            background = background[final_summits.columns]
            all_peaks = pd.concat(
                [final_summits, background], axis=0, ignore_index=True
            ).copy()
        else:
            all_peaks = final_summits.copy()

        # Recenter peaks using summit position
        summit_cols = [c for c in all_peaks.columns if "Summit" in c]
        if summit_cols:
            summit_col = summit_cols[0]
            has_valid_summit = all_peaks[summit_col].notna() & (
                all_peaks[summit_col] > 0
            )

            summit_pos = all_peaks.loc[has_valid_summit, summit_col]
            all_peaks.loc[has_valid_summit, "Start"] = (summit_pos - 1024).clip(
                lower=0
            )
            all_peaks.loc[has_valid_summit, "End"] = summit_pos + 1024

            needs_center = ~has_valid_summit
            if needs_center.any():
                center = (
                    all_peaks.loc[needs_center, "Start"]
                    + all_peaks.loc[needs_center, "End"]
                ) // 2
                all_peaks.loc[needs_center, "Start"] = (center - 1024).clip(lower=0)
                all_peaks.loc[needs_center, "End"] = center + 1024
        else:
            center = (all_peaks["Start"] + all_peaks["End"]) // 2
            all_peaks["Start"] = (center - 1024).clip(lower=0)
            all_peaks["End"] = center + 1024

        all_peaks["Start"] = all_peaks["Start"].clip(lower=0)

        # Mark high-signal peaks as 'for_profile'
        max_value_cols = [
            c for c in all_peaks.columns if "MaxValue_" in c and "log" not in c
        ]
        if max_value_cols:
            filter_expr = " | ".join(
                [f"{col} >= {min_max_value}" for col in max_value_cols]
            )
            all_peaks.loc[
                all_peaks.eval(filter_expr) & (all_peaks.peak_type == "for_count"),
                "peak_type",
            ] = "for_profile"

        self.logger.info(f"  Final peaks: {len(all_peaks)} total")
        self.logger.info(
            f"    - for_count: {len(all_peaks[all_peaks.peak_type == 'for_count'])}"
        )
        self.logger.info(
            f"    - for_profile: {len(all_peaks[all_peaks.peak_type == 'for_profile'])}"
        )
        if len(background) > 0:
            self.logger.info(
                f"    - background: {len(all_peaks[all_peaks.peak_type == 'background'])}"
            )

        return all_peaks
