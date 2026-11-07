"""Differential signal calculation processor.

This processor computes differential signals between conditions by comparing
log-normalized peak signals across celltypes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from caesar.preprocess.core.base import BaseProcessor
from caesar.preprocess.core.exceptions import ValidationError

if TYPE_CHECKING:
    from caesar.preprocess.core.config import CaesarConfig

logger = logging.getLogger(__name__)


from caesar.preprocess.utils.peak_utils import sanitize_column_name


class DifferentialSignalProcessor(BaseProcessor):
    """Processor for calculating differential signals between conditions.

    This processor:
    1. Loads a BED file with summits (from multiple celltypes)
    2. Merges summits with column renaming
    3. Calculates log-normalized values
    4. Computes differential columns for specified comparisons
    5. Outputs annotated BED file
    """

    def __init__(self, config: CaesarConfig):
        """Initialize DifferentialSignalProcessor.

        Args:
            config: Structured CaesarConfig instance
        """
        super().__init__(config)

        # Direct access to configuration
        self.diff_config = config.differential_signal
        self.input_bed = Path(self.diff_config.input_bed)
        self.output_bed = Path(self.diff_config.output_bed)
        self.comparisons = self.diff_config.comparisons or []
        self.use_sanitized_names = self.diff_config.use_sanitized_names
        self.normalization_factor = self.diff_config.normalization_factor

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate differential signal configuration."""
        if not self.input_bed.exists():
            raise ValidationError(f"Input BED file not found: {self.input_bed}")

        if not self.comparisons:
            raise ValidationError(
                "No comparisons specified. Provide at least one (condition1, condition2) tuple."
            )

        for comp in self.comparisons:
            if not isinstance(comp, (list, tuple)) or len(comp) != 2:
                raise ValidationError(
                    f"Invalid comparison format: {comp}. Expected [condition1, condition2] list or tuple."
                )

    def process(self) -> dict[str, Any]:
        """Calculate differential signals.

        Returns:
            Processing results dictionary
        """
        self._start_processing()

        logger.info("Loading input BED file...")
        peaks_df = pd.read_csv(self.input_bed, sep="\t")

        # Parse summits from columns (assuming format: SumValue_celltype, etc.)
        summits_dict = self._parse_summits_from_dataframe(peaks_df)

        if not summits_dict:
            raise ValidationError(
                "Could not parse summits from input BED file. "
                "Expected columns like 'SumValue_celltype', 'MaxValue_celltype', etc."
            )

        # Merge summits
        logger.info("Merging summits...")
        merged_summits = self._merge_summits(summits_dict)

        # Calculate log-normalized values
        logger.info("Calculating log-normalized values...")
        merged_summits = self._calculate_log_normalized_values(
            merged_summits, summits_dict
        )

        # Calculate differential columns
        logger.info("Calculating differential columns...")
        merged_summits = self._calculate_differential_columns(
            merged_summits, summits_dict
        )

        # Save output
        logger.info(f"Saving output to {self.output_bed}...")
        self.output_bed.parent.mkdir(parents=True, exist_ok=True)
        merged_summits.to_csv(self.output_bed, sep="\t", index=False)

        runtime = self._end_processing()

        # Count differential columns created
        diff_cols = [c for c in merged_summits.columns if c.startswith("Diff_")]

        return {
            "output_path": str(self.output_bed),
            "input_peaks": len(peaks_df),
            "output_peaks": len(merged_summits),
            "differential_columns": len(diff_cols),
            "comparisons": self.comparisons,
            "runtime": runtime,
        }

    def _parse_summits_from_dataframe(
        self, peaks_df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Parse summits from DataFrame columns.

        Assumes columns are named like: SumValue_celltype, MaxValue_celltype, etc.

        Args:
            peaks_df: Input DataFrame

        Returns:
            Dictionary mapping celltype -> summits DataFrame
        """
        # Find all celltypes by looking at column suffixes
        celltypes = set()
        for col in peaks_df.columns:
            if "_" in col:
                parts = col.split("_")
                if len(parts) >= 2:
                    # Try to identify celltype (everything after first underscore)
                    celltype = "_".join(parts[1:])
                    celltypes.add(celltype)

        if not celltypes:
            return {}

        # Group columns by celltype
        summits_dict = {}
        for celltype in celltypes:
            celltype_cols = [c for c in peaks_df.columns if c.endswith(f"_{celltype}")]
            if celltype_cols:
                # Create a DataFrame for this celltype
                celltype_df = peaks_df[celltype_cols].copy()
                # Remove celltype suffix from column names
                celltype_df.columns = [
                    c.replace(f"_{celltype}", "") for c in celltype_df.columns
                ]
                # Add coordinate columns if available
                if "Chromosome" in peaks_df.columns:
                    celltype_df["Chromosome"] = peaks_df["Chromosome"]
                if "Start" in peaks_df.columns:
                    celltype_df["Start"] = peaks_df["Start"]
                if "End" in peaks_df.columns:
                    celltype_df["End"] = peaks_df["End"]
                summits_dict[celltype] = celltype_df

        return summits_dict

    def _merge_summits(
        self, summits_dict: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge summit DataFrames from all celltypes with column renaming.

        Args:
            summits_dict: Dictionary of celltype -> summits DataFrame

        Returns:
            Combined DataFrame with renamed columns
        """
        renamed_summits = {}
        celltype_to_sanitized = {}

        for celltype, summits in summits_dict.items():
            renamed = summits.copy()
            if self.use_sanitized_names:
                sanitized_ct = sanitize_column_name(celltype)
                celltype_to_sanitized[celltype] = sanitized_ct
                renamed.columns = [
                    f"{col}_{sanitized_ct}" if col not in ["Chromosome", "Start", "End"] else col
                    for col in renamed.columns
                ]
            else:
                renamed.columns = [
                    f"{col}_{celltype}" if col not in ["Chromosome", "Start", "End"] else col
                    for col in renamed.columns
                ]
            renamed_summits[celltype] = renamed

        all_summits = pd.concat(list(renamed_summits.values()), axis=1)
        all_summits.attrs["celltype_to_sanitized"] = celltype_to_sanitized
        return all_summits

    def _calculate_log_normalized_values(
        self, merged_summits: pd.DataFrame, summits_dict: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate log-normalized values for each celltype.

        Args:
            merged_summits: Merged summits DataFrame
            summits_dict: Original summits dictionary

        Returns:
            DataFrame with added log-normalized columns
        """
        df = merged_summits.copy()
        celltype_to_sanitized = df.attrs.get("celltype_to_sanitized", {})

        for celltype in summits_dict.keys():
            if self.use_sanitized_names:
                sanitized_ct = celltype_to_sanitized.get(
                    celltype, sanitize_column_name(celltype)
                )
            else:
                sanitized_ct = celltype

            # Calculate log-normalized SumValue
            sumval_col = f"SumValue_{sanitized_ct}"
            if sumval_col in df.columns:
                total_sum = df[sumval_col].sum()
                df[f"logSumValue_{sanitized_ct}"] = np.log10(
                    df[sumval_col] / total_sum * self.normalization_factor + 1
                )

            # Calculate log-normalized MaxValue
            maxval_col = f"MaxValue_{sanitized_ct}"
            if maxval_col in df.columns:
                total_max = df[maxval_col].sum()
                df[f"logMaxValue_{sanitized_ct}"] = np.log10(
                    df[maxval_col] / total_max * self.normalization_factor + 1
                )

        return df

    def _calculate_differential_columns(
        self, merged_summits: pd.DataFrame, summits_dict: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate differential columns by comparing conditions.

        Args:
            merged_summits: Merged summits DataFrame
            summits_dict: Original summits dictionary for celltype lookup

        Returns:
            DataFrame with added differential columns
        """
        df = merged_summits.copy()
        celltype_to_sanitized = df.attrs.get("celltype_to_sanitized", {})

        for cond1, cond2 in self.comparisons:
            # Find matching celltypes
            cond1_celltypes = self._find_matching_celltypes(
                cond1, list(summits_dict.keys())
            )
            cond2_celltypes = self._find_matching_celltypes(
                cond2, list(summits_dict.keys())
            )

            if not cond1_celltypes or not cond2_celltypes:
                logger.warning(
                    f"Could not find celltypes matching '{cond1}' or '{cond2}'"
                )
                logger.warning(f"  Available celltypes: {list(summits_dict.keys())}")
                continue

            # Use first matching celltype for each condition
            cond1_ct = cond1_celltypes[0]
            cond2_ct = cond2_celltypes[0]
            logger.info(
                f"  Matched '{cond1}' -> '{cond1_ct}', '{cond2}' -> '{cond2_ct}'"
            )

            # Get column names
            if self.use_sanitized_names:
                cond1_sanitized = celltype_to_sanitized.get(
                    cond1_ct, sanitize_column_name(cond1_ct)
                )
                cond2_sanitized = celltype_to_sanitized.get(
                    cond2_ct, sanitize_column_name(cond2_ct)
                )
            else:
                cond1_sanitized = cond1_ct
                cond2_sanitized = cond2_ct

            # Calculate differential using log-normalized SumValue
            sumval_col1 = f"SumValue_{cond1_sanitized}"
            sumval_col2 = f"SumValue_{cond2_sanitized}"

            if sumval_col1 in df.columns and sumval_col2 in df.columns:
                # Get totals for normalization
                total1 = df[sumval_col1].sum()
                total2 = df[sumval_col2].sum()

                # Calculate log-normalized values and difference
                log1 = np.log10(df[sumval_col1] / total1 * self.normalization_factor + 1)
                log2 = np.log10(df[sumval_col2] / total2 * self.normalization_factor + 1)

                # Create differential column name
                if self.use_sanitized_names:
                    cond1_sanitized_for_col = sanitize_column_name(cond1)
                    cond2_sanitized_for_col = sanitize_column_name(cond2)
                    diff_col_name = (
                        f"Diff_{cond1_sanitized_for_col}_vs_{cond2_sanitized_for_col}"
                    )
                else:
                    diff_col_name = f"Diff_{cond1}_vs_{cond2}"

                df[diff_col_name] = log1 - log2
                logger.info(f"  Created {diff_col_name}")
            else:
                # Try using logSumValue if available
                logsum_col1 = f"logSumValue_{cond1_sanitized}"
                logsum_col2 = f"logSumValue_{cond2_sanitized}"
                if logsum_col1 in df.columns and logsum_col2 in df.columns:
                    if self.use_sanitized_names:
                        cond1_sanitized_for_col = sanitize_column_name(cond1)
                        cond2_sanitized_for_col = sanitize_column_name(cond2)
                        diff_col_name = (
                            f"Diff_{cond1_sanitized_for_col}_vs_{cond2_sanitized_for_col}"
                        )
                    else:
                        diff_col_name = f"Diff_{cond1}_vs_{cond2}"
                    df[diff_col_name] = df[logsum_col1] - df[logsum_col2]
                    logger.info(f"  Created {diff_col_name} (using logSumValue)")

        return df

    def _find_matching_celltypes(
        self, condition: str, available_celltypes: list[str]
    ) -> list[str]:
        """Find celltypes matching a condition name using flexible matching.

        Args:
            condition: Condition name to match
            available_celltypes: List of available celltype names

        Returns:
            List of matching celltype names
        """
        def normalize_for_matching(name):
            return name.lower().replace(".", "_").replace("-", "_")

        cond_normalized = normalize_for_matching(condition)
        matches = []

        for ct in available_celltypes:
            ct_normalized = normalize_for_matching(ct)
            if cond_normalized in ct_normalized or ct_normalized in cond_normalized:
                matches.append(ct)

        return matches

