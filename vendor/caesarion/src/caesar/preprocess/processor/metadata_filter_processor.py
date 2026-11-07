"""Metadata filter processor for dataset-specific cell filtering.

This processor applies dataset-specific metadata filtering to standardize
cell barcodes and filter cells by quality metrics. It supports multiple
published dataset formats and custom filtering rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from caesar.preprocess.core.base import BaseProcessor
from caesar.preprocess.core.exceptions import ValidationError

if TYPE_CHECKING:
    from caesar.preprocess.core.config import CaesarConfig


class MetadataFilterProcessor(BaseProcessor):
    """Process and filter cell metadata for various published datasets.

    This processor:
    - Loads and standardizes cell metadata from various dataset formats
    - Applies dataset-specific filtering rules
    - Standardizes barcode formats
    - Filters cells by quality metrics (e.g., fragment depth)
    - Outputs filtered metadata and barcode mappings

    The processor can be used as a preprocessing step by other processors
    to ensure consistent cell filtering across the pipeline.
    """

    # Dataset-specific filter functions
    DATASET_FILTERS = {
        "joung_tf_atlas": "_filter_joung_tfatlas",
        "t_exhaustion": "_filter_t_exhaustion",
        "linnarsson_brain": "_filter_linnarsson_brain",
        "yanhu_bmms": "_filter_yanhu_bmms",
        "shendure_fetal": "_filter_shendure_fetal",
        "archr_cellline": "_filter_archr_cellline",
        "htan": "_filter_htan",
        "vijay_hematopoiesis": "_filter_vijay_hematopoiesis",
    }

    def __init__(self, config: CaesarConfig):
        """Initialize MetadataFilterProcessor.

        Args:
            config: Full CaesarConfig object
        """
        super().__init__(config)

        # Get metadata filter config (or use defaults)
        self.metadata_config = getattr(config, "metadata_filter", None)
        if self.metadata_config is None:
            # Use defaults if no config section exists
            self.dataset = None
            self.min_fragments = 1000
            self.max_fragments = 100000
            self.custom_filters = []
            self.output_path = None
            self.metadata_path = config.data.metadata
        else:
            self.dataset = self.metadata_config.dataset
            self.min_fragments = self.metadata_config.min_fragments
            self.max_fragments = self.metadata_config.max_fragments
            self.custom_filters = self.metadata_config.custom_filters or []
            self.output_path = self.metadata_config.output_path
            self.metadata_path = (
                self.metadata_config.metadata_path or config.data.metadata
            )

        # Will be populated during processing
        self.filtered_metadata = None
        self.barcode_dict = None

    def validate_config(self) -> None:
        """Validate metadata filter configuration."""
        # Check if metadata path exists
        if self.metadata_path:
            metadata_file = Path(self.metadata_path)
            if not metadata_file.exists():
                raise ValidationError(f"Metadata file not found: {metadata_file}")

        # Validate fragment thresholds
        if self.min_fragments < 0:
            raise ValidationError("min_fragments must be non-negative")
        if self.max_fragments <= self.min_fragments:
            raise ValidationError("max_fragments must be greater than min_fragments")

        # Validate dataset name if provided
        if self.dataset and self.dataset not in self.DATASET_FILTERS:
            self.logger.warning(
                f"Unknown dataset '{self.dataset}'. Available datasets: "
                f"{list(self.DATASET_FILTERS.keys())}"
            )

    def process(self) -> dict[str, Any]:
        """Process metadata filtering.

        Returns:
            Dict containing:
            - filtered_metadata: Filtered metadata DataFrame
            - barcode_dict: Mapping of barcodes to indices
            - n_cells_filtered: Number of cells after filtering
            - n_cells_original: Number of cells before filtering
        """
        self._start_processing()

        # Validate configuration
        self.validate_config()

        if not self.metadata_path:
            raise ValidationError("No metadata path provided")

        # Apply dataset-specific filter if specified
        if self.dataset and self.dataset in self.DATASET_FILTERS:
            filter_method = getattr(self, self.DATASET_FILTERS[self.dataset])
            self.filtered_metadata, self.barcode_dict = filter_method(
                self.metadata_path, self.min_fragments
            )
        else:
            # Generic metadata loading
            self.filtered_metadata, self.barcode_dict = self._filter_generic(
                self.metadata_path, self.min_fragments, self.max_fragments
            )

        # Apply custom filters
        if self.custom_filters:
            self.filtered_metadata = self._apply_custom_filters(
                self.filtered_metadata, self.custom_filters
            )
            # Update barcode dict after custom filtering
            barcodes = self._get_barcode_column(self.filtered_metadata).values
            self.barcode_dict = {k: i for i, k in enumerate(barcodes)}

        # Save filtered metadata if output path provided
        if self.output_path:
            output_file = Path(self.output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self.filtered_metadata.to_csv(output_file, index=False)
            self.logger.info(f"Saved filtered metadata to {output_file}")

        # Store in output directory for other processors
        metadata_output = self.output_dir / "filtered_metadata.csv"
        self.filtered_metadata.to_csv(metadata_output, index=False)

        # Save barcode mapping
        barcode_output = self.output_dir / "barcode_mapping.txt"
        with open(barcode_output, "w") as f:
            for barcode, idx in self.barcode_dict.items():
                f.write(f"{barcode}\t{idx}\n")

        results = {
            "filtered_metadata": self.filtered_metadata,
            "barcode_dict": self.barcode_dict,
            "n_cells_filtered": len(self.filtered_metadata),
            "n_cells_original": len(
                self.filtered_metadata
            ),  # Will be updated if we track original
            "metadata_path": str(metadata_output),
            "barcode_mapping_path": str(barcode_output),
        }

        self._end_processing()
        return results

    def _get_barcode_column(self, df: pd.DataFrame) -> pd.Series:
        """Get barcode column from dataframe, trying common names."""
        barcode_columns = [
            "Barcodes",
            "barcode",
            "Barcode",
            "barcodes",
            "cell_id",
            "cell",
        ]
        for col in barcode_columns:
            if col in df.columns:
                return df[col]

        # Check if index contains barcodes
        if df.index.name in barcode_columns:
            return df.index.to_series()

        # Check if index looks like barcodes (string-based check)
        if len(df) > 0 and isinstance(df.index[0], str) and len(df.index[0]) > 10:
            return df.index.to_series()

        raise ValueError("Could not find barcode column in metadata")

    def _apply_custom_filters(
        self, df: pd.DataFrame, filters: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply custom filters to metadata."""
        for filter_spec in filters:
            field = filter_spec.get("field")
            values = filter_spec.get("values")

            if field and values and field in df.columns:
                df = df[df[field].isin(values)]
                self.logger.info(
                    f"Applied filter: {field} in {values}, {len(df)} cells remaining"
                )

        return df

    # Dataset-specific filter methods
    def _filter_generic(
        self, metadata_path: str, min_fragments: int, max_fragments: int
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Generic metadata filtering."""
        # Try to read with different separators
        metadata = None
        for sep in ["\t", ","]:
            try:
                metadata = pd.read_csv(metadata_path, sep=sep)
                # Check if we got actual columns (not just one column with everything)
                if len(metadata.columns) > 1 or sep == ",":
                    break
            except Exception:
                continue

        if metadata is None:
            raise ValueError(f"Could not read metadata file: {metadata_path}")

        # Apply fragment depth filtering if depth column exists
        depth_columns = [
            "depth",
            "fragments",
            "n_fragments",
            "nFrags",
            "total_fragments",
        ]
        for col in depth_columns:
            if col in metadata.columns:
                original_count = len(metadata)
                metadata = metadata[
                    (metadata[col] >= min_fragments) & (metadata[col] <= max_fragments)
                ]
                self.logger.info(
                    f"Filtered by {col}: {original_count} -> {len(metadata)} cells"
                )
                break

        # Get barcodes
        barcodes = self._get_barcode_column(metadata).values
        barcode_dict = {k: i for i, k in enumerate(barcodes)}

        return metadata, barcode_dict

    def _filter_joung_tfatlas(
        self, metadata_path: str, depth_threshold: int = 0
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Filter Joung TF Atlas dataset metadata."""
        cell_metadata = pd.read_csv(metadata_path, sep=",")

        # Set column name [0] -> Barcodes
        columns = cell_metadata.columns.values
        columns[0] = "Barcodes"
        cell_metadata.columns = columns

        filtered_metadata = cell_metadata

        # Get barcode dict
        barcodes = filtered_metadata.Barcodes.values
        barcodes_dict = {k: i for i, k in enumerate(barcodes)}

        return filtered_metadata, barcodes_dict

    def _filter_t_exhaustion(
        self, metadata_path: str, depth_threshold: int = 0
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Filter T cell exhaustion dataset metadata."""
        cell_metadata = pd.read_csv(metadata_path, sep="\t", compression="gzip")
        filtered_metadata = cell_metadata

        # Get barcode dict
        barcodes = filtered_metadata.Barcodes.values
        barcodes_dict = {k: i for i, k in enumerate(barcodes)}

        return filtered_metadata, barcodes_dict

    def _filter_linnarsson_brain(
        self, metadata_path: str, depth_threshold: int = 0
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Filter Linnarsson brain dataset metadata."""
        cell_metadata = pd.read_csv(metadata_path, sep=",", header=0, index_col=0)
        filtered_metadata = cell_metadata

        # Extract barcodes from index
        barcodes = filtered_metadata.index.str.split(":").str[1]
        # Append '-1' to match fragment file format
        barcodes = barcodes + "-1"
        barcodes = barcodes.values
        barcodes_dict = {k: i for i, k in enumerate(barcodes)}

        return filtered_metadata, barcodes_dict

    def _filter_yanhu_bmms(
        self, metadata_path: str, depth_threshold: int = 500
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Filter Yanhu BMMs dataset metadata."""
        cell_metadata = pd.read_csv(metadata_path, sep="\t")

        filtered_metadata = cell_metadata.sort_values(
            ["cistopic.louvain", "depth"], ascending=[True, False]
        ).query(f"depth > {depth_threshold}")

        # Get barcode dict
        barcodes = filtered_metadata.index.values
        barcodes_dict = {k: i for i, k in enumerate(barcodes)}

        return filtered_metadata, barcodes_dict

    def _filter_shendure_fetal(
        self, metadata_path: str, depth_threshold: int = 0
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Filter Shendure fetal dataset metadata."""
        cell_metadata = pd.read_csv(
            metadata_path,
            sep="\t",
            names=["barcode", "logUMI", "tsse", "sample_id", "cell type", "life stage"],
        )
        filtered_metadata = cell_metadata

        # Get barcode dict
        barcodes = filtered_metadata.barcode.values
        barcodes_dict = {k: i for i, k in enumerate(barcodes)}

        self.logger.info(f"Number of cells: {len(barcodes)}")

        return filtered_metadata, barcodes_dict

    def _filter_archr_cellline(
        self, metadata_path: str, depth_threshold: int = 500
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Filter ArchR cell line dataset metadata."""
        cell_metadata = pd.read_csv(metadata_path, sep=",")
        cell_metadata.columns = ["barcode"] + list(cell_metadata.columns[1:])
        cell_metadata["barcode"] = cell_metadata["barcode"].str.split("#").str[1]

        filtered_metadata = (
            cell_metadata.rename({"nFrags": "depth"}, axis=1)
            .sort_values(["barcode", "depth"], ascending=[True, False])
            .query(f"depth > {depth_threshold}")
            .query('barcode!="."')
        )

        # Get barcode dict
        barcodes = filtered_metadata.barcode.values
        barcodes_dict = {k: i for i, k in enumerate(barcodes)}

        return filtered_metadata, barcodes_dict

    def _filter_htan(
        self, metadata_path: str, depth_threshold: int = 0
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Filter HTAN dataset metadata."""
        cell_metadata = pd.read_csv(metadata_path, sep="\t")
        filtered_metadata = cell_metadata

        # Get barcode dict
        barcodes = filtered_metadata.Barcode.values
        barcodes_dict = {k: i for i, k in enumerate(barcodes)}

        return filtered_metadata, barcodes_dict

    def _filter_vijay_hematopoiesis(
        self, metadata_path: str, depth_threshold: int = 0
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Filter Vijay hematopoiesis dataset metadata."""
        cell_metadata = pd.read_csv(metadata_path).rename(
            {"Unnamed: 0": "barcode"}, axis=1
        )

        # Map cell types
        cell_metadata["STD.CellType"] = cell_metadata["STD.CellType"].map(
            lambda x: x if x != "Refined.HSC" else "HSC"
        )

        filtered_metadata = cell_metadata

        # Standardize barcodes
        filtered_metadata["barcode"] = (
            filtered_metadata.barcode.str.split("-").str[0] + "-1"
        )

        # Get barcode dict
        barcodes = filtered_metadata.barcode.values
        barcodes_dict = {k: i for i, k in enumerate(barcodes)}

        return filtered_metadata, barcodes_dict
