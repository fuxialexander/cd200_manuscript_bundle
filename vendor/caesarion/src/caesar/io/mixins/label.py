from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec
from zarr.core.dtype import VariableLengthUTF8

COMPRESSOR = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")


class LabelMixin:
    """Mixin for standardized label handling in genomic IO classes.

    This mixin provides consistent methods for reading, writing, and managing
    cell type labels across different IO implementations.
    """

    def store_labels(
        self,
        labels: np.ndarray | list[str],
        barcodes: np.ndarray | list[str] | None = None,
        dataset_name: str = "labels",
        dtype: str = "object",
        compressors=None,
        overwrite: bool = True,
    ) -> None:
        """Store cell labels in zarr with validation.

        Parameters
        ----------
        labels : np.ndarray or list[str]
            Array of cell type labels
        barcodes : np.ndarray or list[str], optional
            Array of barcodes. If provided, validates alignment with labels
        dataset_name : str, default="labels"
            Name of the dataset to store labels in
        dtype : str, default="object"
            Data type for labels. Use "object" with VLenUTF8 codec for variable length
        compressors : zarr codec, optional
            Compressor to use. If None, uses default compressor
        overwrite : bool, default=True
            Whether to overwrite existing labels
        """
        # Convert to numpy array
        labels = np.asarray(labels)

        # Validate labels
        labels = self._validate_labels(labels)

        # Validate alignment with barcodes if provided
        if barcodes is not None:
            barcodes = np.asarray(barcodes)
            if len(labels) != len(barcodes):
                raise ValueError(
                    f"Labels length ({len(labels)}) doesn't match "
                    f"barcodes length ({len(barcodes)})"
                )

        # Use default compressor if not provided
        if compressors is None:
            compressors = getattr(
                self, "compressors", BloscCodec(cname="zstd", clevel=3)
            )

        # Store labels
        if dtype == "object":
            # Use VariableLengthUTF8 for variable length strings in zarr v3
            # zarr v3 API requires creating array with shape, then assigning data
            arr = self.dataset.create_array(
                dataset_name,
                shape=labels.shape,
                dtype=VariableLengthUTF8(),
                compressors=compressors,
                overwrite=overwrite,
                chunks=(1000,),
            )
            arr[:] = labels
        else:
            # Fixed length strings - can use data= directly
            self.dataset.create_array(
                dataset_name,
                data=labels,
                compressors=compressors,
                overwrite=overwrite,
                chunks=(1000,),
            )

    def read_labels(
        self, dataset_name: str = "labels", default: str = "unknown"
    ) -> np.ndarray:
        """Read cell labels from zarr with fallback.

        Parameters
        ----------
        dataset_name : str, default="labels"
            Name of the dataset containing labels
        default : str, default="unknown"
            Default label to use if dataset doesn't exist

        Returns
        -------
        np.ndarray
            Array of cell labels
        """
        try:
            labels = self.dataset[dataset_name][:]
            return labels
        except KeyError:
            # Fallback behavior depends on the class
            if hasattr(self, "barcodes"):
                # For single-cell data, use default for all cells
                n_cells = len(self.barcodes)
                if hasattr(self, "data_key") and self.data_key:
                    # Use data_key if available
                    return np.repeat(self.data_key, n_cells)
                else:
                    return np.repeat(default, n_cells)
            else:
                raise KeyError(f"Labels dataset '{dataset_name}' not found")

    def get_unique_labels(self, dataset_name: str = "labels") -> np.ndarray:
        """Get unique labels.

        Parameters
        ----------
        dataset_name : str, default="labels"
            Name of the dataset containing labels

        Returns
        -------
        np.ndarray
            Array of unique labels
        """
        labels = self.read_labels(dataset_name)
        return np.unique(labels)

    def get_label_counts(self, dataset_name: str = "labels") -> pd.Series:
        """Get counts per label.

        Parameters
        ----------
        dataset_name : str, default="labels"
            Name of the dataset containing labels

        Returns
        -------
        pd.Series
            Series with labels as index and counts as values
        """
        labels = self.read_labels(dataset_name)
        return pd.Series(labels).value_counts()

    def rename_labels(
        self,
        label_map: dict[str, str],
        dataset_name: str = "labels",
        create_backup: bool = True,
    ) -> None:
        """Rename labels based on mapping.

        Parameters
        ----------
        label_map : dict[str, str]
            Mapping from old labels to new labels
        dataset_name : str, default="labels"
            Name of the dataset containing labels
        create_backup : bool, default=True
            Whether to create a backup of original labels
        """
        # Read current labels
        labels = self.read_labels(dataset_name)

        # Create backup if requested
        if create_backup:
            backup_name = f"{dataset_name}_backup"
            self.store_labels(labels, dataset_name=backup_name, overwrite=True)

        # Apply mapping
        new_labels = np.array([label_map.get(label, label) for label in labels])

        # Store renamed labels
        self.store_labels(new_labels, dataset_name=dataset_name, overwrite=True)

        logging.info(f"Renamed {len(label_map)} label types")

    def filter_by_labels(
        self, labels_to_keep: list[str] | set[str], dataset_name: str = "labels"
    ) -> np.ndarray:
        """Get indices of cells with specified labels.

        Parameters
        ----------
        labels_to_keep : list[str] or set[str]
            Labels to filter for
        dataset_name : str, default="labels"
            Name of the dataset containing labels

        Returns
        -------
        np.ndarray
            Boolean mask or indices of cells with specified labels
        """
        labels = self.read_labels(dataset_name)
        labels_to_keep = set(labels_to_keep)

        # Create boolean mask
        mask = np.array([label in labels_to_keep for label in labels])
        return mask

    def _validate_labels(
        self, labels: np.ndarray, max_length: int = 50, replace_empty: str = "unknown"
    ) -> np.ndarray:
        """Validate and sanitize labels.

        Parameters
        ----------
        labels : np.ndarray
            Array of labels to validate
        max_length : int, default=50
            Maximum allowed length for labels
        replace_empty : str, default="unknown"
            Value to use for empty labels

        Returns
        -------
        np.ndarray
            Validated labels
        """
        labels = np.asarray(labels, dtype=str)

        # Replace empty labels
        empty_mask = labels == ""
        if empty_mask.any():
            logging.warning(
                f"Found {empty_mask.sum()} empty labels, "
                f"replacing with '{replace_empty}'"
            )
            labels[empty_mask] = replace_empty

        # Check label lengths
        too_long = np.array([len(label) > max_length for label in labels])
        if too_long.any():
            logging.warning(
                f"Found {too_long.sum()} labels exceeding max length {max_length}, "
                "truncating"
            )
            labels[too_long] = np.array(
                [label[:max_length] for label in labels[too_long]]
            )

        return labels

    def get_label_metadata(
        self, dataset_name: str = "labels", include_indices: bool = True
    ) -> pd.DataFrame:
        """Get metadata DataFrame with labels and additional information.

        Parameters
        ----------
        dataset_name : str, default="labels"
            Name of the dataset containing labels
        include_indices : bool, default=True
            Whether to include cell indices in the DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with labels and associated metadata
        """
        labels = self.read_labels(dataset_name)

        # Build metadata DataFrame
        metadata = {"labels": labels}

        # Add barcodes if available
        if hasattr(self, "barcodes"):
            metadata["barcodes"] = self.barcodes

        # Add data key if available
        if hasattr(self, "data_key"):
            metadata["data_key"] = self.data_key

        # Add indices if requested
        if include_indices:
            metadata["cell_idx"] = np.arange(len(labels))

        return pd.DataFrame(metadata)
