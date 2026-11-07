from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_feature_names(file_path: str) -> np.ndarray:
    """
    Load the feature names from the specified file path.

    Parameters:
    file_path (str): The file system path to the feature names file.

    Returns:
    np.ndarray: The feature names loaded from the file.
    """
    return np.loadtxt(file_path, dtype=str)


def load_barcodes(file_path: str) -> pd.DataFrame:
    """
    Load the barcodes from the specified file path.

    Parameters:
    file_path (str): The file system path to the barcodes file.

    Returns:
    pd.DataFrame: The barcodes loaded from the file.
    """
    return pd.read_csv(file_path, index_col=0)


class Metadata:
    def __init__(self, metadata_df, config):
        """
        Initialize the Metadata class.

        Parameters:
        metadata_df (pd.DataFrame): The metadata dataframe.
        config (dict): The configuration dictionary.
        """
        self.config = config
        self.metadata = metadata_df
        self.metadata = metadata_df
        if "labels" not in self.metadata.columns:
            self.metadata["labels"] = self.metadata[self.config.labels_column].apply(
                lambda x: [x]
            )
        else:
            self.metadata["labels"] = self.metadata["labels"].apply(
                lambda x: x.split(",")
            )

    @property
    def labels(self):
        """
        Get the unique labels from the metadata.

        Returns:
        np.ndarray: An array of unique labels.
        """
        if not hasattr(self, "_labels"):
            labels = np.array(
                [item for sublist in self.metadata["labels"].values for item in sublist]
            )
            self._labels = np.unique(labels)
        return self._labels

    def create_queries(self):
        """
        Create queries for each label.

        Returns:
        dict: A dictionary where keys are query names and values are dictionaries of barcodes.
        """
        queries = {}
        for label in self.labels:
            query_metadata = self.metadata[
                self.metadata["labels"].apply(lambda x: label in x)
            ]
            query_names, query = self.subsample_query(label, query_metadata)
            queries[query_names] = query
        return queries

    def subsample_query(self, label, query_metadata):
        """
        Subsample the query metadata for a given label.

        Parameters:
        label (str): The label to filter by.
        query_metadata (pd.DataFrame): The metadata for the query.

        Returns:
        str: The name of the query.
        dict: The barcodes for the query.
        """
        n_insertions = self.config.n_insertions
        if n_insertions is None:
            raise ValueError("n_insertions must be provided in the config.")
        # set random state
        random_state = np.random.RandomState(self.config.seed)
        # shuffle the barcode
        query_metadata = query_metadata.sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
        query_metadata["cumsum_libsize"] = query_metadata["per_cell_libsize"].cumsum()
        # get barcodes until n_insertions library size is reached
        query_metadata = query_metadata.loc[
            query_metadata["cumsum_libsize"] <= n_insertions
        ]
        query_name = f"{label}.L{str(n_insertions // 1000000)}M"
        query_dict = (
            query_metadata[["data_key", "cell_idx"]]
            .groupby("data_key")
            .agg(list)
            .to_dict()["cell_idx"]
        )
        return query_name, query_dict

    def add_labels(self, label_dict: dict):
        """
        Add new labels to the existing labels column based on a dictionary input.

        Parameters:
        label_dict (dict): A dictionary where keys are label names and values are numerical indices of the dataframe.

        Returns:
        None
        """
        for label, indices in label_dict.items():
            self.metadata.loc[indices, "labels"] = self.metadata.loc[
                indices, "labels"
            ].apply(lambda x: x + [label])

        # Update the _labels attribute to include new labels
        if hasattr(self, "_labels"):
            del self._labels

    def get_metadata_by_label(self, label: str) -> pd.DataFrame:
        """
        Retrieve metadata for a specific label.

        Parameters:
        label (str): The label to filter by.

        Returns:
        pd.DataFrame: A DataFrame containing metadata for the specified label.
        """
        metadata_subset = self.metadata[
            self.metadata["labels"].apply(lambda x: label in x)
        ]
        return Metadata(metadata_subset, self.config)

    def to_csv(self, file_path: str):
        """
        Export the metadata to a CSV file.

        Parameters:
        file_path (str): The file system path to save the CSV file.

        Returns:
        None
        """
        self.metadata.to_csv(file_path, index=False)

    @classmethod
    def load_from_csv(cls, file_path: str, config):
        """
        Load metadata from a CSV file.

        Parameters:
        file_path (str): The file system path to the CSV file.

        Returns:
        Metadata: An instance of the Metadata class.
        """
        metadata_df = pd.read_csv(file_path)
        return cls(metadata_df, config)


def load_cell_metadata(
    metadata_path: str | Path,
    barcodes: list[str] | np.ndarray,
    label_column: str,
    barcode_column: str = "barcode",
    sep: str | None = None,
    default_label: str = "unknown",
) -> np.ndarray:
    """
    Load cell metadata from a CSV/TSV file and extract labels for specific barcodes.

    Parameters:
    -----------
    metadata_path : Union[str, Path]
        Path to the metadata CSV/TSV file
    barcodes : Union[List[str], np.ndarray]
        Array of barcodes to match
    label_column : str
        Name of the column containing cell type labels
    barcode_column : str, default="barcode"
        Name of the column containing barcodes
    sep : Optional[str], default=None
        Delimiter for the file. If None, will be inferred from file extension
    default_label : str, default="unknown"
        Label to assign to barcodes not found in metadata

    Returns:
    --------
    np.ndarray
        Array of labels aligned with input barcodes

    Raises:
    -------
    FileNotFoundError
        If metadata file doesn't exist
    KeyError
        If required columns are not found in metadata file
    """
    # Convert to Path object
    metadata_path = Path(metadata_path)

    # Check file exists
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Determine separator
    if sep is None:
        sep = "\t" if metadata_path.suffix.lower() == ".tsv" else ","

    # Load metadata
    try:
        metadata_df = pd.read_csv(metadata_path, sep=sep)
    except Exception as e:
        raise ValueError(f"Error reading metadata file: {e}")

    # Check required columns exist
    if barcode_column not in metadata_df.columns:
        raise KeyError(
            f"Barcode column '{barcode_column}' not found in metadata. "
            f"Available columns: {list(metadata_df.columns)}"
        )
    if label_column not in metadata_df.columns:
        raise KeyError(
            f"Label column '{label_column}' not found in metadata. "
            f"Available columns: {list(metadata_df.columns)}"
        )

    # Create barcode to label mapping
    barcode_to_label = dict(zip(metadata_df[barcode_column], metadata_df[label_column]))

    # Map barcodes to labels
    labels = np.array([barcode_to_label.get(bc, default_label) for bc in barcodes])

    # Log statistics
    n_matched = sum(bc in barcode_to_label for bc in barcodes)
    n_unmatched = len(barcodes) - n_matched
    unique_labels = np.unique(labels)

    print(f"Loaded metadata from {metadata_path}")
    print(f"  - Matched: {n_matched}/{len(barcodes)} barcodes")
    print(f"  - Unmatched: {n_unmatched} barcodes (assigned '{default_label}')")
    print(f"  - Unique labels: {len(unique_labels)}")

    return labels


def load_and_merge_metadata(
    fragments: list,
    barcode_col: str = "barcode",
    sample_id_col: str = "sample_id",
    logger=None,
) -> pd.DataFrame | None:
    """
    Load and merge metadata from multiple fragment file configurations.
    
    Automatically adds sample_id column when multiple fragments are present or when
    a single shared metadata file already contains sample_id. The sample_id is used
    for disambiguating non-unique barcodes across fragments.
    
    Parameters:
    -----------
    fragments : list
        List of fragment file configurations (objects with .metadata and .id attributes)
    barcode_col : str, default="barcode"
        Column name for cell barcodes in metadata files
    sample_id_col : str, default="sample_id"
        Column name for sample IDs in metadata files. Used to disambiguate
        non-unique barcodes across fragments.
    logger : logging.Logger, optional
        Logger instance for warnings/errors. If None, uses print.
    
    Returns:
    --------
    pd.DataFrame | None
        Merged metadata DataFrame, or None if no metadata found.
        When multiple fragments are present, sample_id column is added/used to preserve
        duplicate barcodes across samples. For single fragment, sample_id is still added
        but not strictly necessary.
    
    Examples:
    --------
    >>> from caesar.io.metadata_io import load_and_merge_metadata
    >>> metadata = load_and_merge_metadata(
    ...     fragments=config.data.fragments,
    ...     barcode_col="barcode",
    ...     sample_id_col="sample_id"
    ... )
    """
    all_metadata = []
    log_func = logger.warning if logger else print
    
    # Check if we have a single shared metadata file
    unique_metadata_files = set()
    for frag in fragments:
        if hasattr(frag, "metadata") and frag.metadata:
            unique_metadata_files.add(frag.metadata)
    
    is_single_shared_metadata = len(unique_metadata_files) == 1 and len(fragments) > 1
    
    # If single shared metadata, load it once
    if is_single_shared_metadata:
        shared_metadata_path = Path(list(unique_metadata_files)[0])
        if not shared_metadata_path.exists():
            log_func(f"Shared metadata file not found: {shared_metadata_path}, skipping")
            return None
        
        try:
            shared_meta = pd.read_csv(shared_metadata_path)
            if barcode_col not in shared_meta.columns:
                log_func(
                    f"No '{barcode_col}' column in {shared_metadata_path}. "
                    f"Available columns: {list(shared_meta.columns)}. Skipping."
                )
                return None
            
            # Check if sample_id column exists (REQUIRED for multiple fragments with shared metadata)
            if sample_id_col not in shared_meta.columns:
                # Why sample_id is required:
                # When using a single shared metadata file with multiple fragment files, we need to know
                # which metadata rows correspond to which fragment. Without sample_id, we cannot:
                # 1. Disambiguate barcodes that appear in multiple fragments (non-unique barcodes)
                # 2. Correctly match metadata to the right fragment during processing
                #
                # Why we can't automatically infer sample_id:
                # - Reading fragment files just to extract barcodes is expensive and slow
                # - Fragment files may be large (GBs) - reading them for metadata loading defeats the purpose
                # - For 10x fragment files, parsing TSV/TSV.GZ is slow
                # - The processor will read fragment files anyway during processing, so we'd duplicate work
                # - If barcodes overlap across fragments, we can't uniquely assign rows without sample_id
                #
                # Solution: User must provide sample_id column in metadata matching fragment file prefixes/IDs
                # Example: For fragment files 'fragment1.tsv.gz' and 'fragment2.tsv.gz', metadata should have:
                #   barcode,celltype,sample_id
                #   AAAC...,B_cell,fragment1
                #   AAAC...,T_cell,fragment2
                #
                # Alternative column names can be specified via --sample-id-col (e.g., 'gem_id' for tonsil data)
                available_cols = list(shared_meta.columns)
                raise ValueError(
                    f"Single shared metadata file {shared_metadata_path} is used with {len(fragments)} fragment files, "
                    f"but required '{sample_id_col}' column is missing.\n\n"
                    f"This column is REQUIRED to disambiguate barcodes across fragments. "
                    f"Each row in metadata must have a '{sample_id_col}' value matching the fragment file prefix "
                    f"(e.g., 'fragment1' for 'fragment1.tsv.gz') or fragment ID.\n\n"
                    f"Available columns in metadata: {available_cols}\n\n"
                    f"Solutions:\n"
                    f"  1. Add '{sample_id_col}' column to your metadata file\n"
                    f"  2. If your metadata uses a different column name (e.g., 'gem_id'), specify it with --sample-id-col\n"
                    f"  3. Use separate metadata files (one per fragment) instead of a single shared file"
                )
            else:
                if logger:
                    logger.info(
                        f"Using shared metadata with '{sample_id_col}' column for {len(fragments)} fragments. "
                        f"Sample IDs in metadata: {sorted(shared_meta[sample_id_col].unique())}"
                    )
            
            return shared_meta
        except Exception as e:
            log_func(f"Error reading shared metadata from {shared_metadata_path}: {e}. Skipping.")
            return None
    
    # Per-fragment metadata - load each separately
    for frag in fragments:
        if not hasattr(frag, "metadata") or not frag.metadata:
            continue
            
        metadata_path = Path(frag.metadata)
        if not metadata_path.exists():
            log_func(f"Metadata file not found: {frag.metadata}, skipping")
            continue
        
        try:
            meta = pd.read_csv(metadata_path)
            if barcode_col not in meta.columns:
                log_func(
                    f"No '{barcode_col}' column in {frag.metadata}. "
                    f"Available columns: {list(meta.columns)}. Skipping."
                )
                continue
            
            # Always add sample_id column if fragment has an id and column doesn't exist
            # This ensures we can disambiguate barcodes even if not strictly needed
            if hasattr(frag, "id") and frag.id:
                if sample_id_col not in meta.columns:
                    meta[sample_id_col] = frag.id
                elif logger:
                    logger.info(
                        f"Metadata file {frag.metadata} already has '{sample_id_col}' column, "
                        f"not overwriting with fragment.id={frag.id}"
                    )
            
            all_metadata.append(meta)
        except Exception as e:
            log_func(f"Error reading metadata from {frag.metadata}: {e}. Skipping.")
            continue
    
    if not all_metadata:
        return None
    
    # Concatenate all metadata
    combined = pd.concat(all_metadata, ignore_index=True)
    
    # Handle deduplication based on whether sample_id is available
    if sample_id_col in combined.columns:
        # With sample_id, we preserve all barcodes (even duplicates across samples)
        # Only remove duplicates that have the same barcode AND sample_id
        n_before = len(combined)
        combined = combined.drop_duplicates(subset=[barcode_col, sample_id_col], keep="first")
        n_after = len(combined)
        
        if logger:
            n_unique_barcodes = combined[barcode_col].nunique()
            n_total = len(combined)
            if n_unique_barcodes < n_total:
                logger.info(
                    f"Merged metadata: {n_total} total cells, {n_unique_barcodes} unique barcodes. "
                    f"'{sample_id_col}' column used for disambiguation."
                )
            if n_before != n_after:
                logger.info(
                    f"Removed {n_before - n_after} duplicate barcode+sample_id combinations."
                )
    else:
        # No sample_id - remove duplicates based on barcode only (keep first occurrence)
        n_before = len(combined)
        combined = combined.drop_duplicates(subset=[barcode_col], keep="first")
        n_after = len(combined)
        
        if n_before != n_after and logger:
            logger.info(
                f"Removed {n_before - n_after} duplicate barcodes from merged metadata "
                f"(kept first occurrence). Consider adding '{sample_id_col}' column to preserve "
                f"sample information when barcodes are not unique across fragments."
            )
    
    return combined


def apply_cluster_mapping(
    metadata: pd.DataFrame,
    mapping_file: str | Path,
    source_col: str,
    target_col: str = "labels",
    default_label: str | None = None,
    exclude_unmapped: bool = False,
    logger=None,
) -> pd.DataFrame:
    """
    Apply cluster mapping to metadata DataFrame.
    
    Maps fine-grained clusters to broad groups using a JSON mapping file.
    
    Parameters:
    -----------
    metadata : pd.DataFrame
        Metadata DataFrame with source cluster column
    mapping_file : str | Path
        Path to JSON file with cluster mapping
        Format: {"fine_cluster1": "broad_group1", "fine_cluster2": "broad_group1", ...}
    source_col : str
        Column name containing source clusters to map
    target_col : str, default="labels"
        Column name for mapped labels (will be created or overwritten)
    default_label : str | None, default=None
        Label to assign to unmapped clusters. If None and exclude_unmapped=False,
        unmapped clusters keep their original value.
    exclude_unmapped : bool, default=False
        If True, exclude rows with unmapped clusters from result
    logger : logging.Logger, optional
        Logger instance for warnings. If None, uses print.
    
    Returns:
    --------
    pd.DataFrame
        Metadata DataFrame with mapped labels in target_col
    
    Raises:
    -------
    FileNotFoundError
        If mapping file doesn't exist
    KeyError
        If source_col not found in metadata
    ValueError
        If mapping file is invalid JSON
    
    Examples:
    --------
    >>> from caesar.io.metadata_io import apply_cluster_mapping
    >>> mapping = {
    ...     "Naive_B": "B_cell",
    ...     "Memory_B": "B_cell",
    ...     "CD4_naive": "T_cell",
    ... }
    >>> import json
    >>> with open("mapping.json", "w") as f:
    ...     json.dump(mapping, f)
    >>> metadata_mapped = apply_cluster_mapping(
    ...     metadata=metadata,
    ...     mapping_file="mapping.json",
    ...     source_col="cluster",
    ...     target_col="celltype"
    ... )
    """
    import json
    
    mapping_path = Path(mapping_file)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Cluster mapping file not found: {mapping_file}")
    
    if source_col not in metadata.columns:
        raise KeyError(
            f"Source column '{source_col}' not found in metadata. "
            f"Available columns: {list(metadata.columns)}"
        )
    
    # Load mapping
    try:
        with open(mapping_path, "r") as f:
            cluster_mapping = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in mapping file {mapping_file}: {e}")
    
    log_func = logger.warning if logger else print
    
    # Apply mapping
    metadata = metadata.copy()
    metadata[target_col] = metadata[source_col].map(cluster_mapping)
    
    # Handle unmapped clusters
    unmapped_mask = metadata[target_col].isna()
    n_unmapped = unmapped_mask.sum()
    
    if n_unmapped > 0:
        if exclude_unmapped:
            log_func(f"Excluding {n_unmapped} cells with unmapped clusters")
            metadata = metadata[~unmapped_mask].copy()
        elif default_label is not None:
            log_func(
                f"Assigning '{default_label}' to {n_unmapped} unmapped clusters"
            )
            metadata.loc[unmapped_mask, target_col] = default_label
        else:
            # Keep original values for unmapped
            log_func(
                f"Warning: {n_unmapped} cells have unmapped clusters. "
                "Keeping original values."
            )
            metadata.loc[unmapped_mask, target_col] = metadata.loc[
                unmapped_mask, source_col
            ]
    
    if logger:
        unique_mapped = metadata[target_col].nunique()
        logger.info(
            f"Applied cluster mapping: {len(cluster_mapping)} source clusters -> "
            f"{unique_mapped} target groups"
        )
    
    return metadata


def validate_cell_labels(
    labels: list[str] | np.ndarray,
    max_length: int = 50,
    allowed_chars: str | None = None,
) -> np.ndarray:
    """
    Validate and sanitize cell type labels.

    Parameters:
    -----------
    labels : Union[List[str], np.ndarray]
        Array of cell type labels
    max_length : int, default=50
        Maximum allowed length for labels
    allowed_chars : Optional[str], default=None
        String of allowed characters. If None, allows alphanumeric + underscore

    Returns:
    --------
    np.ndarray
        Validated and sanitized labels

    Raises:
    -------
    ValueError
        If labels contain invalid characters or are too long
    """
    labels = np.asarray(labels, dtype=str)

    # Check for empty labels
    empty_mask = labels == ""
    if empty_mask.any():
        print(
            f"Warning: {empty_mask.sum()} empty labels found, replacing with 'unknown'"
        )
        labels[empty_mask] = "unknown"

    # Check label length
    too_long = np.array([len(label) > max_length for label in labels])
    if too_long.any():
        print(
            f"Warning: {too_long.sum()} labels exceed max length {max_length}, truncating"
        )
        labels[too_long] = np.array([label[:max_length] for label in labels[too_long]])

    # Validate characters if specified
    if allowed_chars is not None:
        import re

        pattern = f"^[{re.escape(allowed_chars)}]+$"
        invalid = []
        for i, label in enumerate(labels):
            if not re.match(pattern, label):
                invalid.append((i, label))

        if invalid:
            raise ValueError(
                f"Invalid characters in {len(invalid)} labels. "
                f"Allowed characters: {allowed_chars}"
            )

    return labels
