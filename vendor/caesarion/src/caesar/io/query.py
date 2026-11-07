from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class BaseQuery(ABC):
    """
    Abstract base class for genomic data query specifications.

    This is the parent class for all query types used to select subsets of data
    from genomic IO classes. Different IO classes support different query types:

    - DenseZarrIO: Supports only DefaultQuery
    - SingleSparseZarrIO: Supports all query types
    - MultiSparseZarrIO: Supports all query types with dataset-specific semantics
    - CelltypeDenseZarrIO: Supports DefaultQuery and KeyQuery
    - BPCellsIO: Support varies by subclass implementation
    """

    pass


class DefaultQuery(BaseQuery):
    """
    Represents selecting the default data without filtering.

    This is the simplest query type that returns all available data within the requested region:
    - In DenseZarrIO: Returns the full signal data
    - In SingleSparseZarrIO: Selects all cells
    - In MultiSparseZarrIO: Treats each sample as a separate group
    - In CelltypeDenseZarrIO: Gets coverage for all configured cell type IDs
    - In SignalDenseBPCellsIO: Returns the single default signal track

    Example:
        ```python
        query = DefaultQuery()
        signal = dense_io.get_track("chr1", 1000, 2000, query=query)
        ```
    """

    def __repr__(self):
        return "DefaultQuery()"


class IndexQuery(BaseQuery):
    """
    Query based on numerical indices to select specific elements.

    Used to select data by position in an array or matrix:
    - In SingleSparseZarrIO: Selects specific cell indices
    - In MultiSparseZarrIO: Selects cells by indices across all samples
    - Not supported in DenseZarrIO and CelltypeDenseZarrIO
    - In SingleSparseBPCellsIO: Maps indices to cell matrix positions

    Parameters:
        indices (Sequence[int]): A sequence of integer indices to select

    Example:
        ```python
        # Select cells at positions 0, 5, and 10
        query = IndexQuery([0, 5, 10])
        track = sparse_io.get_track("chr1", 1000, 2000, query=query)
        ```
    """

    def __init__(self, indices: Sequence[int]):
        """
        Initialize with a sequence of integer indices.

        Args:
            indices (Sequence[int]): Numerical indices for data selection
        """
        self.indices = np.asarray(indices, dtype=int)

    def __repr__(self):
        return f"IndexQuery(indices=<{len(self.indices)} indices>)"


class KeyQuery(BaseQuery):
    """
    Query based on one or more string keys for data selection.

    Keys have different meanings in different IO contexts:
    - In SingleSparseZarrIO: Selects cells by barcode identifiers
    - In MultiSparseZarrIO: Selects samples or cells by label
    - In CelltypeDenseZarrIO: Selects specific cell type IDs
    - In BPCellsIO implementations: Often selects by cell barcode or group name
    - Not supported in DenseZarrIO

    Parameters:
        keys (str | Sequence[str]): Single key or sequence of keys for selection

    Example:
        ```python
        # Select a specific cell type
        query = KeyQuery("CD4+ T cells")
        track = celltype_io.get_track("chr1", 1000, 2000, query=query)

        # Select multiple samples
        query = KeyQuery(["sample1", "sample2"])
        tracks = multi_io.get_track("chr1", 1000, 2000, query=query)
        ```
    """

    def __init__(self, keys: str | Sequence[str]):
        """
        Initialize with string keys for selection.

        Args:
            keys (str | Sequence[str]): String identifier(s) for data selection
        """
        self.keys = [keys] if isinstance(keys, str) else list(keys)

    def __repr__(self):
        return f"KeyQuery(keys={self.keys})"


class MetadataQuery(BaseQuery):
    """
    Query based on a metadata filter string, similar to pandas query syntax.

    Used to filter data based on metadata attributes:
    - In SingleSparseZarrIO: Queries cells based on metadata columns
    - In MultiSparseZarrIO: Queries the combined metadata across samples
    - Not supported in DenseZarrIO and CelltypeDenseZarrIO
    - In SingleSparseBPCellsIO: Filters cells by metadata conditions

    Parameters:
        query_string (str): A filter expression in pandas query format

    Example:
        ```python
        # Select cells where 'cluster' is 3 and 'n_genes' > 500
        query = MetadataQuery("cluster == 3 and n_genes > 500")
        track = sparse_io.get_track("chr1", 1000, 2000, query=query)
        ```
    """

    def __init__(self, query_string: str):
        """
        Initialize with a query string for metadata filtering.

        Args:
            query_string (str): Filter expression to apply to metadata
        """
        self.query_string = query_string

    def __repr__(self):
        return f"MetadataQuery(query_string='{self.query_string}')"


class MultiKeyQuery(BaseQuery):
    """
    Query representing multiple output groups, typically for aggregated or comparative analyses.

    Maps output group names to their constituent elements through nested queries or direct lists:
    - In SingleSparseZarrIO: Groups cells into multiple categories
    - In MultiSparseZarrIO: Creates complex groupings across samples
    - In SingleSparseBPCellsIO: Creates grouped outputs from cell selections
    - Not supported in DenseZarrIO and CelltypeDenseZarrIO

    Parameters:
        mapping (dict): Maps output group names to specifications defining their contents.
                        Values can be sequences of keys/indices or nested query objects.

    Example:
        ```python
        # Group cells by cluster type
        query = MultiKeyQuery(
            {
                "T_cells": MetadataQuery("cell_type in ['CD4', 'CD8']"),
                "B_cells": MetadataQuery("cell_type == 'B cell'"),
                "other": KeyQuery(["NK", "Monocyte"]),
            }
        )
        tracks = sparse_io.get_track("chr1", 1000, 2000, query=query)
        # Result will have keys "T_cells", "B_cells", and "other"
        ```
    """

    def __init__(self, mapping: dict[str, Sequence[str] | Sequence[int] | BaseQuery]):
        """
        Initialize with a mapping from group names to their selection specifications.

        Args:
            mapping (dict): Dictionary where keys are output group names and values
                          are selection specifications (keys, indices, or nested queries)
        """
        # Allow values to be keys, indices, or even nested queries for complex cases
        self.mapping = mapping

    def __repr__(self):
        return f"MultiKeyQuery(mapping=<{len(self.mapping)} groups>)"
