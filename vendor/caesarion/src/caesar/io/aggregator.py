from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy.sparse import csr_matrix, vstack

if TYPE_CHECKING:
    from caesar.io.interface import AggregatedCoverage

DataType = Union[np.ndarray, csr_matrix]
RawCoverageDict = dict[str, DataType]
AggregatedCoverage = Union[
    DataType, dict[str, DataType], list[DataType]
]  # List is for sparse concat


class Aggregator(ABC):
    """Abstract base class for aggregation strategies."""

    @abstractmethod
    def aggregate(self, raw_coverage_dict: RawCoverageDict) -> AggregatedCoverage:
        """Aggregates coverage data from a dictionary."""
        pass

    def get_track_keys(self, raw_coverage_dict: RawCoverageDict) -> list[str]:
        """Returns the keys expected in the final Track object's 'tracks' dict."""
        # Default implementation, subclasses might override
        return sorted(list(raw_coverage_dict.keys()))


class SumAggregator(Aggregator):
    """Aggregates by summing all coverages. Note that this always returns a 1D array, even when sparse=True."""

    def aggregate(self, raw_coverage_dict: RawCoverageDict) -> DataType:
        if not raw_coverage_dict:
            # Return an empty array/matrix - need shape info ideally, but hard here.
            # Let's return a 0-element array. Track constructor handles empty.
            logging.warning("SumAggregator received empty dict, returning empty array.")
            return np.array([], dtype=np.float32)  # Default dtype assumption

        values = list(raw_coverage_dict.values())
        first_val = values[0]
        if isinstance(first_val, csr_matrix):
            if not all(isinstance(v, csr_matrix) for v in values):
                raise TypeError("Cannot sum sparse and dense matrices together.")
            if not values:
                return csr_matrix((1, 0))  # Empty sparse
            # Sum sparse matrices column-wise
            return (
                vstack(values, format="csr").sum(axis=0).A1  # shape is (N,)
            )
        elif isinstance(first_val, np.ndarray):
            if not all(isinstance(v, np.ndarray) for v in values):
                raise TypeError("Cannot sum sparse and dense matrices together.")
            if not values:
                return np.array([], dtype=first_val.dtype)  # Empty dense
            # Sum dense arrays
            return np.sum(np.stack(values, axis=0), axis=0)
        else:
            raise TypeError(f"Unsupported type for summation: {type(first_val)}")

    def get_track_keys(self, raw_coverage_dict: RawCoverageDict) -> list[str]:
        # For sum, the Track object will have a single key (default or user-provided)
        return ["aggregated_sum"]  # Placeholder key


class ByKeyAggregator(Aggregator):
    """Returns the coverage dictionary as is."""

    def aggregate(self, raw_coverage_dict: RawCoverageDict) -> dict[str, DataType]:
        return raw_coverage_dict

    # get_track_keys uses default implementation (sorted keys of input dict)


class ConcatAggregator(Aggregator):
    """Aggregates by concatenating/stacking coverages."""

    def aggregate(
        self, raw_coverage_dict: RawCoverageDict
    ) -> np.ndarray | list[csr_matrix]:
        if not raw_coverage_dict:
            return []  # Return empty list for sparse/dense concat

        values = [
            raw_coverage_dict[k] for k in sorted(raw_coverage_dict.keys())
        ]  # Consistent order
        first_val = values[0]

        if isinstance(first_val, csr_matrix):
            if not all(isinstance(v, csr_matrix) for v in values):
                raise TypeError(
                    "Cannot concatenate sparse and dense matrices together."
                )
            # Return list of sparse matrices for concat (vstacking might create huge dense intermediate)
            return values
        elif isinstance(first_val, np.ndarray):
            if not all(isinstance(v, np.ndarray) for v in values):
                raise TypeError(
                    "Cannot concatenate sparse and dense matrices together."
                )
            # Stack dense arrays
            return np.stack(values, axis=0)
        else:
            raise TypeError(f"Unsupported type for concatenation: {type(first_val)}")

    def get_track_keys(self, raw_coverage_dict: RawCoverageDict) -> list[str]:
        # Keys will be derived from the sorted input keys
        return sorted(list(raw_coverage_dict.keys()))
