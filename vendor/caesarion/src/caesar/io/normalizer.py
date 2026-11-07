from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from caesar.io.interface import AggregatedCoverage, GenomicIO

NormFactors = dict[str, float]


class Normalizer(ABC):
    """Abstract base class for normalization strategies."""

    @abstractmethod
    def prepare_norm_factors(self, track_keys: list[str]) -> NormFactors | None:
        """Prepares the normalization factor dictionary for the Track object."""
        pass

    @abstractmethod
    def transform(self, data: AggregatedCoverage) -> AggregatedCoverage:
        """Applies normalization to the raw aggregated data.
        Used when output_format is not 'track_object'.
        """
        pass


class NoNormalizer(Normalizer):
    """Applies no normalization."""

    def prepare_norm_factors(self, track_keys: list[str]) -> NormFactors | None:
        return None

    def transform(self, data: AggregatedCoverage) -> AggregatedCoverage:
        """Returns data unchanged."""
        return data


class FactorNormalizer(Normalizer):
    """Applies a single normalization factor or an array of factors."""

    def __init__(self, factor: float | Sequence[float] | np.ndarray):
        if isinstance(factor, Sequence | np.ndarray) and not isinstance(factor, str):
            self.factor: float | np.ndarray = np.asarray(factor, dtype=float)
            if self.factor.ndim == 0:  # Convert 0-dim array to scalar float
                self.factor = self.factor.item()
            elif self.factor.ndim > 1:
                raise ValueError(
                    "FactorNormalizer factor array must be 1-dimensional or scalar."
                )
        elif isinstance(factor, int | float | np.number):
            self.factor = float(factor)
        else:
            raise TypeError(
                f"Factor must be numeric or a sequence/array of numerics, got {type(factor)}"
            )

    def prepare_norm_factors(self, track_keys: list[str]) -> NormFactors | None:
        if not track_keys:
            return None

        if isinstance(self.factor, np.ndarray):
            if self.factor.size == 1:  # Treat as scalar
                return {key: self.factor.item() for key in track_keys}
            else:
                logging.warning(
                    "FactorNormalizer with a vector of factors cannot directly prepare "
                    "NormFactors for a Track object with multiple keys without explicit mapping. "
                    "Use DictNormalizer for per-track factors for Track objects."
                )
                return None
        else:  # Scalar float
            return {key: self.factor for key in track_keys}

    def transform(self, data: AggregatedCoverage) -> AggregatedCoverage:
        """Applies the factor(s) to the data."""

        # Handle scalar factor case
        if isinstance(self.factor, float):
            if self.factor == 0:
                logging.warning(
                    "FactorNormalizer: Cannot normalize by zero scalar factor. Returning original data."
                )
                return data

            if isinstance(data, np.ndarray | csr_matrix):
                return data.astype(np.float64) / self.factor * 1e8
            elif isinstance(data, dict):
                # Recursively call transform for each value in the dictionary
                return {k: self.transform(v) for k, v in data.items()}
            elif isinstance(data, list):
                # Recursively call transform for each item in the list
                return [self.transform(item) for item in data]
            else:
                logging.warning(
                    f"FactorNormalizer (scalar) received unsupported data type: {type(data)}. Returning original data."
                )
                return data

        # Handle np.ndarray factor case (vector factor)
        elif isinstance(self.factor, np.ndarray):
            factor_array = self.factor
            if np.any(factor_array == 0):
                logging.warning(
                    "FactorNormalizer (vector): Factor array contains zeros. "
                    "Division by zero will result in inf/nan values."
                )

            if isinstance(data, np.ndarray):
                data_float = data.astype(np.float64)
                if data.ndim == 1:
                    if factor_array.size == 1:
                        return data_float / factor_array.item() * 1e8
                    elif factor_array.size == data.shape[0]:
                        return data_float / factor_array * 1e8
                    else:
                        logging.warning(
                            f"FactorNormalizer (vector): Data shape {data.shape} "
                            f"incompatible with factor array size {factor_array.size} for 1D np.ndarray."
                        )
                        return data
                elif data.ndim == 2:  # (N, M)
                    if factor_array.size == 1:
                        return data_float / factor_array.item() * 1e8
                    elif factor_array.size == data.shape[0]:  # Factor per row
                        return data_float / factor_array[:, np.newaxis] * 1e8
                    else:
                        logging.warning(
                            f"FactorNormalizer (vector): Data shape {data.shape} "
                            f"incompatible with factor array size {factor_array.size} for 2D np.ndarray row-wise normalization."
                        )
                        return data
                else:
                    logging.warning(
                        f"FactorNormalizer (vector): Only supports 1D or 2D np.ndarray, got {data.ndim}D."
                    )
                    return data
            elif isinstance(data, csr_matrix):  # csr_matrix (N, M)
                if factor_array.size == 1:
                    scalar_f = factor_array.item()
                    if scalar_f == 0:
                        logging.warning(
                            "FactorNormalizer (vector): Cannot normalize csr_matrix by zero scalar factor."
                        )
                        return data
                    return data.astype(np.float64) * (1.0 / scalar_f) * 1e8
                elif factor_array.size == data.shape[0]:  # Factor per row
                    new_data = data.copy().astype(np.float64)
                    for i in range(new_data.shape[0]):
                        if factor_array[i] != 0:
                            start_ptr, end_ptr = (
                                new_data.indptr[i],
                                new_data.indptr[i + 1],
                            )
                            new_data.data[start_ptr:end_ptr] /= factor_array[i] / 1e8
                        else:
                            # Values for rows with a zero factor become NaN
                            start_ptr, end_ptr = (
                                new_data.indptr[i],
                                new_data.indptr[i + 1],
                            )
                            new_data.data[start_ptr:end_ptr] = np.nan
                    return new_data
                else:
                    logging.warning(
                        f"FactorNormalizer (vector): csr_matrix shape {data.shape} "
                        f"incompatible with factor array size {factor_array.size} for row-wise normalization."
                    )
                    return data
            elif isinstance(data, dict | list):
                logging.warning(
                    "FactorNormalizer (vector): Applying a vector factor to a dict or list "
                    "is ambiguous. Please iterate and apply scalar factors or use DictNormalizer. Returning original data."
                )
                return data
            else:
                logging.warning(
                    f"FactorNormalizer (vector) received unsupported data type: {type(data)}. Returning original data."
                )
                return data
        else:
            # Should not happen due to __init__ type checking
            logging.error(
                f"FactorNormalizer: self.factor is of unexpected type {type(self.factor)}. Returning original data."
            )
            return data


class ListNormalizer(Normalizer):
    """Applies normalization factors from a list for concatenated array data."""

    def __init__(self, factors: list[float] | Sequence[float] | np.ndarray):
        if isinstance(factors, list | Sequence) and not isinstance(factors, str):
            self.factors = np.asarray(factors, dtype=float)
        elif isinstance(factors, np.ndarray):
            self.factors = factors.astype(float)
        else:
            raise TypeError(f"Factors must be a list, sequence, or array, got {type(factors)}")
        
        if self.factors.ndim != 1:
            raise ValueError("ListNormalizer factors must be 1-dimensional")

    def prepare_norm_factors(self, track_keys: list[str]) -> NormFactors | None:
        """ListNormalizer doesn't prepare factors for Track objects since it works with concatenated arrays."""
        return None

    def transform(self, data: AggregatedCoverage) -> AggregatedCoverage:
        """Applies list-based normalization to concatenated array data."""
        if isinstance(data, np.ndarray):
            data_float = data.astype(np.float64)
            
            if data.ndim == 1:
                # For 1D data, factors should match the array length
                if len(self.factors) == data.shape[0]:
                    # Element-wise normalization
                    mask = self.factors != 0
                    result = np.full_like(data_float, np.nan)
                    result[mask] = data_float[mask] / self.factors[mask] * 1e8
                    if not np.all(mask):
                        logging.warning(
                            f"ListNormalizer: {np.sum(~mask)} elements have zero factors and will be NaN"
                        )
                    return result
                else:
                    logging.warning(
                        f"ListNormalizer: Data length {data.shape[0]} does not match factors length {len(self.factors)}. Returning original data."
                    )
                    return data
            elif data.ndim == 2:
                # For 2D data (samples x features), normalize each sample (row) by corresponding factor
                if len(self.factors) == data.shape[0]:
                    # Row-wise normalization
                    zero_mask = self.factors == 0
                    if np.any(zero_mask):
                        logging.warning(
                            f"ListNormalizer: {np.sum(zero_mask)} factors are zero. Corresponding rows will be NaN."
                        )
                    
                    result = data_float / self.factors[:, np.newaxis] * 1e8
                    result[zero_mask, :] = np.nan
                    return result
                else:
                    logging.warning(
                        f"ListNormalizer: Data shape {data.shape} incompatible with factors length {len(self.factors)} for row-wise normalization. Returning original data."
                    )
                    return data
            else:
                logging.warning(
                    f"ListNormalizer: Only supports 1D or 2D arrays, got {data.ndim}D. Returning original data."
                )
                return data
        elif isinstance(data, csr_matrix):
            # Handle sparse matrix case (similar to FactorNormalizer vector case)
            if len(self.factors) == data.shape[0]:
                new_data = data.copy().astype(np.float64)
                for i in range(new_data.shape[0]):
                    if self.factors[i] != 0:
                        start_ptr, end_ptr = new_data.indptr[i], new_data.indptr[i + 1]
                        new_data.data[start_ptr:end_ptr] /= self.factors[i] / 1e8
                    else:
                        start_ptr, end_ptr = new_data.indptr[i], new_data.indptr[i + 1]
                        new_data.data[start_ptr:end_ptr] = np.nan
                return new_data
            else:
                logging.warning(
                    f"ListNormalizer: Sparse matrix shape {data.shape} incompatible with factors length {len(self.factors)}. Returning original data."
                )
                return data
        elif isinstance(data, dict):
            logging.warning(
                "ListNormalizer is designed for array data, but received dict. Use DictNormalizer for dictionary data. Returning original data."
            )
            return data
        elif isinstance(data, list):
            logging.warning(
                "ListNormalizer received list data. Consider using array format for better performance. Returning original data."
            )
            return data
        else:
            logging.warning(
                f"ListNormalizer received unsupported data type: {type(data)}. Returning original data."
            )
            return data


class DictNormalizer(Normalizer):
    """Applies normalization factors from a dictionary."""

    def __init__(self, factors: dict[str, float]):
        if not isinstance(factors, dict):
            raise TypeError(f"Factors must be a dictionary, got {type(factors)}")
        self.factors = {k: float(v) for k, v in factors.items()}  # Ensure float values

    def prepare_norm_factors(self, track_keys: list[str]) -> NormFactors | None:
        if not track_keys:
            return None
        # Return factors only for the keys present in the track
        return {key: self.factors[key] for key in track_keys if key in self.factors}

    def transform(self, data: AggregatedCoverage) -> AggregatedCoverage:
        """Applies factors from the dictionary to corresponding keys in the data."""
        if isinstance(data, dict):
            transformed_data = {}
            for key, value in data.items():
                if key in self.factors:
                    factor = self.factors[key]
                    if factor == 0:
                        logging.warning(
                            f"Cannot normalize key '{key}' by zero factor. Skipping."
                        )
                        transformed_data[key] = value  # Keep original data
                    else:
                        # Apply normalization similarly to FactorNormalizer
                        try:
                            transformed_data[key] = (
                                value.astype(np.float64) / factor * 1e8
                            )
                        except (TypeError, AttributeError):
                            logging.warning(
                                f"DictNormalizer cannot transform value for key '{key}' (type {type(value)}). Keeping original."
                            )
                            transformed_data[key] = value
                else:
                    logging.warning(
                        f"Key '{key}' not found in DictNormalizer factors. Keeping original data."
                    )
                    transformed_data[key] = value
            return transformed_data
        else:
            # DictNormalizer primarily makes sense for dict data. Warn if used otherwise.
            logging.warning(
                f"DictNormalizer is designed for dict data, received {type(data)}. Returning original data."
            )
            return data


class QuantileNormalizer(Normalizer):
    """
    Applies quantile normalization using background and scale factors
    loaded from the IO object's attributes.

    Normalization formula: normalized = (signal - background) / scale_factor
    """

    def __init__(self, io_object: GenomicIO):
        """
        Initialize QuantileNormalizer.

        Args:
            io_object (GenomicIO): The GenomicIO instance to load attributes from.
        """
        self.io_object = io_object
        self.background: dict[str, float] | None = None
        self.scale_factor: dict[str, float] | None = None
        self._load_params()

    def _load_params(self):
        """Loads background and scale_factor from io_object attributes."""
        try:
            params = self.io_object._load_attr("normalization_params")
            if isinstance(params, dict):
                self.background = params.get("background")
                self.scale_factor = params.get("scale_factor")
                if not isinstance(self.background, dict) or not isinstance(
                    self.scale_factor, dict
                ):
                    logging.error(
                        "QuantileNormalizer: 'normalization_params' attribute has incorrect format."
                    )
                    self.background = None
                    self.scale_factor = None
            else:
                logging.error(
                    "QuantileNormalizer: Could not load 'normalization_params' or it is not a dict."
                )
        except Exception as e:
            logging.error(
                f"QuantileNormalizer: Error loading normalization parameters: {e}"
            )
            self.background = None
            self.scale_factor = None

    def prepare_norm_factors(self, track_keys: list[str]) -> NormFactors | None:
        """Quantile normalization isn't represented by simple factors for Track object."""
        return None

    def transform(self, data: AggregatedCoverage) -> AggregatedCoverage:
        """Applies (signal - background) / scale_factor normalization."""
        if self.background is None or self.scale_factor is None:
            logging.error(
                "QuantileNormalizer parameters not loaded. Returning original data."
            )
            return data

        if isinstance(data, dict):
            transformed_data = {}
            for key, signal in data.items():
                # Ensure signal is numpy array for vectorized operations
                if not isinstance(signal, np.ndarray):
                    logging.warning(
                        f"QuantileNormalizer skipping key '{key}': data is not np.ndarray ({type(signal)})."
                    )
                    transformed_data[key] = signal
                    continue

                if key in self.background and key in self.scale_factor:
                    bg = self.background[key]
                    sf = self.scale_factor[key]

                    if sf == 0:
                        logging.warning(
                            f"QuantileNormalizer skipping key '{key}': scale factor is zero."
                        )
                        transformed_data[key] = (
                            signal  # Keep original or maybe set to NaN?
                        )
                    else:
                        # Apply normalization: (signal - background) / scale_factor
                        try:
                            # Ensure float calculation
                            normalized_signal = (signal.astype(np.float64) - bg) / sf
                            transformed_data[key] = normalized_signal
                        except TypeError:
                            logging.warning(
                                f"QuantileNormalizer cannot transform value for key '{key}' (type {type(signal)}). Keeping original."
                            )
                            transformed_data[key] = signal
                else:
                    logging.warning(
                        f"QuantileNormalizer skipping key '{key}': background or scale factor missing."
                    )
                    transformed_data[key] = signal  # Keep original data
            return transformed_data
        elif isinstance(data, np.ndarray):
            # Applying quantile norm to a single array is ambiguous without a key mapping.
            # We could try finding a 'default' key or averaging params, but it's safer to warn.
            logging.warning(
                "QuantileNormalizer applied to single np.ndarray is ambiguous. Returning original data."
            )
            return data
        elif isinstance(data, list):
            logging.warning(
                "QuantileNormalizer applied to list data is ambiguous. Returning original data."
            )
            return data
        else:
            logging.warning(
                f"QuantileNormalizer received unsupported data type: {type(data)}. Returning original data."
            )
            return data
