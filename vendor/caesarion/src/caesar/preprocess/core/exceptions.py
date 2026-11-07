"""Custom exceptions for preprocessing."""

from __future__ import annotations


class ProcessorError(Exception):
    """Base exception for processor errors."""

    pass


class ValidationError(ProcessorError):
    """Raised when configuration validation fails."""

    pass


class ProcessingError(ProcessorError):
    """Raised when processing fails."""

    pass


class InputError(ProcessorError):
    """Raised when input data is invalid."""

    pass


class OutputError(ProcessorError):
    """Raised when output cannot be written."""

    pass
