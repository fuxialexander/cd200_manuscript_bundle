"""Processor factory for automatic processor selection and CLI generation."""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields
from typing import TYPE_CHECKING, get_type_hints

from caesar.preprocess.core.exceptions import ProcessingError

if TYPE_CHECKING:
    from caesar.preprocess.core.base import BaseProcessor
    from caesar.preprocess.core.config import CaesarConfig


class ProcessorFactory:
    """Factory for creating processors and managing CLI registration."""

    _processors: dict[str, type[BaseProcessor]] = {}

    @classmethod
    def register(cls, command_name: str, processor_class: type[BaseProcessor]) -> None:
        """Register a processor class with a command name.

        Args:
            command_name: CLI command name (e.g., 'fragments', 'bam')
            processor_class: Processor class to register
        """
        cls._processors[command_name] = processor_class

    @classmethod
    def create(cls, command: str, config: CaesarConfig) -> BaseProcessor:
        """Create a processor instance for the given command.

        Args:
            command: CLI command name
            config: Full CaesarConfig object

        Returns:
            Processor instance

        Raises:
            ProcessingError: If command is not registered
        """
        if command not in cls._processors:
            available = ", ".join(cls._processors.keys())
            raise ProcessingError(
                f"Unknown command '{command}'. Available: {available}"
            )

        processor_class = cls._processors[command]
        return processor_class(config)

    @classmethod
    def get_registered_commands(cls) -> list[str]:
        """Get list of all registered command names."""
        return list(cls._processors.keys())

    @classmethod
    def get_processor_class(cls, command: str) -> type[BaseProcessor]:
        """Get processor class for a command.

        Args:
            command: CLI command name

        Returns:
            Processor class

        Raises:
            ProcessingError: If command is not registered
        """
        if command not in cls._processors:
            available = ", ".join(cls._processors.keys())
            raise ProcessingError(
                f"Unknown command '{command}'. Available: {available}"
            )

        return cls._processors[command]


def add_dataclass_args(
    parser: argparse.ArgumentParser, config_class: type, prefix: str = ""
) -> None:
    """Add CLI arguments from a dataclass.

    Args:
        parser: ArgumentParser to add arguments to
        config_class: Dataclass to introspect
        prefix: Prefix for argument names (e.g., 'fragments.')
    """
    try:
        field_list = fields(config_class)
        type_hints = get_type_hints(config_class)
    except Exception as e:
        logging.warning(f"Could not introspect {config_class}: {e}")
        return

    for field in field_list:
        field_name = field.name
        field_type = type_hints.get(field_name, field.type)

        # Skip complex types and private fields
        if field_name.startswith("_") or hasattr(field_type, "__origin__"):
            continue

        # Create argument name
        arg_name = f"--{prefix}{field_name.replace('_', '-')}"

        # Get help text from field metadata or default
        help_text = field.metadata.get("help", f"Set {field_name}")

        # Add argument based on type
        if field_type == bool:
            parser.add_argument(
                arg_name, action="store_true", help=help_text, default=argparse.SUPPRESS
            )
        elif field_type == int:
            parser.add_argument(
                arg_name, type=int, help=help_text, default=argparse.SUPPRESS
            )
        elif field_type == float:
            parser.add_argument(
                arg_name, type=float, help=help_text, default=argparse.SUPPRESS
            )
        elif field_type == str:
            parser.add_argument(
                arg_name, type=str, help=help_text, default=argparse.SUPPRESS
            )
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            # Handle list types
            parser.add_argument(
                arg_name, nargs="+", help=help_text, default=argparse.SUPPRESS
            )


def register_all_processors() -> None:
    """Register all available processors with the factory."""
    # Import processor classes that are working
    processor_classes = []

    # Try to import each processor individually and only register working ones
    processor_imports = [
        ("caesar.preprocess.processor.bam_processor", "BAMProcessor"),
        ("caesar.preprocess.processor.library_size_processor", "LibrarySizeProcessor"),
        ("caesar.preprocess.processor.fragment_processor", "FragmentProcessor"),
        ("caesar.preprocess.processor.clustering_processor", "ClusteringProcessor"),
        ("caesar.preprocess.processor.footprint_processor", "FootprintProcessor"),
        (
            "caesar.preprocess.processor.motif_curator_processor",
            "NonRedundantMotifCuratorProcessor",
        ),
        ("caesar.preprocess.processor.motif_processor", "NonRedundantMotifProcessor"),
        (
            "caesar.preprocess.processor.normalization_processor",
            "NormalizationProcessor",
        ),
        ("caesar.preprocess.processor.peak_calling_processor", "PeakCallingProcessor"),
        ("caesar.preprocess.processor.pseudobulk_processor", "PseudobulkProcessor"),
        ("caesar.preprocess.processor.bpcells_processor", "BPCellsProcessor"),
        ("caesar.preprocess.processor.bigwig_processor", "BigWigProcessor"),
        ("caesar.preprocess.processor.multiome_processor", "MultiomeProcessor"),
        ("caesar.preprocess.processor.multiome_rna_processor", "MultiomeRNAProcessor"),
        (
            "caesar.preprocess.processor.multiome_pseudobulk_processor",
            "MultiomePseudobulkProcessor",
        ),
        ("caesar.preprocess.processor.add_peak_processor", "AddPeakProcessor"),
        ("caesar.preprocess.processor.add_expression", "AddExpressionProcessor"),
        (
            "caesar.preprocess.processor.metadata_filter_processor",
            "MetadataFilterProcessor",
        ),
        (
            "caesar.preprocess.processor.isolibsize_metacell_processor",
            "IsoLibsizeMetacellProcessor",
        ),
        (
            "caesar.preprocess.processor.tss_accessibility_processor",
            "TSSAccessibilityProcessor",
        ),
        (
            "caesar.preprocess.processor.tes_accessibility_processor",
            "TESAccessibilityProcessor",
        ),
        (
            "caesar.preprocess.processor.cnv_detection_processor",
            "CNVDetectionProcessor",
        ),
        (
            "caesar.preprocess.processor.cellstate_tile_count_processor",
            "CellStateTileCountProcessor",
        ),
    ]

    for module_name, class_name in processor_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            processor_class = getattr(module, class_name)
            processor_classes.append(processor_class)
            logging.debug(f"Successfully imported {class_name}")
        except Exception as e:
            logging.debug(f"Failed to import {class_name}: {e}")
            continue

    # Register processors using their get_command_name() method
    for processor_class in processor_classes:
        try:
            command_name = processor_class.get_command_name()
            ProcessorFactory.register(command_name, processor_class)
            logging.debug(f"Registered processor: {command_name}")
        except Exception as e:
            logging.debug(f"Failed to register {processor_class.__name__}: {e}")
