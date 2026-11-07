"""Base processor that accepts full CaesarConfig.

This simplifies processor initialization by passing the full configuration
object and letting each processor extract what it needs.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

    from caesar.preprocess.core.config import CaesarConfig


class BaseProcessor(ABC):
    """Base class for all processors using unified configuration.

    Key improvements:
    1. Accepts full CaesarConfig object
    2. Direct attribute access without get_config_value()
    3. Type-safe configuration access
    4. Simplified initialization
    5. Optional input/output zarr paths that can be set by workflow
    """

    def __init__(self, config: CaesarConfig):
        """Initialize processor with full configuration.

        Args:
            config: Full CaesarConfig object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Common attributes from config
        self.assembly = config.assembly
        # Ensure output_dir is resolved as absolute path from current working directory
        self.output_dir = Path(config.output.dir).resolve()
        self.n_jobs = config.resources.n_jobs

        # Output configuration
        self.backend = config.output.backend
        self.compression = config.output.compression

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Timing attributes
        self._start_time: float | None = None

    @abstractmethod
    def process(self) -> dict[str, Any]:
        """Process data according to processor logic.

        Returns:
            Processing results dictionary
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """Validate processor-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def get_input_path(self) -> Path | None:
        """Get input path for processing.

        This method provides a unified way to resolve input paths from multiple sources:
        1. Workflow-provided input_path attribute
        2. Config data.input field
        3. Processor-specific default paths from config

        Returns:
            Resolved input path or None if no input specified
        """
        # First check if workflow set an input path
        if hasattr(self, "input_path") and self.input_path is not None:
            # Convert to Path if it's not already
            if isinstance(self.input_path, Path):
                return self.input_path
            return Path(self.input_path)

        # Then check config data.input
        if self.config.data.input:
            return Path(self.config.data.input)

        # Finally, check for processor-specific default
        return self._get_default_input_path()

    def _get_default_input_path(self) -> Path | None:
        """Get processor-specific default input path.

        Override in subclasses to provide processor-specific defaults.

        Returns:
            Default input path or None
        """
        return None

    def get_output_path(self, filename: str = "") -> Path:
        """Get output path for a given filename.

        If filename is empty, returns the output directory.

        Args:
            filename: Output filename (optional)

        Returns:
            Full output path
        """
        if filename:
            return self.output_dir / filename
        return self.output_dir

    def validate_input_path(
        self, path: Path | None = None, must_exist: bool = True
    ) -> Path:
        """Validate input path exists and is accessible.

        Args:
            path: Path to validate (if None, uses get_input_path())
            must_exist: Whether the path must exist (set to False for dry-run)

        Returns:
            Validated path

        Raises:
            ValueError: If path is invalid
        """
        if path is None:
            path = self.get_input_path()

        if not path:
            raise ValueError("No input path specified")

        path = Path(path)

        # In dry-run mode or when must_exist is False, just return the path
        if not must_exist:
            return path

        if not path.exists():
            raise ValueError(f"Input path not found: {path}")

        return path

    def is_dry_run(self) -> bool:
        """Check if this is a dry-run execution.

        Returns:
            True if in dry-run mode
        """
        return getattr(self, "_dry_run", False)

    def _start_processing(self) -> None:
        """Mark the start of processing."""
        self._start_time = time.time()
        self.logger.info(f"Starting {self.__class__.__name__}")

    def _end_processing(self) -> float:
        """Mark the end of processing and return runtime.

        Returns:
            Processing runtime in seconds
        """
        if self._start_time is None:
            return 0.0

        runtime = time.time() - self._start_time
        self.logger.info(f"Completed {self.__class__.__name__} in {runtime:.2f}s")
        return runtime

    @classmethod
    def get_command_name(cls) -> str:
        """Get the CLI command name for this processor.

        Default implementation converts class name to kebab-case.
        Override in subclasses for custom command names.

        Returns:
            Command name for CLI
        """
        name = cls.__name__
        if name.endswith("Processor"):
            name = name[:-9]  # Remove "Processor" suffix

        # Convert CamelCase to kebab-case
        result = ""
        for i, char in enumerate(name):
            if i > 0 and char.isupper() and not name[i - 1].isupper():
                result += "-"
            result += char.lower()

        return result

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add processor-specific CLI arguments.

        Default implementation adds common arguments.
        Override in subclasses for processor-specific args.

        Args:
            parser: ArgumentParser to add arguments to
        """
        # Common arguments for all processors
        parser.add_argument("-c", "--config", help="Configuration file (YAML)")
        parser.add_argument(
            "--export-config",
            metavar="PATH",
            help="Export resolved configuration to file",
        )
        parser.add_argument(
            "-n",
            "--n-jobs",
            type=int,
            default=1,
            help="Number of parallel jobs (default: 1)",
        )

    @classmethod
    def get_description(cls) -> str:
        """Get description for CLI help.

        Default implementation uses class docstring.
        Override in subclasses for custom descriptions.

        Returns:
            Description text for CLI help
        """
        if cls.__doc__:
            # Extract first line of docstring
            return cls.__doc__.split("\n")[0].strip()
        return f"{cls.__name__} processor"


class PreprocessingProcessor(BaseProcessor):
    """Base class for processors that convert raw input files to backend storage.

    These processors handle lists of raw input files (fragments, BAM, etc.)
    and create new backend storage (zarr or BPCells).
    """

    def get_input_files(self) -> list[Path]:
        """Get list of input files from configuration.

        Override in subclasses to extract specific file types.

        Returns:
            List of input file paths
        """
        return []

    def _get_default_input_path(self) -> Path | None:
        """Preprocessing processors don't have default input paths."""
        return None

    def _get_default_output_path(self) -> Path | None:
        """Get default output path from processor-specific config.

        Override in subclasses to provide processor-specific defaults.

        Returns:
            Default output path or None
        """
        return None


class TransformationProcessor(BaseProcessor):
    """Base class for processors that transform between backend storages.

    These processors take one backend storage as input and produce
    a different backend storage as output.
    """

    def _get_default_input_path(self) -> Path | None:
        """Get default input path from processor-specific config.

        Override in subclasses to provide processor-specific defaults.

        Returns:
            Default input path relative to output.dir
        """
        # Look for 'input' field in processor-specific config
        processor_name = self.get_command_name().replace("-", "_")
        processor_config = getattr(self.config, processor_name, None)

        if processor_config and hasattr(processor_config, "input"):
            return self.output_dir / processor_config.input

        return None

    def _get_default_output_path(self) -> Path | None:
        """Get default output path from processor-specific config.

        Override in subclasses to provide processor-specific defaults.

        Returns:
            Default output path relative to output.dir
        """
        # Look for 'output' field in processor-specific config
        processor_name = self.get_command_name().replace("-", "_")
        processor_config = getattr(self.config, processor_name, None)

        if processor_config and hasattr(processor_config, "output"):
            return self.output_dir / processor_config.output

        return None


class AnnotationProcessor(BaseProcessor):
    """Base class for processors that modify backend storage in-place.

    These processors add annotations or modify existing backend storage
    without creating new storage. Input and output paths are the same.
    """

    def get_target_path(self) -> Path:
        """Get target path for in-place modification.

        Returns:
            Path to backend storage to modify

        Raises:
            ValueError: If no target path specified
        """
        path = self.get_input_path()
        if not path:
            raise ValueError("No target path specified for annotation")
        return path

    def _get_default_input_path(self) -> Path | None:
        """Get default target path from processor-specific config.

        Override in subclasses to provide processor-specific defaults.

        Returns:
            Default target path relative to output.dir
        """
        # Look for 'input' field in processor-specific config
        processor_name = self.get_command_name().replace("-", "_")
        processor_config = getattr(self.config, processor_name, None)

        if processor_config and hasattr(processor_config, "input"):
            return self.output_dir / processor_config.input

        return None

    def _check_enabled(self, config_section: str | None = None) -> dict[str, Any] | None:
        """Check if this processor is enabled in configuration.
        
        This method provides a consistent way for annotation processors to check
        if they should be skipped based on the `enabled` flag in their config section.
        
        Args:
            config_section: Name of config section to check (e.g., "peaks", "clustering").
                          If None, automatically derives from processor command name.
        
        Returns:
            Dictionary with skipped status if disabled, None if enabled.
            The dictionary has structure:
            {
                "output_path": str,
                "status": "skipped",
                "message": str,
                "runtime": float
            }
        """
        # Auto-detect config section name if not provided
        if config_section is None:
            processor_name = self.get_command_name().replace("-", "_")
            # Map processor names to config section names
            # Some processors use different names (e.g., "library-size" -> "library_size")
            config_section = processor_name
        
        # Get the config section
        processor_config = getattr(self.config, config_section, None)
        
        # Check if enabled flag exists and is False
        if processor_config is not None and hasattr(processor_config, "enabled"):
            if not processor_config.enabled:
                self.logger.info(
                    f"Skipping {self.__class__.__name__} - disabled in configuration"
                )
                runtime = self._end_processing()
                try:
                    target_path = self.validate_input_path(self.get_target_path(), must_exist=False)
                except ValueError:
                    # If we can't get target path, use output_dir
                    target_path = self.output_dir
                
                return {
                    "output_path": str(target_path),
                    "status": "skipped",
                    "message": f"{self.__class__.__name__} disabled in configuration",
                    "runtime": runtime,
                }
        
        return None
