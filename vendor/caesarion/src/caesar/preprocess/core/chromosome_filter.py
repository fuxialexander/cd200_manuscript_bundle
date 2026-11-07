"""Centralized chromosome filtering logic for preprocessing."""

from __future__ import annotations

from typing import Any


class ChromosomeFilter:
    """Handles chromosome filtering based on configuration.

    Provides consistent chromosome filtering across all preprocessing steps.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize chromosome filter with configuration.

        Args:
            config: Configuration dictionary with optional 'chromosomes' section
        """
        # Default filtering rules
        self.defaults = {
            "max_length": 5,  # Maximum chromosome name length
            "exclude_patterns": ["Un", "random", "_alt", "_fix"],  # Patterns to exclude
            "exclude_exact": ["chrM", "chrY"],  # Exact names to exclude
            "include_patterns": [],  # If specified, only include chromosomes matching these
            "include_exact": [],  # If specified, only include these exact names
        }

        # Get chromosome config or use defaults
        chr_config = config.get("chromosomes", {})

        # Merge with defaults
        self.max_length = chr_config.get("max_length", self.defaults["max_length"])
        self.exclude_patterns = chr_config.get(
            "exclude_patterns", self.defaults["exclude_patterns"]
        )
        self.exclude_exact = set(
            chr_config.get("exclude_exact", self.defaults["exclude_exact"])
        )
        self.include_patterns = chr_config.get(
            "include_patterns", self.defaults["include_patterns"]
        )
        self.include_exact = set(
            chr_config.get("include_exact", self.defaults["include_exact"])
        )

        # Custom filter function (if provided as string, will need to eval - not recommended)
        self.custom_filter = chr_config.get("custom_filter", None)

    def is_valid_chromosome(self, chrom: str) -> bool:
        """Check if a chromosome should be included in processing.

        Args:
            chrom: Chromosome name

        Returns:
            True if chromosome should be processed
        """
        # If include_exact is specified, only those are valid
        if self.include_exact:
            return chrom in self.include_exact

        # If include_patterns is specified, must match at least one
        if self.include_patterns:
            if not any(pattern in chrom for pattern in self.include_patterns):
                return False

        # Check length
        if self.max_length and len(chrom) > self.max_length:
            return False

        # Check exact exclusions
        if chrom in self.exclude_exact:
            return False

        # Check pattern exclusions
        return not any(pattern in chrom for pattern in self.exclude_patterns)

    def filter_chromosomes(
        self, chromosomes: list[str], available_in_data: set[str] = None
    ) -> list[str]:
        """Filter a list of chromosomes based on configuration.

        Args:
            chromosomes: List of all possible chromosomes
            available_in_data: Optional set of chromosomes actually present in data

        Returns:
            Filtered list of chromosomes to process
        """
        # Start with all chromosomes
        filtered = []

        for chrom in chromosomes:
            # Check if chromosome is valid according to rules
            if not self.is_valid_chromosome(chrom):
                continue

            # If available_in_data is provided, check if chromosome exists
            if available_in_data and chrom not in available_in_data:
                continue

            filtered.append(chrom)

        return filtered

    def get_standard_chromosomes(self, assembly: str) -> list[str]:
        """Get standard chromosomes for a given assembly.

        Args:
            assembly: Genome assembly (e.g., 'hg38', 'mm10')

        Returns:
            List of standard chromosomes
        """
        if assembly.startswith("hg") or assembly.startswith("GRCh"):
            # Human chromosomes
            return [f"chr{i}" for i in range(1, 23)] + ["chrX"]
        elif assembly.startswith("mm") or assembly.startswith("GRCm"):
            # Mouse chromosomes
            return [f"chr{i}" for i in range(1, 20)] + ["chrX"]
        else:
            # Unknown assembly - return empty list
            return []


def get_chromosomes_to_process(
    config: dict[str, Any],
    all_chromosomes: list[str],
    available_chromosomes: set[str] = None,
) -> list[str]:
    """Convenience function to get filtered chromosomes.

    Args:
        config: Configuration dictionary
        all_chromosomes: List of all chromosomes from reference
        available_chromosomes: Optional set of chromosomes in data

    Returns:
        Filtered list of chromosomes to process
    """
    filter = ChromosomeFilter(config)
    return filter.filter_chromosomes(all_chromosomes, available_chromosomes)
