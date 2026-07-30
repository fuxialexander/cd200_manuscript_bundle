"""
Dataset management for mutation analysis.

Provides Variant dataclass and MutationDataset for managing genetic
variants and integrating with gcell genome interfaces.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator
import torch
import numpy as np
import pandas as pd
from caesar.io.zarr_io import ZarrIO


@dataclass(frozen=True)
class Variant:
    """
    Immutable representation of a genetic variant.

    Follows gcell conventions:
    - 0-based position indexing
    - Chromosome names with 'chr' prefix

    Attributes:
        chromosome: Chromosome name (e.g., 'chr1')
        position: 0-based start position
        ref: Reference allele
        alt: Alternative allele (None for reference-only queries)
        metadata: Additional variant information (p-values, scores, etc.)
    """
    chromosome: str
    position: int
    ref: str
    alt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable representation."""
        alt_str = f"->{self.alt}" if self.alt else ""
        # +1 for display (1-based coordinates)
        return f"{self.chromosome}:{self.position+1} {self.ref}{alt_str}"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"Variant(chromosome='{self.chromosome}', "
                f"position={self.position}, "
                f"ref='{self.ref}', "
                f"alt={self.alt!r})")


class MutationDataset:
    """
    Manages genetic variants and provides sequence retrieval.

    Integrates with gcell SequenceDenseZarrIO for genome access.
    Supports VCF, DataFrame, and programmatic variant creation.

    Example:
        >>> from caesar.io.zarr_io import SequenceDenseZarrIO
        >>> genome = SequenceDenseZarrIO("hg38.zarr", mode='r')
        >>> dataset = MutationDataset.from_vcf("variants.vcf", genome)
        >>> print(len(dataset))
        42
        >>> variant = dataset[0]
        >>> seq = dataset.get_sequence(variant.chromosome, variant.position-1000, variant.position+1000)
    """

    def __init__(self, genome_io: Any, variants: Optional[List[Variant]] = None, atac_io: Optional[Any] = None, cell_types: Optional[List[str]] = None):
        """
        Initialize dataset with genome interface and variants.

        Args:
            genome_io: gcell SequenceDenseZarrIO or compatible interface
                      Must support .get_track(chrom, start, end, output_format='raw_array')
            variants: Optional list of Variant objects
            atac_io: Optional gcell ZarrIO for ATAC data.
            cell_types: Optional list of cell types to select from ATAC data.
        """
        self.genome = genome_io
        self.variants = variants or []
        self.atac_io = atac_io
        self.cell_types = cell_types
        self._validate_genome_interface()
        if self.atac_io:
            self._validate_atac_interface()

    def _validate_genome_interface(self):
        """Validate that genome interface has required methods."""
        if not hasattr(self.genome, 'get_track'):
            raise ValueError(
                "genome_io must have get_track() method "
                "(e.g., caesar.io.zarr_io.SequenceDenseZarrIO)"
            )

    def _validate_atac_interface(self):
        """Validate that ATAC interface has required methods."""
        if not hasattr(self.atac_io, 'get_track'):
            raise ValueError(
                "atac_io must have get_track() method"
            )

    # ========== Constructors ==========

    @classmethod
    def from_vcf(cls, vcf_path: str, genome_io: Any) -> 'MutationDataset':
        """
        Load variants from VCF file.

        Args:
            vcf_path: Path to VCF file
            genome_io: gcell genome interface

        Returns:
            MutationDataset with loaded variants

        Example:
            >>> dataset = MutationDataset.from_vcf("variants.vcf", genome)
            >>> print(len(dataset))
            42
        """
        variants = []

        with open(vcf_path, 'r') as f:
            for line in f:
                # Skip header lines
                if line.startswith('#'):
                    continue

                # Parse VCF fields
                fields = line.strip().split('\t')
                if len(fields) < 5:
                    continue

                chrom = fields[0]
                pos = int(fields[1]) - 1  # Convert to 0-based
                ref = fields[3]
                alt = fields[4]

                # Ensure chr prefix
                if not chrom.startswith('chr'):
                    chrom = f'chr{chrom}'

                # Create variant
                variant = Variant(
                    chromosome=chrom,
                    position=pos,
                    ref=ref,
                    alt=alt
                )
                variants.append(variant)

        return cls(genome_io, variants)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, genome_io: Any,
                      chrom_col: str = 'Chromosome',
                      pos_col: str = 'VARIANT_POS',
                      ref_col: str = 'Ref',
                      alt_col: str = 'Alt') -> 'MutationDataset':
        """
        Load variants from DataFrame.

        Args:
            df: DataFrame with variant information
            genome_io: gcell genome interface
            chrom_col: Column name for chromosome
            pos_col: Column name for position
            ref_col: Column name for reference allele
            alt_col: Column name for alternative allele

        Returns:
            MutationDataset with loaded variants

        Example:
            >>> df = pd.DataFrame({
            ...     'Chromosome': ['chr1', 'chr2'],
            ...     'VARIANT_POS': [1000000, 2000000],
            ...     'Ref': ['A', 'C'],
            ...     'Alt': ['G', 'T']
            ... })
            >>> dataset = MutationDataset.from_dataframe(df, genome)
        """
        variants = []

        for _, row in df.iterrows():
            chrom = str(row[chrom_col])
            pos = int(row[pos_col]) - 1  # Assume 1-based input, convert to 0-based
            ref = str(row[ref_col])
            alt = str(row[alt_col])

            # Ensure chr prefix
            if not chrom.startswith('chr'):
                chrom = f'chr{chrom}'

            # Collect any additional metadata
            metadata = {k: v for k, v in row.items()
                       if k not in [chrom_col, pos_col, ref_col, alt_col]}

            variant = Variant(
                chromosome=chrom,
                position=pos,
                ref=ref,
                alt=alt,
                metadata=metadata
            )
            variants.append(variant)

        return cls(genome_io, variants)

    @classmethod
    def from_region(cls, chrom: str, start: int, end: int,
                   ref: str, alt: str, genome_io: Any) -> 'MutationDataset':
        """
        Create single variant programmatically.

        Args:
            chrom: Chromosome name (e.g., 'chr1')
            start: Start position (0-based)
            end: End position (0-based, exclusive)
            ref: Reference allele
            alt: Alternative allele
            genome_io: gcell genome interface

        Returns:
            MutationDataset with single variant

        Example:
            >>> dataset = MutationDataset.from_region(
            ...     'chr1', 1000000, 1000001, 'A', 'G', genome
            ... )
        """
        # Ensure chr prefix
        if not chrom.startswith('chr'):
            chrom = f'chr{chrom}'

        variant = Variant(
            chromosome=chrom,
            position=start,
            ref=ref,
            alt=alt
        )

        return cls(genome_io, [variant])

    # ========== Sequence Operations ==========

    def get_sequence(self, chrom: str, start: int, end: int) -> torch.Tensor:
        """
        Get sequence using gcell interface.

        Uses genome_io.get_track() directly - no wrapping.

        Args:
            chrom: Chromosome name
            start: Start position (0-based, inclusive)
            end: End position (0-based, exclusive)

        Returns:
            Tensor of shape (4, length) with one-hot encoding
            Bases ordered as [A, C, G, T]

        Example:
            >>> seq = dataset.get_sequence('chr1', 1000000, 1002048)
            >>> seq.shape
            torch.Size([4, 2048])
        """
        # Get sequence from genome
        seq_array = self.genome.get_track(
            chrom, start, end, output_format='raw_array'
        )

        # Convert to numpy if needed
        if not isinstance(seq_array, np.ndarray):
            seq_array = np.array(seq_array)

        # Ensure correct shape (4, length)
        if seq_array.shape[0] != 4:
            seq_array = seq_array.T

        # Convert to tensor
        return torch.tensor(seq_array.astype(np.float32))

    def get_atac_signals(self, chrom: str, start: int, end: int) -> Optional[torch.Tensor]:
        """
        Get ATAC-seq signals for a region.

        Args:
            chrom: Chromosome name
            start: Start position (0-based, inclusive)
            end: End position (0-based, exclusive)

        Returns:
            Tensor of shape (num_samples, length) with ATAC signals, or None if atac_io not set.
        """
        if not self.atac_io:
            return None

        try:
            # Use get_track with parameters from the original script
            track_data = self.atac_io.get_track(
                chrom, start, end,
                output_format='raw_dict',
                aggregator='by_key',
                normalize_factor=1e8,
                conv_size=20
            )

            # Stack data for the specified cell types
            if self.cell_types:
                atac_data = np.stack([track_data[k] for k in self.cell_types], axis=0)
            else:
                # Fallback if no cell types are specified (should not happen with CLI)
                atac_data = np.stack(list(track_data.values()), axis=0)
            
            # If the array has a sample dimension, squeeze it.
            if atac_data.ndim > 1:
                atac_data = atac_data.squeeze()

            # Convert to tensor
            return torch.tensor(atac_data.astype(np.float32))
        except Exception as e:
            print(f"Warning: Could not fetch ATAC data for {chrom}:{start}-{end}. Error: {e}")
            return None

    def apply_mutation(self, sequence: torch.Tensor, position: int,
                      ref: str, alt: str) -> torch.Tensor:
        """
        Apply SNV/indel to sequence tensor.

        Args:
            sequence: Input sequence tensor (4, length)
            position: Position within sequence (0-based)
            ref: Reference allele
            alt: Alternative allele

        Returns:
            Mutated sequence tensor (4, length)

        Example:
            >>> ref_seq = dataset.get_sequence('chr1', 1000000, 1002048)
            >>> alt_seq = dataset.apply_mutation(ref_seq, 1024, 'A', 'G')
        """
        # For now, only support SNVs (single nucleotide variants)
        if len(ref) != 1 or len(alt) != 1:
            raise NotImplementedError(
                f"Currently only SNVs supported. Got ref='{ref}', alt='{alt}'"
            )

        # Create mutated sequence
        mutated = sequence.clone()

        # Base to index mapping
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        # Validate alt base
        alt_upper = alt.upper()
        if alt_upper not in base_to_idx:
            raise ValueError(f"Invalid base: {alt}. Must be A, C, G, or T")

        # Zero out position and set new base
        mutated[:, position] = 0
        mutated[base_to_idx[alt_upper], position] = 1

        return mutated

    def get_variant_sequences(self, variant: Variant,
                             extend_bp: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get reference and alternative sequences for a variant.

        Args:
            variant: Variant to analyze
            extend_bp: Number of base pairs to extend around variant

        Returns:
            Tuple of (ref_seq, alt_seq) tensors, each shape (4, extend_bp)

        Example:
            >>> variant = dataset[0]
            >>> ref_seq, alt_seq = dataset.get_variant_sequences(variant)
            >>> ref_seq.shape
            torch.Size([4, 2048])
        """
        # Calculate window
        center = variant.position
        start = center - extend_bp // 2
        end = start + extend_bp

        # Get reference sequence
        ref_seq = self.get_sequence(variant.chromosome, start, end)

        # Create alternative sequence
        if variant.alt is None:
            # No mutation to apply
            alt_seq = ref_seq
        else:
            # Apply mutation
            position_in_window = extend_bp // 2
            alt_seq = self.apply_mutation(ref_seq, position_in_window,
                                         variant.ref, variant.alt)

        return ref_seq, alt_seq

    # ========== Iterator Protocol ==========

    def __getitem__(self, idx: int) -> Variant:
        """Get variant by index."""
        return self.variants[idx]

    def __len__(self) -> int:
        """Get number of variants."""
        return len(self.variants)

    def __iter__(self) -> Iterator[Variant]:
        """Iterate over variants."""
        return iter(self.variants)

    def __repr__(self) -> str:
        """String representation."""
        return f"MutationDataset(n_variants={len(self.variants)})"