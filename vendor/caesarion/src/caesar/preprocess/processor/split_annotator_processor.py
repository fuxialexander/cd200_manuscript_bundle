"""Split annotator processor for Master Protocol evaluation scheme.

This processor annotates peak BED files with split assignments for rigorous
model evaluation, ensuring test sets are locked and never overlap with training data.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from caesar.preprocess.core.base import BaseProcessor
from caesar.preprocess.core.exceptions import ValidationError

if TYPE_CHECKING:
    from caesar.preprocess.core.config import CaesarConfig

logger = logging.getLogger(__name__)


class SplitAnnotatorProcessor(BaseProcessor):
    """Processor for annotating peaks with split assignments for Master Protocol evaluation.

    This processor creates boolean columns for each split type:
    - Test sets: interpolation, manifold, external benchmark
    - Baseline splits: random, chromosome, NR-only
    - Proposed method splits: pre-training (step 1), fine-tuning (step 2)
    """

    def __init__(self, config: CaesarConfig):
        """Initialize SplitAnnotatorProcessor.

        Args:
            config: Structured CaesarConfig instance
        """
        super().__init__(config)

        # Direct access to configuration
        self.split_config = config.split_annotator
        self.input_bed = Path(self.split_config.input_bed)
        self.cluster_file = Path(self.split_config.cluster_file)
        self.output_bed = Path(self.split_config.output_bed)
        self.random_seed = self.split_config.random_seed
        self.val_chromosome = self.split_config.val_chromosome
        self.interpolation_fraction = self.split_config.interpolation_fraction
        self.manifold_fraction = self.split_config.manifold_fraction
        self.train_val_ratio = self.split_config.train_val_ratio

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate split annotation configuration."""
        if not self.input_bed.exists():
            raise ValidationError(f"Input BED file not found: {self.input_bed}")

        if not self.cluster_file.exists():
            raise ValidationError(f"Cluster file not found: {self.cluster_file}")

        if not 0 < self.interpolation_fraction < 1:
            raise ValidationError(
                f"interpolation_fraction must be between 0 and 1, got: {self.interpolation_fraction}"
            )

        if not 0 < self.manifold_fraction < 1:
            raise ValidationError(
                f"manifold_fraction must be between 0 and 1, got: {self.manifold_fraction}"
            )

        if not 0 < self.train_val_ratio < 1:
            raise ValidationError(
                f"train_val_ratio must be between 0 and 1, got: {self.train_val_ratio}"
            )

    def process(self) -> dict[str, Any]:
        """Annotate peaks with split assignments.

        Returns:
            Processing results dictionary
        """
        self._start_processing()

        logger.info("Loading input BED file...")
        peaks_df = pd.read_csv(self.input_bed, sep="\t")

        # Initialize all boolean columns to False
        split_columns = [
            "is_test_interpolation",
            "is_test_manifold",
            "is_test_external",
            "is_train_baseline_a",
            "is_val_baseline_a",
            "is_train_baseline_b",
            "is_val_baseline_b",
            "is_train_baseline_c",
            "is_val_baseline_c",
            "is_train_proposed_step1",
            "is_val_proposed_step1",
            "is_train_proposed_step2",
            "is_val_proposed_step2",
        ]
        for col in split_columns:
            peaks_df[col] = False

        # Add sequence type columns
        peaks_df["is_nr_seed"] = False
        peaks_df["is_r_variant"] = False

        # Parse cluster file
        logger.info("Parsing cluster file...")
        clusters, nr_seeds, r_variants = self._parse_cluster_file()

        # Identify sequence types
        logger.info("Identifying sequence types...")
        peaks_df = self._identify_sequence_type(peaks_df, clusters, nr_seeds, r_variants)

        # Create golden test sets (highest priority)
        logger.info("Creating golden test sets...")
        peaks_df = self._create_golden_test_sets(peaks_df, clusters)

        # Create baseline splits
        logger.info("Creating baseline splits...")
        peaks_df = self._create_baseline_splits(peaks_df, clusters, nr_seeds, r_variants)

        # Create proposed method splits
        logger.info("Creating proposed method splits...")
        peaks_df = self._create_proposed_method_splits(
            peaks_df, clusters, nr_seeds, r_variants
        )

        # Validate no overlaps
        self._validate_no_overlaps(peaks_df)

        # Save annotated BED file
        logger.info(f"Saving annotated BED file to {self.output_bed}...")
        self.output_bed.parent.mkdir(parents=True, exist_ok=True)
        peaks_df.to_csv(self.output_bed, sep="\t", index=False)

        runtime = self._end_processing()

        # Count splits
        split_counts = {col: int(peaks_df[col].sum()) for col in split_columns}

        return {
            "output_path": str(self.output_bed),
            "input_peaks": len(peaks_df),
            "split_counts": split_counts,
            "runtime": runtime,
        }

    def _parse_cluster_file(self) -> tuple[dict[str, list[str]], set[str], set[str]]:
        """Parse mmseqs cluster TSV file.

        Returns:
            Tuple of (clusters dict, nr_seeds set, r_variants set)
        """
        clusters = defaultdict(list)
        nr_seeds = set()
        r_variants = set()

        with open(self.cluster_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    representative, member = parts
                    # Remove "::" prefix if present
                    representative = representative.lstrip("::")
                    member = member.lstrip("::")
                    clusters[representative].append(member)

        # Identify NR seeds (representatives) and R variants (members)
        for rep, members in clusters.items():
            nr_seeds.add(rep)
            for member in members:
                r_variants.add(member)

        logger.info(
            f"Parsed {len(clusters)} clusters: {len(nr_seeds)} NR seeds, "
            f"{len(r_variants)} R variants"
        )

        return clusters, nr_seeds, r_variants

    def _identify_sequence_type(
        self,
        peaks_df: pd.DataFrame,
        clusters: dict[str, list[str]],
        nr_seeds: set[str],
        r_variants: set[str],
    ) -> pd.DataFrame:
        """Identify sequence types and add to DataFrame.

        Args:
            peaks_df: Peaks DataFrame
            clusters: Cluster dictionary
            nr_seeds: Set of NR seed sequence IDs
            r_variants: Set of R variant sequence IDs

        Returns:
            DataFrame with is_nr_seed and is_r_variant columns
        """
        # Create sequence ID mapping
        peaks_df["seq_id"] = peaks_df.apply(
            lambda row: f"{row['Chromosome']}:{row['Start']}-{row['End']}", axis=1
        )
        seq_id_to_index = {seq_id: idx for idx, seq_id in enumerate(peaks_df["seq_id"])}

        # Map cluster representatives to NR seeds
        missing_seeds = 0
        for seed_id in nr_seeds:
            clean_seed_id = seed_id.split("::", 1)[-1] if "::" in seed_id else seed_id
            if clean_seed_id in seq_id_to_index:
                idx = seq_id_to_index[clean_seed_id]
                peaks_df.loc[idx, "is_nr_seed"] = True
            else:
                missing_seeds += 1

        # Map cluster members to R variants
        missing_variants = 0
        for variant_id in r_variants:
            clean_variant_id = (
                variant_id.split("::", 1)[-1] if "::" in variant_id else variant_id
            )
            if clean_variant_id in seq_id_to_index:
                idx = seq_id_to_index[clean_variant_id]
                peaks_df.loc[idx, "is_r_variant"] = True
            else:
                missing_variants += 1

        if missing_seeds > 0:
            logger.warning(
                f"{missing_seeds} NR seeds from cluster file not found in BED file"
            )
        if missing_variants > 0:
            logger.warning(
                f"{missing_variants} R variants from cluster file not found in BED file"
            )

        # Handle unclustered sequences (treat as singletons/NR seeds)
        unclustered = peaks_df[
            ~peaks_df["is_nr_seed"] & ~peaks_df["is_r_variant"]
        ].index
        if len(unclustered) > 0:
            logger.info(
                f"Found {len(unclustered)} unclustered sequences, treating as NR seeds"
            )
            peaks_df.loc[unclustered, "is_nr_seed"] = True

        return peaks_df

    def _create_golden_test_sets(
        self, peaks_df: pd.DataFrame, clusters: dict[str, list[str]]
    ) -> pd.DataFrame:
        """Create golden test sets (interpolation and manifold).

        Args:
            peaks_df: Peaks DataFrame
            clusters: Cluster dictionary

        Returns:
            DataFrame with test set annotations
        """
        np.random.seed(self.random_seed)

        # Create sequence ID to index mapping
        seq_id_to_idx = {
            seq_id: idx for idx, seq_id in enumerate(peaks_df["seq_id"])
        }

        # Interpolation Set: 10% from each cluster (excluding singletons)
        logger.info("Creating interpolation test set...")
        interpolation_indices = set()
        skipped_singletons = 0

        for rep, members in clusters.items():
            # Skip singleton clusters (members = [rep]) - nothing to interpolate
            # In mmseqs output, singletons have rep == member[0]
            if len(members) == 1 and members[0].lstrip("::") == rep.lstrip("::"):
                skipped_singletons += 1
                continue

            # Get all sequences in cluster (representative + members)
            cluster_seqs = [rep] + members
            valid_seqs = []
            for seq in cluster_seqs:
                clean_seq = seq.split("::", 1)[-1] if "::" in seq else seq
                if clean_seq in seq_id_to_idx:
                    valid_seqs.append(seq)

            if len(valid_seqs) == 0:
                continue

            # Sample 10% from clusters with 2+ members
            n_sample = max(1, int(len(valid_seqs) * self.interpolation_fraction))
            sampled = np.random.choice(valid_seqs, size=n_sample, replace=False)

            for seq_id in sampled:
                clean_seq_id = (
                    seq_id.split("::", 1)[-1] if "::" in seq_id else seq_id
                )
                if clean_seq_id in seq_id_to_idx:
                    interpolation_indices.add(seq_id_to_idx[clean_seq_id])

        peaks_df.loc[list(interpolation_indices), "is_test_interpolation"] = True
        logger.info(
            f"  Interpolation set: {len(interpolation_indices)} sequences "
            f"({len(interpolation_indices)/len(peaks_df)*100:.1f}%)"
        )
        logger.info(
            f"  Skipped {skipped_singletons} singleton clusters (no interpolation needed)"
        )

        # Manifold Set: 50% from largest cluster
        logger.info("Creating manifold test set...")
        largest_cluster_rep = None
        largest_cluster_size = 0

        for rep, members in clusters.items():
            cluster_size = len(members) + 1  # +1 for representative
            if cluster_size > largest_cluster_size:
                largest_cluster_size = cluster_size
                largest_cluster_rep = rep

        if largest_cluster_rep is not None:
            largest_cluster_seqs = [largest_cluster_rep] + clusters[largest_cluster_rep]
            valid_seqs = []
            for seq in largest_cluster_seqs:
                clean_seq = seq.split("::", 1)[-1] if "::" in seq else seq
                if clean_seq in seq_id_to_idx:
                    valid_seqs.append(seq)

            if len(valid_seqs) > 0:
                n_sample = max(1, int(len(valid_seqs) * self.manifold_fraction))
                sampled = np.random.choice(valid_seqs, size=n_sample, replace=False)

                manifold_indices = set()
                for seq_id in sampled:
                    clean_seq_id = (
                        seq_id.split("::", 1)[-1] if "::" in seq_id else seq_id
                    )
                    if clean_seq_id in seq_id_to_idx:
                        idx = seq_id_to_idx[clean_seq_id]
                        # Ensure no overlap with interpolation set
                        if idx not in interpolation_indices:
                            manifold_indices.add(idx)

                peaks_df.loc[list(manifold_indices), "is_test_manifold"] = True
                logger.info(
                    f"  Manifold set: {len(manifold_indices)} sequences "
                    f"({len(manifold_indices)/len(peaks_df)*100:.1f}%)"
                )

        return peaks_df

    def _create_baseline_splits(
        self,
        peaks_df: pd.DataFrame,
        clusters: dict[str, list[str]],
        nr_seeds: set[str],
        r_variants: set[str],
    ) -> pd.DataFrame:
        """Create baseline training/validation splits.

        Args:
            peaks_df: Peaks DataFrame
            clusters: Cluster dictionary
            nr_seeds: Set of NR seed sequence IDs
            r_variants: Set of R variant sequence IDs

        Returns:
            DataFrame with baseline split annotations
        """
        np.random.seed(self.random_seed)

        # Exclude all test sets from pool
        test_mask = (
            peaks_df["is_test_interpolation"]
            | peaks_df["is_test_manifold"]
            | peaks_df["is_test_external"]
        )
        train_pool = peaks_df[~test_mask].copy()

        # Baseline A: Random split
        logger.info("Creating Baseline A (Random) splits...")
        train_pool_shuffled = train_pool.sample(frac=1, random_state=self.random_seed)
        split_idx = int(len(train_pool_shuffled) * self.train_val_ratio)
        train_indices = train_pool_shuffled.index[:split_idx]
        val_indices = train_pool_shuffled.index[split_idx:]

        peaks_df.loc[train_indices, "is_train_baseline_a"] = True
        peaks_df.loc[val_indices, "is_val_baseline_a"] = True
        logger.info(
            f"  Baseline A: {len(train_indices)} train, {len(val_indices)} val"
        )

        # Baseline B: Chromosome split
        logger.info(f"Creating Baseline B (Chromosome: {self.val_chromosome}) splits...")
        train_pool_chr = train_pool.copy()
        val_mask = train_pool_chr["Chromosome"] == self.val_chromosome
        train_mask = ~val_mask

        peaks_df.loc[train_pool_chr[train_mask].index, "is_train_baseline_b"] = True
        peaks_df.loc[train_pool_chr[val_mask].index, "is_val_baseline_b"] = True
        logger.info(
            f"  Baseline B: {train_mask.sum()} train, {val_mask.sum()} val"
        )

        # Baseline C: NR-only split (use 95-5 split to maximize training data)
        logger.info("Creating Baseline C (NR-only) splits...")
        train_pool_nr = train_pool[train_pool["is_nr_seed"]].copy()
        train_pool_nr_shuffled = train_pool_nr.sample(
            frac=1, random_state=self.random_seed
        )
        # Use 95-5 split for NR-only (more data needed since NR set is smaller)
        nr_train_ratio = 0.95
        split_idx = int(len(train_pool_nr_shuffled) * nr_train_ratio)
        train_indices = train_pool_nr_shuffled.index[:split_idx]
        val_indices = train_pool_nr_shuffled.index[split_idx:]

        peaks_df.loc[train_indices, "is_train_baseline_c"] = True
        peaks_df.loc[val_indices, "is_val_baseline_c"] = True
        logger.info(
            f"  Baseline C: {len(train_indices)} train, {len(val_indices)} val (95-5 split)"
        )

        return peaks_df

    def _create_proposed_method_splits(
        self,
        peaks_df: pd.DataFrame,
        clusters: dict[str, list[str]],
        nr_seeds: set[str],
        r_variants: set[str],
    ) -> pd.DataFrame:
        """Create proposed method splits (curriculum learning).

        Args:
            peaks_df: Peaks DataFrame
            clusters: Cluster dictionary
            nr_seeds: Set of NR seed sequence IDs
            r_variants: Set of R variant sequence IDs

        Returns:
            DataFrame with proposed method split annotations
        """
        np.random.seed(self.random_seed)

        # Exclude all test sets from pool
        test_mask = (
            peaks_df["is_test_interpolation"]
            | peaks_df["is_test_manifold"]
            | peaks_df["is_test_external"]
        )
        train_pool = peaks_df[~test_mask].copy()

        # Step 1: Pre-training (same as Baseline C, use 95-5 split)
        logger.info("Creating Proposed Method Step 1 (Pre-training) splits...")
        train_pool_nr = train_pool[train_pool["is_nr_seed"]].copy()
        train_pool_nr_shuffled = train_pool_nr.sample(
            frac=1, random_state=self.random_seed
        )
        # Use 95-5 split for NR-only pre-training
        nr_train_ratio = 0.95
        split_idx = int(len(train_pool_nr_shuffled) * nr_train_ratio)
        train_indices_step1 = train_pool_nr_shuffled.index[:split_idx]
        val_indices_step1 = train_pool_nr_shuffled.index[split_idx:]

        peaks_df.loc[train_indices_step1, "is_train_proposed_step1"] = True
        peaks_df.loc[val_indices_step1, "is_val_proposed_step1"] = True
        logger.info(
            f"  Step 1: {len(train_indices_step1)} train, {len(val_indices_step1)} val (95-5 split)"
        )

        # Step 2: Fine-tuning (R variants belonging to Step 1 training seeds)
        logger.info("Creating Proposed Method Step 2 (Fine-tuning) splits...")

        # Get training seed sequence IDs from Step 1
        train_seeds_step1 = peaks_df.loc[train_indices_step1, "seq_id"].tolist()
        train_seeds_step1_set = set(train_seeds_step1)

        # Find R variants that belong to these seeds
        step2_variants = []
        pairing_info = {}

        for rep, members in clusters.items():
            clean_rep = rep.split("::", 1)[-1] if "::" in rep else rep
            if clean_rep in train_seeds_step1_set:
                # This seed is in Step 1 training set, include its variants
                for member in members:
                    clean_member = (
                        member.split("::", 1)[-1] if "::" in member else member
                    )
                    step2_variants.append(clean_member)
                    pairing_info[clean_member] = clean_rep

        # Filter to variants that exist in DataFrame
        step2_indices = []
        for variant_id in step2_variants:
            variant_mask = train_pool["seq_id"] == variant_id
            if variant_mask.any():
                step2_indices.extend(train_pool[variant_mask].index.tolist())

        if len(step2_indices) > 0:
            # Split Step 2 variants into train/val
            step2_df = train_pool.loc[step2_indices].copy()
            step2_shuffled = step2_df.sample(frac=1, random_state=self.random_seed)
            split_idx = int(len(step2_shuffled) * self.train_val_ratio)
            train_indices_step2 = step2_shuffled.index[:split_idx]
            val_indices_step2 = step2_shuffled.index[split_idx:]

            peaks_df.loc[train_indices_step2, "is_train_proposed_step2"] = True
            peaks_df.loc[val_indices_step2, "is_val_proposed_step2"] = True

            # Store pairing info in DataFrame attributes
            peaks_df.attrs["pairing_info"] = pairing_info

            logger.info(
                f"  Step 2: {len(train_indices_step2)} train, {len(val_indices_step2)} val"
            )
        else:
            logger.warning("  Step 2: No R variants found for Step 1 training seeds")

        return peaks_df

    def _validate_no_overlaps(self, peaks_df: pd.DataFrame) -> None:
        """Validate that test sets don't overlap with training sets.

        Args:
            peaks_df: Annotated peaks DataFrame

        Raises:
            ValidationError: If overlaps are detected
        """
        # Check test sets don't overlap with each other
        test_cols = [
            "is_test_interpolation",
            "is_test_manifold",
            "is_test_external",
        ]
        test_overlaps = []
        for i, col1 in enumerate(test_cols):
            for col2 in test_cols[i + 1 :]:
                overlap = (peaks_df[col1] & peaks_df[col2]).sum()
                if overlap > 0:
                    test_overlaps.append(f"{col1} and {col2}: {overlap} sequences")

        if test_overlaps:
            raise ValidationError(
                f"Test set overlaps detected: {', '.join(test_overlaps)}"
            )

        # Check test sets don't overlap with training sets
        test_mask = (
            peaks_df["is_test_interpolation"]
            | peaks_df["is_test_manifold"]
            | peaks_df["is_test_external"]
        )

        train_cols = [
            "is_train_baseline_a",
            "is_train_baseline_b",
            "is_train_baseline_c",
            "is_train_proposed_step1",
            "is_train_proposed_step2",
        ]

        train_test_overlaps = []
        for col in train_cols:
            overlap = (test_mask & peaks_df[col]).sum()
            if overlap > 0:
                train_test_overlaps.append(f"{col} and test sets: {overlap} sequences")

        if train_test_overlaps:
            raise ValidationError(
                f"Train/test overlaps detected: {', '.join(train_test_overlaps)}"
            )

        logger.info("Validation passed: No overlaps detected")

