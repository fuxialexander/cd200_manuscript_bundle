"""
Filename: dna_atac_zarr_dataset.py
Purpose: Memory-efficient dataset for loading DNA sequences and ATAC signals directly from zarr files

Main Components:
- DNAATACZarrDataset: PyTorch dataset that loads data from zarr files during training
- Support for train/validation/test splits with reproducible random splitting
- Data augmentation: random sequence shifts and reverse complement transformations
- Efficient caching system for improved training performance
- Integration with zarr-based data storage for large-scale genomic data

Dependencies: torch, numpy, pandas, os
"""
from __future__ import annotations

from caesar.io.interface import DictNormalizer
import torch
import numpy as np
import pandas as pd
import os

class DNAATACZarrDataset(torch.utils.data.Dataset):
    """
    Dataset that loads DNA->ATAC data directly from zarr files during training.
    Memory efficient - no pre-curation needed.
    Includes data augmentations: random shift and reverse complement.
    """
    def __init__(
        self,
        cdz_object,
        genome_object,
        summits_csv_path: str = None,
        summits_df: pd.DataFrame = None,
        celltype_ids: list[str] = None,
        extend_bp: int = 2048,
        max_regions: int = None,
        split: str = 'train',
        train_split: float = 0.75,
        val_split: float = 0.15,
        test_split: float = 0.10,
        augment: bool = True,
        random_shift_bp: int = 100,
        aug_prob: float = 0,
        cache_dir: str = None,
        use_curriculum_learning: bool = False,
        curriculum_groups: int = 5,
        curriculum_column: str = "MaxValue",
        curriculum_initial_epsilon: float = 0.2,
        curriculum_min_groups: int = 5,
        val_chr: str = None,
        test_chr: str = None,
        borzoi_fold: bool = False,
        borzoi_val_fold: str = "fold6,fold7",
        sequences_human_bed: str = "sequences_human.bed"
    ):
        """
        Args:
            cdz_object: Pre-initialized CelltypeDenseZarrIO object (already subset to celltype_ids)
            genome_object: Pre-initialized SequenceDenseZarrIO object
            summits_csv_path: Path to pre-computed summits CSV file (either this or summits_df must be provided)
            summits_df: Pre-loaded summits DataFrame (either this or summits_csv_path must be provided)
            celltype_ids: List of celltype IDs to include
            extend_bp: Base pairs to extend around summits (default: 2048)
            max_regions: Maximum number of regions to include (default: None)
            split: Which split to use ('train', 'val', 'test')
            train_split: Fraction for training (default: 0.75)
            val_split: Fraction for validation (default: 0.15)
            test_split: Fraction for test (default: 0.10)
            augment: Whether to apply augmentations (default: True)
            random_shift_bp: Maximum random shift in bp (default: 100)
            aug_prob: Probability of applying each augmentation (default: 0.5)
            cache_dir: Directory to save/load cached .pt files (default: None, no caching)
            use_curriculum_learning: Enable curriculum learning based on specified column quantiles (default: False)
            curriculum_groups: Number of groups to split curriculum column into for curriculum learning (default: 5)
            curriculum_column: Column name to use for curriculum learning (default: "MaxValue")
            val_chr: Chromosome to use for validation set (overrides val_split ratio)
            test_chr: Chromosome to use for test set (overrides test_split ratio)
            borzoi_fold: Enable Borzoi-style fold-based splitting (default: False)
            borzoi_val_fold: Comma-separated validation folds (default: "fold6,fold7")
            sequences_human_bed: Path to sequences_human.bed file for fold definitions (default: "sequences_human.bed")
        """
        # Validate input parameters
        if summits_csv_path is None and summits_df is None:
            raise ValueError("Either summits_csv_path or summits_df must be provided")
        if summits_csv_path is not None and summits_df is not None:
            raise ValueError("Only one of summits_csv_path or summits_df should be provided")

        self.extend_bp = extend_bp
        self.celltype_ids = celltype_ids
        self.split = split
        self.augment = augment and (split == 'train')  # Only augment training data
        self.random_shift_bp = random_shift_bp
        self.aug_prob = aug_prob
        self.cache_dir = cache_dir
        self.max_regions = max_regions  # Store for cache key generation
        self.use_curriculum_learning = use_curriculum_learning
        self.curriculum_groups = curriculum_groups
        self.curriculum_column = curriculum_column
        self.curriculum_initial_epsilon = curriculum_initial_epsilon
        self.curriculum_min_groups = curriculum_min_groups
        self.val_chr = val_chr
        self.test_chr = test_chr
        self.borzoi_fold = borzoi_fold
        self.borzoi_val_fold = borzoi_val_fold
        self.sequences_human_bed = sequences_human_bed
        
        # Use pre-initialized zarr objects
        self.cdz = cdz_object
        self.genome = genome_object
        
        # Load summits from either CSV or DataFrame
        if summits_df is not None:
            # Use provided DataFrame
            summits = summits_df.copy()
            has_header = True  # DataFrame always has column names
            print(f"Using provided DataFrame with {len(summits)} regions and {len(summits.columns)} columns")
        else:
            # Load from CSV file
            print(f"Loading summits from {summits_csv_path}")

            # Try to detect if file has header and number of columns
            first_line = pd.read_csv(summits_csv_path, sep='\t', header=None, nrows=1)

            # Check if first line contains strings that indicate header
            has_header = any(isinstance(val, str) and not val.startswith('chr') for val in first_line.iloc[0])

            if has_header:
                # File has header, read with header=0
                summits = pd.read_csv(summits_csv_path, sep='\t', header=0)
                print(f"Detected header format with {len(summits.columns)} columns")
                # For compatibility, ensure we have the basic columns
                if 'Chromosome' not in summits.columns:
                    summits.columns = ['Chromosome', 'Start', 'End'] + list(summits.columns[3:])
            else:
                # Original logic for headerless files
                has_max_value = len(first_line.columns) >= 4
                if has_max_value:
                    summits = pd.read_csv(summits_csv_path, sep='\t', header=None,
                                        names=['Chromosome', 'Start', 'End', 'MaxValue'])
                    print("Detected 4-column format with MaxValue column")
                else:
                    summits = pd.read_csv(summits_csv_path, sep='\t', header=None,
                                        names=['Chromosome', 'Start', 'End'])

        # Ensure basic columns are correct types
        summits['Start'] = summits['Start'].astype(int)
        summits['End'] = summits['End'].astype(int)

        # Apply max_regions limiting before split (to limit overall dataset size)
        if max_regions and len(summits) > max_regions:
            print(f"Limiting to {max_regions} regions...")
            summits = summits.sample(n=max_regions, random_state=42).reset_index(drop=True)
        
        # Sort summits for reproducibility
        summits = summits.sort_values(['Chromosome', 'Start']).reset_index(drop=True)
        
        # Create train/val/test splits - use fold-based if specified
        if self.borzoi_fold:
            # Borzoi fold-based splitting
            self.regions = self._create_fold_based_split(summits, split)
        elif self.val_chr or self.test_chr:
            # Chromosome-based splitting
            if split == 'train':
                # Training gets all chromosomes except val_chr and test_chr
                mask = True
                if self.val_chr:
                    mask = mask & (summits['Chromosome'] != self.val_chr)
                if self.test_chr:
                    mask = mask & (summits['Chromosome'] != self.test_chr)
                self.regions = summits[mask].reset_index(drop=True)
            elif split == 'val':
                if not self.val_chr:
                    raise ValueError("val_chr must be specified when using chromosome-based splits")
                self.regions = summits[summits['Chromosome'] == self.val_chr].reset_index(drop=True)
            elif split == 'test':
                if not self.test_chr:
                    raise ValueError("test_chr must be specified when using chromosome-based splits")
                self.regions = summits[summits['Chromosome'] == self.test_chr].reset_index(drop=True)
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        else:
            # Random splitting (original behavior)
            n_regions = len(summits)
            # Use a local random state instead of global state for reproducibility
            rng = np.random.RandomState(42)
            indices = rng.permutation(n_regions)
            
            n_train = int(n_regions * train_split)
            n_val = int(n_regions * val_split)
            
            if split == 'train':
                split_indices = indices[:n_train]
            elif split == 'val':
                split_indices = indices[n_train:n_train + n_val]
            elif split == 'test':
                split_indices = indices[n_train + n_val:]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
            # Store regions for this split
            self.regions = summits.iloc[split_indices].reset_index(drop=True)
        
        # IMPORTANT: Curriculum learning (upsampling) must happen AFTER splitting
        # to prevent data leakage. Upsampled regions should only exist in training set.
        # Validation and test sets must contain only unique, non-duplicated regions.
        has_sufficient_columns = has_header or len(self.regions.columns) >= 4

        # Store pre-upsampling region count for validation
        pre_upsampling_count = len(self.regions)

        # Curriculum upsampling ONLY for training split
        if self.use_curriculum_learning and has_sufficient_columns:
            if split == 'train':
                self.regions = self._apply_curriculum_learning(
                    self.regions,
                    initial_epsilon=self.curriculum_initial_epsilon,
                    min_groups=self.curriculum_min_groups
                )
                # Ensure indices are clean after upsampling
                self.regions = self.regions.reset_index(drop=True)
            else:
                # Defensive check: val/test should NEVER have curriculum upsampling
                print(f"Curriculum learning disabled for {split} split (preventing data leakage)")

        print(f"Created {split} dataset with {len(self.regions)} regions (pre-upsampling: {pre_upsampling_count})")
        
        # Create chromosome encoding
        unique_chromosomes = sorted(self.regions['Chromosome'].unique())
        self.chromosome_to_idx = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}
        self.idx_to_chromosome = {idx: chrom for chrom, idx in self.chromosome_to_idx.items()}
        print(f"Chromosome mapping: {self.chromosome_to_idx}")
        
        # Print split information
        if self.borzoi_fold:
            val_folds = [f.strip() for f in self.borzoi_val_fold.split(',')]
            if split == 'train':
                print(f"Borzoi fold-based split - training excludes validation folds: {val_folds}")
            elif split in ['val', 'test']:
                print(f"Borzoi fold-based split - {split} from validation folds: {val_folds}")
        elif self.val_chr or self.test_chr:
            if split == 'train':
                excluded_chrs = []
                if self.val_chr:
                    excluded_chrs.append(f"val: {self.val_chr}")
                if self.test_chr:
                    excluded_chrs.append(f"test: {self.test_chr}")
                print(f"Chromosome-based split - training excludes: {', '.join(excluded_chrs)}")
            elif split == 'val':
                print(f"Chromosome-based split - validation from: {self.val_chr}")
            elif split == 'test':
                print(f"Chromosome-based split - test from: {self.test_chr}")
        else:
            print(f"Random split - train: {n_train}, val: {n_val}, test: {n_regions - n_train - n_val}")
        
        # Initialize cache-related attributes
        self.cached_data = None
        self.use_cache = self.cache_dir is not None
        
        # Try to load from cache if enabled
        if self.use_cache:
            self._setup_cache_path()
            self._try_load_cache()
    
    def _create_fold_based_split(self, summits, split):
        """
        Create fold-based train/val splits using pyranges for overlap detection.
        
        Args:
            summits: DataFrame with summit regions
            split: 'train', 'val', or 'test' (note: test defaults to val for fold-based)
        
        Returns:
            Filtered summits DataFrame for the specified split
        """
        import pyranges as pr
        import os
        
        # Check if sequences_human.bed exists
        if not os.path.exists(self.sequences_human_bed):
            raise FileNotFoundError(f"sequences_human.bed file not found: {self.sequences_human_bed}")
        
        print(f"Loading fold definitions from: {self.sequences_human_bed}")
        
        # Load fold definitions
        fold_df = pd.read_csv(self.sequences_human_bed, sep='\t', header=None, 
                             names=['Chromosome', 'Start', 'End', 'Fold'])
        
        # Parse validation folds
        val_folds = [f.strip() for f in self.borzoi_val_fold.split(',')]
        print(f"Validation folds: {val_folds}")
        
        # Create PyRanges objects with explicit index preservation
        summits_with_idx = summits.copy()
        summits_with_idx['original_idx'] = range(len(summits))
        summits_pr = pr.PyRanges(summits_with_idx[['Chromosome', 'Start', 'End', 'original_idx']])
        
        if split in ['train']:
            # Training uses all folds EXCEPT validation folds
            train_fold_df = fold_df[~fold_df['Fold'].isin(val_folds)].copy()
            train_fold_pr = pr.PyRanges(train_fold_df)
            
            # Find summits that overlap with training folds
            overlap = summits_pr.overlap(train_fold_pr)
            
            if len(overlap) == 0:
                print("Warning: No summits overlap with training folds")
                return summits.iloc[:0].copy()  # Return empty DataFrame with same structure
            
            # Get unique original indices from overlap result
            overlap_indices = sorted(set(overlap.df['original_idx'].tolist()))
            filtered_summits = summits.iloc[overlap_indices].copy()
            
            print(f"Found {len(filtered_summits)} summits in training folds (excluding {val_folds})")
            
        elif split in ['val', 'test']:  # Use validation folds for both val and test
            # Validation uses specified validation folds
            val_fold_df = fold_df[fold_df['Fold'].isin(val_folds)].copy()
            val_fold_pr = pr.PyRanges(val_fold_df)
            
            # Find summits that overlap with validation folds
            overlap = summits_pr.overlap(val_fold_pr)
            
            if len(overlap) == 0:
                print(f"Warning: No summits overlap with validation folds {val_folds}")
                return summits.iloc[:0].copy()  # Return empty DataFrame with same structure
            
            # Get unique original indices from overlap result
            overlap_indices = sorted(set(overlap.df['original_idx'].tolist()))
            filtered_summits = summits.iloc[overlap_indices].copy()
            
            print(f"Found {len(filtered_summits)} summits in validation folds {val_folds}")
            
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        return filtered_summits.reset_index(drop=True)
    
    def _apply_curriculum_learning(self, summits, initial_epsilon=0.2, min_groups=5):
        """
        Apply curriculum learning by splitting regions into quantiles, merging consecutive
        quantiles with small differences, and upsampling all groups to the same size.

        Args:
            summits: DataFrame with Chromosome, Start, End, and curriculum_column
            initial_epsilon: Initial threshold for merging consecutive quantiles
            min_groups: Minimum number of groups required

        Returns:
            Upsampled DataFrame with balanced groups and original peak count stored
        """
        print(f"Applying quantile-based curriculum learning on column '{self.curriculum_column}'...")

        # Check if curriculum column exists
        if self.curriculum_column not in summits.columns:
            raise ValueError(f"Curriculum column '{self.curriculum_column}' not found in DataFrame. Available columns: {list(summits.columns)}")

        # Store original peak count for downsampling
        original_peak_count = len(summits)
        print(f"Original peak count before upsampling: {original_peak_count}")

        epsilon = initial_epsilon

        while True:
            print(f"\nTrying epsilon = {epsilon:.3f}")

            # Step 1: Split into 20 quantiles with equal number of peaks
            n_quantiles = 20
            summits_sorted = summits.sort_values(self.curriculum_column).reset_index(drop=True)
            quantile_size = len(summits_sorted) // n_quantiles

            # Assign quantile groups
            quantile_groups = []
            for i in range(n_quantiles):
                start_idx = i * quantile_size
                end_idx = (i + 1) * quantile_size if i < n_quantiles - 1 else len(summits_sorted)
                quantile_groups.append(summits_sorted.iloc[start_idx:end_idx].copy())

            print(f"Created {len(quantile_groups)} quantiles with {quantile_size} peaks each")

            # Step 2: Calculate mean values for each quantile
            quantile_means = []
            for i, group in enumerate(quantile_groups):
                mean_val = group[self.curriculum_column].mean()
                quantile_means.append(mean_val)
                print(f"  Quantile {i}: mean = {mean_val:.3f}, size = {len(group)}")

            # Step 3: Merge consecutive quantiles if difference < epsilon
            merged_groups = []
            current_group = quantile_groups[0].copy()
            current_mean = quantile_means[0]

            for i in range(1, len(quantile_groups)):
                next_mean = quantile_means[i]
                difference = abs(next_mean - current_mean)

                if difference < epsilon:
                    # Merge with current group
                    current_group = pd.concat([current_group, quantile_groups[i]], ignore_index=True)
                    # Recalculate mean for merged group
                    current_mean = current_group[self.curriculum_column].mean()
                    print(f"  Merged quantile {i} (diff = {difference:.3f} < {epsilon:.3f})")
                else:
                    # Save current group and start new one
                    merged_groups.append(current_group)
                    current_group = quantile_groups[i].copy()
                    current_mean = next_mean
                    print(f"  Kept quantile {i} separate (diff = {difference:.3f} >= {epsilon:.3f})")

            # Don't forget the last group
            merged_groups.append(current_group)

            print(f"After merging: {len(merged_groups)} groups")

            # Step 4: Check if we have at least min_groups
            if len(merged_groups) >= min_groups:
                print(f"Success! Have {len(merged_groups)} groups (>= {min_groups})")
                break
            else:
                print(f"Only {len(merged_groups)} groups < {min_groups}, halving epsilon")
                epsilon = epsilon / 2
                if epsilon < 1e-6:  # Prevent infinite loop
                    print("Epsilon too small, using original quantiles")
                    merged_groups = quantile_groups
                    break

        # Step 5: Assign group labels and prepare for upsampling
        final_summits_list = []
        for i, group in enumerate(merged_groups):
            group = group.copy()
            group['curriculum_group'] = f'group_{i}'
            final_summits_list.append(group)
            mean_val = group[self.curriculum_column].mean()
            print(f"  Final group {i}: {len(group)} peaks, mean = {mean_val:.3f}")

        # Combine all groups
        summits_with_groups = pd.concat(final_summits_list, ignore_index=True)

        # Step 6: Upsample all groups to the same size (size of largest group)
        group_counts = summits_with_groups['curriculum_group'].value_counts()
        max_group_size = group_counts.max()
        print(f"\nTarget size for upsampling: {max_group_size}")

        upsampled_dfs = []
        for group_name in sorted(group_counts.index):
            group_data = summits_with_groups[summits_with_groups['curriculum_group'] == group_name].copy()
            current_size = len(group_data)

            if current_size < max_group_size:
                # Calculate how many times to repeat and remainder
                n_repeats = max_group_size // current_size
                remainder = max_group_size % current_size

                # Create upsampled data
                repeated_data = pd.concat([group_data] * n_repeats, ignore_index=True)
                if remainder > 0:
                    # Add random sample for remainder
                    additional_data = group_data.sample(n=remainder, replace=True, random_state=42)
                    repeated_data = pd.concat([repeated_data, additional_data], ignore_index=True)

                upsampled_dfs.append(repeated_data)
                print(f"  Upsampled {group_name}: {current_size} -> {len(repeated_data)} regions")
            else:
                upsampled_dfs.append(group_data)
                print(f"  {group_name}: {current_size} regions (no upsampling needed)")

        # Combine all upsampled groups
        upsampled_summits = pd.concat(upsampled_dfs, ignore_index=True)

        # Shuffle the final dataset
        upsampled_summits = upsampled_summits.sample(frac=1, random_state=42).reset_index(drop=True)

        print("\nCurriculum learning applied:")
        print(f"  Original regions (before upsampling): {original_peak_count}")
        print(f"  Upsampled regions (after upsampling): {len(upsampled_summits)}")
        print("  Final group distribution:")
        final_counts = upsampled_summits['curriculum_group'].value_counts()
        for group, count in sorted(final_counts.items()):
            print(f"    {group}: {count} regions")
        
        # Store original peak count for potential downsampling
        self.original_peak_count = original_peak_count
        
        return upsampled_summits
    
    def get_random_downsampled_indices(self, epoch_seed=None):
        """
        Get random subset of indices that downsamples back to original peak count.
        This should be called per epoch to get different random subsets.
        
        Args:
            epoch_seed: Optional seed for reproducible epoch-wise sampling
            
        Returns:
            List of indices to use for this epoch (length = original_peak_count)
        """
        if not hasattr(self, 'original_peak_count'):
            # No curriculum learning or no upsampling occurred
            return list(range(len(self.regions)))
        
        total_upsampled_regions = len(self.regions)
        original_count = self.original_peak_count
        
        if total_upsampled_regions <= original_count:
            # No downsampling needed
            print(f"No downsampling needed: {total_upsampled_regions} <= {original_count}")
            return list(range(total_upsampled_regions))
        
        # Use epoch seed for reproducible but different sampling per epoch
        rng = np.random.RandomState(epoch_seed) if epoch_seed is not None else np.random
        
        # Randomly select original_count indices from all available
        downsampled_indices = rng.choice(total_upsampled_regions, 
                                       size=original_count, 
                                       replace=False).tolist()
        
        print(f"Random downsampling per epoch: {total_upsampled_regions} -> {original_count} regions")
        
        # Log curriculum group distribution after downsampling
        if 'curriculum_group' in self.regions.columns:
            downsampled_groups = self.regions.iloc[downsampled_indices]['curriculum_group'].value_counts()
            print("  Downsampled group distribution:")
            for group, count in downsampled_groups.items():
                print(f"    {group}: {count} regions")
        
        return sorted(downsampled_indices)

    def __len__(self):
        return len(self.regions)
    
    def _setup_cache_path(self):
        """Set up cache file path based on dataset parameters"""
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create a unique filename based on dataset parameters
        celltype_str = "_".join(sorted(self.celltype_ids))
        max_regions_str = f"_max{self.max_regions}" if self.max_regions else ""
        curriculum_str = f"_curriculum{self.curriculum_groups}" if self.use_curriculum_learning else ""
        chr_str = ""
        if self.borzoi_fold:
            chr_str = f"_borzoi_{self.borzoi_val_fold}"
        elif self.val_chr or self.test_chr:
            chr_str = f"_val{self.val_chr}_test{self.test_chr}"
        filename = f"zarr_cache_{self.split}_{self.extend_bp}bp_{celltype_str}{max_regions_str}{curriculum_str}{chr_str}.pt"
        self.cache_path = os.path.join(self.cache_dir, filename)
    
    def _try_load_cache(self):
        """Try to load cached data if it exists"""
        if os.path.exists(self.cache_path):
            print(f"Loading cached data from {self.cache_path}")
            try:
                self.cached_data = torch.load(self.cache_path, weights_only=False)
                print(f"Successfully loaded {len(self.cached_data['sequences'])} cached samples")
                return True
            except Exception as e:
                print(f"Failed to load cache: {e}. Will regenerate cache.")
                self.cached_data = None
                return False
        else:
            print(f"No cache found at {self.cache_path}. Will create cache after loading data.")
            return False
    
    def _save_cache(self):
        """Save current data to cache"""
        if not self.use_cache or self.cached_data is None:
            return
            
        print(f"Saving cache to {self.cache_path}")
        try:
            torch.save(self.cached_data, self.cache_path)
            print(f"Successfully saved {len(self.cached_data['sequences'])} samples to cache")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _load_all_data_to_cache(self):
        """Load all data from zarr and store in cache"""
        print("Loading all data from zarr files to create cache...")
        sequences = []
        atac_signals = []
        region_infos = []
        
        for idx in range(len(self.regions)):
            if idx % 100 == 0:
                print(f"Loading region {idx}/{len(self.regions)}")
            
            # Load data without augmentations for caching
            sample = self._load_single_region_from_zarr(idx, apply_augmentations=False)
            if sample is not None:
                sequences.append(sample['sequence'])
                atac_signals.append(sample['atac'])
                region_infos.append(sample['region_info'])
        
        self.cached_data = {
            'sequences': sequences,
            'atac_signals': atac_signals,
            'region_infos': region_infos
        }
        
        # Save to disk
        self._save_cache()
        print("Finished loading all data to cache")
    
    def _load_single_region_from_zarr(self, idx, apply_augmentations=True):
        """Load a single region directly from zarr (used for cache creation and fallback)"""
        row = self.regions.iloc[idx]
        chrom = row['Chromosome']
        center = (row['Start'] + row['End']) // 2
        
        # Apply random shift to genomic coordinates if augmentation is enabled
        shift_amount = 0
        if apply_augmentations and self.augment and np.random.random() < self.aug_prob:
            # Random shift between -random_shift_bp and +random_shift_bp
            shift_amount = np.random.randint(-self.random_shift_bp, self.random_shift_bp + 1)
        
        start = center - self.extend_bp + shift_amount
        end = center + self.extend_bp + shift_amount
        
        try:
            # Get DNA sequence (one-hot encoded)
            sequence = self.genome.get_track(
                chrom, start, end, 
                output_format='raw_array'
            )
            
            # Get ATAC signals for all celltypes
            atac = self.cdz.get_track(
                chrom, start, end,
                output_format='raw_dict',
                aggregator='by_key',  # This gives shape (n_celltypes, seq_len)
                normalizer=DictNormalizer(self.cdz.libsize),
                conv_size=20
            )

            # convert dict of array to single array
            atac = np.stack([atac[celltype] for celltype in self.celltype_ids], axis=0)
            # Validate data shapes
            expected_length = 2 * self.extend_bp
            if (sequence.shape[0] != expected_length or 
                sequence.shape[1] != 4 or
                atac.shape[0] != len(self.celltype_ids) or
                atac.shape[1] != expected_length):
                
                raise ValueError(f"Invalid data shapes for {chrom}:{start}-{end}. "
                               f"Sequence: {sequence.shape}, ATAC: {atac.shape}")
            
            # Convert to torch tensors
            sequence = torch.from_numpy(sequence.T.astype(np.float32))  # (4, seq_len)
            atac = torch.from_numpy(atac.astype(np.float32))  # (n_celltypes, seq_len)
            
            # Apply reverse complement augmentation if enabled
            if apply_augmentations and self.augment and np.random.random() < self.aug_prob:
                sequence, atac = self._apply_reverse_complement(sequence, atac)
            
            # Prepare the basic sample
            sample = {
                'atac': atac,
                'sequence': sequence,
                'chromosome': torch.tensor(self.chromosome_to_idx[chrom], dtype=torch.long),
                'region_info': {
                    'chromosome': chrom,
                    'start': start,
                    'end': end,
                    'center': center,
                    'idx': idx
                }
            }

            # Add differential targets if available in regions DataFrame
            if hasattr(self, 'regions') and 'Diff_scca' in self.regions.columns:
                diff_value = self.regions.iloc[idx]['Diff_scca']
                sample['differential_targets'] = torch.tensor([diff_value], dtype=torch.float32)

            return sample
            
        except Exception as e:
            print(f"Error loading region {idx} ({chrom}:{start}-{end}): {e}")
            # Return dummy data to avoid crashing training
            expected_length = 2 * self.extend_bp
            sequence = torch.zeros(4, expected_length, dtype=torch.float32)
            atac = torch.zeros(len(self.celltype_ids), expected_length, dtype=torch.float32)
            
            # Prepare the basic error sample
            sample = {
                'atac': atac,
                'sequence': sequence,
                'chromosome': torch.tensor(self.chromosome_to_idx[chrom], dtype=torch.long),
                'region_info': {
                    'chromosome': chrom,
                    'start': start,
                    'end': end,
                    'center': center,
                    'idx': idx,
                    'error': str(e)
                }
            }

            # Add dummy differential targets if available in regions DataFrame
            if hasattr(self, 'regions') and 'Diff_scca' in self.regions.columns:
                sample['differential_targets'] = torch.tensor([0.0], dtype=torch.float32)

            return sample
    
    def _apply_reverse_complement(self, sequence, atac):
        """Apply reverse complement to sequence and flip ATAC signal"""
        # Reverse complement DNA sequence (A<->T, C<->G)
        # sequence shape: (4, seq_len) where order is [A, C, G, T]
        sequence_rc = sequence[[3, 2, 1, 0], :].flip(dims=[1])  # Reverse and complement
        
        # Flip ATAC signal along sequence dimension
        atac_flipped = atac.flip(dims=[1])
        
        return sequence_rc, atac_flipped
    
    def __getitem__(self, idx):
        """Load data for a single region, using cache if available"""
        # If we have cached data, use it
        if self.cached_data is not None:
            sequence = self.cached_data['sequences'][idx]#.clone()
            atac = self.cached_data['atac_signals'][idx]#.clone()
            region_info = self.cached_data['region_infos'][idx]#.copy()
            
            # Apply augmentations if enabled
            if self.augment:
                # Reverse complement (50% probability) - random shift already applied during caching
                if np.random.random() < self.aug_prob:
                    sequence, atac = self._apply_reverse_complement(sequence, atac)
            
            # Get chromosome for this sample
            chrom = region_info['chromosome']
            
            # Prepare the basic sample
            sample = {
                'atac': atac,
                'sequence': sequence,
                'chromosome': torch.tensor(self.chromosome_to_idx[chrom], dtype=torch.long),
                'region_info': region_info
            }

            # Add differential targets if available in regions DataFrame
            if hasattr(self, 'regions') and 'Diff_scca' in self.regions.columns:
                diff_value = self.regions.iloc[idx]['Diff_scca']
                sample['differential_targets'] = torch.tensor([diff_value], dtype=torch.float32)

            return sample
        
        # If no cache available, load from zarr and potentially create cache
        else:
            # Load from zarr
            result = self._load_single_region_from_zarr(idx, apply_augmentations=True)
            
            # If this is the first access and caching is enabled, load all data to cache
            if self.use_cache and idx == 0:
                self._load_all_data_to_cache()
            
            return result
    
    def create_cache(self):
        """Manually create cache by loading all data from zarr"""
        if not self.use_cache:
            print("Caching is not enabled")
            return
            
        if self.cached_data is not None:
            print("Cache already exists")
            return
            
        self._load_all_data_to_cache()
    
    def get_region_info(self, idx):
        """Get information about a specific region"""
        row = self.regions.iloc[idx]
        return {
            'index': idx,
            'chromosome': row['Chromosome'],
            'start': row['Start'],
            'end': row['End'],
            'original_csv_index': row.name if hasattr(row, 'name') else 'unknown'
        }
    
    def summary(self):
        """Print dataset summary"""
        print("DNAATACZarrDataset Summary:")
        print(f"  Split: {self.split}")
        print(f"  Total regions: {len(self)}")
        print(f"  Extend bp: {self.extend_bp}")
        print(f"  Sequence length: {2 * self.extend_bp}")
        print(f"  Celltype IDs: {self.celltype_ids}")
        print(f"  Number of celltypes: {len(self.celltype_ids)}")
        print(f"  Caching enabled: {self.use_cache}")
        if self.use_cache:
            print(f"  Cache path: {self.cache_path}")
            print(f"  Cache loaded: {self.cached_data is not None}")
        print(f"  Augmentation enabled: {self.augment}")
        if self.augment:
            print(f"  Random shift bp: {self.random_shift_bp}")
            print(f"  Augmentation probability: {self.aug_prob}")
        
        print(f"  Curriculum learning enabled: {self.use_curriculum_learning}")
        if self.use_curriculum_learning:
            print(f"  Curriculum groups: {self.curriculum_groups}")
            # Show curriculum group distribution if available
            if 'curriculum_group' in self.regions.columns:
                group_dist = self.regions['curriculum_group'].value_counts()
                print("  Curriculum group distribution:")
                for group, count in group_dist.items():
                    print(f"    {group}: {count} regions")
        
        if len(self) > 0:
            # Test loading first sample
            try:
                sample = self[0]
                print(f"  Sample ATAC shape: {sample['atac'].shape}")
                print(f"  Sample sequence shape: {sample['sequence'].shape}")
                print(f"  ATAC data type: {sample['atac'].dtype}")
                print(f"  Sequence data type: {sample['sequence'].dtype}")
                
                # Show data ranges
                atac_data = sample['atac']
                seq_data = sample['sequence']
                print(f"  ATAC value range: [{atac_data.min():.3f}, {atac_data.max():.3f}]")
                print(f"  Sequence value range: [{seq_data.min():.3f}, {seq_data.max():.3f}]")
                
                # Show first region info
                region = sample['region_info']
                print(f"  First region: {region['chromosome']}:{region['start']}-{region['end']}")
                
            except Exception as e:
                print(f"  Error testing first sample: {e}")
    
    def get_celltype_stats(self, idx):
        """Get ATAC statistics for each celltype in a region"""
        sample = self[idx]
        atac_data = sample['atac']
        region_info = sample['region_info']
        
        stats = {
            'region': f"{region_info['chromosome']}:{region_info['start']}-{region_info['end']}",
            'celltypes': {}
        }
        
        for i, celltype in enumerate(self.celltype_ids):
            celltype_signal = atac_data[i]
            stats['celltypes'][celltype] = {
                'mean': float(celltype_signal.mean()),
                'max': float(celltype_signal.max()),
                'std': float(celltype_signal.std()),
                'nonzero_frac': float((celltype_signal > 0).float().mean())
            }
        
        return stats
    
    def create_curriculum_sampler(self, batch_size, shuffle=True, drop_last=False,
                                 group_balance_strategy='equal', random_state=42,
                                 enable_epoch_downsampling=False):
        """
        Create a CurriculumGroupSampler for this dataset.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples within groups
            drop_last: Whether to drop the last incomplete batch
            group_balance_strategy: How to balance groups ('equal', 'proportional', 'random')
            random_state: Random seed for reproducibility
            enable_epoch_downsampling: Whether to randomly downsample to original peak count per epoch
            
        Returns:
            CurriculumGroupSampler instance
        """
        if not self.use_curriculum_learning:
            raise ValueError("Curriculum learning must be enabled to create curriculum sampler")
        
        return CurriculumGroupSampler(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            group_balance_strategy=group_balance_strategy,
            random_state=random_state,
            enable_epoch_downsampling=enable_epoch_downsampling
        )


class CurriculumGroupSampler(torch.utils.data.Sampler):
    """
    Curriculum-aware sampler that randomly samples across different MaxValue groups
    while maintaining balanced representation from each group.
    
    This sampler ensures that each batch contains samples from different curriculum
    groups, promoting learning across the difficulty spectrum while maintaining
    randomization. It also supports per-epoch downsampling to the original peak count.
    """
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, 
                 group_balance_strategy='equal', random_state=42, 
                 enable_epoch_downsampling=True):
        """
        Args:
            dataset: DNAATACZarrDataset with curriculum learning enabled
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples within groups
            drop_last: Whether to drop the last incomplete batch
            group_balance_strategy: How to balance groups ('equal', 'proportional', 'random')
            random_state: Random seed for reproducibility
            enable_epoch_downsampling: Whether to randomly downsample to original peak count per epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.group_balance_strategy = group_balance_strategy
        self.rng = np.random.RandomState(random_state)
        self.enable_epoch_downsampling = enable_epoch_downsampling
        self.current_epoch = 0
        self.current_epoch_indices = None
        
        # Check if dataset has curriculum groups
        if not hasattr(dataset, 'regions') or 'curriculum_group' not in dataset.regions.columns:
            raise ValueError("Dataset must have curriculum learning enabled with curriculum_group column")
        
        # Get group indices (from full upsampled dataset)
        self._build_group_indices()
        
    def _build_group_indices(self):
        """Build indices for each curriculum group."""
        self.group_indices = {}
        self.group_names = []
        
        for group_name in self.dataset.regions['curriculum_group'].unique():
            if pd.notna(group_name):  # Skip NaN groups
                group_mask = self.dataset.regions['curriculum_group'] == group_name
                indices = np.where(group_mask)[0].tolist()
                self.group_indices[group_name] = indices
                self.group_names.append(group_name)
        
        self.group_names = sorted(self.group_names)  # Ensure consistent ordering
        self.n_groups = len(self.group_names)
        
        print("CurriculumGroupSampler initialized:")
        print(f"  Total groups: {self.n_groups}")
        print(f"  Epoch downsampling enabled: {self.enable_epoch_downsampling}")
        for group_name in self.group_names:
            print(f"  {group_name}: {len(self.group_indices[group_name])} samples")
    
    def set_epoch(self, epoch):
        """
        Set current epoch and generate epoch-specific downsampled indices if enabled.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        if self.enable_epoch_downsampling:
            # Get random downsampled indices for this epoch
            epoch_seed = self.rng.randint(0, 100000) + epoch  # Combine base seed with epoch
            downsampled_indices = self.dataset.get_random_downsampled_indices(epoch_seed=epoch_seed)
            self.current_epoch_indices = set(downsampled_indices)
            
            # Rebuild group indices with only the downsampled indices
            self._build_epoch_group_indices()
        else:
            self.current_epoch_indices = None
    
    def _build_epoch_group_indices(self):
        """Build group indices for current epoch (filtered by downsampled indices)."""
        if self.current_epoch_indices is None:
            return
        
        self.epoch_group_indices = {}
        for group_name in self.group_names:
            # Filter original group indices to only include downsampled ones
            filtered_indices = [idx for idx in self.group_indices[group_name] 
                              if idx in self.current_epoch_indices]
            self.epoch_group_indices[group_name] = filtered_indices
        
        if hasattr(self, 'epoch_group_indices'):
            print(f"Epoch {self.current_epoch} downsampled group distribution:")
            for group_name in self.group_names:
                print(f"  {group_name}: {len(self.epoch_group_indices[group_name])} samples")
    
    def _get_active_group_indices(self):
        """Get the currently active group indices (epoch-specific or full)."""
        if self.enable_epoch_downsampling and hasattr(self, 'epoch_group_indices'):
            return self.epoch_group_indices
        else:
            return self.group_indices
    
    def _get_samples_per_group(self, batch_size):
        """Determine how many samples to take from each group for a batch."""
        if self.group_balance_strategy == 'equal':
            # Equal samples from each group
            base_samples = batch_size // self.n_groups
            remainder = batch_size % self.n_groups
            
            samples_per_group = [base_samples] * self.n_groups
            # Distribute remainder randomly across groups
            if remainder > 0:
                extra_groups = self.rng.choice(self.n_groups, remainder, replace=False)
                for i in extra_groups:
                    samples_per_group[i] += 1
            
            return dict(zip(self.group_names, samples_per_group))
            
        elif self.group_balance_strategy == 'proportional':
            # Proportional to group size
            active_indices = self._get_active_group_indices()
            total_samples = sum(len(indices) for indices in active_indices.values())
            samples_per_group = {}
            
            for group_name in self.group_names:
                proportion = len(active_indices[group_name]) / total_samples
                samples_per_group[group_name] = max(1, int(batch_size * proportion))
            
            # Adjust to exact batch size
            current_total = sum(samples_per_group.values())
            if current_total != batch_size:
                # Adjust the largest group
                largest_group = max(self.group_names, key=lambda g: samples_per_group[g])
                samples_per_group[largest_group] += batch_size - current_total
            
            return samples_per_group
            
        elif self.group_balance_strategy == 'random':
            # Randomly distribute samples across groups
            samples_per_group = {}
            remaining = batch_size
            
            for i, group_name in enumerate(self.group_names):
                if i == len(self.group_names) - 1:  # Last group gets remaining
                    samples_per_group[group_name] = remaining
                else:
                    # Random number of samples, but ensure at least 1 and leave room for others
                    max_for_this_group = max(1, remaining - (len(self.group_names) - i - 1))
                    samples = self.rng.randint(1, max_for_this_group + 1)
                    samples_per_group[group_name] = samples
                    remaining -= samples
            
            return samples_per_group
        
        else:
            raise ValueError(f"Unknown group_balance_strategy: {self.group_balance_strategy}")
    
    def _sample_from_group(self, group_name, n_samples):
        """Sample n_samples from a specific curriculum group."""
        group_indices = self.group_indices[group_name]
        
        if len(group_indices) == 0:
            return []
        
        if n_samples >= len(group_indices):
            # Need more samples than available, sample with replacement
            return self.rng.choice(group_indices, n_samples, replace=True).tolist()
        else:
            # Sample without replacement
            return self.rng.choice(group_indices, n_samples, replace=False).tolist()
    
    def __iter__(self):
        """Generate batches with curriculum group awareness."""
        # Shuffle indices within each group if requested
        if self.shuffle:
            for group_name in self.group_names:
                self.rng.shuffle(self.group_indices[group_name])
        
        # Calculate total number of batches using full dataset
        total_samples = len(self.dataset)
        if self.drop_last:
            n_batches = total_samples // self.batch_size
        else:
            n_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        # Generate batches using the original logic for non-downsampling cases
        for batch_idx in range(n_batches):
            # Determine batch size for this batch (handle last batch)
            if batch_idx == n_batches - 1 and not self.drop_last:
                current_batch_size = total_samples - batch_idx * self.batch_size
            else:
                current_batch_size = self.batch_size
            
            # Get samples per group for this batch
            samples_per_group = self._get_samples_per_group(current_batch_size)
            
            # Sample from each group
            batch_indices = []
            for group_name in self.group_names:
                n_samples = samples_per_group[group_name]
                if n_samples > 0:
                    group_samples = self._sample_from_group(group_name, n_samples)
                    batch_indices.extend(group_samples)
            
            # Shuffle batch indices to mix groups within batch
            if self.shuffle:
                self.rng.shuffle(batch_indices)
            
            yield batch_indices
    
    def _iter_with_epoch_downsampling(self):
        """Generate batches for epoch downsampling with exact sample count control."""
        if self.current_epoch_indices is None:
            return iter([])
        
        # Convert to list and shuffle the downsampled indices directly
        available_indices = list(self.current_epoch_indices)
        if self.shuffle:
            self.rng.shuffle(available_indices)
        
        # Generate batches by simply slicing the shuffled indices
        total_samples = len(available_indices)
        batches = []
        
        if self.drop_last:
            n_batches = total_samples // self.batch_size
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batches.append(available_indices[start_idx:end_idx])
        else:
            n_batches = (total_samples + self.batch_size - 1) // self.batch_size
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_samples)
                if start_idx < total_samples:  # Only add non-empty batches
                    batches.append(available_indices[start_idx:end_idx])
        
        return iter(batches)
    
    def __len__(self):
        """Return number of batches."""
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size
    
    def get_group_statistics(self):
        """Get statistics about group distribution in sampling."""
        stats = {
            'total_groups': self.n_groups,
            'group_sizes': {name: len(indices) for name, indices in self.group_indices.items()},
            'balance_strategy': self.group_balance_strategy,
            'batch_size': self.batch_size
        }
        return stats