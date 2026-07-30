"""
Filename: dna_atac_zarr_datamodule.py
Purpose: PyTorch Lightning data module for direct zarr file loading in DNA->ATAC prediction

Main Components:
- DNAATACZarrDataModule: Lightning data module for loading data directly from zarr files
- Memory-efficient data loading without pre-curation requirements
- Support for multiple celltype ATAC signal prediction
- Cache management for improved performance across training runs
- Integration with Caesar zarr I/O system for genomic data access

Dependencies: pytorch_lightning, torch, atac_prediction.dna_atac_zarr_dataset, caesar.io.zarr_io
"""

from __future__ import annotations

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from atac_prediction.dna_atac_zarr_dataset import DNAATACZarrDataset
from caesar.io.zarr_io import CelltypeDenseZarrIO, SequenceDenseZarrIO
from caesar.io.bpcell_io import CelltypeDenseBPCellsIO


class DNAATACZarrDataModule(pl.LightningDataModule):
    """
    Data module for DNA -> ATAC prediction that loads directly from zarr files.
    Memory efficient - no pre-curation needed.
    """

    def __init__(
        self,
        atac_zarr_path: str,
        genome_zarr_path: str,
        summits_csv_path: str = None,
        summits_df: pd.DataFrame = None,
        celltype_ids: list[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        extend_bp: int = 2048,
        max_regions: int = None,
        train_split: float = 0.75,
        val_split: float = 0.15,
        test_split: float = 0.10,
        cache_dir: str = None,
        use_curriculum_learning: bool = False,
        curriculum_groups: int = 5,
        curriculum_column: str = "MaxValue",
        curriculum_sampling_strategy: str = "equal",
        curriculum_initial_epsilon: float = 0.2,
        curriculum_min_groups: int = 5,
        val_chr: str = None,
        test_chr: str = None,
        borzoi_fold: bool = False,
        borzoi_val_fold: str = "fold6,fold7",
        sequences_human_bed: str = "sequences_human.bed",
    ):
        super().__init__()
        self.atac_zarr_path = atac_zarr_path
        self.genome_zarr_path = genome_zarr_path

        # Accept either CSV path or DataFrame directly
        if summits_csv_path is None and summits_df is None:
            raise ValueError("Either summits_csv_path or summits_df must be provided")
        if summits_csv_path is not None and summits_df is not None:
            raise ValueError("Only one of summits_csv_path or summits_df should be provided")

        self.summits_csv_path = summits_csv_path
        self.summits_df = summits_df
        self.celltype_ids = celltype_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.extend_bp = extend_bp
        self.max_regions = max_regions
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.cache_dir = cache_dir
        self.use_curriculum_learning = use_curriculum_learning
        self.curriculum_groups = curriculum_groups
        self.curriculum_column = curriculum_column
        self.curriculum_sampling_strategy = curriculum_sampling_strategy
        self.curriculum_initial_epsilon = curriculum_initial_epsilon
        self.curriculum_min_groups = curriculum_min_groups
        self.val_chr = val_chr
        self.test_chr = test_chr
        self.borzoi_fold = borzoi_fold
        self.borzoi_val_fold = borzoi_val_fold
        self.sequences_human_bed = sequences_human_bed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Zarr objects will be initialized once in setup() (if needed)
        self.cdz = None
        self.genome = None

    def prepare_data(self):
        """Download data if needed (not applicable for zarr files)"""
        pass

    def _check_cache_exists(self, splits=["train", "val", "test"]):
        """Check if cache files exist for all required splits"""
        if not self.cache_dir:
            return False

        import os

        for split in splits:
            # Recreate cache filename logic from DNAATACZarrDataset
            celltype_str = "_".join(sorted(self.celltype_ids))
            max_regions_str = f"_max{self.max_regions}" if self.max_regions else ""
            curriculum_str = (
                f"_curriculum{self.curriculum_groups}"
                if self.use_curriculum_learning
                else ""
            )
            chr_str = ""
            if self.borzoi_fold:
                chr_str = f"_borzoi_{self.borzoi_val_fold}"
            elif self.val_chr or self.test_chr:
                chr_str = f"_val{self.val_chr}_test{self.test_chr}"
            filename = f"zarr_cache_{split}_{self.extend_bp}bp_{celltype_str}{max_regions_str}{curriculum_str}{chr_str}.pt"
            cache_path = os.path.join(self.cache_dir, filename)

            if not os.path.exists(cache_path):
                return False

        return True

    def setup(self, stage: str | None = None):
        """Create datasets for each split"""
        # Check which splits we need based on stage
        if stage == "fit" or stage is None:
            required_splits = ["train", "val"]
        elif stage == "test":
            required_splits = ["test"]
        else:
            required_splits = ["train", "val", "test"]

        # Check if cache exists for all required splits
        cache_available = self._check_cache_exists(required_splits)

        if cache_available:
            print("Cache files found for all required splits. Skipping zarr loading.")
            # Set zarr objects to None to indicate they're not needed
            self.cdz = None
            self.genome = None
        else:
            print("Cache files not found. Loading zarr objects...")
            # Initialize zarr objects once per process (not per worker)
            # Use chunk-based loading instead of loading all to memory
            if self.cdz is None:
                print(f"Loading ATAC-seq zarr from {self.atac_zarr_path}")
                if self.atac_zarr_path.endswith('.zarr'):
                    self.cdz = CelltypeDenseZarrIO(self.atac_zarr_path, mode="r")
                else:
                    self.cdz = CelltypeDenseBPCellsIO(path=self.atac_zarr_path, mode="r")
                # Don't load to memory - use chunk-based loading
                self.cdz = self.cdz.subset(self.celltype_ids)
                if isinstance(self.cdz, CelltypeDenseZarrIO):
                    self.cdz.load_to_memory_dense()

            if self.genome is None:
                print(f"Loading genome zarr from {self.genome_zarr_path}")
                self.genome = SequenceDenseZarrIO(self.genome_zarr_path, mode="r")
                # Don't load to memory - use chunk-based loading

        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                print("Setting up training dataset...")
                self.train_dataset = DNAATACZarrDataset(
                    cdz_object=self.cdz,
                    genome_object=self.genome,
                    summits_csv_path=self.summits_csv_path,
                    summits_df=self.summits_df,
                    celltype_ids=self.celltype_ids,
                    extend_bp=self.extend_bp,
                    max_regions=self.max_regions,
                    split="train",
                    train_split=self.train_split,
                    val_split=self.val_split,
                    test_split=self.test_split,
                    cache_dir=self.cache_dir,
                    use_curriculum_learning=self.use_curriculum_learning,
                    curriculum_groups=self.curriculum_groups,
                    curriculum_column=self.curriculum_column,
                    curriculum_initial_epsilon=self.curriculum_initial_epsilon,
                    curriculum_min_groups=self.curriculum_min_groups,
                    val_chr=self.val_chr,
                    test_chr=self.test_chr,
                    borzoi_fold=self.borzoi_fold,
                    borzoi_val_fold=self.borzoi_val_fold,
                    sequences_human_bed=self.sequences_human_bed,
                )

            if self.val_dataset is None:
                print("Setting up validation dataset...")
                self.val_dataset = DNAATACZarrDataset(
                    cdz_object=self.cdz,
                    genome_object=self.genome,
                    summits_csv_path=self.summits_csv_path,
                    summits_df=self.summits_df,
                    celltype_ids=self.celltype_ids,
                    extend_bp=self.extend_bp,
                    max_regions=self.max_regions,
                    split="val",
                    train_split=self.train_split,
                    val_split=self.val_split,
                    test_split=self.test_split,
                    cache_dir=self.cache_dir,
                    use_curriculum_learning=False,  # Don't use curriculum for validation
                    curriculum_groups=self.curriculum_groups,
                    curriculum_column=self.curriculum_column,
                    curriculum_initial_epsilon=self.curriculum_initial_epsilon,
                    curriculum_min_groups=self.curriculum_min_groups,
                    val_chr=self.val_chr,
                    test_chr=self.test_chr,
                    borzoi_fold=self.borzoi_fold,
                    borzoi_val_fold=self.borzoi_val_fold,
                    sequences_human_bed=self.sequences_human_bed,
                )

        if stage == "test":
            if self.test_dataset is None:
                print("Setting up test dataset...")
                self.test_dataset = DNAATACZarrDataset(
                    cdz_object=self.cdz,
                    genome_object=self.genome,
                    summits_csv_path=self.summits_csv_path,
                    summits_df=self.summits_df,
                    celltype_ids=self.celltype_ids,
                    extend_bp=self.extend_bp,
                    max_regions=self.max_regions,
                    split="test",
                    train_split=self.train_split,
                    val_split=self.val_split,
                    test_split=self.test_split,
                    cache_dir=self.cache_dir,
                    use_curriculum_learning=False,  # Don't use curriculum for test
                    curriculum_groups=self.curriculum_groups,
                    curriculum_column=self.curriculum_column,
                    curriculum_initial_epsilon=self.curriculum_initial_epsilon,
                    curriculum_min_groups=self.curriculum_min_groups,
                    val_chr=self.val_chr,
                    test_chr=self.test_chr,
                    borzoi_fold=self.borzoi_fold,
                    borzoi_val_fold=self.borzoi_val_fold,
                    sequences_human_bed=self.sequences_human_bed,
                )

    def train_dataloader(self):
        # Use curriculum sampler if curriculum learning is enabled
        if self.use_curriculum_learning and hasattr(
            self.train_dataset, "create_curriculum_sampler"
        ):
            print(
                f"Using curriculum sampler with {self.curriculum_sampling_strategy} strategy"
            )
            self.curriculum_sampler = self.train_dataset.create_curriculum_sampler(
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                group_balance_strategy=self.curriculum_sampling_strategy,
                enable_epoch_downsampling=False,  # Disable epoch downsampling
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.curriculum_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
            )
        else:
            # Standard dataloader
            self.curriculum_sampler = None
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
                drop_last=True,  # Helps with batch consistency
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_dataset_info(self):
        """Get information about the datasets"""
        info = {
            "celltype_ids": self.celltype_ids,
            "num_celltypes": len(self.celltype_ids),
            "extend_bp": self.extend_bp,
            "sequence_length": 2 * self.extend_bp,
            "max_regions": self.max_regions,
            "splits": {
                "train": self.train_split,
                "val": self.val_split,
                "test": self.test_split,
            },
        }

        if self.train_dataset is not None:
            info["train_size"] = len(self.train_dataset)
        if self.val_dataset is not None:
            info["val_size"] = len(self.val_dataset)
        if self.test_dataset is not None:
            info["test_size"] = len(self.test_dataset)

        return info

    def summary(self):
        """Print summary of all datasets"""
        info = self.get_dataset_info()

        print("DNAATACZarrDataModule Summary:")
        print(f"  Celltype IDs: {info['celltype_ids']}")
        print(f"  Number of celltypes: {info['num_celltypes']}")
        print(f"  Sequence length: {info['sequence_length']} bp")
        print(f"  Max regions: {info['max_regions']}")
        print(
            f"  Split ratios: train={info['splits']['train']}, val={info['splits']['val']}, test={info['splits']['test']}"
        )

        if "train_size" in info:
            print(f"  Train size: {info['train_size']}")
        if "val_size" in info:
            print(f"  Val size: {info['val_size']}")
        if "test_size" in info:
            print(f"  Test size: {info['test_size']}")

        print(f"  Batch size: {self.batch_size}")
        print(f"  Num workers: {self.num_workers}")
        print(f"  Curriculum learning: {self.use_curriculum_learning}")
        if self.use_curriculum_learning:
            print(f"    Groups: {self.curriculum_groups}")
            print(f"    Sampling strategy: {self.curriculum_sampling_strategy}")

        # Test datasets if available
        if self.train_dataset is not None:
            print("\nTraining dataset:")
            self.train_dataset.summary()

        if self.val_dataset is not None:
            print("\nValidation dataset:")
            self.val_dataset.summary()
