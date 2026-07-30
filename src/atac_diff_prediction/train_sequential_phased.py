#!/usr/bin/env python3
"""
Sequential three-phase ATAC training script.

Runs Phase 1 (bias), Phase 2 (main), and Phase 3 (differential) sequentially
in a single script with separate trainers and pre-filtered datasets.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import platform
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Add parent directory to path for imports
sys.path.append(str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import checkpoint utilities
from checkpoint_manager import PhaseCheckpointManager, CheckpointManager, ModelComponents

from configs.config_utils import ConfigManager
from atac_prediction.train_atac import (
    load_motifs, validate_arguments, _add_data_arguments,
    _add_model_arguments, _add_training_arguments
)
from atac_prediction.dna_atac_zarr_datamodule import DNAATACZarrDataModule
from atac_prediction.dna_atac_lightning_module import DNAATACPredictionLightningModule

from .models import create_phase1_model, create_phase2_model, create_phase3_model
from .lightning_modules import Phase1LightningModule, Phase2LightningModule, Phase3LightningModule

# NOTE: The old Phase1BiasLightningModule has been replaced with the new architecture.
# See the new modules in:
# - models.py: Phase1Model, Phase2Model, Phase3Model (torch-compilable nn.Modules)
# - lightning_modules.py: Phase1LightningModule, Phase2LightningModule, Phase3LightningModule

# Example usage of the new architecture:
"""
from .models import create_phase1_model, create_phase2_model, create_phase3_model
from .lightning_modules import Phase1LightningModule, Phase2LightningModule, Phase3LightningModule

# Phase 1: Create and train bias model
phase1_model = create_phase1_model(config)
phase1_lightning = Phase1LightningModule(phase1_model, learning_rate=1e-3)
# Train phase1_lightning...

# Phase 2: Create and train main model
phase2_model = create_phase2_model(main_atac_model)
phase2_lightning = Phase2LightningModule(phase2_model, learning_rate=1e-3)
# Train phase2_lightning...

# Phase 3: Create and train differential model
phase3_model = create_phase3_model(main_atac_model, config)
phase3_lightning = Phase3LightningModule(phase3_model, learning_rate=1e-3)
# Train phase3_lightning...

# Models can be torch.compile'd for faster training:
compiled_phase1_model = torch.compile(phase1_model)
compiled_phase2_model = torch.compile(phase2_model)
compiled_phase3_model = torch.compile(phase3_model)
"""

# Set precision for improved performance
torch.set_float32_matmul_precision('high')




def load_and_filter_summits(summits_csv_path: str, regions_filter: str, differential_threshold: float = 0.01) -> pd.DataFrame:
    """Load summits CSV and apply phase-specific filtering."""

    # Try to detect if file has header and number of columns
    first_line = pd.read_csv(summits_csv_path, sep='\t', header=None, nrows=1)

    # Check if first line contains strings that indicate header
    has_header = any(isinstance(val, str) and not val.startswith('chr') for val in first_line.iloc[0])

    if has_header:
        # File has header, read with header=0
        summits = pd.read_csv(summits_csv_path, sep='\t', header=0)
        print(f"Detected header format with {len(summits.columns)} columns")
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

    # Apply phase-specific filtering
    if regions_filter == 'background':
        # Phase 1: Only background regions
        if 'peak_type' in summits.columns:
            mask = summits['peak_type'] == 'background'
            filtered_summits = summits[mask].copy()
            print(f"Phase 1: Using {len(filtered_summits)} background regions (from {len(summits)} total)")
            if len(filtered_summits) == 0:
                print("Phase 1: No background regions found, falling back to all regions.")
                filtered_summits = summits.copy()
        else:
            print(f"Phase 1: Warning - no peak_type column found, using all {len(summits)} regions")
            filtered_summits = summits.copy()

    elif regions_filter == 'all':
        # Phase 2: All for_count and for_profile regions
        if 'peak_type' in summits.columns:
            mask = summits['peak_type'].isin(['for_count', 'for_profile'])
            filtered_summits = summits[mask].copy()
            print(f"Phase 2: Using {len(filtered_summits)} for_count/for_profile regions (from {len(summits)} total)")
        else:
            print(f"Phase 2: Warning - no peak_type column found, using all {len(summits)} regions")
            filtered_summits = summits.copy()

    elif regions_filter == 'differential':
        # Phase 3: Only regions with significant differential accessibility
        if 'Diff_DZ_vs_LZ' in summits.columns:
            mask = summits['Diff_DZ_vs_LZ'].abs() >= differential_threshold
            filtered_summits = summits[mask].copy()
            print(f"Phase 3: Using {len(filtered_summits)} regions with |Diff_DZ_vs_LZ| >= {differential_threshold} (from {len(summits)} total)")
        else:
            print(f"Phase 3: Warning - no differential info available, using all {len(summits)} regions")
            filtered_summits = summits.copy()
    else:
        raise ValueError(f"Unknown regions_filter: {regions_filter}")

    return filtered_summits


def create_filtered_dataset(base_dataset, regions_filter: str, differential_threshold: float = 0.01):
    """Create a filtered dataset based on region type (legacy wrapper for compatibility)."""

    class FilteredDataset:
        def __init__(self, base_dataset, regions_filter, differential_threshold=0.01):
            self.base_dataset = base_dataset
            self.regions_filter = regions_filter
            self.differential_threshold = differential_threshold

            # Apply filtering
            self.filtered_indices = self._apply_filter()

        def _apply_filter(self):
            """Apply filtering based on regions_filter type."""
            if not hasattr(self.base_dataset, 'regions'):
                print(f"Warning: No regions info available, using all {len(self.base_dataset)} regions")
                return list(range(len(self.base_dataset)))

            regions = self.base_dataset.regions

            if self.regions_filter == 'background':
                # Phase 1: Only background regions
                mask = regions['peak_type'] == 'background'
                filtered_indices = regions.index[mask].tolist()
                print(f"Phase 1: Using {len(filtered_indices)} background regions (from {len(self.base_dataset)} total)")

            elif self.regions_filter == 'all':
                # Phase 2: All regions
                filtered_indices = list(range(len(self.base_dataset)))
                print(f"Phase 2: Using all {len(filtered_indices)} regions")

            elif self.regions_filter == 'differential':
                # Phase 3: Only regions with significant differential accessibility
                if 'Diff_DZ_vs_LZ' in regions.columns:
                    mask = regions['Diff_DZ_vs_LZ'].abs() >= self.differential_threshold
                    filtered_indices = regions.index[mask].tolist()
                    print(f"Phase 3: Using {len(filtered_indices)} regions with |Diff_DZ_vs_LZ| >= {self.differential_threshold} (from {len(self.base_dataset)} total)")
                else:
                    print(f"Phase 3: Warning - no differential info available, using all {len(self.base_dataset)} regions")
                    filtered_indices = list(range(len(self.base_dataset)))
            else:
                raise ValueError(f"Unknown regions_filter: {self.regions_filter}")

            return filtered_indices

        def __len__(self):
            return len(self.filtered_indices)

        def __getitem__(self, idx):
            # Map to original dataset index
            base_idx = self.filtered_indices[idx]
            sample = self.base_dataset[base_idx]

            # Extract only negative condition
            if hasattr(self.base_dataset, 'celltype_ids'):
                celltype_ids = self.base_dataset.celltype_ids
                neg_indices = [i for i, cid in enumerate(celltype_ids) if 'neg' in cid.lower()]

                if neg_indices:
                    target_idx = neg_indices[0]
                    sample['atac'] = sample['atac'][target_idx:target_idx+1]

            # Add differential targets for Phase 3
            if self.regions_filter == 'differential':
                region_info = self.base_dataset.regions.iloc[base_idx]
                if 'Diff_DZ_vs_LZ' in region_info:
                    sample['differential_targets'] = torch.tensor([
                        region_info['Diff_DZ_vs_LZ'],
                    ], dtype=torch.float32)
                else:
                    sample['differential_targets'] = torch.tensor([0.0], dtype=torch.float32)

            return sample

    return FilteredDataset(base_dataset, regions_filter, differential_threshold)


def run_phase(phase_name: str, model: pl.LightningModule, data_module: pl.LightningDataModule,
              epochs: int, run_name: str, config: Dict[str, Any], logger: Optional[WandbLogger] = None, val_check_interval=0.5) -> Optional[str]:
    """Run a single training phase."""

    if epochs == 0:
        print(f"Skipping {phase_name} (epochs=0)")
        return None

    print(f"\n{'='*60}")
    print(f"{phase_name.upper()}")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")

    phase_run_name = f"{run_name}_{phase_name.lower()}"

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename=f'{phase_run_name}-{{epoch:02d}}-{{val_loss:.2f}}',
            save_top_k=3,
            mode='min',
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15 if phase_name == 'Phase2' else 10,
            mode='min'
        )
    ]

    # Use provided logger or create a new one
    if logger is None:
        logger = WandbLogger(
            project=config.get('logging', {}).get('wandb_project', 'ATAC-Phased-Learning'),
            name=run_name  # Use base run name, not phase-specific
        )

    # Create trainer first
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision="32-true",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=val_check_interval,
        gradient_clip_val=0.5,
        detect_anomaly=False,
    )

    # Log phase transition with enhanced metadata
    if hasattr(logger, 'experiment'):
        logger.experiment.log({
            "training_phase": phase_name,
            "phase_epochs": epochs,
            "phase_start_epoch": trainer.current_epoch if hasattr(trainer, 'current_epoch') else 0,
            f"{phase_name.lower()}_config": {
                "epochs": epochs,
                "val_check_interval": val_check_interval,
                "learning_rate": model.hparams.learning_rate if hasattr(model, 'hparams') else "unknown",
                "model_type": type(model).__name__
            }
        })

    # Test data loading with comprehensive logging
    print("\nTesting data loading...")
    try:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        print(f"Train batches per epoch: {len(train_loader)}")
        print(f"Val batches per epoch: {len(val_loader)}")

        # Test loading a batch
        batch = next(iter(train_loader))
        print(f"Batch ATAC shape: {batch['atac'].shape}")
        print(f"Batch sequence shape: {batch['sequence'].shape}")
        if 'differential_targets' in batch:
            print(f"Batch differential targets shape: {batch['differential_targets'].shape}")

        # Log data loading stats to wandb
        if hasattr(logger, 'experiment'):
            logger.experiment.log({
                f"{phase_name.lower()}_data_loading": {
                    "train_batches_per_epoch": len(train_loader),
                    "val_batches_per_epoch": len(val_loader),
                    "batch_atac_shape": list(batch['atac'].shape),
                    "batch_sequence_shape": list(batch['sequence'].shape),
                    "has_differential_targets": 'differential_targets' in batch,
                    "data_loading_successful": True
                }
            })

    except Exception as e:
        print(f"Error during data loading test: {e}")
        if hasattr(logger, 'experiment'):
            logger.experiment.log({
                f"{phase_name.lower()}_data_loading": {
                    "data_loading_successful": False,
                    "error": str(e)
                }
            })
        # Still continue with training
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

    # Start training with error handling
    print(f"\nStarting {phase_name} training...")
    try:
        # Compile model for better performance
        print(f"Compiling {phase_name} model for optimal performance...")
        model.model = torch.compile(model.model)
        print("✓ Model compilation successful")

        # Log training start
        if hasattr(logger, 'experiment'):
            logger.experiment.log({
                f"{phase_name.lower()}_training_started": True,
                f"{phase_name.lower()}_model_compiled": True
            })

        # Start training
        trainer.fit(model, datamodule=data_module)

        # Log training completion
        if hasattr(logger, 'experiment'):
            logger.experiment.log({
                f"{phase_name.lower()}_training_completed": True,
                f"{phase_name.lower()}_training_successful": True
            })

    except Exception as e:
        print(f"Error during {phase_name} training: {e}")
        if hasattr(logger, 'experiment'):
            logger.experiment.log({
                f"{phase_name.lower()}_training_completed": True,
                f"{phase_name.lower()}_training_successful": False,
                f"{phase_name.lower()}_training_error": str(e)
            })
        # Re-raise the exception to stop the training pipeline
        raise

    print(f"\n{phase_name} completed!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Best validation loss: {callbacks[0].best_model_score}")

    # Log phase completion with comprehensive metrics
    if hasattr(logger, 'experiment'):
        best_score = callbacks[0].best_model_score.item() if callbacks[0].best_model_score is not None else None
        logger.experiment.log({
            f"{phase_name.lower()}_completed": True,
            f"{phase_name.lower()}_best_val_loss": best_score,
            f"{phase_name.lower()}_checkpoint_path": callbacks[0].best_model_path,
            f"{phase_name.lower()}_total_epochs_trained": epochs,
            f"{phase_name.lower()}_early_stopped": trainer.stopped_epoch > 0 if hasattr(trainer, 'stopped_epoch') else False,
            f"{phase_name.lower()}_final_epoch": trainer.current_epoch if hasattr(trainer, 'current_epoch') else epochs,
            f"{phase_name.lower()}_summary": {
                "phase": phase_name,
                "completed": True,
                "best_val_loss": best_score,
                "epochs_planned": epochs,
                "epochs_completed": trainer.current_epoch if hasattr(trainer, 'current_epoch') else epochs,
                "model_checkpoint": callbacks[0].best_model_path
            }
        })

        # Log model architecture summary for this phase
        if hasattr(model, 'model'):
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            logger.experiment.log({
                f"{phase_name.lower()}_model_stats": {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "frozen_parameters": total_params - trainable_params,
                    "trainable_fraction": trainable_params / total_params if total_params > 0 else 0
                }
            })

    return callbacks[0].best_model_path


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sequential three-phase ATAC training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add config argument first
    parser.add_argument(
        "--config",
        default="configs/train_atac_phased.yaml",
        help="Path to YAML configuration file"
    )

    # Add standard training arguments
    _add_data_arguments(parser)
    _add_model_arguments(parser)
    _add_training_arguments(parser)

    # Phase-specific arguments
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=50,
        help="Number of epochs for Phase 1 (0 to skip)"
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=100,
        help="Number of epochs for Phase 2 (0 to skip)"
    )
    parser.add_argument(
        "--phase3-epochs",
        type=int,
        default=50,
        help="Number of epochs for Phase 3 (0 to skip)"
    )
    parser.add_argument(
        "--differential-threshold",
        type=float,
        default=0.01,
        help="Threshold for differential filtering in Phase 3"
    )
    parser.add_argument(
        "--tn5-motifs-path",
        default="artifacts/bias.meme",
        help="Path to TN5 motifs file (MEME format)"
    )

    # Checkpoint loading arguments
    parser.add_argument(
        "--load-phase1-checkpoint",
        type=str,
        help="Path to checkpoint to load for phase 1 bias model"
    )
    parser.add_argument(
        "--load-phase2-checkpoint",
        type=str,
        help="Path to checkpoint to load for phase 2 model (main ATAC model)"
    )
    parser.add_argument(
        "--load-phase3-checkpoint",
        type=str,
        help="Path to checkpoint to load for phase 3 model (differential model)"
    )
    parser.add_argument(
        "--strict-loading",
        action="store_true",
        help="Use strict loading for checkpoint (will fail if keys don't match exactly)"
    )
    parser.add_argument(
        "--phase2-curriculum-column",
        type=str,
        default=None,
        help="Column name in summits file for Phase 2 curriculum learning (auto-detected if not provided)"
    )
    parser.add_argument(
        "--phase3-differential-column",
        type=str,
        default=None,
        help="Column name in summits file for Phase 3 differential filtering (auto-detected if not provided)"
    )

    return parser.parse_args()


def _resolve_curriculum_column(requested_col: Optional[str], summits_df: pd.DataFrame, prefix: str, fallback_prefixes: list) -> str:
    """Resolve curriculum column name from user arg or auto-detect from DataFrame columns."""
    if requested_col and requested_col in summits_df.columns:
        return requested_col
    if requested_col:
        print(f"Warning: requested curriculum column '{requested_col}' not found in DataFrame.")
    # Auto-detect: find first column matching any fallback prefix
    for fp in fallback_prefixes:
        candidates = [c for c in summits_df.columns if c.startswith(fp)]
        if candidates:
            col = candidates[0]
            print(f"Auto-detected curriculum column: '{col}' (from prefix '{fp}')")
            return col
    available = list(summits_df.columns)
    raise ValueError(f"Cannot find a suitable curriculum column with prefix '{prefix}'. Available columns: {available}")


def main():
    """Main sequential training function."""
    # Parse arguments and load configuration
    args = parse_arguments()
    config_manager = ConfigManager("train_atac_phased")
    config = config_manager.load_config(args.config)

    # Validate arguments
    validate_arguments(args)

    # Generate run name if not provided
    if not args.run_name:
        args.run_name = f"ATAC-Phased-{len(args.celltype_ids)}samples-{args.max_regions}regions"

    print(f"Sequential training run name: {args.run_name}")
    print(f"Config file: {args.config}")

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    # Create phase checkpoint manager for all phases
    phase_manager = PhaseCheckpointManager(
        run_name=args.run_name,
        checkpoint_dir="checkpoints",
        strict_loading=not args.strict_loading
    )
    print(f"Created phase checkpoint manager for: {args.run_name}")

    # Create shared wandb logger for all phases with enhanced configuration
    wandb_config = config.get('logging', {})
    shared_logger = WandbLogger(
        project=wandb_config.get('wandb_project', 'ATAC-Phased-Learning-DDR'),
        name=args.run_name,
        tags=wandb_config.get('tags', ['sequential-phased', 'atac-prediction']),
        notes=f"Sequential three-phase ATAC training: Phase1({args.phase1_epochs}), Phase2({args.phase2_epochs}), Phase3({args.phase3_epochs})",
        config={
            'phase1_epochs': args.phase1_epochs,
            'phase2_epochs': args.phase2_epochs,
            'phase3_epochs': args.phase3_epochs,
            'differential_threshold': args.differential_threshold,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate if hasattr(args, 'learning_rate') else '1e-4',
            'max_regions': args.max_regions,
            'celltype_ids': args.celltype_ids,
            'config_file': args.config,
        }
    )
    print(f"Created shared wandb logger for run: {args.run_name}")
    print(f"  Project: {wandb_config.get('wandb_project', 'ATAC-Phased-Learning-DDR')}")
    print(f"  Tags: {wandb_config.get('tags', ['sequential-phased', 'atac-prediction'])}")
    print(f"  Notes: Sequential training with {sum([args.phase1_epochs, args.phase2_epochs, args.phase3_epochs])} total epochs")

    # ========================================
    # PHASE 1: Bias Model Training
    # ========================================

    bias_model_checkpoint = None

    if args.phase1_epochs > 0:
        print(f"\n{'='*80}")
        print("SETTING UP PHASE 1: BIAS MODEL TRAINING")
        print(f"{'='*80}")

        print("Setting up Phase 1 bias model using new architecture...")

        # Pre-filter summits for Phase 1 (background only)
        phase1_summits = load_and_filter_summits(args.summits_csv_path, 'background')

        # Set up data module for Phase 1 with pre-filtered data and curriculum learning
        phase1_data_module = DNAATACZarrDataModule(
            atac_zarr_path=args.atac_zarr_path,
            genome_zarr_path=args.genome_zarr_path,
            summits_df=phase1_summits,  # Use pre-filtered DataFrame
            celltype_ids=args.celltype_ids,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            extend_bp=args.extend_bp,
            max_regions=args.max_regions,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            use_curriculum_learning=False,  # Enable curriculum for Phase 1
        )

        phase1_data_module.setup()

        # Log Phase 1 dataset statistics
        if hasattr(shared_logger, 'experiment'):
            train_dataset_size = len(phase1_data_module.train_dataset) if hasattr(phase1_data_module, 'train_dataset') else "unknown"
            val_dataset_size = len(phase1_data_module.val_dataset) if hasattr(phase1_data_module, 'val_dataset') else "unknown"
            shared_logger.experiment.log({
                "phase1_dataset_stats": {
                    "total_regions": len(phase1_summits),
                    "train_regions": train_dataset_size,
                    "val_regions": val_dataset_size,
                    "filter_type": "background",
                    "curriculum_learning": False,
                    "batch_size": args.batch_size,
                    "extend_bp": args.extend_bp
                }
            })

        # Create Phase 1 Lightning module using new architecture
        phase1_nn_model = create_phase1_model(config)
        phase1_model = Phase1LightningModule(
            model=phase1_nn_model,
            learning_rate=1e-3,
            freeze_tn5_motifs=True
        )

        # Run Phase 1
        bias_model_checkpoint = run_phase('Phase1', phase1_model, phase1_data_module,
                                         args.phase1_epochs, args.run_name, config, shared_logger, val_check_interval=None)

        # Save bias model separately with enhanced metadata
        if bias_model_checkpoint:
            checkpoint_path = phase_manager.save_phase1(
                bias_model=phase1_model.model.bias_model,
                metadata={
                    'epochs': args.phase1_epochs,
                    'best_checkpoint': bias_model_checkpoint,
                    'training_completed': True,
                    'model_architecture': 'Phase1BiasModel',
                },
                config=config
            )

            # Log to wandb
            if hasattr(shared_logger, 'experiment'):
                shared_logger.experiment.log({
                    "phase1_model_saved": True,
                    "phase1_save_path": str(checkpoint_path),
                    "phase1_model_size_mb": checkpoint_path.stat().st_size / (1024 * 1024)
                })

    # ========================================
    # PHASE 2: Main Model Training
    # ========================================

    main_model_checkpoint = None

    if args.phase2_epochs > 0:
        print(f"\n{'='*80}")
        print("SETTING UP PHASE 2: MAIN MODEL TRAINING")
        print(f"{'='*80}")

        # Load motifs for main model
        padded_kernels, _ = load_motifs(config, args)

        # Create base ATAC model
        base_model = DNAATACPredictionLightningModule(
            motif_kernels=padded_kernels,
            num_atac_samples=1,  # Only negative samples
            learning_rate=1e-4,
            celltype_ids=args.celltype_ids,
        )

        # Load bias model from Phase 1 if available
        bias_model = None

        # Create bias model structure first
        from .bias_model import BiasModel, load_tn5_motifs_from_meme
        tn5_motifs = load_tn5_motifs_from_meme(args.tn5_motifs_path)
        bias_model = BiasModel(
            tn5_motifs=tn5_motifs,
            num_atac_samples=1,
            freeze_tn5_layer=True,
            use_learnable_kernels=False
        )

        # Try loading bias model using phase_manager
        if args.load_phase1_checkpoint:
            bias_loaded = phase_manager.load_phase1(bias_model, checkpoint_path=args.load_phase1_checkpoint)
            if bias_loaded:
                print("Bias model loaded from specified Phase 1 checkpoint for Phase 2")
                bias_model.eval()
            else:
                print("Failed to load bias model from specified checkpoint")
        elif bias_model_checkpoint or args.phase1_epochs == 0:
            # Try loading from automatic Phase 1 checkpoint
            bias_loaded = phase_manager.load_phase1(bias_model)
            if bias_loaded:
                print("Bias model loaded from Phase 1 training for Phase 2")
                bias_model.eval()
            else:
                print("No Phase 1 checkpoint found - continuing without bias model")

        # Pre-filter summits for Phase 2 (for_count and for_profile regions)
        phase2_summits = load_and_filter_summits(args.summits_csv_path, 'all')

        # Set up data module for Phase 2 with NEW quantile-based curriculum learning on logSumValue_scca_neg
        phase2_curriculum_col = _resolve_curriculum_column(
            args.phase2_curriculum_column, phase2_summits, "logSumValue_", ["logSumValue_"]
        )
        phase2_data_module = DNAATACZarrDataModule(
            atac_zarr_path=args.atac_zarr_path,
            genome_zarr_path=args.genome_zarr_path,
            summits_df=phase2_summits,  # Use pre-filtered DataFrame
            celltype_ids=args.celltype_ids,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            extend_bp=args.extend_bp,
            max_regions=args.max_regions,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            use_curriculum_learning=True,
            curriculum_groups=10,  # This parameter is now ignored in favor of quantile-based approach
            curriculum_column=phase2_curriculum_col,  # Auto-detected from summits DataFrame
            curriculum_initial_epsilon=0.2,  # NEW: Start with epsilon=0.2 for merging quantiles
            curriculum_min_groups=5,  # NEW: Ensure at least 5 groups after merging
        )

        phase2_data_module.setup()

        # Log Phase 2 dataset statistics
        if hasattr(shared_logger, 'experiment'):
            train_dataset_size = len(phase2_data_module.train_dataset) if hasattr(phase2_data_module, 'train_dataset') else "unknown"
            val_dataset_size = len(phase2_data_module.val_dataset) if hasattr(phase2_data_module, 'val_dataset') else "unknown"
            shared_logger.experiment.log({
                "phase2_dataset_stats": {
                    "total_regions": len(phase2_summits),
                    "train_regions": train_dataset_size,
                    "val_regions": val_dataset_size,
                    "filter_type": "all (for_count + for_profile)",
                    "curriculum_learning": True,
                    "curriculum_column": "logSumValue_DZ",
                    "curriculum_groups": 10,
                    "curriculum_initial_epsilon": 0.2,
                    "curriculum_min_groups": 5,
                    "batch_size": args.batch_size,
                    "extend_bp": args.extend_bp
                }
            })

        # Create Phase 2 Lightning module using new architecture
        if bias_model is None:
            # Create a default bias model if not loaded
            print("Warning: No bias model loaded, creating default bias model for Phase 2")
            from .bias_model import BiasModel, load_tn5_motifs_from_meme
            tn5_motifs = load_tn5_motifs_from_meme(args.tn5_motifs_path)
            bias_model = BiasModel(
                tn5_motifs=tn5_motifs,
                num_atac_samples=1,
                freeze_tn5_layer=True,
                use_learnable_kernels=False
            )

        phase2_nn_model = create_phase2_model(base_model.model, bias_model)
        phase2_model = Phase2LightningModule(
            model=phase2_nn_model,
            learning_rate=1e-4,
            celltype_ids=args.celltype_ids
        )

        # Load checkpoint if specified
        if args.load_phase2_checkpoint:
            print(f"\nLoading Phase 2 checkpoint: {args.load_phase2_checkpoint}")

            # Try to load from Phase 2 checkpoint (handles bias model intelligently)
            main_loaded, bias_loaded = phase_manager.load_phase2(
                main_model=phase2_model.model.main_model,
                bias_model=bias_model if bias_model is not None else None,
                checkpoint_path=args.load_phase2_checkpoint
            )

            if main_loaded:
                print("✓ Main model loaded from Phase 2 checkpoint")
            if bias_loaded:
                print("✓ Bias model loaded from Phase 2 checkpoint")
                bias_model.eval()

            if not (main_loaded or bias_loaded):
                print("Warning: No components loaded from Phase 2 checkpoint")
                print("Continuing with fresh Phase 2 model")

        # Compile the DNAATACMotifCNN model for optimal performance
        print(f"\nCompiling DNAATACMotifCNN model for Phase 2...")
        # phase2_model.model = torch.compile(phase2_model.model)
        print("✓ Model compilation successful - training will be optimized")

        # Run Phase 2
        main_model_checkpoint = run_phase('Phase2', phase2_model, phase2_data_module,
                                         args.phase2_epochs, args.run_name, config, shared_logger)

        # Save full model with enhanced metadata
        if main_model_checkpoint:
            checkpoint_path = phase_manager.save_phase2(
                main_model=phase2_model.model.main_model,
                bias_model=phase2_model.model.bias_model if bias_model is not None else None,
                metadata={
                    'epochs': args.phase2_epochs,
                    'best_checkpoint': main_model_checkpoint,
                    'training_completed': True,
                    'model_architecture': 'Phase2MainModel',
                    'curriculum_learning': True,
                    'curriculum_column': 'logSumValue_DZ',
                    'has_bias_model': bias_model is not None,
                },
                config=config
            )

            # Log to wandb with comprehensive information
            if hasattr(shared_logger, 'experiment'):
                shared_logger.experiment.log({
                    "phase2_model_saved": True,
                    "phase2_save_path": str(checkpoint_path),
                    "phase2_model_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
                    "phase2_includes_bias_model": bias_model is not None,
                    "phase2_curriculum_learning": True,
                    "phase2_curriculum_groups": 10,
                    "phase2_regions_used": len(phase2_summits) if 'phase2_summits' in locals() else "unknown"
                })

    # ========================================
    # PHASE 3: Differential Prediction
    # ========================================

    if args.phase3_epochs > 0:
        print(f"\n{'='*80}")
        print("SETTING UP PHASE 3: DIFFERENTIAL PREDICTION")
        print(f"{'='*80}")

        # Recreate models
        padded_kernels, _ = load_motifs(config, args)
        base_model = DNAATACPredictionLightningModule(
            motif_kernels=padded_kernels,
            num_atac_samples=1,
            learning_rate=5e-5,  # Lower LR for Phase 3
            celltype_ids=args.celltype_ids,
        )

        # Create bias model structure
        from .bias_model import BiasModel, load_tn5_motifs_from_meme
        tn5_motifs = load_tn5_motifs_from_meme(args.tn5_motifs_path)
        bias_model = BiasModel(
            tn5_motifs=tn5_motifs,
            num_atac_samples=1,
            freeze_tn5_layer=True,
            use_learnable_kernels=False
        )

        # Try to load models from Phase 2 for Phase 3 initialization
        print("\nLoading models from Phase 2 for Phase 3 initialization...")
        main_loaded, bias_loaded = phase_manager.load_models_for_phase3(
            main_model=base_model.model,
            bias_model=bias_model
        )

        if main_loaded:
            print("✓ Main model loaded from Phase 2 for Phase 3")
        else:
            print("⚠ Main model not loaded from Phase 2 - using fresh initialization")

        if bias_loaded:
            print("✓ Bias model loaded from Phase 2 for Phase 3")
            bias_model.eval()
        else:
            # Try Phase 1 checkpoint as fallback
            if args.load_phase1_checkpoint:
                bias_loaded = phase_manager.load_phase1(bias_model, checkpoint_path=args.load_phase1_checkpoint)
                if bias_loaded:
                    print("✓ Bias model loaded from Phase 1 checkpoint for Phase 3")
                    bias_model.eval()
                else:
                    print("⚠ Bias model not loaded - using fresh initialization")

        # Pre-filter summits for Phase 3 (differential regions only)
        phase3_summits = load_and_filter_summits(args.summits_csv_path, 'differential', args.differential_threshold)

        # Set up data module for Phase 3 with NEW quantile-based curriculum learning on Diff_scca
        phase3_curriculum_col = _resolve_curriculum_column(
            args.phase3_differential_column, phase3_summits, "Diff_", ["Diff_", "logSumValue_"]
        )
        phase3_data_module = DNAATACZarrDataModule(
            atac_zarr_path=args.atac_zarr_path,
            genome_zarr_path=args.genome_zarr_path,
            summits_df=phase3_summits,  # Use pre-filtered DataFrame
            celltype_ids=args.celltype_ids,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            extend_bp=args.extend_bp,
            max_regions=args.max_regions,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            use_curriculum_learning=True,
            curriculum_groups=5,  # This parameter is now ignored in favor of quantile-based approach
            curriculum_column=phase3_curriculum_col,  # Auto-detected from summits DataFrame
            curriculum_initial_epsilon=0.1,  # NEW: Smaller epsilon for differential values (more sensitive)
            curriculum_min_groups=5,  # NEW: Ensure at least 5 groups after merging
        )

        phase3_data_module.setup()

        # Log Phase 3 dataset statistics
        if hasattr(shared_logger, 'experiment'):
            train_dataset_size = len(phase3_data_module.train_dataset) if hasattr(phase3_data_module, 'train_dataset') else "unknown"
            val_dataset_size = len(phase3_data_module.val_dataset) if hasattr(phase3_data_module, 'val_dataset') else "unknown"
            # Analyze Diff_DZ_vs_LZ distribution for logging
            diff_dz_vs_lz_stats = {}
            if 'Diff_DZ_vs_LZ' in phase3_summits.columns:
                diff_dz_vs_lz_values = phase3_summits['Diff_DZ_vs_LZ'].values
                diff_dz_vs_lz_stats = {
                    "mean": float(diff_dz_vs_lz_values.mean()),
                    "std": float(diff_dz_vs_lz_values.std()),
                    "min": float(diff_dz_vs_lz_values.min()),
                    "max": float(diff_dz_vs_lz_values.max()),
                    "median": float(np.median(diff_dz_vs_lz_values)),
                    "positive_regions": int((diff_dz_vs_lz_values > 0).sum()),
                    "negative_regions": int((diff_dz_vs_lz_values < 0).sum()),
                    "high_diff_regions": int((np.abs(diff_dz_vs_lz_values) >= args.differential_threshold*2).sum())
                }

            shared_logger.experiment.log({
                "phase3_dataset_stats": {
                    "total_regions": len(phase3_summits),
                    "train_regions": train_dataset_size,
                    "val_regions": val_dataset_size,
                    "filter_type": f"differential (|Diff_DZ_vs_LZ| >= {args.differential_threshold})",
                    "differential_threshold": args.differential_threshold,
                    "curriculum_learning": True,
                    "curriculum_column": "Diff_DZ_vs_LZ",
                    "curriculum_groups": 5,
                    "curriculum_initial_epsilon": 0.1,
                    "curriculum_min_groups": 5,
                    "batch_size": args.batch_size,
                    "extend_bp": args.extend_bp,
                    "diff_dz_vs_lz_statistics": diff_dz_vs_lz_stats,
                    "evaluation_metrics": {
                        "tracks": ["val_differential_correlation", "val_differential_r2",
                                  "val_differential_mse", "val_differential_mae", "val_differential_spearman"],
                        "primary_metric": "val_differential_correlation",
                        "success_threshold": 0.5
                    }
                }
            })

        # Create Phase 3 Lightning module using new architecture
        phase3_nn_model = create_phase3_model(base_model.model, bias_model, config)
        phase3_model = Phase3LightningModule(
            model=phase3_nn_model,
            learning_rate=5e-5,
            differential_loss_weight=1.0,
            celltype_ids=args.celltype_ids
        )

        # Load Phase 3 checkpoint if specified
        if args.load_phase3_checkpoint:
            print(f"\nLoading Phase 3 checkpoint: {args.load_phase3_checkpoint}")

            num_loaded, loaded_components = phase_manager.load_phase3(
                full_model=phase3_model,
                main_model=phase3_model.model.main_model,
                differential_head=phase3_model.model.differential_head,
                bias_model=phase3_model.model.bias_model if hasattr(phase3_model.model, 'bias_model') else None,
                checkpoint_path=args.load_phase3_checkpoint
            )

            if num_loaded > 0:
                print(f"✓ Loaded {num_loaded} component(s): {', '.join(loaded_components)}")

                # Log to wandb
                if hasattr(shared_logger, 'experiment'):
                    metadata = phase_manager.phase3_manager.get_metadata(args.load_phase3_checkpoint)
                    shared_logger.experiment.log({
                        "phase3_checkpoint_loading": {
                            "loaded_from": args.load_phase3_checkpoint,
                            "num_components": num_loaded,
                            "components": loaded_components,
                            "loading_successful": True,
                            **metadata
                        }
                    })
            else:
                print("Warning: No components loaded from Phase 3 checkpoint")

                # Log error to wandb
                if hasattr(shared_logger, 'experiment'):
                    shared_logger.experiment.log({
                        "phase3_checkpoint_loading": {
                            "attempted_path": args.load_phase3_checkpoint,
                            "loading_successful": False,
                            "error": "No components could be loaded"
                        }
                    })

        # Compile the DNAATACMotifCNN model for optimal performance
        print(f"\nCompiling DNAATACMotifCNN model for Phase 3...")
        # phase3_model.model = torch.compile(phase3_model.model)
        print("✓ Model compilation successful - training will be optimized")

        # Run Phase 3
        final_checkpoint = run_phase('Phase3', phase3_model, phase3_data_module,
                                   args.phase3_epochs, args.run_name, config, shared_logger)

        # Save final model with comprehensive metadata
        if final_checkpoint:
            checkpoint_path = phase_manager.save_phase3(
                full_model=phase3_model,
                main_model=phase3_model.model.main_model,
                differential_head=phase3_model.model.differential_head,
                bias_model=phase3_model.model.bias_model if hasattr(phase3_model.model, 'bias_model') else None,
                metadata={
                    'epochs': args.phase3_epochs,
                    'best_checkpoint': final_checkpoint,
                    'training_completed': True,
                    'model_architecture': 'Phase3DifferentialModel',
                    'differential_threshold': args.differential_threshold,
                    'curriculum_learning': True,
                    'curriculum_column': 'Diff_DZ_vs_LZ',
                    'all_phases_completed': True,
                    'total_training_epochs': args.phase1_epochs + args.phase2_epochs + args.phase3_epochs,
                    'has_bias_model': bias_model is not None,
                },
                config=config
            )

            # Log comprehensive final model information to wandb
            if hasattr(shared_logger, 'experiment'):
                shared_logger.experiment.log({
                    "phase3_model_saved": True,
                    "phase3_save_path": str(checkpoint_path),
                    "phase3_model_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
                    "phase3_includes_bias_model": bias_model is not None,
                    "phase3_differential_threshold": args.differential_threshold,
                    "phase3_curriculum_learning": True,
                    "phase3_curriculum_groups": 5,
                    "phase3_regions_used": len(phase3_summits) if 'phase3_summits' in locals() else "unknown",
                    "final_model_components": [
                        "main_model (ATAC prediction)",
                        "differential_head (differential prediction)",
                        "bias_model (TN5 bias correction)" if bias_model is not None else "no bias model"
                    ]
                })

    print(f"\n{'='*80}")
    print("THREE-PHASE TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Run name: {args.run_name}")

    # Print summary of saved checkpoints
    checkpoints = []
    for phase, epochs in [('phase1', args.phase1_epochs), ('phase2', args.phase2_epochs), ('phase3', args.phase3_epochs)]:
        if epochs > 0:
            if phase == 'phase1':
                checkpoints.append(f"  Phase 1 (bias): checkpoints/{args.run_name}_phase1_bias_model.ckpt")
            elif phase == 'phase2':
                checkpoints.append(f"  Phase 2 (main): checkpoints/{args.run_name}_phase2_full_model.ckpt")
            elif phase == 'phase3':
                checkpoints.append(f"  Phase 3 (final): checkpoints/{args.run_name}_phase3_final_model.ckpt")

    if checkpoints:
        print("\nSaved checkpoints:")
        for checkpoint in checkpoints:
            print(checkpoint)

    # Log comprehensive final completion summary to wandb
    if hasattr(shared_logger, 'experiment'):
        phases_run = [p for p, e in [
            ('phase1', args.phase1_epochs),
            ('phase2', args.phase2_epochs),
            ('phase3', args.phase3_epochs)
        ] if e > 0]

        total_epochs = sum([args.phase1_epochs, args.phase2_epochs, args.phase3_epochs])

        # Create comprehensive training summary
        training_summary = {
            "all_phases_completed": True,
            "total_phases_run": len(phases_run),
            "phases_executed": phases_run,
            "final_run_name": args.run_name,
            "total_training_epochs": total_epochs,
            "phase_breakdown": {
                "phase1_epochs": args.phase1_epochs,
                "phase2_epochs": args.phase2_epochs,
                "phase3_epochs": args.phase3_epochs
            },
            "training_configuration": {
                "differential_threshold": args.differential_threshold,
                "batch_size": args.batch_size,
                "max_regions": args.max_regions,
                "celltype_ids": args.celltype_ids,
                "config_file": args.config
            },
            "curriculum_learning": {
                "phase1": False,
                "phase2": True,
                "phase2_column": "logSumValue_DZ",
                "phase2_groups": 10,
                "phase3": True,
                "phase3_column": "Diff_DZ_vs_LZ",
                "phase3_groups": 5
            },
            "model_checkpoints": checkpoints if 'checkpoints' in locals() else []
        }

        shared_logger.experiment.log(training_summary)

        # Also log as a summary table for easy viewing
        import wandb
        summary_table_data = [
            ["Phase", "Epochs", "Status", "Curriculum Learning", "Focus"],
            ["Phase 1 (Bias)", args.phase1_epochs, "✓ Completed" if args.phase1_epochs > 0 else "⏭ Skipped", "No", "TN5 bias correction"],
            ["Phase 2 (Main)", args.phase2_epochs, "✓ Completed" if args.phase2_epochs > 0 else "⏭ Skipped", "Yes (logSumValue_DZ)", "ATAC signal prediction"],
            ["Phase 3 (Differential)", args.phase3_epochs, "✓ Completed" if args.phase3_epochs > 0 else "⏭ Skipped", "Yes (Diff_DZ_vs_LZ)", "Differential accessibility"],
            ["Total", total_epochs, f"✓ {len(phases_run)}/{3} phases completed", "-", "Sequential phased training"]
        ]

        summary_table = wandb.Table(
            data=summary_table_data[1:],  # Skip header
            columns=summary_table_data[0]
        )

        shared_logger.experiment.log({
            "training_summary_table": summary_table,
            "experiment_status": "COMPLETED",
            "experiment_type": "sequential_three_phase_atac_training"
        })

    print("\nTraining completed successfully!")
    print(f"All results logged to wandb run: {args.run_name}")

    # Log final experiment artifacts and metadata
    if hasattr(shared_logger, 'experiment'):
        # Log system information

        system_info = {
            "system_platform": platform.system(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

        # Create final experiment metadata
        final_metadata = {
            "experiment_completed_successfully": True,
            "experiment_type": "sequential_three_phase_atac_training",
            "total_runtime_info": "completed_all_phases",
            "system_info": system_info,
            "final_artifacts": {
                "config_file": args.config,
                "run_name": args.run_name,
                "checkpoints_saved": len(checkpoints) if 'checkpoints' in locals() else 0,
                "wandb_project": wandb_config.get('wandb_project', 'ATAC-Phased-Learning')
            },
            "experiment_summary": {
                "phases_completed": len(phases_run),
                "total_epochs": total_epochs,
                "phase1_type": "bias_correction",
                "phase2_type": "main_atac_prediction",
                "phase3_type": "differential_prediction",
                "curriculum_learning_enabled": True,
                "differential_threshold": args.differential_threshold
            }
        }

        shared_logger.experiment.log(final_metadata)

        # Create a final summary image for the experiment
        try:
            import matplotlib.pyplot as plt
            import io
            from PIL import Image

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Sequential ATAC Training Summary: {args.run_name}', fontsize=16, fontweight='bold')

            # Phase breakdown
            phase_names = ['Phase 1\n(Bias)', 'Phase 2\n(Main)', 'Phase 3\n(Diff)']
            phase_epochs = [args.phase1_epochs, args.phase2_epochs, args.phase3_epochs]
            colors = ['lightcoral', 'lightblue', 'lightgreen']

            ax1.bar(phase_names, phase_epochs, color=colors)
            ax1.set_title('Training Epochs per Phase')
            ax1.set_ylabel('Epochs')
            for i, v in enumerate(phase_epochs):
                ax1.text(i, v + max(phase_epochs)*0.01, str(v), ha='center', fontweight='bold')

            # Training progression
            cumulative_epochs = [sum(phase_epochs[:i+1]) for i in range(len(phase_epochs))]
            ax2.plot(range(1, len(cumulative_epochs)+1), cumulative_epochs, 'o-', linewidth=2, markersize=8)
            ax2.set_title('Cumulative Training Progress')
            ax2.set_xlabel('Phase')
            ax2.set_ylabel('Total Epochs')
            ax2.set_xticks(range(1, len(phase_names)+1))
            ax2.set_xticklabels([f'Phase {i+1}' for i in range(len(phase_names))])
            ax2.grid(True, alpha=0.3)

            # Dataset statistics (if available)
            dataset_info = [
                f"Phase 1: Background regions\n(Bias correction)",
                f"Phase 2: All regions\n(ATAC prediction)",
                f"Phase 3: Differential regions\n(|Diff_DZ_vs_LZ| ≥ {args.differential_threshold})"
            ]
            ax3.text(0.1, 0.8, "Dataset Configuration:", fontsize=12, fontweight='bold', transform=ax3.transAxes)
            for i, info in enumerate(dataset_info):
                ax3.text(0.1, 0.7 - i*0.15, f"• {info}", fontsize=10, transform=ax3.transAxes)
            ax3.text(0.1, 0.2, f"Batch size: {args.batch_size}", fontsize=10, transform=ax3.transAxes)
            ax3.text(0.1, 0.1, f"Max regions: {args.max_regions}", fontsize=10, transform=ax3.transAxes)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            ax3.set_title('Training Configuration')

            # Model architecture
            model_info = [
                "Sequential Three-Phase Architecture:",
                "• Phase 1: BiasModel (TN5 correction)",
                "• Phase 2: MainModel (ATAC prediction)",
                "• Phase 3: DifferentialHead (diff. analysis)",
                "",
                "Features:",
                "• Curriculum learning (Phases 2-3)",
                "• Model compilation (torch.compile)",
                "• Comprehensive WandB logging",
                f"• Cell types: {', '.join(args.celltype_ids)}"
            ]
            ax4.text(0.1, 0.9, "\n".join(model_info), fontsize=9, transform=ax4.transAxes, verticalalignment='top')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Model Architecture')

            plt.tight_layout()

            # Convert to wandb image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)

            shared_logger.experiment.log({
                "experiment_summary_visualization": wandb.Image(img)
            })

        except Exception as e:
            print(f"Could not create summary visualization: {e}")

        # Finish wandb run properly
        shared_logger.experiment.finish()
        print("✓ WandB experiment finished and logged successfully")


if __name__ == "__main__":
    main()