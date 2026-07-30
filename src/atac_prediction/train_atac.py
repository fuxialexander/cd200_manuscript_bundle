"""
Filename: train_atac.py
Purpose: Complete training pipeline for DNA->ATAC prediction models using PyTorch Lightning

Main Components:
- GradientMonitorCallback: Callback for monitoring and handling problematic gradients
- Command-line argument parsing for data, model, and training configurations
- Motif loading from various sources (pre-processed .pt files, PWM files)
- Integration with DNAATACZarrDataModule for efficient data loading
- Comprehensive training setup with checkpointing, early stopping, and logging
- Support for resuming training from checkpoints
- Built-in testing functionality for dataset validation

Dependencies: pytorch_lightning, torch, argparse, wandb, utils (motif processing)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from .dna_atac_lightning_module import DNAATACPredictionLightningModule
from .dna_atac_zarr_datamodule import DNAATACZarrDataModule
from utils import pad_motifs_simple, read_pwms_simple

# Add parent directory to path for config imports
sys.path.append(str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config_utils import ConfigManager

# Set precision for improved performance
torch.set_float32_matmul_precision('high')


class GradientMonitorCallback(pl.Callback):
    """Callback to monitor gradients for NaN/Inf values and prevent training crashes.
    
    This callback helps debug training issues by:
    - Detecting NaN/Inf gradients before they propagate
    - Logging batch statistics when problems occur
    - Zeroing problematic gradients to continue training
    """
    
    def __init__(self):
        super().__init__()
        self.current_batch: Optional[dict] = None
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Store current batch for gradient analysis."""
        self.current_batch = batch
    
    def on_after_backward(self, trainer, pl_module):
        """Check for NaN/Inf gradients and log diagnostics."""
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
                
            if torch.isnan(param.grad).any():
                self._handle_nan_gradient(name)
                param.grad.zero_()
            elif torch.isinf(param.grad).any():
                self._handle_inf_gradient(name)
                param.grad.zero_()
    
    def _handle_nan_gradient(self, param_name: str) -> None:
        """Handle NaN gradient detection."""
        print(f"NaN gradient detected in {param_name}")
        if self.current_batch is not None:
            self._log_batch_stats()
    
    def _handle_inf_gradient(self, param_name: str) -> None:
        """Handle Inf gradient detection."""
        print(f"Inf gradient detected in {param_name}")
        if self.current_batch is not None:
            self._log_batch_stats(detailed=True)
    
    def _log_batch_stats(self, detailed: bool = False) -> None:
        """Log current batch statistics."""
        if self.current_batch is None:
            return
            
        atac_data = self.current_batch['atac']
        if detailed:
            print(f"  Current batch ATAC stats:")
            print(f"    Max: {atac_data.max().item():.6f}")
            print(f"    Mean: {atac_data.mean().item():.6f}")
            print(f"    Min: {atac_data.min().item():.6f}")
            print(f"    Std: {atac_data.std().item():.6f}")
            
            max_val = atac_data.max().item()
            if max_val > 1000:
                print(f"    WARNING: Very high ATAC signal detected (>{max_val})")
        else:
            max_atac = atac_data.max().item()
            mean_atac = atac_data.mean().item()
            print(f"  Current batch ATAC - Max: {max_atac:.6f}, Mean: {mean_atac:.6f}")


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add data-related arguments to parser."""
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        "--atac-zarr-path",
        default=os.getenv("ATAC_ZARR_PATH"),
        help="Path to ATAC zarr file"
    )
    data_group.add_argument(
        "--genome-zarr-path",
        default=os.getenv("GENOME_ZARR_PATH"),
        help="Path to genome zarr file"
    )
    data_group.add_argument(
        "--summits-csv-path",
        default=os.getenv("SUMMITS_CSV_PATH"),
        help="Path to summits CSV file"
    )
    data_group.add_argument(
        "--celltype-ids", 
        nargs='+',
        default=os.getenv("CELLTYPE_IDS", "SCCApos.cd200_scc,SCCAneg.cd200_scc").split(','),
        help="List of celltype IDs"
    )
    data_group.add_argument(
        "--batch-size", 
        type=int,
        default=int(os.getenv("BATCH_SIZE", "32")),
        help="Training batch size"
    )
    data_group.add_argument(
        "--num-workers", 
        type=int,
        default=int(os.getenv("NUM_WORKERS", "16")),
        help="Number of data loader workers"
    )
    data_group.add_argument(
        "--extend-bp", 
        type=int,
        default=int(os.getenv("EXTEND_BP", "2048")),
        help="Sequence extension length in base pairs"
    )
    data_group.add_argument(
        "--max-regions", 
        type=int,
        default=int(os.getenv("MAX_REGIONS", "80000")),
        help="Maximum number of regions to process"
    )
    data_group.add_argument(
        "--cache-dir",
        default=os.getenv("CACHE_DIR", "./zarr_cache"),
        help="Cache directory for zarr data"
    )
    data_group.add_argument(
        "--use-data-cache", 
        action="store_true",
        help="Enable data caching (default: load directly from zarr)"
    )
    
    # Curriculum learning arguments
    data_group.add_argument(
        "--use-curriculum-learning",
        action="store_true",
        help="Enable curriculum learning based on MaxValue quantiles"
    )
    data_group.add_argument(
        "--curriculum-groups",
        type=int,
        default=5,
        help="Number of curriculum groups to split MaxValue into"
    )
    data_group.add_argument(
        "--curriculum-sampling-strategy",
        choices=['equal', 'proportional', 'random'],
        default='equal',
        help="Curriculum sampling strategy for balanced group representation"
    )
    
    # Data splits
    data_group.add_argument(
        "--split-strategy",
        choices=['random', 'chromosome', 'borzoi'],
        default='random',
        help="Data splitting strategy: random, chromosome-based, or borzoi fold-based"
    )
    data_group.add_argument(
        "--train-split", 
        type=float,
        default=float(os.getenv("TRAIN_SPLIT", "0.75")),
        help="Training data split ratio (only used with random splitting)"
    )
    data_group.add_argument(
        "--val-split", 
        type=float,
        default=float(os.getenv("VAL_SPLIT", "0.15")),
        help="Validation data split ratio (only used with random splitting)"
    )
    data_group.add_argument(
        "--test-split", 
        type=float,
        default=float(os.getenv("TEST_SPLIT", "0.10")),
        help="Test data split ratio (only used with random splitting)"
    )
    
    # Chromosome-based splits
    data_group.add_argument(
        "--val-chr",
        default="chr14",
        help="Chromosome to use for validation set (overrides val-split ratio)"
    )
    data_group.add_argument(
        "--test-chr",
        default="chr10", 
        help="Chromosome to use for test set (overrides test-split ratio)"
    )
    
    # Borzoi-style fold-based splits
    data_group.add_argument(
        "--borzoi-fold",
        action="store_true",
        help="Use Borzoi-style fold-based data splitting from sequences_human.bed"
    )
    data_group.add_argument(
        "--borzoi-val-fold",
        default="fold6,fold7",
        help="Comma-separated list of validation folds (default: fold6,fold7)"
    )
    data_group.add_argument(
        "--sequences-human-bed",
        default="sequences_human.bed",
        help="Path to sequences_human.bed file for Borzoi-style fold splitting"
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model-related arguments to parser."""
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--pwm-path",
        help="Path to PWM file (overrides config)"
    )
    model_group.add_argument(
        "--motif-filter-prefix",
        help="Motif prefix to filter by (overrides config)"
    )
    model_group.add_argument(
        "--aligned-motifs-path",
        help="Path to aligned motifs with reverse complements .pt file (overrides config)"
    )
    model_group.add_argument(
        "--simple-motifs-path",
        help="Path to simple padded motifs with reverse complements .pt file (overrides config)"
    )
    model_group.add_argument(
        "--learning-rate", 
        type=float,
        help="Learning rate (overrides config)"
    )
    model_group.add_argument(
        "--use-random-motif-init", 
        action="store_true",
        help="Use random motif initialization instead of PWM-based initialization"
    )
    model_group.add_argument(
        "--no-random-motif-init", 
        action="store_true",
        help="Disable random motif initialization (use PWM-based initialization)"
    )
    model_group.add_argument(
        "--num-random-kernels", 
        type=int,
        help="Number of random kernels to use when --use-random-motif-init is enabled (overrides config)"
    )
    
    # Chromosome prediction arguments
    model_group.add_argument(
        "--enable-chromosome-prediction",
        action="store_true",
        help="Enable chromosome classification head"
    )
    model_group.add_argument(
        "--no-chromosome-prediction",
        action="store_true", 
        help="Disable chromosome classification head (explicit override)"
    )
    model_group.add_argument(
        "--num-chromosomes",
        type=int,
        default=25,
        help="Number of chromosome classes to predict (default: 25)"
    )
    model_group.add_argument(
        "--chromosome-loss-weight",
        type=float,
        default=1.0,
        help="Weight for chromosome prediction loss (default: 1.0)"
    )
    model_group.add_argument(
        "--freeze-atac-components",
        action="store_true",
        help="Freeze ATAC prediction components for chromosome-only training"
    )
    model_group.add_argument(
        "--use-compression", 
        action="store_true",
        help="Use compressed model architecture for faster training"
    )
    model_group.add_argument(
        "--compression-strategy", 
        choices=["4bp", "convtranspose"],
        default="4bp",
        help="Compression strategy: '4bp' for 4bp resolution output (fastest), 'convtranspose' for full resolution"
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training-related arguments to parser."""
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument(
        "--max-epochs", 
        type=int,
        default=int(os.getenv("MAX_EPOCHS", "100")),
        help="Maximum number of training epochs"
    )
    training_group.add_argument(
        "--precision",
        default=os.getenv("PRECISION", "16-mixed"),
        help="Training precision"
    )
    training_group.add_argument(
        "--val-check-interval", 
        type=float,
        default=float(os.getenv("VAL_CHECK_INTERVAL", "0.5")),
        help="Validation check interval"
    )
    training_group.add_argument(
        "--gradient-clip-val", 
        type=float,
        default=float(os.getenv("GRADIENT_CLIP_VAL", "0.5")),
        help="Gradient clipping value"
    )
    
    # Logging arguments
    training_group.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "DNA-ATAC-Zarr-Lightning-CD200"),
        help="Weights & Biases project name"
    )
    training_group.add_argument(
        "--run-name",
        default=os.getenv("RUN_NAME"),
        help="Experiment run name"
    )
    training_group.add_argument(
        "--resume-from",
        default=os.getenv("RESUME_FROM"),
        help="Checkpoint to resume from (continues from saved epoch)"
    )
    training_group.add_argument(
        "--load-weights-from",
        default=os.getenv("LOAD_WEIGHTS_FROM"),
        help="Checkpoint to load weights from (starts training from epoch 0)"
    )
    training_group.add_argument(
        "--find-lr",
        action="store_true",
        help="Find optimal learning rate before training"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DNA->ATAC prediction model with PyTorch Lightning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add config argument first
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file"
    )
    
    _add_data_arguments(parser)
    _add_model_arguments(parser)
    _add_training_arguments(parser)
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate required arguments and configurations."""
    required_paths = {
        'atac_zarr_path': 'ATAC zarr path must be provided via --atac-zarr-path or ATAC_ZARR_PATH environment variable',
        'genome_zarr_path': 'Genome zarr path must be provided via --genome-zarr-path or GENOME_ZARR_PATH environment variable',
        'summits_csv_path': 'Summits CSV path must be provided via --summits-csv-path or SUMMITS_CSV_PATH environment variable'
    }
    
    for attr, error_msg in required_paths.items():
        if not getattr(args, attr):
            raise ValueError(error_msg)
    
    # Validate split ratios sum to 1.0
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Train/val/test splits must sum to 1.0, got {total_split}")
    
    # Validate that resume-from and load-weights-from are mutually exclusive
    if getattr(args, 'resume_from', None) and getattr(args, 'load_weights_from', None):
        raise ValueError("Cannot use both --resume-from and --load-weights-from. Use --resume-from to continue training from checkpoint, or --load-weights-from to load only weights and start from epoch 0.")


def _get_random_motif_init_setting(config: dict, args: argparse.Namespace = None) -> bool:
    """Get the random motif initialization setting from config and args."""
    use_random_motif_init = config['model'].get('use_random_motif_init', False)
    if hasattr(args, 'use_random_motif_init') and args.use_random_motif_init:
        use_random_motif_init = True
    if hasattr(args, 'no_random_motif_init') and args.no_random_motif_init:
        use_random_motif_init = False
    return use_random_motif_init


def load_motifs(config: dict, args: argparse.Namespace = None) -> Tuple[torch.Tensor, List[str]]:
    """Load motif kernels from available sources.
    
    Args:
        config: Configuration dictionary
        args: Optional command line arguments to override config
    
    Returns:
        Tuple of (padded_kernels, motif_names)
    """
    use_random_motif_init = _get_random_motif_init_setting(config, args)
    
    if use_random_motif_init:
        # Return None for motif kernels when using random initialization
        # The number of kernels will be handled by the model
        return None, []
    
    # Get paths from config, allowing command line overrides
    simple_motifs_path = getattr(args, 'simple_motifs_path', None) or config['paths'].get('simple_motifs_path')
    aligned_motifs_path = getattr(args, 'aligned_motifs_path', None) or config['paths'].get('aligned_motifs_path')
    pwm_path = getattr(args, 'pwm_path', None) or config['paths'].get('pwm_path')
    
    # Get motif filter prefix from config or args
    motif_filter_prefix = getattr(args, 'motif_filter_prefix', None) or config['model'].get('motif_filter_prefix')
    
    # Check for motif source - prefer simple motifs, then aligned, then PWM
    motif_sources = [
        (simple_motifs_path, "simple motifs"),
        (aligned_motifs_path, "aligned motifs"), 
        (pwm_path, "PWM file")
    ]
    
    available_sources = [(path, desc) for path, desc in motif_sources if path and os.path.exists(path)]
    
    if not available_sources:
        raise ValueError("No motif source found. Check config paths or provide: --simple-motifs-path, --aligned-motifs-path, or --pwm-path")
    
    # Use the first available source
    motif_path, motif_desc = available_sources[0]
    print(f"Using {motif_desc}: {motif_path}")
    
    if motif_path.endswith('.pt'):
        return _load_motifs_from_pt(motif_path, motif_filter_prefix)
    else:
        return _load_motifs_from_pwm(motif_path, motif_filter_prefix)




def _load_motifs_from_pt(motif_path: str, filter_prefix: Optional[str]) -> Tuple[torch.Tensor, List[str]]:
    """Load motifs from .pt file."""
    print(f"Loading motifs with reverse complements from: {motif_path}")
    motif_data = torch.load(motif_path, map_location='cpu', weights_only=False)
    padded_kernels = motif_data['motif_kernels'].numpy()
    motif_names = motif_data['motif_names']
    
    # Apply filter if specified
    if filter_prefix:
        filtered_indices = [i for i, name in enumerate(motif_names) if name.startswith(filter_prefix)]
        if not filtered_indices:
            raise ValueError(f"No motifs found with prefix '{filter_prefix}' in motifs file")
        padded_kernels = padded_kernels[filtered_indices]
        motif_names = [motif_names[i] for i in filtered_indices]
    
    padding_method = motif_data.get('padding_method', 'unknown')
    kernel_length = motif_data.get('kernel_length', padded_kernels.shape[1])
    print(f"Loaded {len(motif_names)} motifs with reverse complements ({padding_method} padding)")
    print(f"Motif dimensions: {padded_kernels.shape}")
    print(f"Kernel length: {kernel_length}")
    
    return torch.tensor(padded_kernels), motif_names


def _load_motifs_from_pwm(motif_path: str, filter_prefix: Optional[str]) -> Tuple[torch.Tensor, List[str]]:
    """Load motifs from PWM file."""
    print(f"Loading from PWM file: {motif_path}")
    motifs_dict = read_pwms_simple(
        motif_path,
        max_motifs=None,
        filter_prefix=filter_prefix
    )
    if not motifs_dict:
        raise ValueError(f"No motifs found with prefix '{filter_prefix}'")
    
    padded_kernels, _ = pad_motifs_simple(motifs_dict)
    motif_names = list(motifs_dict.keys())
    
    print(f"Loaded {len(motifs_dict)} motifs (without reverse complements)")
    print("Consider creating motifs with reverse complements using create_motifs_with_rc_simple.py")
    
    return torch.tensor(padded_kernels), motif_names


def setup_callbacks_and_logger(args: argparse.Namespace) -> Tuple[List[pl.Callback], WandbLogger]:
    """Set up training callbacks and logger."""
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename=f'{args.run_name}-{{epoch:02d}}-{{val_loss:.2f}}',
            save_top_k=3,
            mode='min',
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min'
        ),
        GradientMonitorCallback()
    ]
    
    # Add curriculum learning callback if curriculum learning is enabled
    if hasattr(args, 'use_curriculum_learning') and args.use_curriculum_learning:
        callbacks.append(CurriculumLearningCallback())
        print("Added CurriculumLearningCallback for epoch-wise downsampling")
    
    logger = WandbLogger(project=args.wandb_project, name=args.run_name)
    
    return callbacks, logger


def test_data_loading(data_module: DNAATACZarrDataModule) -> None:
    """Test data loading before training."""
    print("\n6. Testing data loading...")
    try:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"Train batches per epoch: {len(train_loader)}")
        print(f"Val batches per epoch: {len(val_loader)}")
        
        # Test loading a batch
        batch = next(iter(train_loader))
        print(f"Batch ATAC shape: {batch['atac'].shape}")
        print(f"Batch sequence shape: {batch['sequence'].shape}")
        print("Data loading test successful!")
        
    except Exception as e:
        print(f"Error testing data loading: {e}")
        raise


def _override_config_with_args(config: dict, args: argparse.Namespace) -> None:
    """Override configuration values with command line arguments."""
    # Model-related overrides
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    if hasattr(args, 'motif_filter_prefix') and args.motif_filter_prefix is not None:
        config.setdefault('model', {})['motif_filter_prefix'] = args.motif_filter_prefix
    
    # Random motif initialization overrides
    if hasattr(args, 'use_random_motif_init') and args.use_random_motif_init:
        config.setdefault('model', {})['use_random_motif_init'] = True
    
    if hasattr(args, 'no_random_motif_init') and args.no_random_motif_init:
        config.setdefault('model', {})['use_random_motif_init'] = False
    
    if hasattr(args, 'num_random_kernels') and args.num_random_kernels is not None:
        config.setdefault('model', {})['num_random_kernels'] = args.num_random_kernels


def main() -> None:
    """Main training function."""
    # Parse arguments and load configuration
    args = parse_arguments()
    config_manager = ConfigManager("train_atac")
    config = config_manager.load_config(args.config)
    
    # Load split strategy from config if not specified via command line
    if hasattr(args, 'split_strategy') and args.split_strategy == 'random':
        # Check if config specifies a different strategy
        split_strategy = config.get('splits', {}).get('strategy', 'random')
        if split_strategy != 'random':
            args.split_strategy = split_strategy
            print(f"Using split strategy from config: {split_strategy}")
    
    # Load split-specific config values
    if 'splits' in config:
        splits_config = config['splits']
        if not hasattr(args, 'val_chr') or args.val_chr == "chr14":  # Default value
            args.val_chr = splits_config.get('val_chr', args.val_chr)
        if not hasattr(args, 'test_chr') or args.test_chr == "chr10":  # Default value
            args.test_chr = splits_config.get('test_chr', args.test_chr)
        if not hasattr(args, 'borzoi_val_fold') or args.borzoi_val_fold == "fold6,fold7":  # Default value
            args.borzoi_val_fold = splits_config.get('borzoi_val_fold', args.borzoi_val_fold)
        if not hasattr(args, 'sequences_human_bed') or args.sequences_human_bed == "sequences_human.bed":  # Default value
            args.sequences_human_bed = splits_config.get('sequences_human_bed', args.sequences_human_bed)
    
    # Override config with command line arguments
    _override_config_with_args(config, args)
    
    # Validate arguments
    validate_arguments(args)
    
    # Generate run name if not provided
    if not args.run_name:
        args.run_name = f"DNA-ATAC-zarr-{len(args.celltype_ids)}samples-{args.max_regions}regions"

    print("=== DNA->ATAC Training with Direct Zarr Loading ===")
    print(f"ATAC zarr: {args.atac_zarr_path}")
    print(f"Genome zarr: {args.genome_zarr_path}")
    print(f"Summits CSV: {args.summits_csv_path}")
    print(f"Celltype IDs: {args.celltype_ids}")
    print(f"Max regions: {args.max_regions}")
    print(f"Batch size: {args.batch_size}")
    print(f"Extend bp: {args.extend_bp}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Use data cache: {args.use_data_cache}")
    print(f"Curriculum learning: {args.use_curriculum_learning}")
    if args.use_curriculum_learning:
        print(f"  Curriculum groups: {args.curriculum_groups}")
        print(f"  Sampling strategy: {args.curriculum_sampling_strategy}")
    
    print(f"Split strategy: {args.split_strategy}")
    if args.split_strategy == 'borzoi' or args.borzoi_fold:
        print(f"Borzoi fold-based splitting: enabled")
        print(f"  Validation folds: {args.borzoi_val_fold}")
        print(f"  Sequences bed file: {args.sequences_human_bed}")
    elif args.split_strategy == 'chromosome':
        print(f"Chromosome-based splitting: enabled")
        print(f"  Validation chromosome: {args.val_chr}")
        print(f"  Test chromosome: {args.test_chr}")
    else:  # random
        print(f"Random splitting: enabled")
        print(f"  Train: {args.train_split}, Val: {args.val_split}, Test: {args.test_split}")

    # Load motifs
    print("\n1. Loading motifs...")
    padded_kernels, motif_names = load_motifs(config, args)
    
    # Set up Data Module
    print("\n2. Setting up zarr data module...")
    # Only use cache if explicitly requested
    cache_dir = args.cache_dir if args.use_data_cache else None
    
    # Determine split parameters based on strategy
    if args.split_strategy == 'random':
        use_val_chr = None
        use_test_chr = None
        use_borzoi_fold = False
    elif args.split_strategy == 'chromosome':
        use_val_chr = args.val_chr
        use_test_chr = args.test_chr
        use_borzoi_fold = False
    elif args.split_strategy == 'borzoi':
        use_val_chr = None
        use_test_chr = None
        use_borzoi_fold = True
    else:
        # Fallback to legacy behavior for backward compatibility
        use_val_chr = args.val_chr if not args.borzoi_fold else None
        use_test_chr = args.test_chr if not args.borzoi_fold else None
        use_borzoi_fold = args.borzoi_fold
    
    data_module = DNAATACZarrDataModule(
        atac_zarr_path=args.atac_zarr_path,
        genome_zarr_path=args.genome_zarr_path,
        summits_csv_path=args.summits_csv_path,
        celltype_ids=args.celltype_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        extend_bp=args.extend_bp,
        max_regions=args.max_regions,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        cache_dir=cache_dir,
        use_curriculum_learning=args.use_curriculum_learning,
        curriculum_groups=args.curriculum_groups,
        curriculum_sampling_strategy=args.curriculum_sampling_strategy,
        val_chr=use_val_chr,
        test_chr=use_test_chr,
        borzoi_fold=use_borzoi_fold,
        borzoi_val_fold=args.borzoi_val_fold,
        sequences_human_bed=args.sequences_human_bed
    )
    
    # Setup datasets to get info
    data_module.setup()
    data_module.summary()

    # --- 3. Set up Lightning Module ---
    print("\n3. Setting up lightning module...")
    
    use_random_motif_init = _get_random_motif_init_setting(config, args)
    
    # Get number of random kernels
    num_random_kernels = getattr(args, 'num_random_kernels', None) or config['model'].get('num_random_kernels')
    
    # Choose model type based on compression setting
    if getattr(args, 'use_compression', False):
        print(f"Compression requested but DNAATACPredictionLightningModuleCompressed not available")
        print("Using standard model instead")
    
    print("Using standard model")
    
    # Get chromosome prediction parameters (CLI args override config)
    enable_chromosome_prediction = config['model'].get('enable_chromosome_prediction', False)
    if hasattr(args, 'enable_chromosome_prediction') and args.enable_chromosome_prediction:
        enable_chromosome_prediction = True
    elif hasattr(args, 'no_chromosome_prediction') and args.no_chromosome_prediction:
        enable_chromosome_prediction = False
    
    num_chromosomes = getattr(args, 'num_chromosomes', None) or config['model'].get('num_chromosomes', 25)
    chromosome_loss_weight = getattr(args, 'chromosome_loss_weight', None) or config['model'].get('chromosome_loss_weight', 1.0)
    freeze_atac_components = getattr(args, 'freeze_atac_components', False) or config['model'].get('freeze_atac_components', False)
    
    model = DNAATACPredictionLightningModule(
        motif_kernels=padded_kernels,
        num_atac_samples=len(args.celltype_ids),
        learning_rate=args.learning_rate,
        celltype_ids=args.celltype_ids,  # Pass celltype IDs for plotting
        use_random_motif_init=use_random_motif_init,
        num_random_kernels=num_random_kernels,
        enable_chromosome_prediction=enable_chromosome_prediction,
        num_chromosomes=num_chromosomes,
        chromosome_loss_weight=chromosome_loss_weight,
        freeze_atac_components=freeze_atac_components,
    )
    
    if enable_chromosome_prediction:
        print(f"Chromosome prediction enabled:")
        print(f"  Number of chromosomes: {num_chromosomes}")
        print(f"  Loss weight: {chromosome_loss_weight}")
        if freeze_atac_components:
            print(f"  ATAC components: FROZEN (chromosome-only training)")
        else:
            print(f"  ATAC components: trainable (joint training)")

    # --- 4. Set up Callbacks and Logger ---
    print("\n4. Setting up callbacks and logger...")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename=f'{args.run_name}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=3,  # Save top 3 models
        mode='min',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min'
    )

    gradient_monitor_callback = GradientMonitorCallback()

    # Use standard callbacks
    callbacks = [checkpoint_callback, early_stopping_callback, gradient_monitor_callback]

    wandb_logger = WandbLogger(project=args.wandb_project, name=args.run_name)

    # --- 5. Initialize and Run Trainer ---
    print("\n5. Starting training...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=args.gradient_clip_val,
        detect_anomaly=False,  # Enable anomaly detection for debugging
    )

    # Test data loading before training
    test_data_loading(data_module)

    # Learning rate finder (optional)
    if args.find_lr:
        print("\n7. Finding optimal learning rate...")
        try:
            # Try the newer Lightning API first
            from pytorch_lightning.tuner import Tuner
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(model, datamodule=data_module)
        except (ImportError, AttributeError):
            # Fallback to older API if available
            try:
                lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)
            except AttributeError:
                # Manual implementation as last resort
                print("Lightning's LR finder not available. Use the standalone find_lr.py script.")
                print("Example: python atac_prediction/find_lr.py --config your_config.yaml --find-lr")
                lr_finder = None
        
        if lr_finder is not None:
            # Plot the learning rate finder results
            fig = lr_finder.plot(suggest=True)
            fig.savefig('lr_finder_plot.png', dpi=150, bbox_inches='tight')
            print(f"Learning rate finder plot saved to: lr_finder_plot.png")
            
            # Get the suggested learning rate
            new_lr = lr_finder.suggestion()
            print(f"Suggested learning rate: {new_lr}")
            
            # Option 1: Use the suggested learning rate
            model.hparams.learning_rate = new_lr
            
            # Option 2: Let user decide
            # print(f"Current learning rate: {model.hparams.learning_rate}")
            # print(f"Suggested learning rate: {new_lr}")
            # user_choice = input("Use suggested learning rate? (y/n): ")
            # if user_choice.lower() == 'y':
            #     model.hparams.learning_rate = new_lr
            
            print(f"Using learning rate: {model.hparams.learning_rate}")
        else:
            print("Skipping learning rate finder - continuing with current learning rate")

    # Load weights from checkpoint if specified (without resuming epoch)
    if args.load_weights_from:
        print(f"\nLoading weights from checkpoint: {args.load_weights_from}")
        from checkpoint_manager import CheckpointManager, ModelComponents
        manager = CheckpointManager(run_name=args.run_name, phase="training", strict_loading=False)
        components = ModelComponents(full_model=model)
        num_loaded, _ = manager.load_into_components(
            components,
            checkpoint_path=args.load_weights_from
        )
        if num_loaded > 0:
            print("Successfully loaded weights from checkpoint (starting from epoch 0)")
        else:
            print("Warning: failed to load checkpoint")
    
    # Handle checkpoint loading with chromosome prediction compatibility
    resume_checkpoint = args.resume_from
    if resume_checkpoint and (enable_chromosome_prediction or freeze_atac_components):
        print(f"\nChecking checkpoint compatibility for chromosome prediction...")
        try:
            # Try to load checkpoint to check for chromosome head keys
            checkpoint = torch.load(resume_checkpoint, map_location='cpu')
            state_dict = checkpoint.get('state_dict', {})
            
            # Check if checkpoint has chromosome head parameters
            has_chromosome_head = any('chromosome_head' in key for key in state_dict.keys())
            
            if enable_chromosome_prediction and not has_chromosome_head:
                print("⚠️  Checkpoint doesn't have chromosome prediction head - using flexible loading")
                print("   The chromosome head will be initialized randomly and trained from scratch")
                
                # Load the model with flexible loading instead of using trainer's resume
                model = DNAATACPredictionLightningModule.load_from_checkpoint_flexible(
                    resume_checkpoint,
                    strict=False,
                    enable_chromosome_prediction=enable_chromosome_prediction,
                    num_chromosomes=num_chromosomes,
                    chromosome_loss_weight=chromosome_loss_weight,
                    freeze_atac_components=freeze_atac_components,
                )
                resume_checkpoint = None  # Don't use trainer's resume since we loaded manually
                print("✅ Checkpoint loaded with flexible matching - chromosome head initialized randomly")
                
        except Exception as e:
            print(f"⚠️  Error checking checkpoint compatibility: {e}")
            print("   Attempting standard resume...")
    
    # Start training - compile after loading weights
    model.model = torch.compile(model.model)
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint)
    
    print("\nTraining completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score}")


def test_zarr_dataset():
    """Test the zarr dataset functionality"""

    print("=== Testing Zarr Dataset ===")
    
    ATAC_ZARR_PATH = 'data/zarr/cd200_scc_dense_new.zarr'
    GENOME_ZARR_PATH = 'data/zarr/mm10.zarr' 
    SUMMITS_CSV_PATH = 'training/data/total.bed'
    
    CELLTYPE_IDS = [
        'SCCApos.cd200_scc', 
        'SCCAneg.cd200_scc'
    ]
    
    # Test data module (zarr objects will be initialized in setup())
    data_module = DNAATACZarrDataModule(
        atac_zarr_path=ATAC_ZARR_PATH,
        genome_zarr_path=GENOME_ZARR_PATH,
        summits_csv_path=SUMMITS_CSV_PATH,
        celltype_ids=CELLTYPE_IDS,
        batch_size=32,
        num_workers=0,  # Use 0 for testing
        max_regions=1000  # Small number for testing
    )
    
    data_module.setup()
    data_module.summary()
    
    # Test individual samples
    print("\nTesting individual samples:")
    train_dataset = data_module.train_dataset
    
    for i in range(min(3, len(train_dataset))):
        print(f"\nSample {i}:")
        sample = train_dataset[i]
        region_info = sample['region_info']
        
        print(f"  Region: {region_info['chromosome']}:{region_info['start']}-{region_info['end']}")
        print(f"  ATAC shape: {sample['atac'].shape}")
        print(f"  Sequence shape: {sample['sequence'].shape}")
        
        # Show celltype stats
        stats = train_dataset.get_celltype_stats(i)
        for celltype, stat in stats['celltypes'].items():
            print(f"  {celltype}: mean={stat['mean']:.3f}, max={stat['max']:.3f}")
    
    # Test batch loading
    print("\nTesting batch loading:")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch ATAC shape: {batch['atac'].shape}")
    print(f"Batch sequence shape: {batch['sequence'].shape}")
    print("Zarr dataset test completed successfully!")


if __name__ == "__main__":
    # Handle test mode
    import argparse as base_argparse
    test_parser = base_argparse.ArgumentParser(add_help=False)
    test_parser.add_argument('--test', action='store_true', help='Run test mode instead of training')
    test_args, remaining_args = test_parser.parse_known_args()
    
    if test_args.test:
        test_zarr_dataset()
    else:
        # Restore sys.argv for the main argument parser
        sys.argv = [sys.argv[0]] + remaining_args
        main()
