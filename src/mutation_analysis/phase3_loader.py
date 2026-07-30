"""
Phase 3 differential model loader.

Encapsulates the complexity of loading Phase 3 models which require:
- Base ATAC model initialization
- Bias model creation (with TN5 motifs)
- Differential head assembly
- Complex checkpoint format handling

Provides a clean interface compatible with the mutation_analysis framework.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Phase3ModelLoader:
    """
    Loader for Phase 3 differential ATAC prediction models.

    Handles all the complexity of Phase 3 model initialization:
    - Loads configuration files
    - Creates base ATAC model with motifs
    - Creates bias model with TN5 motifs
    - Assembles Phase 3 model with differential head
    - Loads checkpoint weights

    Example:
        >>> loader = Phase3ModelLoader(config_file='configs/env_dlbcl.yml')
        >>> model, heads = loader.load_from_checkpoint('checkpoint.ckpt')
        >>> # Use with mutation_analysis
        >>> analyzer = MutationAnalyzer(model, heads, genome)
    """

    def __init__(self,
                 config_file: str = 'configs/env_dlbcl.yml',
                 simple_motifs_path: str = 'artifacts/motifs_with_rc_aligned.pt',
                 tn5_motifs_path: str = 'artifacts/bias.meme',
                 phase_config_file: str = 'configs/train_atac_phased.yaml'):
        """
        Initialize Phase 3 model loader.

        Args:
            config_file: Path to environment config (contains paths to data/motifs)
            simple_motifs_path: Path to aligned motifs for base model
            tn5_motifs_path: Path to TN5 motifs for bias model
            phase_config_file: Path to phase training config (model architecture)
        """
        self.config_file = config_file
        self.simple_motifs_path = simple_motifs_path
        self.tn5_motifs_path = tn5_motifs_path
        self.phase_config_file = phase_config_file

        # Will be loaded on demand
        self.config = None
        self.phase_config = None

    def load_from_checkpoint(self,
                           checkpoint_path: str,
                           device: Optional[str] = None) -> Tuple[nn.Module, List[str]]:
        """
        Load Phase 3 model from checkpoint.

        Args:
            checkpoint_path: Path to Phase 3 checkpoint file
            device: Device to load model on ('cuda' or 'cpu', auto-detected if None)

        Returns:
            Tuple of (model, output_heads)
            - model: Phase3 nn.Module with differential_head
            - output_heads: List of output head names

        Raises:
            ValueError: If checkpoint or required files not found
            RuntimeError: If model initialization fails
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Loading Phase 3 model from: {checkpoint_path}")

        # Load configurations
        self._load_configs()

        # Load and clean checkpoint
        checkpoint = self._load_and_clean_checkpoint(checkpoint_path)

        # Create model components
        base_model = self._create_base_model()
        bias_model = self._create_bias_model()

        # Assemble Phase 3 model
        phase3_model = self._create_phase3_model(base_model, bias_model)

        # Create Lightning wrapper
        lightning_module = self._create_lightning_module(phase3_model)

        # Load weights
        self._load_weights(lightning_module, checkpoint)

        # Move to device and set eval mode
        lightning_module = lightning_module.to(device)
        lightning_module.eval()

        logger.info(f"✓ Phase 3 model loaded successfully on {device}")

        # Extract output heads
        output_heads = self._get_output_heads()

        return lightning_module, output_heads

    def _load_configs(self):
        """Load configuration files."""
        # Import here to avoid circular dependencies.
        # Paths are resolved from the active bundle working directory.
        from configs.config_utils import load_env_config, ConfigManager

        # Load environment config
        logger.info(f"Loading environment config: {self.config_file}")
        try:
            self.config = load_env_config(self.config_file)
        except Exception as e:
            logger.warning(f"Could not load environment config: {e}")
            self.config = {}  # Use empty config, will fall back to defaults

        # Load phase config (model architecture)
        try:
            config_manager = ConfigManager("train_atac_phased")
            self.phase_config = config_manager.load_config(self.phase_config_file)
            logger.info(f"✓ Loaded phase config: {self.phase_config_file}")
        except Exception as e:
            logger.warning(f"Could not load phase config, using defaults: {e}")
            # Fallback to sensible defaults
            self.phase_config = {
                'model': {
                    'num_motifs': 746,
                    'motif_width': 21,
                    'num_residual_blocks': 4,
                    'num_filters': 128,
                    'dilations': [1, 2, 4, 8]
                },
                'bias_model': {
                    'use_learnable_kernels': False,
                    'freeze_motif_layer': True
                }
            }

    def _load_and_clean_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint and clean torch.compile artifacts."""
        from checkpoint_manager import CheckpointManager

        # Use CheckpointManager for automatic cleaning
        manager = CheckpointManager(
            run_name='inference',
            phase='phase3',
            checkpoint_dir=os.path.dirname(checkpoint_path) or 'checkpoints',
            clean_compiled=True
        )

        checkpoint, success = manager.load_checkpoint(checkpoint_path)

        if not success:
            raise ValueError(f"Failed to load checkpoint: {checkpoint_path}")

        # Ensure we have state_dict
        if 'state_dict' not in checkpoint:
            if 'full_model_state_dict' in checkpoint:
                checkpoint['state_dict'] = checkpoint['full_model_state_dict']
            elif 'main_model_state_dict' in checkpoint:
                checkpoint['state_dict'] = checkpoint['main_model_state_dict']
            else:
                raise ValueError(
                    f"Cannot find model state dict in checkpoint. "
                    f"Found keys: {list(checkpoint.keys())}"
                )

        logger.info("✓ Checkpoint loaded and cleaned")
        return checkpoint

    def _create_base_model(self) -> nn.Module:
        """Create base ATAC prediction model."""
        from atac_prediction.dna_atac_lightning_module import DNAATACPredictionLightningModule
        from atac_prediction.train_atac import load_motifs

        # Get PWM path from config or use default
        pwm_path = 'artifacts/data/hocomoco/H13CORE_pwms.txt'
        if self.config and 'paths' in self.config and 'motifs_file' in self.config['paths']:
            pwm_path = self.config['paths']['motifs_file']

        # Load motifs
        args_obj = type('Args', (), {
            'tn5_motifs_path': self.tn5_motifs_path,
            'simple_motifs_path': self.simple_motifs_path,
            'aligned_motifs_path': self.simple_motifs_path,
            'pwm_path': pwm_path,
            'motif_filter_prefix': None
        })()

        try:
            padded_kernels, _ = load_motifs(self.phase_config, args_obj)
            logger.info(f"✓ Loaded motifs: {padded_kernels.shape}")
        except Exception as e:
            raise RuntimeError(
                "Failed to load motif kernels for phase-3 model reconstruction. "
                "Refusing to fall back to random kernels."
            ) from e

        # Create base model
        base_model = DNAATACPredictionLightningModule(
            motif_kernels=padded_kernels,
            num_atac_samples=1,  # Phase 3 uses single negative sample
            learning_rate=1e-4,
            celltype_ids=['DZ'],  # Default for DLBCL
        )

        logger.info("✓ Created base ATAC model")
        return base_model

    def _create_bias_model(self) -> nn.Module:
        """Create bias model with TN5 motifs."""
        from atac_diff_prediction.bias_model import BiasModel, load_tn5_motifs_from_meme

        try:
            tn5_motifs = load_tn5_motifs_from_meme(self.tn5_motifs_path)
            bias_model = BiasModel(
                tn5_motifs=tn5_motifs,
                num_atac_samples=1,
                freeze_tn5_layer=True,
                use_learnable_kernels=False
            )
            logger.info(f"✓ Created bias model with TN5 motifs: {tn5_motifs.shape}")
        except Exception as e:
            logger.warning(f"Could not load TN5 motifs, using learnable kernels: {e}")
            bias_model = BiasModel(
                num_atac_samples=1,
                num_filters=8,
                filter_size=30,
                use_learnable_kernels=True
            )

        return bias_model

    def _create_phase3_model(self, base_model: nn.Module, bias_model: nn.Module) -> nn.Module:
        """Assemble Phase 3 model from components."""
        from atac_diff_prediction.models import create_phase3_model

        # Extract the actual nn.Module from Lightning wrapper
        base_nn_model = base_model.model if hasattr(base_model, 'model') else base_model

        phase3_model = create_phase3_model(base_nn_model, bias_model, self.phase_config)

        logger.info("✓ Created Phase 3 model with differential head")
        return phase3_model

    def _create_lightning_module(self, phase3_model: nn.Module) -> nn.Module:
        """Wrap Phase 3 model in Lightning module."""
        from atac_diff_prediction.lightning_modules import Phase3LightningModule

        lightning_module = Phase3LightningModule(
            model=phase3_model,
            learning_rate=5e-5,
            differential_loss_weight=1.0,
            celltype_ids=['DZ']
        )

        logger.info("✓ Created Phase 3 Lightning module")
        return lightning_module

    def _load_weights(self, lightning_module: nn.Module, checkpoint: Dict[str, Any]):
        """Load weights from checkpoint into model."""
        try:
            lightning_module.load_state_dict(checkpoint['state_dict'], strict=True)
            logger.info("✓ Loaded weights with strict=True")
        except Exception as e:
            logger.warning(f"Strict loading failed, trying non-strict: {e}")
            missing_keys, unexpected_keys = lightning_module.load_state_dict(
                checkpoint['state_dict'], strict=False
            )
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")

    def _get_output_heads(self) -> List[str]:
        """Get list of output head names for Phase 3 model."""
        # Phase 3 models have all three heads
        return ['regression', 'count', 'differential']


def load_phase3_model(checkpoint_path: str,
                     config_file: str = 'configs/env_dlbcl.yml',
                     **kwargs) -> Tuple[nn.Module, List[str]]:
    """
    Convenience function to load Phase 3 model.

    Args:
        checkpoint_path: Path to Phase 3 checkpoint
        config_file: Path to environment config file
        **kwargs: Additional arguments for Phase3ModelLoader

    Returns:
        Tuple of (model, output_heads)

    Example:
        >>> model, heads = load_phase3_model('checkpoint.ckpt')
        >>> print(heads)  # ['regression', 'count', 'differential']
    """
    loader = Phase3ModelLoader(config_file=config_file, **kwargs)
    return loader.load_from_checkpoint(checkpoint_path)
