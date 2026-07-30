"""
Torch-compilable nn.Module implementations for ATAC differential prediction.

This module contains pure nn.Module classes that are torch.compile compatible:
- Phase1Model: Standalone bias model for Phase 1 training
- Phase2Model: Standalone main model for Phase 2 training
- Phase3Model: Combined main model + differential head for Phase 3 training

All model computation logic is contained here, separate from Lightning modules.
No conditional branches - each phase has its own dedicated nn.Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from .bias_model import BiasModel, DifferentialHead


class Phase1Model(nn.Module):
    """
    Torch-compilable Phase 1 model (bias model only).

    This model only runs the bias model for background region training.
    No conditional logic - pure bias model forward pass.
    """

    def __init__(
        self,
        bias_model: BiasModel
    ):
        """
        Initialize Phase 1 model.

        Args:
            bias_model: Bias model for negative baseline prediction
        """
        super().__init__()
        self.bias_model = bias_model

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through bias model only.

        Args:
            x: Input DNA sequence tensor of shape (batch, 4, seq_len)

        Returns:
            Dictionary with 'bias_regression' and 'bias_count' predictions
        """
        return self.bias_model(x)


class Phase2Model(nn.Module):
    """
    Torch-compilable Phase 2 model (main ATAC model + bias model).

    This model runs both the main ATAC model and bias model, combining their outputs.
    Final regression = bias_regression + main_regression
    Final count = bias_count + main_count
    """

    def __init__(
        self,
        main_model: nn.Module,
        bias_model: BiasModel
    ):
        """
        Initialize Phase 2 model.

        Args:
            main_model: Main ATAC prediction model (DNAATACMotifCNN)
            bias_model: Bias model for background prediction
        """
        super().__init__()
        self.main_model = main_model
        self.bias_model = bias_model

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through main model and bias model, combining outputs.

        Args:
            x: Input DNA sequence tensor of shape (batch, 4, seq_len)

        Returns:
            Dictionary with combined 'regression' and 'count' predictions
        """
        # Get outputs from both models
        main_outputs = self.main_model(x)
        bias_outputs = self.bias_model(x)

        # Combine outputs: final = bias + main
        combined_outputs = {
            'regression': bias_outputs['bias_regression'] + main_outputs['regression'],
            'count': bias_outputs['bias_count'] + main_outputs['count']
        }

        return combined_outputs


class Phase3Model(nn.Module):
    """
    Torch-compilable Phase 3 model (main model + bias model + differential head).

    This model runs the main ATAC model, bias model, and differential head.
    Final regression = bias_regression + main_regression
    Final count = bias_count + main_count
    Also includes differential prediction.
    """

    def __init__(
        self,
        main_model: nn.Module,
        bias_model: BiasModel,
        differential_head: DifferentialHead
    ):
        """
        Initialize Phase 3 model.

        Args:
            main_model: Main ATAC prediction model (DNAATACMotifCNN)
            bias_model: Bias model for background prediction
            differential_head: Differential prediction head (required)
        """
        super().__init__()
        self.main_model = main_model
        self.bias_model = bias_model
        self.differential_head = differential_head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through main model + bias model + differential head.

        Args:
            x: Input DNA sequence tensor of shape (batch, 4, seq_len)

        Returns:
            Dictionary with combined 'regression', 'count', and 'differential' predictions
        """
        # Get outputs from both models
        main_outputs = self.main_model(x)
        bias_outputs = self.bias_model(x)

        # Combine outputs: final = bias + main
        combined_outputs = {
            'regression': bias_outputs['bias_regression'] + main_outputs['regression'],
            'count': bias_outputs['bias_count'] + main_outputs['count']
        }

        # Always add differential prediction (no conditionals)
        features = self._extract_main_model_features(x)
        differential_output = self.differential_head(features)
        combined_outputs['differential'] = differential_output

        return combined_outputs

    def _extract_main_model_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features from main model for differential head.

        This method extracts backbone features from the main model.
        It's designed to work with DNAATACMotifCNN architecture.
        """
        # For DNAATACMotifCNN, we need features after the residual blocks
        # Extract motif features
        features = self.main_model.motif_conv(x)
        if hasattr(self.main_model, 'motif_gn'):
            features = self.main_model.motif_gn(features)

        # Pass through residual backbone
        for block in self.main_model.res_blocks:
            features = block(features)

        return features




def create_phase1_model(config: Dict) -> Phase1Model:
    """
    Create Phase 1 model (bias model only) for Phase 1 training.

    Args:
        config: Configuration dictionary

    Returns:
        Phase1Model instance
    """
    bias_config = config.get('bias_model', {})

    use_learnable_kernels = bias_config.get('use_learnable_kernels', True)
    target_condition = bias_config.get('target_condition', 'neg')
    num_atac_samples = 1 if target_condition == 'neg' else 2

    if use_learnable_kernels:
        num_filters = bias_config.get('num_filters', 8)
        filter_size = bias_config.get('filter_size', 30)

        bias_model = BiasModel(
            num_atac_samples=num_atac_samples,
            num_filters=num_filters,
            filter_size=filter_size,
            use_learnable_kernels=True
        )
    else:
        # Legacy TN5 motifs
        from .bias_model import load_tn5_motifs_from_meme

        tn5_motifs_path = bias_config.get('tn5_motifs_path', 'artifacts/bias.meme')
        motif_filter_prefix = bias_config.get('motif_filter_prefix', 'TN5_')
        freeze_motif_layer = bias_config.get('freeze_motif_layer', True)

        tn5_motifs = load_tn5_motifs_from_meme(tn5_motifs_path, motif_filter_prefix)

        bias_model = BiasModel(
            tn5_motifs=tn5_motifs,
            num_atac_samples=num_atac_samples,
            freeze_tn5_layer=freeze_motif_layer,
            use_learnable_kernels=False
        )

    return Phase1Model(bias_model)


def create_phase2_model(main_model: nn.Module, bias_model: BiasModel) -> Phase2Model:
    """
    Create Phase 2 model (main model + bias model) for Phase 2 training.

    Args:
        main_model: Main ATAC prediction model
        bias_model: Bias model for background prediction

    Returns:
        Phase2Model instance
    """
    return Phase2Model(main_model, bias_model)


def create_phase3_model(
    main_model: nn.Module,
    bias_model: BiasModel,
    config: Optional[Dict] = None
) -> Phase3Model:
    """
    Create Phase 3 model (main model + bias model + differential head) for Phase 3 training.

    Args:
        main_model: Main ATAC prediction model
        bias_model: Bias model for background prediction
        config: Configuration dictionary

    Returns:
        Phase3Model instance
    """
    # Always create differential head for Phase 3
    input_channels = 128  # Adjust based on main model architecture
    num_conditions = 1  # Default

    if config is not None:
        phase3_config = config.get('phased_learning', {}).get('phase3_config', {})
        input_channels = phase3_config.get('input_channels', 128)
        num_conditions = phase3_config.get('num_conditions', 1)

    differential_head = DifferentialHead(input_channels, num_conditions)

    return Phase3Model(
        main_model=main_model,
        bias_model=bias_model,
        differential_head=differential_head
    )