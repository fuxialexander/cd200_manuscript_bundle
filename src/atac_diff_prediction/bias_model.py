"""
BiasModel implementation with TN5 motif convolution for ATAC-seq prediction.

This module implements the bias model that uses fixed TN5 motif kernels for
initial sequence processing, followed by trainable CNN layers and dual heads
for regression and count prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class BiasModel(nn.Module):
    """
    Bias model with learnable convolution for ATAC-seq prediction.

    Architecture:
    - Layer 1: Random initialized learnable convolution (instead of fixed TN5 motifs)
    - Layer 2-3: Trainable CNN layers (smaller than main model)
    - Dual heads: regression and count prediction for negative samples only
    """

    def __init__(
        self,
        tn5_motifs: Optional[torch.Tensor] = None,
        num_atac_samples: int = 1,
        freeze_tn5_layer: bool = False,
        num_filters: int = 128,
        filter_size: int = 31,
        use_learnable_kernels: bool = True
    ):
        """
        Initialize BiasModel.

        Args:
            tn5_motifs: TN5 motif kernels tensor (legacy, kept for compatibility)
            num_atac_samples: Number of ATAC samples (should be 1 for negative only)
            freeze_tn5_layer: Whether to freeze first convolution layer (legacy)
            num_filters: Number of filters for learnable kernels
            filter_size: Size of learnable kernels
            use_learnable_kernels: Whether to use learnable kernels (default: True)
        """
        super().__init__()

        if use_learnable_kernels:
            # Use random initialized learnable kernels
            self.num_motifs = num_filters
            self.motif_length = filter_size
        else:
            # Legacy: use provided TN5 motifs
            if tn5_motifs is None:
                raise ValueError("tn5_motifs must be provided when use_learnable_kernels=False")
            self.num_motifs = tn5_motifs.shape[0]
            self.motif_length = tn5_motifs.shape[2]

        self.num_atac_samples = num_atac_samples

        # Layer 1: Learnable convolution (random initialized) or Fixed TN5 motifs (legacy)
        if use_learnable_kernels:
            self.tn5_conv = self._create_learnable_conv_layer()
        else:
            # Legacy code for TN5 motifs
            self.tn5_conv = self._create_tn5_motif_layer(tn5_motifs)
            if freeze_tn5_layer:
                # Freeze TN5 convolution weights
                for param in self.tn5_conv.parameters():
                    param.requires_grad = False

        self.tn5_norm = nn.GroupNorm(self.num_motifs, self.num_motifs)

        # Layer 2-3: Trainable CNN layers (smaller than main model)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.num_motifs, 64, kernel_size=3, padding='same'),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding='same', dilation=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding='same', dilation=4),
            nn.GroupNorm(2, 16),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding='same', dilation=8),
            nn.GroupNorm(2, 8),
            nn.ReLU()
        )

        # Dual heads for bias model (negative samples only)
        self.bias_model_regression = nn.Sequential(
            nn.Conv1d(8, num_atac_samples, kernel_size=31, padding='same'),
            nn.Softplus()
        )

        self.bias_model_count = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(8, num_atac_samples),
            nn.Softplus()
        )

    def _create_learnable_conv_layer(self) -> nn.Conv1d:
        """
        Create learnable convolution layer with random initialization.

        Returns:
            Conv1d layer with random initialized weights
        """
        conv_layer = nn.Conv1d(
            in_channels=4,
            out_channels=self.num_motifs,
            kernel_size=self.motif_length,
            padding='same',
            bias=False
        )

        # Initialize with Xavier/Glorot normal initialization
        nn.init.xavier_normal_(conv_layer.weight)

        return conv_layer

    def _create_tn5_motif_layer(self, tn5_motifs: torch.Tensor) -> nn.Conv1d:
        """
        Create TN5 motif convolution layer with pre-defined kernels (legacy).

        Args:
            tn5_motifs: TN5 motif kernels of shape (num_motifs, 4, motif_length)

        Returns:
            Conv1d layer with TN5 motif kernels
        """
        conv_layer = nn.Conv1d(
            in_channels=4,
            out_channels=self.num_motifs,
            kernel_size=self.motif_length,
            padding='same',
            bias=False
        )

        # Set the weights to TN5 motifs
        with torch.no_grad():
            conv_layer.weight.copy_(tn5_motifs)

        return conv_layer

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through bias model.

        Args:
            x: Input sequence tensor of shape (batch, 4, seq_len)

        Returns:
            Dictionary with 'bias_regression' and 'bias_count' predictions
        """
        # Layer 1: TN5 motif convolution
        tn5_features = self.tn5_conv(x)  # (batch, num_motifs, seq_len)
        tn5_features = self.tn5_norm(tn5_features)

        # Layer 2-3: Trainable CNN layers
        features = self.conv_layers(tn5_features)  # (batch, 16, seq_len)

        # Dual heads
        regression_output = self.bias_model_regression(features)  # (batch, num_atac_samples, seq_len)
        count_output = self.bias_model_count(features)  # (batch, num_atac_samples)

        return {
            'bias_regression': regression_output,
            'bias_count': count_output
        }


class DifferentialHead(nn.Module):
    """
    Differential head module for predicting differential values (pos - neg).

    Predicts differential values separately from main regression using only
    global features, not sequence-level features.
    """

    def __init__(self, input_channels: int = 128, num_conditions: int = 1):
        """
        Initialize DifferentialHead.

        Args:
            input_channels: Number of input channels from main trunk
            num_conditions: Number of differential conditions to predict
        """
        super().__init__()

        self.diff_blocks = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_conditions)
            # No activation - raw differential values for MSE loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through differential head.

        Args:
            x: Input tensor of shape (batch, channels, seq_len) from main trunk

        Returns:
            Differential predictions of shape (batch, num_conditions)
        """
        # Global average pooling
        pooled = x.mean(dim=2)  # (batch, channels)

        # Differential prediction
        differential_output = self.diff_blocks(pooled)  # (batch, num_conditions)

        return differential_output


def load_tn5_motifs_from_meme(meme_file_path: str, filter_prefix: str = "TN5_") -> torch.Tensor:
    """
    Load TN5 motifs from MEME format file with reverse complements.

    Args:
        meme_file_path: Path to MEME format file containing TN5 motifs
        filter_prefix: Prefix to filter motif names (default: "TN5_")

    Returns:
        Tensor of TN5 motifs with reverse complements, shape (num_motifs*2, 4, motif_length)
    """
    import os


    motifs = []
    current_motif = None
    current_motif_name = None

    with open(meme_file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('MOTIF '):
                # Save previous motif if it matches filter
                if current_motif is not None and current_motif_name is not None:
                    if current_motif_name.startswith(filter_prefix):
                        motifs.append(torch.tensor(current_motif))

                # Start new motif
                current_motif_name = line.split()[1]  # Extract motif name
                current_motif = []

            elif line.startswith('letter-probability matrix:'):
                # Skip motif dimension parsing - not needed
                pass

            elif current_motif is not None and not line.startswith('MOTIF') and line:
                # Parse probability line
                try:
                    probs = [float(x) for x in line.split()]
                    if len(probs) == 4:  # A, C, G, T probabilities
                        current_motif.append(probs)
                except ValueError:
                    continue

    # Add the last motif if it matches filter
    if current_motif is not None and current_motif_name is not None:
        if current_motif_name.startswith(filter_prefix):
            motifs.append(torch.tensor(current_motif))

    if motifs:
        # Standardize all motifs to 51bp
        target_length = 51

        # Pad/truncate motifs to the same length and transpose to (num_motifs, 4, length) format
        standardized_motifs = []
        for m in motifs:
            m_transposed = m.transpose(0, 1)  # (4, length)
            current_length = m_transposed.shape[1]

            if current_length < target_length:
                # Pad with uniform distribution (0.25 for each nucleotide)
                padding = torch.full((4, target_length - current_length), 0.25)
                m_standardized = torch.cat([m_transposed, padding], dim=1)
            elif current_length > target_length:
                # Truncate from the end
                m_standardized = m_transposed[:, :target_length]
            else:
                m_standardized = m_transposed
            standardized_motifs.append(m_standardized)

        # Create reverse complements for each motif
        all_motifs = []
        for motif in standardized_motifs:
            # Original motif
            all_motifs.append(motif)
            # Reverse complement: reverse sequence and complement bases (A<->T, C<->G)
            motif_rc = motif.flip(dims=[1])  # Reverse the sequence
            motif_rc = motif_rc[[3, 2, 1, 0], :]  # Complement: A(0)<->T(3), C(1)<->G(2)
            all_motifs.append(motif_rc)

        motif_tensor = torch.stack(all_motifs)
        print(f"Loaded {len(motifs)} TN5 motifs from {meme_file_path} with filter '{filter_prefix}'")
        print(f"Created {len(motifs)} reverse complements (total: {len(all_motifs)} motifs)")
        for i, m in enumerate(motifs):
            print(f"  TN5 Motif {i}: {m.shape[0]} positions (standardized to {target_length})")
        return motif_tensor
    else:
        # No motifs found - raise error
        raise ValueError(f"No motifs found in {meme_file_path} with filter '{filter_prefix}'")



def create_bias_model_from_config(config: dict) -> Optional[BiasModel]:
    """
    Create BiasModel from configuration.

    Args:
        config: Configuration dictionary with bias_model settings

    Returns:
        BiasModel instance or None if not enabled
    """
    bias_config = config.get('bias_model', {})

    if not bias_config.get('enable', False):
        return None

    # Check if using learnable kernels (default: True)
    use_learnable_kernels = bias_config.get('use_learnable_kernels', True)

    # Get number of ATAC samples (should be 1 for negative-only prediction)
    target_condition = bias_config.get('target_condition', 'neg')
    num_atac_samples = 1 if target_condition == 'neg' else 2

    if use_learnable_kernels:
        # Use random initialized learnable kernels
        num_filters = bias_config.get('num_filters', 8)
        filter_size = bias_config.get('filter_size', 30)

        bias_model = BiasModel(
            num_atac_samples=num_atac_samples,
            num_filters=num_filters,
            filter_size=filter_size,
            use_learnable_kernels=True
        )

        print(f"Created BiasModel with {num_filters} learnable filters of size {filter_size}")
        print(f"Target condition: {target_condition}")
        print("Using random initialized learnable kernels")

    else:
        # Legacy: Load TN5 motifs from file
        tn5_motifs_path = bias_config.get('tn5_motifs_path', 'artifacts/bias.meme')
        motif_filter_prefix = bias_config.get('motif_filter_prefix', 'TN5_')

        tn5_motifs = load_tn5_motifs_from_meme(tn5_motifs_path, motif_filter_prefix)

        freeze_motif_layer = bias_config.get('freeze_motif_layer', True)

        bias_model = BiasModel(
            tn5_motifs=tn5_motifs,
            num_atac_samples=num_atac_samples,
            freeze_tn5_layer=freeze_motif_layer,
            use_learnable_kernels=False
        )

        print(f"Created BiasModel with {tn5_motifs.shape[0]} TN5 motifs (legacy mode)")
        print(f"Target condition: {target_condition}")
        print(f"Motif layer frozen: {freeze_motif_layer}")

    return bias_model