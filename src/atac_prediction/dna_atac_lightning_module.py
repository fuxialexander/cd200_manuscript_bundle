"""
Filename: dna_atac_lightning_module.py
Purpose: Complete PyTorch Lightning implementation for DNA->ATAC prediction with motif-based CNN architecture

Main Components:
- ResidualBlock: Residual block with dilated convolutions and batch normalization
- DNAATACMotifCNN: CNN model using pre-defined motif kernels for sequence analysis
- DNAATACPredictionLightningModule: Full Lightning module with training, validation, and comprehensive metrics
- Multi-head architecture: regression head for signal prediction, classification head for peak detection
- Support for multiple loss functions: MSE, BCE, region-level losses, and fold-change prediction
- Advanced visualization and logging integration with Weights & Biases


  Model Architecture (DNAATACMotifCNN)

  Fixed Motif Layer: Uses pre-computed, non-trainable motif kernels as convolutional filters
  Backbone: 6 residual blocks with dilated convolutions (dilations: 1, 2, 4, 8, 16, 32)
  Multi-head Output:
  - Regression head: Predicts ATAC signal with Softplus activation
  - Classification head: Region-level peak classification with global average pooling
  - FC head: Fold-change prediction between cell types

  Loss Functions

  Primary Losses:
  - MSE Loss: Log-transformed signal prediction on central 2048bp
  - BCE Loss: Sequence-level binary classification using dynamic thresholds

  Additional Losses:
  - Region Classification: Binary cross-entropy for region-level peak detection
  - Region MSE: Log-transformed region means for positive regions only
  - FC Huber Loss: Fold-change prediction between cell types (SCCApos - SCCAneg)

  Loss Processing:
  - Applies high-count filtering (threshold > 1.0)
  - Log transformation with pseudocount of 1.0
  - Region scaling factor of 2048 for log transformation
  - Dynamic binary thresholds using 80th percentile with 0.2 floor

  The model supports both 2 and 4 cell type configurations and includes comprehensive validation
  metrics with correlation analysis and interactive visualizations.


Dependencies: pytorch_lightning, torch, torch.nn, numpy, wandb, matplotlib, PIL
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image


class ClusteredFrequencyDropout(nn.Module):
    def __init__(
        self,
        cluster_labels_path: str = "atac_prediction/motif_cluster_labels_thresh0.9.npy",
        mean_activations_path: str = "atac_prediction/motif_mean_activations.npy",
        p_min: float = 0.02,
        p_max: float = 0.2,
        temperature: float = 1,
    ):
        """
        Performs dropout on clusters of motifs, where the dropout probability
        for a cluster is proportional to the sum of the mean activations of its members.

        Args:
            cluster_labels_path (str): Path to the .npy file with motif cluster labels.
            mean_activations_path (str): Path to the .npy file with motif mean activations.
            p_min (float): The base dropout probability for the least important cluster.
            p_max (float): The max dropout probability for the most important cluster.
            temperature (float): Controls the sharpness of the probability distribution. Lower is sharper.
        """
        super().__init__()

        # Check if files exist before loading
        import os

        if not os.path.exists(cluster_labels_path) or not os.path.exists(
            mean_activations_path
        ):
            print(f"ClusteredFrequencyDropout: Required files not found, will be disabled until files are available")
            print(f"  - Cluster labels: {cluster_labels_path} ({'found' if os.path.exists(cluster_labels_path) else 'missing'})")
            print(f"  - Mean activations: {mean_activations_path} ({'found' if os.path.exists(mean_activations_path) else 'missing'})")
            self.enabled = False
            self.cluster_labels_path = cluster_labels_path
            self.mean_activations_path = mean_activations_path
            return

        cluster_labels = np.load(cluster_labels_path)
        mean_activations = np.load(mean_activations_path)
        self.enabled = True

        num_motifs = len(cluster_labels)
        num_clusters = len(np.unique(cluster_labels))

        # --- Calculate cluster importance and dropout probabilities ---
        cluster_importance = np.zeros(num_clusters)
        for i in range(num_motifs):
            cluster_id = cluster_labels[i]
            cluster_importance[cluster_id] += mean_activations[i]

        # Use softmax to convert importance scores to a relative scale
        # Note: We don't use a negative sign, as we want high importance -> high probability
        relative_probs = self.softmax(cluster_importance, temp=temperature)

        # Scale probabilities to the desired [p_min, p_max] range
        # This makes the max dropout probability predictable and capped at p_max
        p_scaled = p_min + (relative_probs / relative_probs.max()) * (p_max - p_min)

        self.register_buffer(
            "cluster_dropout_probs", torch.tensor(p_scaled, dtype=torch.float32)
        )

        # Create a map from motif index to cluster index for the forward pass
        unique_clusters, cluster_map = torch.unique(
            torch.tensor(cluster_labels), return_inverse=True
        )
        self.register_buffer("cluster_map", cluster_map)  # Shape: (num_motifs,)

    def try_enable_if_files_available(self, p_min: float = 0.02, p_max: float = 0.2, temperature: float = 1):
        """Try to enable ClusteredFrequencyDropout if required files are now available."""
        if self.enabled:
            return  # Already enabled
            
        import os
        if not hasattr(self, 'cluster_labels_path') or not hasattr(self, 'mean_activations_path'):
            return  # Paths not stored
            
        if os.path.exists(self.cluster_labels_path) and os.path.exists(self.mean_activations_path):
            print("ClusteredFrequencyDropout: Required files found, enabling dropout")
            
            cluster_labels = np.load(self.cluster_labels_path)
            mean_activations = np.load(self.mean_activations_path)
            
            num_motifs = len(cluster_labels)
            num_clusters = len(np.unique(cluster_labels))

            # --- Calculate cluster importance and dropout probabilities ---
            cluster_importance = np.zeros(num_clusters)
            for i in range(num_motifs):
                cluster_id = cluster_labels[i]
                cluster_importance[cluster_id] += mean_activations[i]

            # Use softmax to convert importance scores to a relative scale
            relative_probs = self.softmax(cluster_importance, temp=temperature)

            # Scale probabilities to the desired [p_min, p_max] range
            p_scaled = p_min + (relative_probs / relative_probs.max()) * (p_max - p_min)

            self.register_buffer(
                "cluster_dropout_probs", torch.tensor(p_scaled, dtype=torch.float32)
            )

            # Create a map from motif index to cluster index for the forward pass
            _, cluster_map = torch.unique(
                torch.tensor(cluster_labels), return_inverse=True
            )
            self.register_buffer("cluster_map", cluster_map)  # Shape: (num_motifs,)
            
            self.enabled = True

    def softmax(self, x, temp=1.0):
        # Helper function for numpy arrays
        e_x = np.exp(x / temp)
        return e_x / e_x.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where C is num_motifs.
        """
        if not self.training or not self.enabled:
            return x

        # 1. Generate a dropout decision for each CLUSTER
        # We sample once per cluster using its specific dropout probability
        cluster_mask = torch.bernoulli(1.0 - self.cluster_dropout_probs).to(x.device)

        # 2. Expand the cluster mask to a full motif mask
        motif_mask = cluster_mask[self.cluster_map]  # Shape: (num_motifs,)

        # 3. Apply the mask and scale the output
        motif_mask = motif_mask.view(1, -1, 1)  # Reshape for broadcasting

        # Calculate the keep probability for scaling
        # This is the average keep probability across all motifs
        avg_keep_prob = (1.0 - self.cluster_dropout_probs[self.cluster_map]).mean()

        # Scale by the average keep probability to maintain the overall activation magnitude
        # Add safeguard to prevent extreme scaling when many channels are dropped
        safe_scaling = torch.clamp(1.0 / (avg_keep_prob + 1e-8), max=2.0)
        return x * motif_mask * safe_scaling


class ShapeScaleLoss(torch.nn.Module):
    """
    Hybrid loss for profile prediction.
    Focuses on shape (scale-invariant) with optional scale regularization.

    Args:
        shape_mode: 'ce' | 'cosine' | 'pearson'
        tau: temperature for shape='ce' (<=1 sharpens). Typical 0.5–1.0
        scale_weight: λ for the scale term (0 disables)
        eps: numerical epsilon
        clamp_negatives: if True, applies softplus to predictions before normalization
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self, shape_mode="ce", tau=1.0, scale_weight=5, eps=1e-6, reduction="mean"
    ):
        super().__init__()
        self.shape_mode = shape_mode
        self.tau = tau
        self.scale_weight = scale_weight
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred, y_true, mask=None):
        """
        y_pred, y_true: (B, L) non-negative profiles (counts or continuous)
        mask: (B, L) boolean or 0/1 to exclude positions from shape term (optional)
        return_components: if True, returns (total_loss, shape_loss, scale_loss)
        """
        eps = self.eps

        # Optionally mask invalid positions for shape term
        if mask is not None:
            y_true = y_true * mask
            y_pred = y_pred * mask

        # --- SHAPE TERM ---
        if self.shape_mode == "ce":
            # L1 normalize to get distributions with better numerical stability
            sum_true = y_true.sum(dim=1, keepdim=True)
            sum_pred = y_pred.sum(dim=1, keepdim=True)

            # Skip samples with very low or very high signal to avoid numerical issues
            # Increased threshold to prevent near-zero divisions
            valid_mask = (
                (sum_true > eps * 1000)
                & (sum_pred > eps * 1000)
                & (sum_true < 1e6)
                & (sum_pred < 1e6)
            )

            # Add smoothing to prevent extreme distributions
            p_true = (y_true + eps) / (sum_true + eps * y_true.shape[1])
            p_pred = (y_pred + eps) / (sum_pred + eps * y_pred.shape[1])

            # Temperature (optional): sharpen/flatten target matching
            # Clamp tau to prevent extreme sharpening that causes instability
            safe_tau = max(self.tau, 0.1)  # Prevent tau < 0.1
            if safe_tau != 1.0:
                # Work in logits space for stability with better clamping
                logits = (
                    torch.log(torch.clamp(p_pred, min=eps, max=1 - eps))
                ) / safe_tau
                log_p = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            else:
                log_p = torch.log(torch.clamp(p_pred, min=eps, max=1 - eps))

            # Compute cross-entropy only for valid samples with gradient clamping
            ce_term = p_true * log_p
            # Clamp individual terms to prevent extreme gradients
            ce_term = torch.clamp(ce_term, min=-10.0, max=0.0)
            shape_loss = -ce_term.sum(dim=1)

            # Apply valid mask to avoid inf/nan from low-signal samples
            shape_loss = torch.where(
                valid_mask.squeeze(), shape_loss, torch.zeros_like(shape_loss)
            )

            # Additional safeguards: clamp to prevent extreme values and check for nan/inf
            shape_loss = torch.clamp(shape_loss, min=0, max=10.0)
            shape_loss = torch.where(
                torch.isnan(shape_loss) | torch.isinf(shape_loss),
                torch.zeros_like(shape_loss),
                shape_loss,
            )

        elif self.shape_mode == "cosine":
            # Cosine distance = 1 - cosine similarity
            # add eps to avoid division by zero
            num = (y_true * y_pred).sum(dim=1)
            den = y_true.norm(dim=1) * y_pred.norm(dim=1) + eps
            shape_loss = 1.0 - (num / den)

        elif self.shape_mode == "pearson":
            # 1 - Pearson correlation (scale & shift invariant)
            yt = y_true - y_true.mean(dim=1, keepdim=True)
            yp = y_pred - y_pred.mean(dim=1, keepdim=True)
            num = (yt * yp).sum(dim=1)
            den = yt.norm(dim=1) * yp.norm(dim=1) + eps
            shape_loss = 1.0 - (num / den)

        else:
            raise ValueError("shape_mode must be 'ce', 'cosine', or 'pearson'.")

        # --- SCALE TERM (log-mass MSE) ---
        if self.scale_weight > 0:
            # Clamp means before log to prevent extreme scale terms
            mean_true = torch.clamp(y_true.mean(dim=1), max=1e4)
            mean_pred = torch.clamp(y_pred.mean(dim=1), max=1e4)

            m_true = torch.log1p(mean_true)
            m_pred = torch.log1p(mean_pred)
            scale_loss = (m_true - m_pred).pow(2)

            # Clamp scale loss to prevent extreme values
            scale_loss = torch.clamp(scale_loss, max=10.0)

            loss = shape_loss + self.scale_weight * scale_loss
        else:
            scale_loss = torch.zeros_like(shape_loss)
            loss = shape_loss

        # Clamp total loss to prevent extreme spikes
        loss = torch.clamp(loss, max=15.0)

        # Apply reduction
        if self.reduction == "mean":
            total_loss = loss.mean()
        elif self.reduction == "sum":
            total_loss = loss.sum()
        else:
            total_loss = loss  # (B,)

        return total_loss


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions, group normalization, and skip connections.

    This block implements a standard residual connection with:
    - Two 1D convolutions with group normalization
    - Configurable dilation for expanded receptive field
    - Dropout for regularization
    - Skip connection with channel adjustment if needed
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.1,
        num_groups: int = 8,
    ) -> None:
        """Initialize ResidualBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernels
            dilation: Dilation rate for convolutions
            dropout: Dropout probability
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()

        # Use GroupNorm instead of BatchNorm
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding="same"
        )
        self.gn1 = nn.GroupNorm(
            num_groups=min(num_groups, out_channels), num_channels=out_channels
        )

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=1, padding="same"
        )
        self.gn2 = nn.GroupNorm(
            num_groups=min(num_groups, out_channels), num_channels=out_channels
        )

        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Output tensor with residual connection applied
        """
        identity = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


class DNAATACMotifCNN(nn.Module):
    """CNN model for DNA->ATAC prediction using motif kernels.

    This model uses pre-defined motif kernels as fixed convolutional filters,
    followed by residual blocks and a regression head for ATAC signal prediction.

    Architecture:
    1. Motif convolution layer (fixed or trainable)
    2. Series of dilated residual blocks
    3. Regression head for multi-sample ATAC prediction

    Supports:
    - Multiple ATAC samples (celltypes) as output
    - Motifs with reverse complements
    - Variable motif kernel sizes
    - Random motif initialization (trainable kernels)
    """

    def __init__(
        self,
        motif_kernels: np.ndarray,
        num_atac_samples: int = 1,
        use_random_motif_init: bool = False,
        num_random_kernels: int = None,
        enable_chromosome_prediction: bool = False,
        num_chromosomes: int = 25,
        freeze_atac_components: bool = False,
    ) -> None:
        """Initialize the DNA->ATAC motif CNN.

        Args:
            motif_kernels: Pre-computed motif kernels of shape (num_motifs, length, 4) or None for random init
            num_atac_samples: Number of ATAC samples/celltypes to predict
            use_random_motif_init: Whether to use random initialization instead of fixed motifs
            num_random_kernels: Number of random kernels to use (overrides motif_kernels shape)
            enable_chromosome_prediction: Whether to add chromosome prediction head
            num_chromosomes: Number of chromosomes to predict (default: 25 for human + unknown)
            freeze_atac_components: Whether to freeze ATAC prediction components (for chromosome-only training)
        """
        super().__init__()

        self.num_atac_samples = num_atac_samples
        self.use_random_motif_init = use_random_motif_init
        self.enable_chromosome_prediction = enable_chromosome_prediction
        self.num_chromosomes = num_chromosomes
        self.freeze_atac_components = freeze_atac_components

        if use_random_motif_init:
            # Use random initialization
            self.num_motifs = (
                num_random_kernels if num_random_kernels is not None else 512
            )
            kernel_length = 19  # Standard motif length
            print(f"Initializing DNAATACMotifCNN with {self.num_motifs} random kernels")
            print(f"Kernel length: {kernel_length}")
            self.motif_conv = self._create_random_motif_layer(kernel_length)
        else:
            # Use provided motif kernels
            # Ensure kernel size is odd to avoid padding warnings
            motif_kernels = self._ensure_odd_kernel_size(motif_kernels)

            self.num_motifs = motif_kernels.shape[0]
            kernel_length = motif_kernels.shape[1]

            print(f"Initializing DNAATACMotifCNN with {self.num_motifs} motifs")
            print(f"Kernel length: {kernel_length}")

            # Initialize model components
            self.motif_conv = self._create_motif_layer(motif_kernels, kernel_length)
        # self.motif_gn = nn.BatchNorm1d(self.num_motifs)
        self.motif_gn = nn.GroupNorm(num_groups=self.num_motifs, num_channels=self.num_motifs)
        # Add ClusteredFrequencyDropout before residual blocks
        # self.clustered_dropout = ClusteredFrequencyDropout()

        # Use motif activations directly without projection
        self.res_blocks = self._create_backbone(self.num_motifs)
        self.count_blocks = self._create_count_backbone(128)  # Count-specific blocks
        self.regression_head = self._create_regression_head(num_atac_samples)
        self.count_head = self._create_count_head(num_atac_samples)
        
        # Chromosome prediction head (optional)
        if self.enable_chromosome_prediction:
            self.chromosome_head = self._create_chromosome_head()
        
        # Apply freezing after all components are initialized
        if self.freeze_atac_components:
            self._freeze_atac_components()

    def _ensure_odd_kernel_size(self, motif_kernels: np.ndarray) -> np.ndarray:
        """Ensure motif kernels have odd size to avoid padding warnings."""
        if motif_kernels.shape[1] % 2 == 0:
            return np.pad(motif_kernels, ((0, 0), (0, 1), (0, 0)), "constant")
        return motif_kernels

    def _create_motif_layer(
        self, motif_kernels: np.ndarray, kernel_length: int
    ) -> nn.Conv1d:
        """Create fixed motif convolution layer."""
        motif_conv = nn.Conv1d(
            4, self.num_motifs, kernel_length, padding="same", bias=False
        )

        # Set motif weights and freeze them
        # Handle both numpy arrays and torch tensors
        if isinstance(motif_kernels, torch.Tensor):
            motif_weights = motif_kernels.permute(0, 2, 1).float()
        else:
            motif_weights = torch.tensor(
                motif_kernels.transpose(0, 2, 1), dtype=torch.float32
            )
        motif_conv.weight.data = motif_weights
        motif_conv.weight.requires_grad = False

        return motif_conv

    def _create_random_motif_layer(self, kernel_length: int) -> nn.Conv1d:
        """Create random motif convolution layer (trainable)."""
        motif_conv = nn.Conv1d(
            4, self.num_motifs, kernel_length, padding="same", bias=True
        )

        # Use standard PyTorch initialization (weights and bias remain trainable)
        # No manual weight setting or gradient freezing

        return motif_conv

    def _create_backbone(self, input_channels: int) -> nn.ModuleList:
        """Create residual block backbone."""
        dilations = [1, 2, 3, 5, 11, 41, 81, 167]
        channels = [input_channels, 128, 128, 128, 128, 128, 128, 128, 128]

        res_blocks = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            res_blocks.append(
                ResidualBlock(
                    channels[i], channels[i + 1], kernel_size=3, dilation=dilation
                )
            )

        return res_blocks

    def _create_count_backbone(self, input_channels: int) -> nn.ModuleList:
        """Create count-specific residual blocks with dilation=1 and kernel_size=3."""
        count_blocks = nn.ModuleList()
        
        # Add 3 count-specific residual blocks
        for i in range(3):
            count_blocks.append(
                ResidualBlock(
                    input_channels, input_channels, kernel_size=3, dilation=1
                )
            )
        
        return count_blocks

    def _create_regression_head(self, num_atac_samples: int) -> nn.Sequential:
        """Create regression head for ATAC prediction."""
        return nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=64),  # Stabilize activations
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Conv1d(64, num_atac_samples, kernel_size=1),
            nn.Softplus(),  # Standard softplus
            # Note: Output range will be ~0.01-5.0, scale in loss function if needed
        )

    def _create_count_head(self, num_atac_samples: int) -> nn.Sequential:
        """Create count head for log10(x+1) sum profile prediction."""
        return nn.Sequential(
            nn.Linear(128, 64),  # Input from count blocks mean pooling
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_atac_samples),
            nn.Softplus(),  # Output will be passed through log10(x+1)
        )
    
    def _create_chromosome_head(self) -> nn.ModuleDict:
        """Create chromosome prediction head for chromosome classification."""
        return nn.ModuleDict({
            'conv_block': ResidualBlock(
                in_channels=128, 
                out_channels=128, 
                kernel_size=3, 
                dilation=1,
                dropout=0.2
            ),
            'classifier': nn.Sequential(
                nn.Linear(128, 64),  # Input from conv block mean pooling
                nn.ReLU(),
                nn.Dropout(0.2),  # Higher dropout for regularization
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, self.num_chromosomes),
                # No activation - use raw logits for CrossEntropyLoss
            )
        })
    
    def _freeze_atac_components(self) -> None:
        """Freeze all ATAC prediction components, keeping only chromosome head trainable."""
        print("Freezing ATAC prediction components for chromosome-only training...")
        
        frozen_components = []
        trainable_components = []
        
        # Freeze motif convolution layer
        if hasattr(self, 'motif_conv'):
            for param in self.motif_conv.parameters():
                param.requires_grad = False
            frozen_components.append("motif_conv")
        
        # Freeze motif group normalization
        if hasattr(self, 'motif_gn'):
            for param in self.motif_gn.parameters():
                param.requires_grad = False
            frozen_components.append("motif_gn")
        
        # Freeze residual backbone blocks
        if hasattr(self, 'res_blocks'):
            for block in self.res_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            frozen_components.append("res_blocks")
        
        # Freeze count-specific blocks  
        if hasattr(self, 'count_blocks'):
            for block in self.count_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            frozen_components.append("count_blocks")
        
        # Freeze regression head (ATAC signal prediction)
        if hasattr(self, 'regression_head'):
            for param in self.regression_head.parameters():
                param.requires_grad = False
            frozen_components.append("regression_head")
        
        # Freeze count head
        if hasattr(self, 'count_head'):
            for param in self.count_head.parameters():
                param.requires_grad = False
            frozen_components.append("count_head")
        
        # Keep chromosome head trainable (if it exists)
        if hasattr(self, 'chromosome_head'):
            for param in self.chromosome_head.parameters():
                param.requires_grad = True
            trainable_components.append("chromosome_head")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"  Frozen components: {', '.join(frozen_components)}")
        print(f"  Trainable components: {', '.join(trainable_components)}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    def forward(self, dna_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            dna_seq: DNA sequence tensor of shape (batch, 4, seq_len)

        Returns:
            Dictionary containing:
                - 'regression': ATAC predictions of shape (batch, num_atac_samples, seq_len)
        """
        # Extract motif features using kernels
        x = self.motif_conv(dna_seq)
        # add group norm
        x = self.motif_gn(x)

        # Apply clustered frequency dropout
        # x = self.clustered_dropout(x)

        # Pass through residual backbone
        for block in self.res_blocks:
            x = block(x)

        # Generate regression output
        regression_output = self.regression_head(x)
        
        # Pass through count-specific blocks
        count_features = x  # Start from same features as regression
        for block in self.count_blocks:
            count_features = block(count_features)
        
        # Mean pooling for count features (torch.compile friendly)
        count_pooled = count_features.mean(dim=2)  # (batch, 128) - global average pool
        
        # Generate count output (raw values, log transform in loss)
        count_output = self.count_head(count_pooled)  # (batch, num_atac_samples)

        result = {"regression": regression_output, "count": count_output}
        
        # Add chromosome prediction if enabled
        if self.enable_chromosome_prediction:
            # Use features after the backbone for chromosome prediction
            # Apply conv1d residual block then global average pooling
            chrom_conv_features = self.chromosome_head['conv_block'](x)  # (batch, 128, seq_len)
            chrom_pooled = chrom_conv_features.mean(dim=2)  # (batch, 128) - global average pool
            chromosome_output = self.chromosome_head['classifier'](chrom_pooled)  # (batch, num_chromosomes)
            result["chromosome"] = chromosome_output

        return result

    def forward_with_motif_activations(
        self, dna_seq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model with motif activations returned.

        Args:
            dna_seq: DNA sequence tensor of shape (batch, 4, seq_len)

        Returns:
            Dictionary containing:
                - 'regression': ATAC predictions of shape (batch, num_atac_samples, seq_len)
                - 'motif_activations': Motif activations of shape (batch, num_motifs, seq_len)
        """
        # Extract motif features using kernels
        motif_activations = self.motif_conv(dna_seq)

        # Apply clustered frequency dropout
        # x = self.clustered_dropout(motif_activations)
        x = motif_activations

        # Pass through residual backbone
        for block in self.res_blocks:
            x = block(x)

        # Generate regression output
        regression_output = self.regression_head(x)
        
        # Pass through count-specific blocks
        count_features = x  # Start from same features as regression
        for block in self.count_blocks:
            count_features = block(count_features)
        
        # Mean pooling for count features (torch.compile friendly)
        count_pooled = count_features.mean(dim=2)  # (batch, 128) - global average pool
        
        # Generate count output (log10 of profile sum + 1)
        count_output = self.count_head(count_pooled)  # (batch, num_atac_samples)

        result = {
            "regression": regression_output,
            "count": count_output,
            "motif_activations": F.relu(motif_activations).mean(dim=2),
        }
        
        # Add chromosome prediction if enabled
        if self.enable_chromosome_prediction:
            # Use features after the backbone for chromosome prediction
            # Apply conv1d residual block then global average pooling
            chrom_conv_features = self.chromosome_head['conv_block'](x)  # (batch, 128, seq_len)
            chrom_pooled = chrom_conv_features.mean(dim=2)  # (batch, 128) - global average pool
            chromosome_output = self.chromosome_head['classifier'](chrom_pooled)  # (batch, num_chromosomes)
            result["chromosome"] = chromosome_output

        return result


class DNAATACPredictionLightningModule(pl.LightningModule):
    """PyTorch Lightning module for DNA->ATAC prediction.

    This module wraps the DNAATACMotifCNN model and provides:
    - Training, validation, and prediction steps
    - Multiple loss functions (MSE, BCE, etc.)
    - Comprehensive metrics and logging
    - Interactive visualizations via Weights & Biases

    The module supports multi-celltype ATAC prediction with various
    loss weighting strategies and advanced gradient monitoring.
    """

    def __init__(
        self,
        motif_kernels: Optional[np.ndarray] = None,
        num_atac_samples: int = 1,
        learning_rate: float = 1e-3,
        high_count_threshold: float = 0,
        celltype_ids: Optional[List[str]] = None,
        use_random_motif_init: bool = False,
        num_random_kernels: int = None,
        enable_chromosome_prediction: bool = False,
        num_chromosomes: int = 25,
        chromosome_loss_weight: float = 1.0,
        freeze_atac_components: bool = False,
    ) -> None:
        """Initialize the Lightning module.

        Args:
            motif_kernels: Pre-computed motif kernels or None for random init
            num_atac_samples: Number of ATAC samples/celltypes
            learning_rate: Learning rate for optimization
            high_count_threshold: Threshold for high-count region filtering
            celltype_ids: List of celltype identifiers for logging
            use_random_motif_init: Whether to use random initialization instead of fixed motifs
            num_random_kernels: Number of random kernels to use
            enable_chromosome_prediction: Whether to add chromosome prediction head
            num_chromosomes: Number of chromosomes to predict
            chromosome_loss_weight: Weight for chromosome prediction loss
            freeze_atac_components: Whether to freeze ATAC components (for chromosome-only training)
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize model (no projection layer)
        self.model = DNAATACMotifCNN(
            motif_kernels,
            num_atac_samples,
            use_random_motif_init=use_random_motif_init,
            num_random_kernels=num_random_kernels,
            enable_chromosome_prediction=enable_chromosome_prediction,
            num_chromosomes=num_chromosomes,
            freeze_atac_components=freeze_atac_components,
        )

        # Initialize ShapeScaleLoss
        self.shape_scale_loss = ShapeScaleLoss(
            shape_mode="ce", tau=1.0, scale_weight=0.2, eps=1e-6, reduction="mean"
        )
        
        # Initialize PoissonNLL loss for count prediction
        self.poisson_loss = nn.PoissonNLLLoss(log_input=True, reduction='mean')
        
        # Initialize chromosome prediction loss if enabled
        if enable_chromosome_prediction:
            self.chromosome_loss = nn.BCEWithLogitsLoss(reduction='mean')
            # Create chromosome mapping for label encoding
            self.chromosome_to_idx = self._create_chromosome_mapping(num_chromosomes)

        # Storage for validation epoch aggregation
        self.validation_preds: List[Dict[str, torch.Tensor]] = []
        self.validation_targets: List[torch.Tensor] = []

        # Storage for first epoch mean motif activations (save memory)
        self.first_epoch_mean_motif_activations: List[torch.Tensor] = []
        self.first_epoch_complete = False

    def _create_chromosome_mapping(self, num_chromosomes: int) -> Dict[str, int]:
        """Create mapping from chromosome names to indices for classification."""
        # Standard human chromosomes plus X, Y, and unknown
        chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'unknown']
        
        # Ensure we don't exceed num_chromosomes
        if len(chromosomes) > num_chromosomes:
            chromosomes = chromosomes[:num_chromosomes]
        
        # Add padding if needed
        while len(chromosomes) < num_chromosomes:
            chromosomes.append(f'chr_extra_{len(chromosomes)}')
            
        chromosome_to_idx = {chrom: idx for idx, chrom in enumerate(chromosomes)}
        
        print(f"Created chromosome mapping with {len(chromosome_to_idx)} chromosomes:")
        for chrom, idx in sorted(chromosome_to_idx.items(), key=lambda x: x[1]):
            print(f"  {chrom}: {idx}")
            
        return chromosome_to_idx
    
    def _encode_chromosomes(self, chromosome_names: List[str]) -> torch.Tensor:
        """Convert chromosome names to indices for classification."""
        if not hasattr(self, 'chromosome_to_idx'):
            raise ValueError("Chromosome mapping not initialized. Enable chromosome prediction first.")
            
        indices = []
        for chrom in chromosome_names:
            if chrom in self.chromosome_to_idx:
                indices.append(self.chromosome_to_idx[chrom])
            else:
                # Use 'unknown' category for unmapped chromosomes
                if 'unknown' in self.chromosome_to_idx:
                    indices.append(self.chromosome_to_idx['unknown'])
                else:
                    # Fallback to last index if no unknown category
                    indices.append(len(self.chromosome_to_idx) - 1)
                    
        return torch.tensor(indices, dtype=torch.long)


    def forward(
        self, dna_seq: torch.Tensor, return_motif_activations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            dna_seq: DNA sequence tensor
            return_motif_activations: Whether to return motif activations

        Returns:
            Model predictions dictionary
        """
        if return_motif_activations:
            return self.model.forward_with_motif_activations(dna_seq)
        else:
            return self.model(dna_seq)

    def _calculate_losses(
        self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate ShapeScaleLoss for training with nucleotide masking support.

        Args:
            batch: Batch dictionary containing 'atac' targets and 'sequence' input
            preds: Predictions dictionary containing 'regression' output

        Returns:
            Dictionary of calculated losses
        """
        targets = batch["atac"]  # (batch, N, seq_len) - N-dimensional ATAC targets
        sequence = batch["sequence"]  # (batch, 4, seq_len) - DNA sequence one-hot
        input_len = targets.shape[-1]
        crop_len = 2048

        # Check for NaN/Inf in inputs
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: NaN/Inf detected in targets")
            targets = torch.where(
                torch.isnan(targets) | torch.isinf(targets),
                torch.zeros_like(targets),
                targets,
            )

        if (
            torch.isnan(preds["regression"]).any()
            or torch.isinf(preds["regression"]).any()
        ):
            print("Warning: NaN/Inf detected in predictions")
            preds["regression"] = torch.where(
                torch.isnan(preds["regression"]) | torch.isinf(preds["regression"]),
                torch.zeros_like(preds["regression"]),
                preds["regression"],
            )

        start = (input_len - crop_len) // 2
        end = start + crop_len

        # Crop predictions and targets to central 2048bp
        pred_reg_cropped = preds["regression"][:, :, start:end]  # (batch, N, 2048)
        targets_cropped = targets[:, :, start:end]  # (batch, N, 2048)

        # Create sample-level mask based on nucleotide count in center 2048bp
        sequence_cropped = sequence[:, :, start:end]  # (batch, 4, 2048)
        # Count valid nucleotides (non-zero one-hot positions) per sample
        valid_nucleotides_per_sample = sequence_cropped.sum(dim=1).sum(dim=1)  # (batch,) - sum across 4 bases and 2048 positions

        # Sample is valid if it has >=(2048-200)=1848 valid nucleotides in center 2048bp
        # This means we exclude samples with more than 200 masked positions
        sample_mask = valid_nucleotides_per_sample >= (2048 - 50)  # (batch,) - True for valid samples

        losses = {}

        # Only compute ATAC losses if chromosome prediction is not enabled
        if not self.hparams.enable_chromosome_prediction:
            # Calculate ShapeScaleLoss while maintaining proper per-sequence normalization
            batch_size, num_samples, seq_len = pred_reg_cropped.shape

            # Only compute loss for valid samples (those with >200 nucleotides)
            if sample_mask.any():
                # Filter to valid samples
                valid_indices = torch.where(sample_mask)[0]
                pred_valid = pred_reg_cropped[valid_indices]  # (valid_batch, num_samples, seq_len)
                target_valid = targets_cropped[valid_indices]  # (valid_batch, num_samples, seq_len)

                # Reshape to treat each sequence independently: (valid_batch * num_samples, seq_len)
                pred_reshaped = pred_valid.view(-1, seq_len)
                target_reshaped = target_valid.view(-1, seq_len)

                # Compute loss for valid samples only
                shape_scale_loss = self.shape_scale_loss(pred_reshaped, target_reshaped)
            else:
                # No valid samples in this batch
                shape_scale_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
            
            # Calculate count loss using PoissonNLL for valid samples only
            if "count" in preds:
                if sample_mask.any():
                    # Filter to valid samples
                    valid_indices = torch.where(sample_mask)[0]
                    target_valid = targets_cropped[valid_indices]  # (valid_batch, num_samples, seq_len)
                    pred_count_valid = preds["count"][valid_indices]  # (valid_batch, num_samples)

                    # Calculate count loss for valid samples
                    target_sums = target_valid.sum(dim=2)  # (valid_batch, num_samples)
                    target_sums_log10 = torch.log10(target_sums + 1)

                    # Reshape for PoissonNLL: (valid_batch * num_samples)
                    pred_count_flat = pred_count_valid.view(-1)
                    target_sums_flat = target_sums_log10.view(-1)

                    count_loss = self.poisson_loss(pred_count_flat, target_sums_flat)
                else:
                    # No valid samples
                    count_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)

                losses["count_loss"] = count_loss

                # Combine losses with weighting
                total_loss = shape_scale_loss + count_loss
            else:
                total_loss = shape_scale_loss
                losses["count_loss"] = torch.tensor(0.0, device=shape_scale_loss.device)
        else:
            # When chromosome prediction is enabled, skip ATAC losses
            shape_scale_loss = torch.tensor(0.0, device=targets.device)
            total_loss = torch.tensor(0.0, device=targets.device)
            losses["count_loss"] = torch.tensor(0.0, device=targets.device)
        
        # Calculate chromosome loss if enabled (also apply sample masking)
        if self.hparams.enable_chromosome_prediction and "chromosome" in preds:
            # Use chromosome labels directly from batch
            if "chromosome" in batch:
                if sample_mask.any():
                    # Filter to valid samples
                    valid_indices = torch.where(sample_mask)[0]
                    chromosome_labels = batch["chromosome"][valid_indices].to(preds["chromosome"].device)
                    chromosome_preds = preds["chromosome"][valid_indices]

                    # Convert chromosome labels to multi-hot encoding for BCELoss
                    # Handle both single labels and multi-labels
                    if chromosome_labels.dim() == 1:
                        # Single label case - convert to multi-hot
                        valid_batch_size = chromosome_labels.size(0)
                        multi_hot_labels = torch.zeros(valid_batch_size, self.hparams.num_chromosomes,
                                                     device=chromosome_labels.device, dtype=torch.float32)
                        multi_hot_labels.scatter_(1, chromosome_labels.unsqueeze(1), 1.0)
                    else:
                        # Already multi-hot encoded
                        multi_hot_labels = chromosome_labels.float()

                    # Calculate BCE loss for valid samples only
                    chromosome_loss = self.chromosome_loss(chromosome_preds, multi_hot_labels)
                else:
                    # No valid samples
                    chromosome_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
                    multi_hot_labels = None
                    chromosome_preds = None

                losses["chromosome_loss"] = chromosome_loss
                
                # Set total loss based on whether ATAC losses are enabled
                if self.hparams.enable_chromosome_prediction:
                    # Only chromosome loss (ATAC losses are disabled)
                    total_loss = self.hparams.chromosome_loss_weight * chromosome_loss
                else:
                    # Add to existing ATAC losses with weighting
                    total_loss = total_loss + self.hparams.chromosome_loss_weight * chromosome_loss
                
                # Calculate multiple accuracy metrics for multi-label classification (only for valid samples)
                if sample_mask.any() and multi_hot_labels is not None:
                    with torch.no_grad():
                        # Convert predictions to probabilities for valid samples
                        pred_probs = torch.sigmoid(chromosome_preds)

                        # Multi-label accuracy metrics
                        # Exact match accuracy (all labels must match)
                        pred_binary = (pred_probs > 0.5).float()
                        exact_match_accuracy = (pred_binary == multi_hot_labels).all(dim=1).float().mean()
                        losses["chromosome_accuracy"] = exact_match_accuracy

                        # Hamming accuracy (per-label accuracy)
                        hamming_accuracy = (pred_binary == multi_hot_labels).float().mean()
                        losses["chromosome_hamming_accuracy"] = hamming_accuracy

                        # Top-k recall (if any of top-k predictions match any true label)
                        _, top_k_indices = torch.topk(pred_probs, k=5, dim=1)
                        top_k_binary = torch.zeros_like(pred_probs)
                        top_k_binary.scatter_(1, top_k_indices, 1.0)
                        top_k_recall = ((top_k_binary * multi_hot_labels).sum(dim=1) > 0).float().mean()
                        losses["chromosome_top5_recall"] = top_k_recall

                        # Confidence-based accuracy (only evaluate confident predictions)
                        max_probs = pred_probs.max(dim=1)[0]
                        confident_mask = max_probs > 0.7
                        if confident_mask.sum() > 0:
                            confident_accuracy = (pred_binary[confident_mask] == multi_hot_labels[confident_mask]).all(dim=1).float().mean()
                            losses["chromosome_soft_accuracy"] = confident_accuracy
                        else:
                            losses["chromosome_soft_accuracy"] = torch.tensor(1.0, device=pred_probs.device)

                        # F1 Score (harmonic mean of precision and recall)
                        # Handle case where no predictions are made
                        eps = 1e-8
                        tp = (pred_binary * multi_hot_labels).sum(dim=1)
                        fp = (pred_binary * (1 - multi_hot_labels)).sum(dim=1)
                        fn = ((1 - pred_binary) * multi_hot_labels).sum(dim=1)

                        precision = tp / (tp + fp + eps)
                        recall = tp / (tp + fn + eps)
                        f1_score = 2 * precision * recall / (precision + recall + eps)
                        losses["chromosome_f1_score"] = f1_score.mean()
                else:
                    # No valid samples - set all chromosome metrics to zero
                    losses["chromosome_accuracy"] = torch.tensor(0.0, device=targets.device)
                    losses["chromosome_hamming_accuracy"] = torch.tensor(0.0, device=targets.device)
                    losses["chromosome_top5_recall"] = torch.tensor(0.0, device=targets.device)
                    losses["chromosome_soft_accuracy"] = torch.tensor(0.0, device=targets.device)
                    losses["chromosome_f1_score"] = torch.tensor(0.0, device=targets.device)
            else:
                # No chromosome information available - set all metrics to zero
                losses["chromosome_loss"] = torch.tensor(0.0, device=shape_scale_loss.device)
                losses["chromosome_accuracy"] = torch.tensor(0.0, device=shape_scale_loss.device)
                losses["chromosome_hamming_accuracy"] = torch.tensor(0.0, device=shape_scale_loss.device)
                losses["chromosome_top5_recall"] = torch.tensor(0.0, device=shape_scale_loss.device)
                losses["chromosome_soft_accuracy"] = torch.tensor(0.0, device=shape_scale_loss.device)
                losses["chromosome_f1_score"] = torch.tensor(0.0, device=shape_scale_loss.device)
        else:
            # Chromosome prediction disabled - set all metrics to zero
            losses["chromosome_loss"] = torch.tensor(0.0, device=shape_scale_loss.device)
            losses["chromosome_accuracy"] = torch.tensor(0.0, device=shape_scale_loss.device)
            losses["chromosome_hamming_accuracy"] = torch.tensor(0.0, device=shape_scale_loss.device)
            losses["chromosome_top5_recall"] = torch.tensor(0.0, device=shape_scale_loss.device)
            losses["chromosome_soft_accuracy"] = torch.tensor(0.0, device=shape_scale_loss.device)
            losses["chromosome_f1_score"] = torch.tensor(0.0, device=shape_scale_loss.device)

        # Store losses
        losses["shape_scale_loss"] = shape_scale_loss
        losses["loss"] = total_loss

        # Store example data for validation plotting (don't log per step)
        max_vals_per_seq = targets_cropped.max(dim=2)[0].max(dim=1)[0]  # (batch,)

        # Store examples for different ranges for validation plotting
        value_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        range_examples = {}
        for low, high in value_ranges:
            in_range_mask = (max_vals_per_seq >= low) & (max_vals_per_seq < high)
            if in_range_mask.any():
                first_idx = torch.where(in_range_mask)[0][0]
                range_examples[f"{low}_{high}"] = {
                    "batch_idx": first_idx.item(),
                    "max_val": max_vals_per_seq[first_idx].item(),
                    "pred_signal": pred_reg_cropped[first_idx].cpu(),
                    "target_signal": targets_cropped[first_idx].cpu(),
                }

        # Store for validation epoch plotting (only during validation)
        if self.training:
            # Don't store examples during training to save memory
            pass
        else:
            # Only store one example per range across entire validation epoch
            if not hasattr(self, "range_examples_epoch"):
                self.range_examples_epoch = {}

            # Only add examples if we don't already have one for this range
            for range_key, example_data in range_examples.items():
                if range_key not in self.range_examples_epoch:
                    self.range_examples_epoch[range_key] = example_data

        # Calculate and log sample masking statistics
        valid_sample_fraction = sample_mask.float().mean()  # Fraction of samples that are valid
        avg_nucleotide_count = valid_nucleotides_per_sample.float().mean()  # Average valid nucleotides per sample
        losses["valid_sample_fraction"] = valid_sample_fraction
        losses["avg_nucleotide_count"] = avg_nucleotide_count

        return losses

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Single training step.

        Args:
            batch: Batch of training data
            batch_idx: Index of current batch

        Returns:
            Loss tensor for backpropagation
        """
        # Check if we need to collect motif activations (first epoch only)
        collect_motif_activations = (
            self.current_epoch == 0 and not self.first_epoch_complete
        )
        collect_motif_activations=False
        if collect_motif_activations:
            preds = self.forward(batch["sequence"], return_motif_activations=True)
            # motif_activations are already ReLU'd and mean-pooled in the model
            # Shape: (batch, num_motifs)
            mean_motif_activations = preds["motif_activations"]
            # Move to CPU and store
            self.first_epoch_mean_motif_activations.append(
                mean_motif_activations.detach().cpu()
            )
        else:
            preds = self.forward(batch["sequence"])

        losses = self._calculate_losses(batch, preds)

        loss = losses["loss"]

        # Handle NaN/Inf losses
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected at batch {batch_idx}")
            self._log_problematic_losses(losses)
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        # Log training metrics
        batch_size = batch["sequence"].size(0)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Log detailed losses periodically to reduce overhead
        if batch_idx % 100 == 0:
            for name, value in losses.items():
                if name != "loss":  # Exclude already logged metrics
                    self.log(
                        f"train_{name}",
                        value,
                        on_step=True,
                        on_epoch=False,
                        batch_size=batch_size,
                    )

        return loss

    def _log_problematic_losses(self, losses: Dict[str, torch.Tensor]) -> None:
        """Log information about problematic loss values."""
        for name, value in losses.items():
            if torch.isnan(value) or torch.isinf(value):
                print(f"  {name}: {value}")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Single validation step.

        Args:
            batch: Batch of validation data
            batch_idx: Index of current batch

        Returns:
            Validation loss
        """
        preds = self.forward(batch["sequence"])
        losses = self._calculate_losses(batch, preds)

        # Log validation losses
        batch_size = batch["sequence"].size(0)
        for name, value in losses.items():
            self.log(
                f"val_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        # Store predictions and targets for epoch-end analysis
        self.validation_preds.append({"regression": preds["regression"]})
        self.validation_targets.append(batch["atac"])

        return losses["loss"]

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Single prediction step with masking support.

        Args:
            batch: Batch of input data
            batch_idx: Index of current batch
            dataloader_idx: Index of current dataloader

        Returns:
            Dictionary with predictions and observations
        """
        preds = self.forward(batch["sequence"])

        targets = batch["atac"]
        sequence = batch["sequence"]
        input_len = targets.shape[-1]
        crop_len = 2048

        start = (input_len - crop_len) // 2
        end = start + crop_len

        # Crop predictions and targets to central 2048bp
        pred_reg_cropped = preds["regression"][:, :, start:end]
        targets_cropped = targets[:, :, start:end]

        # Create sample-level mask based on nucleotide count in center 2048bp
        sequence_cropped = sequence[:, :, start:end]
        valid_nucleotides_per_sample = sequence_cropped.sum(dim=1).sum(dim=1)  # (batch,)
        sample_mask = valid_nucleotides_per_sample >= (2048 - 50)  # (batch,) - True for valid samples

        result = {
            "predictions": pred_reg_cropped,  # Full predictions for all positions
            "observations": targets_cropped,   # Full observations for all positions
            "sample_mask": sample_mask,  # Sample-level mask for downstream analysis
            "nucleotide_count": valid_nucleotides_per_sample  # Nucleotide counts per sample
        }

        # Add count predictions if available
        if "count" in preds:
            result["count_predictions"] = preds["count"]
            # For evaluation: use full count observations (all positions)
            target_sums = targets_cropped.sum(dim=2)  # (batch, num_samples)
            target_sums_log10 = torch.log10(target_sums + 1)
            result["count_observations"] = target_sums_log10
        
        # Add chromosome predictions if available
        if "chromosome" in preds:
            result["chromosome_predictions"] = preds["chromosome"]
            # Add chromosome labels if available
            if "chromosome" in batch:
                chromosome_labels = batch["chromosome"]
                result["chromosome_observations"] = chromosome_labels
                # Also get chromosome names for logging if region_info is available
                if "region_info" in batch:
                    chromosome_names = [info["chromosome"] for info in batch["region_info"]]
                    result["chromosome_names"] = chromosome_names
            
        return result

    def on_validation_epoch_end(self) -> None:
        """Process validation results at the end of each epoch."""
        if not self.validation_preds or not self.validation_targets:
            return
            
        # Aggregate predictions and targets from all validation steps
        all_preds = torch.cat(
            [pred["regression"] for pred in self.validation_preds], dim=0
        )
        all_targets = torch.cat(self.validation_targets, dim=0)

        # Crop to central 2048bp for fair comparison
        input_len = all_targets.shape[-1]
        crop_len = 2048
        start = (input_len - crop_len) // 2
        end = start + crop_len
        all_preds_cropped = all_preds[:, :, start:end]
        all_targets_cropped = all_targets[:, :, start:end]

        # Use shared plotting logic
        self._shared_validation_plotting(all_preds_cropped, all_targets_cropped)
        
        # Clear stored validation data to free memory
        self.validation_preds.clear()
        self.validation_targets.clear()
        
        # Reset specific regions logging flag for next epoch
        if hasattr(self, '_specific_regions_logged'):
            delattr(self, '_specific_regions_logged')

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Single test step - same as validation step but logs with test_ prefix."""
        preds = self.forward(batch["sequence"])
        losses = self._calculate_losses(batch, preds)
        
        # Log test losses
        batch_size = batch["sequence"].size(0)
        for name, value in losses.items():
            self.log(
                f"test_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        
        # Store predictions and targets for epoch-end analysis
        if not hasattr(self, 'test_preds'):
            self.test_preds = []
            self.test_targets = []
        
        self.test_preds.append({"regression": preds["regression"]})
        self.test_targets.append(batch["atac"])
        
        return losses["loss"]
    
    def on_test_epoch_end(self) -> None:
        """Process test results at the end of the test epoch with plotting."""
        if not hasattr(self, 'test_preds') or not self.test_preds or not self.test_targets:
            return
            
        # Aggregate predictions and targets from all test steps
        all_preds = torch.cat(
            [pred["regression"] for pred in self.test_preds], dim=0
        )
        all_targets = torch.cat(self.test_targets, dim=0)

        # Crop to central 2048bp for fair comparison (same as validation)
        input_len = all_targets.shape[-1]
        crop_len = 2048
        start = (input_len - crop_len) // 2
        end = start + crop_len
        
        all_preds_cropped = all_preds[:, :, start:end]
        all_targets_cropped = all_targets[:, :, start:end]
        
        # Use the same plotting and metrics logic as validation
        self._shared_validation_plotting(all_preds_cropped, all_targets_cropped, prefix="test")
        
        # Clear stored test data to free memory
        self.test_preds.clear()
        self.test_targets.clear()
    
    @classmethod
    def load_from_checkpoint_flexible(
        cls,
        checkpoint_path: str,
        strict: bool = False,
        **kwargs
    ) -> "DNAATACPredictionLightningModule":
        """Load model from checkpoint with flexible parameter matching.
        
        This allows loading from checkpoints that may not have chromosome_head or count_head
        parameters (for backwards compatibility with pretrained models).
        
        Args:
            checkpoint_path: Path to the checkpoint file
            strict: If False, allows missing keys (like chromosome_head parameters)
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Loaded model instance
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract hyperparameters from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        hparams.update(kwargs)  # Allow override of hyperparameters
        
        # Create model instance
        model = cls(**hparams)
        
        # Handle _orig_mod prefix from torch.compile
        state_dict = checkpoint['state_dict']
        if any(key.startswith('model._orig_mod') for key in state_dict.keys()):
            print("Removing _orig_mod prefix from compiled model checkpoint")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model._orig_mod'):
                    new_key = key.replace('model._orig_mod', 'model')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # Load state dict with strict=False to handle missing chromosome_head/count_head
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=strict
        )
        
        if not strict and (missing_keys or unexpected_keys):
            print(f"Loaded checkpoint with flexible matching:")
            if missing_keys:
                # Filter chromosome and count head related keys for cleaner output
                chrom_keys = [k for k in missing_keys if 'chromosome_head' in k]
                count_keys = [k for k in missing_keys if 'count_head' in k]
                other_keys = [k for k in missing_keys if 'chromosome_head' not in k and 'count_head' not in k]
                
                if chrom_keys:
                    print(f"  Missing chromosome_head keys: {len(chrom_keys)} (new feature)")
                if count_keys:
                    print(f"  Missing count_head keys: {len(count_keys)} (new feature)")
                if other_keys:
                    print(f"  Other missing keys: {other_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")
                
        return model
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
    def _shared_validation_plotting(self, all_preds_cropped: torch.Tensor, all_targets_cropped: torch.Tensor, prefix: str = "val") -> None:
        """Shared plotting logic used by both compressed and original models for validation and test."""
        
        num_samples = all_targets_cropped.shape[1]
        total_val_batch_size = all_targets_cropped.shape[0]  # Total validation samples

        def compute_correlation_vectorized(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            """Compute Pearson correlation coefficients between tensors."""
            # Center the data
            x_centered = x - x.mean(dim=-1, keepdim=True)
            y_centered = y - y.mean(dim=-1, keepdim=True)

            # Calculate correlation components
            numerator = torch.sum(x_centered * y_centered, dim=-1)
            x_var = torch.sum(x_centered**2, dim=-1)
            y_var = torch.sum(y_centered**2, dim=-1)

            # Safe correlation calculation
            eps = 1e-8
            denominator = torch.sqrt((x_var + eps) * (y_var + eps))

            # Create output tensor and handle valid correlations
            corr = torch.zeros_like(numerator)
            valid_mask = (x_var > eps) & (y_var > eps) & (denominator > eps)

            if valid_mask.any():
                corr_values = numerator[valid_mask] / denominator[valid_mask]
                corr_values = torch.clamp(corr_values, -1.0, 1.0)
                corr_values = torch.where(
                    torch.isnan(corr_values), torch.zeros_like(corr_values), corr_values
                )
                corr[valid_mask] = corr_values

            return corr

        # Calculate metrics for each ATAC sample
        for sample_idx in range(num_samples):
            # Region-level metrics using log10(x/1e3+1) transformation for counts
            region_pred_sum = all_preds_cropped[:, sample_idx, :].sum(
                dim=1
            )  # (batch,) - sum across positions (counts)
            region_target_sum = all_targets_cropped[:, sample_idx, :].sum(
                dim=1
            )  # (batch,) - sum across positions (counts)
            region_pred_log_count = torch.log10(region_pred_sum / 1e3 + 1)
            region_target_log_count = torch.log10(region_target_sum / 1e3 + 1)

            region_mse = F.mse_loss(region_pred_log_count, region_target_log_count)

            region_corr = compute_correlation_vectorized(
                region_pred_log_count.unsqueeze(0),
                region_target_log_count.unsqueeze(0),
            )[0]

            # Log metrics for each sample
            self.log(
                f"val_region_mse_sample_{sample_idx}",
                region_mse,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )
            self.log(
                f"val_region_correlation_sample_{sample_idx}",
                region_corr,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )

        # Calculate average metrics across all samples
        avg_region_corr = torch.stack(
            [
                self.trainer.logged_metrics.get(
                    f"{prefix}_region_correlation_sample_{i}", torch.tensor(0.0)
                ).to(all_preds_cropped.device)
                for i in range(num_samples)
            ]
        ).mean()

        self.log(
            f"{prefix}_region_correlation_avg",
            avg_region_corr,
            on_epoch=True,
            batch_size=total_val_batch_size,
        )
        
        # Calculate count metrics if available in logged metrics
        if f"{prefix}_count_loss" in self.trainer.logged_metrics:
            count_loss = self.trainer.logged_metrics[f"{prefix}_count_loss"]
            self.log(
                f"{prefix}_count_loss_avg",
                count_loss,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )

        # Calculate FC correlations for both 2 and 4 sample cases
        if num_samples >= 2:
            # Use log10(x/1e3+1) transformation for FC calculation
            pred_region_sum_all = all_preds_cropped.sum(
                dim=2
            )  # (batch, num_samples) - sum across positions
            target_region_sum_all = all_targets_cropped.sum(
                dim=2
            )  # (batch, num_samples) - sum across positions
            pred_region_log_count = torch.log10(
                pred_region_sum_all / 1e3 + 1
            )  # (batch, num_samples) - log10 count scale
            target_region_log_count = torch.log10(
                target_region_sum_all / 1e3 + 1
            )  # (batch, num_samples) - log10 count scale

            if num_samples == 2:
                # Calculate FC for SCCA condition using log10 count transformation
                pred_fc_scca = (
                    pred_region_log_count[:, 0] - pred_region_log_count[:, 1]
                )  # SCCApos - SCCAneg
                target_fc_scca = (
                    target_region_log_count[:, 0] - target_region_log_count[:, 1]
                )  # SCCApos - SCCAneg

                # Calculate correlation for FC using vectorized function
                fc_corr_scca = compute_correlation_vectorized(
                    pred_fc_scca.unsqueeze(0), target_fc_scca.unsqueeze(0)
                )[0]

                # Log FC metrics
                self.log(
                    "val_fc_correlation_scca",
                    fc_corr_scca,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )

            elif num_samples == 4:
                # Calculate FC using log10 count transformation
                pred_fc_scc1 = (
                    pred_region_log_count[:, 0] - pred_region_log_count[:, 1]
                )  # SCC1pos - SCC1neg
                pred_fc_scca = (
                    pred_region_log_count[:, 2] - pred_region_log_count[:, 3]
                )  # SCCApos - SCCAneg

                target_fc_scc1 = target_region_log_count[:, 0] - target_region_log_count[:, 1]
                target_fc_scca = target_region_log_count[:, 2] - target_region_log_count[:, 3]

                # Calculate correlations for FC using vectorized function
                fc_corr_scc1 = compute_correlation_vectorized(
                    pred_fc_scc1.unsqueeze(0), target_fc_scc1.unsqueeze(0)
                )[0]

                fc_corr_scca = compute_correlation_vectorized(
                    pred_fc_scca.unsqueeze(0), target_fc_scca.unsqueeze(0)
                )[0]

                # Log FC metrics
                self.log(
                    "val_fc_correlation_scc1",
                    fc_corr_scc1,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )
                self.log(
                    "val_fc_correlation_scca",
                    fc_corr_scca,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )
                self.log(
                    "val_fc_correlation_avg",
                    (fc_corr_scc1 + fc_corr_scca) / 2,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )

        # Plotting for all samples - region-level scatter plots and high signal examples
        if self.logger and num_samples > 0:
            # Create scatter plots for each sample showing region-level counts
            for sample_idx in range(num_samples):
                # Scatter plot of log10(x/1e3+1) transformed region counts
                region_pred_sum = all_preds_cropped[:, sample_idx, :].sum(dim=1)
                region_target_sum = all_targets_cropped[:, sample_idx, :].sum(
                    dim=1
                )
                region_pred_log_count = torch.log10(region_pred_sum / 1e3 + 1)
                region_target_log_count = torch.log10(region_target_sum / 1e3 + 1)

                table_data = [
                    [x.item(), y.item()]
                    for (x, y) in zip(
                        region_target_log_count, region_pred_log_count
                    )
                ]
                table = wandb.Table(
                    data=table_data,
                    columns=["Observed Log10 Count", "Predicted Log10 Count"],
                )
                celltype_name = (
                    self.hparams.celltype_ids[sample_idx]
                    if self.hparams.celltype_ids
                    else f"Sample_{sample_idx}"
                )
                self.logger.experiment.log(
                    {
                        f"val_region_scatter_{celltype_name}": wandb.plot.scatter(
                            table,
                            "Observed Log10 Count",
                            "Predicted Log10 Count",
                            title=f"Region-Level Log10 Count ATAC Signal - {celltype_name}",
                        )
                    }
                )

            # High signal example visualization
            # Find the sample with highest signal across all samples
            max_signals_per_sample = all_targets_cropped.max(dim=2)[
                0
            ]  # (batch, num_samples)
            max_signal_overall = max_signals_per_sample.max(dim=1)[0]  # (batch,)

            if (
                max_signal_overall.max() > 0.5
            ):  # Only create if there's meaningful signal
                # Find the sample with the highest overall signal
                highest_signal_batch_idx = torch.argmax(max_signal_overall)
                highest_signal_sample_idx = torch.argmax(
                    max_signals_per_sample[highest_signal_batch_idx]
                )

                # Extract the sequences for this high-signal example
                pred_signal = (
                    all_preds_cropped[
                        highest_signal_batch_idx, highest_signal_sample_idx
                    ]
                    .cpu().float()
                    .numpy()
                )
                target_signal = (
                    all_targets_cropped[
                        highest_signal_batch_idx, highest_signal_sample_idx
                    ]
                    .cpu().float()
                    .numpy()
                )

                # Create position array - ensure it matches the actual signal length
                positions = np.arange(len(pred_signal))
                
                # Verify pred_signal and target_signal have same length
                if len(pred_signal) != len(target_signal):
                    print(f"Warning: Signal length mismatch - pred: {len(pred_signal)}, target: {len(target_signal)}")
                    # Use the shorter length to avoid plotting errors
                    min_len = min(len(pred_signal), len(target_signal))
                    pred_signal = pred_signal[:min_len]
                    target_signal = target_signal[:min_len]
                    positions = np.arange(min_len)

                celltype_name = (
                    self.hparams.celltype_ids[highest_signal_sample_idx]
                    if self.hparams.celltype_ids
                    else f"Sample_{highest_signal_sample_idx}"
                )

                # Main ATAC signal plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(
                    positions,
                    target_signal,
                    color="blue",
                    label="Observed ATAC Signal",
                    linewidth=1,
                )
                ax.plot(
                    positions,
                    pred_signal,
                    color="red",
                    label="Predicted ATAC Signal",
                    linewidth=1,
                )
                ax.set_xlabel("Position (bp)")
                ax.set_ylabel("Signal")
                ax.set_title(
                    f"High Signal Example - {celltype_name} (Max Signal: {max_signal_overall[highest_signal_batch_idx]:.2f})"
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Convert matplotlib figure to wandb image
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                img = Image.open(buf)
                plt.close(fig)

                self.logger.experiment.log(
                    {"val_high_signal_example": wandb.Image(img)}
                )

        # Plot signal magnitude range examples
        # Calculate max values per sequence for range categorization
        max_vals_per_seq = all_targets_cropped.max(dim=2)[0].max(dim=1)[0]  # (batch,)
        
        # Store examples for different ranges for validation plotting
        value_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        range_examples = {}
        for low, high in value_ranges:
            in_range_mask = (max_vals_per_seq >= low) & (max_vals_per_seq < high)
            if in_range_mask.any():
                first_idx = torch.where(in_range_mask)[0][0]
                range_examples[f"{low}_{high}"] = {
                    "batch_idx": first_idx.item(),
                    "max_val": max_vals_per_seq[first_idx].item(),
                    "pred_signal": all_preds_cropped[first_idx].cpu(),
                    "target_signal": all_targets_cropped[first_idx].cpu(),
                }

        if self.logger and range_examples:
            # Create line plots for each signal magnitude range
            for range_key, example_data in range_examples.items():
                low, high = range_key.split("_")
                pred_signal = example_data["pred_signal"]  # (num_samples, seq_len)
                target_signal = example_data["target_signal"]  # (num_samples, seq_len)
                max_val = example_data["max_val"]

                # Find the sample with highest signal in this example
                max_signals_per_sample = target_signal.max(dim=1)[0]  # (num_samples,)
                highest_sample_idx = torch.argmax(max_signals_per_sample)

                # Extract signals for the highest sample
                pred_seq = pred_signal[highest_sample_idx].float().numpy()
                target_seq = target_signal[highest_sample_idx].float().numpy()
                
                # Verify sequences have same length
                if len(pred_seq) != len(target_seq):
                    print(f"Warning: Range example sequence length mismatch - pred: {len(pred_seq)}, target: {len(target_seq)}")
                    # Use the shorter length to avoid plotting errors
                    min_len = min(len(pred_seq), len(target_seq))
                    pred_seq = pred_seq[:min_len]
                    target_seq = target_seq[:min_len]
                
                positions = np.arange(len(pred_seq))

                celltype_name = (
                    self.hparams.celltype_ids[highest_sample_idx]
                    if self.hparams.celltype_ids
                    else f"Sample_{highest_sample_idx}"
                )

                # Create line plot similar to high signal example
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(
                    positions,
                    target_seq,
                    color="blue",
                    label="Observed ATAC Signal",
                    linewidth=1,
                )
                ax.plot(
                    positions,
                    pred_seq,
                    color="red",
                    label="Predicted ATAC Signal",
                    linewidth=1,
                )
                ax.set_xlabel("Position (bp)")
                ax.set_ylabel("Signal")
                ax.set_title(
                    f"Signal Range {low}-{high} Example - {celltype_name} (Max Signal: {max_val:.2f})"
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Convert matplotlib figure to wandb image
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                img = Image.open(buf)
                plt.close(fig)

                self.logger.experiment.log(
                    {f"{prefix}_signal_range_{low}_{high}_example": wandb.Image(img)}
                )

        # Add specific genomic region visualizations
        if self.logger and prefix == "val":  # Only during validation
            self._plot_specific_genomic_regions()

    def _plot_specific_genomic_regions(self) -> None:
        """Plot specific genomic regions of interest to wandb."""
        # Skip if already logged in this epoch or no trainer access
        if hasattr(self, '_specific_regions_logged') or not hasattr(self.trainer, 'datamodule'):
            return
            
        self._specific_regions_logged = True
        
        # Define the specific regions to visualize
        specific_regions = [
            {"chr": "chr16", "start": 45408722, "end": 45409426, "name": "region1"},
            {"chr": "chr16", "start": 45383763, "end": 45384762, "name": "region2"},
            {"chr": "chr16", "start": 45379545, "end": 45379801, "name": "region3"},
            {"chr": "chr16", "start": 45356164, "end": 45357191, "name": "region4"}
        ]
        
        try:
            # Access the datamodule to get zarr datasets
            datamodule = self.trainer.datamodule
            if not hasattr(datamodule, 'cdz') or not hasattr(datamodule, 'genome'):
                # Try to setup if not already done
                if datamodule.cdz is None or datamodule.genome is None:
                    from caesar.io.zarr_io import CelltypeDenseZarrIO, SequenceDenseZarrIO
                    datamodule.cdz = CelltypeDenseZarrIO(datamodule.atac_zarr_path)
                    datamodule.genome = SequenceDenseZarrIO(datamodule.genome_zarr_path)
            
            # Get celltype information
            celltype_ids = datamodule.celltype_ids if hasattr(datamodule, 'celltype_ids') else self.hparams.get('celltype_ids', ['sample_0'])
            
            for region in specific_regions:
                self._plot_single_genomic_region(
                    region, datamodule.cdz, datamodule.genome, celltype_ids
                )
                
        except Exception as e:
            # Fallback to placeholder if data access fails
            print(f"Could not access zarr data for region plotting: {e}")
            self._create_placeholder_plot()
    
    def _plot_single_genomic_region(self, region: dict, cdz, genome, celltype_ids: list) -> None:
        """Plot a single genomic region with ATAC predictions vs observations."""
        try:
            chr_name = region['chr']
            start = region['start'] 
            end = region['end']
            region_name = region['name']
            
            # Add padding for model context
            extend_bp = 2048
            padded_start = max(0, start - extend_bp)
            padded_end = end + extend_bp
            
            # Get sequence data
            sequence = genome.get_track(chr_name, padded_start, padded_end, output_format='raw_array')
            sequence_tensor = torch.tensor(sequence.T.astype(np.float32), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Get ATAC signals for all celltypes
            atac = cdz.get_track(
                chr_name, padded_start, padded_end,
                output_format='raw_dict',
                aggregator='by_key',  # This gives shape (n_celltypes, seq_len)
                normalize_factor = 1e8,
                conv_size=20
            )
            # convert dict of array to single array
            atac = np.stack([atac[celltype] for celltype in celltype_ids], axis=0)
            
            # Stack ATAC signals: (num_celltypes, seq_len)
            atac_tensor = torch.tensor(atac.astype(np.float32), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Run model prediction
            self.eval()
            with torch.no_grad():
                pred_dict = self.forward(sequence_tensor.to(self.device))
                pred_atac = pred_dict['regression'].cpu()  # (1, num_celltypes, seq_len)
            
            # Create the plot
            fig, axes = plt.subplots(len(celltype_ids), 1, figsize=(12, 3 * len(celltype_ids)))
            if len(celltype_ids) == 1:
                axes = [axes]
            
            # Calculate positions relative to original region
            seq_len = sequence_tensor.shape[2]
            positions = np.arange(seq_len) + padded_start
            
            for i, celltype in enumerate(celltype_ids):
                ax = axes[i]
                
                # Plot observed and predicted
                ax.plot(positions, atac_tensor[0, i, :].numpy(), 
                       label='Observed', color='blue', alpha=0.7)
                ax.plot(positions, pred_atac[0, i, :].numpy(), 
                       label='Predicted', color='red', alpha=0.7)
                
                # Highlight the original region of interest
                ax.axvspan(start, end, alpha=0.2, color='gray', label='Region of Interest')
                
                ax.set_ylabel('ATAC Signal')
                ax.set_title(f'{celltype} - {chr_name}:{start:,}-{end:,}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Genomic Position (bp)')
            plt.tight_layout()
            
            # Convert to wandb image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            self.logger.experiment.log(
                {f"val_specific_region_{region_name}_{chr_name}_{start}_{end}": wandb.Image(img)}
            )
            
        except Exception as e:
            print(f"Error plotting region {region}: {e}")
    
    def _create_placeholder_plot(self) -> None:
        """Create placeholder plot when data access fails."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Specific Region Visualization\n\nRegions configured:\n"
                          f"chr16:45408722-45409426\nchr16:45383763-45384762\n"
                          f"chr16:45379545-45379801\nchr16:45356164-45357191\n\n"
                          f"Data access not available in current context",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Specific Genomic Region Plots - Configuration Ready")
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        self.logger.experiment.log(
            {"val_specific_regions_status": wandb.Image(img)}
        )
    
