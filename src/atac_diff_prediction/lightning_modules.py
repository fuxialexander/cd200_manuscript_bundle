"""
Separate Lightning modules for each training phase of ATAC differential prediction.

This module contains pure Lightning modules without model computation logic:
- Phase1LightningModule: For bias model training (Phase 1)
- Phase2LightningModule: For main model training (Phase 2)
- Phase3LightningModule: For differential prediction training (Phase 3)

Each Lightning module handles only training/validation logic and delegates
model computation to the corresponding phase-specific nn.Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any, Optional, List
import io
from PIL import Image

from atac_prediction.dna_atac_lightning_module import ShapeScaleLoss


def compute_jsd_vectorized(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Jensen-Shannon Divergence between predicted and target profiles.

    Both inputs are normalized to probability distributions before computing JSD.

    Args:
        pred: Predicted profiles tensor (batch, samples, seq_len)
        target: Target profiles tensor (batch, samples, seq_len)
        eps: Small constant for numerical stability

    Returns:
        JSD values tensor (batch, samples)
    """
    # Normalize to probability distributions (add eps for numerical stability)
    pred_norm = pred + eps
    target_norm = target + eps

    pred_prob = pred_norm / pred_norm.sum(dim=-1, keepdim=True)
    target_prob = target_norm / target_norm.sum(dim=-1, keepdim=True)

    # Compute M = (P + Q) / 2
    m = (pred_prob + target_prob) / 2

    # Compute KL divergences: KL(P||M) and KL(Q||M)
    kl_pm = torch.sum(pred_prob * torch.log(pred_prob / (m + eps) + eps), dim=-1)
    kl_qm = torch.sum(target_prob * torch.log(target_prob / (m + eps) + eps), dim=-1)

    # JSD = 0.5 * (KL(P||M) + KL(Q||M))
    jsd = 0.5 * (kl_pm + kl_qm)

    # Clamp to avoid numerical issues
    jsd = torch.clamp(jsd, 0.0, 1.0)

    return jsd


class Phase1LightningModule(pl.LightningModule):
    """
    Lightning module for Phase 1 bias model training.

    Handles training/validation of bias model on background regions with negative samples only.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        freeze_tn5_motifs: bool = True,
        **kwargs
    ):
        """
        Initialize Phase 1 Lightning module.

        Args:
            model: Phase1Model instance
            learning_rate: Learning rate for optimization
            freeze_tn5_motifs: Whether to freeze TN5 motif weights during training
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.freeze_tn5_motifs = freeze_tn5_motifs
        self.save_hyperparameters(ignore=['model'])

        # Storage for validation epoch aggregation
        self.validation_preds: List[Dict[str, torch.Tensor]] = []
        self.validation_targets: List[torch.Tensor] = []

        # Phase 1 specific shape scale loss (scale component disabled)
        self.bias_shape_scale_loss = ShapeScaleLoss(
            shape_mode="ce", tau=1.0, scale_weight=0.2, eps=1e-6, reduction="sum"
        )

        # Freeze TN5 motifs if requested
        if self.freeze_tn5_motifs:
            self._freeze_tn5_motifs()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through Phase 1 model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for Phase 1."""
        try:
            sequence = batch['sequence']
            atac_targets = batch['atac']

            # Skip if sequence is empty
            if sequence.numel() == 0 or atac_targets.numel() == 0:
                return None

            # Forward pass through Phase 1 model
            outputs = self.forward(sequence)

            # Calculate losses using shape scale loss
            if 'bias_regression' in outputs:
                regression_loss = self.bias_shape_scale_loss(outputs['bias_regression'], atac_targets)
                total_loss = regression_loss

                # Log losses
                self.log('train_loss', total_loss, prog_bar=True)
                self.log('train_bias_regression_loss', regression_loss)

                return total_loss
            else:
                return None

        except Exception as e:
            print(f"Phase 1 training step error: {e}")
            return None

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for Phase 1."""
        try:
            sequence = batch['sequence']
            atac_targets = batch['atac']

            # Skip if sequence is empty
            if sequence.numel() == 0 or atac_targets.numel() == 0:
                return None

            # Forward pass through Phase 1 model
            outputs = self.forward(sequence)

            # Calculate losses using shape scale loss
            if 'bias_regression' in outputs:
                regression_loss = self.bias_shape_scale_loss(outputs['bias_regression'], atac_targets)
                total_loss = regression_loss

                # Log losses
                batch_size = sequence.size(0)
                self.log('val_loss', total_loss, prog_bar=True, batch_size=batch_size)
                self.log('val_bias_regression_loss', regression_loss, batch_size=batch_size)

                # Store predictions and targets for epoch-end analysis
                self.validation_preds.append({"regression": outputs["bias_regression"]})
                self.validation_targets.append(atac_targets)

                return total_loss
            else:
                return None

        except Exception as e:
            print(f"Phase 1 validation step error: {e}")
            return None

    def on_validation_epoch_end(self) -> None:
        """End of validation epoch with Phase 1 bias model plotting."""
        if self.validation_preds and self.validation_targets:
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

            # Call shared validation plotting with Phase 1 prefix
            self._shared_validation_plotting(all_preds_cropped, all_targets_cropped, prefix="val_phase1")

            # Clear stored validation data to free memory
            self.validation_preds.clear()
            self.validation_targets.clear()

    def configure_optimizers(self):
        """Configure optimizer for Phase 1."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _freeze_tn5_motifs(self) -> None:
        """
        Freeze TN5 motif parameters in the bias model to prevent training.

        This method searches for TN5 convolution layers in the bias model and
        freezes their parameters (sets requires_grad=False).
        """
        if hasattr(self.model, 'bias_model'):
            bias_model = self.model.bias_model

            # Freeze TN5 convolution layer if it exists
            if hasattr(bias_model, 'tn5_conv'):
                for param in bias_model.tn5_conv.parameters():
                    param.requires_grad = False
                print("TN5 motif weights frozen in Phase 1 bias model training")
            else:
                print("Warning: No tn5_conv layer found in bias model")
        else:
            print("Warning: No bias_model found in Phase 1 model")

    def on_train_start(self) -> None:
        """Called at the start of training to ensure TN5 motifs remain frozen."""
        if self.freeze_tn5_motifs:
            self._freeze_tn5_motifs()

    def _shared_validation_plotting(self, all_preds_cropped: torch.Tensor, all_targets_cropped: torch.Tensor, prefix: str = "val") -> None:
        """Shared plotting logic for Phase 1 bias model validation."""
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        num_samples = all_targets_cropped.shape[1]
        total_val_batch_size = all_targets_cropped.shape[0]

        def compute_correlation_vectorized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
            # Region-level metrics using log10 transformation
            region_pred_means_orig = all_preds_cropped[:, sample_idx, :].sum(dim=1)
            region_target_means_orig = all_targets_cropped[:, sample_idx, :].sum(dim=1)
            region_pred_means_log10 = torch.log10(region_pred_means_orig/1e3 + 1)
            region_target_means_log10 = torch.log10(region_target_means_orig/1e3 + 1)

            region_mse = F.mse_loss(region_pred_means_log10, region_target_means_log10)

            region_corr = compute_correlation_vectorized(
                region_pred_means_log10.unsqueeze(0),
                region_target_means_log10.unsqueeze(0),
            )[0]

            # Profile-level JSD metrics (comparing normalized probability distributions)
            sample_jsd_values = compute_jsd_vectorized(
                all_preds_cropped[:, sample_idx:sample_idx+1, :],  # (batch, 1, seq_len)
                all_targets_cropped[:, sample_idx:sample_idx+1, :]  # (batch, 1, seq_len)
            )
            region_jsd = sample_jsd_values.mean()  # Average JSD across all regions

            # Log metrics for each sample
            self.log(
                f"{prefix}_region_mse_sample_{sample_idx}",
                region_mse,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )
            self.log(
                f"{prefix}_region_correlation_sample_{sample_idx}",
                region_corr,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )
            self.log(
                f"{prefix}_profile_jsd_sample_{sample_idx}",
                region_jsd,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )

        # Calculate average correlation across all samples
        avg_region_corr = torch.stack(
            [
                self.trainer.logged_metrics.get(
                    f"{prefix}_region_correlation_sample_{i}", torch.tensor(0.0)
                ).to(all_preds_cropped.device)
                for i in range(num_samples)
            ]
        ).mean()

        # Calculate average JSD across all samples
        avg_profile_jsd = torch.stack(
            [
                self.trainer.logged_metrics.get(
                    f"{prefix}_profile_jsd_sample_{i}", torch.tensor(0.0)
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
        self.log(
            f"{prefix}_profile_jsd_avg",
            avg_profile_jsd,
            on_epoch=True,
            batch_size=total_val_batch_size,
        )

        # Plotting for all samples - region-level scatter plots
        if self.logger and num_samples > 0:
            # Create scatter plots for each sample showing region-level means
            for sample_idx in range(num_samples):
                # Scatter plot of log10-transformed region means
                region_pred_means_orig = all_preds_cropped[:, sample_idx, :].sum(dim=1)
                region_target_means_orig = all_targets_cropped[:, sample_idx, :].sum(dim=1)
                region_pred_means_log10 = torch.log10(region_pred_means_orig/1e3 + 1)
                region_target_means_log10 = torch.log10(region_target_means_orig/1e3 + 1)

                table_data = [
                    [x.item(), y.item()]
                    for (x, y) in zip(
                        region_target_means_log10, region_pred_means_log10
                    )
                ]
                table = wandb.Table(
                    data=table_data,
                    columns=["Observed Log10 Mean", "Predicted Log10 Mean"],
                )
                celltype_name = f"Sample_{sample_idx}"  # Phase 1 uses generic names

                self.logger.experiment.log(
                    {
                        f"{prefix}_region_scatter_{celltype_name}": wandb.plot.scatter(
                            table,
                            "Observed Log10 Mean",
                            "Predicted Log10 Mean",
                            title=f"Region-Level Log10 Mean ATAC Signal - {celltype_name} (Phase 1 Bias Model)",
                        )
                    }
                )

            # High signal example visualization
            # Find the sample with highest signal across all samples
            max_signals_per_sample = all_targets_cropped.max(dim=2)[0]  # (batch, num_samples)
            max_signal_overall = max_signals_per_sample.max(dim=1)[0]  # (batch,)

            if max_signal_overall.max() > 0.5:  # Only create if there's meaningful signal
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

                celltype_name = f"Sample_{highest_signal_sample_idx}"

                # Main ATAC signal plot
                fig, ax = plt.subplots(figsize=(12, 6))
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
                    f"High Signal Example - {celltype_name} (Max Signal: {max_signal_overall[highest_signal_batch_idx]:.2f}) - Phase 1 Bias Model"
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
                    {f"{prefix}_high_signal_example": wandb.Image(img)}
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

                celltype_name = f"Sample_{highest_sample_idx}"

                # Create line plot similar to high signal example
                fig, ax = plt.subplots(figsize=(12, 6))
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
                    f"Signal Range {low}-{high} Example - {celltype_name} (Max Signal: {max_val:.2f}) - Phase 1 Bias Model"
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

        # Add specific genomic region visualizations for Phase 1
        if self.logger and prefix == "val_phase1":  # Only during validation
            self._plot_specific_genomic_regions()

    def _plot_specific_genomic_regions(self) -> None:
        """Plot specific genomic regions of interest to wandb."""
        # Skip if no trainer access
        if not hasattr(self.trainer, 'datamodule'):
            return

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
            celltype_ids = datamodule.celltype_ids if hasattr(datamodule, 'celltype_ids') else ['sample_0']

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
                pred_atac = pred_dict['bias_regression'].cpu()  # (1, num_celltypes, seq_len)

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
                ax.set_title(f'{celltype} - {chr_name}:{start:,}-{end:,} (Phase 1 Bias Model)')
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
                {f"val_phase1_specific_region_{region_name}_{chr_name}_{start}_{end}": wandb.Image(img)}
            )

        except Exception as e:
            print(f"Error plotting region {region}: {e}")

    def _create_placeholder_plot(self) -> None:
        """Create placeholder plot when data access fails."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Specific Region Visualization (Phase 1)\n\nRegions configured:\n"
                          f"chr16:45408722-45409426\nchr16:45383763-45384762\n"
                          f"chr16:45379545-45379801\nchr16:45356164-45357191\n\n"
                          f"Data access not available in current context",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Specific Genomic Region Plots - Configuration Ready (Phase 1)")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        self.logger.experiment.log(
            {"val_phase1_specific_regions_status": wandb.Image(img)}
        )


class Phase2LightningModule(pl.LightningModule):
    """
    Lightning module for Phase 2 main model training.

    Handles training/validation of main ATAC model on negative samples only.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        celltype_ids: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize Phase 2 Lightning module.

        Args:
            model: Phase2Model instance
            learning_rate: Learning rate for optimization
            celltype_ids: List of celltype identifiers for logging
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.celltype_ids = celltype_ids
        self.save_hyperparameters(ignore=['model'])

        # Storage for validation epoch aggregation
        self.validation_preds: List[Dict[str, torch.Tensor]] = []
        self.validation_targets: List[torch.Tensor] = []

        # Shape scale loss (with scale component enabled)
        self.shape_scale_loss = ShapeScaleLoss(
            shape_mode="ce", tau=1.0, scale_weight=0.2, eps=1e-6, reduction="sum"
        )

        # Poisson loss for count prediction
        self.poisson_loss = nn.PoissonNLLLoss(log_input=True, reduction='mean')

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through Phase 2 model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for Phase 2."""
        sequence = batch['sequence']
        preds = self.forward(sequence)
        losses = self._calculate_losses(batch, preds)

        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True)

        return losses['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for Phase 2."""
        sequence = batch['sequence']
        preds = self.forward(sequence)
        losses = self._calculate_losses(batch, preds)

        # Log losses
        batch_size = sequence.size(0)
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, on_epoch=True, batch_size=batch_size)

        # Store predictions and targets for epoch-end analysis
        if 'regression' in preds:
            pred_dict = {"regression": preds["regression"]}
            if 'count' in preds:
                pred_dict["count"] = preds["count"]
            self.validation_preds.append(pred_dict)
            self.validation_targets.append(batch["atac"])

        return losses['loss']

    def _calculate_losses(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate Phase 2 losses (main model)."""
        losses = {}

        # Profile loss
        if 'regression' in preds:
            profile_loss = self.shape_scale_loss(preds['regression'], batch['atac'])
            losses['profile_loss'] = profile_loss

        # Count loss
        if 'count' in preds:
            target_counts = batch['atac'].sum(dim=2)  # Sum across sequence length
            count_loss = self.poisson_loss(torch.log(preds['count'] + 1e-8), target_counts)
            losses['count_loss'] = count_loss

        # Total loss
        total_loss = sum(losses.values())
        losses['loss'] = total_loss

        return losses

    def on_validation_epoch_end(self) -> None:
        """End of validation epoch with Phase 2 plotting."""
        if self.validation_preds and self.validation_targets:
            # Aggregate predictions and targets
            all_preds = torch.cat([pred["regression"] for pred in self.validation_preds], dim=0)
            all_targets = torch.cat(self.validation_targets, dim=0)

            # Aggregate count predictions if available
            all_count_preds = None
            if self.validation_preds[0].get("count") is not None:
                all_count_preds = torch.cat([pred["count"] for pred in self.validation_preds], dim=0)

            # Crop to central 2048bp
            input_len = all_targets.shape[-1]
            crop_len = 2048
            start = (input_len - crop_len) // 2
            end = start + crop_len
            all_preds_cropped = all_preds[:, :, start:end]
            all_targets_cropped = all_targets[:, :, start:end]

            # Use shared plotting logic with Phase 2 prefix
            self._shared_validation_plotting(all_preds_cropped, all_targets_cropped, all_count_preds, prefix="val_phase2")

            # Clear stored validation data
            self.validation_preds.clear()
            self.validation_targets.clear()

    def _shared_validation_plotting(self, all_preds_cropped: torch.Tensor, all_targets_cropped: torch.Tensor, all_count_preds: torch.Tensor = None, prefix: str = "val") -> None:
        """Shared plotting logic for Phase 2 main model validation."""
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        import numpy as np

        num_samples = all_targets_cropped.shape[1]
        total_val_batch_size = all_targets_cropped.shape[0]

        def compute_correlation_vectorized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
            # Region-level metrics using count predictions if available, otherwise sum of regression
            if all_count_preds is not None:
                # Use count head output (already in log10 scale)
                region_pred_log_count = all_count_preds[:, sample_idx]  # (batch,) - from count head
                region_target_sum = all_targets_cropped[:, sample_idx, :].sum(dim=1)  # (batch,) - sum across positions
                region_target_log_count = torch.log10(region_target_sum + 1)  # Convert targets to log10 scale
            else:
                # Fallback to regression sum with log10(x/1e3+1) transformation
                region_pred_sum = all_preds_cropped[:, sample_idx, :].sum(dim=1)  # (batch,) - sum across positions
                region_target_sum = all_targets_cropped[:, sample_idx, :].sum(dim=1)  # (batch,) - sum across positions
                region_pred_log_count = torch.log10(region_pred_sum / 1e3 + 1)
                region_target_log_count = torch.log10(region_target_sum / 1e3 + 1)

            region_mse = F.mse_loss(region_pred_log_count, region_target_log_count)

            region_corr = compute_correlation_vectorized(
                region_pred_log_count.unsqueeze(0),
                region_target_log_count.unsqueeze(0),
            )[0]

            # Profile-level JSD metrics (comparing normalized probability distributions)
            sample_jsd_values = compute_jsd_vectorized(
                all_preds_cropped[:, sample_idx:sample_idx+1, :],  # (batch, 1, seq_len)
                all_targets_cropped[:, sample_idx:sample_idx+1, :]  # (batch, 1, seq_len)
            )
            region_jsd = sample_jsd_values.mean()  # Average JSD across all regions

            # Log metrics for each sample
            self.log(
                f"{prefix}_region_mse_sample_{sample_idx}",
                region_mse,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )
            self.log(
                f"{prefix}_region_correlation_sample_{sample_idx}",
                region_corr,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )
            self.log(
                f"{prefix}_profile_jsd_sample_{sample_idx}",
                region_jsd,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )

        # Calculate average correlation across all samples
        avg_region_corr = torch.stack(
            [
                self.trainer.logged_metrics.get(
                    f"{prefix}_region_correlation_sample_{i}", torch.tensor(0.0)
                ).to(all_preds_cropped.device)
                for i in range(num_samples)
            ]
        ).mean()

        # Calculate average JSD across all samples
        avg_profile_jsd = torch.stack(
            [
                self.trainer.logged_metrics.get(
                    f"{prefix}_profile_jsd_sample_{i}", torch.tensor(0.0)
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
        self.log(
            f"{prefix}_profile_jsd_avg",
            avg_profile_jsd,
            on_epoch=True,
            batch_size=total_val_batch_size,
        )

        # Calculate FC correlations for both 2 and 4 sample cases
        if num_samples >= 2:
            # Use log10(x/1e3+1) transformation for FC calculation
            pred_region_sum_all = all_preds_cropped.sum(dim=2)  # (batch, num_samples) - sum across positions
            target_region_sum_all = all_targets_cropped.sum(dim=2)  # (batch, num_samples) - sum across positions
            pred_region_log_count = torch.log10(pred_region_sum_all / 1e3 + 1)  # (batch, num_samples) - log10 count scale
            target_region_log_count = torch.log10(target_region_sum_all / 1e3 + 1)  # (batch, num_samples) - log10 count scale

            if num_samples == 2:
                # Calculate FC for SCCA condition using log10 count transformation
                pred_fc_scca = pred_region_log_count[:, 0] - pred_region_log_count[:, 1]  # SCCApos - SCCAneg
                target_fc_scca = target_region_log_count[:, 0] - target_region_log_count[:, 1]  # SCCApos - SCCAneg

                # Calculate correlation for FC using vectorized function
                fc_corr_scca = compute_correlation_vectorized(
                    pred_fc_scca.unsqueeze(0), target_fc_scca.unsqueeze(0)
                )[0]

                # Log FC metrics
                self.log(
                    f"{prefix}_fc_correlation_scca",
                    fc_corr_scca,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )

            elif num_samples == 4:
                # Calculate FC using log10 count transformation
                pred_fc_scc1 = pred_region_log_count[:, 0] - pred_region_log_count[:, 1]  # SCC1pos - SCC1neg
                pred_fc_scca = pred_region_log_count[:, 2] - pred_region_log_count[:, 3]  # SCCApos - SCCAneg

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
                    f"{prefix}_fc_correlation_scc1",
                    fc_corr_scc1,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )
                self.log(
                    f"{prefix}_fc_correlation_scca",
                    fc_corr_scca,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )
                self.log(
                    f"{prefix}_fc_correlation_avg",
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
                region_target_sum = all_targets_cropped[:, sample_idx, :].sum(dim=1)
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
                    self.celltype_ids[sample_idx]
                    if self.celltype_ids
                    else f"Sample_{sample_idx}"
                )
                self.logger.experiment.log(
                    {
                        f"{prefix}_region_scatter_{celltype_name}": wandb.plot.scatter(
                            table,
                            "Observed Log10 Count",
                            "Predicted Log10 Count",
                            title=f"Region-Level Log10 Count ATAC Signal - {celltype_name} (Phase 2 Main Model)",
                        )
                    }
                )

            # High signal example visualization
            # Find the sample with highest signal across all samples
            max_signals_per_sample = all_targets_cropped.max(dim=2)[0]  # (batch, num_samples)
            max_signal_overall = max_signals_per_sample.max(dim=1)[0]  # (batch,)

            if max_signal_overall.max() > 0.5:  # Only create if there's meaningful signal
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
                    self.celltype_ids[highest_signal_sample_idx]
                    if self.celltype_ids
                    else f"Sample_{highest_signal_sample_idx}"
                )

                # Main ATAC signal plot
                fig, ax = plt.subplots(figsize=(12, 6))
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
                    f"High Signal Example - {celltype_name} (Max Signal: {max_signal_overall[highest_signal_batch_idx]:.2f}) - Phase 2 Main Model"
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
                    {f"{prefix}_high_signal_example": wandb.Image(img)}
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
                    self.celltype_ids[highest_sample_idx]
                    if self.celltype_ids
                    else f"Sample_{highest_sample_idx}"
                )

                # Create line plot similar to high signal example
                fig, ax = plt.subplots(figsize=(12, 6))
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
                    f"Signal Range {low}-{high} Example - {celltype_name} (Max Signal: {max_val:.2f}) - Phase 2 Main Model"
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

        # Add specific genomic region visualizations for Phase 2
        if self.logger and prefix == "val_phase2":  # Only during validation
            self._plot_specific_genomic_regions()

    def _plot_specific_genomic_regions(self) -> None:
        """Plot specific genomic regions of interest to wandb."""
        # Skip if no trainer access
        if not hasattr(self.trainer, 'datamodule'):
            return

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
            celltype_ids = datamodule.celltype_ids if hasattr(datamodule, 'celltype_ids') else self.celltype_ids if self.celltype_ids else ['sample_0']

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
                ax.set_title(f'{celltype} - {chr_name}:{start:,}-{end:,} (Phase 2 Main Model)')
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
                {f"val_phase2_specific_region_{region_name}_{chr_name}_{start}_{end}": wandb.Image(img)}
            )

        except Exception as e:
            print(f"Error plotting region {region}: {e}")

    def _create_placeholder_plot(self) -> None:
        """Create placeholder plot when data access fails."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Specific Region Visualization (Phase 2)\n\nRegions configured:\n"
                          f"chr16:45408722-45409426\nchr16:45383763-45384762\n"
                          f"chr16:45379545-45379801\nchr16:45356164-45357191\n\n"
                          f"Data access not available in current context",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Specific Genomic Region Plots - Configuration Ready (Phase 2)")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        self.logger.experiment.log(
            {"val_phase2_specific_regions_status": wandb.Image(img)}
        )

    def configure_optimizers(self):
        """Configure optimizer for Phase 2."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Phase3LightningModule(pl.LightningModule):
    """
    Lightning module for Phase 3 differential prediction training.

    Handles training/validation of main model + differential head.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        differential_loss_weight: float = 1.0,
        celltype_ids: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize Phase 3 Lightning module.

        Args:
            model: Phase3Model instance
            learning_rate: Learning rate for optimization
            differential_loss_weight: Weight for differential loss
            celltype_ids: List of celltype identifiers for logging
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.differential_loss_weight = differential_loss_weight
        self.celltype_ids = celltype_ids
        self.save_hyperparameters(ignore=['model'])

        # Storage for validation epoch aggregation
        self.validation_preds: List[Dict[str, torch.Tensor]] = []
        self.validation_targets: List[torch.Tensor] = []
        self.validation_diff_preds: List[torch.Tensor] = []
        self.validation_diff_targets: List[torch.Tensor] = []

        # Shape scale loss (with scale component enabled)
        self.shape_scale_loss = ShapeScaleLoss(
            shape_mode="ce", tau=1.0, scale_weight=0.2, eps=1e-6, reduction="sum"
        )

        # Poisson loss for count prediction
        self.poisson_loss = nn.PoissonNLLLoss(log_input=True, reduction='mean')

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through Phase 3 model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for Phase 3."""
        sequence = batch['sequence']
        preds = self.forward(sequence)
        losses = self._calculate_losses(batch, preds)

        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True)

        return losses['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for Phase 3."""
        sequence = batch['sequence']
        preds = self.forward(sequence)
        losses = self._calculate_losses(batch, preds)

        # Log losses
        batch_size = sequence.size(0)
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, on_epoch=True, batch_size=batch_size)

        # Store predictions and targets for epoch-end analysis
        if 'regression' in preds:
            pred_dict = {"regression": preds["regression"]}
            if 'count' in preds:
                pred_dict["count"] = preds["count"]
            self.validation_preds.append(pred_dict)
            self.validation_targets.append(batch["atac"])

        # Store differential predictions
        if 'differential' in preds and 'differential_targets' in batch:
            self.validation_diff_preds.append(preds['differential'].cpu())
            self.validation_diff_targets.append(batch['differential_targets'].cpu())
            # Debug logging for first batch
            if len(self.validation_diff_preds) == 1:
                print(f"DEBUG: Phase 3 differential predictions found!")
                print(f"  Differential preds shape: {preds['differential'].shape}")
                print(f"  Differential targets shape: {batch['differential_targets'].shape}")
                print(f"  Sample pred values: {preds['differential'][:3].cpu().numpy()}")
                print(f"  Sample target values: {batch['differential_targets'][:3].cpu().numpy()}")
        else:
            # Debug logging for missing differential data
            if len(self.validation_diff_preds) == 0:  # Only log once
                print(f"DEBUG: Phase 3 differential data missing!")
                print(f"  'differential' in preds: {'differential' in preds}")
                print(f"  'differential_targets' in batch: {'differential_targets' in batch}")
                print(f"  Available pred keys: {list(preds.keys())}")
                print(f"  Available batch keys: {list(batch.keys())}")

        return losses['loss']

    def _calculate_losses(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate Phase 3 losses (main model + differential)."""
        losses = {}

        # Main model losses (same as Phase 2)
        if 'regression' in preds:
            profile_loss = self.shape_scale_loss(preds['regression'], batch['atac'])
            losses['profile_loss'] = profile_loss

        if 'count' in preds:
            target_counts = batch['atac'].sum(dim=2)
            count_loss = self.poisson_loss(torch.log(preds['count'] + 1e-8), target_counts)
            losses['count_loss'] = count_loss

        # Differential loss
        if 'differential' in preds and 'differential_targets' in batch:
            diff_mse_loss = F.mse_loss(preds['differential'], batch['differential_targets'])
            losses['differential_mse_loss'] = diff_mse_loss

        # Total loss
        main_loss = losses.get('profile_loss', 0) + losses.get('count_loss', 0)
        diff_loss = self.differential_loss_weight * losses.get('differential_mse_loss', 0)
        total_loss = main_loss + diff_loss
        losses['loss'] = total_loss

        return losses

    def on_validation_epoch_end(self) -> None:
        """End of validation epoch with Phase 3 plotting."""
        # Standard ATAC validation plotting
        if self.validation_preds and self.validation_targets:
            # Aggregate and plot similar to Phase 2
            all_preds = torch.cat([pred["regression"] for pred in self.validation_preds], dim=0)
            all_targets = torch.cat(self.validation_targets, dim=0)

            # Aggregate count predictions if available
            all_count_preds = None
            if self.validation_preds[0].get("count") is not None:
                all_count_preds = torch.cat([pred["count"] for pred in self.validation_preds], dim=0)

            # Crop to central 2048bp
            input_len = all_targets.shape[-1]
            crop_len = 2048
            start = (input_len - crop_len) // 2
            end = start + crop_len
            all_preds_cropped = all_preds[:, :, start:end]
            all_targets_cropped = all_targets[:, :, start:end]

            # Use shared plotting logic with Phase 3 prefix
            self._shared_validation_plotting(all_preds_cropped, all_targets_cropped, all_count_preds, prefix="val_phase3")

            # Clear stored validation data
            self.validation_preds.clear()
            self.validation_targets.clear()

        # Differential value analysis
        if self.validation_diff_preds:
            self._log_differential_metrics()

    def _log_differential_metrics(self) -> None:
        """Log comprehensive differential value metrics and visualizations."""
        all_diff_preds = torch.cat(self.validation_diff_preds, dim=0)
        all_diff_targets = torch.cat(self.validation_diff_targets, dim=0)

        # Flatten for analysis
        diff_preds_flat = all_diff_preds.flatten()
        diff_targets_flat = all_diff_targets.flatten()

        # Calculate comprehensive metrics
        diff_mse = F.mse_loss(diff_preds_flat, diff_targets_flat)
        diff_mae = F.l1_loss(diff_preds_flat, diff_targets_flat)
        diff_rmse = torch.sqrt(diff_mse)

        # Calculate R² (coefficient of determination)
        diff_r2 = self._compute_r_squared(diff_preds_flat, diff_targets_flat)

        # Calculate Pearson correlation
        diff_corr = self._compute_correlation(diff_preds_flat, diff_targets_flat)

        # Calculate Spearman correlation (rank-based)
        diff_spearman = self._compute_spearman_correlation(diff_preds_flat, diff_targets_flat)

        # Log comprehensive scalar metrics
        self.log("val_differential_mse", diff_mse, on_epoch=True)
        self.log("val_differential_mae", diff_mae, on_epoch=True)
        self.log("val_differential_rmse", diff_rmse, on_epoch=True)
        self.log("val_differential_r2", diff_r2, on_epoch=True)
        self.log("val_differential_correlation", diff_corr, on_epoch=True)
        self.log("val_differential_spearman", diff_spearman, on_epoch=True)

        # Create comprehensive visualizations if logger is available
        if self.logger:
            self._create_differential_visualizations(diff_preds_flat, diff_targets_flat,
                                                   diff_corr, diff_r2, diff_mse, diff_mae)

        # Clear for next epoch
        self.validation_diff_preds.clear()
        self.validation_diff_targets.clear()

    def _compute_r_squared(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute R² (coefficient of determination)."""
        # Handle case where all targets are the same (would cause division by zero)
        targets_var = torch.var(targets)
        if targets_var < 1e-8:
            return torch.tensor(0.0, device=predictions.device)

        # Calculate R² = 1 - (SS_res / SS_tot)
        ss_res = torch.sum((targets - predictions) ** 2)  # Sum of squares of residuals
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)  # Total sum of squares

        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return torch.clamp(r2, -10.0, 1.0)  # Clamp to reasonable range

    def _compute_spearman_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Spearman rank correlation coefficient."""
        try:
            # Convert to ranks
            x_ranks = self._compute_ranks(x)
            y_ranks = self._compute_ranks(y)

            # Calculate Pearson correlation of ranks
            return self._compute_correlation(x_ranks, y_ranks)
        except Exception:
            # Fallback to 0 if ranking fails
            return torch.tensor(0.0, device=x.device)

    def _compute_ranks(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ranks of tensor values."""
        # Sort values and get indices
        _, sorted_indices = torch.sort(x)

        # Create ranks tensor
        ranks = torch.zeros_like(x)
        ranks[sorted_indices] = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device)

        return ranks

    def _create_differential_visualizations(self, diff_preds: torch.Tensor, diff_targets: torch.Tensor,
                                          corr: torch.Tensor, r2: torch.Tensor, mse: torch.Tensor, mae: torch.Tensor) -> None:
        """Create comprehensive visualizations for differential predictions."""
        import matplotlib.pyplot as plt
        import numpy as np

        # Convert to numpy for plotting
        pred_np = diff_preds.cpu().numpy()
        target_np = diff_targets.cpu().numpy()

        # 1. Main scatter plot with regression line
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Scatter plot with regression line
        ax1.scatter(target_np, pred_np, alpha=0.6, s=20, color='blue')

        # Add diagonal reference line (perfect prediction)
        min_val = min(target_np.min(), pred_np.min())
        max_val = max(target_np.max(), pred_np.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')

        # Add best fit line
        try:
            coeffs = np.polyfit(target_np, pred_np, 1)
            fit_line = np.poly1d(coeffs)
            ax1.plot(target_np, fit_line(target_np), 'g-', alpha=0.8, linewidth=2, label=f'Best Fit (slope={coeffs[0]:.3f})')
        except:
            pass

        ax1.set_xlabel('Observed Diff_scca')
        ax1.set_ylabel('Predicted Diff_scca')
        ax1.set_title(f'Differential Prediction Scatter Plot\nPearson r={corr:.3f}, R²={r2:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Residual plot
        residuals = pred_np - target_np
        ax2.scatter(target_np, residuals, alpha=0.6, s=20, color='purple')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Observed Diff_scca')
        ax2.set_ylabel('Residuals (Predicted - Observed)')
        ax2.set_title(f'Residual Plot\nMSE={mse:.4f}, MAE={mae:.4f}')
        ax2.grid(True, alpha=0.3)

        # 3. Distribution comparison
        ax3.hist(target_np, bins=50, alpha=0.7, label='Observed', color='blue', density=True)
        ax3.hist(pred_np, bins=50, alpha=0.7, label='Predicted', color='red', density=True)
        ax3.set_xlabel('Diff_scca Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Q-Q plot (quantile-quantile)
        target_sorted = np.sort(target_np)
        pred_sorted = np.sort(pred_np)

        ax4.scatter(target_sorted, pred_sorted, alpha=0.6, s=20, color='green')
        ax4.plot([target_sorted.min(), target_sorted.max()],
                [target_sorted.min(), target_sorted.max()], 'r--', alpha=0.8, linewidth=2)
        ax4.set_xlabel('Observed Quantiles')
        ax4.set_ylabel('Predicted Quantiles')
        ax4.set_title('Q-Q Plot')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to wandb image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        self.logger.experiment.log({
            "val_phase3_differential_analysis": wandb.Image(img)
        })

        # 2. Create wandb scatter plot table for interactive exploration
        table_data = [[target_val, pred_val, abs(target_val - pred_val)]
                     for target_val, pred_val in zip(target_np[:1000], pred_np[:1000])]  # Limit to 1000 points for performance

        table = wandb.Table(
            data=table_data,
            columns=["Observed_Diff_scca", "Predicted_Diff_scca", "Absolute_Error"]
        )

        self.logger.experiment.log({
            "val_phase3_differential_scatter": wandb.plot.scatter(
                table,
                "Observed_Diff_scca",
                "Predicted_Diff_scca",
                title=f"Differential Prediction Scatter (r={corr:.3f}, R²={r2:.3f})"
            )
        })

        # 3. Performance metrics summary plot
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_names = ['Pearson r', 'Spearman ρ', 'R²', 'RMSE', 'MAE']
        metrics_values = [
            corr.cpu().item(),
            self._compute_spearman_correlation(diff_preds, diff_targets).cpu().item(),
            r2.cpu().item(),
            torch.sqrt(mse).cpu().item(),
            mae.cpu().item()
        ]

        colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral', 'plum']
        bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Metric Value')
        ax.set_title('Differential Prediction Performance Metrics Summary')
        ax.grid(True, alpha=0.3, axis='y')

        # Add horizontal reference lines for correlation metrics
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good correlation (0.7)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate correlation (0.5)')
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Convert to wandb image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        self.logger.experiment.log({
            "val_phase3_differential_metrics_summary": wandb.Image(img)
        })

    def _compute_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute correlation between two tensors."""
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
        return numerator / (denominator + 1e-8)

    def _shared_validation_plotting(self, all_preds_cropped: torch.Tensor, all_targets_cropped: torch.Tensor, all_count_preds: torch.Tensor = None, prefix: str = "val") -> None:
        """Shared plotting logic for Phase 3 validation."""
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        import numpy as np

        num_samples = all_targets_cropped.shape[1]
        total_val_batch_size = all_targets_cropped.shape[0]

        def compute_correlation_vectorized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
            # Region-level metrics using count predictions if available, otherwise sum of regression
            if all_count_preds is not None:
                # Use count head output (already in log10 scale)
                region_pred_log_count = all_count_preds[:, sample_idx]  # (batch,) - from count head
                region_target_sum = all_targets_cropped[:, sample_idx, :].sum(dim=1)  # (batch,) - sum across positions
                region_target_log_count = torch.log10(region_target_sum + 1)  # Convert targets to log10 scale
            else:
                # Fallback to regression sum with log10(x/1e3+1) transformation
                region_pred_sum = all_preds_cropped[:, sample_idx, :].sum(dim=1)  # (batch,) - sum across positions
                region_target_sum = all_targets_cropped[:, sample_idx, :].sum(dim=1)  # (batch,) - sum across positions
                region_pred_log_count = torch.log10(region_pred_sum / 1e3 + 1)
                region_target_log_count = torch.log10(region_target_sum / 1e3 + 1)

            region_mse = F.mse_loss(region_pred_log_count, region_target_log_count)

            region_corr = compute_correlation_vectorized(
                region_pred_log_count.unsqueeze(0),
                region_target_log_count.unsqueeze(0),
            )[0]

            # Profile-level JSD metrics (comparing normalized probability distributions)
            sample_jsd_values = compute_jsd_vectorized(
                all_preds_cropped[:, sample_idx:sample_idx+1, :],  # (batch, 1, seq_len)
                all_targets_cropped[:, sample_idx:sample_idx+1, :]  # (batch, 1, seq_len)
            )
            region_jsd = sample_jsd_values.mean()  # Average JSD across all regions

            # Log metrics for each sample
            self.log(
                f"{prefix}_region_mse_sample_{sample_idx}",
                region_mse,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )
            self.log(
                f"{prefix}_region_correlation_sample_{sample_idx}",
                region_corr,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )
            self.log(
                f"{prefix}_profile_jsd_sample_{sample_idx}",
                region_jsd,
                on_epoch=True,
                batch_size=total_val_batch_size,
            )

        # Calculate average correlation across all samples
        avg_region_corr = torch.stack(
            [
                self.trainer.logged_metrics.get(
                    f"{prefix}_region_correlation_sample_{i}", torch.tensor(0.0)
                ).to(all_preds_cropped.device)
                for i in range(num_samples)
            ]
        ).mean()

        # Calculate average JSD across all samples
        avg_profile_jsd = torch.stack(
            [
                self.trainer.logged_metrics.get(
                    f"{prefix}_profile_jsd_sample_{i}", torch.tensor(0.0)
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
        self.log(
            f"{prefix}_profile_jsd_avg",
            avg_profile_jsd,
            on_epoch=True,
            batch_size=total_val_batch_size,
        )

        # Calculate FC correlations for both 2 and 4 sample cases
        if num_samples >= 2:
            # Use log10(x/1e3+1) transformation for FC calculation
            pred_region_sum_all = all_preds_cropped.sum(dim=2)  # (batch, num_samples) - sum across positions
            target_region_sum_all = all_targets_cropped.sum(dim=2)  # (batch, num_samples) - sum across positions
            pred_region_log_count = torch.log10(pred_region_sum_all / 1e3 + 1)  # (batch, num_samples) - log10 count scale
            target_region_log_count = torch.log10(target_region_sum_all / 1e3 + 1)  # (batch, num_samples) - log10 count scale

            if num_samples == 2:
                # Calculate FC for SCCA condition using log10 count transformation
                pred_fc_scca = pred_region_log_count[:, 0] - pred_region_log_count[:, 1]  # SCCApos - SCCAneg
                target_fc_scca = target_region_log_count[:, 0] - target_region_log_count[:, 1]  # SCCApos - SCCAneg

                # Calculate correlation for FC using vectorized function
                fc_corr_scca = compute_correlation_vectorized(
                    pred_fc_scca.unsqueeze(0), target_fc_scca.unsqueeze(0)
                )[0]

                # Log FC metrics
                self.log(
                    f"{prefix}_fc_correlation_scca",
                    fc_corr_scca,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )

            elif num_samples == 4:
                # Calculate FC using log10 count transformation
                pred_fc_scc1 = pred_region_log_count[:, 0] - pred_region_log_count[:, 1]  # SCC1pos - SCC1neg
                pred_fc_scca = pred_region_log_count[:, 2] - pred_region_log_count[:, 3]  # SCCApos - SCCAneg

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
                    f"{prefix}_fc_correlation_scc1",
                    fc_corr_scc1,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )
                self.log(
                    f"{prefix}_fc_correlation_scca",
                    fc_corr_scca,
                    on_epoch=True,
                    batch_size=total_val_batch_size,
                )
                self.log(
                    f"{prefix}_fc_correlation_avg",
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
                region_target_sum = all_targets_cropped[:, sample_idx, :].sum(dim=1)
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
                    self.celltype_ids[sample_idx]
                    if self.celltype_ids
                    else f"Sample_{sample_idx}"
                )
                self.logger.experiment.log(
                    {
                        f"{prefix}_region_scatter_{celltype_name}": wandb.plot.scatter(
                            table,
                            "Observed Log10 Count",
                            "Predicted Log10 Count",
                            title=f"Region-Level Log10 Count ATAC Signal - {celltype_name} (Phase 3 Differential Model)",
                        )
                    }
                )

            # High signal example visualization
            # Find the sample with highest signal across all samples
            max_signals_per_sample = all_targets_cropped.max(dim=2)[0]  # (batch, num_samples)
            max_signal_overall = max_signals_per_sample.max(dim=1)[0]  # (batch,)

            if max_signal_overall.max() > 0.5:  # Only create if there's meaningful signal
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
                    self.celltype_ids[highest_signal_sample_idx]
                    if self.celltype_ids
                    else f"Sample_{highest_signal_sample_idx}"
                )

                # Main ATAC signal plot
                fig, ax = plt.subplots(figsize=(12, 6))
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
                    f"High Signal Example - {celltype_name} (Max Signal: {max_signal_overall[highest_signal_batch_idx]:.2f}) - Phase 3 Differential Model"
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
                    {f"{prefix}_high_signal_example": wandb.Image(img)}
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
                    self.celltype_ids[highest_sample_idx]
                    if self.celltype_ids
                    else f"Sample_{highest_sample_idx}"
                )

                # Create line plot similar to high signal example
                fig, ax = plt.subplots(figsize=(12, 6))
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
                    f"Signal Range {low}-{high} Example - {celltype_name} (Max Signal: {max_val:.2f}) - Phase 3 Differential Model"
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

        # Add specific genomic region visualizations for Phase 3
        if self.logger and prefix == "val_phase3":  # Only during validation
            self._plot_specific_genomic_regions()

    def _plot_specific_genomic_regions(self) -> None:
        """Plot specific genomic regions of interest to wandb."""
        # Skip if no trainer access
        if not hasattr(self.trainer, 'datamodule'):
            return

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
            celltype_ids = datamodule.celltype_ids if hasattr(datamodule, 'celltype_ids') else self.celltype_ids if self.celltype_ids else ['sample_0']

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
                ax.set_title(f'{celltype} - {chr_name}:{start:,}-{end:,} (Phase 3 Differential Model)')
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
                {f"val_phase3_specific_region_{region_name}_{chr_name}_{start}_{end}": wandb.Image(img)}
            )

        except Exception as e:
            print(f"Error plotting region {region}: {e}")

    def _create_placeholder_plot(self) -> None:
        """Create placeholder plot when data access fails."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Specific Region Visualization (Phase 3)\n\nRegions configured:\n"
                          f"chr16:45408722-45409426\nchr16:45383763-45384762\n"
                          f"chr16:45379545-45379801\nchr16:45356164-45357191\n\n"
                          f"Data access not available in current context",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Specific Genomic Region Plots - Configuration Ready (Phase 3)")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        self.logger.experiment.log(
            {"val_phase3_specific_regions_status": wandb.Image(img)}
        )

    def configure_optimizers(self):
        """Configure optimizer for Phase 3."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)