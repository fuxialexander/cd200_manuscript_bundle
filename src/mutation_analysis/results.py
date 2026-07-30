"""
Result classes for mutation analysis.

Each result type knows how to plot itself, following the principle
of clear ownership. Type-specific plots are implemented in subclasses,
while shared plots are in the base class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .dataset import Variant

if TYPE_CHECKING:
    from .analyzer import MutationAnalyzer


@dataclass
class BaseResult(ABC):
    """
    Abstract base class for all analysis results.

    Provides common interface and shared plotting methods.
    Subclasses implement their own plot_summary().
    """
    variant: Variant
    output_heads: List[str]

    @abstractmethod
    def plot_summary(self, figsize: Optional[Tuple[int, int]] = None,
                    save_path: Optional[str] = None, **kwargs) -> Figure:
        """
        Create comprehensive summary plot.

        Each result type implements its own layout and panels.
        """
        pass

    def plot_motif_activation(self, ref_motifs: pd.DataFrame,
                             alt_motifs: pd.DataFrame,
                             top_n: int = 20,
                             p_threshold: float = 1e-4,
                             figsize: Tuple[int, int] = (12, 6),
                             save_path: Optional[str] = None) -> Figure:
        """
        Plot motif activation changes (shared by all result types).

        Shows top motifs with significant changes (p < threshold).

        Args:
            ref_motifs: Reference motif matches (from FIMO)
            alt_motifs: Alternative motif matches
            top_n: Number of top motifs to show
            p_threshold: P-value threshold for significance
            figsize: Figure size
            save_path: Optional path to save plot

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Placeholder implementation
        ax.text(0.5, 0.5, 'Motif activation plot\n(Implementation pending)',
               ha='center', va='center', fontsize=14)
        ax.set_title('Motif Activation Changes')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


@dataclass
class VariantResult(BaseResult):
    """
    Single variant analysis result.

    Contains ref/alt predictions for all output heads and motif matches.
    Provides comprehensive visualization methods.
    """
    ref_outputs: Dict[str, torch.Tensor]
    alt_outputs: Dict[str, torch.Tensor]
    effects: Dict[str, Dict[str, float]]
    ref_motifs: pd.DataFrame
    alt_motifs: pd.DataFrame
    analyzer: Optional['MutationAnalyzer'] = field(default=None, repr=False)

    def scan_motifs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scan reference and alternate sequences for motifs.

        Requires the analyzer to be configured with a FimoMotifScanner.
        The results are stored in self.ref_motifs and self.alt_motifs.

        Returns:
            A tuple of (ref_motifs_df, alt_motifs_df).
        """
        if not self.analyzer or not self.analyzer.motif_scanner:
            raise RuntimeError(
                "Motif scanning requires an analyzer with a configured FimoMotifScanner."
            )
        
        # This assumes the sequences were generated with the same extend_bp
        # and are available in the analyzer's dataset.
        extend_bp = self.ref_outputs[self.output_heads[0]].shape[-1]
        
        ref_seq, alt_seq = self.analyzer.dataset.get_variant_sequences(self.variant, extend_bp)

        ref_hits = self.analyzer.motif_scanner.scan(ref_seq)
        alt_hits = self.analyzer.motif_scanner.scan(alt_seq)

        # Apply filter if analyzer has one
        if self.analyzer.motif_filter:
            allowed = self.analyzer.motif_filter.filter_motifs(self.analyzer.motif_scanner.motif_names)
            ref_hits = ref_hits[ref_hits['motif_name'].isin(allowed)]
            alt_hits = alt_hits[alt_hits['motif_name'].isin(allowed)]

        self.ref_motifs = ref_hits
        self.alt_motifs = alt_hits
        
        print(f"Found {len(ref_hits)} REF motifs and {len(alt_hits)} ALT motifs.")
        return self.ref_motifs, self.alt_motifs

    def plot_summary(self, figsize: Optional[Tuple[int, int]] = None,
                    save_path: Optional[str] = None, **kwargs) -> Figure:
        """
        Comprehensive multi-panel plot for variant analysis.

        Layout:
        - One row per output head (showing ref vs alt)
        - One row for motif activation changes

        Example:
            >>> result.plot_summary(figsize=(15, 12))
        """
        n_heads = len(self.output_heads)
        figsize = figsize or (15, 4 * n_heads)

        fig, axes = plt.subplots(n_heads, 1, figsize=figsize)

        # Handle single head case
        if n_heads == 1:
            axes = [axes]

        # Plot each output head
        for i, head in enumerate(self.output_heads):
            self._plot_head_panel(axes[i], head)

        # Add main title
        fig.suptitle(f'Variant Analysis: {self.variant}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_head_output(self, head_name: str,
                        figsize: Tuple[int, int] = (12, 4),
                        save_path: Optional[str] = None) -> Figure:
        """
        Plot specific output head in detail.

        Automatically determines plot type based on tensor dimensionality:
        - 1D tensor (length > 1): Line plot (e.g., regression profile)
        - 0D or scalar: Bar plot (e.g., count, fc)

        Args:
            head_name: Name of output head to plot
            figsize: Figure size
            save_path: Optional path to save

        Returns:
            matplotlib Figure

        Example:
            >>> result.plot_head_output('regression')
            >>> result.plot_head_output('count')
        """
        if head_name not in self.output_heads:
            raise ValueError(f"Unknown head '{head_name}'. Available: {self.output_heads}")

        ref_out = self.ref_outputs[head_name]
        alt_out = self.alt_outputs[head_name]

        # Determine plot type by dimensionality
        if ref_out.ndim >= 1 and ref_out.numel() > 1:
            # 1D profile
            return self._plot_profile(ref_out, alt_out, head_name, figsize, save_path)
        else:
            # Scalar value
            return self._plot_scalar(ref_out, alt_out, head_name, figsize, save_path)

    def _plot_profile(self, ref: torch.Tensor, alt: torch.Tensor,
                     head_name: str, figsize: Tuple[int, int],
                     save_path: Optional[str]) -> Figure:
        """Line plot for 1D profiles (e.g., regression head)."""
        fig, ax = plt.subplots(figsize=figsize)

        ref_data = ref.squeeze().cpu().numpy()
        alt_data = alt.squeeze().cpu().numpy()
        positions = np.arange(len(ref_data))

        # Plot profiles
        ax.plot(positions, ref_data, label='REF', color='blue', linewidth=2, alpha=0.8)
        ax.plot(positions, alt_data, label='ALT', color='red', linewidth=2, alpha=0.8)

        # Mark variant position (center)
        center = len(positions) // 2
        ax.axvline(center, color='black', linestyle='--', alpha=0.5,
                  label='Variant position')

        # Formatting
        ax.set_xlabel('Position (bp)', fontsize=11)
        ax.set_ylabel(f'{head_name} Output', fontsize=11)
        ax.set_title(f'{head_name.capitalize()} Head: REF vs ALT', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add effect metrics as text
        effect_text = '\n'.join(f"{k}: {v:.4f}"
                               for k, v in self.effects[head_name].items())
        ax.text(0.02, 0.98, effect_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_scalar(self, ref: torch.Tensor, alt: torch.Tensor,
                    head_name: str, figsize: Tuple[int, int],
                    save_path: Optional[str]) -> Figure:
        """Bar plot for scalar outputs (e.g., count, fc heads)."""
        fig, ax = plt.subplots(figsize=figsize)

        ref_val = ref.item() if ref.numel() == 1 else ref.sum().item()
        alt_val = alt.item() if alt.numel() == 1 else alt.sum().item()

        # Create bar plot
        bars = ax.bar(['REF', 'ALT'], [ref_val, alt_val],
                     color=['blue', 'red'], alpha=0.7, width=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, [ref_val, alt_val]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11)

        # Formatting
        ax.set_ylabel(f'{head_name} Output', fontsize=11)
        ax.set_title(f'{head_name.capitalize()} Head: REF vs ALT', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add effect metrics
        effect_text = '\n'.join(f"{k}: {v:.4f}"
                               for k, v in self.effects[head_name].items())
        ax.text(0.02, 0.98, effect_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_head_panel(self, ax: plt.Axes, head_name: str):
        """Plot single head panel for summary plot."""
        ref_out = self.ref_outputs[head_name]
        alt_out = self.alt_outputs[head_name]

        if ref_out.ndim >= 1 and ref_out.numel() > 1:
            # Profile plot
            ref_data = ref_out.squeeze().cpu().numpy()
            alt_data = alt_out.squeeze().cpu().numpy()
            positions = np.arange(len(ref_data))

            ax.plot(positions, ref_data, label='REF', color='blue', linewidth=2, alpha=0.8)
            ax.plot(positions, alt_data, label='ALT', color='red', linewidth=2, alpha=0.8)

            center = len(positions) // 2
            ax.axvline(center, color='black', linestyle='--', alpha=0.5)
        else:
            # Scalar plot
            ref_val = ref_out.item() if ref_out.numel() == 1 else ref_out.sum().item()
            alt_val = alt_out.item() if alt_out.numel() == 1 else alt_out.sum().item()

            ax.bar(['REF', 'ALT'], [ref_val, alt_val],
                  color=['blue', 'red'], alpha=0.7)

        ax.set_title(f'{head_name.capitalize()} Head', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)


@dataclass
class ISMResult(BaseResult):
    """
    In-silico mutagenesis result.

    Contains L x 4 effect matrices for each output head.
    Provides logo plot visualization (ISM-specific).
    """
    effect_matrices: Dict[str, np.ndarray]
    contribution_scores: Dict[str, np.ndarray]
    ref_sequence: str
    window_start: int

    def plot_summary(self, figsize: Optional[Tuple[int, int]] = None,
                    save_path: Optional[str] = None, **kwargs) -> Figure:
        """
        Multi-panel logo plots for all output heads.

        Layout:
        - One logo plot per output head
        - Stacked vertically with shared x-axis

        Example:
            >>> ism_result.plot_summary()
        """
        n_heads = len(self.output_heads)
        figsize = figsize or (15, 3 * n_heads)

        fig, axes = plt.subplots(n_heads, 1, figsize=figsize, sharex=True)

        if n_heads == 1:
            axes = [axes]

        # Create logo for each head
        for i, head in enumerate(self.output_heads):
            self.plot_logo(head, ax=axes[i])

        # Add main title
        fig.suptitle(f'In Silico Mutagenesis: {self.variant}',
                    fontsize=16, fontweight='bold', y=0.995)

        # Set x-label on bottom plot
        axes[-1].set_xlabel('Position', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_logo(self, head_name: str, ax: Optional[plt.Axes] = None,
                 figsize: Tuple[int, int] = (15, 3),
                 save_path: Optional[str] = None,
                 show_sequence: bool = True) -> Figure:
        """
        ISM-specific logo plot for given metric.

        Creates sequence logo showing effect of each nucleotide at each position.

        Args:
            head_name: Output head to visualize
            ax: Optional axis to plot on
            figsize: Figure size (if creating new figure)
            save_path: Optional path to save
            show_sequence: Whether to show reference sequence

        Returns:
            matplotlib Figure

        Example:
            >>> ism_result.plot_logo('log_fold_change')
            >>> ism_result.plot_logo('count', show_sequence=True)
        """
        if head_name not in self.output_heads:
            raise ValueError(f"Unknown head '{head_name}'. Available: {self.output_heads}")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Get effect matrix
        effect_matrix = self.effect_matrices[head_name]

        # Create simple heatmap visualization (simplified logo)
        im = ax.imshow(effect_matrix.T, aspect='auto', cmap='RdBu_r',
                      interpolation='nearest')

        # Formatting
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['A', 'C', 'G', 'T'])
        ax.set_ylabel('Base', fontsize=11)
        ax.set_title(f'ISM: {head_name.capitalize()}', fontsize=12, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Effect')

        # Show sequence if requested
        if show_sequence:
            ax.text(0.5, -0.15, f'Reference: {self.ref_sequence[:min(50, len(self.ref_sequence))]}...',
                   transform=ax.transAxes, ha='center', fontsize=8,
                   fontfamily='monospace')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_contribution_scores(self, head_name: str,
                                figsize: Tuple[int, int] = (15, 3),
                                save_path: Optional[str] = None) -> Figure:
        """
        Plot contribution scores (reference base effects).

        Shows the effect of each reference base (from mean-centered ISM).

        Args:
            head_name: Output head to visualize
            figsize: Figure size
            save_path: Optional path to save

        Returns:
            matplotlib Figure

        Example:
            >>> ism_result.plot_contribution_scores('log_fold_change')
        """
        if head_name not in self.output_heads:
            raise ValueError(f"Unknown head '{head_name}'. Available: {self.output_heads}")

        fig, ax = plt.subplots(figsize=figsize)

        scores = self.contribution_scores[head_name]
        L = len(scores)
        positions = np.arange(self.window_start, self.window_start + L)

        # Plot contribution scores
        ax.plot(positions, scores, linewidth=2, color='darkblue')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)

        # Highlight variant position
        variant_idx = self.variant.position - self.window_start
        if 0 <= variant_idx < L:
            ax.axvline(self.variant.position, color='red',
                      linestyle='--', alpha=0.7, label='Variant')
            ax.legend()

        # Formatting
        ax.set_xlabel('Genomic Position', fontsize=12)
        ax.set_ylabel('Contribution Score', fontsize=12)
        ax.set_title(f'Contribution Scores: {head_name.capitalize()}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


@dataclass
class DeletionResult(BaseResult):
    """
    Sliding window deletion result.
    Contains deletion effects for each output head across window positions.
    Provides deletion effect visualization.
    Attributes:
        deletion_effects: Dict mapping head names to window-level deletion effects
        window_positions: Genomic positions of window centers
        deletion_size: Size of deletion window (bp)
        stride: Step size between windows
        ref_motifs: Optional motif annotations for reference
        alt_motifs: Optional motif annotations for alternative
        nucleotide_scores: Optional dict of per-nucleotide scores (if distribute_effects=True)
        region_start: Start position of analyzed region
        region_end: End position of analyzed region
        motif_activations: Optional dict with motif activation data from model
            - 'activations': Tensor (num_motifs, seq_len)
            - 'filtered_motif_names': List of motif names after filtering
            - 'high_score_regions': Dict of high-scoring regions per head
    analyzer: Optional['MutationAnalyzer'] = field(default=None, repr=False)
    baseline_predictions: Optional[Dict[str, np.ndarray]] = None
    observed_atac: Optional[np.ndarray] = None
    core_region: Optional[Tuple[int, int]] = None
    """
    deletion_effects: Dict[str, np.ndarray]
    window_positions: np.ndarray
    deletion_size: int
    stride: int
    ref_motifs: Optional[pd.DataFrame] = None
    alt_motifs: Optional[pd.DataFrame] = None
    nucleotide_scores: Optional[Dict[str, np.ndarray]] = None
    region_start: Optional[int] = None
    region_end: Optional[int] = None
    motif_activations: Optional[Dict[str, any]] = None
    analyzer: Optional['MutationAnalyzer'] = field(default=None, repr=False)
    baseline_predictions: Optional[Dict[str, np.ndarray]] = None
    observed_atac: Optional[np.ndarray] = None
    core_region: Optional[Tuple[int, int]] = None

    def scan_and_plot_motif_changes(self, head_name: str,
                                    sigma_threshold: float = 4.0,
                                    max_regions: int = 5,
                                    context_bp: int = 50,
                                    figsize: Tuple[int, int] = (15, 6),
                                    save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Identify high-impact regions, scan them for motifs, and plot the changes.

        Args:
            head_name: The output head to use for identifying high-impact regions.
            sigma_threshold: The number of standard deviations to define a high-impact region.
            max_regions: The maximum number of distinct regions to analyze.
            context_bp: The number of base pairs to include around each region.
            figsize: Figure size for the plot.
            save_path: Optional path to save the plot.

        Returns:
            A matplotlib Figure, or None if no high-impact regions are found.
        """
        if not self.analyzer or not self.analyzer.motif_scanner:
            raise RuntimeError(
                "Motif scanning requires an analyzer with a configured FimoMotifScanner."
            )
            
        if self.nucleotide_scores is None:
            print("Nucleotide scores are required for this analysis.")
            return None
            
        scores = self.nucleotide_scores[head_name]
        threshold = np.mean(scores) + sigma_threshold * np.std(scores)
        high_indices = np.where(scores > threshold)[0]
        
        if len(high_indices) == 0:
            print(f"No regions found above {sigma_threshold}-sigma threshold.")
            return None
            
        # Group consecutive indices into regions
        regions = []
        start = high_indices[0]
        for i in range(1, len(high_indices)):
            if high_indices[i] > high_indices[i-1] + 1:
                regions.append((start, high_indices[i-1]))
                start = high_indices[i]
        regions.append((start, high_indices[-1]))
        
        # Get top regions by max score
        region_scores = [np.max(scores[s:e+1]) for s, e in regions]
        top_regions_indices = np.argsort(region_scores)[-max_regions:]
        top_regions = [regions[i] for i in top_regions_indices]

        # Scan each region and store results
        results = []
        for start_idx, end_idx in top_regions:
            scan_start = self.region_start + start_idx - context_bp
            scan_end = self.region_start + end_idx + context_bp
            
            ref_seq, alt_seq = self.analyzer.dataset.get_variant_sequences(self.variant, (scan_end - scan_start))
            
            # This is a simplification; a full implementation would need to
            # carefully align the variant within these arbitrary regions.
            # For now, we scan the original ref sequence of the region.
            ref_motifs = self.analyzer.scan_motifs(self.variant.chromosome, scan_start, scan_end)
            
            results.append({
                "region": (scan_start, scan_end),
                "motifs": ref_motifs
            })
            
        # For this example, we just print the results. A full implementation
        # would create a comparative plot.
        print("Motif Scan Results for High-Impact Regions:")
        for res in results:
            print(f"\n--- Region {res['region'][0]}-{res['region'][1]} ---")
            if res['motifs'].empty:
                print("No significant motifs found.")
            else:
                print(res['motifs'][['motif_name', 'start', 'p_value']].to_string(index=False))
        
        # Placeholder for plotting logic
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Plotting implementation is pending.\nSee console for scan results.', 
                ha='center', va='center', fontsize=12)
        ax.set_title(f"Motif Changes in High-Impact Regions for '{head_name}'")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return fig

    def plot_summary(self, figsize: Optional[Tuple[int, int]] = None,
                    save_path: Optional[str] = None, **kwargs) -> Figure:
        """
        Multi-panel deletion effect plots.
        Layout:
        - One plot per output head showing deletion effects
        - Shared x-axis (genomic position)
        Example:
            >>> del_result.plot_summary()
        """
        n_heads = len(self.output_heads)
        has_extra_panel = (self.baseline_predictions and any(p is not None for p in self.baseline_predictions.values())) or \
                          (self.observed_atac is not None)
        
        num_panels = n_heads + 1 if has_extra_panel else n_heads
        
        figsize = figsize or (15, 4 * num_panels)

        fig, axes = plt.subplots(num_panels, 1, figsize=figsize, sharex=True)

        if num_panels == 1:
            axes = [axes]

        if has_extra_panel:
            self._plot_prediction_observation_panel(axes[0])
            plot_axes = axes[1:]
        else:
            plot_axes = axes

        for i, head in enumerate(self.output_heads):
            self.plot_deletion_effects(head, ax=plot_axes[i])

        fig.suptitle(f'Sliding Window Deletion: {self.variant} '
                    f'({self.deletion_size}bp window)',
                    fontsize=16, fontweight='bold', y=0.995)

        axes[-1].set_xlabel('Genomic Position', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_prediction_observation_panel(self, ax: plt.Axes):
        """Plot baseline prediction and observed ATAC in a panel."""
        if self.region_start is None or self.region_end is None:
            return

        positions = np.arange(self.region_start, self.region_end)
        
        if self.baseline_predictions:
            for head, pred in self.baseline_predictions.items():
                if 'regression' in head and pred is not None and pred.ndim == 1 and len(pred) == len(positions):
                    ax.plot(positions, pred, label=f'Predicted ({head})', color='green', linewidth=2, alpha=0.8)

        if self.observed_atac is not None and self.observed_atac.ndim == 1 and len(self.observed_atac) == len(positions):
            ax.plot(positions, self.observed_atac, label='Observed ATAC', color='purple', linewidth=2, alpha=0.8)

        if self.core_region is not None:
            core_start, core_end = self.core_region
            ax.axvspan(core_start, core_end, color='gray', alpha=0.2, label='Core Region')

        ax.set_ylabel('Signal', fontsize=11)
        ax.set_title('Baseline Prediction and Observation', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_deletion_effects(self, head_name: str,
                             ax: Optional[plt.Axes] = None,
                             figsize: Tuple[int, int] = (15, 4),
                             save_path: Optional[str] = None,
                             highlight_top: int = 5,
                             motif_cache: Optional[str] = None,
                             motif_filter: Optional['MotifFilter'] = None,
                             motif_pvalue_threshold: float = 1e-4,
                             use_nucleotide_scores: bool = False) -> Figure:
        """
        Plot deletion effects for specific output head.

        Shows effect of deleting each window on model output.

        Args:
            head_name: Output head to visualize
            ax: Optional axis to plot on
            figsize: Figure size (if creating new figure)
            save_path: Optional path to save
            highlight_top: Number of top effects to highlight
            motif_cache: Optional path to motif cache (zarr) for overlay
            motif_filter: Optional MotifFilter to select which motifs to show
            motif_pvalue_threshold: P-value threshold for motif filtering
            use_nucleotide_scores: If True and available, plot nucleotide-level scores
                                  instead of window-level effects

        Returns:
            matplotlib Figure

        Example:
            >>> del_result.plot_deletion_effects('regression')
            >>> del_result.plot_deletion_effects('count', highlight_top=10)
            >>> # With motif overlay
            >>> from mutation_analysis.motif_filters import create_annotation_filter
            >>> filt = create_annotation_filter('annot.csv', 'expressed')
            >>> del_result.plot_deletion_effects('regression',
            ...     motif_cache='motifs.zarr', motif_filter=filt)
        """
        if head_name not in self.output_heads:
            raise ValueError(f"Unknown head '{head_name}'. Available: {self.output_heads}")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Choose data to plot (nucleotide vs window level)
        if use_nucleotide_scores and self.nucleotide_scores is not None:
            effects = self.nucleotide_scores[head_name]
            positions = np.arange(self.region_start, self.region_start + len(effects))
            plot_label = 'Nucleotide-level'
        else:
            effects = self.deletion_effects[head_name]
            positions = self.window_positions
            plot_label = 'Window-level'

        # Plot deletion effects
        ax.plot(positions, effects, linewidth=2, marker='o' if len(positions) < 500 else '',
               markersize=3, color='darkred', alpha=0.7, label=plot_label)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)

        # Highlight top effects
        if highlight_top > 0:
            top_indices = np.argsort(np.abs(effects))[-highlight_top:]
            ax.scatter(positions[top_indices], effects[top_indices],
                      s=100, color='orange', edgecolors='black', linewidths=2,
                      zorder=10, label=f'Top {highlight_top}')

        # Mark variant position
        ax.axvline(self.variant.position, color='blue',
                  linestyle='--', alpha=0.7, label='Variant')

        # Add motif overlay if requested
        if motif_cache is not None:
            self._add_motif_overlay(ax, motif_cache, motif_filter, motif_pvalue_threshold,
                                   positions[0], positions[-1])

        # Formatting
        ax.set_ylabel(f'Deletion Effect ({head_name})', fontsize=11)
        title = f'Deletion Effects: {head_name.capitalize()} '
        if use_nucleotide_scores and self.nucleotide_scores is not None:
            title += f'(nucleotide-level, {self.deletion_size}bp window)'
        else:
            title += f'({self.deletion_size}bp window, {self.stride}bp stride)'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Genomic Position', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Add statistics
        stats_text = (f"Mean: {np.mean(effects):.3f}\n"
                     f"Std: {np.std(effects):.3f}\n"
                     f"Max |effect|: {np.max(np.abs(effects)):.3f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _add_motif_overlay(self, ax: plt.Axes, motif_cache_path: str,
                          motif_filter: Optional['MotifFilter'],
                          pvalue_threshold: float,
                          plot_start: int, plot_end: int,
                          max_motifs: int = 20):
        """
        Add motif overlay to deletion plot.

        Args:
            ax: Matplotlib axis
            motif_cache_path: Path to motif cache zarr
            motif_filter: Optional filter for motifs
            pvalue_threshold: P-value threshold for motifs
            plot_start: Start position of plot
            plot_end: End position of plot
            max_motifs: Maximum number of motifs to display
        """
        try:
            from .motif_cache import MotifCache
            from adjustText import adjust_text
        except ImportError:
            import warnings
            warnings.warn("adjustText not available, motif labels may overlap")

        # Load motif cache
        cache = MotifCache(motif_cache_path, mode='r')

        # Query region
        scan_result = cache.query_region(
            self.variant.chromosome,
            int(plot_start),
            int(plot_end),
            pvalue_threshold=pvalue_threshold
        )

        if len(scan_result.positions) == 0:
            return

        # Apply motif filter if provided
        if motif_filter is not None:
            scan_result = scan_result.filter_by_motifs(
                motif_filter.filter_motifs(scan_result.motif_names)
            )

        if len(scan_result.positions) == 0:
            return

        # Convert to DataFrame for easier manipulation
        motif_df = scan_result.to_dataframe()

        # Limit number of motifs if too many
        if len(motif_df) > max_motifs:
            # Keep top scoring motifs
            motif_df = motif_df.nlargest(max_motifs, 'score')

        # Get y-axis limits for positioning
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        motif_height = y_range * 0.03
        motif_y_base = y_min - y_range * 0.15

        # Plot motifs as rectangles
        from matplotlib.patches import Rectangle

        text_objects = []
        for _, motif in motif_df.iterrows():
            # Draw rectangle for motif
            rect = Rectangle(
                (motif['position'], motif_y_base),
                15,  # Approximate motif width
                motif_height,
                facecolor='gray',
                edgecolor='black',
                alpha=0.7,
                linewidth=1
            )
            ax.add_patch(rect)

            # Add label
            text_obj = ax.text(
                motif['position'] + 7.5,  # Center of rectangle
                motif_y_base + motif_height / 2,
                motif['motif_name'].split('.')[0],  # Remove suffix
                fontsize=7,
                ha='center',
                va='center',
                rotation=45
            )
            text_objects.append(text_obj)

        # Adjust text to avoid overlaps
        try:
            from adjustText import adjust_text
            adjust_text(text_objects, ax=ax,
                       force_text=(0.1, 0.2),
                       force_objects=(0.1, 0.2),
                       expand_text=(1.05, 1.2),
                       expand_objects=(1.05, 1.2))
        except:
            pass  # Skip if adjustText not available

        # Extend y-axis to show motifs
        ax.set_ylim(motif_y_base - y_range * 0.05, y_max)

    def plot_motif_activations(self, head_name: str,
                               figsize: Tuple[int, int] = (15, 8),
                               save_path: Optional[str] = None,
                               max_motifs: int = 10,
                               min_activation_percentile: float = 95) -> Optional[Figure]:
        """
        Plot motif activations from model for high-scoring regions.

        Requires motif_activations to be populated by analyzer.

        Args:
            head_name: Output head to analyze
            figsize: Figure size
            save_path: Optional path to save
            max_motifs: Maximum number of motifs to show
            min_activation_percentile: Percentile threshold for showing motifs

        Returns:
            matplotlib Figure or None if no motif activations available

        Example:
            >>> # After running analyzer with extract_motif_activations=True
            >>> del_result.plot_motif_activations('regression')
        """
        if self.motif_activations is None:
            print("No motif activations available. Run analyzer with extract_motif_activations=True")
            return None

        if head_name not in self.output_heads:
            raise ValueError(f"Unknown head '{head_name}'. Available: {self.output_heads}")

        activations = self.motif_activations['activations']  # (num_motifs, seq_len)
        motif_names = self.motif_activations.get('filtered_motif_names', None)
        high_score_regions = self.motif_activations['high_score_regions'].get(head_name, [])

        if motif_names is None or len(motif_names) == 0:
            print("No motif names available")
            return None

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

        # Top panel: Deletion effects with high-scoring regions highlighted
        if self.nucleotide_scores is not None:
            effects = self.nucleotide_scores[head_name]
            positions = np.arange(self.region_start, self.region_start + len(effects))
        else:
            effects = self.deletion_effects[head_name]
            positions = self.window_positions

        ax1.plot(positions, effects, linewidth=2, color='darkred', alpha=0.7)
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)

        # Highlight high-scoring regions
        for region_start, region_end in high_score_regions:
            genomic_start = self.region_start + region_start
            genomic_end = self.region_start + region_end
            ax1.axvspan(genomic_start, genomic_end, alpha=0.2, color='orange',
                       label='High-scoring region' if region_start == high_score_regions[0][0] else '')

        ax1.set_ylabel(f'Deletion Effect ({head_name})', fontsize=11)
        ax1.set_title(f'Deletion Effects with High-Scoring Regions', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if high_score_regions:
            ax1.legend(loc='best')

        # Bottom panel: Top motif activations in high-scoring regions
        if len(high_score_regions) > 0:
            # Calculate average activation for each motif in high-scoring regions
            motif_scores = []
            for motif_idx in range(len(motif_names)):
                total_activation = 0
                for region_start, region_end in high_score_regions:
                    region_activations = activations[motif_idx, region_start:region_end+1]
                    total_activation += torch.sum(region_activations).item()
                motif_scores.append(total_activation)

            # Get top motifs
            top_indices = np.argsort(motif_scores)[-max_motifs:][::-1]
            top_motifs = [motif_names[i] for i in top_indices]
            top_scores = [motif_scores[i] for i in top_indices]

            # Plot bar chart
            y_pos = np.arange(len(top_motifs))
            ax2.barh(y_pos, top_scores, color='steelblue', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([m.split('.')[0] for m in top_motifs], fontsize=9)
            ax2.set_xlabel('Total Activation in High-Scoring Regions', fontsize=11)
            ax2.set_title(f'Top {len(top_motifs)} Motifs by Activation', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, 'No high-scoring regions found',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig