"""
Mutation analyzer for genomic variant effect prediction.

Orchestrates mutation analysis and returns result objects with plotting methods.
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import Variant, MutationDataset
from .metrics import calculate_effects
from .motif_cache import FimoMappingsCache
from .fimo import pvalue_from_logpdf, trim_motif_padding


if TYPE_CHECKING:
    from .motif_filters import MotifFilter


class FimoMotifScanner:
    """
    Performs ultra-fast, FIMO-like motif scanning using a precached
    statistical model of p-value distributions.
    """
    def __init__(self, model: nn.Module, motif_names: List[str],
                 fimo_cache: FimoMappingsCache, p_value_threshold: float = 1e-4):
        """
        Args:
            model: The trained PyTorch model containing the motif_conv layer.
            motif_names: A list of names corresponding to the motifs in the model.
            fimo_cache: A loaded FimoMappingsCache instance.
            p_value_threshold: The p-value cutoff for reporting motif hits.
        """
        self.model = model
        self.motif_names = motif_names
        self.fimo_cache = fimo_cache
        self.p_value_threshold = p_value_threshold

        self.motif_weights = self._extract_motif_weights(model).cpu()
        self.motif_conv = self._find_motif_conv_layer(model)
        
        if len(motif_names) != self.motif_weights.shape[0]:
            raise ValueError(
                f"Mismatch between number of motif names ({len(motif_names)}) and "
                f"number of motifs in model ({self.motif_weights.shape[0]})"
            )

    def scan(self, sequence: torch.Tensor, verbose: bool = False) -> pd.DataFrame:
        """
        Scan a sequence for motif occurrences.

        Args:
            sequence: A one-hot encoded sequence tensor of shape (4, length).
            verbose: If True, print detailed debug logs for p-value filtering.

        Returns:
            A pandas DataFrame of significant motif hits.
        """
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)  # Add batch dimension

        device = next(self.model.parameters()).device
        sequence = sequence.to(device)
        
        # Use convolution to get raw scores for all motifs at all positions
        with torch.no_grad():
            scores = self.motif_conv(sequence)[0] # Shape: (n_motifs, seq_len)
        
        all_hits = []
        
        for motif_idx in range(scores.shape[0]):
            motif_log_odds = self.motif_weights[motif_idx].numpy()
            trimmed_log_odds = trim_motif_padding(motif_log_odds)
            
            if trimmed_log_odds.shape[1] == 0:
                continue

            # Use cache to get p-value mapping
            motif_key = (trimmed_log_odds.shape, trimmed_log_odds.tobytes(), self.fimo_cache.metadata['bin_size'])
            mapping = self.fimo_cache.get_mapping(motif_key)
            
            if mapping is None:
                continue
                
            smallest, logpdf = mapping
            
            # Get scores for this specific motif
            motif_scores = scores[motif_idx, :].cpu().numpy()
            
            # Find positions exceeding the minimum possible score
            min_score = smallest
            candidate_indices = np.where(motif_scores > min_score)[0]
            
            if len(candidate_indices) == 0:
                continue

            candidate_scores = motif_scores[candidate_indices]
            
            # Convert scores to p-values
            pvalues = pvalue_from_logpdf(candidate_scores, smallest, logpdf, self.fimo_cache.metadata['bin_size'])
            
            # Filter by p-value threshold
            significant_mask = pvalues < self.p_value_threshold

            if verbose:
                for idx, score, pval, is_sig in zip(candidate_indices, candidate_scores, pvalues, significant_mask):
                    status = "passed" if is_sig else "failed"
                    print(
                        f"  DEBUG: [Detailed] Key motif '{self.motif_names[motif_idx]}' {status} "
                        f"p-value filter with p={pval:.6f} (score {score:.2f})"
                    )
            
            if np.any(significant_mask):
                hit_positions = candidate_indices[significant_mask]
                
                for pos in hit_positions:
                    # Currently, strand is not determined, defaulting to '+'
                    # A more complete implementation would check both strands
                    all_hits.append({
                        "motif_name": self.motif_names[motif_idx],
                        "start": pos,
                        "end": pos + trimmed_log_odds.shape[1],
                        "score": motif_scores[pos],
                        "p_value": pvalues[np.where(candidate_indices == pos)[0][0]],
                        "strand": '+'
                    })

        if not all_hits:
            return pd.DataFrame(columns=["motif_name", "start", "end", "score", "p_value", "strand"])
            
        return pd.DataFrame(all_hits).sort_values("p_value").reset_index(drop=True)

    @staticmethod
    def _extract_motif_weights(model: nn.Module) -> torch.Tensor:
        layer = FimoMotifScanner._find_motif_conv_layer(model)
        return layer.weight.data

    @staticmethod
    def _find_motif_conv_layer(model: nn.Module) -> nn.Conv1d:
        possible_paths = [
            lambda m: m.motif_conv,
            lambda m: m.model.motif_conv,
            lambda m: m.model.main_model.motif_conv,
            lambda m: m.main_model.motif_conv,
        ]
        for path_fn in possible_paths:
            try:
                layer = path_fn(model)
                if layer is not None and isinstance(layer, torch.nn.Conv1d):
                    return layer
            except AttributeError:
                continue
        raise AttributeError("Could not find a 'motif_conv' layer in the model.")


class MutationAnalyzer:
    """
    Main analysis interface - returns Result objects with plotting methods.

    All analysis methods return result objects that know how to plot themselves.
    No plotting logic in this class - keeps analysis and visualization separate.

    Example:
        >>> analyzer = MutationAnalyzer(model, output_heads, dataset)
        >>> result = analyzer.analyze_variant(dataset[0])
        >>> result.plot_summary()
    """

    def __init__(self, model: nn.Module, output_heads: List[str],
                 dataset: MutationDataset, device: Optional[str] = None,
                 motif_filter: Optional['MotifFilter'] = None,
                 motif_names: Optional[List[str]] = None,
                 motif_scanner: Optional[FimoMotifScanner] = None):
        """
        Initialize analyzer with loaded model and dataset.

        Args:
            model: PyTorch model (from load_model, not wrapped)
            output_heads: List of output head names (from load_model)
            dataset: MutationDataset with genome interface
            device: Device for inference (auto-detected if None)
            motif_filter: Optional MotifFilter for filtering motifs in analysis
            motif_names: Optional list of motif names (for motif activation extraction)
            motif_scanner: Optional FimoMotifScanner for fast motif scanning.

        Example:
            >>> from mutation_analysis import load_model, MutationDataset, MutationAnalyzer
            >>> from mutation_analysis.motif_filters import create_annotation_filter
            >>> model, heads = load_model("model.ckpt")
            >>> motif_filter = create_annotation_filter('annot.csv', 'expressed')
            >>> analyzer = MutationAnalyzer(model, heads, dataset, motif_filter=motif_filter)
        """
        self.model = model
        self.output_heads = output_heads
        self.dataset = dataset
        self.device = device or str(next(model.parameters()).device)
        self.motif_filter = motif_filter
        self.motif_names = motif_names
        self.motif_scanner = motif_scanner
        self.model.eval()  # Always in eval mode for inference

    def scan_motifs(self, chromosome: str, start: int, end: int, verbose: bool = False) -> pd.DataFrame:
        """
        Scan a specific genomic region for motifs using the configured scanner.

        Args:
            chromosome: Chromosome name.
            start: Start genomic coordinate.
            end: End genomic coordinate.
            verbose: If True, print detailed debug logs for p-value filtering.

        Returns:
            A pandas DataFrame of significant motif hits, filtered by the
            analyzer's motif_filter if one is configured.
        """
        if not self.motif_scanner:
            raise RuntimeError(
                "Motif scanner is not configured. Please initialize MutationAnalyzer "
                "with a FimoMotifScanner instance."
            )

        # Get sequence for the specified region
        sequence = self.dataset.get_sequence(chromosome, start, end)

        # Scan for motifs
        motif_hits = self.motif_scanner.scan(sequence, verbose=verbose)
        
        # Adjust positions to be genomic coordinates
        motif_hits['start'] += start
        motif_hits['end'] += start

        # Apply motif filter if provided
        if self.motif_filter:
            allowed_motifs = self.motif_filter.filter_motifs(
                self.motif_scanner.motif_names
            )
            motif_hits = motif_hits[motif_hits['motif_name'].isin(allowed_motifs)]

        return motif_hits

    def analyze_variant(self, variant: Variant,
                       extend_bp: int = 2048) -> 'VariantResult':
        """
        Comprehensive analysis of single variant.

        Args:
            variant: Variant to analyze
            extend_bp: Context window size around variant

        Returns:
            VariantResult with plotting methods:
            - .plot_summary() - comprehensive multi-panel plot
            - .plot_head_output(head_name) - specific output head
            - .plot_motif_activation() - motif changes

        Example:
            >>> result = analyzer.analyze_variant(dataset[0])
            >>> result.plot_summary()
            >>> result.plot_head_output('regression')
        """
        from .results import VariantResult

        # Get sequences centered on variant
        ref_seq, alt_seq = self.dataset.get_variant_sequences(variant, extend_bp)

        # Add batch dimension and move to device
        ref_seq = ref_seq.unsqueeze(0).to(self.device)
        alt_seq = alt_seq.unsqueeze(0).to(self.device)

        # Forward pass (use original model)
        with torch.no_grad():
            ref_outputs = self._forward_model(ref_seq)
            alt_outputs = self._forward_model(alt_seq)

        # Calculate effects for all heads
        effects = calculate_effects(ref_outputs, alt_outputs, self.output_heads)

        # Motif scanning is now decoupled. Users can call scan_motifs for a region.
        ref_motifs = pd.DataFrame()
        alt_motifs = pd.DataFrame()

        # Return result object with plotting methods
        return VariantResult(
            variant=variant,
            ref_outputs={k: v.cpu() for k, v in ref_outputs.items()},
            alt_outputs={k: v.cpu() for k, v in alt_outputs.items()},
            effects=effects,
            ref_motifs=ref_motifs,
            alt_motifs=alt_motifs,
            output_heads=self.output_heads,
            analyzer=self
        )

    def in_silico_mutagenesis(self, variant: Variant,
                             window_size: int = 50,
                             center_position: Optional[int] = None) -> 'ISMResult':
        """
        Systematic mutagenesis - mutate all positions to all nucleotides.

        Args:
            variant: Variant defining the region
            window_size: Number of positions to mutate
            center_position: Position to center window on (default: variant.position)

        Returns:
            ISMResult with plotting methods:
            - .plot_summary() - all logos
            - .plot_logo(head_name) - specific metric logo
            - .plot_contribution_scores(head_name) - contribution scores

        Example:
            >>> ism = analyzer.in_silico_mutagenesis(dataset[0], window_size=50)
            >>> ism.plot_logo('log_fold_change')
        """
        from .results import ISMResult

        center = center_position or variant.position

        # Get reference sequence
        window_start = center - window_size // 2
        window_end = window_start + window_size
        ref_seq = self.dataset.get_sequence(variant.chromosome, window_start, window_end)

        # Get baseline prediction
        with torch.no_grad():
            ref_outputs = self._forward_model(ref_seq.unsqueeze(0).to(self.device))

        # Initialize effect matrices (L x 4) for each head
        effect_matrices = {head: np.zeros((window_size, 4))
                          for head in self.output_heads}

        # Mutate each position to each nucleotide
        bases = ['A', 'C', 'G', 'T']
        base_to_idx = {b: i for i, b in enumerate(bases)}

        for pos_idx in tqdm(range(window_size), desc="ISM"):
            ref_base_idx = torch.argmax(ref_seq[:, pos_idx]).item()
            ref_base = bases[ref_base_idx]

            for alt_idx, alt_base in enumerate(bases):
                if alt_base == ref_base:
                    continue  # Will be filled by mean-centering

                # Create mutated sequence
                mut_seq = ref_seq.clone()
                mut_seq[:, pos_idx] = 0
                mut_seq[alt_idx, pos_idx] = 1

                # Get prediction
                with torch.no_grad():
                    mut_outputs = self._forward_model(mut_seq.unsqueeze(0).to(self.device))

                # Calculate effects for all heads
                effects = calculate_effects(ref_outputs, mut_outputs, self.output_heads)

                # Store in matrices
                for head in self.output_heads:
                    effect_matrices[head][pos_idx, alt_idx] = effects[head]['primary_metric']

        # Apply mean-centering
        for head in self.output_heads:
            effect_matrices[head] = self._mean_center(effect_matrices[head], ref_seq)

        # Calculate contribution scores (reference base values)
        contribution_scores = {}
        for head in self.output_heads:
            scores = np.zeros(window_size)
            for pos_idx in range(window_size):
                ref_base_idx = torch.argmax(ref_seq[:, pos_idx]).item()
                scores[pos_idx] = effect_matrices[head][pos_idx, ref_base_idx]
            contribution_scores[head] = scores

        # Get reference sequence string
        ref_sequence = ''.join(bases[torch.argmax(ref_seq[:, i]).item()]
                              for i in range(window_size))

        return ISMResult(
            variant=variant,
            effect_matrices=effect_matrices,
            contribution_scores=contribution_scores,
            ref_sequence=ref_sequence,
            output_heads=self.output_heads,
            window_start=window_start
        )

    def sliding_window_deletion(self, variant: Variant,
                               deletion_size: int = 15,
                               stride: int = 1,
                               extend_bp: int = 2048,
                               distribute_effects: bool = False,
                               smoothing_window: int = 31,
                               core_region: Optional[Tuple[int, int]] = None,
                               real_deletion: bool = False,
                               extract_motif_activations: bool = False,
                               motif_threshold_sigma: float = 4.0,
                               save_diff_matrix_path: Optional[str] = None) -> 'DeletionResult':
        """
        Sliding window deletion analysis.

        Args:
            variant: Variant defining the region
            deletion_size: Size of deletion window (bp)
            stride: Step size between windows
            extend_bp: Context size (full sequence length for model input)
            distribute_effects: If True, distribute window effects across nucleotides
                               for per-base resolution scoring
            smoothing_window: Window size for smoothing (applied to nucleotide scores if
                           distribute_effects=True, otherwise applied to window effects)
            core_region: Optional (start, end) genomic coordinates for core region.
                        Effects are computed only on predictions within this region.
                        Deletions still scan across full extend_bp context.
            real_deletion: If True, remove sequence and stitch together (more realistic).
                          If False, zero out the region (faster, but less realistic).
            extract_motif_activations: If True, extract motif activations from model
                                      intermediate layers for high-scoring regions.
            motif_threshold_sigma: Threshold in standard deviations for identifying
                                  high-scoring regions for motif extraction.
            save_diff_matrix_path: If provided, save a matrix of regression output
                                  differences (ref - alt) to this .npy file path.

        Returns:
            DeletionResult with plotting methods:
            - .plot_summary() - all deletion effects
            - .plot_deletion_effects(head_name) - specific head
            - .plot_motif_activation() - motif changes in top regions

        Example:
            >>> del_result = analyzer.sliding_window_deletion(dataset[0])
            >>> del_result.plot_summary()

            >>> # With core region, distributed scoring, and real deletion
            >>> del_result = analyzer.sliding_window_deletion(
            ...     dataset[0],
            ...     distribute_effects=True,
            ...     core_region=(136117626, 136117842),
            ...     real_deletion=True,
            ...     extract_motif_activations=True
            ... )
        """
        from .results import DeletionResult

        # Get reference sequence
        center = variant.position
        window_start = center - extend_bp // 2
        window_end = window_start + extend_bp
        ref_seq = self.dataset.get_sequence(variant.chromosome, window_start, window_end)

        # Get observed ATAC data if available
        observed_atac = self.dataset.get_atac_signals(variant.chromosome, window_start, window_end)

        # Calculate core region in sequence coordinates if specified
        core_start_seq = None
        core_end_seq = None
        if core_region is not None:
            core_start_genomic, core_end_genomic = core_region
            # Convert genomic coordinates to sequence indices
            core_start_seq = core_start_genomic - window_start
            core_end_seq = core_end_genomic - window_start
            # Validate
            if core_start_seq < 0 or core_end_seq > extend_bp:
                raise ValueError(
                    f"Core region {core_region} outside extended region "
                    f"[{window_start}, {window_end}]"
                )

        # Baseline prediction
        with torch.no_grad():
            baseline = self._forward_model(ref_seq.unsqueeze(0).to(self.device))

        # Generate deletion windows
        n_windows = (extend_bp - deletion_size) // stride + 1
        window_positions = []
        deletion_effects = {head: np.zeros(n_windows) for head in self.output_heads}

        # For distributed effects, track nucleotide-level scores
        if distribute_effects:
            nucleotide_scores = {head: np.zeros(extend_bp) for head in self.output_heads}
            nucleotide_scores_unsmoothed = {head: np.zeros(extend_bp) for head in self.output_heads}
        else:
            nucleotide_scores = None
            nucleotide_scores_unsmoothed = None

        # For saving raw difference matrix
        diff_matrices = [] if save_diff_matrix_path is not None else None

        for i, start_pos in tqdm(enumerate(range(0, extend_bp - deletion_size + 1, stride)),
                                 total=n_windows, desc="Deletion scan"):
            # Create deleted sequence
            if real_deletion:
                # Real deletion: remove the region and stitch the remaining sequence
                left_part = ref_seq[:, :start_pos]
                right_part = ref_seq[:, start_pos + deletion_size:]
                del_seq = torch.cat([left_part, right_part], dim=-1)
            else:
                # Zero out (faster, original behavior)
                del_seq = ref_seq.clone()
                del_seq[:, start_pos:start_pos + deletion_size] = 0

            # Get prediction
            with torch.no_grad():
                del_outputs = self._forward_model(del_seq.unsqueeze(0).to(self.device))

            # If saving diff matrix, calculate and store it
            if diff_matrices is not None:
                ref_regr = baseline['regression'].squeeze().cpu().numpy()
                alt_regr = del_outputs['regression'].squeeze().cpu().numpy()

                # Ensure lengths match if real_deletion=False (zero-padding)
                if not real_deletion and ref_regr.shape != alt_regr.shape:
                    pad_width = ref_regr.shape[0] - alt_regr.shape[0]
                    alt_regr = np.pad(alt_regr, (0, pad_width), 'constant')

                # For real_deletion=True, reconstruct full-length profile
                if real_deletion:
                    left_part = alt_regr[:start_pos]
                    right_part = alt_regr[start_pos:]
                    padded_alt_regr = np.concatenate([left_part, np.zeros(deletion_size), right_part])
                    alt_regr = padded_alt_regr
                
                diff = ref_regr - alt_regr
                diff_matrices.append(diff)

            # Calculate effects with core region slicing
            # Convert core_region from sequence coords if specified
            core_region_seq = None
            if core_start_seq is not None:
                core_region_seq = (core_start_seq, core_end_seq)

            effects = calculate_effects(baseline, del_outputs, self.output_heads,
                                      core_region=core_region_seq)

            # Store window-level effects
            for head in self.output_heads:
                effect_value = effects[head]['primary_metric']
                deletion_effects[head][i] = effect_value

                # Distribute effect across nucleotides if requested
                if distribute_effects:
                    # Spread window effect evenly across deleted nucleotides
                    for pos in range(start_pos, min(start_pos + deletion_size, extend_bp)):
                        nucleotide_scores[head][pos] += effect_value / deletion_size

            window_positions.append(window_start + start_pos + deletion_size // 2)

        # After the loop, save the matrix if requested
        if diff_matrices is not None and save_diff_matrix_path is not None:
            # Stack along a new axis to get (L, num_deletions)
            final_matrix = np.stack(diff_matrices, axis=-1)
            np.save(save_diff_matrix_path, final_matrix)
            print(f"✓ Saved regression difference matrix to {save_diff_matrix_path}")

        # Apply smoothing
        if smoothing_window > 1:
            if distribute_effects:
                # Smooth nucleotide-level scores (run_analysis_dlbcl.py line 998)
                for head in self.output_heads:
                    nucleotide_scores[head] = np.convolve(
                        nucleotide_scores[head],
                        np.ones(smoothing_window) / smoothing_window,
                        mode='same'
                    )
            else:
                # Also smooth window-level effects in traditional mode
                # (run_analysis_dlbcl.py line 1003)
                for head in self.output_heads:
                    deletion_effects[head] = np.convolve(
                        deletion_effects[head],
                        np.ones(smoothing_window) / smoothing_window,
                        mode='same'
                    )

        # Extract motif activations if requested
        motif_activations = None
        if extract_motif_activations:
            motif_activations = self._extract_motif_activations(
                ref_seq, deletion_effects, nucleotide_scores,
                distribute_effects, motif_threshold_sigma
            )

        # Motif scanning (placeholder for BED-based motifs)
        ref_motifs = None
        alt_motifs = None

        return DeletionResult(
            variant=variant,
            deletion_effects=deletion_effects,
            window_positions=np.array(window_positions),
            deletion_size=deletion_size,
            stride=stride,
            ref_motifs=ref_motifs, # TODO: Integrate FIMO scanning here as well
            alt_motifs=alt_motifs, # TODO: Integrate FIMO scanning here as well
            output_heads=self.output_heads,
            nucleotide_scores=nucleotide_scores,
            region_start=window_start,
            region_end=window_end,
            motif_activations=motif_activations,
            analyzer=self,
            baseline_predictions={k: v.squeeze().cpu().numpy() for k, v in baseline.items()},
            observed_atac=observed_atac.numpy() if observed_atac is not None else None,
            core_region=core_region
        )

    # ========== Helper Methods ==========

    def _forward_model(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model and collect output heads.

        Args:
            sequence: Input sequence tensor (batch, 4, length)

        Returns:
            Dictionary mapping head names to output tensors
        """
        # Get model outputs
        outputs = self.model(sequence)

        # Handle different output formats
        if isinstance(outputs, dict):
            # Model already returns dict
            return outputs
        elif isinstance(outputs, tuple):
            # Model returns tuple - map to head names
            if len(outputs) != len(self.output_heads):
                raise RuntimeError(
                    f"Model returned {len(outputs)} outputs but expected "
                    f"{len(self.output_heads)} heads: {self.output_heads}"
                )
            return {head: out for head, out in zip(self.output_heads, outputs)}
        else:
            # Single output - assume it's the first head
            return {self.output_heads[0]: outputs}

    def _mean_center(self, effect_matrix: np.ndarray,
                    ref_seq: torch.Tensor) -> np.ndarray:
        """
        Apply mean-centering to ISM effect matrix.

        Args:
            effect_matrix: Effect matrix (L, 4)
            ref_seq: Reference sequence (4, L)

        Returns:
            Mean-centered effect matrix
        """
        centered = effect_matrix.copy()

        L = effect_matrix.shape[0]
        for pos_idx in range(L):
            # Calculate mean of alternative bases
            ref_base_idx = torch.argmax(ref_seq[:, pos_idx]).item()
            alt_indices = [i for i in range(4) if i != ref_base_idx]
            mean_alt = np.mean([effect_matrix[pos_idx, i] for i in alt_indices])

            # Set reference base value to negative of mean
            centered[pos_idx, ref_base_idx] = -mean_alt

        return centered

    def _extract_motif_activations(self, ref_seq: torch.Tensor,
                                   deletion_effects: Dict[str, np.ndarray],
                                   nucleotide_scores: Optional[Dict[str, np.ndarray]],
                                   distribute_effects: bool,
                                   threshold_sigma: float) -> Optional[Dict[str, any]]:
        """
        Extract motif activations from model intermediate layers for high-scoring regions.

        Args:
            ref_seq: Reference sequence tensor (4, L)
            deletion_effects: Window-level deletion effects per head
            nucleotide_scores: Nucleotide-level scores per head (if distribute_effects=True)
            distribute_effects: Whether distributed effects were used
            threshold_sigma: Threshold in standard deviations for high-scoring regions

        Returns:
            Dictionary with:
            - 'activations': Motif activation tensor (num_motifs, seq_len)
            - 'filtered_motif_names': List of motif names after filtering
            - 'high_score_regions': Dict mapping head -> list of (start, end) positions
            Or None if extraction fails
        """
        try:
            # Try to access motif_conv layer from different possible locations
            motif_conv = None

            # Common patterns in different model architectures
            possible_paths = [
                lambda: self.model.motif_conv,                    # Direct attribute
                lambda: self.model.model.motif_conv,              # Nested under .model
                lambda: self.model.model.main_model.motif_conv,   # Phase 2 architecture
                lambda: self.model.main_model.motif_conv,         # Alternative nesting
            ]

            for path_fn in possible_paths:
                try:
                    motif_conv = path_fn()
                    if motif_conv is not None:
                        print(f"✓ Found motif_conv layer")
                        break
                except AttributeError:
                    continue

            if motif_conv is None:
                print("Warning: Could not find motif_conv layer in model. Skipping motif extraction.")
                return None

            # Get motif activations for reference sequence
            with torch.no_grad():
                ref_seq_batch = ref_seq.unsqueeze(0).to(self.device)
                motif_activations = motif_conv(ref_seq_batch)[0]  # Shape: (num_motifs, seq_len)

            print(f"✓ Extracted motif activations: shape {motif_activations.shape}")

            # Apply motif filter if provided
            filtered_motif_names = self.motif_names
            if self.motif_filter is not None and self.motif_names is not None:
                filtered_motif_names = self.motif_filter.filter_motifs(self.motif_names)
                print(f"✓ Motif filter: {len(filtered_motif_names)}/{len(self.motif_names)} motifs passed")
                print(f"  Filter description: {self.motif_filter.get_description()}")

            # Identify high-scoring regions for each head
            high_score_regions = {}
            for head_idx, head in enumerate(self.output_heads):
                if distribute_effects and nucleotide_scores is not None:
                    # Use nucleotide-level scores
                    scores = nucleotide_scores[head]
                else:
                    # Use window-level effects (less resolution)
                    scores = deletion_effects[head]

                # Calculate threshold
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                threshold = mean_score + threshold_sigma * std_score

                # Identify consecutive high-scoring regions
                high_positions = np.where(scores > threshold)[0]
                regions = self._identify_consecutive_regions(high_positions, max_gap=5)

                high_score_regions[head] = regions
                print(f"  Head '{head}': {len(regions)} high-scoring regions "
                      f"({len(high_positions)} positions above {threshold_sigma}-sigma)")

            return {
                'activations': motif_activations.cpu(),
                'filtered_motif_names': filtered_motif_names,
                'high_score_regions': high_score_regions
            }

        except Exception as e:
            print(f"Warning: Failed to extract motif activations: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _identify_consecutive_regions(self, positions: np.ndarray,
                                     max_gap: int = 5) -> List[Tuple[int, int]]:
        """
        Group consecutive positions into regions.

        Args:
            positions: Array of position indices
            max_gap: Maximum gap between positions to consider them consecutive

        Returns:
            List of (start, end) tuples for each region
        """
        if len(positions) == 0:
            return []

        regions = []
        current_start = positions[0]
        current_end = positions[0]

        for pos in positions[1:]:
            if pos - current_end <= max_gap:
                # Extend current region
                current_end = pos
            else:
                # Start new region
                regions.append((current_start, current_end))
                current_start = pos
                current_end = pos

        # Add final region
        regions.append((current_start, current_end))

        return regions