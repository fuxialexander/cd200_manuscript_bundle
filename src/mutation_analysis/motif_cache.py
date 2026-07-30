#!/usr/bin/env python3
"""
Manages the precaching of FIMO PWM-to-p-value computations.

This module provides a class to build, save, and load a cache of statistical
mappings that are used to rapidly convert raw motif scores into p-values,
avoiding expensive recomputation during analysis.
"""

import os
import sys
import torch
import numpy as np
import pickle
from tqdm import tqdm
import warnings

# Add project paths - consider a more robust path management solution
# sys.path appends for local checkouts belong in scripts/bundle_env.sh (CAESAR_SRC).

# These imports are expected to be available in the environment.
# They are part of the broader project structure.
from mutation_analysis.inference import ModelInference
from mutation_analysis.fimo import _pwm_to_mapping, trim_motif_padding


class FimoMappingsCache:
    """Handles the creation, loading, and saving of FIMO p-value mappings."""

    def __init__(self, cache_data: dict):
        """
        Initialize with loaded cache data. Use .load() to create an instance.
        """
        self.cache = cache_data.get('fimo_cache', {})
        self.metadata = {k: v for k, v in cache_data.items() if k != 'fimo_cache'}

    @classmethod
    def load(cls, cache_path: str) -> 'FimoMappingsCache':
        """
        Load precached FIMO mappings from a file.

        Args:
            cache_path: Path to the cached mappings file.

        Returns:
            A FimoMappingsCache instance.
        """
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        print(f"✓ Loaded FIMO cache from {cache_path}")
        print(f"  - Cache entries: {len(cache_data.get('fimo_cache', {}))}")
        print(f"  - Bin size: {cache_data.get('bin_size')}")
        print(f"  - From checkpoint: {os.path.basename(cache_data.get('checkpoint_path', ''))}")

        return cls(cache_data)

    def get_mapping(self, motif_key):
        """
        Retrieve a p-value mapping for a given motif key.

        Args:
            motif_key: The unique key for the motif.

        Returns:
            The (smallest, logpdf) tuple for the mapping, or None if not found.
        """
        return self.cache.get(motif_key)

    @staticmethod
    def build(checkpoint_path: str, output_path: str, bin_size: float = 0.1):
        """
        Build the FIMO PWM-to-mapping cache from a trained model checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint file.
            output_path: Path to save the cached mappings.
            bin_size: Bin size for the FIMO calculations.
        """
        print(f"=== Building FIMO Mappings Cache ===")
        print(f"  - Checkpoint: {checkpoint_path}")
        print(f"  - Output: {output_path}")
        print(f"  - Bin size: {bin_size}\n")

        # Initialize model to get motif weights
        print("1. Loading model...")
        # Assuming ModelInference can be instantiated this way
        model_inference = ModelInference(checkpoint_path, model_type='phase2')
        model = model_inference.model

        # Extract motif weights
        print("2. Extracting motif weights...")
        motif_weights = FimoMappingsCache._extract_motif_weights(model)
        motif_weights_np = motif_weights.cpu().numpy()
        print(f"   ✓ Found {motif_weights_np.shape[0]} motifs with width {motif_weights_np.shape[2]}")

        # Compute and cache mappings
        print("\n3. Computing FIMO mappings for all motifs...")
        fimo_cache = {}
        successful_motifs = 0
        failed_motifs = []

        for motif_idx in tqdm(range(motif_weights_np.shape[0]), desc="Processing motifs"):
            try:
                motif_log_odds = motif_weights_np[motif_idx]
                trimmed_log_odds = trim_motif_padding(motif_log_odds)

                if trimmed_log_odds.shape[1] == 0:
                    warnings.warn(f"Motif {motif_idx} is empty after trimming padding")
                    continue

                motif_key = (trimmed_log_odds.shape, trimmed_log_odds.tobytes(), bin_size)
                if motif_key in fimo_cache:
                    continue

                smallest, logpdf = _pwm_to_mapping(trimmed_log_odds, bin_size)
                fimo_cache[motif_key] = (smallest, logpdf)
                successful_motifs += 1

            except Exception as e:
                failed_motifs.append((motif_idx, str(e)))
                print(f"Failed to process motif {motif_idx}: {e}")

        print(f"\n   ✓ Successfully cached mappings for {successful_motifs} motifs")
        if failed_motifs:
            print(f"   - Failed to cache {len(failed_motifs)} motifs.")

        # Save cache to file
        print(f"\n4. Saving cache to {output_path}...")
        cache_data = {
            'fimo_cache': fimo_cache,
            'bin_size': bin_size,
            'num_motifs': motif_weights_np.shape[0],
            'successful_motifs': successful_motifs,
            'failed_motifs': failed_motifs,
            'checkpoint_path': checkpoint_path
        }

        with open(output_path, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"   ✓ Cache saved successfully.")
        print(f"   - Total cache entries: {len(fimo_cache)}")
        print(f"   - File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        print("\n=== Cache building complete! ===")

    @staticmethod
    def _extract_motif_weights(model: torch.nn.Module) -> torch.Tensor:
        """
        Find and extract the motif convolutional weights from the model.
        Searches for the 'motif_conv' layer in common model architectures.
        """
        # Common patterns in different model architectures
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
                    print(f"   ✓ Found motif_conv layer")
                    return layer.weight.data
            except AttributeError:
                continue

        raise AttributeError("Could not find a 'motif_conv' layer in the model.")


if __name__ == '__main__':
    """
    Example usage: run this script to build the cache.
    """
    print("Running FIMO Mappings Cache Builder...")

    # --- Configuration ---
    # IMPORTANT: Update this path to your model checkpoint
    CHECKPOINT_PATH = "checkpoints/ATAC-Sequential--20250923-205442_phase2_full_model.ckpt"
    CACHE_OUTPUT_PATH = "fimo_mappings_cache.pkl"
    BIN_SIZE = 0.1
    # ---------------------

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint file not found at: {CHECKPOINT_PATH}")
        print("Please update the CHECKPOINT_PATH variable in this script.")
        sys.exit(1)

    try:
        # Build and save the cache
        FimoMappingsCache.build(
            checkpoint_path=CHECKPOINT_PATH,
            output_path=CACHE_OUTPUT_PATH,
            bin_size=BIN_SIZE
        )

        # Example of loading the cache
        print("\n--- Testing cache loading ---")
        loaded_cache = FimoMappingsCache.load(CACHE_OUTPUT_PATH)
        print("Cache loaded successfully.")
        print(f"Metadata 'bin_size': {loaded_cache.metadata.get('bin_size')}")

    except Exception as e:
        print(f"\n❌ An error occurred during cache building: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
