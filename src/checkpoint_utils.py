#!/usr/bin/env python3
"""
Utility functions for handling checkpoint loading with torch.compile compatibility.

This module provides functions to safely load PyTorch Lightning checkpoints
that may have been saved with torch.compile, which adds '_orig_mod' prefixes
to parameter names.
"""

import torch
from typing import Dict, Any, Tuple, Optional


def clean_compiled_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove '_orig_mod' prefixes from compiled model state dict.

    When torch.compile() is used, it wraps the model and adds '_orig_mod'
    prefixes to all parameter names. This function removes those prefixes
    to make the state dict compatible with non-compiled models.

    Args:
        state_dict: The state dictionary from a checkpoint

    Returns:
        Cleaned state dictionary with '_orig_mod' prefixes removed
    """
    if not any('_orig_mod' in key for key in state_dict.keys()):
        return state_dict

    print("Detected compiled model checkpoint - removing _orig_mod prefixes")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if '_orig_mod.' in key:
            # Remove _orig_mod prefix from anywhere in the key
            new_key = key.replace('_orig_mod.', '')
            cleaned_state_dict[new_key] = value
        else:
            cleaned_state_dict[key] = value

    return cleaned_state_dict


def load_checkpoint_safe(checkpoint_path: str,
                        model: torch.nn.Module,
                        strict: bool = False,
                        map_location: str = 'cpu') -> Tuple[Dict[str, Any], bool]:
    """
    Safely load a PyTorch Lightning checkpoint with compiled model support.

    This function handles checkpoints that may have been saved with torch.compile,
    automatically cleaning '_orig_mod' prefixes and providing detailed logging
    about the loading process.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load the checkpoint into
        strict: Whether to use strict loading (default: False for flexibility)
        map_location: Device to map the checkpoint to (default: 'cpu')

    Returns:
        Tuple of (checkpoint_dict, success_flag)
    """
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Handle different checkpoint structures
        state_dict = None
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'base_model_state_dict' in checkpoint:
            # Custom checkpoint structure with separate base and bias models
            print("Found custom checkpoint structure with base_model_state_dict")
            state_dict = checkpoint['base_model_state_dict']
        else:
            print("Warning: No compatible state_dict found in checkpoint")
            available_keys = list(checkpoint.keys())
            print(f"Available keys: {available_keys}")
            return checkpoint, False

        # Clean the state dict of any compiled model artifacts
        state_dict = clean_compiled_state_dict(state_dict)

        # Load the state dict into the model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        # Report loading results
        print(f"Checkpoint loaded successfully:")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")

        if missing_keys and not strict:
            print("  Missing keys (expected for flexible loading):")
            for key in missing_keys[:5]:  # Show first 5
                print(f"    - {key}")
            if len(missing_keys) > 5:
                print(f"    ... and {len(missing_keys) - 5} more")

        if unexpected_keys and not strict:
            print("  Unexpected keys (will be ignored):")
            for key in unexpected_keys[:5]:  # Show first 5
                print(f"    - {key}")
            if len(unexpected_keys) > 5:
                print(f"    ... and {len(unexpected_keys) - 5} more")

        return checkpoint, True

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return {}, False


def load_lightning_checkpoint_safe(checkpoint_path: str,
                                 lightning_module_class,
                                 *args, **kwargs) -> Optional[torch.nn.Module]:
    """
    Safely load a PyTorch Lightning module from checkpoint with compiled model support.

    This is a higher-level function that creates a Lightning module instance
    and loads the checkpoint into it, handling compiled model artifacts.

    Args:
        checkpoint_path: Path to the checkpoint file
        lightning_module_class: The Lightning module class to instantiate
        *args, **kwargs: Arguments to pass to the Lightning module constructor

    Returns:
        Loaded Lightning module or None if loading failed
    """
    try:
        # First try the standard Lightning loading
        try:
            model = lightning_module_class.load_from_checkpoint(
                checkpoint_path, *args, **kwargs
            )
            print("✓ Standard Lightning checkpoint loading successful")
            return model
        except Exception as e:
            print(f"Standard Lightning loading failed: {e}")
            print("Attempting manual loading with compiled model support...")

        # Manual loading with compiled model support
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract hyperparameters
        hparams = checkpoint.get('hyper_parameters', {})

        # Update with provided kwargs
        hparams.update(kwargs)

        # Create model instance
        model = lightning_module_class(*args, **hparams)

        # Load the state dict with compiled model support
        if 'state_dict' in checkpoint:
            state_dict = clean_compiled_state_dict(checkpoint['state_dict'])
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            print(f"✓ Manual checkpoint loading successful:")
            print(f"  Missing keys: {len(missing_keys)}")
            print(f"  Unexpected keys: {len(unexpected_keys)}")

        return model

    except Exception as e:
        print(f"✗ Failed to load Lightning checkpoint: {e}")
        return None


def fix_checkpoint_compiled_keys(checkpoint_path: str, output_path: Optional[str] = None) -> bool:
    """
    Fix a checkpoint file by removing compiled model artifacts.

    This function loads a checkpoint, cleans any '_orig_mod' prefixes,
    and saves it back, making it compatible with non-compiled models.

    Args:
        checkpoint_path: Path to the checkpoint file to fix
        output_path: Path to save the fixed checkpoint (default: overwrites original)

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Fixing compiled checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' not in checkpoint:
            print("No state_dict found in checkpoint")
            return False

        # Clean the state dict
        original_keys = len(checkpoint['state_dict'])
        checkpoint['state_dict'] = clean_compiled_state_dict(checkpoint['state_dict'])
        cleaned_keys = len(checkpoint['state_dict'])

        # Save the fixed checkpoint
        save_path = output_path or checkpoint_path
        torch.save(checkpoint, save_path)

        print(f"✓ Fixed checkpoint saved to: {save_path}")
        print(f"  Original keys: {original_keys}")
        print(f"  Cleaned keys: {cleaned_keys}")

        return True

    except Exception as e:
        print(f"✗ Failed to fix checkpoint: {e}")
        return False