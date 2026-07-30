"""
Unified checkpoint management system for multi-phase training.

This module provides a clean abstraction for saving and loading model checkpoints
across different training phases, with support for:
- Component-based model architectures (bias, main, differential heads)
- Compiled model state dict cleaning
- Flexible loading strategies with fallbacks
- Comprehensive metadata tracking
- Phase-specific checkpoint paths
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import pytorch_lightning as pl
from checkpoint_utils import clean_compiled_state_dict


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving/loading."""
    run_name: str
    phase: str
    checkpoint_dir: Path = Path("checkpoints")
    save_metadata: bool = True
    strict_loading: bool = False
    clean_compiled: bool = True  # Clean torch.compile artifacts

    def get_checkpoint_path(self, suffix: str = "") -> Path:
        """Generate checkpoint path for this phase."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.run_name}_{self.phase}"
        if suffix:
            filename += f"_{suffix}"
        return self.checkpoint_dir / f"{filename}.ckpt"


@dataclass
class ModelComponents:
    """Container for model components with metadata."""
    # Core model components
    bias_model: Optional[torch.nn.Module] = None
    main_model: Optional[torch.nn.Module] = None
    differential_head: Optional[torch.nn.Module] = None
    full_model: Optional[Union[torch.nn.Module, pl.LightningModule]] = None

    # Metadata
    component_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Track which components are present."""
        self.component_names = []
        if self.bias_model is not None:
            self.component_names.append("bias_model")
        if self.main_model is not None:
            self.component_names.append("main_model")
        if self.differential_head is not None:
            self.component_names.append("differential_head")
        if self.full_model is not None:
            self.component_names.append("full_model")

    def to_dict(self, clean_compiled: bool = True) -> Dict[str, Any]:
        """Convert components to state dict dictionary."""
        state_dict = {}

        if self.bias_model is not None:
            bias_state = self.bias_model.state_dict()
            state_dict['bias_model_state_dict'] = (
                clean_compiled_state_dict(bias_state) if clean_compiled else bias_state
            )

        if self.main_model is not None:
            main_state = self.main_model.state_dict()
            state_dict['main_model_state_dict'] = (
                clean_compiled_state_dict(main_state) if clean_compiled else main_state
            )

        if self.differential_head is not None:
            diff_state = self.differential_head.state_dict()
            state_dict['differential_head_state_dict'] = (
                clean_compiled_state_dict(diff_state) if clean_compiled else diff_state
            )

        if self.full_model is not None:
            full_state = self.full_model.state_dict()
            state_dict['full_model_state_dict'] = (
                clean_compiled_state_dict(full_state) if clean_compiled else full_state
            )
            # Also save standard PyTorch Lightning format
            state_dict['state_dict'] = state_dict['full_model_state_dict']

        return state_dict


class CheckpointManager:
    """
    Unified checkpoint manager for multi-phase training.

    Features:
    - Save/load individual model components or full models
    - Automatic fallback strategies for flexible loading
    - Metadata tracking for reproducibility
    - Support for compiled models

    Example:
        >>> manager = CheckpointManager(run_name="experiment1", phase="phase2")
        >>>
        >>> # Save checkpoint
        >>> components = ModelComponents(main_model=model, bias_model=bias)
        >>> metadata = {"epochs": 50, "best_loss": 0.123}
        >>> manager.save_checkpoint(components, metadata)
        >>>
        >>> # Load checkpoint
        >>> loaded = manager.load_checkpoint()
        >>> model.load_state_dict(loaded['main_model_state_dict'])
    """

    def __init__(
        self,
        run_name: str,
        phase: str,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        strict_loading: bool = False,
        clean_compiled: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            run_name: Experiment run name
            phase: Training phase identifier (e.g., "phase1", "phase2")
            checkpoint_dir: Directory to save/load checkpoints
            strict_loading: Whether to use strict state dict loading
            clean_compiled: Whether to clean torch.compile artifacts
        """
        self.config = CheckpointConfig(
            run_name=run_name,
            phase=phase,
            checkpoint_dir=Path(checkpoint_dir),
            strict_loading=strict_loading,
            clean_compiled=clean_compiled
        )

    def save_checkpoint(
        self,
        components: ModelComponents,
        metadata: Optional[Dict[str, Any]] = None,
        suffix: str = "model",
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model checkpoint with components and metadata.

        Args:
            components: ModelComponents container with model parts
            metadata: Additional metadata to save
            suffix: Filename suffix (e.g., "bias_model", "full_model")
            config: Training configuration to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.config.get_checkpoint_path(suffix)

        # Build checkpoint dictionary
        checkpoint = components.to_dict(clean_compiled=self.config.clean_compiled)

        # Add metadata
        if metadata is None:
            metadata = {}

        checkpoint.update({
            'run_name': self.config.run_name,
            'phase': self.config.phase,
            'component_names': components.component_names,
            'metadata': metadata
        })

        if config is not None:
            checkpoint['config'] = config

        # Save to disk
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print(f"  Components: {', '.join(components.component_names)}")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        suffix: str = "model"
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Load checkpoint with automatic fallback strategies.

        Args:
            checkpoint_path: Path to checkpoint (if None, uses default path)
            suffix: Filename suffix if using default path

        Returns:
            Tuple of (checkpoint_dict, success_flag)
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.get_checkpoint_path(suffix)
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            return {}, False

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"✓ Checkpoint loaded: {checkpoint_path}")

            # Clean compiled state dicts if requested
            if self.config.clean_compiled:
                checkpoint = self._clean_checkpoint(checkpoint)

            return checkpoint, True

        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            return {}, False

    def load_into_components(
        self,
        components: ModelComponents,
        checkpoint_path: Optional[Union[str, Path]] = None,
        suffix: str = "model"
    ) -> Tuple[int, List[str]]:
        """
        Load checkpoint and populate ModelComponents with fallback strategies.

        Strategy:
        1. Try loading full_model_state_dict (preferred)
        2. Try loading individual components
        3. Try standard PyTorch Lightning state_dict

        Args:
            components: ModelComponents to populate
            checkpoint_path: Path to checkpoint
            suffix: Filename suffix if using default path

        Returns:
            Tuple of (num_components_loaded, list_of_loaded_components)
        """
        checkpoint, success = self.load_checkpoint(checkpoint_path, suffix)
        if not success:
            return 0, []

        loaded_components = []

        # Strategy 1: Full model (best for Lightning modules)
        if 'full_model_state_dict' in checkpoint and components.full_model is not None:
            try:
                missing, unexpected = components.full_model.load_state_dict(
                    checkpoint['full_model_state_dict'],
                    strict=self.config.strict_loading
                )
                loaded_components.append("full_model")
                print(f"  ✓ Full model loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
                return len(loaded_components), loaded_components
            except Exception as e:
                print(f"  ✗ Full model loading failed: {e}")

        # Strategy 2: Individual components
        component_map = {
            'bias_model_state_dict': ('bias_model', components.bias_model),
            'main_model_state_dict': ('main_model', components.main_model),
            'differential_head_state_dict': ('differential_head', components.differential_head)
        }

        for key, (name, module) in component_map.items():
            if key in checkpoint and module is not None:
                try:
                    missing, unexpected = module.load_state_dict(
                        checkpoint[key],
                        strict=self.config.strict_loading
                    )
                    loaded_components.append(name)
                    print(f"  ✓ {name} loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
                except Exception as e:
                    print(f"  ✗ {name} loading failed: {e}")

        # Strategy 3: Standard PyTorch Lightning format
        if not loaded_components and 'state_dict' in checkpoint and components.full_model is not None:
            try:
                missing, unexpected = components.full_model.load_state_dict(
                    checkpoint['state_dict'],
                    strict=self.config.strict_loading
                )
                loaded_components.append("full_model (lightning)")
                print(f"  ✓ Lightning checkpoint loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
            except Exception as e:
                print(f"  ✗ Lightning checkpoint loading failed: {e}")

        if loaded_components:
            print(f"✓ Successfully loaded {len(loaded_components)} component(s): {', '.join(loaded_components)}")
        else:
            print(f"✗ No components could be loaded from checkpoint")

        return len(loaded_components), loaded_components

    def _clean_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Clean torch.compile artifacts from all state dicts in checkpoint."""
        keys_to_clean = [
            'full_model_state_dict',
            'main_model_state_dict',
            'differential_head_state_dict',
            'bias_model_state_dict',
            'state_dict'
        ]

        for key in keys_to_clean:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint[key] = clean_compiled_state_dict(checkpoint[key])

        return checkpoint

    def get_metadata(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        suffix: str = "model"
    ) -> Dict[str, Any]:
        """
        Extract metadata from checkpoint without loading models.

        Args:
            checkpoint_path: Path to checkpoint
            suffix: Filename suffix if using default path

        Returns:
            Metadata dictionary
        """
        checkpoint, success = self.load_checkpoint(checkpoint_path, suffix)
        if not success:
            return {}

        metadata = {
            'run_name': checkpoint.get('run_name', 'unknown'),
            'phase': checkpoint.get('phase', 'unknown'),
            'component_names': checkpoint.get('component_names', []),
        }

        if 'metadata' in checkpoint:
            metadata.update(checkpoint['metadata'])

        if 'config' in checkpoint:
            metadata['config'] = checkpoint['config']

        return metadata


class PhaseCheckpointManager:
    """
    Higher-level manager for multi-phase training with automatic phase transitions.

    Example:
        >>> phase_manager = PhaseCheckpointManager(run_name="experiment1")
        >>>
        >>> # Phase 1: Save bias model
        >>> phase_manager.save_phase1(bias_model)
        >>>
        >>> # Phase 2: Load bias model and save main model
        >>> bias_model = phase_manager.load_bias_model_for_phase2()
        >>> phase_manager.save_phase2(main_model, bias_model)
        >>>
        >>> # Phase 3: Load all components
        >>> main_model, bias_model = phase_manager.load_models_for_phase3()
        >>> phase_manager.save_phase3(full_model, main_model, diff_head, bias_model)
    """

    def __init__(
        self,
        run_name: str,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        strict_loading: bool = False
    ):
        """
        Initialize phase checkpoint manager.

        Args:
            run_name: Experiment run name
            checkpoint_dir: Directory for checkpoints
            strict_loading: Whether to use strict loading
        """
        self.run_name = run_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.strict_loading = strict_loading

        # Create phase-specific managers
        self.phase1_manager = CheckpointManager(
            run_name=run_name,
            phase="phase1",
            checkpoint_dir=checkpoint_dir,
            strict_loading=strict_loading
        )
        self.phase2_manager = CheckpointManager(
            run_name=run_name,
            phase="phase2",
            checkpoint_dir=checkpoint_dir,
            strict_loading=strict_loading
        )
        self.phase3_manager = CheckpointManager(
            run_name=run_name,
            phase="phase3",
            checkpoint_dir=checkpoint_dir,
            strict_loading=strict_loading
        )

    # ========================================
    # Phase 1: Bias Model
    # ========================================

    def save_phase1(
        self,
        bias_model: torch.nn.Module,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save Phase 1 bias model checkpoint."""
        components = ModelComponents(bias_model=bias_model)
        return self.phase1_manager.save_checkpoint(
            components=components,
            metadata=metadata,
            suffix="bias_model",
            config=config
        )

    def load_phase1(
        self,
        bias_model: torch.nn.Module,
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """Load Phase 1 bias model."""
        components = ModelComponents(bias_model=bias_model)
        num_loaded, _ = self.phase1_manager.load_into_components(
            components=components,
            checkpoint_path=checkpoint_path,
            suffix="bias_model"
        )
        return num_loaded > 0

    # ========================================
    # Phase 2: Main Model + Bias Model
    # ========================================

    def save_phase2(
        self,
        main_model: torch.nn.Module,
        bias_model: Optional[torch.nn.Module] = None,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save Phase 2 checkpoint with main model and optional bias model."""
        components = ModelComponents(
            main_model=main_model,
            bias_model=bias_model
        )
        return self.phase2_manager.save_checkpoint(
            components=components,
            metadata=metadata,
            suffix="full_model",
            config=config
        )

    def load_phase2(
        self,
        main_model: torch.nn.Module,
        bias_model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, bool]:
        """
        Load Phase 2 models.

        Returns:
            Tuple of (main_model_loaded, bias_model_loaded)
        """
        components = ModelComponents(
            main_model=main_model,
            bias_model=bias_model
        )
        num_loaded, loaded_names = self.phase2_manager.load_into_components(
            components=components,
            checkpoint_path=checkpoint_path,
            suffix="full_model"
        )

        main_loaded = "main_model" in loaded_names
        bias_loaded = "bias_model" in loaded_names

        return main_loaded, bias_loaded

    def load_bias_model_for_phase2(
        self,
        bias_model: torch.nn.Module,
        prefer_phase1: bool = True
    ) -> bool:
        """
        Load bias model for Phase 2, preferring Phase 1 checkpoint.

        Args:
            bias_model: Bias model to populate
            prefer_phase1: Try Phase 1 checkpoint first

        Returns:
            Success flag
        """
        if prefer_phase1:
            # Try Phase 1 first
            if self.load_phase1(bias_model):
                print("✓ Loaded bias model from Phase 1 checkpoint")
                return True

        # Fall back to Phase 2 checkpoint
        components = ModelComponents(bias_model=bias_model)
        num_loaded, _ = self.phase2_manager.load_into_components(
            components=components,
            suffix="full_model"
        )

        if num_loaded > 0:
            print("✓ Loaded bias model from Phase 2 checkpoint")
            return True

        print("✗ Could not load bias model")
        return False

    # ========================================
    # Phase 3: Full Model + Components
    # ========================================

    def save_phase3(
        self,
        full_model: Union[torch.nn.Module, pl.LightningModule],
        main_model: Optional[torch.nn.Module] = None,
        differential_head: Optional[torch.nn.Module] = None,
        bias_model: Optional[torch.nn.Module] = None,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save Phase 3 final checkpoint with all components."""
        components = ModelComponents(
            full_model=full_model,
            main_model=main_model,
            differential_head=differential_head,
            bias_model=bias_model
        )
        return self.phase3_manager.save_checkpoint(
            components=components,
            metadata=metadata,
            suffix="final_model",
            config=config
        )

    def load_phase3(
        self,
        full_model: Optional[Union[torch.nn.Module, pl.LightningModule]] = None,
        main_model: Optional[torch.nn.Module] = None,
        differential_head: Optional[torch.nn.Module] = None,
        bias_model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Tuple[int, List[str]]:
        """
        Load Phase 3 checkpoint with all available components.

        Returns:
            Tuple of (num_loaded, list_of_loaded_components)
        """
        components = ModelComponents(
            full_model=full_model,
            main_model=main_model,
            differential_head=differential_head,
            bias_model=bias_model
        )
        return self.phase3_manager.load_into_components(
            components=components,
            checkpoint_path=checkpoint_path,
            suffix="final_model"
        )

    def load_models_for_phase3(
        self,
        main_model: torch.nn.Module,
        bias_model: Optional[torch.nn.Module] = None
    ) -> Tuple[bool, bool]:
        """
        Load main and bias models for Phase 3 initialization from Phase 2.

        Returns:
            Tuple of (main_model_loaded, bias_model_loaded)
        """
        return self.load_phase2(
            main_model=main_model,
            bias_model=bias_model
        )

    # ========================================
    # Utility Methods
    # ========================================

    def get_available_checkpoints(self) -> Dict[str, List[Path]]:
        """List all available checkpoints for this run."""
        checkpoints = {
            'phase1': [],
            'phase2': [],
            'phase3': []
        }

        if not self.checkpoint_dir.exists():
            return checkpoints

        for ckpt_file in self.checkpoint_dir.glob(f"{self.run_name}_*.ckpt"):
            filename = ckpt_file.stem
            if "_phase1_" in filename:
                checkpoints['phase1'].append(ckpt_file)
            elif "_phase2_" in filename:
                checkpoints['phase2'].append(ckpt_file)
            elif "_phase3_" in filename:
                checkpoints['phase3'].append(ckpt_file)

        return checkpoints

    def print_checkpoint_summary(self):
        """Print summary of available checkpoints."""
        checkpoints = self.get_available_checkpoints()

        print(f"\n{'='*60}")
        print(f"Checkpoint Summary: {self.run_name}")
        print(f"{'='*60}")

        for phase, ckpt_list in checkpoints.items():
            print(f"\n{phase.upper()}:")
            if ckpt_list:
                for ckpt_path in ckpt_list:
                    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {ckpt_path.name} ({size_mb:.2f} MB)")
            else:
                print("  (no checkpoints)")

        print(f"\n{'='*60}")
