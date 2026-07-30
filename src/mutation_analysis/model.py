"""
Model loading utilities for mutation analysis.

Provides simple function-based interface to load PyTorch Lightning
checkpoints and extract model metadata.

Integrates with the unified CheckpointManager for consistent checkpoint handling.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import torch
import torch.nn as nn
from checkpoint_manager import CheckpointManager, ModelComponents


def load_model(
    checkpoint_path: str,
    device: Optional[str] = None,
    config_file: Optional[str] = None,
    **kwargs
) -> Tuple[nn.Module, List[str]]:
    """
    Load model from checkpoint and extract output heads.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt or .pt)
        device: Device to load model on (auto-detected if None)
        config_file: Optional config file for Phase 3 models (e.g., 'configs/env_dlbcl.yml')
        **kwargs: Additional arguments to pass to model constructor
                 (e.g., for overriding hyperparameters)
                 For Phase 3: simple_motifs_path, tn5_motifs_path, phase_config_file

    Returns:
        model: Loaded PyTorch model (original, not wrapped)
        output_heads: List of output head names (e.g., ['count', 'regression'])

    Raises:
        ValueError: If checkpoint not found or invalid format
        RuntimeError: If output_heads cannot be determined

    Example:
        >>> # Standard ATAC model
        >>> model, heads = load_model("model.ckpt")
        >>> print(heads)
        ['count', 'regression']

        >>> # Phase 3 differential model
        >>> model, heads = load_model("phase3.ckpt", config_file='configs/env_dlbcl.yml')
        >>> print(heads)
        ['count', 'regression', 'differential']
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")

    # Determine checkpoint format
    # Check if it's a Lightning or CheckpointManager format
    has_model_state = (
        'state_dict' in checkpoint or
        'base_model_state_dict' in checkpoint or
        'main_model_state_dict' in checkpoint or
        'full_model_state_dict' in checkpoint
    )

    if has_model_state:
        # Lightning or CheckpointManager checkpoint
        model, output_heads = _load_lightning_checkpoint(
            checkpoint_path, checkpoint, device, config_file, **kwargs
        )
    else:
        # Unknown format
        raise ValueError(
            f"Unknown checkpoint format. Expected Lightning or CheckpointManager format. "
            f"Found keys: {list(checkpoint.keys())}"
        )

    # Set model to eval mode
    model.eval()

    return model, output_heads


def _load_lightning_checkpoint(
    checkpoint_path: str,
    checkpoint: Dict[str, Any],
    device: str,
    config_file: Optional[str] = None,
    **kwargs
) -> Tuple[nn.Module, List[str]]:
    """
    Load Lightning checkpoint and extract model/heads.

    Args:
        checkpoint_path: Path to checkpoint file
        checkpoint: Loaded checkpoint dictionary
        device: Target device
        config_file: Optional config file for Phase 3 models
        **kwargs: Additional model constructor arguments

    Returns:
        model: Unwrapped model (not Lightning module)
        output_heads: List of output head names
    """
    # Check for Phase 3 checkpoint structure first (before state dict detection).
    # Some Phase 2 checkpoints also carry bias-model metadata, so only use
    # signals that uniquely identify a full Phase 3 checkpoint.
    is_phase3_checkpoint = (
        'differential_head_state_dict' in checkpoint or
        'main_model_state_dict' in checkpoint or
        'full_model_state_dict' in checkpoint or
        str(checkpoint.get('phase', '')).lower() == 'phase3'
    )
    
    # Handle different checkpoint formats
    state_dict = None
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'base_model_state_dict' in checkpoint:
        state_dict = checkpoint['base_model_state_dict']
    elif 'main_model_state_dict' in checkpoint:
        state_dict = checkpoint['main_model_state_dict']
    elif 'full_model_state_dict' in checkpoint:
        state_dict = checkpoint['full_model_state_dict']

    if state_dict is None:
        raise ValueError(
            f"Cannot find model state dict in checkpoint. "
            f"Found keys: {list(checkpoint.keys())}"
        )

    # Auto-detect model type from state dict keys
    # If checkpoint structure indicates Phase 3, use that (more reliable)
    if is_phase3_checkpoint:
        model_type = 'atac_phase3'
    else:
        model_type = _detect_model_type(state_dict)

    # Load Lightning module based on detected type
    if model_type == 'atac':
        lightning_model = _load_atac_model(checkpoint_path, device, **kwargs)
        # Extract the actual model (unwrap from Lightning module)
        model = lightning_model.model
        # Extract output heads from the model
        output_heads = _extract_output_heads(model, model_type)
    elif model_type == 'atac_phase3':
        # Phase 3 loader returns both model and heads
        lightning_model, output_heads = _load_atac_phase3_model(checkpoint_path, device, config_file, **kwargs)
        # Extract the actual model (unwrap from Lightning module)
        model = lightning_model.model
    elif model_type == 'chip':
        lightning_model = _load_chip_model(checkpoint_path, device, **kwargs)
        # Extract the actual model (unwrap from Lightning module)
        model = lightning_model.model
        # Extract output heads from the model
        output_heads = _extract_output_heads(model, model_type)
    else:
        raise RuntimeError(
            f"Unknown model type. Could not detect model architecture from "
            f"checkpoint state dict keys."
        )

    return model, output_heads


def _detect_model_type(state_dict: Dict[str, Any]) -> str:
    """
    Detect model type from state dict keys.

    Args:
        state_dict: Model state dictionary

    Returns:
        Model type string: 'atac', 'atac_phase3', 'chip', or 'unknown'
    """
    keys = set(state_dict.keys())

    # Check for Phase 3 differential model (has differential_head)
    if any('differential_head' in k for k in keys):
        return 'atac_phase3'

    # Check for ATAC-specific keys
    if any('count_head' in k for k in keys):
        return 'atac'

    # Check for ChIP-specific patterns (if distinguishable)
    # For now, assume regression_head without count_head is ChIP
    if any('regression_head' in k for k in keys):
        if not any('count_head' in k for k in keys):
            return 'chip'
        return 'atac'  # Has both regression and count

    return 'unknown'


def _load_atac_model(
    checkpoint_path: str,
    device: str,
    **kwargs
) -> nn.Module:
    """
    Load ATAC prediction Lightning module.

    Uses CheckpointManager for consistent checkpoint handling with
    automatic cleaning of compiled model artifacts.

    Args:
        checkpoint_path: Path to checkpoint
        device: Target device
        **kwargs: Override hyperparameters

    Returns:
        Lightning module with loaded weights
    """
    from atac_prediction.dna_atac_lightning_module import DNAATACPredictionLightningModule

    load_kwargs = dict(kwargs)
    simple_motifs_path = load_kwargs.pop('simple_motifs_path', None)
    if simple_motifs_path and 'motif_kernels' not in load_kwargs:
        motif_data = torch.load(simple_motifs_path, map_location='cpu', weights_only=False)
        load_kwargs['motif_kernels'] = motif_data['motif_kernels'].detach().clone()

    # Use CheckpointManager for consistent handling
    # Extract run_name and phase from checkpoint path if available
    checkpoint_file = os.path.basename(checkpoint_path)
    parts = checkpoint_file.replace('.ckpt', '').split('_')

    # Try to parse phase from filename (e.g., "phase2")
    phase = 'inference'  # default
    for part in parts:
        if 'phase' in part.lower():
            phase = part
            break

    # Create checkpoint manager (run_name doesn't matter for loading)
    manager = CheckpointManager(
        run_name='inference',
        phase=phase,
        checkpoint_dir=os.path.dirname(checkpoint_path) or 'checkpoints',
        clean_compiled=True  # Automatically clean _orig_mod artifacts
    )

    # Load checkpoint with automatic cleaning
    checkpoint, success = manager.load_checkpoint(checkpoint_path)

    if not success:
        raise ValueError(f"Failed to load checkpoint: {checkpoint_path}")

    # Ensure checkpoint has 'state_dict' key for Lightning loading
    # Handle different checkpoint formats
    if 'state_dict' not in checkpoint:
        if 'base_model_state_dict' in checkpoint:
            checkpoint['state_dict'] = {
                k if k.startswith('model.') else f'model.{k}': v
                for k, v in checkpoint['base_model_state_dict'].items()
            }
        elif 'main_model_state_dict' in checkpoint:
            checkpoint['state_dict'] = checkpoint['main_model_state_dict']
        elif 'full_model_state_dict' in checkpoint:
            checkpoint['state_dict'] = checkpoint['full_model_state_dict']
        else:
            raise ValueError(
                f"Cannot find model state dict in checkpoint. "
                f"Found keys: {list(checkpoint.keys())}"
            )

    # Add Lightning metadata if missing (for CheckpointManager checkpoints)
    if 'pytorch-lightning_version' not in checkpoint:
        checkpoint['pytorch-lightning_version'] = '2.0.0'
    if 'epoch' not in checkpoint:
        checkpoint['epoch'] = 0
    if 'global_step' not in checkpoint:
        checkpoint['global_step'] = 0

    # Save cleaned checkpoint temporarily if needed
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
        tmp_path = tmp.name
        torch.save(checkpoint, tmp_path)

    try:
        # Load from cleaned checkpoint
        lightning_model = DNAATACPredictionLightningModule.load_from_checkpoint(
            tmp_path,
            map_location=device,
            **load_kwargs
        )
    except Exception as e:
        # Catch all exceptions to handle motif-related errors
        error_msg = str(e)
        if ('motif' in error_msg.lower() or 'NoneType' in error_msg) and simple_motifs_path is None:
            print(f"  Model initialization failed, retrying with default simple_motifs_path...")
            default_motifs_path = "artifacts/motifs_with_rc_aligned.pt"
            if os.path.exists(default_motifs_path):
                motif_data = torch.load(default_motifs_path, map_location='cpu', weights_only=False)
                retry_kwargs = dict(load_kwargs)
                retry_kwargs['motif_kernels'] = motif_data['motif_kernels'].detach().clone()
                lightning_model = DNAATACPredictionLightningModule.load_from_checkpoint(
                    tmp_path,
                    map_location=device,
                    **retry_kwargs
                )
            else:
                raise ValueError(
                    f"Model requires simple_motifs_path but {default_motifs_path} not found. "
                    f"Please provide simple_motifs_path as a kwarg."
                ) from e
        else:
            raise
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return lightning_model


def _load_atac_phase3_model(
    checkpoint_path: str,
    device: str,
    config_file: Optional[str] = None,
    **kwargs
) -> Tuple[nn.Module, List[str]]:
    """
    Load ATAC Phase 3 differential prediction model using Phase3ModelLoader.

    Args:
        checkpoint_path: Path to checkpoint
        device: Target device
        config_file: Path to environment config file (required for Phase 3)
        **kwargs: Additional arguments (simple_motifs_path, tn5_motifs_path, etc.)

    Returns:
        Tuple of (Lightning module with Phase3 model, list of output heads)

    Raises:
        ValueError: If config_file not provided for Phase 3 model
    """
    from .phase3_loader import Phase3ModelLoader

    # Default config file if not provided
    if config_file is None:
        config_file = kwargs.get('config_file', 'configs/env_dlbcl.yml')
        print(f"Using default Phase 3 config file: {config_file}")

    # Create loader
    loader = Phase3ModelLoader(
        config_file=config_file,
        simple_motifs_path=kwargs.get('simple_motifs_path', 'artifacts/motifs_with_rc_aligned.pt'),
        tn5_motifs_path=kwargs.get('tn5_motifs_path', 'artifacts/bias.meme'),
        phase_config_file=kwargs.get('phase_config_file', 'configs/train_atac_phased.yaml')
    )

    # Load model and return both model AND heads
    lightning_model, output_heads = loader.load_from_checkpoint(checkpoint_path, device)

    return lightning_model, output_heads


def _load_chip_model(
    checkpoint_path: str,
    device: str,
    **kwargs
) -> nn.Module:
    """
    Load ChIP-seq prediction Lightning module.

    Args:
        checkpoint_path: Path to checkpoint
        device: Target device
        **kwargs: Override hyperparameters

    Returns:
        Lightning module with loaded weights
    """
    from chip_prediction.lightning_module import GenomicPredictionLightningModule

    # Load from checkpoint
    lightning_model = GenomicPredictionLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        **kwargs
    )

    return lightning_model


def _extract_output_heads(model: nn.Module, model_type: str) -> List[str]:
    """
    Extract output head names from model.

    Args:
        model: PyTorch model
        model_type: Type of model ('atac' or 'chip')

    Returns:
        List of output head names
    """
    output_heads = []

    # Common heads to check for
    possible_heads = [
        'regression',
        'count',
        'differential',
        'fc',
        'mean',
        'shape',
        'chromosome',
    ]

    # Check which heads exist in the model
    for head_name in possible_heads:
        head_attr = f'{head_name}_head'
        if hasattr(model, head_attr):
            # Verify it's not None and is a module
            head = getattr(model, head_attr)
            if head is not None and isinstance(head, nn.Module):
                output_heads.append(head_name)

    # If no heads found, try to infer from model type
    if not output_heads:
        if model_type == 'atac':
            # ATAC models typically have regression and count
            output_heads = ['regression', 'count']
        elif model_type == 'chip':
            # ChIP models typically have regression
            output_heads = ['regression']

    if not output_heads:
        raise RuntimeError(
            "Could not determine output heads from model. "
            "Model may have non-standard architecture."
        )

    return output_heads


def get_model_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Extract model metadata without fully loading the model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with model metadata:
        - model_type: Type of model ('atac' or 'chip')
        - output_heads: List of output head names
        - hyperparameters: Model hyperparameters
        - epoch: Training epoch
        - pytorch_version: PyTorch version used for training

    Example:
        >>> info = get_model_info("model.ckpt")
        >>> print(info['output_heads'])
        ['count', 'regression']
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint (map to CPU to avoid GPU memory usage)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    state_dict = None
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'base_model_state_dict' in checkpoint:
        # Phase2 checkpoint from CheckpointManager
        state_dict = checkpoint['base_model_state_dict']
    elif 'main_model_state_dict' in checkpoint:
        # Phase2/3 checkpoint with main model
        state_dict = checkpoint['main_model_state_dict']
    elif 'full_model_state_dict' in checkpoint:
        # Full model checkpoint
        state_dict = checkpoint['full_model_state_dict']

    if state_dict is None:
        raise ValueError(
            f"Invalid checkpoint format. Expected 'state_dict', 'base_model_state_dict', "
            f"'main_model_state_dict', or 'full_model_state_dict'. "
            f"Found keys: {list(checkpoint.keys())}"
        )

    # Detect model type
    model_type = _detect_model_type(state_dict)

    # Extract output heads from state dict keys
    output_heads = []
    for head_name in ['regression', 'count', 'differential', 'fc', 'mean', 'shape']:
        head_pattern = f'model.{head_name}_head'
        if any(head_pattern in k for k in state_dict.keys()):
            output_heads.append(head_name)

    # Extract other metadata
    info = {
        'model_type': model_type,
        'output_heads': output_heads,
        'hyperparameters': checkpoint.get('hyper_parameters', {}),
        'epoch': checkpoint.get('epoch', None),
        'pytorch_version': checkpoint.get('pytorch-lightning_version', 'unknown'),
    }

    return info