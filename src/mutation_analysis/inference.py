#!/usr/bin/env python3
"""
Handles model loading and inference.
"""

import os
import torch
from typing import Dict, Optional

from configs.config_utils import load_env_config

class ModelInference:
    """
    Handles loading and inference for Phase 2 and Phase 3 ATAC/differential prediction models.
    """

    def __init__(self, checkpoint_path: str, model_type: str = 'phase3',
                 scoring_mode: str = 'auto', device: Optional[str] = None):
        """
        Initialize model inference.

        Args:
            checkpoint_path: Path to model checkpoint
            model_type: Type of model ('phase2' or 'phase3')
            scoring_mode: Scoring mode ('auto', 'regression', 'count', 'differential')
            device: Device to run on (None for auto-selection)
        """
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.model_type = model_type.lower()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if self.model_type not in ['phase2', 'phase3']:
            raise ValueError("model_type must be 'phase2' or 'phase3'")

        if scoring_mode == 'auto':
            self.scoring_mode = 'differential' if self.model_type == 'phase3' else 'regression'
        else:
            self.scoring_mode = scoring_mode

        self.config = load_env_config('configs/env.yaml')
        self.model = None
        self._load_model()

        print(f"✓ Model loaded successfully")
        print(f"  - Type: {self.model_type.upper()}")
        print(f"  - Device: {self.device}")

    def _load_model(self):
        """Load the appropriate model based on type"""
        if self.model_type == 'phase2':
            self._load_phase2_model()
        else:
            self._load_phase3_model()

    def _load_phase2_model(self):
        # Placeholder for Phase 2 model loading logic
        from atac_prediction.dna_atac_lightning_module import DNAATACPredictionLightningModule
        # This is a simplified instantiation. A real one would load state_dict.
        self.model = DNAATACPredictionLightningModule(
            motif_kernels=torch.randn(746, 4, 21), num_atac_samples=1
        ).model.to(self.device).eval()
        
        # Load state dict
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = self._clean_checkpoint(checkpoint)
        # Adjust keys if needed
        state_dict = {k.replace('model.main_model.', ''): v for k, v in state_dict.items() if 'main_model' in k}
        self.model.load_state_dict(state_dict, strict=False)


    def _load_phase3_model(self):
        """Load Phase 3 differential model"""
        from atac_diff_prediction.models import create_phase3_model
        from atac_diff_prediction.lightning_modules import Phase3LightningModule
        from atac_diff_prediction.bias_model import BiasModel
        from atac_prediction.dna_atac_lightning_module import DNAATACPredictionLightningModule
        from atac_prediction.train_atac import load_motifs
        from configs.config_utils import ConfigManager

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = self._clean_checkpoint(checkpoint)

        try:
            config_manager = ConfigManager("train_atac_phased")
            model_config = config_manager.load_config("configs/train_atac_phased.yaml")
        except Exception:
            model_config = self._get_default_config()

        args_obj = self._create_args_object()
        try:
            padded_kernels, _ = load_motifs(model_config, args_obj)
        except Exception:
            padded_kernels = torch.randn(746, 4, 21)

        base_model = DNAATACPredictionLightningModule(
            motif_kernels=padded_kernels, num_atac_samples=1
        ).model

        bias_model = BiasModel(num_atac_samples=1, num_filters=16, filter_size=51, use_learnable_kernels=True)
        
        phase3_nn_model = create_phase3_model(base_model, bias_model, model_config)
        
        lightning_module = Phase3LightningModule(model=phase3_nn_model, celltype_ids=['DZ'])
        
        lightning_module.load_state_dict(state_dict, strict=False)
        
        self.model = lightning_module.model.to(self.device).eval()

    def _clean_checkpoint(self, checkpoint: Dict) -> Dict:
        """Clean checkpoint state dict from torch.compile prefixes"""
        if 'state_dict' in checkpoint:
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
                return new_state_dict
            return state_dict
        elif 'full_model_state_dict' in checkpoint:
            return checkpoint['full_model_state_dict']
        # Add other potential checkpoint structures from the old file
        elif 'main_model_state_dict' in checkpoint and 'bias_model_state_dict' in checkpoint:
            # Combine separate state dicts
            combined_state_dict = {}
            base_state = checkpoint['main_model_state_dict']
            bias_state = checkpoint['bias_model_state_dict']

            # Add base model under main_model prefix
            for key, value in base_state.items():
                if key.startswith('model.'):
                    new_key = f"model.main_model.{key[6:]}"
                else:
                    new_key = f"model.main_model.{key}"
                combined_state_dict[new_key] = value

            # Add bias model under bias_model prefix
            for key, value in bias_state.items():
                if key.startswith('model.'):
                    new_key = f"model.bias_model.{key[6:]}"
                else:
                    new_key = f"model.bias_model.{key}"
                combined_state_dict[new_key] = value

            return combined_state_dict
        return checkpoint # Fallback

    def predict(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run model prediction on sequence."""
        with torch.no_grad():
            sequence = sequence.to(self.device)
            outputs = self.model(sequence)
            if isinstance(outputs, dict):
                return {key: value.cpu() for key, value in outputs.items()}
            else:
                return {'regression': outputs.cpu()}

    def _get_default_config(self) -> Dict:
        """Get default model configuration"""
        return {
            'model': {
                'num_motifs': 746, 'motif_width': 21, 'num_residual_blocks': 4,
                'num_filters': 128, 'dilations': [1, 2, 4, 8]
            },
            'bias_model': {
                'use_learnable_kernels': False, 'tn5_motifs_path': "artifacts/bias.meme",
                'freeze_motif_layer': True
            }
        }
    
    def _create_args_object(self):
        """Create arguments object for motif loading"""
        paths = self.config.get('paths', {}) if self.config else {}
        return type('Args', (), {
            'tn5_motifs_path': "artifacts/bias.meme",
            'simple_motifs_path': 'artifacts/motifs_with_rc_aligned.pt',
            'aligned_motifs_path': 'artifacts/motifs_with_rc_aligned.pt',
            'pwm_path': paths.get('motifs_file', 'artifacts/data/hocomoco/H13CORE_pwms.txt'),
            'motif_filter_prefix': None
        })()
