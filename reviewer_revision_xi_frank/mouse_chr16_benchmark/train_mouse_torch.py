#!/usr/bin/env python3
"""Reuse the completed human benchmark trainer with truthful mouse-target metadata.

Architecture, optimization, augmentation and checkpoint selection are unchanged.
The mouse cache holds unnormalized/unsmoothed stored signal, not verified Tn5 ends.
"""
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / 'gm12878_benchmark/train_torch.py'
SPEC = importlib.util.spec_from_file_location('mouse_shared_torch', SOURCE)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
UNITS = 'unsmoothed_per_base_ATAC_counts_from_bundled_Zarr'
COUNT_TRANSFORM = 'ln1p_stored_atac_count'
_original_provenance = shared.provenance
_original_save = shared.save_json


def provenance(args, train, valid, count_weight, validation_indices):
    meta = _original_provenance(args, train, valid, count_weight, validation_indices)
    manifest = json.loads((args.cache_dir / 'cache_manifest.json').read_text())
    if manifest.get('target_units') != UNITS or manifest.get('assembly') != 'mm10':
        raise ValueError('Expected explicitly labeled mouse stored-signal cache')
    if set(meta['training_chromosomes']) & {'chr16','chr8','chr9','chr14'}:
        raise ValueError('Mouse holdout chromosome in training')
    if set(meta['validation_chromosomes']) != {'chr14'}:
        raise ValueError('Unexpected mouse validation chromosomes')
    meta.update(target_units=UNITS, target_kind='stored_atac_signal_counts', genome_build='mm10',
        count_target='ln(1 + sum stored unsmoothed ATAC signal in central1000)',
        count_transform=COUNT_TRANSFORM,
        modl_adaptation='Original fixed-motif architecture; center1000 profile crop and common stored-signal multinomial/logcount objectives; single stage; no historical weights.',
        profile_loss='-sum(stored target counts * log(unit-sum profile)), per-window mean; multinomial constant omitted',
        upstream_signal_provenance='Tn5 shift/endpoint and sample-merge provenance not fully verified; no library normalization or smoothing in this benchmark',
        mouse_adapter_sha256=shared.sha256(__file__))
    return meta


def save_json(path, value):
    if Path(path).name == 'prediction_metadata.json':
        value.update(genome_build='mm10', target_units=UNITS, target_kind='stored_atac_signal_counts',
            input_length=2114, output_length=1000, count_transform=COUNT_TRANSFORM,
            prediction_counts_units='predicted ln(1 + central1000 stored ATAC signal count)',
            mouse_adapter_sha256=shared.sha256(__file__))
    _original_save(path, value)


shared.provenance = provenance
shared.save_json = save_json
if __name__ == '__main__':
    args = shared.parse_args()
    shared.setup_seed(args.seed)
    shared.predict(args) if args.predict else shared.fit(args)
