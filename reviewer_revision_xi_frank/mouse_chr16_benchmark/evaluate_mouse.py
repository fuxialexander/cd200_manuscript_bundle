#!/usr/bin/env python3
"""Common evaluator with mouse stored-signal semantics, no changed metric equations."""
import importlib.util
import json
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / 'gm12878_benchmark/evaluate.py'
SPEC = importlib.util.spec_from_file_location('mouse_common_evaluator', SOURCE)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
shared.COUNT_TRANSFORM = 'ln1p_stored_atac_count'
_original = shared.evaluate


def evaluate(*args, **kwargs):
    result = _original(*args, **kwargs)
    directory = Path(args[0]); output = Path(args[2])
    manifest = json.loads((directory/'cache_manifest.json').read_text())
    if manifest.get('target_units') != 'unsmoothed_per_base_ATAC_counts_from_bundled_Zarr':
        raise ValueError('Target units not established')
    result['contract'].update(genome_build='mm10', target_units=manifest['target_units'],
        target_kind='stored_atac_signal_counts',
        count_definition='ln(1 + sum unsmoothed unnormalized stored ATAC counts in central1000)',
        upstream_provenance='Original Tn5 shift/endpoint provenance not independently established')
    (output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

shared.evaluate = evaluate
if __name__ == '__main__':
    shared.main()
