#!/usr/bin/env python3
"""Common held-out evaluator; no fitting, calibration, or test-set filtering.

Count predictions are ln(1+raw central-1000-bp insertion counts). Profile
predictions are nonnegative probabilities over those same 1000 bases. Native
ExplaiNN scalar predictions have no profile metric. CI resampling units are
chromosome-stratified genomic blocks, merging blocks when input windows overlap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import scipy
from pathlib import Path

import numpy as np
from scipy.special import rel_entr
from scipy.stats import spearmanr

TARGET_START, TARGET_END = 500, 1500
COUNT_TRANSFORM = 'ln1p_raw_insertion_count'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def region_order_hash(ids: list[str]) -> str:
    return hashlib.sha256(''.join(x + '\n' for x in ids).encode()).hexdigest()


def read_regions(path: Path) -> list[dict]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        required = {'region_id', 'chrom', 'center', 'is_peak'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: required columns {sorted(required)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty test region manifest')
    ids = [r['region_id'] for r in rows]
    if len(set(ids)) != len(ids) or any(not x or '\n' in x for x in ids):
        raise ValueError('region_id must be nonempty and unique')
    for r in rows:
        r['center'] = int(r['center'])
        if r['center'] < 1057 or not r['chrom']:
            raise ValueError('Invalid chromosome or center')
        v = str(r['is_peak']).lower()
        if v not in {'0', '1', 'false', 'true'}:
            raise ValueError('is_peak must be 0/1/false/true')
        r['is_peak'] = v in {'1', 'true'}
    return rows


def genomic_blocks(rows: list[dict], block_bp: int = 1_000_000):
    """Merge fixed bins connected by overlapping 2114-bp input windows."""
    if block_bp < 1:
        raise ValueError('block_bp must be positive')
    keys = [(r['chrom'], r['center'] // block_bp) for r in rows]
    unique = sorted(set(keys))
    parent = {k: k for k in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for chrom in sorted({r['chrom'] for r in rows}):
        indices = sorted((i for i, r in enumerate(rows) if r['chrom'] == chrom),
                         key=lambda i: rows[i]['center'])
        right, previous = -1, None
        for i in indices:
            left, end = rows[i]['center'] - 1057, rows[i]['center'] + 1057
            if left < right:
                parent[find(keys[i])] = find(previous)
            right = max(right, end)
            previous = keys[i]
    roots = sorted({find(k) for k in keys})
    lookup = {k: i for i, k in enumerate(roots)}
    return np.array([lookup[find(k)] for k in keys]), [k[0] for k in roots]


def bootstrap_weights(block_chroms: list[str], n_bootstrap: int, seed: int):
    if n_bootstrap < 2:
        raise ValueError('At least two bootstrap replicates are required')
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_bootstrap, len(block_chroms)), dtype=np.int32)
    labels = np.asarray(block_chroms)
    for chrom in sorted(set(block_chroms)):
        positions = np.flatnonzero(labels == chrom)
        weights[:, positions] = rng.multinomial(len(positions),
            np.full(len(positions), 1 / len(positions)), size=n_bootstrap)
    return weights


def weighted_count_metrics(stats):
    """Stats columns n,x,y,x²,y²,xy,(x-y)²; supports batched bootstrap."""
    n, sx, sy, sxx, syy, sxy, se = np.asarray(stats).T
    with np.errstate(invalid='ignore', divide='ignore'):
        vx, vy = sxx - sx * sx / n, syy - sy * sy / n
        corr = (sxy - sx * sy / n) / np.sqrt(vx * vy)
        corr = np.where((n >= 2) & (vx > 1e-12) & (vy > 1e-12), corr, np.nan)
        rmse = np.sqrt(np.maximum(se, 0) / n)
    return {'count_pearson_ln1p': np.clip(corr, -1, 1), 'count_rmse_ln1p': rmse}


def count_block_stats(target, prediction, mask, blocks, nblocks):
    x, y = target[mask], prediction[mask]
    columns = [np.ones(len(x)), x, y, x*x, y*y, x*y, (x-y)**2]
    return np.stack([np.bincount(blocks[mask], weights=v, minlength=nblocks)
                     for v in columns], axis=1)


def row_pearson(a, b):
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    numerator = (a*b).sum(axis=1)
    denominator = np.sqrt((a*a).sum(axis=1) * (b*b).sum(axis=1))
    out = np.full(len(a), np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 1e-15)
    return np.clip(out, -1, 1)


def profile_metrics(raw_target, probabilities, batch_size=2048):
    n = len(raw_target)
    result = {k: np.full(n, np.nan) for k in ['profile_js_divergence_nats',
              'profile_js_distance', 'profile_pearson_1bp', 'profile_pearson_20bp']}
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        y = np.asarray(raw_target[start:end], dtype=np.float64)
        p = np.asarray(probabilities[start:end], dtype=np.float64)
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('Profiles must be finite and nonnegative')
        if not np.allclose(p.sum(axis=1), 1, atol=1e-4, rtol=1e-5):
            raise ValueError('Each prediction profile must sum to one')
        totals = y.sum(axis=1)
        valid = totals > 0
        if not valid.any():
            continue
        q, pvalid = y[valid] / totals[valid, None], p[valid]
        # Only rounding correction after validating the probability contract.
        pvalid = pvalid / pvalid.sum(axis=1, keepdims=True)
        midpoint = .5 * (pvalid + q)
        divergence = .5 * (rel_entr(pvalid, midpoint).sum(axis=1)
                            + rel_entr(q, midpoint).sum(axis=1))
        pos = np.arange(start, end)[valid]
        result['profile_js_divergence_nats'][pos] = np.maximum(divergence, 0)
        result['profile_js_distance'][pos] = np.sqrt(np.maximum(divergence, 0))
        result['profile_pearson_1bp'][pos] = row_pearson(y[valid], pvalid)
        result['profile_pearson_20bp'][pos] = row_pearson(
            y[valid].reshape(-1, 50, 20).sum(axis=2),
            pvalid.reshape(-1, 50, 20).sum(axis=2))
    return result


def interval(samples):
    finite = np.asarray(samples)[np.isfinite(samples)]
    return {'lower': float(np.quantile(finite, .025)) if len(finite) else None,
            'upper': float(np.quantile(finite, .975)) if len(finite) else None,
            'valid_replicates': int(len(finite)),
            'undefined_replicates': int(len(samples) - len(finite))}


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def evaluate(data_dir: Path, predictions: dict[str, Path], output: Path,
             n_bootstrap=2000, seed=1729, block_bp=1_000_000):
    region_path, target_path = data_dir/'test.regions.tsv', data_dir/'test.y.npy'
    rows = read_regions(region_path)
    ids = [r['region_id'] for r in rows]
    order_hash, manifest_hash = region_order_hash(ids), sha256(region_path)
    raw = np.load(target_path, mmap_mode='r', allow_pickle=False)
    if raw.shape != (len(rows), 2000):
        raise ValueError(f'Target shape must be ({len(rows)},2000), got {raw.shape}')
    for start in range(0, len(raw), 4096):
        chunk = raw[start:start+4096]
        if not np.isfinite(chunk).all() or (chunk < 0).any():
            raise ValueError('Raw test insertion targets must be finite and nonnegative')
    raw = raw[:, TARGET_START:TARGET_END]
    totals = raw.sum(axis=1, dtype=np.float64)
    target = np.log1p(totals)
    blocks, block_chroms = genomic_blocks(rows, block_bp)
    weights = bootstrap_weights(block_chroms, n_bootstrap, seed)
    nblocks = len(block_chroms)
    chroms = np.array([r['chrom'] for r in rows])
    peaks = np.array([r['is_peak'] for r in rows])
    masks = {'all': np.ones(len(rows), dtype=bool), 'peaks': peaks, 'nonpeaks': ~peaks}
    masks.update({f'chrom:{c}': chroms == c for c in sorted(set(chroms))})
    result = {'contract': {'target_start': 500, 'target_end': 1500,
        'count_transform': COUNT_TRANSFORM, 'region_ids_sha256': order_hash,
        'model_input_length': 2114, 'model_input_half_width': 1057,
        'profile_definition': 'per-window shape; 20bp bins are nonoverlapping sums',
        'js_distance': 'sqrt natural-log Jensen-Shannon divergence, matching scipy.spatial.distance.jensenshannon',
        'js_divergence': 'natural-log Jensen-Shannon divergence, range [0,ln(2)]',
        'spearman_ci': 'point estimate only; block CI computed for Pearson/RMSE and mean profile metrics',
        'population': 'fixed supplied test regions; no filtering on target/prediction quality',
        'uncertainty': 'conditional on trained checkpoint; excludes training seed/assay uncertainty'},
        'bootstrap': {'replicates': n_bootstrap, 'seed': seed, 'block_bp': block_bp,
            'unit': 'genomic bins merged if 2114bp input windows overlap; resampled within chromosome',
            'blocks_per_chromosome': {c: block_chroms.count(c) for c in sorted(set(block_chroms))},
            'warning': 'Chromosomes with one block contribute no within-chromosome resampling variation'},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'evaluator_sha256': sha256(Path(__file__))},
        'sources': [{'path': str(region_path), 'sha256': manifest_hash},
                    {'path': str(target_path), 'sha256': sha256(target_path)}],
        'scopes': {}, 'paired_deltas': []}
    all_boot, all_profiles = {}, {}
    for name, directory in predictions.items():
        meta_path = directory/'prediction_metadata.json'
        meta = json.loads(meta_path.read_text())
        valid_identity = (meta.get('region_ids_sha256') == order_hash or
                          meta.get('test_regions_sha256') == manifest_hash)
        if (meta.get('region_ids_sha256') not in (None, order_hash) or
                meta.get('test_regions_sha256') not in (None, manifest_hash)):
            valid_identity = False
        if not valid_identity:
            raise ValueError(f'{name}: test region order/hash mismatch or missing identity')
        for key, expected in [('target_start',500), ('target_end',1500), ('count_transform',COUNT_TRANSFORM)]:
            if meta.get(key) != expected:
                raise ValueError(f'{name}: metadata {key} must be {expected!r}')
        cp = directory/'prediction_counts.npy'
        counts = np.load(cp, allow_pickle=False)
        if counts.shape != (len(rows),) or not np.isfinite(counts).all():
            raise ValueError(f'{name}: counts must be finite vector of length {len(rows)}')
        pp = directory/'prediction_profiles.npy'
        per_profile = None
        if pp.exists():
            profiles = np.load(pp, mmap_mode='r', allow_pickle=False)
            if profiles.shape != (len(rows),1000):
                raise ValueError(f'{name}: profile shape must be ({len(rows)},1000)')
            per_profile = profile_metrics(raw, profiles)
        all_profiles[name] = per_profile
        for path in [meta_path, cp] + ([pp] if pp.exists() else []):
            result['sources'].append({'model': name, 'path': str(path), 'sha256': sha256(path)})
        for scope, mask in masks.items():
            n = int(mask.sum())
            record = {'n_regions': n, 'zero_count_targets': int((totals[mask] == 0).sum()), 'metrics': {}}
            bs = count_block_stats(target, counts.astype(np.float64), mask, blocks, nblocks)
            point = weighted_count_metrics(bs.sum(axis=0))
            boot = weighted_count_metrics(weights @ bs)
            for metric, estimate in point.items():
                record['metrics'][metric] = {'value': float(estimate),
                    'n_defined': n if np.isfinite(estimate) else 0,
                    'n_undefined': 0 if np.isfinite(estimate) else n,
                    'ci95': interval(boot[metric])}
                all_boot[(scope,name,metric)] = boot[metric]
            spearman = (float(spearmanr(target[mask], counts[mask]).statistic)
                         if n > 1 and np.ptp(target[mask]) > 0 and np.ptp(counts[mask]) > 0 else np.nan)
            record['metrics']['count_spearman_ln1p'] = {'value': spearman,
                'n_defined': n if np.isfinite(spearman) else 0,
                'n_undefined': 0 if np.isfinite(spearman) else n,
                'ci95': None, 'reason': 'Point estimate only; no fixed-rank approximation used'}
            if per_profile is None:
                record['profile_status'] = 'N/A: checkpoint provides scalar counts only'
            else:
                record['profile_status'] = 'evaluated'
                for metric, values in per_profile.items():
                    finite = mask & np.isfinite(values)
                    valid_n = int(finite.sum())
                    sums = np.bincount(blocks[finite], weights=values[finite], minlength=nblocks)
                    ns = np.bincount(blocks[finite], minlength=nblocks)
                    with np.errstate(invalid='ignore',divide='ignore'):
                        sample_values = (weights @ sums) / (weights @ ns)
                    estimate = float(values[finite].mean()) if valid_n else np.nan
                    record['metrics'][metric] = {'value': estimate, 'n_defined': valid_n,
                        'n_undefined': n-valid_n, 'ci95': interval(sample_values),
                        'undefined_rule': 'zero observed total or (for correlation) constant target/prediction'}
                    all_boot[(scope,name,metric)] = sample_values
            result['scopes'].setdefault(scope,{})[name] = record
    for scope, models in result['scopes'].items():
        for a,b in itertools.combinations(models,2):
            common = models[a]['metrics'].keys() & models[b]['metrics'].keys()
            for metric in sorted(common):
                va,vb = models[a]['metrics'][metric]['value'],models[b]['metrics'][metric]['value']
                delta = float(va-vb)
                ci = None
                n_common = int(masks[scope].sum())
                keya,keyb = (scope,a,metric),(scope,b,metric)
                if metric.startswith('profile_'):
                    xa,xb = all_profiles[a][metric], all_profiles[b][metric]
                    common_mask = masks[scope] & np.isfinite(xa) & np.isfinite(xb)
                    n_common = int(common_mask.sum())
                    differences = xa[common_mask] - xb[common_mask]
                    delta = float(differences.mean()) if n_common else np.nan
                    sums = np.bincount(blocks[common_mask], weights=differences, minlength=nblocks)
                    ns = np.bincount(blocks[common_mask], minlength=nblocks)
                    with np.errstate(invalid='ignore',divide='ignore'):
                        ci = interval((weights @ sums) / (weights @ ns))
                elif keya in all_boot:
                    ci = interval(all_boot[keya]-all_boot[keyb])
                result['paired_deltas'].append({'scope':scope,'model_a':a,'model_b':b,
                    'metric':metric,'delta_a_minus_b':delta,'n_common_defined_regions':n_common,
                    'ci95':ci})
    result = clean_json(result)
    output.mkdir(parents=True, exist_ok=True)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'metrics.tsv').open('w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerow(['scope','model','metric','value','n_regions','n_defined','n_undefined','ci_lower','ci_upper'])
        for scope, models in result['scopes'].items():
            for name, record in models.items():
                for metric,m in record['metrics'].items():
                    ci=m['ci95'] or {}
                    writer.writerow([scope,name,metric,m['value'],record['n_regions'],m['n_defined'],m['n_undefined'],ci.get('lower'),ci.get('upper')])
    np.save(output/'region_block_ids.npy',blocks)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--prediction',action='append',required=True,help='NAME=DIRECTORY')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--block-bp',type=int,default=1_000_000)
    args=parser.parse_args()
    predictions={}
    for spec in args.prediction:
        name,path=spec.split('=',1)
        if not name or name in predictions:
            raise ValueError('Prediction names must be nonempty and unique')
        predictions[name]=Path(path)
    result=evaluate(args.data_dir,predictions,args.output,args.bootstrap,args.seed,args.block_bp)
    print(json.dumps({'evaluation':str(args.output/'evaluation.json'),
        'models':list(predictions),'n_regions':result['scopes']['all'][next(iter(predictions))]['n_regions']}))


if __name__=='__main__':
    main()
