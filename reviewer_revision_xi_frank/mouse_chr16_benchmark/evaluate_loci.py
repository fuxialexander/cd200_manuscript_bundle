#!/usr/bin/env python3
"""Evaluate the three plotted Cd200 loci without fitting or across-locus tests.

Input directory: locus_windows.tsv, observed_profiles.npy (N,1000),
target_metadata.json (genome_build=mm10, target_units, target_kind,
input_length=2114, output_length=1000). Prediction directories contain
prediction_counts.npy (N, ln1p total target signal), optional unit-sum
prediction_profiles.npy (N,1000), prediction_metadata.json with matching fields
and window_ids_sha256, region_ids_sha256 or locus_windows_sha256. Coordinates are BED half-open.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln, rel_entr, xlogy

LOCI = {
    'P1': ('chr16', 45383863, 45384762),
    'P2': ('chr16', 45356164, 45357191),
    'promoter': ('chr16', 45408000, 45409426),
}
INPUT_LENGTH, OUTPUT_LENGTH = 2114, 1000
COUNT_TRANSFORM = 'ln1p_stored_atac_count'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def window_order_hash(ids):
    return hashlib.sha256(''.join(x+'\n' for x in ids).encode()).hexdigest()


def planned_windows():
    """One standardized central window plus coverage tiles for wider loci."""
    rows=[]
    for name,(chrom,start,end) in LOCI.items():
        center=(start+end)//2
        starts=[('central',center-500)]
        if end-start>OUTPUT_LENGTH:
            starts += [('tile_left',start),('tile_right',end-OUTPUT_LENGTH)]
        for role,a in starts:
            rows.append({'window_id':f'{name}_{role}', 'locus':name,'chrom':chrom,
                         'output_start':a,'output_end':a+OUTPUT_LENGTH,'role':role})
    return rows


def write_window_manifest(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(planned_windows()[0]),delimiter='\t')
        writer.writeheader();writer.writerows(planned_windows())


def read_windows(path):
    with Path(path).open() as f:
        reader=csv.DictReader(f,delimiter='\t')
        required={'window_id','locus','chrom','output_start','output_end','role'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing locus window columns: {sorted(required)}')
        rows=list(reader)
    if not rows or len({r['window_id'] for r in rows})!=len(rows):
        raise ValueError('Window IDs must be unique and nonempty manifest required')
    for r in rows:
        r['output_start'],r['output_end']=int(r['output_start']),int(r['output_end'])
        if not r['window_id'] or r['locus'] not in LOCI:
            raise ValueError('Unknown locus or empty window ID')
        chrom,a,b=LOCI[r['locus']]
        if r['chrom']!=chrom or r['output_end']-r['output_start']!=OUTPUT_LENGTH:
            raise ValueError('Each output window must be 1000bp on chr16')
        if r['output_start']>=b or r['output_end']<=a:
            raise ValueError('Output window does not overlap its locus')
        if r['role']=='central' and r['output_start']!=(a+b)//2-500:
            raise ValueError('Central output window is not the frozen locus midpoint')
    for name in LOCI:
        if sum(r['locus']==name and r['role']=='central' for r in rows)!=1:
            raise ValueError(f'Exactly one central window required for {name}')
    return rows


def pearson(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    a,b=a-a.mean(),b-b.mean()
    norm=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return None if norm<=1e-15 else float(np.clip(np.dot(a,b)/norm,-1,1))


def bin_sum(a,width=20):
    return np.add.reduceat(np.asarray(a),np.arange(0,len(a),width))


def shape_metrics(observed, probabilities, target_kind):
    y,p=np.asarray(observed,dtype=np.float64),np.asarray(probabilities,dtype=np.float64)
    if y.ndim!=1 or p.shape!=y.shape or not np.isfinite(y).all() or (y<0).any():
        raise ValueError('Observed profile must be finite, nonnegative vector matching prediction')
    if not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-4,rtol=1e-5):
        raise ValueError('Predicted profile must be finite, nonnegative and sum to one')
    p=p/p.sum()  # Rounding correction only after unit-sum validation.
    total=float(y.sum())
    integer=bool(np.allclose(y,np.rint(y),rtol=0,atol=1e-6))
    likelihood_valid=(target_kind in {'raw_insertion_counts','stored_atac_signal_counts'} and integer)
    result={'observed_total_signal':total,'observed_nonzero_bases':int((y>0).sum()),
        'target_values_integer':integer,'multinomial_likelihood_valid':likelihood_valid,
        'loss_interpretation':('discrete multinomial count likelihood for stated stored counts; upstream Tn5 shift is not established here' if likelihood_valid else
          'target-weighted score; gamma extension of multinomial formula, not a discrete count likelihood'),
        'profile_bin20_count':int((len(y)+19)//20),'profile_bin20_final_width':len(y)%20 or 20}
    if total==0:
        result.update({'profile_nll_without_constant':0.,'profile_nll_with_constant':0.,
            'profile_cross_entropy_per_target_unit':None,'profile_js_divergence_nats':None,
            'profile_js_distance':None,'profile_pearson_1bp':None,'profile_pearson_20bp':None,
            'shape_status':'undefined: observed total is zero'})
        return result
    q=y/total
    q=q/q.sum()  # Align rounding-only probability normalization with prediction.
    nll=float(-xlogy(y,p).sum())
    constant=float(-gammaln(total+1)+gammaln(y+1).sum())
    midpoint=(p+q)/2
    js=float(.5*(rel_entr(p,midpoint).sum()+rel_entr(q,midpoint).sum()))
    result.update({'profile_nll_without_constant':nll if np.isfinite(nll) else None,
        'profile_nll_with_constant':nll+constant if np.isfinite(nll) else None,
        'profile_cross_entropy_per_target_unit':nll/total if np.isfinite(nll) else None,
        'nll_status':'finite' if np.isfinite(nll) else 'infinite: positive target at predicted zero probability',
        'profile_js_divergence_nats':max(js,0),'profile_js_distance':float(np.sqrt(max(js,0))),
        'profile_pearson_1bp':pearson(y,p),'profile_pearson_20bp':pearson(bin_sum(y),bin_sum(p)),
        'shape_status':'evaluated; undefined correlations retained as null'})
    return result


def stitch_interval(rows,profiles,locus,require_consistent=False):
    """Average each covered base once; never sum overlapping window totals."""
    chrom,start,end=LOCI[locus];n=end-start
    signal=np.zeros(n,dtype=np.float64);coverage=np.zeros(n,dtype=np.int64)
    first=np.full(n,np.nan)
    for row,profile in zip(rows,profiles,strict=True):
        if row['locus']!=locus:continue
        a,b=max(start,row['output_start']),min(end,row['output_end'])
        if b<=a:continue
        destination=slice(a-start,b-start);source=slice(a-row['output_start'],b-row['output_start'])
        values=np.asarray(profile[source],dtype=np.float64)
        existing=first[destination]
        if require_consistent:
            common=np.isfinite(existing)
            if not np.allclose(existing[common],values[common],rtol=1e-5,atol=1e-6):
                raise ValueError('Overlapping observed windows disagree on common genomic bases')
        existing[np.isnan(existing)]=values[np.isnan(existing)]
        signal[destination]+=values;coverage[destination]+=1
    if (coverage==0).any():
        return None,coverage
    return signal/coverage,coverage


def evaluate(data_dir,predictions,output):
    data_dir,output=Path(data_dir),Path(output)
    manifest=data_dir/'locus_windows.tsv';rows=read_windows(manifest)
    meta_path=data_dir/'target_metadata.json';target_meta=json.loads(meta_path.read_text())
    expected={'genome_build':'mm10','input_length':INPUT_LENGTH,'output_length':OUTPUT_LENGTH}
    for k,v in expected.items():
        if target_meta.get(k)!=v:raise ValueError(f'Target metadata {k} must be {v!r}')
    if not target_meta.get('target_units') or target_meta.get('target_kind') not in {'raw_insertion_counts','stored_atac_signal_counts','processed_signal'}:
        raise ValueError('target_units and target_kind (stored_atac_signal_counts, raw_insertion_counts or processed_signal) required')
    expected.update({k:target_meta[k] for k in ['target_units','target_kind']})
    observed_path=data_dir/'observed_profiles.npy';observed=np.load(observed_path,allow_pickle=False)
    if observed.shape!=(len(rows),OUTPUT_LENGTH) or not np.isfinite(observed).all() or (observed<0).any():
        raise ValueError('Observed profile matrix must be finite, nonnegative (N,1000)')
    order_hash=window_order_hash([r['window_id'] for r in rows]);manifest_hash=sha256(manifest)
    result={'contract':{**expected,'coordinate_convention':'0-based half-open BED',
        'window_ids_sha256':order_hash,'count_transform':COUNT_TRANSFORM,
        'loss_nll_without_constant':'-sum(y_i * ln(p_i)); no probability flooring',
        'loss_nll_with_constant':'-sum(y_i ln(p_i)) - gammaln(sum(y)+1) + sum(gammaln(y_i+1))',
        'profile_cross_entropy_per_target_unit':'NLL_without_constant / sum(y); undefined at zero total',
        'native_js_match':'JS distance = sqrt(natural-log JS divergence)',
        'bin20_policy':'nonoverlapping sums anchored at evaluated interval start; final shorter bin retained',
        'count_evaluation':'native scalar count loss per 1000bp window; full-interval count is reconstructed from stitched perbase predictions',
        'tiling':'perbase predicted mass = unit-sum profile * expm1(logcount); average overlapping-base estimates; sum each genomic base once',
        'inference_limit':'descriptive locus case studies; no across-three-loci count correlation or base-level significance test',
        'loci':{name:{'chrom':v[0],'start':v[1],'end':v[2],'length':v[2]-v[1]} for name,v in LOCI.items()}},
        'sources':[{'path':str(p),'sha256':sha256(p)} for p in [manifest,meta_path,observed_path,Path(__file__)]],
        'windows':{},'full_intervals':{}}
    full_targets={name:stitch_interval(rows,observed,name,True) for name in LOCI}
    for model,directory in predictions.items():
        directory=Path(directory);mp=directory/'prediction_metadata.json';metadata=json.loads(mp.read_text())
        if not (metadata.get('window_ids_sha256')==order_hash or metadata.get('region_ids_sha256')==order_hash or metadata.get('locus_windows_sha256')==manifest_hash):
            raise ValueError(f'{model}: window identity/order mismatch')
        if (metadata.get('window_ids_sha256') not in (None,order_hash) or
                metadata.get('region_ids_sha256') not in (None,order_hash) or
                metadata.get('locus_windows_sha256') not in (None,manifest_hash)):
            raise ValueError(f'{model}: contradictory window identity metadata')
        for k,v in {**expected,'count_transform':COUNT_TRANSFORM}.items():
            if metadata.get(k)!=v:raise ValueError(f'{model}: metadata {k} must be {v!r}')
        cp=directory/'prediction_counts.npy';counts=np.load(cp,allow_pickle=False)
        if counts.shape!=(len(rows),) or not np.isfinite(counts).all():raise ValueError('Count predictions must be a finite length-N vector')
        pp=directory/'prediction_profiles.npy';profiles=np.load(pp,allow_pickle=False) if pp.exists() else None
        if profiles is not None and profiles.shape!=observed.shape:raise ValueError('Prediction profile shape mismatch')
        paths=[mp,cp]+([pp] if pp.exists() else [])
        result['sources'] += [{'model':model,'path':str(p),'sha256':sha256(p)} for p in paths]
        for i,row in enumerate(rows):
            total=float(observed[i].sum(dtype=np.float64));truth=float(np.log1p(total))
            record={'locus':row['locus'],'role':row['role'],'chrom':row['chrom'],
                'output_start':row['output_start'],'output_end':row['output_end'],
                'observed_log1p_total_signal':truth,'predicted_log1p_total_signal':float(counts[i]),
                'count_log_mse':float((counts[i]-truth)**2),
                'count_loss_status':'native 1000bp scalar-head objective in stated target units'}
            if profiles is None:record['profile_status']='N/A: scalar-only model'
            else:record.update(shape_metrics(observed[i],profiles[i],target_meta['target_kind']))
            result['windows'].setdefault(row['window_id'],{})[model]=record
        for name in LOCI:
            y,coverage=full_targets[name]
            full={'coverage_complete':y is not None,'bases_covered':int((coverage>0).sum()),
                  'maximum_window_coverage':int(coverage.max()),
                  'count_loss_status':'not a native full-interval count head; derived from stitched 1000bp predictions'}
            if profiles is None:full['profile_status']='N/A: scalar-only model; overlapping scalar counts cannot reconstruct a spatial profile'
            elif y is None:full['profile_status']='N/A: output windows do not cover full original interval'
            elif any(counts[i]<0 for i,r in enumerate(rows) if r['locus']==name):
                full['profile_status']='N/A: negative inverse count estimate; no clipping applied for tiling'
            else:
                with np.errstate(over='ignore',invalid='ignore'):
                    mass=profiles*np.expm1(counts.astype(np.float64))[:,None]
                if not np.isfinite(mass).all():raise ValueError('Nonfinite reconstructed profile mass')
                signal,_=stitch_interval(rows,mass,name)
                predicted_total=float(signal.sum());observed_total=float(y.sum())
                full.update({'observed_total_signal':observed_total,'reconstructed_predicted_total_signal':predicted_total,
                    'reconstructed_count_log_mse':float((np.log1p(predicted_total)-np.log1p(observed_total))**2)})
                if predicted_total==0:full['profile_status']='undefined: reconstructed total prediction is zero'
                else:full.update(shape_metrics(y,signal/predicted_total,target_meta['target_kind']))
            result['full_intervals'].setdefault(name,{})[model]=full
    output.mkdir(parents=True,exist_ok=True)
    (output/'locus_evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'locus_metrics.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['scope','window_or_locus','model','metric','value'])
        for scope in ['windows','full_intervals']:
            for key,models in result[scope].items():
                for model,record in models.items():
                    for metric,value in record.items():
                        if isinstance(value,(int,float)) and not isinstance(value,bool):w.writerow([scope,key,model,metric,value])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-window-manifest',type=Path)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--prediction',action='append',default=[],help='MODEL=DIRECTORY')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.write_window_manifest:
        write_window_manifest(a.write_window_manifest)
        print(json.dumps({'window_manifest':str(a.write_window_manifest),'windows':len(planned_windows())}))
        return
    if not a.data_dir or not a.output or not a.prediction:p.error('--data-dir, --prediction and --output required')
    predictions={}
    for value in a.prediction:
        model,path=value.split('=',1)
        if not model or model in predictions:raise ValueError('Model names must be nonempty and unique')
        predictions[model]=Path(path)
    result=evaluate(a.data_dir,predictions,a.output)
    print(json.dumps({'evaluation':str(a.output/'locus_evaluation.json'),'windows':len(result['windows']),'models':list(predictions)}))


if __name__=='__main__':main()
