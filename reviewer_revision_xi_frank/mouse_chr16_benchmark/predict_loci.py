#!/usr/bin/env python3
"""Predict prespecified held-out locus windows with trained mouse benchmark models."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
BASE = Path(__file__).resolve().parent
UNITS = 'unsmoothed_per_base_ATAC_counts_from_bundled_Zarr'


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['modl','explainn','chrombpnet'],required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bias-model',type=Path)
    args=p.parse_args()
    shared_path=BASE.parent/'gm12878_benchmark'
    if args.model=='chrombpnet':
        tfm=load('locus_native',shared_path/'train_chrombpnet.py');tfm.bootstrap()
        tf=tfm.setup_tf(1234)
        meta=json.loads((args.run_dir/'metadata.json').read_text())
        if args.bias_model is None or digest(args.bias_model)!=meta['bias_model_sha256']:
            raise ValueError('Bias model missing or changed')
        model=tfm.fresh_model(tf,args.bias_model,meta['count_weight'],0.001,meta['seed'])
        checkpoint=args.run_dir/'best.weights.h5';model.load_weights(checkpoint)
        dataset=tfm.Cache(args.cache_dir,'loci');x,unused=dataset.batch(np.arange(len(dataset)))
        logits,count=model(x,training=False)
        counts=count.numpy()[:,0];profiles=tf.nn.softmax(logits,axis=-1).numpy()
    else:
        import torch
        tm=load('locus_torch',shared_path/'train_torch.py')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        checkpoint=args.run_dir/'best.pt'
        allowed=[np._core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32'))]
        with torch.serialization.safe_globals(allowed):saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['model']!=args.model:raise ValueError('Wrong checkpoint architecture')
        meta=saved['provenance']
        dataset=tm.CacheDataset(args.cache_dir,'loci',False)
        model=tm.make_model(args.model,BASE.parents[1]/'artifacts/motifs_with_rc_aligned.pt',saved['num_cnns']).cuda()
        model.load_state_dict(saved['state_dict'],strict=True);model.eval()
        x=torch.stack([dataset[i][0] for i in range(len(dataset))]).cuda()
        with torch.no_grad():count,profile=tm.predictions(model,args.model,x)
        counts=count.cpu().numpy();profiles=None if profile is None else profile.cpu().numpy()
    chroms={r['chrom'] for r in dataset.regions}
    if chroms!={'chr16'} or chroms & set(meta['training_chromosomes']+meta['validation_chromosomes']):
        raise ValueError('Locus chromosomes overlap model fitting')
    if meta.get('target_units')!=UNITS:raise ValueError('Model target units not established')
    if not np.isfinite(counts).all() or (profiles is not None and not np.isfinite(profiles).all()):
        raise FloatingPointError('Nonfinite locus predictions')
    windows=list(csv.DictReader((args.cache_dir/'locus_windows.tsv').open(),delimiter='\t'))
    ids=[r.get('window_id',r.get('region_id')) for r in windows]
    if ids!=[r['region_id'] for r in dataset.regions]:raise ValueError('Locus window order differs')
    args.output.mkdir(parents=True,exist_ok=False)
    np.save(args.output/'prediction_counts.npy',counts)
    if profiles is not None:np.save(args.output/'prediction_profiles.npy',profiles)
    metadata={'model':args.model,'checkpoint_sha256':digest(checkpoint),
        'target_units':UNITS,'target_kind':'stored_atac_signal_counts','genome_build':'mm10',
        'count_transform':'ln1p_stored_atac_count','input_length':2114,'output_length':1000,
        'region_ids_sha256':hashlib.sha256(''.join(i+'\n' for i in ids).encode()).hexdigest(),
        'locus_windows_sha256':digest(args.cache_dir/'locus_windows.tsv'),
        'source_sha256':digest(__file__),'n_windows':len(counts)}
    (args.output/'prediction_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'prediction':str(args.output),'windows':len(counts)}))

if __name__=='__main__':main()
