#!/usr/bin/env python3
"""Predict a fixed 2048bp Cd200 interval from three overlapping ChromBPNet windows.

This shared raw-count contrast is distinct from the historical curated Diff_scca
label unless the historical target transformation has been independently matched.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
LOCI = {'P2': (45356164,45357191), 'P1': (45383863,45384762),
        'promoter': (45408000,45409426)}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',choices=['pos','neg'],required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--bias-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    tm=load('contrast_native',REVISION/'gm12878_benchmark/train_chrombpnet.py')
    tm.bootstrap()
    tf=tm.setup_tf(1234)
    prep=load('contrast_reader',REVISION/'mouse_chr16_benchmark/prepare_mouse_cache.py')
    from pyfaidx import Fasta
    repo=REVISION.parent
    meta_path=args.run_dir/'metadata.json'
    meta=json.loads(meta_path.read_text())
    checkpoint=args.run_dir/'best.weights.h5'
    if digest(args.bias_model)!=meta['bias_model_sha256']:raise ValueError('Bias checkpoint differs')
    if 'chr16' in meta['training_chromosomes']+meta['validation_chromosomes']:
        raise ValueError('Cd200 chromosome was used in fitting')
    if meta['target_units']!='unsmoothed_per_base_ATAC_counts_from_bundled_Zarr':
        raise ValueError('Unexpected target units')
    cache_manifest=json.loads(Path(meta['args']['cache_dir']).joinpath('cache_manifest.json').read_text())
    if cache_manifest['sample']!=f'SCCA{args.state}.cd200_scc':raise ValueError('Wrong cell state')
    fasta_path=Path('/mnt/storage/get_data/mm10.fa')
    fasta=Fasta(str(fasta_path),as_raw=True,sequence_always_upper=True,rebuild=False)
    signal=prep.DenseZarrReader(repo/'data/zarr/cd200_scc_dense_new.zarr'/f'SCCA{args.state}.cd200_scc'/'chrs','<u2')
    windows=[];sequences=[];observed=[]
    for locus,(a,b) in LOCI.items():
        center=(a+b)//2
        left,right=center-1024,center+1024
        observed.append(signal.read('chr16',left,right).astype(np.float64))
        for offset in [0,524,1048]:
            out_start=left+offset
            sequence=prep.encode_sequence(fasta['chr16'][out_start-557:out_start+1557])
            assert len(sequence)==2114
            sequences.append(np.eye(5,dtype=np.float32)[sequence,:4])
            windows.append({'locus':locus,'chrom':'chr16','center':center,
                            'interval_start':left,'interval_end':right,
                            'output_start':out_start,'output_end':out_start+1000})
    x=np.stack(sequences)
    model=tm.fresh_model(tf,args.bias_model,meta['count_weight'],.001,meta['seed'])
    model.load_weights(checkpoint)
    logits,count=model(x,training=False)
    probabilities=tf.nn.softmax(logits,axis=-1).numpy().astype(np.float64)
    logcounts=count.numpy().reshape(-1).astype(np.float64)
    totals=np.expm1(logcounts)
    if not np.isfinite(totals).all() or (totals<0).any():raise ValueError('Invalid predicted totals')
    predicted=probabilities*totals[:,None]
    stitched=[];coverages=[]
    for locus in LOCI:
        profile=np.zeros(2048);coverage=np.zeros(2048,dtype=np.int64)
        for w,y in zip(windows,predicted):
            if w['locus']!=locus:continue
            start=w['output_start']-w['interval_start']
            profile[start:start+1000]+=y;coverage[start:start+1000]+=1
        assert (coverage>0).all()
        stitched.append(profile/coverage);coverages.append(coverage)
    # Recover the historical PRE-filter normalization from retained sum/log-sum
    # columns. Sums of the final filtered/deduplicated BED are NOT that denominator.
    bed_path=repo/'training/data/scc_all_peaks.deduplicated.bed'
    with bed_path.open() as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    lib={};recovery={}
    for state in ['pos','neg']:
        counts=np.array([float(r[f'SumValue_scca_{state}']) for r in rows])
        logs=np.array([float(r[f'logSumValue_scca_{state}']) for r in rows])
        keep=(counts>0)&(logs>0)
        recovered=counts[keep]*1e5/np.expm1(logs[keep]*np.log(10))
        denominator=float(np.median(recovered))
        assert np.allclose(recovered,denominator,atol=.01,rtol=1e-10)
        lib[state]=denominator
        recovery[state]={'n_rows':int(keep.sum()),'maximum_absolute_residual':float(np.max(np.abs(recovered-denominator))),
                         'final_BED_sum_for_comparison_only':float(counts.sum())}
    diff=np.array([float(r['Diff_scca']) for r in rows])
    check=np.array([float(r['logSumValue_scca_pos'])-float(r['logSumValue_scca_neg']) for r in rows])
    assert np.allclose(diff,check,atol=1e-12,rtol=0)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'state_predictions.npz',loci=np.array(list(LOCI)),
        observed_profiles=np.stack(observed),predicted_profiles=np.stack(stitched),
        coverage=np.stack(coverages),window_logcounts=logcounts,window_profiles=probabilities,
        window_input_onehot=x)
    with (args.output/'windows.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]),delimiter='\t')
        writer.writeheader();writer.writerows(windows)
    report={'state':args.state,'sample':cache_manifest['sample'],'loci_order':list(LOCI),
        'interval_length':2048,'input_length':2114,'output_length':1000,
        'count_transform':'ln1p(stored ATAC counts)','profile_reconstruction':'softmax(logits)*expm1(predicted_logcount)',
        'tiling':'three1000bpwindows starting at interval_start+[0,524,1048]; average overlapping per-base counts, sum each base once',
        'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
        'bias_checkpoint':str(args.bias_model),'bias_checkpoint_sha256':digest(args.bias_model),
        'training_metadata_sha256':digest(meta_path),'libraries':lib,'library_recovery_audit':recovery,
        'contrast_definition':'log10(pos_sum / lib_pos *1e5+1) - log10(neg_sum / lib_neg *1e5+1)',
        'contrast_limit':'Historical transform recovered from retained sum/log-sum columns. Current stored signal sums and centered windows can differ from historical BED counts/windows; preserve observed counts and archived labels separately.',
        'library_source':str(bed_path),'library_source_sha256':digest(bed_path),
        'source_sha256':digest(__file__),'signal_source_chunks':signal.sources,
        'reference_fasta':str(fasta_path),'reference_fai_sha256':digest(str(fasta_path)+'.fai'),
        'upstream_target_limitation':meta.get('target_provenance_limitation'),
        'observed_totals':np.stack(observed).sum(-1).tolist(),
        'predicted_totals':np.stack(stitched).sum(-1).tolist()}
    (args.output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    fasta.close()
    print(json.dumps({'state':args.state,'observed_totals':report['observed_totals'],
                      'predicted_totals':report['predicted_totals']}))

if __name__=='__main__':main()
