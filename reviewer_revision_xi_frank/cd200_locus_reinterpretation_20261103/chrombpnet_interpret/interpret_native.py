#!/usr/bin/env python3
"""Native ChromBPNet bias-free interpretation with preserved references.

export: load new mouse full checkpoint and export its bias-free component.
deepshap: exact Kundaje fork + official ChromBPNet target/projection functions.
deeplift: standalone Captum DeepLIFT Rescale, single preserved reference, after
strict native/PyTorch parity on all inputs and all shuffled references.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

BASE=Path(__file__).resolve().parent
REVIEW=BASE.parents[1]
HUMAN=REVIEW/'gm12878_benchmark'
DEFAULT_RUN=Path('/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/runs/chrombpnet_seed1234')
DEFAULT_CACHE=Path('/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/cache')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def package_paths():
    sys.path.insert(0,str(BASE/'runtime_deps'))


def export(args):
    package_paths()
    native=load('native_export',HUMAN/'train_chrombpnet.py');native.bootstrap()
    tf=native.setup_tf(args.seed)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from deeplift.dinuc_shuffle import dinuc_shuffle
    metadata=json.loads((args.run_dir/'metadata.json').read_text())
    if metadata.get('genome_build')!='mm10' or metadata.get('pilot'):
        raise ValueError('Expected complete mm10 benchmark checkpoint')
    if 'chr16' in metadata['training_chromosomes']+metadata['validation_chromosomes']:
        raise ValueError('Model fitting included chr16')
    bias=Path(metadata['args']['bias_model'])
    if digest(bias)!=metadata['bias_model_sha256']:raise ValueError('Bias component provenance mismatch')
    full=native.fresh_model(tf,bias,metadata['count_weight'],0.001,metadata['seed'])
    full.load_weights(args.run_dir/'best.weights.h5')
    component=full.get_layer('model_wo_bias')
    cache=native.Cache(args.cache_dir,'loci')
    indices=[i for i,r in enumerate(cache.regions) if r.get('role')=='central']
    if len(indices)!=3:raise ValueError('Exactly three frozen central locus windows required')
    inputs,_=cache.batch(indices)
    rows=[cache.regions[i] for i in indices]
    references=[]
    for i,x in enumerate(inputs):
        references.append(dinuc_shuffle(x,num_shufs=100,rng=np.random.RandomState(args.seed+i)))
    references=np.asarray(references,dtype=np.float32)
    for x,rs in zip(inputs,references):
        tokens=np.argmax(x,axis=-1)
        original=np.bincount(tokens[:-1]*4+tokens[1:],minlength=16)
        for r in rs:
            t=np.argmax(r,axis=-1)
            np.testing.assert_array_equal(np.bincount(t[:-1]*4+t[1:],minlength=16),original)
    args.output.mkdir(parents=True,exist_ok=False)
    component.save(args.output/'bias_free_model.h5',include_optimizer=False)
    weights={}
    for layer in component.layers:
        for j,value in enumerate(layer.get_weights()):weights[f'{layer.name}__{j}']=value
    np.savez_compressed(args.output/'native_weights.npz',**weights)
    np.savez_compressed(args.output/'inputs_references.npz',inputs=inputs,references=references)
    all_inputs=np.concatenate([inputs,references.reshape(-1,2114,4)])
    logits,counts=[],[]
    for begin in range(0,len(all_inputs),16):
        p,c=component(all_inputs[begin:begin+16],training=False)
        logits.append(p.numpy());counts.append(c.numpy()[:,0])
    logits,counts=np.concatenate(logits),np.concatenate(counts)
    np.savez_compressed(args.output/'native_predictions.npz',logits=logits,counts=counts)
    # Verify component/full recombination directly on the unmodified input loci.
    bias_model=next(m for m in full.layers if isinstance(m,tf.keras.Model) and m.name!='model_wo_bias')
    bp,bc=bias_model(inputs,training=False);fp,fc=full(inputs,training=False)
    cp,cc=component(inputs,training=False)
    rp=cp.numpy()+bp.numpy()
    rc=np.logaddexp(cc.numpy()[:,0],bc.numpy()[:,0])
    np.testing.assert_allclose(rp,fp.numpy(),rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(rc,fc.numpy()[:,0],rtol=1e-5,atol=1e-5)
    windows=[]
    for row in rows:
        center=int(row['center'])
        windows.append({**row,'input_start':center-1057,'input_end':center+1057,'output_start':center-500,'output_end':center+500})
    save(args.output/'export_metadata.json',{'checkpoint':str(args.run_dir/'best.weights.h5'),
        'checkpoint_sha256':digest(args.run_dir/'best.weights.h5'),'bias_sha256':digest(bias),
        'bias_free_model_sha256':digest(args.output/'bias_free_model.h5'),'weights_sha256':digest(args.output/'native_weights.npz'),
        'inputs_references_sha256':digest(args.output/'inputs_references.npz'),'seed':args.seed,
        'reference_policy':'100 dinucleotide shuffles per locus; numpy.RandomState(seed+locus_index); first20 subset preserved',
        'windows':windows,'all_native_forward_inputs':len(all_inputs),'component_recombination':'PASS',
        'tensorflow':tf.__version__,'tf32_enabled':False,'target_units':metadata['target_units'],'state':'SCCApos',
        'no_negative_state_difference_claimed':True,'source_sha256':digest(__file__)})
    print(json.dumps({'export':str(args.output),'native_forward_inputs':len(all_inputs)}),flush=True)


def fork_module():
    """Load only the official fork's TF backend, bypassing unrelated plotting imports."""
    root=BASE/'runtime_deps/shap'
    for name,path in [('shap',root),('shap.explainers',root/'explainers'),('shap.explainers.deep',root/'explainers/deep')]:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    load('shap.explainers.explainer',root/'explainers/explainer.py')
    load('shap.explainers.deep.misc',root/'explainers/deep/misc.py')
    return load('shap.explainers.deep.deep_tf',root/'explainers/deep/deep_tf.py')


def deepshap(args):
    package_paths()
    native=load('native_shap',HUMAN/'train_chrombpnet.py');native.bootstrap()
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    native.setup_tf(args.seed)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    session=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
    tf.compat.v1.keras.backend.set_session(session)
    deep=fork_module()
    import tf_keras
    deep.keras=tf_keras
    # AddV2 is the modern spelling of linear Add; no attribution rule changes.
    deep.op_handlers['AddV2']=deep.passthrough
    utilities=load('official_shap_utils',native.VENDOR/'chrombpnet/evaluation/interpret/shap_utils.py')
    model=tf.keras.models.load_model(args.export_dir/'bias_free_model.h5',compile=False)
    data=np.load(args.export_dir/'inputs_references.npz')
    n=args.references
    native_predictions=np.load(args.export_dir/'native_predictions.npz')
    from scipy.special import softmax
    count_target=tf.reduce_sum(model.outputs[1],axis=-1)
    profile_target=utilities.get_weightedsum_meannormed_logits(model)
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for head in ['counts','profile','profile_frozen_input']:
        all_hypothetical=[]
        all_projected=[]
        for i in range(args.loci):
            x=data['inputs'][i:i+1];references=data['references'][i,:n]
            zx=native_predictions['logits'][i].astype(np.float64)
            zr=native_predictions['logits'][3+i*100:3+i*100+n].astype(np.float64)
            wx=softmax(zx);wr=softmax(zr,axis=1)
            zxn=zx-zx.mean();zrn=zr-zr.mean(axis=1,keepdims=True)
            if head=='counts':target=count_target
            elif head=='profile':target=profile_target
            else:
                logits=model.outputs[0]
                target=tf.reduce_sum((logits-tf.reduce_mean(logits,axis=1,keepdims=True))*tf.constant(wx.astype(np.float32)[None]),axis=1)
            explainer=deep.TFDeepExplainer((model.input,target),[references[:args.reference_batch_size]],session=session,
                combine_mult_and_diffref=utilities.combine_mult_and_diffref)
            weighted_scores=[]
            for begin in range(0,n,args.reference_batch_size):
                chunk=references[begin:begin+args.reference_batch_size]
                explainer.data=[chunk]
                weighted_scores.append(np.asarray(explainer.shap_values(x),dtype=np.float32)*len(chunk))
            scores=np.sum(weighted_scores,axis=0)/n
            if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite official DeepSHAP scores')
            projected=scores*x
            target_x=float(session.run(target,feed_dict={model.input:x})[0])
            target_refs=np.concatenate([session.run(target,feed_dict={model.input:references[b:b+args.reference_batch_size]}) for b in range(0,n,args.reference_batch_size)])
            residual=float(projected.sum())-(target_x-float(np.mean(target_refs)))
            # The official profile target cuts the softmax-weight path. The fork's
            # product rule retains only the logit branch with pair-averaged weights.
            expected_partial=float(np.mean(np.sum(0.5*(wx[None]+wr)*(zxn[None]-zrn),axis=1))) if head=='profile' else target_x-float(np.mean(target_refs))
            conservation_residual=float(projected.sum())-expected_partial
            if abs(conservation_residual)>1e-3:raise AssertionError(f'{head} conservation failed: {conservation_residual}')
            records.append({'locus_index':i,'head':head,'references':n,'attribution_sum':float(projected.sum()),
                'target_input':target_x,'mean_target_reference':float(np.mean(target_refs)),
                'scalar_forward_additivity_residual':residual,
                'scalar_forward_additivity_applicable':head!='profile',
                'expected_attributed_difference':expected_partial,'conservation_residual':conservation_residual,
                'operator_types':sorted(explainer.used_types)})
            all_hypothetical.append(scores[0]);all_projected.append(projected[0])
            print(json.dumps(records[-1]),flush=True)
        np.savez_compressed(args.output/f'{head}_scores.npz',hypothetical=np.asarray(all_hypothetical),
                            projected=np.asarray(all_projected),inputs=data['inputs'][:args.loci],
                            references=data['references'][:args.loci,:n])
    save(args.output/'metadata.json',{'method':'official Kundaje TFDeepExplainer / DeepSHAP',
        'references':n,'reference_batch_size':args.reference_batch_size,'seed':args.seed,'model':'bias-free component of newly trained mouse ChromBPNet',
        'algorithm_note':'DeepSHAP uses DeepLIFT-style modified backpropagation and an ensemble of references; not independent from DeepLIFT',
        'profile_target':'official weighted mean-centered profile logits, softmax weights behind stop_gradient; partial conservation uses pair-averaged input/reference softmax weights',
        'profile_frozen_input_target':'mean-centered logits weighted by original-input softmax held fixed for both input and reference; standard scalar completeness applies',
        'chunking':'official per-reference multipliers/projections averaged in small independent chunks; no batch-dependent layers',
        'compatibility':['load official TF backend directly without unrelated plotting dependencies','linear AddV2 alias to linear Add passthrough'],
        'checks':records,'tf_backend_sha256':digest(Path(deep.__file__)),'official_shap_utils_sha256':digest(Path(utilities.__file__)),
        'source_sha256':digest(__file__)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['export','deepshap'],required=True)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--cache-dir',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--export-dir',type=Path,default=BASE/'export_fp32')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--references',type=int,choices=[2,20,100],default=20)
    p.add_argument('--reference-batch-size',type=int,default=2)
    p.add_argument('--loci',type=int,choices=[1,3],default=3)
    p.add_argument('--seed',type=int,default=1234)
    args=p.parse_args()
    export(args) if args.stage=='export' else deepshap(args)


if __name__=='__main__':main()
