#!/usr/bin/env python3
"""Standalone Captum DeepLIFT Rescale on a forward-verified native model mirror."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'runtime_deps'))
import torch
from torch import nn
from captum.attr import DeepLift
from scipy.special import softmax


class NativeMirror(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.first=nn.Conv1d(4,512,21)
        self.first_relu=nn.ReLU()
        self.convs=nn.ModuleList([nn.Conv1d(512,512,3,dilation=2**i) for i in range(1,9)])
        self.relus=nn.ModuleList([nn.ReLU() for _ in range(8)])
        self.profile=nn.Conv1d(512,1,75)
        self.count=nn.Linear(512,1)
        for layer,name in [(self.first,'wo_bias_bpnet_1st_conv'),
                           *[(layer,f'wo_bias_bpnet_{i}conv') for i,layer in enumerate(self.convs,1)],
                           (self.profile,'wo_bias_bpnet_prof_out_precrop')]:
            layer.weight.data.copy_(torch.from_numpy(weights[name+'__0'].transpose(2,1,0).copy()))
            layer.bias.data.copy_(torch.from_numpy(weights[name+'__1']))
        self.count.weight.data.copy_(torch.from_numpy(weights['wo_bias_bpnet_logcount_predictions__0'].T.copy()))
        self.count.bias.data.copy_(torch.from_numpy(weights['wo_bias_bpnet_logcount_predictions__1']))
        for p in self.parameters():p.requires_grad_(False)

    def forward(self,x):
        x=self.first_relu(self.first(x))
        for i,(conv,relu) in enumerate(zip(self.convs,self.relus),1):
            d=2**i
            x=x[:,:,d:-d]+relu(conv(x))
        return self.profile(x)[:,0,:],self.count(x.mean(-1))[:,0]


class ScalarTarget(nn.Module):
    def __init__(self,model,head,weights=None):
        super().__init__();self.model=model;self.head=head
        self.register_buffer('weights',None if weights is None else torch.from_numpy(weights[None].astype(np.float32)))

    def forward(self,x):
        p,c=self.model(x)
        return c if self.head=='counts' else ((p-p.mean(-1,keepdim=True))*self.weights).sum(-1)


def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export-dir',type=Path,default=BASE/'export_fp32')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--references',type=int,default=100)
    p.add_argument('--reference-batch-size',type=int,default=4)
    p.add_argument('--loci',type=int,choices=[1,3],default=3)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('GPU required for authorized attribution; no CPU fallback')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=NativeMirror(np.load(args.export_dir/'native_weights.npz')).cuda().eval()
    data=np.load(args.export_dir/'inputs_references.npz')
    original=np.load(args.export_dir/'native_predictions.npz')
    all_inputs=np.concatenate([data['inputs'],data['references'].reshape(-1,2114,4)])
    pp,cc=[],[]
    with torch.no_grad():
        for begin in range(0,len(all_inputs),args.reference_batch_size):
            x=torch.from_numpy(all_inputs[begin:begin+args.reference_batch_size].transpose(0,2,1).copy()).cuda()
            pr,ct=model(x);pp.append(pr.cpu().numpy());cc.append(ct.cpu().numpy())
    pp,cc=np.concatenate(pp),np.concatenate(cc)
    np.testing.assert_allclose(pp,original['logits'],rtol=3e-5,atol=3e-5)
    np.testing.assert_allclose(cc,original['counts'],rtol=3e-5,atol=3e-5)
    args.output.mkdir(parents=True,exist_ok=False)
    parity={'status':'PASS','n_inputs':len(all_inputs),'scope':'all three observed inputs and all 300 reference sequences',
        'max_abs_profile_logit_difference':float(np.max(np.abs(pp-original['logits']))),
        'max_abs_count_difference':float(np.max(np.abs(cc-original['counts']))),'rtol':3e-5,'atol':3e-5,
        'tf32_enabled':False,'weight_conversion':'Conv1D Keras[L,in,out]→PyTorch[out,in,L]; Dense[in,out]→[out,in]; native VALID cropping/residual additions retained'}
    save(args.output/'forward_parity.json',parity)
    records=[]
    for head in ['counts','profile_frozen_input']:
        outputs=[];hypotheticals=[];differences=[]
        for i in range(args.loci):
            weights=softmax(original['logits'][i]).astype(np.float32)
            target=ScalarTarget(model,head,weights if head!='counts' else None).cuda().eval()
            explainer=DeepLift(target,multiply_by_inputs=False,eps=1e-6)
            x=torch.from_numpy(data['inputs'][i:i+1].transpose(0,2,1).copy()).cuda()
            per_hyp=[];per_diff=[];residuals=[]
            for begin in range(0,args.references,args.reference_batch_size):
                refs=torch.from_numpy(data['references'][i,begin:min(begin+args.reference_batch_size,args.references)].transpose(0,2,1).copy()).cuda()
                xx=x.repeat(len(refs),1,1).requires_grad_(True)
                multipliers=explainer.attribute(xx,baselines=refs)
                diff=multipliers*(xx-refs)
                hypothetical=multipliers-(multipliers*refs).sum(1,keepdim=True)
                with torch.no_grad():expected=target(xx)-target(refs)
                residual=diff.sum((1,2))-expected
                if not bool(torch.isfinite(multipliers).all()):raise FloatingPointError('Nonfinite standalone DeepLIFT multiplier')
                residuals.extend(residual.detach().cpu().tolist())
                per_hyp.extend(hypothetical.detach().cpu().numpy().transpose(0,2,1))
                per_diff.extend(diff.detach().cpu().numpy().transpose(0,2,1))
            per_hyp,per_diff=np.asarray(per_hyp),np.asarray(per_diff)
            if np.max(np.abs(residuals))>1e-3:raise AssertionError(f'DeepLIFT conservation failed: {max(np.abs(residuals))}')
            average=per_hyp.mean(0)
            outputs.append(average*data['inputs'][i]);hypotheticals.append(average);differences.append(per_diff)
            # Demonstrate actual modified-backprop multipliers, rather than an ordinary gradient placeholder.
            gx=x.clone().detach().requires_grad_(True)
            gradient=torch.autograd.grad(target(gx).sum(),gx)[0].detach().cpu().numpy()[0].T
            ref0=data['references'][i,0]
            grad_hyp=gradient-(gradient*ref0).sum(-1,keepdims=True)
            distance=float(np.linalg.norm(per_hyp[0]-grad_hyp)/(np.linalg.norm(grad_hyp)+1e-12))
            record={'head':head,'locus_index':i,'references':args.references,
                'max_abs_single_reference_additivity_residual':float(np.max(np.abs(residuals))),
                'mean_reference_additivity_residual':float(np.mean(residuals)),
                'first_reference_relative_l2_difference_from_ordinary_gradient_hypothetical':distance}
            records.append(record);print(json.dumps(record),flush=True)
            np.savez_compressed(args.output/f'{head}_locus{i}_per_reference.npz',hypothetical=per_hyp,
                difference_attributions=per_diff,inputs=data['inputs'][i],references=data['references'][i,:args.references],
                profile_weights=weights if head!='counts' else np.array([]))
        np.savez_compressed(args.output/f'{head}_scores.npz',hypothetical=np.asarray(hypotheticals),
            projected=np.asarray(outputs),inputs=data['inputs'][:args.loci],references=data['references'][:args.loci,:args.references],
            difference_attributions=np.asarray(differences))
    save(args.output/'metadata.json',{'method':'standalone Captum DeepLIFT Rescale, independently implemented model mirror',
        'references':args.references,'reference_batch_size':args.reference_batch_size,'checks':records,
        'algorithm_relationship':'DeepSHAP also uses DeepLIFT-family rules. Averaging independent single-reference Rescale attributions here is an implementation/baseline comparison, not a claim of unrelated methods.',
        'profile_target':'mean-centered logits weighted by unperturbed-input softmax probabilities held fixed for BOTH input and each reference',
        'single_reference_sensitivity':'first member in each per-reference NPZ; same seed/reference as DeepSHAP prefix',
        'forward_parity':parity,'torch':torch.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__=='__main__':main()
