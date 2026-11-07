#!/usr/bin/env python3
"""Regenerate descriptive Cd200 profile comparison with matched positions/target."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cd200_s5_new_mpl')
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'
CACHE=Path('/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/cache')
NEW=Path('/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/runs/chrombpnet_seed1234/locus_predictions')
OLD=HERE/'original_modl/results_core_reproduction'
COLORS={'MoDL':'#1964B0','ChromBPNet':'#DB5829','Observed':'#333333'}

def corr(a,b):
    a=a-a.mean();b=b-b.mean();denom=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return np.nan if denom<1e-15 else float(np.dot(a,b)/denom)

def main():
    sys.path.insert(0,'/home/xf2217/.codex/skills/nature-figure/scripts')
    from nature_matplotlib import set_nature_style
    set_nature_style();plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    OUT.mkdir(exist_ok=True)
    windows=pd.read_csv(OLD/'matched1000_windows.tsv',sep='\t')
    cache_windows=pd.read_csv(CACHE/'locus_windows.tsv',sep='\t')
    assert windows.equals(cache_windows)
    old=np.load(OLD/'matched1000_profiles.npz')
    new=np.load(NEW/'prediction_profiles.npy')
    raw=np.load(CACHE/'observed_profiles.npy')
    assert new.shape==old['native4096_profile'].shape==raw.shape==(7,1000)
    assert np.array_equal(raw,old['observed_stored_counts'])
    assert np.allclose(new.sum(-1),1,atol=1e-4)
    # Matching original 20bp mean target: do NOT smooth the native MoDL output again.
    new_smoothed=convolve1d(new.astype(float),np.ones(20)/20,axis=-1,mode='nearest')
    observed=old['observed_overlay_conv20'].astype(float)
    rows=[];pairrows=[];profiles=[]
    rng=np.random.default_rng(1234)
    loci=['P2','P1','promoter']
    for locus in loci:
        idx=int(windows.index[(windows.locus==locus)&(windows.role=='central')][0])
        row=windows.loc[idx]
        y=observed[idx,10:-10]
        predictions={'MoDL':old['native4096_profile'][idx,10:-10].astype(float),
                     'ChromBPNet':new_smoothed[idx,10:-10]}
        adapted=old['adapted2114_profile'][idx,10:-10].astype(float)
        assert len(y)==980 and np.isfinite(y).all()
        assert np.allclose(y,convolve1d(raw[idx].astype(float),np.ones(20)/20,mode='nearest')[10:-10],atol=1e-6)
        # Ten spatial blocks of98bp, paired across both fixed models and observed target.
        blocks=np.arange(980).reshape(10,98)
        boot=[]
        for _ in range(2000):
            ids=blocks[rng.integers(0,10,10)].reshape(-1)
            values=[corr(p[ids],y[ids]) for p in predictions.values()]
            if np.isfinite(values).all():boot.append(values)
        boot=np.asarray(boot)
        assert len(boot)>=1900
        for j,(name,p) in enumerate(predictions.items()):
            lo,hi=np.quantile(boot[:,j],[.025,.975])
            rows.append({'locus':locus,'model':name,'pearson':corr(p,y),
                'ci_lower':lo,'ci_upper':hi,'bootstrap_valid':len(boot),
                'n_genomic_positions':980,'block_bp':98,'n_blocks':10,
                'input_context_bp':4096 if name=='MoDL' else 2114,
                'adapted2114_modl_pearson':corr(adapted,y) if name=='MoDL' else np.nan,
                'ci_interpretation':'spatial block resampling of this locus; not across model folds or biological replicates'})
        delta=boot[:,0]-boot[:,1]
        lo,hi=np.quantile(delta,[.025,.975])
        pairrows.append({'locus':locus,'pearson_difference_modl_minus_chrombpnet':corr(predictions['MoDL'],y)-corr(predictions['ChromBPNet'],y),
                         'ci_lower':lo,'ci_upper':hi,'test':'paired spatial-block bootstrap difference interval; no across-fold hypothesis test'})
        positions=np.arange(int(row.output_start)+10,int(row.output_end)-10)
        for pos,obs,a,b in zip(positions,y,predictions['MoDL'],predictions['ChromBPNet']):
            profiles.append({'locus':locus,'position_0':pos,'observed_conv20_mean':obs,
                             'modl_native4096_prediction':a,'chrombpnet_softmax_conv20_mean':b})
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'profile_metrics.tsv',sep='\t',index=False)
    pd.DataFrame(pairrows).to_csv(OUT/'paired_profile_differences.tsv',sep='\t',index=False)
    frame=pd.DataFrame(profiles);frame.to_csv(OUT/'profile_source_data.tsv',sep='\t',index=False)
    fig=plt.figure(figsize=(183/25.4,110/25.4))
    gs=fig.add_gridspec(2,3,left=.09,right=.985,bottom=.13,top=.87,wspace=.34,hspace=.75)
    ax=fig.add_subplot(gs[0,:])
    for j,name in enumerate(['MoDL','ChromBPNet']):
        vals=metrics[metrics.model==name].set_index('locus').loc[loci]
        x=np.arange(3)+(j-.5)*.24
        ax.errorbar(x,vals.pearson,yerr=np.vstack([vals.pearson-vals.ci_lower,vals.ci_upper-vals.pearson]),
                    fmt='o',color=COLORS[name],markersize=3,capsize=2,lw=.7,label=name)
    ax.set_xticks(range(3),['Peak 2','Peak 1','Promoter']);ax.set_ylim(-.05,1.03)
    ax.set_ylabel('Profile Pearson r');ax.spines[['top','right']].set_visible(False)
    ax.text(-.065,1.13,'a',transform=ax.transAxes,fontweight='bold',fontsize=8)
    for i,locus in enumerate(loci):
        ax=fig.add_subplot(gs[1,i]);d=frame[frame.locus==locus]
        x=d.position_0.to_numpy();x=x-x[len(x)//2]
        for name,key in [('Observed','observed_conv20_mean'),('MoDL','modl_native4096_prediction'),('ChromBPNet','chrombpnet_softmax_conv20_mean')]:
            y=d[key].to_numpy();y=y/y.sum()
            ax.plot(x,y,color=COLORS[name],lw=.55,label=name,alpha=.9)
        ax.set_title(['Peak 2','Peak 1','Promoter'][i],fontsize=7,fontweight='normal')
        ax.set_xlabel('Position from locus center (bp)')
        if i==0:ax.set_ylabel('Normalized profile')
        ax.tick_params(labelsize=5.5);ax.spines[['top','right']].set_visible(False)
        ax.text(-.25,1.2,chr(98+i),transform=ax.transAxes,fontweight='bold',fontsize=8)
    fig.legend(handles=[Line2D([],[],color=COLORS[n],label=n) for n in ['Observed','MoDL','ChromBPNet']],
               loc='upper center',ncol=3,frameon=False,fontsize=6)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'regenerated_s5a_profiles.{ext}',dpi=450,facecolor='white')
    plt.close(fig)
    caption='''Selected-locus profile comparison. Original MoDL epoch 105 uses its native 4096-bp input context, cropped to the common central 1000-bp output; the new chromosome-held-out ChromBPNet model uses a 2114-bp input and 1000-bp output. ChromBPNet predictions are averaged over 20 bp to match the original MoDL target; MoDL predictions are not smoothed again. Excluding ten bases at each output edge leaves 980 common positions and avoids smoothing boundary artifacts. Curve normalization affects display only. Error bars are 95% paired spatial-block bootstrap intervals (2000 resamples of ten 98-bp blocks). They describe conditional within-locus resampling, rather than variation across fitted folds or biological replicates. Paired difference intervals are supplied in paired_profile_differences.tsv. One completed native mouse fold was available; no fold-based hypothesis test is reported. The 2114-bp MoDL input sensitivity is retained in the source table. These selected loci illustrate local fit; the separate matched benchmarks provide generalization evidence.\n'''
    (OUT/'profile_caption.txt').write_text(caption)
    inputs=[OLD/'matched1000_profiles.npz',OLD/'matched1000_windows.tsv',OLD/'manifest.json',
            NEW/'prediction_profiles.npy',NEW/'prediction_metadata.json',
            CACHE/'observed_profiles.npy',CACHE/'locus_windows.tsv',Path(__file__)]
    (OUT/'profile_manifest.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'model_fold_replicates':1,'target':'original20bpmeanstoredATAC','n_loci':3,'bootstrap_replicates':2000,'base_level_independence_assumed':False},indent=2)+'\n')
    print(metrics[['locus','model','pearson','ci_lower','ci_upper']].to_string(index=False))

if __name__=='__main__':main()
