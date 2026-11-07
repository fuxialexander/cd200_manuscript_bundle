#!/usr/bin/env python3
"""Plot locus-level MoDL deletion effects and new ChromBPNet attributions."""
import argparse,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cd200_s5_interpret_mpl')
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
LOCI=['P2','P1','promoter']
ORDER={'P1':0,'P2':1,'promoter':2}
# The two archived TEAD scans use overlapping PWM widths/alignments; display
# their 19bp union as a motif-containing interval, not one exact PWM instance.
MOTIFS={'P1':(45384058,45384077,'TEAD'), 'P2':(45356358,45356371,'ETS')}

def correlation(a,b):
    a=a-a.mean();b=b-b.mean();n=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return None if n<1e-15 else float(np.dot(a,b)/n)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deepshap-dir',type=Path,required=True)
    parser.add_argument('--deeplift-dir',type=Path,default=HERE/'chrombpnet_interpret/deeplift_100ref')
    args=parser.parse_args()
    sys.path.insert(0,'/home/xf2217/.codex/skills/nature-figure/scripts')
    from nature_matplotlib import set_nature_style
    set_nature_style();plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    out=HERE/'figures';out.mkdir(exist_ok=True)
    counts_shap=np.load(args.deepshap_dir/'counts_scores.npz')
    counts_lift=np.load(args.deeplift_dir/'counts_scores.npz')
    profile_shap=np.load(args.deepshap_dir/'profile_scores.npz')
    frozen_shap=np.load(args.deepshap_dir/'profile_frozen_input_scores.npz')
    frozen_lift=np.load(args.deeplift_dir/'profile_frozen_input_scores.npz')
    assert np.array_equal(counts_shap['inputs'],counts_lift['inputs'])
    assert np.array_equal(counts_shap['references'],counts_lift['references'])
    for scores in [counts_shap,counts_lift,profile_shap,frozen_shap,frozen_lift]:
        assert scores['projected'].shape==(3,2114,4)
        assert np.isfinite(scores['projected']).all()
        assert np.allclose(scores['projected'],scores['hypothetical']*scores['inputs'],atol=1e-7)
    profiles=pd.read_csv(out/'profile_source_data.tsv',sep='\t')
    windows=pd.read_csv(HERE/'original_modl/results_core_reproduction/matched1000_windows.tsv',sep='\t')
    export_meta_path=HERE/'chrombpnet_interpret/export_fp32/export_metadata.json'
    export_metadata=json.loads(export_meta_path.read_text())
    assert [w['locus'] for w in export_metadata['windows']]==['P1','P2','promoter']
    assert export_metadata['inputs_references_sha256']==hashlib.sha256((HERE/'chrombpnet_interpret/export_fp32/inputs_references.npz').read_bytes()).hexdigest()
    source=[];checks=[]
    fig,axes=plt.subplots(5,3,figsize=(183/25.4,155/25.4))
    fig.subplots_adjust(left=.145,right=.99,bottom=.09,top=.92,wspace=.38,hspace=.40)
    labels=['Normalized\nprofile','MoDL deletion\nprofile effect','ChromBPNet\nDeepSHAP count','ChromBPNet\nDeepLIFT count','ChromBPNet\nDeepSHAP profile']
    for col,locus in enumerate(LOCI):
        idx=ORDER[locus]
        window=windows[(windows.locus==locus)&(windows.role=='central')].iloc[0]
        attribution_window=export_metadata['windows'][idx]
        assert int(attribution_window['output_start'])==int(window.output_start)
        assert int(attribution_window['output_end'])==int(window.output_end)
        positions=np.arange(window.output_start,window.output_end,dtype=int)
        center=int(window.output_start)+500
        deletion_path=HERE/'original_modl/results_core_reproduction/archived_deletion_tracks'/f'{locus}_regression_epoch105.csv'
        deletion=pd.read_csv(deletion_path).set_index('position').loc[positions,'effect'].to_numpy()
        tracks=[deletion,counts_shap['projected'][idx,557:1557].sum(-1),
                counts_lift['projected'][idx,557:1557].sum(-1),
                profile_shap['projected'][idx,557:1557].sum(-1)]
        for row in range(5):
            ax=axes[row,col]
            if row==0:
                d=profiles[profiles.locus==locus]
                for name,key,color in [('Observed','observed_conv20_mean','#333333'),('MoDL','modl_native4096_prediction','#1964B0'),('ChromBPNet','chrombpnet_softmax_conv20_mean','#DB5829')]:
                    y=d[key].to_numpy();y=y/y.sum()
                    ax.plot(d.position_0-center,y,color=color,lw=.5,label=name)
                ax.set_title({'P2':'Peak 2','P1':'Peak 1','promoter':'Promoter'}[locus],fontsize=7,fontweight='normal',pad=12)
            else:
                y=tracks[row-1]
                color='#1964B0' if row==1 else '#DB5829'
                ax.axhline(0,color='#999999',lw=.35,zorder=0)
                ax.plot(positions-center,y,color=color,lw=.45)
                for pos,value in zip(positions,y):source.append({'locus':locus,'position_0':int(pos),'track':labels[row].replace('\n',' '),'score':float(value)})
                if locus in MOTIFS:
                    a,b,label=MOTIFS[locus]
                    ax.axvspan(a-center,b-center,color='#AE75A2',alpha=.20,lw=0)
            if col==0:ax.set_ylabel(labels[row],fontsize=6)
            ax.set_xlim(-500,500);ax.set_xticks([-400,0,400])
            ax.tick_params(labelsize=5)
            if row==4:ax.set_xlabel('Position from locus center (bp)',fontsize=6)
            else:ax.tick_params(labelbottom=False)
            ax.spines[['top','right']].set_visible(False)
            if col==0:ax.text(-.41,1.01,chr(97+row),transform=ax.transAxes,fontweight='bold',fontsize=8)
        for target,shap,lift in [('counts',counts_shap,counts_lift),('profile_frozen_input',frozen_shap,frozen_lift)]:
            a=shap['projected'][idx].sum(-1);b=lift['projected'][idx].sum(-1)
            for extent,sel in [('full2114',slice(None)),('central1000',slice(557,1557))]:
                checks.append({'locus':locus,'target':target,'extent':extent,
                    'pearson_between_implementations':correlation(a[sel],b[sel]),
                    'max_absolute_difference':float(np.max(np.abs(a[sel]-b[sel]))),
                    'reference_count':100})
    fig.legend(handles=[Line2D([],[],color=color,label=name,lw=.7) for name,color in
                        [('Observed','#333333'),('MoDL','#1964B0'),('ChromBPNet','#DB5829')]],
               loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,fontsize=6)
    for ext in ['pdf','svg','png']:fig.savefig(out/f'locus_interpretation_comparison.{ext}',dpi=450,facecolor='white')
    plt.close(fig)
    pd.DataFrame(source).to_csv(out/'interpretation_source_data.tsv',sep='\t',index=False)
    pd.DataFrame(checks).to_csv(out/'deepshap_deeplift_agreement.tsv',sep='\t',index=False)
    caption='''Locus-level interpretation of original MoDL and new mouse ChromBPNet. Columns show common central 1000-bp windows at Peak 2, Peak 1 and the promoter. The first row shows observed/predicted profiles on the common 20-bp-smoothed target. MoDL effects are the maximum reference profile minus maximum deletion profile within the annotated core, distributed across each 15-bp deletion at 1-bp stride; positive effects indicate a reduction after deletion. Archived tracks were checked by selected fresh deletion inferences. ChromBPNet tracks are signed observed-base projections from its bias-free component, using official DeepSHAP count/profile targets and standalone Captum DeepLIFT Rescale count attributions. Both implementations use identical inputs and 100 dinucleotide-shuffled references. Single-reference DeepLIFT explanations are averaged; DeepSHAP uses related DeepLIFT-family rules, so agreement is an implementation check. Axes retain distinct target units: regional profile-maximum deletion effects, natural-log count contributions and weighted centered-logit profile contributions. Purple spans mark the archived ETS motif at Peak 2 and the 19-bp union of overlapping archived TEAD scans at Peak 1. Scores are neither smoothed nor converted to absolute values. Target definitions, forward-parity, projection and conservation checks are supplied separately. No TF-MoDISco analysis or global fragmentation claim is made.\n'''
    (out/'interpretation_caption.txt').write_text(caption)
    files=[Path(__file__),args.deepshap_dir/'counts_scores.npz',args.deepshap_dir/'profile_scores.npz',args.deeplift_dir/'counts_scores.npz',
           args.deepshap_dir/'profile_frozen_input_scores.npz',args.deeplift_dir/'profile_frozen_input_scores.npz',
           export_meta_path,HERE/'chrombpnet_interpret/export_fp32/inputs_references.npz',
           out/'profile_source_data.tsv',HERE/'original_modl/results_core_reproduction/matched1000_windows.tsv']
    files += [HERE/'original_modl/results_core_reproduction/archived_deletion_tracks'/f'{locus}_regression_epoch105.csv' for locus in LOCI]
    motif_sources=[Path('/mnt/storage/cd200/chrombpnet_moa/tead.bed'),
                   HERE.parents[1]/'analysis/results/cd200_analysis_new/motif_scan_deg_chr16_45383863_45384762.tsv',
                   HERE.parents[1]/'analysis/results/cd200_analysis_new/motif_scan_deg_chr16_45356164_45357191.tsv']
    files += motif_sources
    (out/'motif_annotations.json').write_text(json.dumps({'coordinate_convention':'mm10 zero-based half-open',
        'intervals':MOTIFS,'P1_definition':'union of archived TEAD interval45384058–45384070 and original MoDL scan45384064–45384077; not a newly scanned motif',
        'sources':list(map(str,motif_sources))},indent=2)+'\n')
    (out/'interpretation_manifest.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'score_units_shared_between_models':False,'n_references':100},indent=2)+'\n')
    print(pd.DataFrame(checks).to_string(index=False))

if __name__=='__main__':main()
