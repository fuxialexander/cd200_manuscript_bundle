#!/usr/bin/env python3
"""Compact two-panel proposed replacement for the submitted Fig. S5."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cd200_s5_compact_mpl')
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-display',action='store_true',help='Show the documented original3.307237 MoDL differential display multiplier; does not affect profile correlations')
    args=parser.parse_args()
    sys.path.insert(0,'/home/xf2217/.codex/skills/nature-figure/scripts')
    from nature_matplotlib import set_nature_style
    set_nature_style();plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    fit=pd.read_csv(OUT/'profile_metrics.tsv',sep='\t')
    contrast=pd.read_csv(OUT/'state_contrast_source_data.tsv',sep='\t').set_index('locus')
    loci=['P2','P1','promoter'];names=['Peak 2','Peak 1','Promoter']
    colors={'MoDL':'#1964B0','ChromBPNet':'#DB5829','Observed':'#333333'}
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,75/25.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.17,top=.73,wspace=.38)
    for j,name in enumerate(['MoDL','ChromBPNet']):
        d=fit[fit.model==name].set_index('locus').loc[loci]
        axes[0].errorbar(np.arange(3)+(j-.5)*.22,d.pearson,
            yerr=np.vstack([d.pearson-d.ci_lower,d.ci_upper-d.pearson]),
            color=colors[name],fmt='o' if j==0 else 's',markersize=3,capsize=2,lw=.7)
    axes[0].set_ylabel('Profile Pearson r');axes[0].set_ylim(-.05,1.03)
    modl_key='original_modl_legacy_scaled_for_historical_display_only' if args.legacy_display else 'original_modl_native_differential'
    keys=[('Observed','observed_contrast'),('MoDL',modl_key),('ChromBPNet','chrombpnet_contrast')]
    for j,(name,key) in enumerate(keys):
        x=np.arange(3)+(j-1)*.23;y=contrast.loc[loci,key].to_numpy()
        axes[1].bar(x,y,width=.21,color=colors[name])
        for pos,value in zip(x,y):
            axes[1].annotate(f'{value:.3f}' if abs(value)>=.001 else f'{value:.2g}',
                (pos,value),xytext=(0,3 if value>=0 else -5),textcoords='offset points',
                ha='center',va='bottom' if value>=0 else 'top',fontsize=5)
    axes[1].set_ylabel('Positive–negative contrast')
    axes[1].axhline(0,color='#777777',lw=.4);axes[1].margins(y=.25)
    for i,ax in enumerate(axes):
        ax.set_xticks(range(3),names);ax.tick_params(labelsize=6)
        ax.spines[['top','right']].set_visible(False)
        title=['Profile fit','Accessibility contrast'][i]
        if i==1 and args.legacy_display:title+=' (MoDL ×3.307)'
        ax.set_title(title,fontsize=7,fontweight='normal',pad=10)
        ax.text(-.17,1.22,chr(97+i),transform=ax.transAxes,fontsize=8,fontweight='bold')
    fig.legend(handles=[Line2D([],[],color=colors[n],lw=1.5,label='MoDL (original)' if n=='MoDL' else n)
                        for n in ['Observed','MoDL','ChromBPNet']],loc='upper center',ncol=3,
               bbox_to_anchor=(.5,.995),fontsize=6,frameon=False)
    stem='FigS5_regenerated_legacy_calibrated' if args.legacy_display else 'FigS5_regenerated'
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{stem}.{ext}',dpi=450,facecolor='white')
    plt.close(fig)
    caption='''Proposed replacement Fig. S5. (a) Profile Pearson correlations for original positive-state MoDL and new positive-state ChromBPNet over 980 common positions per locus, after matching the 20-bp target smoothing and excluding ten output-edge bases. Error bars are 95% paired spatial-block bootstrap intervals (2000 resamples of ten 98-bp blocks), rather than model-fold or biological-replicate uncertainty. Native inputs are 4096 bp for MoDL and 2114 bp for ChromBPNet. (b) Positive-versus-negative contrasts over common centered 2048-bp intervals. The new ChromBPNet models use the same held-out chromosomes and state-specific bias models. Overlapping predicted per-base counts are averaged before integration. Contrast = log10(1 + positive_sum/Lpos × 100000) − log10(1 + negative_sum/Lneg × 100000), using recovered historical constants Lpos = 97,975,505 and Lneg = 110,158,866. MoDL is its native phase-3 differential output with an 8192-bp input; the earlier 3.307237 display multiplier is not applied in this primary version. Historical/current count differences and legacy values remain in source tables. Panel b is descriptive, with one model per state and no fold-based test. All contrast values are labeled. Original S5B's archived Peak 1 ChromBPNet value was 0.0004167. The separate matched genome-wide benchmarks provide comparative generalization evidence.\n'''
    if args.legacy_display:
        caption=caption.replace('the earlier 3.307237 display multiplier is not applied in this primary version','the documented earlier 3.307237 display multiplier IS applied to MoDL in this alternative; no new multiplier is fitted')
        caption='LEGACY CALIBRATION SENSITIVITY VERSION. '+caption
    (OUT/(stem+'_caption.txt')).write_text(caption)
    inputs=[Path(__file__),OUT/'profile_metrics.tsv',OUT/'state_contrast_source_data.tsv']
    (OUT/(stem+'_manifest.json')).write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'width_mm':183,'height_mm':75,'model_folds_per_state':1,'legacy_modl_differential_multiplier':3.307237044473427 if args.legacy_display else None},indent=2)+'\n')
    print(OUT/(stem+'.pdf'))

if __name__=='__main__':main()
