#!/usr/bin/env python3
"""Plot common centered2048bp state contrasts with recovered historical scaling."""
import argparse,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cd200_s5_state_mpl')
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positive-dir',type=Path,default=HERE/'positive_contrast')
    parser.add_argument('--negative-dir',type=Path,default=HERE/'negative_contrast')
    args=parser.parse_args()
    sys.path.insert(0,'/home/xf2217/.codex/skills/nature-figure/scripts')
    from nature_matplotlib import set_nature_style
    set_nature_style();plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    pos=np.load(args.positive_dir/'state_predictions.npz')
    neg=np.load(args.negative_dir/'state_predictions.npz')
    assert np.array_equal(pos['loci'],neg['loci'])
    assert np.array_equal(pos['window_input_onehot'],neg['window_input_onehot'])
    pw=pd.read_csv(args.positive_dir/'windows.tsv',sep='\t')
    nw=pd.read_csv(args.negative_dir/'windows.tsv',sep='\t')
    assert pw.equals(nw)
    old=HERE/'original_modl/results_core_reproduction'
    native=pd.read_csv(old/'native_diff_recovered_contrasts.tsv',sep='\t').set_index('locus')
    audit=json.loads((old/'native_diff_normalization_audit.json').read_text())
    lp=audit['positive_state']['denominator'];ln=audit['negative_state']['denominator']
    assert lp==97975505 and ln==110158866
    def contrast(a,b):return np.log10(1+a/lp*1e5)-np.log10(1+b/ln*1e5)
    original=pd.read_csv(old/'s5b_original_and_redefined_contrasts.tsv',sep='\t').set_index('locus')
    loci=[str(x) for x in pos['loci']]
    rows=[];track_rows=[]
    for idx,locus in enumerate(loci):
        op=float(pos['observed_profiles'][idx].sum());on=float(neg['observed_profiles'][idx].sum())
        pp=float(pos['predicted_profiles'][idx].sum());pn=float(neg['predicted_profiles'][idx].sum())
        archived=original.loc[locus]
        assert np.isclose(op,archived.new_center2048_stored_pos_sum)
        assert np.isclose(on,archived.new_center2048_stored_neg_sum)
        rows.append({'locus':locus,'observed_pos_sum':op,'observed_neg_sum':on,
            'chrombpnet_pos_predicted_sum':pp,'chrombpnet_neg_predicted_sum':pn,
            'observed_contrast':contrast(op,on),'chrombpnet_contrast':contrast(pp,pn),
            'original_modl_native_differential':archived.phase3_native_differential_unscaled,
            'original_modl_legacy_scaled_for_historical_display_only':archived.legacy_scaled_differential_for_historical_display_only,
            'archived_observed_Diff_scca':archived.archived_observed_Diff_scca,
            'archived_chrombpnet_contrast':archived.archived_chrombpnet_contrast,
            'normalizer_positive':lp,'normalizer_negative':ln,
            'n_model_folds':1,'interval_start':int(archived.new_center2048_start),
            'interval_end':int(archived.new_center2048_end)})
        for offset in range(2048):
            track_rows.append({'locus':locus,'position_0':int(archived.new_center2048_start)+offset,
                'observed_pos':float(pos['observed_profiles'][idx,offset]),
                'observed_neg':float(neg['observed_profiles'][idx,offset]),
                'chrombpnet_pos':float(pos['predicted_profiles'][idx,offset]),
                'chrombpnet_neg':float(neg['predicted_profiles'][idx,offset])})
    out=HERE/'figures';out.mkdir(exist_ok=True)
    data=pd.DataFrame(rows)
    data.to_csv(out/'state_contrast_source_data.tsv',sep='\t',index=False)
    pd.DataFrame(track_rows).to_csv(out/'state_profile_source_data.tsv',sep='\t',index=False)
    fig,ax=plt.subplots(figsize=(110/25.4,67/25.4))
    fig.subplots_adjust(left=.16,right=.98,bottom=.20,top=.80)
    for j,(label,key,color) in enumerate([
        ('Observed','observed_contrast','#333333'),
        ('Original MoDL','original_modl_native_differential','#1964B0'),
        ('ChromBPNet','chrombpnet_contrast','#DB5829')]):
        values=data[key].to_numpy()
        x=np.arange(len(loci))+(j-1)*.23
        ax.bar(x,values,width=.21,color=color,label=label)
        for a,b in zip(x,values):
            ax.annotate(f'{b:.3f}' if abs(b)>=.001 else f'{b:.2g}',(a,b),
                        xytext=(0,3 if b>=0 else -8),textcoords='offset points',
                        ha='center',va='bottom' if b>=0 else 'top',fontsize=5)
    ax.axhline(0,color='#777777',lw=.4)
    ax.set_xticks(range(len(loci)),[{'P2':'Peak 2','P1':'Peak 1','promoter':'Promoter'}[x] for x in loci])
    ax.set_ylabel('Positive–negative accessibility contrast')
    ax.spines[['top','right']].set_visible(False)
    ax.margins(y=.25)
    fig.legend(loc='upper center',ncol=3,frameon=False,fontsize=6)
    for ext in ['pdf','svg','png']:fig.savefig(out/f'regenerated_s5b_contrast.{ext}',dpi=450,facecolor='white')
    plt.close(fig)
    caption='''Positive-versus-negative accessibility contrast at three selected loci. Observed and new ChromBPNet state-specific profiles are integrated over common 2048-bp intervals centered on each locus. Three overlapping 1000-bp predictions cover each interval; overlapping per-base estimates are averaged before summing. The state models were trained separately from random initialization, with state-specific bias models, test chromosomes chr16/chr8/chr9, validation chromosome chr14 and training-derived count-loss weights. Contrast = log10(1 + positive_sum/Lpos × 100000) − log10(1 + negative_sum/Lneg × 100000), with Lpos = 97,975,505 and Lneg = 110,158,866. These historical pre-filter denominators were recovered from retained sum/log-sum columns without fitting to the selected loci. Original MoDL is its unscaled native phase-3 differential output with an 8192-bp input at the same center. The original 3.307237 display multiplier is retained separately in the source table. Historical labels and current stored signals differ and are supplied separately. This panel is descriptive, with one fitted model per state. All bars are numerically labeled. The archived S5B Peak 1 ChromBPNet value was 0.0004167; it is distinct from the newly computed contrast.\n'''
    (out/'contrast_caption.txt').write_text(caption)
    files=[Path(__file__),old/'native_diff_recovered_contrasts.tsv',old/'s5b_original_and_redefined_contrasts.tsv',old/'native_diff_normalization_audit.json']
    for directory in [args.positive_dir,args.negative_dir]:files += [directory/'state_predictions.npz',directory/'metadata.json',directory/'windows.tsv']
    (out/'contrast_manifest.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'normalization':{'positive':lp,'negative':ln},'legacy_modl_shrinkage_applied':False,'n_folds_per_state':1},indent=2)+'\n')
    print(data[['locus','observed_contrast','original_modl_native_differential','chrombpnet_contrast']].to_string(index=False))

if __name__=='__main__':main()
