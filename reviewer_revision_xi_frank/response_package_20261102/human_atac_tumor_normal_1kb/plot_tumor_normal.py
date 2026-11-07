#!/usr/bin/env python3
"""First-pass Fig. 4D-normalized tumor distributions and normal reference lines."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cd200_tumor_normal_mpl')
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/storage/cd200/figures/human_cd200_atac_locus')
DATA = ROOT / 'data'
INPUTS = {
    'tumors': DATA/'tcga/bigwig_1mb_normalized/tcga_display_donors_normalized.tsv.gz',
    'normals': DATA/'catlas_normals_1mb_normalized/catlas_display_normalized.tsv.gz',
    'normal_manifest': DATA/'catlas_normals_1mb_normalized/catlas_normal_manifest.tsv',
    'tumor_contract': DATA/'tcga/bigwig_1mb_normalized/normalization_contract.json',
    'normal_contract': DATA/'catlas_normals_1mb_normalized/normalization_contract.json',
}
# Centers from original source constants: annotated transcript start and motif midpoint.
WINDOWS = {'Promoter': (112333155-500, 112333155+500),
           'TEAD motif': (112360767-500, 112360767+500)}
ORIGINAL_COHORTS = ['KIRP', 'KIRC', 'MESO', 'LUSC', 'UCEC', 'SKCM', 'STAD']
PRIORITY = {'exact_lineage': 0, 'tissue_proxy': 1, 'lineage_proxy': 2,
            'developmental_proxy': 3, 'tissue_lineage_proxy': 4}

def main():
    sys.path.insert(0, '/home/xf2217/.codex/skills/nature-figure/scripts')
    from nature_matplotlib import set_nature_style
    set_nature_style()
    # Reuse installed Arial, including nonstandard paths already indexed by Matplotlib.
    for p in Path('/usr/share/fonts').rglob('*Arial*.ttf'):
        font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
    tumors = pd.read_csv(INPUTS['tumors'], sep='\t')
    normals = pd.read_csv(INPUTS['normals'], sep='\t').set_index('normal_id')
    manifest = pd.read_csv(INPUTS['normal_manifest'], sep='\t').set_index('normal_id')
    bins = [c for c in tumors if c.startswith('chr3:')]
    normal_bins = list(normals.columns)
    bounds = np.array([list(map(int, b.split(':')[1].split('-'))) for b in bins])
    assert np.all(bounds[:-1, 1] == bounds[1:, 0])
    assert len(tumors) == 404 and tumors.donor_id.nunique() == 404
    assert tumors.case_id.str[:12].nunique() == 404
    for x in [tumors[bins].to_numpy(), normals.to_numpy()]:
        assert np.isfinite(x).all() and (x >= 0).all() and (x <= 1+1e-8).all()
    # Exactly the original normal matching priority, stable manifest order for ties.
    matching = {}
    for cohort in sorted(tumors.cohort.unique()):
        candidates = [idx for idx, r in manifest.iterrows()
                      if cohort in str(r.matched_cancers).split(';')]
        candidates.sort(key=lambda idx: PRIORITY[manifest.loc[idx, 'match_class']])
        matching[cohort] = candidates[0] if candidates else None
    rows, weights = [], []
    for region, (start, end) in WINDOWS.items():
        for source, frame in [('tumor', tumors.set_index('donor_id')[bins]), ('normal', normals)]:
            source_bins = list(frame.columns)
            source_bounds = np.array([list(map(int, b.split(':')[1].split('-'))) for b in source_bins])
            assert np.all(source_bounds[:-1, 1] == source_bounds[1:, 0])
            overlap = np.maximum(0, np.minimum(source_bounds[:, 1], end)-np.maximum(source_bounds[:, 0], start))
            assert overlap.sum() == end-start == 1000
            w = overlap/(end-start)
            keep = overlap > 0
            for b, bp in zip(np.array(source_bins)[keep], overlap[keep]):
                weights.append({'source':source,'region':region, 'bin':b, 'overlap_bp':int(bp), 'weight':bp/1000})
            means = frame.to_numpy() @ w
            maxima = frame.to_numpy()[:, keep].max(axis=1)
            for idx, mean, maximum in zip(frame.index, means, maxima):
                if source == 'tumor':
                    t = tumors.set_index('donor_id').loc[idx]
                    annotation = {'cohort':t.cohort, 'case_id':t.case_id}
                else:
                    annotation = {'cohort':'', 'case_id':''}
                rows.append({'source':source, 'source_id':idx, **annotation, 'region':region,
                             'start_0':start, 'end_0':end, 'mean_signal':mean,
                             'area_signal_bp':mean*1000, 'max_bin_signal':maximum})
    values = pd.DataFrame(rows)
    values.to_csv(OUT/'source_data.tsv',sep='\t',index=False)
    pd.DataFrame(weights).to_csv(OUT/'interval_bin_weights.tsv',sep='\t',index=False)
    summaries = []
    for cohort, normal_id in matching.items():
        for region in WINDOWS:
            for metric in ['mean_signal', 'max_bin_signal']:
                x = values[(values.source=='tumor') & (values.cohort==cohort) & (values.region==region)][metric]
                normal = float(values[(values.source=='normal') & (values.source_id==normal_id) &
                                      (values.region==region)][metric].iloc[0]) if normal_id else None
                summaries.append({'cohort':cohort, 'region':region, 'metric':metric,
                    'n_tumors':len(x), 'normal_id':normal_id,
                    'normal_label':manifest.loc[normal_id,'short_label'] if normal_id else '',
                    'match_class':manifest.loc[normal_id,'match_class'] if normal_id else '',
                    'normal_reference_signal':normal, 'tumor_median':float(x.median()),
                    'tumor_min':float(x.min()), 'tumor_max':float(x.max()),
                    'tumors_above_reference':int((x>normal).sum()) if normal is not None else None,
                    'tumors_equal_reference':int((x==normal).sum()) if normal is not None else None})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'cohort_summary.tsv',sep='\t',index=False)
    rng = np.random.default_rng(1729)
    for label, cohorts in [('fig4d_cohorts', ORIGINAL_COHORTS), ('all_cohorts', sorted(matching))]:
        for metric in ['mean_signal','max_bin_signal']:
            compact = label == 'fig4d_cohorts'
            if compact:
                fig, axes = plt.subplots(1,2,figsize=(183/25.4,72/25.4))
                fig.subplots_adjust(left=.075,right=.99,bottom=.17,top=.72,wspace=.25)
            else:
                fig, axes = plt.subplots(2,1,figsize=(183/25.4,110/25.4))
                fig.subplots_adjust(left=.10,right=.985,bottom=.16,top=.83,hspace=.65)
            selected_normals = [matching[c] for c in cohorts if matching[c]]
            relevant = values[(values.source.eq('tumor') & values.cohort.isin(cohorts)) |
                              (values.source.eq('normal') & values.source_id.isin(selected_normals))]
            ymax = min(1.02, max(.05, float(relevant[metric].max())*1.12))
            for panel, (region, ax) in enumerate(zip(WINDOWS, axes)):
                for i, cohort in enumerate(cohorts):
                    x = values[(values.source=='tumor') & (values.cohort==cohort) & (values.region==region)][metric].to_numpy()
                    if len(x)>=3 and np.ptp(x)>1e-10:
                        parts=ax.violinplot(x,positions=[i],widths=.72,showextrema=False,bw_method=.35)
                        for body in parts['bodies']:
                            body.set_facecolor('#DB5829');body.set_edgecolor('#DB5829');body.set_alpha(.22)
                    ax.scatter(i+rng.uniform(-.18,.18,len(x)),x,s=6,color='#DB5829',alpha=.7,linewidths=0,zorder=3)
                    ax.plot([i-.15,i+.15],[np.median(x)]*2,color='black',lw=.8,zorder=4)
                    normal_id=matching[cohort]
                    if normal_id:
                        y=summary[(summary.cohort==cohort)&(summary.region==region)&(summary.metric==metric)].normal_reference_signal.iloc[0]
                        ax.plot([i-.35,i+.35],[y,y],color='#1964B0',lw=1,zorder=5)
                    ax.text(i,1.025,f'n={len(x)}',transform=ax.get_xaxis_transform(),ha='center',fontsize=5)
                ax.set_xlim(-.6,len(cohorts)-.4)
                ax.set_xticks(range(len(cohorts)),cohorts,rotation=0 if compact else 45,ha='center' if compact else 'right')
                ax.tick_params(axis='both',labelsize=5.5)
                ax.set_ylim(-ymax*.015,ymax)
                ax.set_ylabel((('Mean' if metric=='mean_signal' else 'Maximum')+' relative ATAC signal') if not compact or panel==0 else '')
                ax.set_title(('Promoter' if region=='Promoter' else 'TEAD enhancer')+' (±500 bp)',pad=17,fontsize=7,fontweight='normal')
                ax.text(-.15 if compact else -.08,1.23,chr(97+panel),transform=ax.transAxes,fontweight='bold',fontsize=8)
                ax.spines[['top','right']].set_visible(False)
            if not compact:
                axes[-1].set_xlabel('Cancer type')
            fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#DB5829',label='Tumor donor'),
                       Line2D([],[],color='black',label='Tumor median'),
                       Line2D([],[],color='#1964B0',label='Normal')],
                       loc='upper center',ncol=3,frameon=False,fontsize=6,
                       bbox_to_anchor=(.5,.99),handlelength=1.5,columnspacing=1.5)
            stem=OUT/f'{label}_{metric}'
            for ext in ['pdf','svg','png']:
                fig.savefig(stem.with_suffix('.'+ext),dpi=450,facecolor='white')
            plt.close(fig)
    audit={'status':'AUTHOR_REVIEW_FIRST_PASS','n_tumors':len(tumors),'n_cohorts':tumors.cohort.nunique(),
           'fig4d_tumor_donors':int(tumors.cohort.isin(ORIGINAL_COHORTS).sum()),
           'n_normal_profiles':len(normals),'windows_hg38_half_open':WINDOWS,
           'all_tumors_retained':True,'noise_exclusions':0,'signal_gates':None,
           'no_normal_reference':[c for c,n in matching.items() if n is None],
           'normalization':'Existing Fig4D track values: background-subtracted, clipped at zero, divided by each track own 1Mb maximum; technical replicates already averaged for tumors.',
           'display_scale':'Original shared pair y-scale changes axes only; no additional pair scaling applied here. Both panels in each export share limits based on the included values; underlying values remain on the 0–1 scale.',
           'within_bin_approximation':'100bp source bins; partial bins weighted by overlap for mean/area. Maximum is maximum overlapping bin, not base-resolution maximum.',
           'source_grid_check':'Tumor and normal grids differ slightly; each source uses its own genomic bin coordinates. Original builder reassigns normal columns to tumor headers by position; this new regional summary does not.',
           'violin':'Only n>=3 with nonconstant values; observed-range KDE plus every donor point; no outlier exclusion.',
           'inference':'Descriptive relative signal, not absolute accessibility/fold change or tumor-normal hypothesis tests.',
           'software':{'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
           'inputs':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in INPUTS.items()}}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'README.txt').write_text('AUTHOR REVIEW — first-pass comparison using the existing Fig.4D normalization.\n\n'
      'Start with fig4d_cohorts_mean_signal.pdf: the seven TCGA cancer types visible in submitted Fig4D; all 120 donors, rather than only the strongest donor. Cell lines remain in the original browser figure and are not pooled with patient donors here.\n'
      'fig4d_cohorts_max_bin_signal.pdf uses the maximum overlapping 100bp bin. Both endpoints use exactly 1kb windows. Mean equals area divided by 1000bp; area is retained in source_data.tsv. The promoter is centered on original TRANSCRIPT_START_0=112333155, and enhancer on original TEAD motif midpoint=112360767. This updates the original asymmetric promoter/501bp ROI windows.\n'
      'all_cohorts files include every recovered TCGA donor (404/23 cohorts), including those without a normal reference. Normal matching follows original priority and stable manifest order, without choosing references by their signal. cohort_summary.tsv records each exact normal identity and proxy class.\n\n'
      'Source bin grids differ slightly between tumors and normals. This summary respects each source genomic headers and uses overlap weighting separately. The original browser builder relabels normal bins using tumor headers by position; no such relabeling is used here. This is a minor coordinate-alignment correction, not a change to normalized source values.\n\n'
      'Normalization check: original tracks subtract mean of the lowest 80% of 100bp bins in a 1Mb window, clip negative values, and divide by their own background-subtracted 1Mb maximum. Tumor/normal share the plotted y-axis within a pair; that does not create an absolute calibration. These plots compare relative local signal using those SAME stored normalized tracks, never mixing publisher-normalized tumors with maximum-normalized normals. No extra normalization or background subtraction is performed.\n\n'
      'Normal lines are one cell-type pseudobulk per cohort, not normal-donor distributions; all are displayed as solid blue lines labeled Normal. Reference identities and match classes, including developmental/lineage/tissue proxies, remain in cohort_summary.tsv. The same reference may serve multiple cohorts. PRAD and TGCT have no normal line. There are no p-values, independent normal replicates, or biological activity thresholds. All tumor donors retained, no noise/outlier exclusions. Cross-study assay, cell composition, proxy-lineage and independent maximum normalization limitations remain. Promoter accessibility is not an RNA expression measurement.\n\n'
      'Rebuild: python '+str(Path(__file__).resolve())+'\n')
    print(json.dumps({k:audit[k] for k in ['n_tumors','n_cohorts','fig4d_tumor_donors','no_normal_reference']},indent=2))

if __name__=='__main__': main()
