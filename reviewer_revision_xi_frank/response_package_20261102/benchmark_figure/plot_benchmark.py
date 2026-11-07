#!/usr/bin/env python3
"""Render the three completed benchmarks from frozen evaluation JSONs only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import numpy as np

REPO=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SKILL=Path('/home/xf2217/.codex/skills/nature-figure')
MODELS=['MoDL','ChromBPNet_fresh','ChromBPNet_published','ExplaiNN']
LABELS={'MoDL':'MoDL','ChromBPNet_fresh':'ChromBPNet (fresh)',
        'ChromBPNet_published':'ChromBPNet (published)','ExplaiNN':'ExplaiNN'}
COLORS={'MoDL':'#1964B0','ChromBPNet_fresh':'#DB5829',
        'ChromBPNet_published':'#008A69','ExplaiNN':'#882D71'}
MARKERS={'MoDL':'o','ChromBPNet_fresh':'s','ChromBPNet_published':'^','ExplaiNN':'D'}
PANELS=[('a','peaks','count_pearson_ln1p','Peak counts','Count correlation (Pearson r)',(.48,.77),[.5,.6,.7]),
        ('b','peaks','profile_js_distance','Peak profiles','Profile JS distance (natural log)',(.30,.52),[.30,.35,.40,.45,.50]),
        ('c','all','count_pearson_ln1p','Peaks + background counts','Count correlation (Pearson r)',(.55,.86),[.6,.7,.8])]
DEFAULTS={
 'GM12878':REPO/'reviewer_revision_xi_frank/gm12878_benchmark/evaluation_matched_training_restart_20261027T025330Z/evaluation.json',
 'K562':REPO/'reviewer_revision_xi_frank/k562_benchmark/evaluation_matched_training_storage_retry/evaluation.json',
 'Mouse cSCC':Path('/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/evaluation_final/evaluation.json')}


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_inputs(paths):
    docs={};provenance=[]
    for dataset,path in paths.items():
        d=json.loads(path.read_text())
        assert d['bootstrap']['replicates']==2000
        assert d['bootstrap']['block_bp']==1_000_000
        assert d['bootstrap']['seed']==1729
        assert d['contract']['model_input_length']==2114
        assert d['contract']['target_start']==500 and d['contract']['target_end']==1500
        expected_transform='ln1p_stored_atac_count' if dataset=='Mouse cSCC' else 'ln1p_raw_insertion_count'
        assert d['contract']['count_transform']==expected_transform
        provenance.append({'role':'evaluation','dataset':dataset,'path':str(path.resolve()),'sha256':digest(path)})
        # Verify the lightweight prediction metadata against the frozen evaluation.
        # Checkpoints/large prediction arrays are not re-read to draw a figure.
        for s in d['sources']:
            if not s.get('model') or not s['path'].endswith('prediction_metadata.json'):continue
            p=Path(s['path']);p=p if p.is_absolute() else REPO/p
            assert digest(p)==s['sha256'],f'Prediction metadata changed: {p}'
            provenance.append({'role':'prediction_metadata','dataset':dataset,'model':s['model'],
                               'path':str(p.resolve()),'sha256':digest(p)})
            if s['model']!='ChromBPNet_published':
                training=p.parent.parent/'metadata.json';meta=json.loads(training.read_text())
                assert meta['seed']==1234
                provenance.append({'role':'training_metadata','dataset':dataset,'model':s['model'],
                                   'path':str(training.resolve()),'sha256':digest(training),'seed':meta['seed']})
        docs[dataset]=d
    return docs,provenance


def source_rows(docs):
    rows=[]
    for panel,scope,metric,*_ in PANELS:
        for dataset,d in docs.items():
            for model in MODELS:
                source_model='ChromBPNet' if dataset=='Mouse cSCC' and model=='ChromBPNet_fresh' else model
                record=d['scopes'][scope].get(source_model)
                row={'panel':panel,'dataset':dataset,'scope':scope,'metric':metric,'model':model,
                     'source_model':source_model,'value':'','ci95_lower':'','ci95_upper':'',
                     'n_regions':'','n_defined':'','n_undefined':'','status':'evaluated'}
                if record is None:
                    row['status']='N/A: published checkpoint not included for mouse'
                elif metric not in record['metrics']:
                    assert model=='ExplaiNN' and metric=='profile_js_distance'
                    row['status']='N/A: native ExplaiNN is scalar-only'
                else:
                    m=record['metrics'][metric];ci=m['ci95']
                    assert m['value'] is not None and ci and ci['valid_replicates']==2000
                    assert ci['lower']<=m['value']<=ci['upper']
                    row.update(value=m['value'],ci95_lower=ci['lower'],ci95_upper=ci['upper'],
                               n_regions=record['n_regions'],n_defined=m['n_defined'],n_undefined=m['n_undefined'])
                rows.append(row)
    assert len(rows)==36 and sum(r['status']=='evaluated' for r in rows)==30
    return rows


def write_tsv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)


def caption(docs):
    return '''Chromosome-held-out accessibility prediction across three datasets. (a) Pearson correlation between predicted and observed ln(1 + total counts) in held-out peak windows. (b) Mean per-window Jensen–Shannon (JS) distance between predicted and observed 1-bp profile distributions in held-out peak windows; smaller values indicate better agreement. JS distance is the square root of natural-log JS divergence. (c) Count correlation across the complete held-out evaluation sets, including peaks and sampled background regions. All outputs cover the same central 1,000 bp of a 2,114-bp sequence input. Points are checkpoint point estimates; bars are 95% percentile confidence intervals from 2,000 paired chromosome-stratified genomic-block bootstrap replicates (nominal 1-Mb blocks merged where input windows overlap; bootstrap seed 1729). Intervals quantify sampling uncertainty conditional on each trained checkpoint, not variation across training runs or experimental replicates. Newly trained models use one predeclared split per dataset and one training seed (1234). Published human ChromBPNet checkpoints are evaluated as released; they are not additional independently trained repeats. Native ExplaiNN predicts scalar counts and has no profile output; its missing profile points are N/A, not zero. No published mouse ChromBPNet comparator is included.

The human held-out chromosomes are chr1, chr3 and chr6; the mouse held-out chromosomes are chr16, chr8 and chr9. Peak-window counts are 62,929 (GM12878), 66,474 (K562) and 13,593 (mouse cSCC); complete evaluation-set sizes are 314,645, 332,370 and 26,350, respectively. Human targets are insertion counts; mouse targets are unsmoothed per-base ATAC counts from the bundled Zarr store, for which original Tn5 endpoint/shift provenance has not been independently established. Absolute JS distances are therefore not interpreted as a cross-dataset ranking. These are fixed held-out region sets, not an exhaustive base-by-base genome scan.

MoDL has the highest peak count correlation and lowest peak profile JS distance among the evaluated models in all three datasets. The complete-region count comparison favors ChromBPNet in the two human benchmarks. In mouse, its count-correlation point estimate is slightly higher, but the paired 95% confidence interval for the difference includes zero. Model comparisons should use the paired difference intervals in the source evaluation files rather than infer significance from overlap of the individual intervals shown here.
'''


def make_figure(rows,output):
    spec=importlib.util.spec_from_file_location('nature_matplotlib',SKILL/'scripts/nature_matplotlib.py')
    style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style);style.set_nature_style()
    # Register actual Arial files, then fail rather than silently substitute fonts.
    for p in Path('/usr/share/fonts/truetype/msttcorefonts').glob('Arial*.ttf'):font_manager.fontManager.addfont(str(p))
    font=font_manager.findfont('Arial',fallback_to_default=False)
    matplotlib.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':7,
        'axes.titleweight':'normal','xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5})
    fig=plt.figure(figsize=(180/25.4,80/25.4),facecolor='white')
    axes=[];xpos=np.arange(3);offsets=dict(zip(MODELS,[-.24,-.08,.08,.24]))
    for i,(panel,scope,metric,title,ylabel,ylim,ticks) in enumerate(PANELS):
        ax=fig.add_axes([[.075,.20,.23,.58],[.402,.20,.23,.58],[.725,.20,.23,.58]][i]);axes.append(ax)
        ax.set_gid(f'panel_{panel}_{scope}_{metric}')
        for r in rows:
            if r['panel']!=panel or r['status']!='evaluated':continue
            x=list(DEFAULTS).index(r['dataset'])+offsets[r['model']]
            y=r['value'];lo,hi=r['ci95_lower'],r['ci95_upper']
            artists=ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt=MARKERS[r['model']],
                markersize=3.8,color=COLORS[r['model']],markerfacecolor=COLORS[r['model']],
                markeredgecolor='black',markeredgewidth=.3,elinewidth=.7,capsize=2,capthick=.7,
                linestyle='none',zorder=3)
            artists.lines[0].set_gid(f'{panel}_{r["dataset"]}_{r["model"]}_estimate')
        ax.set_xlim(-.55,2.55);ax.set_ylim(*ylim);ax.set_yticks(ticks)
        ax.set_xticks(xpos,list(DEFAULTS));ax.set_xlabel('Held-out dataset',labelpad=5)
        ax.set_ylabel(ylabel,labelpad=4);ax.set_title(title,pad=7)
        style.add_panel_label(ax,panel,x=-.24,y=1.075)
        ax.tick_params(axis='x',pad=4)
    handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],linestyle='none',
        markeredgecolor='black',markeredgewidth=.3,markersize=4,label=LABELS[m]) for m in MODELS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.965),ncol=4,
               handletextpad=.4,columnspacing=1.5,frameon=False)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();canvas=fig.bbox
    clipping=[]
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        box=text.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>canvas.width+.5 or box.y1>canvas.height+.5:
            clipping.append(text.get_text())
    assert not clipping,clipping
    for ext in ['pdf','svg']:
        fig.savefig(output/f'benchmark_main.{ext}',bbox_inches=None,facecolor='white',transparent=False,dpi=450)
    fig.savefig(output/'benchmark_main.png',bbox_inches=None,facecolor='white',transparent=False,dpi=450)
    # Final-size grayscale QA proof; data, axes and editable main exports unchanged.
    from PIL import Image,ImageOps
    im=Image.open(output/'benchmark_main.png')
    ImageOps.grayscale(im).save(output/'benchmark_main_grayscale.png')
    # QA only: Machado et al. 2009, severity100 linear-sRGB matrices.
    # Primary implementation/data: https://github.com/njsmith/colorspacious/blob/master/colorspacious/cvd.py
    matrices={
      'protanopia':[[.152286,1.052583,-.204868],[.114503,.786281,.099216],[-.003882,-.048116,1.051998]],
      'deuteranopia':[[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.011820,.042940,.968881]],
      'tritanopia':[[1.255528,-.076749,-.178779],[-.078411,.930809,.147602],[.004733,.691367,.303900]]}
    rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    for name,matrix in matrices.items():
        transformed=np.clip(linear@np.asarray(matrix,dtype=np.float32).T,0,1)
        srgb=np.where(transformed<=.0031308,12.92*transformed,1.055*transformed**(1/2.4)-.055)
        Image.fromarray(np.rint(255*srgb).astype(np.uint8)).save(output/f'benchmark_main_{name}.png')
    info={'width_mm':180,'height_mm':80,'font_family':'Arial','font_file':font,
          'editable_svg_text':True,'pdf_fonttype':42,'preview_dpi':450,
          'png_size_px':list(im.size),'text_clipping_failures':clipping,
          'series':[{'model':m,'color':COLORS[m],'marker':MARKERS[m]} for m in MODELS]}
    plt.close(fig)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--gm-json',type=Path,default=DEFAULTS['GM12878'])
    p.add_argument('--k562-json',type=Path,default=DEFAULTS['K562'])
    p.add_argument('--mouse-json',type=Path,default=DEFAULTS['Mouse cSCC'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    docs,provenance=load_inputs({'GM12878':a.gm_json,'K562':a.k562_json,'Mouse cSCC':a.mouse_json})
    rows=source_rows(docs);write_tsv(a.output/'source_data.tsv',rows)
    (a.output/'caption.txt').write_text(caption(docs))
    paired=[]
    for dataset,d in docs.items():
        for x in d['paired_deltas']:
            if (x['scope']=='peaks' and x['metric'] in ['count_pearson_ln1p','profile_js_distance'] or
                x['scope']=='all' and x['metric']=='count_pearson_ln1p') and 'MoDL' in [x['model_a'],x['model_b']]:
                paired.append({'dataset':dataset,'scope':x['scope'],'model_a':x['model_a'],'model_b':x['model_b'],
                    'metric':x['metric'],'delta_a_minus_b':x['delta_a_minus_b'],
                    'ci95_lower':x['ci95']['lower'],'ci95_upper':x['ci95']['upper'],
                    'n_common_defined_regions':x['n_common_defined_regions']})
    write_tsv(a.output/'paired_comparisons.tsv',paired)
    figure=make_figure(rows,a.output)
    manifest={'figure_message':'MoDL leads peak metrics in these runs; full-region count results retain ChromBPNet advantages.',
        'profile':'main research figure; conservative Nature baseline',
        'dimensions':figure,'input_provenance':provenance,
        'uncertainty':'2000 paired genomic-block bootstrap replicates, conditional on checkpoint; training seed1234 once per fresh model',
        'runtime':{'python':platform.python_version(),'matplotlib':matplotlib.__version__,'numpy':np.__version__,'platform':platform.platform()},
        'source_script':{'path':str(Path(__file__).resolve()),'sha256':digest(__file__)},
        'skill_helper':{'path':str(SKILL/'scripts/nature_matplotlib.py'),'sha256':digest(SKILL/'scripts/nature_matplotlib.py')},
        'warnings':['180mm canvas follows requested size; strict Nature main-figure baseline is183mm. Resize only if target journal requires it.',
                    'Target journal not specified; final journal-specific artwork requirements remain to be confirmed.',
                    'SVG retains editable text but font is referenced, not embedded; PDF font embedding checked separately.',
                    'Error bars are region-sampling intervals, not seed-to-seed reproducibility.'],
        'artifacts':[{'path':x.name,'sha256':digest(x),'bytes':x.stat().st_size} for x in sorted(a.output.iterdir())
                     if x.is_file() and x.name in ['benchmark_main.pdf','benchmark_main.svg','benchmark_main.png',
                       'benchmark_main_grayscale.png','benchmark_main_protanopia.png','benchmark_main_deuteranopia.png',
                       'benchmark_main_tritanopia.png','source_data.tsv','paired_comparisons.tsv','caption.txt']]}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'plotted_estimates':30,'source_rows':36,'dimensions_mm':[180,80]}))

if __name__=='__main__':main()
