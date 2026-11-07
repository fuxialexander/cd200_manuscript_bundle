#!/usr/bin/env python3
"""Read fresh SCCAneg targets on the immutable positive benchmark region universe."""
from __future__ import annotations
import argparse,copy,csv,hashlib,importlib.util,json,shutil,time
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[3]
PREP=REPO/'reviewer_revision_xi_frank/mouse_chr16_benchmark/prepare_mouse_cache.py'
spec=importlib.util.spec_from_file_location('mouse_cache_source',PREP)
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
SAMPLE='SCCAneg.cd200_scc'
UNITS='unsmoothed_per_base_ATAC_counts_from_bundled_Zarr'

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--positive-cache',type=Path,default=Path('/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16/cache'))
 p.add_argument('--signal-zarr',type=Path,default=REPO/'data/zarr/cd200_scc_dense_new.zarr')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists() or a.report.exists():p.error('Fresh output/report paths required; preserve earlier artifacts')
 posmanifest=a.positive_cache/'cache_manifest.json';base=json.loads(posmanifest.read_text())
 if base['status']!='complete' or base['sample']!='SCCApos.cd200_scc' or base['assembly']!='mm10':
  raise ValueError('Expected verified completed positive mm10 cache')
 if set(base['fold']['test'])!={'chr16','chr8','chr9'} or base['fold']['valid']!=['chr14']:
  raise ValueError('Unexpected frozen chromosome split')
 attrs=a.signal_zarr/SAMPLE/'.zattrs';metadata=json.loads(attrs.read_text())
 if metadata['assembly']!='mm10' or metadata['celltype']!='SCCAneg':raise ValueError('Negative sample identity mismatch')
 signal=source.DenseZarrReader(a.signal_zarr/SAMPLE/'chrs','<u2')
 a.output.mkdir(parents=True,exist_ok=False);a.report.parent.mkdir(parents=True,exist_ok=True)
 report=copy.deepcopy(base)
 report.update(status='in_progress',sample=SAMPLE,cache_directory=str(a.output),splits={},
   normalization='none',smoothing='none',test_outcome_filtering=False,
   source_code_sha256=sha(__file__),source_reader_sha256=sha(PREP),
   positive_universe_reuse={'manifest':str(posmanifest),'manifest_sha256':sha(posmanifest),
      'policy':'exact same eligible rows/order/splits and DNA; x arrays linked read-only at consumer mmap; new negative target arrays',
      'source_sample':'SCCApos.cd200_scc','peak_label_universe':'unchanged shared source region peak_type classification'},
   zarr_root_metadata={str(p):{'sha256':sha(p),'content':json.loads(p.read_text())}
      for p in [a.signal_zarr/'.zattrs',attrs]},
   genome_sequence_concordance_provenance='Inherited from unchanged positive x arrays; each linked file SHA256 reverified against completed cache manifest',
   signal_sample_metadata=metadata)
 for split in ['train','valid','test','loci']:
  t=time.monotonic();regions=a.positive_cache/f'{split}.regions.tsv';x=a.positive_cache/f'{split}.x.npy'
  old=base['splits'][split]
  if sha(regions)!=old['regions_sha256'] or sha(x)!=old['x_sha256']:
   raise ValueError('Positive source region/DNA artifact changed')
  rows=list(csv.DictReader(regions.open(),delimiter='\t'))
  if len(rows)!=old['regions']:raise ValueError('Source manifest region count mismatch')
  if any((r['chrom'] not in base['fold'][split]) for r in rows) if split!='loci' else any(r['chrom']!='chr16' for r in rows):
   raise ValueError('Unexpected chromosome in frozen split')
  shutil.copyfile(regions,a.output/regions.name)
  (a.output/x.name).symlink_to(x.resolve())
  y=np.lib.format.open_memmap(a.output/f'{split}.y.npy',mode='w+',dtype=np.float32,shape=(len(rows),2000))
  maximum=zeros=0
  for i,r in enumerate(rows):
   center=int(r['center']);v=signal.read(r['chrom'],center-1000,center+1000)
   if v.shape!=(2000,) or v.dtype!=np.uint16:raise ValueError('Unexpected negative signal representation')
   y[i]=v;maximum=max(maximum,int(v.max()));zeros+=int((v==0).sum())
   if (i+1)%20000==0:print(f'{split}: {i+1}/{len(rows)}',flush=True)
  y.flush();del y
  v=copy.deepcopy(old);v.update(y_sha256=sha(a.output/f'{split}.y.npy'),maximum_stored_count=maximum,
   zero_signal_bases=zeros,signal_bases=len(rows)*2000,elapsed_seconds=time.monotonic()-t,
   DNA_reuse='symlink to checksum-verified positive-cache sequence array',
   x_source_path=str(x.resolve()),y_target_sample=SAMPLE)
  report['splits'][split]=v;save(a.report,report)
  print(f'{split}: complete {len(rows)} rows in {time.monotonic()-t:.1f}s',flush=True)
 for name in ['locus_windows.tsv','excluded_regions.jsonl']:
  shutil.copyfile(a.positive_cache/name,a.output/name)
 windows=list(csv.DictReader((a.output/'locus_windows.tsv').open(),delimiter='\t'))
 observed=np.load(a.output/'loci.y.npy',mmap_mode='r')[:,500:1500]
 np.save(a.output/'observed_profiles.npy',np.asarray(observed))
 report['stored_signal_locus_audit']=[{'window_id':w['window_id'],'locus':w['locus'],'role':w['role'],
   'output_start':int(w['output_start']),'output_end':int(w['output_end']),
   'stored_signal_sum':float(v.sum(dtype=np.float64)),'nonzero_bases':int(np.count_nonzero(v)),
   'maximum_stored_count':int(v.max()),'all_values_integer':True} for w,v in zip(windows,observed,strict=True)]
 save(a.output/'target_metadata.json',{'genome_build':'mm10','target_units':UNITS,
  'target_kind':'stored_atac_signal_counts','sample':SAMPLE,'input_length':2114,'output_length':1000,
  'count_transform':'ln1p_stored_atac_count','window_ids_sha256':source.hashlib.sha256(''.join(w['window_id']+'\n' for w in windows).encode()).hexdigest(),
  'locus_windows_sha256':sha(a.output/'locus_windows.tsv'),'source_cache':str(a.output/'loci.y.npy'),
  'upstream_provenance':base['upstream_provenance']})
 chunklog=a.output/'source_chunks.json';save(chunklog,{'signal':signal.sources,'genome_sequence_checks':'inherited unchanged DNA source hashes'})
 report['source_chunk_log']={'path':str(chunklog),'sha256':sha(chunklog),'signal_chunks_and_metadata':len(signal.sources)}
 # Independent indexing checks on actual cache rows, before fitting.
 checks=[];rng=np.random.default_rng(1234)
 for split in ['train','valid','test','loci']:
  rows=list(csv.DictReader((a.output/f'{split}.regions.tsv').open(),delimiter='\t'))
  y=np.load(a.output/f'{split}.y.npy',mmap_mode='r')
  for i in sorted(set([0,len(rows)-1,*rng.choice(len(rows),min(8,len(rows)),replace=False).tolist()])):
   r=rows[i];center=int(r['center']);direct=signal.read(r['chrom'],center-1000,center+1000)
   match=bool(np.array_equal(y[i],direct));checks.append({'split':split,'index':i,'region_id':r['region_id'],'matches_negative_source':match})
   if not match:raise ValueError('Cache/source negative target alignment failed')
  if report['splits'][split]['y_sha256']==base['splits'][split]['y_sha256']:
   raise ValueError('Negative targets unexpectedly identical to positive targets')
 report['independent_negative_target_index_checks']=checks
 report['status']='complete';save(a.report,report);save(a.output/'cache_manifest.json',report)
 print('Negative cache complete',flush=True)

if __name__=='__main__':main()
