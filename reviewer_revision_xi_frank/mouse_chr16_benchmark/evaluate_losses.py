#!/usr/bin/env python3
"""Common held-out profile/count losses; conditional test evaluation, no fitting."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import gammaln

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--prediction',action='append',required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--count-weight',type=float,required=True)
 a=p.parse_args();rows=list(csv.DictReader((a.cache_dir/'test.regions.tsv').open(),delimiter='\t'))
 ids=hashlib.sha256(''.join(r['region_id']+'\n' for r in rows).encode()).hexdigest()
 y=np.load(a.cache_dir/'test.y.npy',mmap_mode='r')[:,500:1500];total=y.sum(1,dtype=np.float64);obs=np.log1p(total)
 peak=np.array([r['is_peak'] in ['1','true','True'] for r in rows]);chrom=np.array([r['chrom'] for r in rows])
 scopes={'all':np.ones(len(rows),bool),'peaks':peak,'nonpeaks':~peak}
 scopes.update({c:chrom==c for c in sorted(set(chrom))});results=[]
 for spec in a.prediction:
  name,path=spec.split('=',1);directory=Path(path);m=json.loads((directory/'prediction_metadata.json').read_text())
  if m.get('region_ids_sha256')!=ids or m.get('count_transform')!='ln1p_stored_atac_count':raise ValueError('Prediction identity/units mismatch')
  pred=np.load(directory/'prediction_counts.npy');assert pred.shape==obs.shape and np.isfinite(pred).all()
  count=(pred-obs)**2;profile_file=directory/'prediction_profiles.npy';nll=constant=None
  if profile_file.exists():
   probs=np.load(profile_file,mmap_mode='r');assert probs.shape==y.shape
   nll=np.empty(len(rows));constant=np.empty(len(rows))
   for i in range(0,len(rows),1024):
    z=np.asarray(y[i:i+1024],dtype=np.float64);q=np.asarray(probs[i:i+1024],dtype=np.float64)
    if not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-5):raise ValueError('Invalid profile')
    nll[i:i+len(z)]=-(z*np.log(np.maximum(q,1e-8))).sum(1)
    constant[i:i+len(z)]=gammaln(z.sum(1)+1)-gammaln(z+1).sum(1)
  for scope,mask in scopes.items():
   if not mask.any():continue
   row={'model':name,'scope':scope,'n_regions':int(mask.sum()),'count_log_mse':float(count[mask].mean()),'profile_nll_without_constant':None,'profile_multinomial_nll':None,'profile_cross_entropy_per_observed_count':None,'combined_loss_without_constant':None}
   if nll is not None:
    row.update(profile_nll_without_constant=float(nll[mask].mean()),profile_multinomial_nll=float((nll-constant)[mask].mean()),combined_loss_without_constant=float((nll+a.count_weight*count)[mask].mean()))
    positive=mask&(total>0)
    if positive.any():row['profile_cross_entropy_per_observed_count']=float((nll[positive]/total[positive]).mean())
   results.append(row)
 a.output.mkdir(parents=True,exist_ok=False)
 report={'count_transform':'ln1p_stored_atac_count','target_units':'unsmoothed_per_base_ATAC_counts_from_bundled_Zarr','profile_loss_probability_floor':1e-8,'count_weight':a.count_weight,'scope':'all eligible heldout rows; zeros retained; no test fitting; original manuscript loss uses different targets','rows':results}
 (a.output/'losses.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 with (a.output/'losses.tsv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]),delimiter='\t');w.writeheader();w.writerows(results)
 print(json.dumps({'losses':str(a.output/'losses.json')}))
if __name__=='__main__':main()
