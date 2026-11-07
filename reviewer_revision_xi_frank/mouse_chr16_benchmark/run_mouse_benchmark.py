#!/usr/bin/env python3
"""Persistent sequential mouse fit/predict/locus-evaluation supervisor."""
import argparse,csv,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
TORCH=ROOT.parent/'motif_chip_model/.venv/bin/python'
TF=ROOT.parent/'get_model_seq/.venv/bin/python'
UNITS='unsmoothed_per_base_ATAC_counts_from_bundled_Zarr'

def stamp():return datetime.now(timezone.utc).isoformat()
def save(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--storage',type=Path,default=Path('/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16'));p.add_argument('--resume',action='store_true');a=p.parse_args()
 storage=a.storage;cache=storage/'cache';state=BASE/'benchmark_status.json'
 if state.exists():
  old=json.loads(state.read_text())
  if not a.resume or old.get('state')=='complete':raise RuntimeError('Status exists; inspect run and use --resume only for interrupted work')
  try:os.kill(old['supervisor_pid'],0)
  except ProcessLookupError:pass
  else:raise RuntimeError('Prior supervisor still alive')
 else:old=None
 storage.mkdir(parents=True,exist_ok=True);(BASE/'logs').mkdir(exist_ok=True)
 status={'state':'waiting_for_cache','started_utc':stamp(),'updated_utc':stamp(),'supervisor_pid':os.getpid(),'storage':str(storage),'stages':[], 'error':None,'target_units':UNITS,'sample':'SCCApos.cd200_scc','configuration':{'seed':1234,'test_chromosomes':['chr16','chr8','chr9'],'validation_chromosomes':['chr14'],'max_epochs':50,'patience':5,'batch_size':64,'negative_ratio':0.1,'historical_weights_used':False}}
 if old:status['previous_attempt']=old
 save(state,status)
 def stage(name,command):
  log=BASE/'logs'/f'{name}.log'
  if log.exists():log=BASE/'logs'/f'{name}_{int(time.time())}.log'
  entry={'name':name,'state':'running','started_utc':stamp(),'command':list(map(str,command)),'log':str(log)};status['stages'].append(entry);status.update(state='running',current_stage=name);save(state,status)
  with log.open('w') as f:
   proc=subprocess.Popen(entry['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);entry['pid']=proc.pid;save(state,status)
   while proc.poll() is None:
    time.sleep(20);status['updated_utc']=stamp()
    if '--output-dir' in entry['command']:
     directory=Path(entry['command'][entry['command'].index('--output-dir')+1]);hist=directory/'history.json'
     if hist.exists():
      try:
       rows=json.loads(hist.read_text());entry['last_completed_epoch']=rows[-1] if rows else None
      except (ValueError,IndexError):pass
    save(state,status)
  entry.update(state='passed' if proc.returncode==0 else 'failed',exit_code=proc.returncode,finished_utc=stamp());save(state,status)
  if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; see {log}')
 def trained(directory):
  r=directory/'training_result.json'
  if r.exists() and not json.loads(r.read_text()).get('pilot',True):return True
  if directory.exists():raise RuntimeError(f'Incomplete run preserved at {directory}; inspect before recovery')
  return False
 def evaluate(preds,suffix):
  out=storage/f'evaluation_{suffix}';cmd=[TORCH,BASE/'evaluate_mouse.py','--data-dir',cache,'--output',out,'--bootstrap',2000]
  losses=[TORCH,BASE/'evaluate_losses.py','--cache-dir',cache,'--output',storage/f'losses_{suffix}','--count-weight',weight]
  loci=[TORCH,BASE/'evaluate_loci.py','--data-dir',cache,'--output',storage/f'locus_evaluation_{suffix}']
  for label,d in preds.items():
   cmd+=['--prediction',f'{label}={d}/test_predictions'];losses+=['--prediction',f'{label}={d}/test_predictions'];loci+=['--prediction',f'{label}={d}/locus_predictions']
  for label,command,marker in [(f'evaluate_{suffix}',cmd,out/'evaluation.json'),(f'losses_{suffix}',losses,storage/f'losses_{suffix}/losses.json'),(f'loci_{suffix}',loci,storage/f'locus_evaluation_{suffix}/locus_evaluation.json')]:
   if not marker.exists():stage(label,command)
  status['latest_evaluation']=str(out/'evaluation.json');status['latest_locus_evaluation']=str(storage/f'locus_evaluation_{suffix}/locus_evaluation.json');save(state,status)
 try:
  start=time.monotonic()
  while not (cache/'cache_manifest.json').exists():
   unit=subprocess.run(['systemctl','--user','show','cd200-mouse-chr16-cache.service','--property=ActiveState','--value'],capture_output=True,text=True)
   if unit.stdout.strip()=='failed':raise RuntimeError('Cache preparation service failed; inspect prepare_cache.log')
   if time.monotonic()-start>7200:raise TimeoutError('Cache completion marker absent after two hours')
   time.sleep(15);status['updated_utc']=stamp();save(state,status)
  manifest=json.loads((cache/'cache_manifest.json').read_text());assert manifest['status']=='complete' and manifest['target_units']==UNITS
  status['cache_manifest_sha256']=sha(cache/'cache_manifest.json')
  train_rows=list(csv.DictReader((cache/'train.regions.tsv').open(),delimiter='\t'));peak=np.array([r['is_peak']=='1' for r in train_rows]);y=np.load(cache/'train.y.npy',mmap_mode='r');counts=y[:,500:1500].sum(1,dtype=np.float64)
  weight=max(1,float(np.median(counts[peak]))/10);status['configuration']['count_weight']=weight
  audit={'training_peak_count':int(peak.sum()),'training_background_count':int((~peak).sum()),'peak_quantiles':np.quantile(counts[peak],[0,.01,.5,.99,1]).tolist(),'background_quantiles':np.quantile(counts[~peak],[0,.01,.5,.99,1]).tolist(),'fit_source':'training only'}
  save(BASE/'training_target_audit.json',audit);save(state,status)
  snap=storage/'source_snapshots';snap.mkdir(exist_ok=True)
  for source in list(BASE.glob('*.py'))+[BASE.parent/'gm12878_benchmark'/n for n in ['train_torch.py','train_chrombpnet.py','evaluate.py']]:
   dest=snap/source.name
   if dest.exists():assert sha(dest)==sha(source),'Source changed since run start'
   else:shutil.copy2(source,dest)
  preds={}
  modl=storage/'runs/modl_seed1234'
  common=[TORCH,BASE/'train_mouse_torch.py','--model','modl','--cache-dir',cache,'--output-dir',modl,'--count-weight',weight,'--seed',1234,'--batch-size',64,'--workers',4]
  if not trained(modl):stage('modl_train',common+['--max-epochs',50,'--patience',5,'--amp'])
  if not (modl/'test_predictions/prediction_metadata.json').exists():stage('modl_predict',common+['--predict'])
  if not (modl/'locus_predictions/prediction_metadata.json').exists():stage('modl_loci_predict',[TORCH,BASE/'predict_loci.py','--model','modl','--cache-dir',cache,'--run-dir',modl,'--output',modl/'locus_predictions'])
  preds['MoDL']=modl;evaluate(preds,'modl_preliminary')
  bias=storage/'runs/mouse_bias_seed1234';biasmodel=bias/'bias_model_scaled.h5'
  native=[TF,BASE/'train_mouse_chrombpnet.py','--cache-dir',cache,'--seed',1234,'--batch-size',64,'--heldout-chroms','chr16','chr8','chr9']
  if not biasmodel.exists():
   if bias.exists():raise RuntimeError('Incomplete bias run preserved; inspect before recovery')
   stage('mouse_bias_train',native+['--mode','bias','--output-dir',bias,'--max-epochs',50,'--patience',5])
  cbp=storage/'runs/chrombpnet_seed1234';common=native+['--output-dir',cbp,'--bias-model',biasmodel,'--count-weight',weight]
  if not trained(cbp):stage('chrombpnet_train',common+['--mode','train','--max-epochs',50,'--patience',5])
  if not (cbp/'test_predictions/prediction_metadata.json').exists():stage('chrombpnet_predict',common+['--mode','predict'])
  if not (cbp/'locus_predictions/prediction_metadata.json').exists():stage('chrombpnet_loci_predict',[TF,BASE/'predict_loci.py','--model','chrombpnet','--cache-dir',cache,'--run-dir',cbp,'--bias-model',biasmodel,'--output',cbp/'locus_predictions'])
  preds['ChromBPNet']=cbp;evaluate(preds,'modl_chrombpnet')
  ex=storage/'runs/explainn_seed1234';common=[TORCH,BASE/'train_mouse_torch.py','--model','explainn','--cache-dir',cache,'--output-dir',ex,'--count-weight',weight,'--seed',1234,'--batch-size',64,'--workers',4]
  if not trained(ex):stage('explainn_train',common+['--max-epochs',50,'--patience',5,'--amp'])
  if not (ex/'test_predictions/prediction_metadata.json').exists():stage('explainn_predict',common+['--predict'])
  if not (ex/'locus_predictions/prediction_metadata.json').exists():stage('explainn_loci_predict',[TORCH,BASE/'predict_loci.py','--model','explainn','--cache-dir',cache,'--run-dir',ex,'--output',ex/'locus_predictions'])
  preds['ExplaiNN']=ex;evaluate(preds,'final')
  d=json.loads((storage/'evaluation_final/evaluation.json').read_text());lines=['# Mouse chromosome-held-out benchmark','',f'SCCApos; test chr16/chr8/chr9, validation chr14. One fold and seed. Targets: {UNITS}. Upstream insertion/Tn5-shift provenance remains unverified.','', '| Model | Peak count Pearson | All-region count Pearson | Peak profile JS distance |','| --- | ---: | ---: | ---: |']
  for name in preds:
   peakmetrics=d['scopes']['peaks'][name]['metrics'];allmetrics=d['scopes']['all'][name]['metrics'];js=peakmetrics.get('profile_js_distance',{}).get('value');lines.append(f"| {name} | {peakmetrics['count_pearson_ln1p']['value']:.4f} | {allmetrics['count_pearson_ln1p']['value']:.4f} | {js if js is not None else 'N/A'} |")
  lines+=['',f'Full metrics and paired block intervals: {storage}/evaluation_final/',f'Common profile/count losses: {storage}/losses_final/',f'P1/P2/promoter native-window and full-interval metrics: {storage}/locus_evaluation_final/','Original human benchmark files and original manuscript checkpoints were not changed.']
  (BASE/'BENCHMARK_RESULTS.md').write_text('\n'.join(lines)+'\n');status.update(state='complete',current_stage='complete',finished_utc=stamp(),updated_utc=stamp());save(state,status)
 except BaseException as e:
  status.update(state='failed',error=repr(e),updated_utc=stamp());save(state,status);raise
if __name__=='__main__':main()
