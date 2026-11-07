#!/usr/bin/env python3
"""Persistent negative-state-only native ChromBPNet training/prediction job."""
from __future__ import annotations
import csv,hashlib,json,os,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
MOUSE=REPO/'reviewer_revision_xi_frank/mouse_chr16_benchmark'
SHARED=REPO/'reviewer_revision_xi_frank/gm12878_benchmark'
STORE=Path('/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16_negative_20261103')
CACHE=STORE/'cache'
TF=REPO.parent/'get_model_seq/.venv/bin/python'
PY=REPO.parent/'motif_chip_model/.venv/bin/python'


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)


def main():
 statuspath=HERE/'job_status.json'
 if statuspath.exists():raise FileExistsError('Fresh single-run supervisor only; inspect preserved previous status before recovery')
 manifest=json.loads((CACHE/'cache_manifest.json').read_text())
 if manifest['sample']!='SCCAneg.cd200_scc' or manifest['status']!='complete':raise ValueError('Expected completed negative-state cache')
 if set(manifest['fold']['test'])!={'chr16','chr8','chr9'} or manifest['fold']['valid']!=['chr14']:
  raise ValueError('Requested immutable split differs')
 audit=json.loads((HERE/'training_target_audit.json').read_text())
 if audit['cache_manifest_sha256']!=sha(CACHE/'cache_manifest.json'):raise ValueError('Cache changed since training audit')
 weight=audit['count_loss_weight']
 run=STORE/'runs/chrombpnet_seed1234';bias=STORE/'runs/negative_bias_seed1234';biasmodel=bias/'bias_model_scaled.h5'
 for directory in [run,bias]:
  if directory.exists():raise FileExistsError(f'Existing run preserved: {directory}')
 (HERE/'logs').mkdir(exist_ok=True)
 status={'state':'starting','started_utc':now(),'updated_utc':now(),'supervisor_pid':os.getpid(),
  'sample':'SCCAneg.cd200_scc','storage':str(STORE),'cache_manifest_sha256':sha(CACHE/'cache_manifest.json'),
  'configuration':{'seed':1234,'max_epochs':50,'patience':5,'batch_size':64,'negative_ratio':0.1,
   'learning_rate':.001,'count_loss_weight':weight,'count_weight_rule':audit['count_weight_rule'],
   'test_chromosomes':['chr16','chr8','chr9'],'validation_chromosomes':['chr14'],
   'bias_initialization':'random; newly fitted negative-state training nonpeaks only',
   'positive_or_human_checkpoint_used':False},'stages':[]}
 save(statuspath,status)
 srcpaths=[Path(__file__),HERE/'prepare_negative_cache.py',MOUSE/'prepare_mouse_cache.py',MOUSE/'train_mouse_chrombpnet.py',
  MOUSE/'predict_loci.py',MOUSE/'evaluate_loci.py',MOUSE/'evaluate_mouse.py',MOUSE/'evaluate_losses.py',
  SHARED/'train_chrombpnet.py',SHARED/'evaluate.py',
  SHARED/'vendor/chrombpnet/chrombpnet/training/models/bpnet_model.py',
  SHARED/'vendor/chrombpnet/chrombpnet/training/models/chrombpnet_with_bias_model.py',
  SHARED/'vendor/chrombpnet/chrombpnet/training/utils/losses.py']
 snapshots=STORE/'source_snapshots';snapshots.mkdir(exist_ok=False)
 srcrecord=[]
 for i,p in enumerate(srcpaths):
  snapshot=snapshots/f'{i:02d}_{p.name}';shutil.copyfile(p,snapshot)
  srcrecord.append({'source':str(p),'snapshot':str(snapshot),'sha256':sha(p)})
 save(HERE/'implementation_manifest.json',{'sources':srcrecord,'read_only_positive_cache_reused':manifest['positive_universe_reuse']})
 def stage(name,command,history=None):
  log=HERE/'logs'/f'{name}.log'
  record={'name':name,'command':list(map(str,command)),'started_utc':now(),'state':'running','log':str(log)}
  status['stages'].append(record);status.update(state='running',current_stage=name,updated_utc=now());save(statuspath,status)
  env=dict(os.environ);env.update(PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
  with log.open('x') as handle:
   proc=subprocess.Popen(record['command'],cwd=REPO,stdout=handle,stderr=subprocess.STDOUT,env=env)
   record['pid']=proc.pid;save(statuspath,status)
   while proc.poll() is None:
    time.sleep(20);status['updated_utc']=now()
    if history and Path(history).exists():
     try:
      rows=json.loads(Path(history).read_text());record['last_completed_epoch']=rows[-1] if rows else None
     except (ValueError,IndexError):pass
    save(statuspath,status)
  record.update(state='passed' if proc.returncode==0 else 'failed',exit_code=proc.returncode,finished_utc=now());save(statuspath,status)
  if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; inspect {log}')
 try:
  native=[TF,MOUSE/'train_mouse_chrombpnet.py','--cache-dir',CACHE,'--seed',1234,'--batch-size',64,
    '--heldout-chroms','chr16','chr8','chr9','--max-epochs',50,'--patience',5,'--negative-ratio',.1]
  stage('negative_bias_train',native+['--mode','bias','--output-dir',bias],bias/'history.json')
  stage('negative_chrombpnet_train',native+['--mode','train','--output-dir',run,'--bias-model',biasmodel,'--count-weight',weight],run/'history.json')
  stage('negative_test_predict',native+['--mode','predict','--output-dir',run,'--bias-model',biasmodel,'--count-weight',weight])
  stage('negative_loci_predict',[TF,MOUSE/'predict_loci.py','--model','chrombpnet','--cache-dir',CACHE,
      '--run-dir',run,'--bias-model',biasmodel,'--output',run/'locus_predictions'])
  stage('negative_loci_evaluate',[PY,MOUSE/'evaluate_loci.py','--data-dir',CACHE,'--prediction',f'ChromBPNet_negative={run}/locus_predictions',
      '--output',STORE/'locus_evaluation_final'])
  stage('negative_test_evaluate',[PY,MOUSE/'evaluate_mouse.py','--data-dir',CACHE,'--prediction',f'ChromBPNet_negative={run}/test_predictions',
      '--output',STORE/'evaluation_final','--bootstrap',2000])
  stage('negative_test_losses',[PY,MOUSE/'evaluate_losses.py','--cache-dir',CACHE,'--prediction',f'ChromBPNet_negative={run}/test_predictions',
      '--output',STORE/'losses_final','--count-weight',weight])
  sourceverify=[{'source':r['source'],'unchanged_since_start':sha(r['source'])==r['sha256']} for r in srcrecord]
  if not all(r['unchanged_since_start'] for r in sourceverify):raise ValueError('Training/evaluation source changed during run')
  checkpoints=[biasmodel,bias/'best.weights.h5',run/'best.weights.h5']
  final={'sample':'SCCAneg.cd200_scc','configuration':status['configuration'],
    'bias_training_result':json.loads((bias/'training_result.json').read_text()),
    'full_training_result':json.loads((run/'training_result.json').read_text()),
    'checkpoint_sha256':{str(p):sha(p) for p in checkpoints},'source_unchanged_checks':sourceverify,
    'locus_metrics':str(STORE/'locus_evaluation_final/locus_evaluation.json'),
    'test_metrics':str(STORE/'evaluation_final/evaluation.json'),'locus_predictions':str(run/'locus_predictions'),
    'test_losses':str(STORE/'losses_final/losses.json'),
    'limits':['One training seed and one chromosome split','Stored counts; original Tn5 endpoint provenance unverified',
      'Per-state counts are not normalized for library depth here; contrast calibration is a separate analysis',
      'Native bias model motif purity has not been independently established']}
  save(HERE/'RESULTS.json',final)
  status.update(state='complete',current_stage='complete',finished_utc=now(),updated_utc=now(),results=str(HERE/'RESULTS.json'));save(statuspath,status)
 except BaseException as error:
  status.update(state='failed',error=repr(error),updated_utc=now());save(statuspath,status);raise

if __name__=='__main__':main()
