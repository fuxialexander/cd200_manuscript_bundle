#!/usr/bin/env python3
"""Wait for the negative checkpoint/evaluations, then run authorized S5B outputs."""
import hashlib,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
REPO=WORK.parents[1]
STORE=Path('/mnt/storage/cd200/reviewer_benchmarks/mouse_chr16_negative_20261103')

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def main():
 statepath=HERE/'postcontrast_status.json'
 if statepath.exists():raise FileExistsError('Preserve previous postcontrast state')
 state={'state':'waiting_for_negative_pipeline','started_utc':now(),'commands':[]};save(statepath,state)
 while True:
  status=json.loads((HERE/'job_status.json').read_text())
  if status['state']=='failed':
   state.update(state='blocked_by_failed_training_pipeline',training_error=status.get('error'));save(statepath,state)
   raise RuntimeError('Negative fit/evaluation pipeline failed')
  if status['state']=='complete':break
  time.sleep(20)
 output=WORK/'negative_contrast'
 if output.exists():raise FileExistsError(output)
 for name in ['regenerated_s5b_contrast.pdf','regenerated_s5b_contrast.svg','regenerated_s5b_contrast.png']:
  if (WORK/'figures'/name).exists():raise FileExistsError('Preserve previously generated contrast figure')
 commands=[('negative_state_contrast',[
  str(REPO.parent/'get_model_seq/.venv/bin/python'),str(WORK/'predict_state_contrast.py'),
  '--state','neg','--run-dir',str(STORE/'runs/chrombpnet_seed1234'),
  '--bias-model',str(STORE/'runs/negative_bias_seed1234/bias_model_scaled.h5'),
  '--output',str(output)]),('plot_state_contrast',['/usr/bin/python3',str(WORK/'plot_state_contrast.py')])]
 try:
  for name,command in commands:
   log=HERE/'logs'/f'{name}.log'
   entry={'name':name,'command':command,'source_sha256':sha(command[1]),'state':'running','started_utc':now(),'log':str(log)}
   state['commands'].append(entry);state.update(state='running',current_stage=name,updated_utc=now());save(statepath,state)
   with log.open('x') as handle:result=subprocess.run(command,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
   entry.update(state='passed' if result.returncode==0 else 'failed',exit_code=result.returncode,finished_utc=now());save(statepath,state)
   if result.returncode:raise RuntimeError(f'{name} failed; see {log}')
  state.update(state='complete',finished_utc=now(),negative_prediction_metadata=str(output/'metadata.json'),
    figure=str(WORK/'figures/regenerated_s5b_contrast.pdf'),source_values=str(WORK/'figures/state_contrast_source_data.tsv'))
  save(statepath,state)
 except BaseException as error:
  state.update(state='failed',error=repr(error),updated_utc=now());save(statepath,state);raise
if __name__=='__main__':main()
