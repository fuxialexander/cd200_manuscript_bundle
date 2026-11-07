"""Resume only interrupted inference/evaluation; preserve fitted models and partial output."""
import importlib.util,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent
C=B.parent/'gm12878_benchmark'
p=B/'benchmark_status.json'
s=json.loads(p.read_text())
def save():
 s['updated_utc']=datetime.now(timezone.utc).isoformat();p.write_text(json.dumps(s,indent=2))
def run(name,cmd):
 s.update(state='running',current_stage=name);save()
 with (B/'logs'/f'{name}_storage_recovery.log').open('w') as log:
  subprocess.run([str(x) for x in cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
try:
 r=B/'runs/chrombpnet_seed1234'
 partial=r/'test_predictions'; preserved=r/'test_predictions_interrupted_disk_full'
 assert not preserved.exists()
 assert partial.exists() and not (partial/'prediction_metadata.json').exists()
 partial.rename(preserved)
 command=next(x['command'] for x in s['stages'] if x['name']=='chrombpnet_predict')
 run('chrombpnet_predict_recovery',command)
 out=B/'evaluation_matched_training'
 assert not out.exists()
 cmd=[sys.executable,C/'evaluate.py','--data-dir','/tmp/cd200_k562_benchmark/cache','--output',out,'--bootstrap','2000']
 for label,name in [('ChromBPNet_published','chrombpnet_published_fold0'),('ExplaiNN','explainn_seed1234'),('MoDL','modl_seed1234'),('ChromBPNet_fresh','chrombpnet_seed1234')]:
  cmd+=['--prediction',f'{label}={B / "runs" / name / "test_predictions"}']
 run('evaluation_matched_training',cmd)
 spec=importlib.util.spec_from_file_location('supervisor',C/'run_benchmark.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 report=B/'EVALUATION_MATCHED_TRAINING.md';m.summarize(out,report,'K562 held-out prediction results','K562','ENCSR868FGK')
 s.update(state='complete',current_stage='complete',error=None,matched_comparison_available=str(report),finished_utc=datetime.now(timezone.utc).isoformat());save()
except BaseException as e:
 s.update(state='failed',error=repr(e));save();raise
