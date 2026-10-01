"""Record complete attempts and confirmation; root audit/publication remain separate."""
import sys, csv, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_extended_common_v3 import *

def run(dataset):
 p=verify(dataset);out=BASE/dataset;trials=read(out/'TRIALS.json');sel=read(out/'SELECTION.json');assert len(trials)<=100 and not sel['confirmation_used']
 for t in trials:
  if t['state']=='COMPLETE':
   d=out/t['result_dir'];s=read(d/'SCORE.json');assert s['objective']==t['objective'] and s['params']==t['params'] and sha(d/'AUDIT.json')==s['audit_sha256']
 with (out/'TRIALS.csv').open('w',newline='') as f:
  fields=['number','tag','stage','state',*anchor(p),'objective_pp','reused_from']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for t in trials:w.writerow(dict(number=t['number'],tag=t['tag'],stage=t['stage'],state=t['state'],**t['params'],objective_pp=None if t['objective'] is None else t['objective']*100,reused_from=t.get('reused_from','')))
 confirmation={}
 for name in ['anchor','selected']:
  d=out/'confirmation'/name
  if (d/'REUSE.json').exists():
   r=read(d/'REUSE.json');d=out/r['source'];assert sha(d/r['receipt'])==r['sha256']
  if (d/'NUMERICAL_FAILURE.json').exists():confirmation[name]=read(d/'NUMERICAL_FAILURE.json');continue
  assert read(d/'PREDICTION_BARRIER.json')['time']>sel['time']
  confirmation[name]=dict(score=read(d/'SCORE.json'),summary=read(d/'SUMMARY.json'))
 write(out/'CONFIRMATION_SUMMARY.json',confirmation)
 write(out/'COMPLETION.json',dict(status='executed_pending_root_audit_curves_publication',scheduled=len(trials),failed=sum(t['state']=='FAIL' for t in trials),selection_sha256=sha(out/'SELECTION.json'),time=time.time()))
 (out/'REPORT.md').write_text(f'# {dataset} extended sensitivity\n\nExecuted {len(trials)} scheduled configurations. Full records, failures, compute costs, and confirmation summaries are retained. Final configuration: {sel["params"]}. Historically exposed data; no confirmation reselection or production promotion. Root audit, sensitivity curves and verified public export remain required.\n')

if __name__=='__main__':run(sys.argv[1])
