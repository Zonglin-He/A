"""Read completed development trials; never pick using confirmation scores."""
import sys,json,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_optuna_common_v1 import *
def run(dataset):
 p=verify(dataset);out=BASE/dataset;trials=read(out/'TRIALS.json');sel=read(out/'SELECTION.json');assert len(trials)==48 and all(x['state'] in ['COMPLETE','FAIL'] for x in trials);valid=[x for x in trials if x['state']=='COMPLETE'];best=min(valid,key=lambda x:(-x['value'],x['number']));assert sel['trial']==best['number'] and not sel['confirmation_used']
 checks=0
 for x in valid:
  td=out/'trials'/f"{x['number']:05}";score=read(td/'SCORE.json');assert score['objective']==x['value'] and score['params']==x['params'] and sha(td/'AUDIT.json')==score['audit_sha256'] and read(td/'AUDIT.json')['status']=='pass';assert read(td/'PREDICTION_BARRIER.json')['time']<score['time'];checks+=1
 confirmation={}
 for name in ['default','selected']:
  d=out/'confirmation'/name
  if (d/'REUSE.json').exists():d=out/'confirmation/default'
  if (d/'NUMERICAL_FAILURE.json').exists():confirmation[name]=dict(status='numerical_failure',no_reselection=True);continue
  score=read(d/'SCORE.json');assert read(d/'PREDICTION_BARRIER.json')['time']>sel['time'];assert sha(d/'AUDIT.json')==score['audit_sha256'];confirmation[name]=dict(score=score,summary=read(d/'SUMMARY.json'));checks+=1
 with (out/'TRIALS.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['trial','state','lr','rho','teacher_temperature','objective_pp']);writer.writeheader()
  for x in trials:writer.writerow(dict(trial=x['number'],state=x['state'],**x['params'],objective_pp='' if x['value'] is None else 100*x['value']))
 write(out/'CONFIRMATION_SUMMARY.json',confirmation);write(out/'AUDIT.json',dict(status='pass',attempts=48,complete=len(valid),numerical_failures=48-len(valid),trial_and_confirmation_checks=checks,selection_before_confirmation=True,confirmation_not_used_for_selection=True,prior_Paper48_unchanged=True))
 lines=[f'# {dataset}: wide Optuna development search','', '32 source search / 16 disjoint-source confirmation, two orders, six conditions; all historical project exposure. No full-dataset parameter selection.','',f"Selected trial {sel['trial']}: `{json.dumps(sel['params'])}`.",f"Search corrupt nonexpert delta vIoU: default {100*sel['default_objective']:+.6f} pp; selected {100*sel['objective']:+.6f} pp. These are selection-biased development outcomes.",'','| Confirmation arm | Corrupt nonexpert delta vIoU pp [95% source CI] |','|---|---|']
 for name,c in confirmation.items():
  if c.get('status')=='numerical_failure':lines.append(f'| {name} | numerical failure; no reselection |')
  else:
   m=c['summary']['corruption']['nonexpert']['metrics']['delta_m_vIoU'];lines.append(f"| {name} | {100*m['mean']:+.6f} [{100*m['ci95'][0]:+.6f}, {100*m['ci95'][1]:+.6f}] |")
 lines+=['','All trials, failures, all-stream/clean/negative tails are retained. A small historically exposed confirmation panel does not establish a fresh full-test result. No automatic promotion or full-dataset rerun.','']
 (out/'REPORT.md').write_text('\n'.join(lines));write(out/'COMPLETION.json',dict(status='completed_pending_root_audit_publication',audit_sha256=sha(out/'AUDIT.json'),selection_sha256=sha(out/'SELECTION.json'),time=time.time()))
if __name__=='__main__':run(sys.argv[1])
