"""Record every attempted ordered configuration and fixed confirmation arm."""
import sys,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_coordinate_common_v2 import *
def run(dataset):
 p=verify(dataset);out=BASE/dataset;trials=read(out/'TRIALS.json');sel=read(out/'SELECTION.json');assert len(trials)<=72
 for letter in ['A','B']:
  valid=[r for r in trials if r['stage']==letter and r['state']=='COMPLETE'];best=min(valid,key=lambda r:(-r['objective'],r['number']));assert sel['stages'][letter]['tag']==best['tag']
  for r in valid:
   d=out/r['result_dir'];score=read(d/'SCORE.json');assert score['objective']==r['objective'] and score['params']==r['params'] and sha(d/'AUDIT.json')==score['audit_sha256']
 with (out/'TRIALS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['number','tag','stage','state','lr','rho','teacher_temperature','objective_pp','reused_from']);w.writeheader()
  for r in trials:w.writerow(dict(number=r['number'],tag=r['tag'],stage=r['stage'],state=r['state'],**r['params'],objective_pp='' if r['objective'] is None else r['objective']*100,reused_from=r.get('reused_from','')))
 confirmation={}
 for name in ['default','lr_only','selected']:
  d=out/'confirmation'/name
  if (d/'REUSE.json').exists():reuse=read(d/'REUSE.json');d=out/reuse['source'];assert sha(d/reuse['receipt'])==reuse['sha256']
  if (d/'NUMERICAL_FAILURE.json').exists():confirmation[name]=read(d/'NUMERICAL_FAILURE.json');continue
  score=read(d/'SCORE.json');assert read(d/'PREDICTION_BARRIER.json')['time']>sel['time'];confirmation[name]=dict(score=score,summary=read(d/'SUMMARY.json'))
 write(out/'CONFIRMATION_SUMMARY.json',confirmation)
 lines=[f'# {dataset}: learning rate then temperature','',f'Selected parameters: {sel["params"]}. Source-disjoint confirmation is historically exposed data. No confirmation reselection.','', '| Arm | Corruption all ΔvIoU pp [95% CI] | Nonexpert ΔvIoU pp [95% CI] |','|---|---:|---:|']
 for name,z in confirmation.items():
  if z.get('status')=='numerical_failure':lines.append(f'| {name} | numerical failure | no reselection |');continue
  entries=[]
  for sub in ['all','nonexpert']:
   m=z['summary']['corruption'][sub]['metrics']['delta_m_vIoU'];entries.append(f"{100*m['mean']:+.5f} [{100*m['ci95'][0]:+.5f}, {100*m['ci95'][1]:+.5f}]")
  lines.append('| '+name+' | '+' | '.join(entries)+' |')
 lines+=['','All scheduled points, numerical failures and clean/expert/nonexpert/harm outcomes are retained. A single coordinate pass is not a global optimum. No production promotion or full-data rerun. Root publication and parameter curves remain required.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');write(out/'COMPLETION.json',dict(status='executed_pending_root_audit_curves_publication',scheduled=len(trials),numerical_failures=sum(r['state']=='FAIL' for r in trials),selection_sha256=sha(out/'SELECTION.json'),time=time.time()))
if __name__=='__main__':run(sys.argv[1])
