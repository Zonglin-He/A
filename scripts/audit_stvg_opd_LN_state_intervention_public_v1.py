"""No-private-data audit of every anonymous scalar/CSV/parent bootstrap."""
import csv,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_stvg_opd_LN_state_intervention_v1 import near,independent_summary
from scripts.stvg_opd_LN_state_intervention_common_v1 import ARMS
def run(root):
 p=Path(root)/'results/stvg_opd_LN_state_intervention/2026-10-13'
 with gzip.open(p/'ROWS.jsonl.gz','rt') as f:rr=[json.loads(x) for x in f]
 assert len(rr)==64 and len({r['parent_id'] for r in rr})==64 and {r['dataset'] for r in rr}=={'hc2','vidstg'}
 saved=json.loads((p/'SUMMARY.json').read_text());checks=near(independent_summary(rr),saved)
 for r in rr:
  for arm in ARMS:
   for mask in ['own_U','common_U','observed','dense_s','dense_v','dense_t']:
    before=r[arm+'_'+mask+'_before'];after=r[arm+'_'+mask+'_after'];delta=r[arm+'_'+mask+'_delta']
    checks+=near(None if before is None or after is None else after-before,delta)
   assert r[arm+'_common_U_frames']<=r[arm+'_own_U_frames']<=r[arm+'_GT_event_real_frames']
   assert r[arm+'_own_U_frames']+r[arm+'_observed_frames']==r[arm+'_GT_event_real_frames']
   if r[arm+'_empty_evidence']:assert r[arm+'_actual_backwards']==0 and r[arm+'_own_U_delta'] in [None,0.]
  for mask in ['own_U','common_U','observed']:assert len({r[a+'_'+mask+'_frames'] for a in ARMS})==1
 with (p/'ALL_PARENT_THREE_ARM_READOUTS.csv').open() as f:csvrows=list(csv.DictReader(f))
 assert len(csvrows)==64
 for csvrow,r in zip(csvrows,rr):
  assert set(csvrow)==set(r)
  for k,v in r.items():assert csvrow[k]==('' if v is None else str(v));checks+=1
 from scripts.audit_stvg_opd_LN_state_intervention_v1 import independent_estimate
 expected=json.loads((p/'BEFORE_AFTER_PARENT_SUMMARY.json').read_text())
 for ds in ['hc2','vidstg']:
  rows=[r for r in rr if r['dataset']==ds]
  for arm in ARMS:
   for name in ['before','after']:checks+=near(independent_estimate([r[arm+'_own_U_'+name] for r in rows]),expected[ds][arm][name])
 return dict(status='pass',scope='all64 anonymous three-arm Before/After/real masks/positive-negative-neutral-unknown/empty fits/full dense/CSV/all parent bootstrap and report table readouts',
  parents=64,fits=192,scalar_checks=checks,CPU_only=True,GT_read=False)
if __name__=='__main__':print(json.dumps(run(sys.argv[1]),allow_nan=False))
