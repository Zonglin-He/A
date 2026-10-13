"""Pure anonymous technical-scalar arithmetic; no private vectors or model."""
import json,math,sys
from pathlib import Path
import numpy as np

def run(checkout):
 p=Path(checkout)/'results/stvg_opd_observation_response_projection/2026-10-13'
 read=lambda name:json.loads((p/name).read_text())
 rr=read('QUALIFICATION_ROWS.json');s=read('QUALIFICATION_SUMMARY.json');m=read('TECHNICAL_STATISTICS.json')
 ff=read('FINITE_DIFFERENCE_ROWS.json');ind=read('INDEPENDENT_FD_READBACK.json');checked=0
 assert len(rr)==144 and len(ff)==len(ind)==84 and not s['formal_eligible'] and not m['formal_eligible']
 assert len({(r['parent_id'],r['arm'],r['boundary'],r['repeat']) for r in rr})==144
 assert len({r['parent_id'] for r in rr})==8
 for f in ff:
  assert f['pass_check']==(f['absolute_error']<=f['tolerance'])
  assert abs(f['relative_error']-f['absolute_error']/max(1e-12,f['predicted_norm']))<1e-12
  assert abs(f['tolerance']-(1e-4+.02*f['predicted_norm']))<1e-12
  assert f['JVP_error']<=f['JVP_tolerance'] and f['VJP_error']<=f['VJP_tolerance'];checked+=5
 for f in ind:
  name='fixed_random' if f['direction']==0 else 'original_Adam'
  target=next(x for x in ff if (x['parent_id'],x['dataset'],x['boundary'],x['direction'],x['h'])==(f['parent_id'],f['dataset'],f['boundary'],name,f['h']))
  for k in ['absolute_error','relative_error','tolerance','pass_check']:assert target[k]==f[k];checked+=1
 for ds in ['hc2','vidstg']:
  rows=[x for x in rr if x['dataset']==ds];f=[x for x in ff if x['dataset']==ds];c=[x for x in s['actual_derivative_checks'] if x['dataset']==ds]
  assert len(rows)==72 and m['datasets'][ds]['local_proposals']==72
  assert m['datasets'][ds]['informative']==sum(x['updated'] for x in rows)
  assert m['datasets'][ds]['no_op']==sum(not x['updated'] for x in rows)
  assert m['datasets'][ds]['FD_pass']==sum(x['pass_check'] for x in f)
  assert m['datasets'][ds]['FD_checks']==len(f)
  assert m['datasets'][ds]['FD_all_pass_contexts']==sum(x['status']=='pass' for x in c)
  assert m['datasets'][ds]['FD_relative_error_median']==float(np.median([x['relative_error'] for x in f]))
  for a in ['P','F','R']:
   vals=[x['seconds'] for x in rows if x['arm']==a and x['repeat']==1 and x['updated']]
   assert m['datasets'][ds]['repeat1_no_directional_FD_local_seconds'][a]==float(np.median(vals));checked+=1
  F=[x for x in rows if x['arm']=='F' and x['updated']];v=s['datasets'][ds]
  assert v['accepted']==sum(x['accepted'] for x in F) and v['informative_probes']==len(F)
  assert v['fallback']==sum(x['fallback'] for x in F) and v['fraction']==v['accepted']/len(F)
  assert m['datasets'][ds]['repeat_bitwise_fail']==sum(x['repeat_scientific_status']=='failed_bitwise' for x in rows);checked+=15
 assert sum(x['pass_check'] for x in ff)==25 and sum(not x['updated'] for x in rr)==18
 assert sum(x['repeat_scientific_status']=='failed_bitwise' for x in rr)==2
 assert m['actual_total_complete_A_replays_including_engineering']==24 and m['complete_candidate_formal_fits']==0 and not m['GT_read']
 return dict(status='pass',scope='all144 anonymous technical proposals/all84 actual FD and independent FD scalar equality/all positive-negative-no-op-unknown costs; not GT efficacy',scalar_checks=checked,count=144,total=144,FD_checks=84,CPU_only=True,GT_read=False)

if __name__=='__main__':print(json.dumps(run(sys.argv[1]),allow_nan=False))
