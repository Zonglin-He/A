"""Independent no-GT full input/current-baseline audit after all256 commits."""
from pathlib import Path
import os,sys,time,traceback,gc
sys.path.insert(0,str(Path(__file__).parent))
import current_evidence_runtime_v2 as execution
import current_evidence_v1 as original
from current_evidence_v1 import *

def validate_audit():
 execution.validate_execution();p=read(N/'current_audit_plan_v1.json')
 assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p

def run():
 import torch,numpy as np
 p=validate_audit();torch.set_num_threads(1);rc.gt_guard();original.math=execution.compatible_math
 who=identity();assert not(N/'current_audit_worker_v1.json').exists();atomic(N/'current_audit_worker_v1.json',who)
 state=read(N/'state_v1.json');assert state['status']=='all256_current_expert_contexts_and_matched_baselines_prepared_pending_independent_audit'
 for x in [state['worker'],read(N/'first_two_new_current_expert_bridge.json')['worker']]:
  stat=Path('/proc')/str(x['pid'])/'stat'
  assert not stat.exists()or int(stat.read_text().split(') ',1)[1].split()[19])!=x['startticks']
 assert not(N/'failure_v1.json').exists();start=time.perf_counter();totals=dict(queries=0,cached_current_expert=0,new_current_expert=0);domains={}
 try:
  for ds in ['hc2','vidstg']:
   seal=read(N/ds/'sealed.json');assert seal['queries']==128 and not seal['GT_read'];rows=pilot_rows(ds);assert set(seal['files'])=={r['id']for r in rows};proof=[];cached=fresh=0
   for i,row in enumerate(rows,1):
    resources(start);e=seal['files'][row['id']];assert sha(N/e['receipt_path'])==e['receipt_sha256'];assert read(N/e['receipt_path'])=={k:v for k,v in e.items()if k not in ['receipt_path','receipt_sha256']}
    v=original.checked(ds,row,e);v['receipt_sha256']=e['receipt_sha256'];v['compact_sha256']=e['compact_sha256'];proof.append(v)
    if e['new_current_expert']:
     fresh+=1;assert e['fresh_current_uses_genuinely_new_geometry_and_Epool']and e['current_teacher']['new_actual_current_expert'];tp=Path(e['current_teacher']['payload_path']);assert tp.resolve().is_relative_to((N/'new_current_expert_cache').resolve())
    else:
     cached+=1;assert e['reused_current_matches_published_selected_exact'];pe,ce=rc.paths(ds,row['id'])
     with np.load(N/e['path'])as z,np.load(rc.RUN/ds/pe['path'])as a,np.load(rc.selected.D/ce['path'])as b:
      assert np.array_equal(z['Simple'],a['Interpolation'])and np.array_equal(z['Ours'],b['FullMAP'])and np.array_equal(z['Ours_interval'],b['FullMAP_interval'])
    assert e['new_Source']==e['new_optimizer']==0 and not e['GT_read'];atomic(N/'current_audit_progress_v1.json',dict(dataset=ds,completed=i,total=128,worker=who,GT_read=False));gc.collect()
   rec=dict(status='verified128_same_input_current_expert_full_chain_twice_cold_exact_no_GT',queries=128,cached_current_expert=cached,new_current_expert=fresh,seal_sha256=sha(N/ds/'sealed.json'),all_parent_RGB_teacher_raw_compact_prediction_receipt_SHA_and_arrays_exact=True,proof=proof,worker=who,new_Source=0,new_expert=0,new_GT=0)
   atomic(N/ds/'current_audit_receipt_v1.json',rec);domains[ds]=dict(path=str((N/ds/'current_audit_receipt_v1.json').relative_to(R)),sha256=sha(N/ds/'current_audit_receipt_v1.json'),queries=128,cached_current_expert=cached,new_current_expert=fresh);totals['queries']+=128;totals['cached_current_expert']+=cached;totals['new_current_expert']+=fresh
  assert totals==dict(queries=256,cached_current_expert=250,new_current_expert=6)
  rec=dict(status='verified_all256_current_expert_contexts_and_matched_baselines_independent_no_GT_audit',totals=totals,domains=domains,plan_fingerprint=p['fingerprint'],worker=who,elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT=0,preparation_is_not_external_comparison_score=True)
  atomic(N/'current_audit_completion_v1.json',rec);atomic(PUB/'current_audit_receipt_v1.json',{k:v for k,v in rec.items()if k!='domains'})
 except BaseException:
  atomic(N/'current_audit_failure_v1.json',dict(error=traceback.format_exc(),worker=who,elapsed_wall_seconds=time.perf_counter()-start,GT_read=False));raise

if __name__=='__main__':run()
