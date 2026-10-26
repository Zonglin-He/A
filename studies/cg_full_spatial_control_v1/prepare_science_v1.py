"""Freeze math and prove old controls plus predeclared full-spatial chain."""
from spatial_common_v1 import *
import argparse,ast

def prepare():
 assert not(D/'science_plan_v1.json').exists();base.validate();old=base.legacy();index=fixed(base.CGOLD/'reuse_index.json')['hc2'];assert len(index)==227
 inventory=[]
 for key,e in index.items():
  cp=Path(e['root'])/e['entry']['capture_path'];assert sha(cp)==e['entry']['capture_sha256'];inventory.append(dict(id=key,capture_sha256=e['entry']['capture_sha256'],bytes=cp.stat().st_size))
 atomic(D/'retained_raw_inventory_v1.json',dict(queries=227,files=inventory,missing_raw_queries=1675,new_Source_or_prefix_upper_bound=1675,new_expert=0))
 code=[S/'spatial_common_v1.py',S/'spatial_core_v1.py',Path(__file__)]
 inputs=[base.D/'plan.json',base.D/'hc2_sealed.json',base.D/'hc2_reporting_receipt.json',base.D/'hc2_publication.json',base.CGOLD/'reuse_index.json',R/'outputs/paper_followup_v1/cg_calibrations.json',R/'outputs/review_strengthening_v1/CG_full_spatial_preflight_plan_v1.json',R/'outputs/review_strengthening_v1/CG_full_spatial_preflight_receipt_v1.json',D/'retained_raw_inventory_v1.json']
 p=dict(schema='predeclared_complete_spatial_CG_HC_science_v1',cohort=dict(hc2=1902,parents=227),profile=PROFILE,arms=ARMS,pairs=PAIRS,scope='Enable the predeclared full spatial profile gamma.5/unobserved.5/link.25 on HC only, preserving original beta.5/context1.5/kappa0/current shared-E ROI time inference and finalDP; no first auxiliary DP. Old gamma0/u0/link0 currentHC remains an immutable conservative control. Not an isolated single-factor effect, no full-set parameter selection; current Vid already full-spatial is reused after complete audit, not rerun.',numerics='Actual fresh expert masks/seeds/bidirectional geometry, newly corrected E/native-grid E ROI pool, own prototype/posterior/legaljointMAP interval/finalDP. Original Source/native logits/q0/old profile pooled features/all arrays and current conservative all arrays exact before new inference. Cold sufficient-state all arrays exact, not cached final-box renaming.',reuse=dict(actual_retained_raw_SHA_checked=227,missing_raw=1675,max_new_Source=1675,max_new_expert=0,original_teacher_context_exact_only=True),execution='This science freeze and actual retained-grid CPU preflight do not authorize a GPU launch by themselves. Separate execution freeze must wait actual CG and gamma successful exit/whole audits/root publication plus explicit root release, preserve complete raw Source outputs+grids checkpoint before postprocessing/teacher/ROI; first2 new acquisitions root bridge, at-most-once paid Source, original420FP32/model/frames/expert8, no fallback/retry or old-root deletion. Full HC1902 seal before independent official CPU scorer then independent parent/native/teacher/pred/compact SHA/allarrays/means auditor. Root signed positive-negative tails/duration/parentCI/cost and ALLremote publication required.',code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs});p['fingerprint']=digest(p);atomic(D/'science_plan_v1.json',p);atomic(P/'science_protocol_v1.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'])

def preflight():
 import torch,numpy as np
 from spatial_core_v1 import build,replay
 torch.set_num_threads(1);validate();sys.addaudithook(__import__('stvg.runtime',fromlist=['guard']).guard)
 sys.path.insert(0,str(R/'studies/review_strengthening_v1'));import cg_full_spatial_preflight_v1 as declared
 assert declared.PROFILE==PROFILE;old=base.legacy();rows={r['id']:r for r in old.rows('hc2')};reuse=fixed(base.CGOLD/'reuse_index.json')['hc2'];proof=[];start=time.perf_counter()
 for key,e in list(reuse.items())[:2]:
  cap,teacher,origin=old.load_legacy(e,rows[key],'hc2');a,c,p,cost=build(cap,teacher,key);b,cc,pp,_=build(cap,teacher,key);assert all(np.array_equal(v,b[k])for k,v in a.items())
  original=declared.derive(cap,teacher,PROFILE);assert np.array_equal(a['FullSpatial'],original['boxes'])and np.array_equal(a['FullSpatial_interval'],original['interval'])and np.array_equal(c['features'],original['features'])and np.array_equal(c['posterior'],original['posterior'])and np.array_equal(c['E'],original['E'])
  torch.set_num_threads(4)
  try:again=replay(c)
  finally:torch.set_num_threads(1)
  assert all(np.array_equal(v,again[k])for k,v in a.items())
  proof.append(dict(raw_capture_sha256=origin['capture_sha256'],current_conservative_prediction_sha256=p['current_conservative_prediction_sha256'],original_teacher_sha256=origin['teacher_sha256'],old_full_and_native_and_current_conservative_exact=True,new_complete_spatial_chain_and_predeclared_profile_exact=True,repeat_all_arrays_exact=True,Torch1_vs4_cold_all_arrays_exact=True))
 assert not torch.cuda.is_initialized();rec=dict(status='passed_two_actual_HC_raw_grids_full_spatial_and_current_control_exact',cases=proof,science_fingerprint=read(D/'science_plan_v1.json')['fingerprint'],worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0,actual_new_GPU_bridge=False,full_HC_complete=False);atomic(D/'preflight_receipt_v1.json',rec);atomic(P/'preflight_receipt_v1.json',rec);print(rec['status'])
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight']);globals()[a.parse_args().command]()
