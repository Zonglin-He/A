import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_full_b1_online_v1 import source_capture,OUT
if __name__=='__main__':
 tick=time.monotonic()
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_final_simplification_v1 import lease
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.exact_frame_decode_audit_v2 import decode
 from methods.tastvg_dual_evidence_j0_v1.method import OnlineMethod
 torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 slot=lease();install_clean_loader()
 from scripts.run_spatial_regression_alignment_v1 import model_load
 model=model_load('hcstvg1_test').eval().requires_grad_(False);initial=state_hash(model.state_dict());support=load(ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1/PARAMETER_SUPPORT.pt');deltas=[{n:(v-support['center'][n]).cuda() for n,v in x.items()} for x in support['states']];policy=OnlineMethod(model,deltas)
 j=ROOT/'artifacts/tastvg_schedule_j01_v1';p=read(j/'LOCK.json');rows=p['rows'];records=[]
 try:
  for arrival,parent in enumerate(p['orders']['order1'][:2]):
   row=next(r for r in rows if r['ordinal']==parent);frames,ids=decode(row['input']);before=policy.actor.state();policy.actor.restore(policy.actor.initial)
   data=source_capture(model,frames,row,row['parses']['subject']);old=load(ROOT/f'artifacts/tastvg_spatial_expansion_s0_v1/capture/clean/{parent:03}.pt')
   assert torch.equal(data['prediction']['boxes'],old['prediction']['boxes'])
   assert all(torch.equal(a['H'],b['H']) for a,b in zip(data['views'],old['views']))
   data['pixel_sha256']=old['pixel_sha256'];policy.actor.restore(before)
   def tp():return load(ROOT/f'artifacts/tastvg_temporal_fourarm_v1/c2/clean/{parent:03}.pt')
   def sp():return load(ROOT/f'artifacts/tastvg_spatial_expansion_s0_v1/expert/clean/{parent:03}.pt')
   x,_=policy.arrive(device_tree(data,'cuda'),arrival%4==0,tp,sp);ref=load(j/f'online/order1/clean/{parent:03}.pt')
   for key in ['pre_state','post_state']:assert all(torch.equal(v,ref[key][n]) for n,v in x[key].items())
   assert torch.equal(x['output_prediction']['boxes'],ref['output_prediction']['boxes']) and x['output_prediction']['indices']==ref['output_prediction']['indices']
   records.append(dict(arrival=arrival,parent=parent,H_exact=True,native_exact=True,inherited_state_exact=True,output_exact=True))
 finally:policy.close();slot.close()
 assert state_hash(model.state_dict())==initial
 result=dict(status='pass',scope='Two old dev fixtures, one scheduled update and one inherited nonexpert; new full capture equals original cache and frozen J01 outputs',records=records,seconds=time.monotonic()-tick,GT_read=False)
 write(OUT/'CAPTURE_INTEGRATION_AUDIT.json',result);write(OUT/'allocations'/f'{time.time_ns()}.json',dict(stage='engineering_validation',status='completed',seconds=result['seconds'],GT_read=False));print(result)
