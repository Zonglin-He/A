"""Independent no-model/no-GT audit of actual alternative-expert pilot outputs."""
from refer_common_v1 import *
import argparse

def checked(ds,row,e):
 import torch,numpy as np
 import math_endpoint_contract_v2 as math
 from refer_worker_v1 import payload,cpu_tree,validate_pilot
 from interface import selected_eight
 validate_pilot();cap,raw,frames,old,refs,proof=payload(ds,row)
 assert e['input_binding']==proof and not e['GT_read']
 for path,checksum in [('path','sha256'),('compact_path','compact_sha256')]:assert sha(D/e[path])==e[checksum]
 for name in ['expert_payload','raw_expert_output']:
  assert sha(D/e[name]['path'])==e[name]['sha256']
 cp=torch.load(D/e['raw_expert_output']['path'],map_location='cpu',weights_only=False,mmap=True)
 assert cp['input_binding']==proof and not cp['GT_read']and cp['worker']==e['worker']
 fresh=selected_eight(cp['outputs'],cp['input_shape']);teacher=torch.load(D/e['expert_payload']['path'],map_location='cpu',weights_only=False,mmap=True)
 for k,v in fresh.items():
  assert torch.equal(v,teacher[k])if torch.is_tensor(v)else v==teacher[k],k
 assert teacher['text_sha256']==proof['caption_sha256']and e['official_CUDA_kernel_calls']>0
 assert e['raw_expert_output']['before_selected_query_validation']and e['raw_expert_output']['before_ROI_or_geometry']and e['expert_payload']['before_both_inference_arms']
 a,c=math.build(cap,raw,teacher,ds);b,cc=math.build(cap,raw,teacher,ds)
 assert all(np.array_equal(v,b[k])for k,v in a.items());cold=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);replayed=math.replay(cold)
 assert all(np.array_equal(v,replayed[k])for k,v in a.items())
 a.update(refs);a['CurrentSimple_interval']=a['Source_interval'].copy()
 with np.load(D/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
 current,_=math.build(cap,raw,old,ds);assert np.array_equal(current['Ours'],refs['CurrentOurs'])and np.array_equal(current['Ours_interval'],refs['CurrentOurs_interval'])and np.array_equal(current['Simple'],refs['CurrentSimple'])
 assert e['new_expert_queries']==1 and e['new_expert_observations']==8 and e['new_Source']==e['new_original_expert']==e['new_optimizer']==0
 return dict(receipt_sha256=sha(D/e['receipt_path']),raw_expert_output_sha256=e['raw_expert_output']['sha256'],expert_payload_sha256=e['expert_payload']['sha256'],all_arrays_fresh_repeat_and_cold_exact=True,actual_original_Source_grid_and_context_verified=True,official_selected_single_query_all8_exact=True)

def bridge(ds):
 import torch
 torch.set_num_threads(1);rc.gt_guard();folder=D/'pilot_v1'/ds;waiting=folder/'bridge_waiting.json';w=read(waiting)
 assert w['status']=='waiting_root_actual_first_two_new_expert_bridge'and len(w['files'])==2
 assert not(folder/'root_actual_bridge.json').exists();model=read(D/'pilot_v1/model_load.json')
 assert sha(D/'pilot_v1/model_load.json')==w['model_load_sha256']and model['load_audit']['all_learned_parameters_covered']and model['load_audit']['all_shared_state_tensors_array_exact']
 start=time.perf_counter();proof=[]
 for row,e in zip(pilot_rows(ds)[:2],w['files']):proof.append(checked(ds,row,e))
 assert not torch.cuda.is_initialized();atomic(folder/'root_actual_bridge.json',dict(status='passed_actual_first_two_new_expert_bridge',waiting_sha256=sha(waiting),cases=proof,root_worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_model=0,new_GT_scoring=0))

def audit(ds):
 import torch
 from eg_pilot_gate_v1 import live
 torch.set_num_threads(1);rc.gt_guard();folder=D/'pilot_v1'/ds;assert not live(read(D/'pilot_v1/worker.json'))
 assert read(D/'pilot_v1/state.json')['status']=='all256_pilot_committed_pending_independent_audit_root_cost_feasibility_and_full_release'
 seal=read(folder/'sealed.json');assert seal['queries']==128 and len(seal['files'])==128 and not seal['GT_read'];start=time.perf_counter();proof=[]
 for row in pilot_rows(ds):
  e=seal['files'][row['id']];assert read(D/e['receipt_path'])=={k:v for k,v in e.items()if k not in ['receipt_path','receipt_sha256']};assert sha(D/e['receipt_path'])==e['receipt_sha256'];proof.append(checked(ds,row,e))
 assert not torch.cuda.is_initialized();atomic(folder/'independent_audit.json',dict(status='verified_all128_pilot_outputs_context_checkpoints_and_fresh_math_no_GT',queries=128,seal_sha256=sha(folder/'sealed.json'),cases=proof,all_arrays_cold_rederived_exact=True,new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,full_release_allowed=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['bridge','audit']);p.add_argument('--dataset',choices=rc.COUNTS,required=True);a=p.parse_args();globals()[a.command](a.dataset)
