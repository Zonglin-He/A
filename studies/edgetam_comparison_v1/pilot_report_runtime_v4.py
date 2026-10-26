"""Unchanged output checks, actual continuation model-load proof for new Vid bridge."""
from pilot_report_runtime_v2 import *
from pilot_execution_v4 import validate_execution_v4
def bridge(ds):
 import torch
 torch.set_num_threads(1);validate();parent.gt_guard();folder=D/'pilot_v1'/ds;wait=folder/'bridge_waiting.json';w=read(wait);assert w['status']=='waiting_root_first_two_actual_EdgeTAM_queries'and len(w['files'])==2;assert not(folder/'root_actual_bridge.json').exists();load=read(D/'pilot_v1/model_load_runtime_v4.json');assert load['load']['all_official_state_tensors_exact']and load['actual_CUDA_kernel_contract_passed'];start=time.perf_counter()
 proof=[checked(ds,row,e,raw=True)for row,e in zip(rows(ds)[:2],w['files'])];assert not torch.cuda.is_initialized();atomic(folder/'root_actual_bridge.json',dict(status='passed_first_two_actual_EdgeTAM_queries',waiting_sha256=sha(wait),cases=proof,model_load_sha256=sha(D/'pilot_v1/model_load_runtime_v4.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_original_expert=0,new_GT_scoring=0))
def dtype_bridge():
 import torch
 torch.set_num_threads(1);validate_execution_v4();parent.gt_guard();folder=D/'pilot_v1/hc2';wait=folder/'dtype_bridge_waiting_v4.json';w=read(wait);assert w['status']=='waiting_actual_failed_query_dtype_boundary_root_bridge';e=w['file'];assert e['actual_memory_BF16_to_FP32_V_projection_casts']>0;row=next(x for x in rows('hc2')if x['id']==Path(e['path']).stem);proof=checked('hc2',row,e,raw=True);control=read(D/'dtype_positive_control_v1/completion.json');assert control['all_original_raw_mask_bytes_SHA_exact'];atomic(folder/'dtype_root_bridge_v4.json',dict(status='passed_actual_BF16_storage_to_FP32_V_projection_full_raw_bridge',waiting_sha256=sha(wait),case=proof,actual_dtype_casts=e['actual_memory_BF16_to_FP32_V_projection_casts'],positive_control_sha256=sha(D/'dtype_positive_control_v1/completion.json'),worker=identity(),new_model=0,new_Source=0,new_original_expert=0,new_GT=0))
if __name__=='__main__':
 validate_execution_v4();validate_bridge();a=argparse.ArgumentParser();a.add_argument('command',choices=['bridge','score','audit','dtype_bridge']);a.add_argument('--dataset',choices=['hc2','vidstg'],required=True);q=a.parse_args();original.checked=checked
 if q.command=='bridge':bridge(q.dataset)
 elif q.command=='dtype_bridge':assert q.dataset=='hc2';dtype_bridge()
 else:getattr(original,q.command)(q.dataset)
