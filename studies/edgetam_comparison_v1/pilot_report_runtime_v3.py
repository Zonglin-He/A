"""Unchanged output checks, actual continuation model-load proof for new Vid bridge."""
from pilot_report_runtime_v2 import *
from pilot_execution_v3 import validate_execution_v3
def bridge(ds):
 import torch
 torch.set_num_threads(1);validate();parent.gt_guard();folder=D/'pilot_v1'/ds;wait=folder/'bridge_waiting.json';w=read(wait);assert w['status']=='waiting_root_first_two_actual_EdgeTAM_queries'and len(w['files'])==2;assert not(folder/'root_actual_bridge.json').exists();load=read(D/'pilot_v1/model_load_runtime_v3.json');assert load['load']['all_official_state_tensors_exact']and load['actual_CUDA_kernel_contract_passed'];start=time.perf_counter()
 proof=[checked(ds,row,e,raw=True)for row,e in zip(rows(ds)[:2],w['files'])];assert not torch.cuda.is_initialized();atomic(folder/'root_actual_bridge.json',dict(status='passed_first_two_actual_EdgeTAM_queries',waiting_sha256=sha(wait),cases=proof,model_load_sha256=sha(D/'pilot_v1/model_load_runtime_v3.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_original_expert=0,new_GT_scoring=0))
if __name__=='__main__':
 validate_execution_v3();validate_bridge();a=argparse.ArgumentParser();a.add_argument('command',choices=['bridge','score','audit']);a.add_argument('--dataset',choices=['hc2','vidstg'],required=True);q=a.parse_args();original.checked=checked
 if q.command=='bridge':bridge(q.dataset)
 else:getattr(original,q.command)(q.dataset)
