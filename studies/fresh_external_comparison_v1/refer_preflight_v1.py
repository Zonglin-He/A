"""Prepared-current numerical compatibility only, no alternative-model construction."""
from refer_common_v1 import *
def run():
 import torch,numpy as np
 import math_endpoint_contract_v2 as math
 validate_pilot();torch.set_num_threads(1);rc.gt_guard();start=time.perf_counter();cases=[]
 for ds in ['hc2','vidstg']:
  for row in pilot_rows(ds)[:2]:
   cap,raw,frames,t,ref,proof=payload(ds,row);a,_=math.build(cap,raw,t,ds);b,_=math.build(cap,raw,t,ds);assert all(np.array_equal(v,b[k])for k,v in a.items());assert np.array_equal(a['Ours'],ref['CurrentOurs'])and np.array_equal(a['Ours_interval'],ref['CurrentOurs_interval'])and np.array_equal(a['Simple'],ref['CurrentSimple']);cases.append(dict(dataset=ds,prepared_current_full_chain_repeat_exact=True,current_receipt_sha256=proof['current_baseline_receipt_sha256'],new_model=0,new_expert=0,new_Source=0,new_GT=0))
 rec=dict(status='passed_four_prepared_current_expert_full_chain_repeat_exact',cases=cases,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_alternative_expert_forward=0,new_model=0,new_Source=0,new_GT=0);atomic(N/'refer_math_preflight_v1.json',rec);atomic(P/'math_preflight_receipt_v1.json',rec)
if __name__=='__main__':run()
