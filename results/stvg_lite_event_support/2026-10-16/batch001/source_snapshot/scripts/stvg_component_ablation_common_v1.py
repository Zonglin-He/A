"""Finite post-search component-deletion protocol. No inference on import."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import BASE as SEARCH,read,write,sha,roster,load_case
BASE=ROOT/'artifacts/stvg_component_ablation128_v1'
SPATIAL=[
 ('A000','baseline',{}),
 ('A001','uniform_reliability',{'uniform_reliability':True}),
 ('A002','no_control_variate',{'no_control_variate':True}),
 ('A003','no_policy_KL',{'no_policy_KL':True}),
 ('A004','iid_sampling',{'iid_sampling':True}),
 ('A005','no_reference_gate',{'no_reference_gate':True}),
 ('A006','no_response_gate',{'no_response_gate':True}),
 ('A007','no_backtracking',{'no_backtracking':True}),
 ('A008','no_rejection_stop',{'no_rejection_stop':True}),
 ('A009','no_flat_skip',{'no_flat_skip':True}),
 ('A010','no_equal_norm_diagnostic',{'no_equal_norm_diagnostic':True}),
 ('A011','query_only',{'query_only':True}),
 ('A012','ln_only',{'ln_only':True}),
 ('A013','no_soft_direction',{'no_soft_direction':True}),
 ('A014','no_response_module',{'no_soft_direction':True,'no_response_gate':True,'no_backtracking':True}),
 ('A015','no_acceptance_controls',{'no_reference_gate':True,'no_response_gate':True,'no_backtracking':True}),
]
TEMPORAL=['no_grid_trust','no_temporal_l2','shared_offset_candidate','projection_only']
ARMS=('F','T','S_M2','TS_M2')
def verify_final():
 p=BASE/'FINAL_PARAMETER_LOCK.json';f=read(p)
 assert f['status']=='locked'and f['search_finished']and f['GT_read']is False and f['count']==128
 assert f['roster_sha256']==sha(SEARCH/'ROSTER_LOCK.json')
 assert f['search_completion_sha256']==sha(SEARCH/'SEARCH_COMPLETION006.json')
 for rel,h in {**f['pins'],**f['protected']}.items():assert sha(ROOT/rel)==h,rel
 return f

def verify_preparation():
 p=BASE/'PREPARATION_RUNTIME.json';r=read(p);assert r['status']=='locked'and not r['GT_read']
 for rel,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/rel)==h,rel
 return r
