"""Original actual CG state name, unchanged worker/science and publication gate."""
from gamma_full_common import *
import runtime_common_v2 as worker_runtime
def validate_runtime():
 worker_runtime.validate_runtime();p=read(D/'execution_v4_plan.json')
 assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 assert sha(D/'plan.json')==p['science_plan_sha256']and sha(D/'execution_v2_plan.json')==p['worker_execution_plan_sha256']
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
