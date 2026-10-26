"""Original pilot/math unchanged; user paused CG, EdgeTAM small remains first."""
from eg_execution_common_v3 import *
def validate_execution_v4():
 validate_execution_v3();p=read(D/'execution_v4_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
