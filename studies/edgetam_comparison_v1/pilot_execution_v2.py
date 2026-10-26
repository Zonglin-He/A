"""Execution-only CG deferral; original fixed256 numeric science unchanged."""
from pilot_common_v1 import *
def validate_execution():
 validate();p=read(D/'execution_v2_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
