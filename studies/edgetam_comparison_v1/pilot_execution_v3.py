"""Frozen tracking math preserved, bounded two-commit recovery and retention fix."""
from pilot_execution_v2 import *
def validate_execution_v3():
 validate_execution();p=read(D/'execution_v3_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
