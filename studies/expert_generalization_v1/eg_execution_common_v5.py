"""Exact same fixed expert pilot; EdgeTAM retention-recovery owner version."""
from eg_execution_common_v4 import *
def validate_execution_v5():
 validate_execution_v4();p=read(D/'execution_v5_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
