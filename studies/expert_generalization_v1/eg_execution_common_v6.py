"""Frozen expert pilot unchanged, actual EdgeTAM FP32 storage-interface recovery owner."""
from eg_execution_common_v5 import *
def validate_execution_v6():
 validate_execution_v5();p=read(D/'execution_v6_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
