"""Scheduling-only version: explicitly deferred gamma is not a completed task."""
from eg_common_v1 import *
from eg_pilot_v1 import validate_pilot

def validate_execution():
 validate_pilot();p=read(D/'execution_v2_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 pause=read(R/'outputs/maintenance/gamma_paused_by_user_20261026/review_gamma_full_v1/owned_exit.json')
 assert pause['status']=='paused_by_user'and pause['owned_controller_actual_exit']and pause['new_Source']==pause['new_expert']==0
 return p
