"""Only replace the protected-service identity gate; all numerical worker code intact."""
from pathlib import Path
import json,hashlib,argparse
import worker_v1 as frozen
from protected_service_gate_runtime_v2 import foreign_check
D=frozen.D;R=frozen.R

def validate_execution():
 p=json.loads((D/'execution_v2_plan.json').read_text());assert p['scientific_fingerprint']==frozen.read(D/'plan.json')['fingerprint']
 assert p['fingerprint']==frozen.digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert frozen.sha(R/f)==h,f
 frozen.validate();return p
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['GPU']);a.add_argument('--expert',choices=['mttr'],required=True);q=a.parse_args();validate_execution();code=frozen.run.__code__;frozen.foreign_check=foreign_check;assert frozen.run.__code__ is code;frozen.run(q.expert)
