from common import *
BASE=D
D=BASE/'runtime_v4'
def validate_runtime():
 validate();p=read(BASE/'plan_v2.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 assert sha(D/'plan.json')==sha(BASE/'plan_v2.json') and sha(D/'selection.json')==sha(BASE/'selection.json')
 return p
