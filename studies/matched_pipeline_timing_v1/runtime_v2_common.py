from common import *
BASE=D
D=BASE/'runtime_v2'
def validate_runtime():
 p=validate();e=read(BASE/'execution_v2_plan.json');assert e['science_plan_sha256']==sha(BASE/'plan.json') and digest({k:v for k,v in e.items()if k!='fingerprint'})==e['fingerprint']
 for f,h in e['code_sha256'].items():assert sha(R/f)==h,f
 return p
