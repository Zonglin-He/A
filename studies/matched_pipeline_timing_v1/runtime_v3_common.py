from runtime_v2_common import *
_validate_v2=validate_runtime
D=BASE/'runtime_v3'
def validate_runtime():
 p=_validate_v2();e=read(BASE/'execution_v3_plan.json');assert e['science_plan_sha256']==sha(BASE/'plan.json') and digest({k:v for k,v in e.items()if k!='fingerprint'})==e['fingerprint']
 for f,h in e['code_sha256'].items():assert sha(R/f)==h,f
 assert sha(D/'plan.json')==sha(BASE/'plan.json') and sha(D/'selection.json')==sha(BASE/'selection.json')
 return p
