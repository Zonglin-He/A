"""Preserved partial-cache resume: version only worker-registration identity."""
from pathlib import Path
import sys,argparse,inspect,subprocess,traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime_v5 as prior
original=C=prior.original
_validate5=prior.validate

def validate():
 _validate5();p=C.read(C.D/'execution_v6_plan.json');assert p['fingerprint']==C.digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert C.sha(C.R/f)==h,f
 return p
prior.rt.validate=validate
src=inspect.getsource(prior.rt.predict);assert src.count('predict_worker_v4.json')==2
patched=src.replace('predict_worker_v4.json','predict_worker_v6.json');assert patched.replace('predict_worker_v6.json','predict_worker_v4.json')==src
scope={**vars(prior.rt),'validate':validate};exec(compile(patched,'<version_only_CPU_worker_registration_v6>','exec'),scope);predict=scope['predict']

def all_domains():
 validate();assert not(C.D/'owner_v6.json').exists();C.atomic(C.D/'owner_v6.json',C.identity())
 try:
  for ds in C.COUNTS:
   if(C.D/ds/'reporting_receipt.json').exists():
    a=C.read(C.D/ds/'reporting_receipt.json');assert a['status']=='verified_whole_domain'and a['queries']==C.COUNTS[ds];continue
   for stage in ['predict','score','audit']:
    C.atomic(C.D/'state_v6.json',dict(stage=stage,dataset=ds,owner=C.identity()));log=(C.D/(ds+'_'+stage+'_v6.log')).open('xb');ch=subprocess.Popen([sys.executable,'-B',__file__,stage,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);C.atomic(C.D/(ds+'_'+stage+'_child_v6.json'),C.identity(ch.pid));code=ch.wait();log.close();C.atomic(C.D/(ds+'_'+stage+'_exit_v6.json'),dict(code=code));assert code==0,'Preserve failure, no automatic retry'
  C.atomic(C.D/'state_v6.json',dict(stage='both_domains_scored_audited_pending_root_analysis_publication',owner=C.identity()))
 except BaseException:C.atomic(C.D/'state_v6.json',dict(stage='failed_preserved',error=traceback.format_exc(),owner=C.identity()));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['predict','score','audit','all_domains']);p.add_argument('--dataset',choices=C.COUNTS);v=p.parse_args();validate()
 if v.command=='all_domains':all_domains()
 elif v.command=='predict':predict(v.dataset)
 else:getattr(prior.rt,v.command)(v.dataset)
