"""Explicit local module identity; original numerical core and plan unchanged."""
import importlib.util,sys,json,hashlib,argparse
from pathlib import Path
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
import run as original
assert Path(original.__file__).resolve()==S/'run.py'
assert Path(sys.modules['common'].__file__).resolve()==S/'common.py'
spec=importlib.util.spec_from_file_location('_full_tube_controls_core_v1',S/'core.py');numeric=importlib.util.module_from_spec(spec);sys.modules[spec.name]=numeric;spec.loader.exec_module(numeric);sys.modules['core']=numeric
base_validate=original.validate

def validate():
 p=original.read(original.D/'execution_v2_plan.json');assert p['fingerprint']==original.digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert original.sha(original.R/f)==h,f
 return base_validate()
original.validate=validate
# The numeric source, original code and original scientific plan are never edited.
# The owner spawns this exact runtime for all subsequent independent stages.
def all_domains():
 import subprocess,traceback,time
 D=original.D;validate();assert not(D/'owner_v2.json').exists();original.atomic(D/'owner_v2.json',original.identity())
 try:
  for ds in original.COUNTS:
   for command in ['predict','score','audit']:
    original.atomic(D/'state_v2.json',dict(stage=command,dataset=ds,owner=original.identity()));log=(D/(ds+'_'+command+'_v2.log')).open('xb');c=subprocess.Popen([sys.executable,'-B',__file__,command,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);original.atomic(D/(ds+'_'+command+'_child_v2.json'),original.identity(c.pid));code=c.wait();log.close();original.atomic(D/(ds+'_'+command+'_exit_v2.json'),dict(code=code));assert code==0,'Preserve all failed stage evidence; no automatic retry'
  original.atomic(D/'state_v2.json',dict(stage='both_domains_scored_audited_pending_root_analysis_publication',owner=original.identity()))
 except BaseException:original.atomic(D/'state_v2.json',dict(stage='failed_preserved',error=traceback.format_exc(),owner=original.identity()));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preflight','predict','score','audit','all_domains']);ap.add_argument('--dataset',choices=original.COUNTS);v=ap.parse_args();validate()
 if v.command=='all_domains':all_domains()
 else:getattr(original,v.command)(*([]if v.command=='preflight'else[v.dataset]))
