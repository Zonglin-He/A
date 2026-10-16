"""Exact frozen Temporal/WF-Top1 protocol on 64 added development sources only."""
from pathlib import Path
import sys,copy,importlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
def fresh():return [r for r in roster()if 'cached_parent'not in r]
def setup(m):
 m.BASE=BASE;m.roster=lambda *a:fresh();m.load_case=load_case;m.verify_runtime=verify_runtime
 original_write=write
 def counted(p,d,mutable=False):
  d=copy.deepcopy(d)
  if Path(p).name in ['TEMPORAL_TEACHER_GLOBAL_BARRIER.json','TEMPORAL_FIT_BARRIER.json','CANDIDATE_INPUT_BARRIER.json']:
   assert d['count']==32;d['count']=len(fresh())
  if Path(p).name in ['TEMPORAL_STATUS.json','CANDIDATE_STATUS.json']:d['total']=len(fresh())
  return original_write(p,d,mutable=mutable)
 m.write=counted
 return m
def main():
 verify_runtime();stage=sys.argv[1];assert stage in ['temporal','fit','inputs','teacher','references']
 sys.addaudithook(guard)
 if stage=='temporal':
  m=setup(importlib.import_module('scripts.run_stvg_evidence_confirmation_temporal012'));m.temporal()
 elif stage=='fit':
  import types
  p=ROOT/'scripts/fit_stvg_evidence_confirmation012.py';raw=p.read_text();old="assert bar['count']==32";assert raw.count(old)==1;raw=raw.replace(old,"assert bar['count']==64")
  m=types.ModuleType('_coordinate128_temporal_fit');m.__file__=str(p);m.__package__='scripts';exec(compile(raw,str(p),'exec'),m.__dict__);setup(m);m.main()
 elif stage=='inputs':
  sys.path.insert(0,str(ROOT/'artifacts/stvg_weightfree_spatial_v1/deps'))
  import cv2
  assert cv2.__version__=='4.11.0'and hasattr(cv2,'ximgproc')
  m=setup(importlib.import_module('scripts.prepare_stvg_evidence_confirmation012'))
  def check(cid,t):
   import torch
   pp=BASE/'confirmation/requests'/(cid+'.json');tt=torch.load(BASE/'confirmation/temporal_teacher'/(cid+'.pt'),map_location='cpu',weights_only=False)
   assert tt['request_sha256']==t['request_sha256']==sha(pp) and t['source_temporal_teacher_sha256']==sha(BASE/'confirmation/temporal_teacher'/(cid+'.pt')) and not tt['GT_read']
  m.check_request=check;sys.argv=['inputs','confirmation'];m.main()
 elif stage=='teacher':
  sys.argv=['teacher','unused'];m=importlib.import_module('scripts.run_stvg_evidence_teacher_v1');m.BASE=BASE/'confirmation';m.main()
 else:
  m=setup(importlib.import_module('scripts.stvg_evidence_spatial_phase009'));m.references('confirmation')
 print('expanded preparation stage complete',stage,flush=True)
if __name__=='__main__':main()
