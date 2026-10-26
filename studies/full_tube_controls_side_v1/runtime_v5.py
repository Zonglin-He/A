"""Kernel resource-file fast path only; all dataset label guards/math unchanged."""
from pathlib import Path
import sys,os,re,argparse,subprocess,traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime_v4 as rt
original=C=rt.original
_validate4=rt.validate

def validate():
 _validate4();p=C.read(C.D/'execution_v5_plan.json');assert p['fingerprint']==C.digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert C.sha(C.R/f)==h,f
 return p
rt.validate=validate

def no_GT():
 paths={(C.selected.PROVIDER/j['labels']).resolve()for j in C.read(C.selected.RUN/'plan.json')['jobs'].values()}
 assert all(not str(p).startswith('/proc/')for p in paths)
 kernel=re.compile(r'/proc/(?:meminfo|[0-9]+/(?:stat|cmdline|environ))\Z')
 def guard(event,args):
  if event!='open'or not args or not isinstance(args[0],(str,bytes,Path)):return
  raw=os.fsdecode(args[0])
  # These exact kernel-owned files cannot be dataset files or user symlinks.
  # /proc/*/root, fd, cwd and every dataset/cache path still resolve normally.
  if kernel.fullmatch(raw):return
  if Path(raw).resolve()in paths:raise PermissionError('No labels in prediction or cold audit')
 sys.addaudithook(guard)
C.no_GT=no_GT

def preflight():
 import numpy as np,torch,time
 torch.set_num_threads(1);validate();no_GT();blocked=0
 for j in C.read(C.selected.RUN/'plan.json')['jobs'].values():
  try:open(C.selected.PROVIDER/j['labels'],'rb')
  except PermissionError:blocked+=1
  else:raise AssertionError('Label guard failed before any data access')
 assert blocked==2
 cases=[]
 for ds in C.COUNTS:
  for row in C.records(ds)[:2]:
   a,_=rt.core.derive(ds,row);b,_=rt.rt.numeric.derive(ds,row);assert set(a)==set(b)and all(np.array_equal(v,b[k])and v.dtype==b[k].dtype for k,v in a.items());cases.append(dict(dataset=ds,all_arrays_and_dtypes_equal_to_fresh_native_and_full_DP=True))
 start=time.perf_counter()
 for _ in range(3):C.wait_resources(start)
 z=dict(status='passed_four_actual_array_dtype_bridges_and_label_guard',cases=cases,actual_GT_label_open_attempts_blocked_before_access=blocked,actual_GT_reads=0,new_Source=0,new_expert=0,new_optimizer=0,three_resource_checks_elapsed_seconds=time.perf_counter()-start,fastpath_only='/proc/meminfo or /proc/numeric-pid/stat|cmdline|environ; canonical checks unchanged for every dataset/cache/fd/root/cwd path');C.atomic(C.D/'preflight_v5.json',z);C.atomic(C.P/'preflight_v5.json',z);print(z,flush=True)

def all_domains():
 validate();assert not(C.D/'owner_v5.json').exists();C.atomic(C.D/'owner_v5.json',C.identity())
 try:
  for ds in C.COUNTS:
   if(C.D/ds/'reporting_receipt.json').exists():
    a=C.read(C.D/ds/'reporting_receipt.json');assert a['status']=='verified_whole_domain'and a['queries']==C.COUNTS[ds];continue
   for stage in['predict','score','audit']:
    C.atomic(C.D/'state_v5.json',dict(stage=stage,dataset=ds,owner=C.identity()));log=(C.D/(ds+'_'+stage+'_v5.log')).open('xb');ch=subprocess.Popen([sys.executable,'-B',__file__,stage,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);C.atomic(C.D/(ds+'_'+stage+'_child_v5.json'),C.identity(ch.pid));code=ch.wait();log.close();C.atomic(C.D/(ds+'_'+stage+'_exit_v5.json'),dict(code=code));assert code==0,'Preserve failure without automatic retry'
  C.atomic(C.D/'state_v5.json',dict(stage='both_domains_scored_audited_pending_root_analysis_publication',owner=C.identity()))
 except BaseException:C.atomic(C.D/'state_v5.json',dict(stage='failed_preserved',error=traceback.format_exc(),owner=C.identity()));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preflight','predict','score','audit','all_domains']);ap.add_argument('--dataset',choices=C.COUNTS);v=ap.parse_args();validate()
 if v.command=='preflight':preflight()
 elif v.command=='all_domains':all_domains()
 else:getattr(rt,v.command)(v.dataset)
