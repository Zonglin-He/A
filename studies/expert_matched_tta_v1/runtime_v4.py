"""Exact old/new TTA receipt schemas, resume committed CPU compositions only."""
import sys,importlib.util,inspect,copy,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
c=load('common',ROOT/'common.py');old=load('_expert_matched_frozen_run',ROOT/'run.py')
def validate_execution():
 p=c.read(c.D/'execution_v4_plan.json');assert p['fingerprint']==c.digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert c.sha(c.R/f)==h,f
 c.validate()
def normalized_entry(ds,key):
 e=old.fixed(c.T/f'{ds}_sealed.json')['files'][key]
 if e.get('all_readouts_cold_exact')is True:return e
 assert e['reused_completed']and 'all_readouts_cold_exact'not in e
 proof=old.fixed(c.D/'legacy_schema_compatibility.json')[ds][key]
 for k in ['sha256','logits_sha256','receipt_sha256']:assert proof[k]==e[k]
 assert proof['actual_saved_native_logits_readouts_exact']and proof['all_three_files_SHA_exact']
 n=copy.deepcopy(e);n['all_readouts_cold_exact']=True
 for m in ['Tent','SAR']:
  assert 'logits_changed'not in n['traces'][m];n['traces'][m]['logits_changed']=proof['logits_changed'][m]
 return n
def compile_derive():
 src=inspect.getsource(old.derive);a="te=fixed(T/f'{ds}_sealed.json')['files'][row['id']]";b="te=normalized_entry(ds,row['id'])";assert src.count(a)==1
 changed=src.replace(a,b);assert changed.replace(b,a)==src
 env={**vars(old),'normalized_entry':normalized_entry};exec(compile(changed,'<receipt-schema-only-v4>','exec'),env);return env['derive']
derive=compile_derive()
def predict(ds):
 import numpy as np,torch
 torch.set_num_threads(1);validate_execution();c.rc.gt_guard();dest=c.D/ds;assert not(dest/'worker_runtime_v4.json').exists();who=c.identity();c.atomic(dest/'worker_runtime_v4.json',who);start=time.perf_counter();files={};workers={};stats={m:dict(queries=0,nonzero_parameter_delta=0,logits_changed=0,iterations=0)for m in ['Tent','SAR']};reused=0
 for i,row in enumerate(c.records(ds),1):
  assert c.resources()and time.perf_counter()-start<259200;a,proof=derive(ds,row);out=dest/'predictions'/(row['id']+'.npz');rp=dest/'receipts'/(row['id']+'.json')
  if out.exists()or rp.exists():
   assert out.exists()and rp.exists();rec=c.read(rp);assert rec['provenance']==proof and rec['sha256']==c.sha(out);reused+=1
  else:
   out.parent.mkdir(parents=True,exist_ok=True)
   with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
   c.os.replace(out.with_suffix('.tmp'),out);rec=dict(path=str(out.relative_to(c.D)),sha256=c.sha(out),provenance=proof,worker=who,fingerprint=c.read(c.D/'plan.json')['fingerprint'],GT_read=False);c.atomic(rp,rec)
  with np.load(out)as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
  files[row['id']]=dict(rec,receipt_path=str(rp.relative_to(c.D)),receipt_sha256=c.sha(rp));workers[str(rec['worker']['pid'])]=rec['worker'];te=normalized_entry(ds,row['id'])
  for m,t in te['traces'].items():stats[m]['queries']+=1;stats[m]['nonzero_parameter_delta']+=int(t['parameter_delta']>0);stats[m]['logits_changed']+=int(t['logits_changed']);stats[m]['iterations']+=len(t['steps'])
  if i%25==0:c.atomic(dest/'progress.json',dict(completed=i,total=c.COUNTS[ds],elapsed_wall_seconds=time.perf_counter()-start,reused_committed_cpu_predictions=reused,GT_read=False))
 assert len(files)==c.COUNTS[ds]
 for m in stats:assert stats[m]['queries']==c.COUNTS[ds]and stats[m]['iterations']==5*c.COUNTS[ds]
 c.atomic(dest/'sealed.json',dict(queries=len(files),files=files,worker=who,actual_workers=workers,seconds=time.perf_counter()-start,reused_committed_cpu_predictions=reused,previous_failed_attempt_elapsed_wall_seconds=None,previous_failed_cost_status='Actual first worker and48commits retained; standalone wall clock not saved, not fabricated zero.',update_diagnostics=stats,GT_read=False,fingerprint=c.read(c.D/'plan.json')['fingerprint']))
def score(ds):validate_execution();old.score(ds)
def audit(ds):
 validate_execution();old.derive=derive;old.audit(ds)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['predict','score','audit']);a.add_argument('--dataset',choices=c.COUNTS,required=True);v=a.parse_args();globals()[v.command](v.dataset)
