"""Execution-only checkpoint helper completion; original mathematical body unchanged."""
from refer_common_v1 import *
import argparse,ast,dis,builtins,inspect
V2=N/'referformer_v4';V2P=R/'results/fresh_external_comparison_v1/referformer_v4'

def validate_v2():
 p=read(N/'refer_execution_v4_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 validate_pilot();return p

def bind():
 import refer_worker_v1 as worker,refer_root_v1 as root,refer_evaluation_v1 as evaluation
 source=(S/'eg_pilot_v1.py').read_text();tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='save');segment=ast.get_source_segment(source,node);namespace={'os':os};exec(compile(segment,str(S/'eg_pilot_v1.py'),'exec'),namespace)
 original=worker.run.__code__;worker.save=namespace['save'];worker.D=V2;root.D=V2;evaluation.D=V2;evaluation.P=V2P
 assert worker.run.__code__ is original
 missing=sorted({x.argval for x in dis.get_instructions(worker.run)if x.opname=='LOAD_GLOBAL'and x.argval not in worker.__dict__ and not hasattr(builtins,x.argval)})
 assert not missing,missing
 return worker,root,evaluation,segment

def preflight():
 import torch
 validate_v2();w,root,e,segment=bind();torch.set_num_threads(1);folder=V2/'synthetic_save_contract_v2';path=folder/'checkpoint.pt';assert not path.exists();value=dict(nested={'tensor':torch.arange(6,dtype=torch.float32).reshape(2,3)},before_selected_validation=True);w.save(path,value);loaded=torch.load(path,map_location='cpu',weights_only=False);assert torch.equal(loaded['nested']['tensor'],value['nested']['tensor'])and loaded['before_selected_validation'];assert not path.with_suffix('.tmp').exists();assert not torch.cuda.is_initialized()
 atomic(V2/'preflight_v4.json',dict(status='passed_exact_original_save_helper_atomic_CPU_roundtrip_and_all_worker_global_symbols_resolved',save_helper_source_sha256=hashlib.sha256(segment.encode()).hexdigest(),original_save_helper_source=str((S/'eg_pilot_v1.py').relative_to(R)),original_save_helper_file_sha256=sha(S/'eg_pilot_v1.py'),worker_numeric_code_object_unchanged=True,namespace_only_changed=True,new_model=0,new_expert=0,new_Source=0,new_GT_scoring=0,saved_synthetic_checkpoint_sha256=sha(path),worker=identity()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','GPU','bridge','input_audit','score','audit','export']);p.add_argument('--dataset',choices=['hc2','vidstg']);a=p.parse_args()
 if a.command=='preflight':preflight()
 else:
  validate_v2();assert read(V2/'preflight_v4.json')['status']=='passed_exact_original_save_helper_atomic_CPU_roundtrip_and_all_worker_global_symbols_resolved';w,root,e,_=bind()
  if a.command=='GPU':w.run()
  elif a.command=='bridge':root.bridge(a.dataset)
  elif a.command=='input_audit':root.audit(a.dataset)
  else:getattr(e,a.command)(a.dataset)
