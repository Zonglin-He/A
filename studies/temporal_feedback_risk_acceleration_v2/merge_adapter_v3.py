"""Supply original prefix-local folder to the unchanged extracted suffix."""
import runtime_v2 as v
from pathlib import Path
import sys,dis,builtins,json,hashlib
def bind():
 p=v.c.read(v.D/'execution_plan_v3.json')
 assert p['adapter_sha256']==v.c.sha(Path(__file__))
 assert p['runtime_v2_sha256']==v.c.sha(Path(v.__file__))
 v.original.folder=v.c.D/'vidstg'
 v.original.sp=v.original.folder/'sealed.json'
def preflight():
 bind();proof={}
 for stage in ['score','audit']:
  _,_,suffix=v.blocks(stage);args='ds,out,checks,maxerr,start'if stage=='score'else'ds,sha_checks,start'
  code='def _merge('+args+'):\n import torch,numpy as np\n from full_score import metric,cluster\n torch.set_num_threads(1)'+suffix+'\n';ns=v.original.__dict__.copy();exec(compile(code,'merge_preflight_v3','exec'),ns)
  def names(code):
   found={i.argval for i in dis.get_instructions(code)if i.opname=='LOAD_GLOBAL'}
   for x in code.co_consts:
    if hasattr(x,'co_code'):found|=names(x)
   return found
  need=names(ns['_merge'].__code__);assert all(x in ns or hasattr(builtins,x)for x in need),need-set(ns)-set(dir(builtins));proof[stage]=dict(all_generated_global_names_resolved=True,original_suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest(),canonical_folder=str(v.original.folder))
 v.put(v.D/'merge_preflight_v3.json',dict(status='both_unchanged_original_suffixes_compile_all_globals_resolved_exact_folder',proof=proof,new_model=0,new_GT_scoring=0))
if __name__=='__main__':
 if sys.argv[1]=='preflight':preflight()
 else:bind();v.merge(sys.argv[1])
