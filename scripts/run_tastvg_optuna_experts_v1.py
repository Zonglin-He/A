import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_optuna_common_v1 import *
def run(dataset,stage):
 p=verify(dataset);bind_decode(dataset)
 from scripts import run_tastvg_paper48_experts_v1 as w
 old=ROOT/'artifacts/tastvg_paper48_v1';old=old/'experts' if dataset=='vidstg' else old/'P5/experts';out=BASE/dataset/'experts'
 def reuse(stage,ep):
  for i in ep['expert_needed']:
   for cond in ep['conditions_by_parent'][str(i)]:
    src=old/stage/cond/f"{p['rows'][i]['original_parent']:05}.json";dst=out/stage/cond/f'{i:05}.json'
    if dst.exists() or not src.exists():continue
    rr=read(src);cf=old/rr['cache'];assert sha(cf)==rr['cache_sha256'];dest=out/rr['cache'];dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():os.link(cf,dest)
    write(dst,{**rr,'parent':i,'reused_from':str(src),'new_inference':False})
 w.OUT=out;w.OLD=old;w.verify=lambda:(verify(dataset),read(BASE/dataset/'EXPERT_PLAN.json'))[1];w.reuse_receipts=reuse;w.guard=guard;w.budget=lambda tick:budget();w.run(stage)
if __name__=='__main__':run(*sys.argv[1:])
