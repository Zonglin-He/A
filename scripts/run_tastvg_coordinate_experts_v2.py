"""Reuse matching cached specialists; prepare only missing query observations."""
import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_coordinate_common_v2 import *
def run(dataset,stage):
 p=verify(dataset);bind_decode(dataset)
 from scripts import run_tastvg_paper48_experts_v1 as w
 paper=ROOT/'artifacts/tastvg_paper48_v1';paper=paper/'experts' if dataset=='vidstg' else paper/'P5/experts'
 prior=ROOT/'artifacts/tastvg_optuna_v1'/dataset;op=read(prior/'PLAN.json');lookup={r['key']:i for i,r in enumerate(op['rows'])};out=BASE/dataset/'experts'
 def reuse(stage,ep):
  for i in ep['expert_needed']:
   row=p['rows'][i]
   for cond in ep['conditions_by_parent'][str(i)]:
    dst=out/stage/cond/f'{i:05}.json'
    if dst.exists():continue
    sources=[]
    if row['key'] in lookup:
     j=lookup[row['key']];assert op['rows'][j]['input']==row['input'] and op['rows'][j]['frame_ids']==row['frame_ids'];sources.append((prior/'experts',j))
    sources.append((paper,row['original_parent']))
    for folder,j in sources:
     src=folder/stage/cond/f'{j:05}.json'
     if not src.exists():continue
     rr=read(src);cf=folder/rr['cache'];assert sha(cf)==rr['cache_sha256'];dest=out/rr['cache'];dest.parent.mkdir(parents=True,exist_ok=True)
     if not dest.exists():os.link(cf,dest)
     else:assert sha(dest)==rr['cache_sha256']
     write(dst,{**rr,'parent':i,'reused_from':str(src),'new_inference':False});break
 w.OUT=out;w.OLD=paper;w.verify=lambda:(verify(dataset),read(BASE/dataset/'EXPERT_PLAN.json'))[1];w.reuse_receipts=reuse;w.guard=guard;w.budget=lambda tick:budget();w.run(stage)
if __name__=='__main__':run(*sys.argv[1:])
