import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_best_quick_common_v1 import *
def run(dataset,stage):
 p=verify(dataset);bind_decode(dataset)
 from scripts import tastvg_best_quick_expert_worker_v1 as w
 out=BASE/dataset/'experts';index=read(BASE/dataset/'EXPERT_REUSE_INDEX.json');assert sha(BASE/dataset/'EXPERT_REUSE_INDEX.json')==read(BASE/dataset/'SUBJECT_BARRIER.json')['expert_reuse_index_sha256']
 def reuse(stage,ep):
  for i in ep['expert_needed']:
   for cond in ep['conditions']:
    rel=f'{stage}/{cond}/{i:05}.json';dst=out/rel
    if dst.exists() or rel not in index:continue
    rec=index[rel];rf=Path(rec['receipt']);assert sha(rf)==rec['receipt_sha256'];rr=read(rf);src=Path(rec['folder'])/rr['cache'];assert sha(src)==rr['cache_sha256'];dest=out/rr['cache'];dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():os.link(src,dest)
    else:assert sha(dest)==rr['cache_sha256']
    write(dst,{**rr,'parent':i,'reused_from':str(rf),'new_inference':False})
 # Exact receipts need no specialist residency or new GPU calls.
 import time
 tick=time.monotonic();ep=read(BASE/dataset/'EXPERT_PLAN.json');reuse(stage,ep)
 expected=[out/stage/cond/f'{i:05}.json' for i in ep['expert_needed'] for cond in ep['conditions']]
 if all(f.exists() for f in expected):
  files={}
  for f in expected:
   rr=read(f);assert sha(out/rr['cache'])==rr['cache_sha256'];files[str(f.relative_to(out))]=sha(f)
  write(out/f'{stage.upper()}_BARRIER.json',dict(cells=ep['total'],files=files,GT_read=False,time=time.time()))
  status(out/f'{stage.upper()}_STATUS.json',dict(status='completed',done=ep['total'],total=ep['total'],new_inferences_this_run=0,seconds=time.monotonic()-tick,specialist_loaded=False,exact_receipts_reused=True))
  print('EXPERT_CACHE_REUSED',dataset,stage,ep['total'],flush=True);return
 w.OUT=out;w.verify=lambda:(verify(dataset),read(BASE/dataset/'EXPERT_PLAN.json'))[1];w.reuse_receipts=reuse;w.guard=guard;w.budget=budget;w.run(stage)
if __name__=='__main__':run(*sys.argv[1:])
