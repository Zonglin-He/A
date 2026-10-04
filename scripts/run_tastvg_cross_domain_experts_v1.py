"""Bind the unchanged specialist worker to this clean, isolated target cohort."""
import sys, time, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_cross_domain_common_v1 import *

def run(direction, stage):
    p=verify(direction); bind_decode(p['target_dataset'])
    from scripts import tastvg_best_quick_expert_worker_v1 as w
    out=BASE/direction/'experts'; ep=read(BASE/direction/'EXPERT_PLAN.json')
    # Only an immutable exact-input index can import historical receipts.
    w.OUT=out; w.OLD=out/'no_implicit_historical_cache'
    w.verify=lambda:(verify(direction), ep)[1]
    indexfile=BASE/direction/'EXPERT_REUSE_INDEX.json'
    assert sha(indexfile)==read(BASE/direction/'EXPERT_REUSE_BARRIER.json')['index_sha256']
    index=read(indexfile)
    def reuse(stage, plan):
        for i in plan['expert_needed']:
            rel=f'{stage}/clean/{i:05}.json';dst=out/rel
            if dst.exists() or rel not in index:continue
            rec=index[rel];f=Path(rec['receipt']);assert sha(f)==rec['receipt_sha256']
            old=read(f);src=Path(rec['folder'])/old['cache'];assert sha(src)==old['cache_sha256']
            dest=out/old['cache'];dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():os.link(src,dest)
            else:assert sha(dest)==old['cache_sha256']
            write(dst,{**old,'parent':i,'reused_from':str(f),'new_inference':False})
    w.reuse_receipts=reuse
    w.guard=guard; w.budget=budget
    w.run(stage)

if __name__=='__main__':run(*sys.argv[1:])
