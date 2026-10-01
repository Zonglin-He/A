"""Lock matching inputs and baseline before new inference; no labels."""
import time
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_proximal_common_v1 import *
def run():
 from scripts.tastvg_extended_common_v3 import verify as previous_verify
 for ds in DATASETS:
  p=previous_verify(ds);sel=read(POOL/ds/'SELECTION.json');cfg=sel['params']
  ts=read(POOL/ds/'TRIALS.json');matched=[x for x in ts if x['state']=='COMPLETE' and x['params']==cfg];assert matched
  tag=matched[0]['result_dir'];assert read(POOL/ds/tag/'PREDICTION_BARRIER.json')['cells']==384
  p={**p,'params':cfg,'selected_result_dir':tag,'splits':{'search':p['splits']['search']}}
  write(BASE/ds/'PLAN.json',p)
  write(BASE/ds/'A/REQUEST.json',dict(arm='A',tag='A',split='search',params=cfg,
   method=dict(target_mode='rank',s_ref=1.,arrival_radius=None)))
 write(BASE/'STATUS.json',dict(status='registered_pending_runtime_and_smoke',GPU_predictions=0,GT_read=False))
if __name__=='__main__':run()
