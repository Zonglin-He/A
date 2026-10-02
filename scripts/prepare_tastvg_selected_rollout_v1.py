"""The same original development cohort; no annotations or new teachers."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_selected_rollout_common_v1 import *
def run():
 from scripts.tastvg_directional_common_v1 import verify as previous_verify
 for ds in DATASETS:
  p=previous_verify(ds)
  p={**p,'baseline_result_dir':f'artifacts/tastvg_directional_preference_v1/{ds}/A'}
  assert read(ROOT/p['baseline_result_dir']/'PREDICTION_BARRIER.json')['cells']==384
  write(BASE/ds/'PLAN.json',p)
  for arm,mode in [('A','rkl'),('G','selected_rollout')]:
   write(BASE/ds/arm/'REQUEST.json',dict(arm=arm,tag=arm,split='search',params=p['params'],method=dict(target_mode='rank',s_ref=1.,arrival_radius=None,actuation=mode)))
 write(BASE/'STATUS.json',dict(status='registered_pending_runtime_and_smoke',GPU_predictions=0,GT_read=False,time=time.time()))
if __name__=='__main__':run()
