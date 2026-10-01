"""Read sealed A rewards, not GT; freeze calibration and the remaining arms."""
import time,numpy as np
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_proximal_common_v1 import *
def run():
 for ds in DATASETS:
  p=verify(ds);out=BASE/ds/'A';bar=read(out/'PREDICTION_BARRIER.json');assert bar['cells']==384
  spreads=[];receipts={};empty=0
  for cond in p['conditions']:
   for order,seq in p['splits']['search']['orders'].items():
    for at in range(0,len(seq),4):
     rf=out/'online'/cond/order/f'{at:05}.json';f=rf.with_suffix('.pt')
     assert sha(rf)==bar['files'][str(rf.relative_to(out))] and sha(f)==read(rf)['sha256']
     receipts[str(rf.relative_to(BASE))]=sha(rf);x=load(f);assert x['GT_read'] is False
     rr=x['update_steps'][0]['rewards']
     if rr is None:empty+=1;spreads.append(0.)
     else:spreads.append(float(np.ptp(rr)))
  positive=[v for v in spreads if v>0.];assert positive
  sr=float(np.median(positive));avg=float(np.mean([v/(v+sr) for v in spreads]))
  write(BASE/ds/'CALIBRATION.json',dict(s_ref=sr,fixed_lambda=avg,spreads=spreads,
   empty=empty,receipts=receipts,GT_read=False,arm_A_barrier_sha256=sha(out/'PREDICTION_BARRIER.json'),time=time.time()))
  radius=read(out/'SUPPORT.json')['spec']['radius']*.1
  for arm,mode,cap in [('B','proximal',None),('C','rank',radius),('D','proximal',radius)]:
   write(BASE/ds/arm/'REQUEST.json',dict(arm=arm,tag=arm,split='search',params=p['params'],
    method=dict(target_mode=mode,s_ref=sr,arrival_radius=cap)))
 write(BASE/'CALIBRATION_BARRIER.json',dict(GT_read=False,
  files={str(f.relative_to(BASE)):sha(f) for ds in DATASETS for f in
   [BASE/ds/'CALIBRATION.json',*[BASE/ds/a/'REQUEST.json' for a in ['B','C','D']]]},time=time.time()))
if __name__=='__main__':run()
