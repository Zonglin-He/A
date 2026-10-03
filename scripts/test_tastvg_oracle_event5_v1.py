"""Analytic controls for endpoint/support, oracle bounds and matched interventions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.tastvg_oracle_event5_v1 import event_positions,DenseTube,official,box_iou
def run():
    z=event_positions([0,2,4,6,8,10],[2,9]);assert not z['eligible'] and z['available']==4 and not z['positions']
    z=event_positions(list(range(11)),[2,9]);assert z['positions']==[2,4,5,6,8]
    row=dict(frame_ids=[0,2,4,6],input=dict(width=10,height=10));b=np.tile([.5,.5,1,1],(4,1));truth={i:[0,0,10,10] for i in range(1,6)};span=[1,6]
    for ds in ['vidstg','hc2']:
        a=DenseTube(b,row,truth,span,ds=='hc2');m=official(b,row,truth,span,[0,7],ds)
        assert abs(a.score([0,7])['v']-m['v'])<1e-12
        assert official(None,row,truth,span,span,ds,True)['v']==1
        assert abs(official(None,row,truth,span,[0,7],ds,True)['v']-5/7)<1e-12
    # No extrapolation: GT space is perfect even when A observed support is short.
    short=dict(row,frame_ids=[2,4]);a=DenseTube(b[:2],short,truth,span)
    assert a.score(span)['v']==3/5 and official(None,short,truth,span,span,'vidstg',True)['v']==1
    # The official denominator keeps the full GT duration for a short sub-event.
    ids=list(range(10));r=dict(row,frame_ids=ids);q=np.tile([2,2,.1,.1],(10,1));q[:2]=[.5,.5,1,1]
    a=DenseTube(q,r,{i:[0,0,10,10] for i in ids},[0,10]);assert a.score([0,2])['v']==a.score([0,10])['v']==.2
    assert float(box_iou([0,0,1,1],[0,0,1,1]))==1
    from vg_tta.tastvg_current_correction_views_v1 import select
    assert select(None)['selected']==0 and select([0,1,1])['selected']==0
    print('PASS 8 analytic oracle/event5 controls',flush=True)
if __name__=='__main__':run()
