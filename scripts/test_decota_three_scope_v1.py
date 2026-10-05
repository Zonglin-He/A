"""Small contracts for physical alignment, selection and training exclusion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.decota_three_scope_v1 import *

def main():
    checks=0
    ids=[0,1,20,21,80,90]; a=tts_positions(ids,[0,5],[0,1,2,3,4,5]);assert len(a)==4 and len(set(a))==4;checks+=1
    assert tts_positions([0,1,2,3],[0,3],[1,1,1,1])==[0,1,2,3];checks+=1
    assert tts_positions([0,1],[0,1],[1,2])==[0,1];checks+=1
    assert tts_positions([0,1,2,3,4],[1,3],[1,0,9,0,1])==[1,2,3];checks+=1
    rec=[dict(frame_ids=[0,2,4]),dict(frame_ids=[1,3,5])]
    logits=[np.array([[4.,0],[1,1],[0,4]]),np.array([[4.,0],[1,1],[0,4]])]
    q=boundary(logits,rec,list(range(6)));qq=boundary(logits[::-1],rec[::-1],list(range(6)))
    assert all(q[k]==qq[k] for k in ['interval','indices','start','end','js_start','js_end','js_mean']) and q['indices'][0]<q['indices'][1];checks+=1
    assert abs(sum(q['start'])-1)<1e-14 and abs(sum(q['end'])-1)<1e-14;checks+=1
    rr=boundary([logits[0],logits[0]],[rec[0],rec[0]],rec[0]['frame_ids']);assert rr['js_mean']==0;checks+=1
    assert rr['interval']==[0,5];checks+=1
    x=np.arange(30).reshape(10,3);y=np.arange(10.)*.2;z=ridge_predict(x,y,x)
    assert np.isfinite(z).all() and np.corrcoef(z,y)[0,1]>.999;checks+=1
    d=dict(query=[1,0],spatial=[0,1],geometry=[.2,.4]);r=dict(query=[0,1],spatial=[0,1],geometry=[.3,.4])
    assert pair_features(d,r)[-2:].tolist()==[0.,1.];checks+=1
    print('THREE_SCOPE_CONTRACTS_PASS',checks)
if __name__=='__main__':main()
