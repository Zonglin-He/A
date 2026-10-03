"""Meaningful CPU tests for coverage strata, native ties and oracle denominator."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.tastvg_temporal_candidate_coverage_v1 import *
from vg_tta.tastvg_oracle_event5_v1 import DenseTube, official

def run():
    ids=list(range(0,24,2)); records=[dict(frame_ids=ids[::2]),dict(frame_ids=ids[1::2])]
    logits=[np.zeros((1,len(r['frame_ids']),2)) for r in records]
    prior=endpoint_prior(logits,records,ids); assert np.allclose(prior,1/12)
    pool=allocate([3,8],ids,logits,records)
    assert pool[0]['indices']==[3,8] and len({tuple(c['indices']) for c in pool})==8
    assert not any(c['fallback'] for c in pool)
    assert choose(pool,[],[])['selected']==0
    assert allocate([3,8],ids,logits,records)==pool
    for q in pool[1:]:
        a,z=q['physical_interval'];d=z-a;cen=a+z-2*ids[0];L=ids[-1]+1-ids[0]
        b=min(2,3*cen//(2*L))
        assert q['position_bin'] is None or q['position_bin']==b
        band=0 if 3*d<=L else 1 if 3*d<=2*L else 2
        assert q['duration_band']==band
    row=dict(frame_ids=ids,input=dict(width=10,height=10));boxes=np.tile([.5,.5,1,1],(len(ids),1))
    truth={f:[0,0,10,10] for f in range(3,18)}
    for ds in ['vidstg','hc2']:
        s=DenseTube(boxes,row,truth,[3,18],ds=='hc2');v=grid_values(s,ids)
        brute=np.array([s.score(i)['v'] for i in v['intervals']])
        assert np.allclose(v['values'],brute,rtol=0,atol=1e-14)
        assert v['count']==len(ids)*(len(ids)-1)//2
        for at in [0,v['best'],len(brute)-1]:
            assert abs(official(boxes,row,truth,[3,18],v['intervals'][at],ds)['v']-brute[at])<1e-14
        assert brute.max()<=s.score([3,18])['v']+1e-14
    # A zero-quality tube keeps all scores zero, with deterministic grid/native ties.
    zero=np.tile([2.,2.,.1,.1],(len(ids),1));s=DenseTube(zero,row,truth,[3,18])
    assert grid_values(s,ids)['values'].max()==0
    # Sparse small grid explicitly records exhausted-stratum fallback, without labels.
    ids2=list(range(5));rs=[dict(frame_ids=ids2[::2]),dict(frame_ids=ids2[1::2])]
    q=allocate([0,4],ids2,[np.zeros((1,len(r['frame_ids']),2)) for r in rs],rs)
    assert len(q)==8 and any(c['fallback'] for c in q)
    print('PASS 10 analytic coverage/grid/scorer controls',flush=True)
if __name__=='__main__': run()
