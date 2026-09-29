import numpy as np
from vg_tta.tastvg_temporal_qualification_v1 import positions,corrupt_local,critic_scores,FAMILIES

def test_locality_determinism_and_severity():
    frames=np.random.default_rng(10).integers(0,256,(20,40,60,3),dtype=np.uint8)
    ids=list(range(0,40,2));pos,n=positions(ids,[8,32],10)
    assert pos==[9,10] and n==12
    for family in FAMILIES:
        a=corrupt_local(frames,pos,family,'example');b=corrupt_local(frames,pos,family,'example')
        assert np.array_equal(a,b) and np.array_equal(a[:9],frames[:9]) and np.array_equal(a[11:],frames[11:])
        assert not np.array_equal(a[pos],frames[pos])
    assert positions(ids,[8,32],1)[0]==positions(ids,[8,32],5)[0]

def test_candidate_bridge_and_native_ties():
    c=[[0,10],[2,6],[20,30]];r=critic_scores(c,[[2,6]],[.8])
    assert np.allclose(r,[.32,.8,0]) and np.argmax(r)==1
    assert np.argmax(critic_scores(c,[],[]))==0

def test_full_video_burst_independent_of_query_sampling():
    from vg_tta.tastvg_deployment_corruption_v2 import burst_spec
    for pct in [1,5,10]:
        full=burst_spec('video',1000,list(range(1000)),pct)
        cropped=burst_spec('video',1000,list(range(400,600,3)),pct)
        for k in ['physical_start','physical_end','length','uniform_draw']:
            assert full[k]==cropped[k]
        assert full['length']==10*pct
        assert 0<=full['physical_start']<full['physical_end']<=1000
        absent=[i for i in range(1000) if not full['physical_start']<=i<full['physical_end']]
        assert burst_spec('video',1000,absent,pct)['positions']==[]
