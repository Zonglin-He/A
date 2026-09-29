import numpy as np
from vg_tta.tastvg_preference_memory_v1 import append,select

def test_empty_memory_uses_native():
    assert select([],np.eye(3),[0.,2.,1.])[0]==1

def test_orientation_reversal_preserves_result():
    phi=np.eye(4);m=append([],phi,[4,3,2,1],1)
    rev=[{**z,'direction':-z['direction'],'label':-z['label']} for z in m]
    a=select(m,phi,[0]*4);b=select(rev,phi,[0]*4)
    assert a[:2]==b[:2]
    assert [x['vote'] for x in a[2]]==[x['vote'] for x in b[2]]

def test_top3_distinct_and_zero_differences():
    phi=np.eye(4);m=append([],phi,[4,3,2,1],1);_,_,pairs=select(m,phi,[0]*4)
    assert all(len(z['neighbors'])==3 and len({x['memory_index'] for x in z['neighbors']})==3 for z in pairs)
    assert append([],np.ones((2,4)),[1,0],1)==[]
    assert select(m,np.ones((2,4)),[0,1])[0]==1
