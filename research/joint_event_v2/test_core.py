import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from research.joint_event_v1.core import physical_membership
from research.joint_event_v2.core import excess_scores
ids=list(range(40));pairs=np.stack(np.triu_indices(40,1),-1);overlap,width,lengths=physical_membership(ids,pairs)
for mu in [.1,.3,.8]:
    score,logs=excess_scores(np.full((40,2),mu),overlap,width,lengths)
    assert np.max(abs(score))<1e-13
p=np.full((40,2),.01);p[10:20,0]=.99;p[20:30,1]=.99
s,l=excess_scores(p,overlap,width,lengths)
def idx(a,b):return int(np.flatnonzero((pairs==[a,b]).all(1))[0])
assert abs(s[idx(0,39)])<1e-13
assert s[idx(10,29)]>s[idx(10,19)]and s[idx(10,29)]>s[idx(0,39)]
print('PASS: constant-mask all-interval zero, arbitrary-mask full-video zero, multi-phase completeness, no reward from length alone')
