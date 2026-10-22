"""Mechanism tests; do not access development labels or models."""
import numpy as np
from core import boundary_curve, groups_for, reweight

ids = list(range(64))
h = np.zeros((6,1,64,2));h[:,:,:32,0]=1;h[:,:,32:,1]=1
offsets=[dict(hidden=h[:,:,o::2],frame_ids=ids[o::2])for o in range(2)]
c, raw = boundary_curve(offsets,ids)
assert all(abs(r['cuts'][np.argmax(r['strengths'])]-32)<2 for r in raw)
assert c[0,31] > c[0,15]
constant=[dict(hidden=np.ones_like(o['hidden']),frame_ids=o['frame_ids'])for o in offsets]
assert not boundary_curve(constant,ids)[0].any()
pairs=np.stack(np.triu_indices(64,1),axis=-1)
patterns,g=groups_for(ids,[0,9,18,27,36,45,54,63],pairs)
q=np.zeros((64,64));q[pairs[:,0],pairs[:,1]]=np.random.default_rng(7).uniform(size=len(pairs));q/=q.sum()
s=c[0,pairs[:,0]]+c[1,pairs[:,1]]
t=reweight(q,pairs,s,g)
assert np.max(abs(np.bincount(g,weights=t[pairs[:,0],pairs[:,1]])-np.bincount(g,weights=q[pairs[:,0],pairs[:,1]])))<1e-14
assert np.max(abs(q-reweight(q,pairs,np.zeros(len(pairs)),g)))<1e-15
assert np.max(abs(t-reweight(q,pairs,s+10,g)))<1e-15
scaled=[dict(hidden=o['hidden']*7,frame_ids=[10+3*x for x in o['frame_ids']])for o in offsets]
# Physical window centers retain correct location under affine time transform.
_, sr=boundary_curve(scaled,[10+3*x for x in ids])
assert np.max(abs(sr[0]['strengths']-raw[0]['strengths']))<1e-14
print('PASS: actual change-point localization, constant-evidence no-op, conditional native-mass conservation, score gauge, physical time transform')
