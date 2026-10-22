"""Finite checks for completeness, spill, nuisance cancellation and safe fallback."""
import numpy as np
from core import relative_event,physical_membership,interval_scores,calibrated_masks,ridge_predict,guard
ids=list(range(40));pairs=np.stack(np.triu_indices(40,1),-1)
overlap,width,lengths=physical_membership(ids,pairs)
p=np.full((40,2),.01);p[10:20,0]=.99;p[20:30,1]=.99
s,_=interval_scores(p,overlap,width,lengths)
def index(a,b):return int(np.flatnonzero((pairs==[a,b]).all(1))[0])
# Full description has two phases. A strong endpoint peak or first phase alone
# must not replace coverage of both; appending background must also cost score.
assert s[index(10,29)]>s[index(10,19)]
assert s[index(10,29)]>s[index(0,39)]
z=np.random.default_rng(4).normal(size=(8,15,12));a=relative_event(z,[[3],[4]],[[1]],0)
# Object-independent additive token or frame offsets are true softmax gauges.
assert np.max(abs(a-relative_event(z+np.arange(8)[:,None,None],[[3],[4]],[[1]],0)))<1e-13
models=[[np.random.default_rng(9).normal(size=(40,2))]]
mask=calibrated_masks(models,width,.3)[0][0]
assert np.max(abs((width[:,None]*mask).sum(0)/width.sum()-.3))<1e-13
x=np.random.default_rng(3).normal(size=(40,5));pos=np.arange(0,40,5);y=x[pos,:2]
v,mu,sd=ridge_predict(x,y,np.arange(8)!=3,pos)
y2=y.copy();y2[3]+=100
vv,mm,ss=ridge_predict(x,y2,np.arange(8)!=3,pos)
assert np.array_equal(v,vv)and np.array_equal(mu,mm)and np.array_equal(sd,ss)
good=[dict(total_gain=1.),dict(total_gain=.1)]
scores=[np.array([0.,1.])for _ in range(18)];logs=np.array([[0.,0.],[1.,1.]])
assert guard(1,0,scores,logs,good)[0]
scores[-1]=np.array([0.,-.01]);assert not guard(1,0,scores,logs,good)[0]
assert not guard(1,0,scores,logs,[dict(total_gain=0.)])[0]
print('PASS: two-phase joint coverage, background-spill rejection, exact ownership gauge, native occupancy matching, no heldout-label leakage, disagreement/no-skill fallback')
