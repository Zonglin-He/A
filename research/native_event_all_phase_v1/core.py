"""Exact ordered event-assignment marginal, retaining first start and final end."""
import numpy as np
from research.native_event_order_v1.core import ordered


def all_ordered(phases):
 assert len(phases)>=2
 if len(phases)==2:return ordered(*phases)
 q=[np.asarray(x,dtype=np.float64)for x in phases];n=len(q[0]);legal=np.triu(np.ones((n,n),bool),1)
 for x in q:assert x.shape==(n,n)and(x>=0).all()and not x[~legal].any()and abs(x.sum()-1)<1e-10
 # Axis0 preserves the original start. Axes1,2 are the most recent event.
 dp=np.zeros((n,n,n));idx=np.arange(n);dp[idx,idx,:]=q[0];mass=1.
 for event in q[1:]:
  dp=dp.cumsum(axis=1).cumsum(axis=2)*event[None,:,:]
  z=float(dp.sum());assert 0<z<=1+1e-12;mass*=z;dp/=z
 out=dp.sum(axis=1);assert not out[~legal].any()and abs(out.sum()-1)<1e-12
 return out,mass
