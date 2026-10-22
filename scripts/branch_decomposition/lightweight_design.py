"""Generic single-high-view posterior interface, not an adopted method.

Deployment must really compute one high-resolution view; slicing outputs after
computing both is not a speedup. No hidden/attention/auxiliary-head input.
"""
import numpy as np

def physical_view(ids,start_indices,end_indices,probabilities,physical_grid):
 """Map a model's normalized interval posterior to a common physical grid."""
 ids=np.asarray(ids,dtype=np.int64);U=np.asarray(physical_grid,dtype=np.int64);a=np.asarray(start_indices,dtype=np.int64);b=np.asarray(end_indices,dtype=np.int64);p=np.asarray(probabilities,float)
 assert p.shape==a.shape==b.shape and (a<b).all()and np.isfinite(p).all()and(p>=0).all()and abs(p.sum()-1)<1e-10
 assert (np.diff(ids)>0).all()and(np.diff(U)>0).all();loc={int(v):i for i,v in enumerate(U)};q=np.zeros((len(U),len(U)),float);ia=np.asarray([loc[int(ids[k])]for k in a]);ib=np.asarray([loc[int(ids[k])]for k in b]);np.add.at(q,(ia,ib),p);return q

def single_high_view_fusion(global_low,high_view):
 """Keep the old equal weighting; change the expensive view count only."""
 g=np.asarray(global_low,float);h=np.asarray(high_view,float);assert g.shape==h.shape and g.ndim==2 and g.shape[0]==g.shape[1]
 legal=np.triu(np.ones(g.shape,bool),1)
 for q in [g,h]:assert np.isfinite(q).all()and(q>=0).all()and not q[~legal].any()and abs(q.sum()-1)<1e-10
 return .5*g+.5*h
