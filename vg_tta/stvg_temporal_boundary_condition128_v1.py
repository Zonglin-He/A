"""Physical boundary key masks and conditional labels; no new model."""
import numpy as np

def key_mask(frame_ids,boundary,direction,variant):
 ids=np.asarray(frame_ids);assert ids.ndim==1 and np.all(np.diff(ids)>0)
 keep=ids<=boundary if direction=='start'else ids>=boundary
 assert direction in ['start','end']and variant in ['Correct','Wrong']and keep.any()
 if variant=='Wrong':
  count=int(keep.sum());keep=np.zeros(len(ids),bool)
  if direction=='start':keep[-count:]=True
  else:keep[:count]=True
 return ~keep

def conditional(q,p0,pair,direction):
 i,j=map(int,pair);assert p0[i,j]>0
 if direction=='start':mask=p0[:,j]>0;v=q[:,j].copy()
 else:mask=p0[i,:]>0;v=q[i,:].copy()
 v[~mask]=0.;assert v[mask].sum()>0 and np.isfinite(v).all();v/=v.sum();assert np.array_equal(v>0,mask)and abs(v.sum()-1)<1e-12
 return v

def properties():
 ids=np.array([0,8,16,24]);a=key_mask(ids,16,'start','Correct');b=key_mask(ids,16,'start','Wrong');assert np.array_equal(a,[False,False,False,True])and np.array_equal(b,[True,False,False,False])and a.sum()==b.sum();assert not key_mask(ids,24,'start','Correct').any();assert not key_mask(ids,0,'end','Correct').any()
 q=np.triu(np.arange(1,26).reshape(5,5).astype(float),1);q/=q.sum();v=conditional(q,q,[1,3],'start');assert np.array_equal(v>0,q[:,3]>0)
 return dict(status='pass',physical_predicates=True,opposite_side_same_count=True,full_mask_identity=True,conditional_support_normalization=True)
