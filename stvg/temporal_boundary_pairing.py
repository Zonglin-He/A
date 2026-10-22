"""Cached interval pairing probe; no Source forward or spatial modification."""
import numpy as np
from scipy.special import softmax,logsumexp
from .temporal import probabilities,mass_table
from .temporal_readout import BASE as OLD_BASE,tables as old_tables,decode as old_decode
from .temporal_source_calibration import pair_geometry
BASE=dict(pair_strength=0.,radius_cells=1,risk_power=1.,offset_agreement=0.,last_layers=6)
def validate(p):
 if set(p)!=set(BASE)or not all(np.isfinite(x)for x in p.values()):raise ValueError('Invalid pairing parameters')
 if not(-2<=p['pair_strength']<=8 and p['radius_cells']in [0,1,2]and .25<=p['risk_power']<=4 and 0<=p['offset_agreement']<=4 and p['last_layers']in [1,3,6]):raise ValueError('Pairing bounds exceeded')
def native_grids(records,logits,p):
 grids=probabilities(records,logits)
 if p['pair_strength']==0:return grids
 for g,z in zip(grids,logits):
  z=np.asarray(z)[0].astype(np.float64);i,j=g['ij'];mass=softmax(z[:,0])+softmax(z[:,1]);prefix=np.r_[0,np.cumsum(mass)];lo=np.minimum(i+int(p['radius_cells'])+1,len(mass));hi=np.maximum(j-int(p['radius_cells']),0);interior=np.where(hi>lo,prefix[hi]-prefix[lo],0);score=z[i,0]+z[j,1]-p['pair_strength']*interior;g['p']=np.exp(score-logsumexp(score))
 return grids
def tables(records,logits,ids,p):
 validate(p)
 if p['pair_strength']==0 and p['offset_agreement']==0:return old_tables(records,logits,ids,OLD_BASE)
 result=[]
 for l in range(6):
  grids=native_grids(records,[z[l]for z in logits],p)
  if p['offset_agreement']==0:
   actual,q=mass_table(grids);assert actual==list(ids)
  else:
   key,iou=pair_geometry(grids,ids);w=grids[0]['p'][:,None]*grids[1]['p'][None]*np.exp(p['offset_agreement']*iou);w/=w.sum();q=np.bincount(key,weights=w.ravel(),minlength=len(ids)**2).reshape(len(ids),len(ids))
  assert np.isfinite(q).all()and abs(q.sum()-1)<1e-12;result.append(q)
 return np.stack(result)
def combine(a,ids,p):
 validate(p);assert np.asarray(a).shape==(6,len(ids),len(ids));return np.mean(a[-int(p['last_layers']):],axis=0)
def decode(ids,q,p):return old_decode(ids,q,{**OLD_BASE,'risk_power':p['risk_power']})
