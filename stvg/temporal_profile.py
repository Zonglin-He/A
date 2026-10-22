"""Output-only temporal calibration; no network, evidence or MBR changes."""
import numpy as np
from scipy.special import logsumexp
from .temporal import probabilities,mass_table,final_mbr
BASE=dict(endpoint_temperature=1.,layer_slope=0.,length_power=0.)
def validate(params):
 if set(params)!=set(BASE):raise ValueError('Expected only the three locked temporal parameters')
 if not all(np.isfinite(x)for x in params.values()):raise ValueError('Nonfinite temporal profile')
 if not(0<params['endpoint_temperature']<=4 and -4<=params['layer_slope']<=4 and -2<=params['length_power']<=2):raise ValueError('Temporal profile exceeds protocol bounds')
def tables(records,logits,frame_ids,temperature=1.):
 out=[]
 for layer in range(6):
  grids=probabilities(records,[z[layer]for z in logits])
  if temperature!=1.:
   for g,z in zip(grids,logits):
    z=np.asarray(z[layer])[0].astype(np.float64);ij=g['ij'];score=(z[ij[0],0]+z[ij[1],1])/temperature;g['p']=np.exp(score-logsumexp(score))
  ids,q=mass_table(grids)
  if ids!=list(frame_ids):raise ValueError('Physical grid changed')
  out.append(q)
 return np.stack(out)
def combine(layer_tables,frame_ids,params):
 validate(params);layer_tables=np.asarray(layer_tables);assert layer_tables.shape==(6,len(frame_ids),len(frame_ids))
 slope=params['layer_slope']
 if slope==0:q=np.mean(layer_tables,axis=0)
 else:
  w=np.exp(slope*np.arange(6)/5);w/=w.sum();q=np.sum(layer_tables*w[:,None,None],axis=0)
 beta=params['length_power']
 if beta!=0:
  ids=np.asarray(frame_ids,dtype=np.float64);length=ids[None,:]+1-ids[:,None];legal=np.triu(np.ones(q.shape,dtype=bool),1);fraction=np.where(legal,length/(ids[-1]+1-ids[0]),1.)
  q=q*np.exp(-beta*np.log(fraction));q/=q.sum()
 if not np.isfinite(q).all()or abs(q.sum()-1)>1e-10:raise ValueError('Invalid calibrated posterior')
 return q
def posterior(records,logits,frame_ids,params):return combine(tables(records,logits,frame_ids,params['endpoint_temperature']),frame_ids,params)
