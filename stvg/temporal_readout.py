"""Research-only temporal readout knobs; never registered as the maintained default.

Every trial uses the same saved native logits and physical support. Composition of
native offset distributions and MBR utility changes are explicit, not compatibility
exceptions for the completed full queue.
"""
import numpy as np
from scipy.special import logsumexp
from .temporal import probabilities,mass_table,final_mbr
BASE=dict(start_temperature=1.,end_temperature=1.,hull_weight=1.,last_layers=6,risk_power=1.)

def validate(p):
    if set(p)!=set(BASE)or not all(np.isfinite(x)for x in p.values()):raise ValueError('Invalid temporal readout profile')
    if not(.25<=p['start_temperature']<=2 and .25<=p['end_temperature']<=2 and 0<=p['hull_weight']<=1 and p['last_layers']in (1,3,6)and .25<=p['risk_power']<=4):raise ValueError('Temporal readout exceeds protocol bounds')

def tables(records,logits,frame_ids,params):
    validate(params);result=[];idx={int(x):i for i,x in enumerate(frame_ids)};n=len(idx)
    for layer in range(6):
        grids=probabilities(records,[z[layer]for z in logits])
        if params['start_temperature']!=1. or params['end_temperature']!=1.:
            for g,z in zip(grids,logits):
                z=np.asarray(z[layer])[0].astype(np.float64);i,j=g['ij'];v=z[i,0]/params['start_temperature']+z[j,1]/params['end_temperature'];g['p']=np.exp(v-logsumexp(v))
        ids,hull=mass_table(grids)
        if ids!=list(frame_ids):raise ValueError('Physical support changed')
        if params['hull_weight']==1.:q=hull
        else:
            mixture=np.zeros_like(hull)
            for g in grids:
                mapped=np.array([idx[int(x)]for x in g['ids']]);i,j=g['ij'];np.add.at(mixture,(mapped[i],mapped[j]),g['p']*.5)
            q=params['hull_weight']*hull+(1-params['hull_weight'])*mixture
        if not np.isfinite(q).all()or abs(q.sum()-1)>1e-12:raise ValueError('Invalid layer posterior')
        result.append(q)
    return np.stack(result)

def combine(layer_tables,frame_ids,params):
    validate(params);a=np.asarray(layer_tables,dtype=np.float64);assert a.shape==(6,len(frame_ids),len(frame_ids))
    return np.mean(a[-int(params['last_layers']):],axis=0)

def decode(ids,q,params):
    validate(params)
    if params['risk_power']==1.:return final_mbr(ids,q)
    # Exact blocked finite E[IoU**power]; no posterior pruning or grid coarsening.
    ids=np.asarray(ids);n=len(ids);q=np.asarray(q,dtype=np.float64);i,j=np.triu_indices(n,1)
    assert q.shape==(n,n)and abs(q.sum()-1)<1e-10 and(q>=0).all()and not q[np.tril_indices(n)].any()
    starts=ids[i].astype(np.float64);ends=ids[j].astype(np.float64)+1;weights=q[i,j];values=np.empty(len(i))
    for at in range(0,len(i),128):
        sl=slice(at,min(at+128,len(i)));inter=np.maximum(0,np.minimum(ends[sl,None],ends[None])-np.maximum(starts[sl,None],starts[None]));union=np.maximum(ends[sl,None],ends[None])-np.minimum(starts[sl,None],starts[None]);iou=inter/union;values[sl]=(np.power(iou,params['risk_power'])*weights[None]).sum(1)
    u=np.full((n,n),-np.inf);u[i,j]=values;k=int(np.argmax(u));a,b=divmod(k,n)
    return dict(interval=[int(ids[a]),int(ids[b])+1],indices=[a,b],utility=float(u[a,b]),utility_grid=u)

def posterior(records,logits,ids,params):return combine(tables(records,logits,ids,params),ids,params)
