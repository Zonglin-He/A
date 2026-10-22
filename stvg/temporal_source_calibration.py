"""Source-primary cached temporal calibration research; no spatial/model changes."""
import numpy as np
from scipy.special import softmax,logsumexp
from .temporal import probabilities,mass_table
from .temporal_readout import BASE as OLD_BASE,validate as old_validate,tables as old_tables,decode as old_decode
BASE={**OLD_BASE,'pool_power':1.,'offset_agreement':0.,'adaptive_start':0.,'adaptive_threshold':.75,'source_mix':0.,'source_trust':0.,'nucleus_mass':1.}
_PAIR_CACHE=None

def validate(p):
    if set(p)!=set(BASE)or not all(np.isfinite(v)for v in p.values()):raise ValueError('Invalid Source-primary parameters')
    old_validate({k:p[k]for k in OLD_BASE})
    if not(-1<=p['pool_power']<=1 and -4<=p['offset_agreement']<=8 and 0<=p['adaptive_start']<=1 and 0<=p['adaptive_threshold']<1 and 0<=p['source_mix']<=1 and 0<=p['source_trust']<=4 and .5<=p['nucleus_mass']<=1):raise ValueError('Source-primary bounds exceeded')

def pair_geometry(grids,ids):
    global _PAIR_CACHE
    signature=tuple(tuple(g['ids'])for g in grids)
    if _PAIR_CACHE is not None and _PAIR_CACHE[0]==signature:return _PAIR_CACHE[1:]
    s=[];e=[];ss=[];ee=[];lookup={int(x):i for i,x in enumerate(ids)}
    for g in grids:
        i,j=g['ij'];f=np.asarray(g['ids']);s.append(f[i]);e.append(f[j]+1);ss.append(np.array([lookup[int(x)]for x in f[i]]));ee.append(np.array([lookup[int(x)]for x in f[j]]))
    intersection=np.maximum(0,np.minimum(e[0][:,None],e[1][None])-np.maximum(s[0][:,None],s[1][None]));union=np.maximum(e[0][:,None],e[1][None])-np.minimum(s[0][:,None],s[1][None]);iou=intersection/union;key=(np.minimum(ss[0][:,None],ss[1][None])*len(ids)+np.maximum(ee[0][:,None],ee[1][None])).ravel()
    _PAIR_CACHE=(signature,key,iou)if key.nbytes+iou.nbytes<=256*2**20 else None
    return key,iou

def tables(records,logits,ids,p):
    validate(p);original=old_tables(records,logits,ids,OLD_BASE)
    unchanged=all(p[k]==BASE[k]for k in ['start_temperature','end_temperature','hull_weight','offset_agreement','adaptive_start','nucleus_mass'])
    if unchanged:return np.stack([original,original])
    modified=[]
    for l in range(6):
        grids=probabilities(records,[z[l]for z in logits])
        for g,z in zip(grids,logits):
            z=np.asarray(z[l])[0].astype(np.float64);i,j=g['ij'];ps=softmax(z[:,0]);h=float(-(ps*np.log(np.maximum(ps,1e-300))).sum()/np.log(len(ps)));ambiguity=np.clip((h-p['adaptive_threshold'])/(1-p['adaptive_threshold']),0,1);ts=p['start_temperature']+p['adaptive_start']*ambiguity;v=z[i,0]/ts+z[j,1]/p['end_temperature'];prob=np.exp(v-logsumexp(v))
            if p['nucleus_mass']<1:
                order=np.argsort(-prob,kind='stable');n=min(len(order),int(np.searchsorted(np.cumsum(prob[order]),p['nucleus_mass'],side='left'))+1);keep=order[:n];truncated=np.zeros_like(prob);truncated[keep]=prob[keep];prob=truncated/truncated.sum()
            g['p']=prob
        if p['offset_agreement']==0:
            actual,q=mass_table(grids);assert actual==list(ids)
        else:
            key,iou=pair_geometry(grids,ids);w=grids[0]['p'][:,None]*grids[1]['p'][None]*np.exp(p['offset_agreement']*iou);w/=w.sum();q=np.bincount(key,weights=w.ravel(),minlength=len(ids)**2).reshape(len(ids),len(ids))
        if p['hull_weight']!=1:
            mixture=np.zeros_like(q);lookup={int(f):i for i,f in enumerate(ids)}
            for g in grids:
                mapped=np.array([lookup[int(f)]for f in g['ids']]);i,j=g['ij'];np.add.at(mixture,(mapped[i],mapped[j]),.5*g['p'])
            q=p['hull_weight']*q+(1-p['hull_weight'])*mixture
        assert np.isfinite(q).all()and abs(q.sum()-1)<1e-12
        modified.append(q)
    return np.stack([np.stack(modified),original])

def combine(blocks,ids,p,trust_reference_js):
    validate(p);a,original=np.asarray(blocks);assert a.shape==original.shape==(6,len(ids),len(ids));active=a[-int(p['last_layers']):]
    if p['pool_power']==1:q=active.mean(0)
    else:
        valid=np.any(active>0,axis=0);values=np.maximum(active[:,valid],1e-300);power=p['pool_power'];q=np.zeros(active.shape[1:]);q[valid]=np.exp(np.log(values).mean(0))if power==0 else np.power(np.power(values,power).mean(0),1/power);q/=q.sum()
    if p['source_mix']!=0 or p['source_trust']!=0:
        assert trust_reference_js>0
        mean=original.mean(0);js=float((original*np.log(np.maximum(original,1e-300)/np.maximum(mean[None],1e-300))).sum()/6);weight=(1-p['source_mix'])/(1+p['source_trust']*js/trust_reference_js);q=(1-weight)*original[-1]+weight*q
    assert np.isfinite(q).all()and abs(q.sum()-1)<1e-10 and not q[np.tril_indices(len(ids))].any()
    return q

def decode(ids,q,p):return old_decode(ids,q,{k:p[k]for k in OLD_BASE})
