"""Pure NumPy cached geometry/statistics; no models, labels or data loader."""
import numpy as np


def box_stats(b):
    b=np.asarray(b,np.float64);lo=b[...,:2]-b[...,2:]/2;hi=b[...,:2]+b[...,2:]/2
    finite=np.isfinite(b).all(-1);coord=(b>0)&(b<1)
    intersection=np.maximum(np.minimum(hi,1)-np.maximum(lo,0),0).prod(-1)
    area=b[...,2:].prod(-1)
    return dict(legal=finite&coord.all(-1),finite=finite,coordinate_legal=coord,
                inside=(lo>=0).all(-1)&(hi<=1).all(-1),visible=np.divide(intersection,area,out=np.zeros_like(area),where=area>0))


def iou(a,b):
    a=np.asarray(a,np.float64).reshape(-1,4);b=np.asarray(b,np.float64).reshape(-1,4)
    if not len(b):return np.empty((len(a),0))
    al=a[:,:2]-a[:,2:]/2;ah=a[:,:2]+a[:,2:]/2;bl=b[:,:2]-b[:,2:]/2;bh=b[:,:2]+b[:,2:]/2
    inter=np.maximum(np.minimum(ah[:,None],bh)-np.maximum(al[:,None],bl),0).prod(-1)
    union=a[:,2:].prod(-1)[:,None]+b[:,2:].prod(-1)-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)


def quant(v):
    x=np.asarray([float(z)for z in v if z is not None],np.float64)
    return dict(n=len(x),mean=float(x.mean())if len(x)else None,
        quantiles=None if not len(x)else dict(zip(['min','q25','median','q75','q90','q95','q99','max'],map(float,np.quantile(x,[0,.25,.5,.75,.9,.95,.99,1])))))


def slotbin(slot):
    return '000-009'if slot<10 else '010-049'if slot<50 else '050-199'if slot<200 else '200-499'if slot<500 else '500-899'


def overlapbin(x):
    return 'unknown'if x is None else '[0,.25)'if x<.25 else '[.25,.50)'if x<.5 else '[.50,.75)'if x<.75 else '[.75,1]'


def visiblebin(x):return '[0,.5)'if x<.5 else '[.5,.9)'if x<.9 else '[.9,1)'if x<1-1e-12 else '1'


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])


def score_summary(rows):
    scores=np.array([r['score']for r in rows],float)
    hist,edges=np.histogram(scores,[0,.05,.10,.20,.25,.30,.35,.40,.50,1.00000001])
    match=[r for r in rows if r['matched_native_slot_score']is not None]
    confident=[r for r in match if r['matched_native_slot_score']>=.35]
    return dict(**quant(scores),accepted=int((scores>=.35).sum()),rejected=int((scores<.35).sum()),
        near_below_030_035=int(((scores>=.30)&(scores<.35)).sum()),far_below_030=int((scores<.30).sum()),
        histogram=[dict(low=float(a),high=min(1.,float(b)),count=int(n))for a,b,n in zip(edges[:-1],edges[1:],hist)],
        known_original_Top3_slot_matches=len(match),
        original_same_slot_score_ge035=len(confident),
        original_ge035_but_conditional_below035=sum(r['score']<.35 for r in confident),
        conditional_minus_original_score=quant([r['score']-r['matched_native_slot_score']for r in match]))


def public_value(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return public_value(x.tolist())
    if isinstance(x,dict):return {k:public_value(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [public_value(v)for v in x]
    return x
