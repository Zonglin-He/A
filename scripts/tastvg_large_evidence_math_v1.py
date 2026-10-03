"""Pure NumPy rules for the pre-specified cached boundary evidence audit."""
import collections
import numpy as np

EPS=1e-12
SIGNALS=['N','U','S']
GROUPS=['all_changed','large','large_trim_start','large_trim_end','large_expand',
        'large_shift','large_trim_both']

def logsumexp(x,axis):
    m=np.max(x,axis=axis,keepdims=True)
    return np.squeeze(m,axis)+np.log(np.exp(x-m).sum(axis=axis))

def merged_logprior(logits,offset_positions,n):
    result=np.empty((n,2));seen=set()
    for x,pos in zip(logits,offset_positions):
        x=np.asarray(x,float).reshape(-1,2)
        assert len(x)==len(pos) and np.isfinite(x).all()
        result[pos]=x-logsumexp(x,0)-np.log(2.)
        assert not seen.intersection(pos);seen.update(pos)
    assert len(logits)==2 and seen==set(range(n))
    assert np.allclose(np.exp(result).sum(0),1,atol=1e-12,rtol=0)
    return result

def boundary_density(intervals,proposals,bandwidth):
    x=np.asarray(intervals,float);p=np.asarray(proposals,float).reshape(-1,2)
    if not len(p):return None
    assert bandwidth>0 and np.isfinite(p).all() and (p[:,1]>p[:,0]).all()
    d=(x[:,None,:]-p[None,:,:])/bandwidth
    # Normalization is constant across candidates, so omit 1/(h*sqrt(2*pi)).
    return logsumexp(-.5*d*d,1)-np.log(len(p))

def scores(e):
    pos=e['candidate_indices'];lp=np.asarray(e['native_logprior']);a=e['anchor_index']
    n=np.array([lp[i,0]+lp[j,1] for i,j in pos])
    bd=[boundary_density(e['intervals'],e['proposals_'+v],e['bandwidth_normalized']) for v in ['view0','view1']]
    u=[b.sum(1) if b is not None else np.zeros(32) for b in bd]
    s=np.minimum(u[0]-u[0][a],u[1]-u[1][a]) if all(b is not None for b in bd) else np.zeros(32)
    return dict(N=n.tolist(),U=u[0].tolist(),S=s.tolist()),dict(
        view0=u[0].tolist(),view1=u[1].tolist(),
        view0_endpoint_logdensity=bd[0].tolist() if bd[0] is not None else None,
        view1_endpoint_logdensity=bd[1].tolist() if bd[1] is not None else None,
        U_available=bd[0] is not None,S_available=all(b is not None for b in bd))

def choose(values,a,n):
    x=np.asarray(values[:n],float);mx=x.max();top=np.flatnonzero(mx-x<=EPS)
    if mx<=x[a]+EPS or len(top)!=1:return a
    return int(top[0])

def geometry(anchor,interval):
    ds,de=np.asarray(interval,float)-anchor;d=abs(ds)+abs(de)
    b=2*min(abs(ds),abs(de))/d if d>EPS else 0.
    if d<=EPS:kind='same'
    elif b<=.25+EPS:
        if abs(ds)>=abs(de):kind='trim_start' if ds>0 else 'expand'
        else:kind='trim_end' if de<0 else 'expand'
    elif ds<0<de:kind='expand'
    elif de<0<ds:kind='trim_both'
    else:kind='shift'
    r=d/(anchor[1]-anchor[0]);large=r>=.5-EPS
    return dict(ds=float(ds),de=float(de),balance=float(b),radius=float(r),kind=kind,large=bool(large))

def candidate_mask(geo,a,group):
    return [i for i,g in enumerate(geo) if i!=a and g['kind']!='same' and
        (group=='all_changed' or (g['large'] and (group=='large' or group=='large_'+g['kind'])))]

def binary_cell(delta,ev,indices):
    ids=[i for i in indices if abs(delta[i])>EPS]
    if not ids:return None
    y=np.array([delta[i]>EPS for i in ids]);z=np.array([ev[i]>EPS for i in ids]);den=len(ids)
    pos=np.flatnonzero(y);neg=np.flatnonzero(~y)
    auc=None
    if len(pos) and len(neg):
        a=np.array([ev[ids[i]] for i in pos]);b=np.array([ev[ids[i]] for i in neg])
        d=a[:,None]-b[None,:];auc=float(np.mean(np.where(d>EPS,1.,np.where(d< -EPS,0.,.5))))
    return dict(tp=float((y&z).sum()/den),fp=float((~y&z).sum()/den),
        fn=float((y&~z).sum()/den),tn=float((~y&~z).sum()/den),
        positive=float(y.mean()),negative=float((~y).mean()),
        accepted=float(z.mean()),auc=auc,candidates=den,
        positives=int(y.sum()),negatives=int((~y).sum()),
        TP=int((y&z).sum()),FP=int((~y&z).sum()),FN=int((y&~z).sum()),TN=int((~y&~z).sum()))

def matrix(rows,fields):
    sources=sorted({r['source_id'] for r in rows});orders=collections.defaultdict(list);mat=[]
    for src in sources:
        oo=[]
        for order in sorted({r['order'] for r in rows}):
            rr=[r for r in rows if r['source_id']==src and r['order']==order]
            if not rr:continue
            cc=[np.mean([[r[f] for f in fields] for r in rr if r['condition']==c],0)
                for c in sorted({r['condition'] for r in rr})]
            x=np.mean(cc,0);oo.append(x);orders[order].append(x)
        mat.append(np.mean(oo,0))
    return sources,np.array(mat),orders

def aggregate(rows,fields,ratios=None):
    if not rows:return dict(cells=0,sources=0,metrics={},ratios={})
    sources,mat,orders=matrix(rows,fields);rng=np.random.default_rng(20261003)
    boot=np.concatenate([mat[rng.integers(0,len(mat),(100,len(mat)))].mean(1) for _ in range(100)])
    q=np.percentile(boot,[2.5,97.5],0);metrics={};at={f:i for i,f in enumerate(fields)}
    for j,f in enumerate(fields):
        x=mat[:,j];loo=(x.sum()-x)/(len(x)-1) if len(x)>1 else x
        metrics[f]=dict(mean=float(x.mean()),ci95=q[:,j].tolist(),
            source_values={str(s):float(v) for s,v in zip(sources,x)},
            order_values={o:float(np.mean(vals,0)[j]) for o,vals in orders.items()},
            leave_one_out_range=[float(loo.min()),float(loo.max())],
            cell_mean=float(np.mean([r[f] for r in rows])))
    rr={}
    for name,(num,den) in (ratios or {}).items():
        d=boot[:,at[den]];keep=d>EPS;total=mat[:,at[den]].mean()
        rr[name]=dict(mean=float(mat[:,at[num]].mean()/total) if total>EPS else None,
            ci95=np.percentile(boot[keep,at[num]]/d[keep],[2.5,97.5]).tolist() if keep.any() else None,
            bootstrap_zero_denominator_draws=int((~keep).sum()),valid_draws=int(keep.sum()))
    return dict(cells=len(rows),sources=len(sources),metrics=metrics,ratios=rr,
        bootstrap_draws=10000,seed=20261003)
