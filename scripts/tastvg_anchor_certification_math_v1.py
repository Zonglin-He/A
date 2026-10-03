"""CPU calibration. Its bootstrap curve is not an individual safety bound."""
import numpy as np

EPS=1e-12
SEED=20261003
DRAWS=10000
QUANTILES=[.5,.2,.1,.05,.025,.01]

def top1(scores,anchor):
    z=np.asarray(scores,dtype=np.float64); assert z.shape==(32,) and np.isfinite(z).all()
    ids=np.flatnonzero(z.max()-z<=EPS)
    return int(ids[0]) if len(ids)==1 and z.max()>z[anchor]+EPS else int(anchor)

def isotonic(x,y,weight=None):
    x=np.asarray(x,float);y=np.asarray(y,float)
    w=np.ones(len(x)) if weight is None else np.asarray(weight,float)
    assert x.shape==y.shape==w.shape and np.isfinite(x).all() and np.isfinite(y).all()
    assert np.isfinite(w).all() and (w>=0).all() and w.sum()>0
    keep=w>0;x,y,w=x[keep],y[keep],w[keep]
    order=np.argsort(x,kind='stable');x,y,w=x[order],y[order],w[order]
    knots,inverse=np.unique(x,return_inverse=True)
    mass=np.bincount(inverse,weights=w);total=np.bincount(inverse,weights=w*y)
    blocks=[]
    for j,(a,b) in enumerate(zip(mass,total)):
        blocks.append([j,j,float(a),float(b)])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            q=blocks.pop();p=blocks.pop();blocks.append([p[0],q[1],p[2]+q[2],p[3]+q[3]])
    fitted=np.empty(len(knots))
    for start,end,a,b in blocks:fitted[start:end+1]=b/a
    return knots,fitted

def fit_calibration(rows):
    eligible=[r for r in rows if r['eligible']]
    if not eligible:return dict(available=False,eligible_sources=0,knots=[],mean=[],probability=[],lower={},thresholds=[])
    x=np.array([r['margin'] for r in eligible]);y=np.array([r['true_delta_t'] for r in eligible]);b=(y>EPS).astype(float)
    knots,mean=isotonic(x,y);pk,prob=isotonic(x,b);assert np.array_equal(knots,pk)
    rng=np.random.default_rng(SEED);ys=np.empty((DRAWS,len(knots)));ps=np.empty_like(ys)
    for i in range(DRAWS):
        w=np.bincount(rng.integers(0,len(x),len(x)),minlength=len(x))
        k,v=isotonic(x,y,w);ys[i]=np.interp(knots,k,v)
        k,v=isotonic(x,b,w);ps[i]=np.interp(knots,k,v)
    lower={str(q):np.minimum(np.quantile(ys,q,axis=0),mean).tolist() for q in QUANTILES}
    return dict(available=True,eligible_sources=len(x),knots=knots.tolist(),mean=mean.tolist(),probability=prob.tolist(),
        probability_ci95=np.quantile(ps,[.025,.975],axis=0).tolist(),lower=lower,
        thresholds=np.unique(np.quantile(x,np.linspace(0,1,21))).tolist(),bootstrap_draws=DRAWS,seed=SEED,
        source_validation_rows=len(rows),domain=[float(knots[0]),float(knots[-1])],
        interpretation='pointwise bootstrap lower curve of isotonic mean delta; no individual/target safety guarantee')

def evaluate(model,margin,q=.05):
    if not model['available']:return dict(in_domain=False,mean=None,probability=None,lower=None)
    in_domain=model['knots'][0]<=margin<=model['knots'][-1]
    if not in_domain:return dict(in_domain=False,mean=None,probability=None,lower=None)
    def ip(v):return float(np.interp(margin,model['knots'],v))
    return dict(in_domain=True,mean=ip(model['mean']),probability=ip(model['probability']),lower=ip(model['lower'][str(q)]))

def decide(scores,anchor,model):
    k=top1(scores,anchor);m=float(scores[k]-scores[anchor]);e=evaluate(model,m)
    if k==anchor:reason='no_unique_positive_margin'
    elif not model['available']:reason='no_source_calibrator'
    elif not e['in_domain']:reason='outside_source_margin_support'
    elif e['lower']<=EPS:reason='nonpositive_lower_mean_delta'
    else:reason='accepted'
    choices=dict(A=anchor,L32=k,Selective=k if reason=='accepted' else anchor)
    for q in QUANTILES:
        z=evaluate(model,m,q);choices['LCB_'+str(q)]=k if k!=anchor and z['in_domain'] and z['lower']>EPS else anchor
    for j,t in enumerate(model['thresholds']):choices['Margin_'+str(j)]=k if k!=anchor and m>=t else anchor
    return dict(top_index=k,margin=m,calibration=e,reason=reason,choices=choices)

def source_matrix(rows,fields):
    sources=sorted({r['source_id'] for r in rows});matrix=[];orders={}
    for s in sources:
        rr=[r for r in rows if r['source_id']==s];oo=[]
        for o in sorted({r['order'] for r in rr}):
            ro=[r for r in rr if r['order']==o];cc=[]
            for c in sorted({r['condition'] for r in ro}):cc.append(np.mean([[r[f] for f in fields] for r in ro if r['condition']==c],axis=0))
            z=np.mean(cc,axis=0);oo.append(z);orders.setdefault(o,[]).append(z)
        matrix.append(np.mean(oo,axis=0))
    return sources,np.asarray(matrix),orders

def summarize(rows,fields,ratios=None):
    if not rows:return dict(cells=0,sources=0,metrics={},ratios={})
    sources,mat,orders=source_matrix(rows,fields);rng=np.random.default_rng(SEED)
    boot=mat[rng.integers(0,len(mat),(DRAWS,len(mat)))].mean(1);q=np.quantile(boot,[.025,.975],axis=0)
    metrics={};at={f:j for j,f in enumerate(fields)}
    for j,f in enumerate(fields):
        v=mat[:,j];loo=(v.sum()-v)/(len(v)-1) if len(v)>1 else v
        metrics[f]=dict(mean=float(v.mean()),ci95=q[:,j].tolist(),source_values={str(s):float(z) for s,z in zip(sources,v)},
            order_values={o:float(np.mean(z,axis=0)[j]) for o,z in orders.items()},leave_one_out_range=[float(loo.min()),float(loo.max())])
    out={}
    for name,(num,den) in (ratios or {}).items():
        n,d=boot[:,at[num]],boot[:,at[den]];valid=d>EPS;point=mat[:,at[den]].mean()
        out[name]=dict(mean=float(mat[:,at[num]].mean()/point) if point>EPS else None,
            ci95=np.quantile(n[valid]/d[valid],[.025,.975]).tolist() if valid.any() else None,
            valid_draws=int(valid.sum()),bootstrap_zero_denominator_draws=int((~valid).sum()))
    return dict(cells=len(rows),sources=len(mat),metrics=metrics,ratios=out,bootstrap_draws=DRAWS,seed=SEED)
