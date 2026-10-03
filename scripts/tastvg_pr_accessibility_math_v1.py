"""Bounded CPU readout audit math. No model/provider/file imports."""
from collections import defaultdict
import numpy as np

EPS=1e-12
SEED=20261003
DRAWS=10000
FITS=['source_fit','target_search_fit']
LADDERS=['predicted','GT_anchor','GT_winner','GT_precision','GT_recall','GT_all']
ROLES=['P_A','R_A','P_W','R_W']

def source_weights(groups):
    g=np.asarray(groups);u,n=np.unique(g,return_counts=True)
    lookup=dict(zip(u.tolist(),n.tolist()))
    w=np.array([1/lookup[s] for s in g.tolist()],float)
    return w/w.sum()

def fit_ridge(x,y,groups,alpha):
    x=np.asarray(x,float);y=np.asarray(y,float);w=source_weights(groups)
    mean=w@x;std=np.sqrt(w@((x-mean)**2));std=np.where(std<1e-8,1.,std)
    a=(x-mean)/std;bias=float(w@y)
    cov=a.T@(w[:,None]*a);rhs=a.T@(w*(y-bias))
    ev,vec=np.linalg.eigh(cov);ev=np.maximum(ev,0.)
    coef=vec@((vec.T@rhs)/(ev+alpha))
    return dict(mean=mean,std=std,weight=coef,bias=bias,alpha=float(alpha),model='ridge',intercept=True)

def predict(model,x):
    return (np.asarray(x,float)-model['mean'])/model['std']@model['weight']+model['bias']

def interval_pr(intervals,span):
    z=np.asarray(intervals,float);s,e=span
    assert e>s and np.all(z[:,1]>z[:,0])
    inter=np.maximum(0.,np.minimum(z[:,1],e)-np.maximum(z[:,0],s))
    p=inter/(z[:,1]-z[:,0]);r=inter/(e-s)
    t=inter/(z[:,1]-z[:,0]+e-s-inter)
    return p,r,t

def analytic_t(p,r):
    p=np.clip(np.asarray(p,float),0.,1.);r=np.clip(np.asarray(r,float),0.,1.)
    den=p+r-p*r
    return np.divide(p*r,den,out=np.zeros(np.broadcast_shapes(p.shape,r.shape)),where=den>0)

def ladder(pred,true,mode):
    z=dict(pred)
    changed={'predicted':[],'GT_anchor':['P_A','R_A'],'GT_winner':['P_W','R_W'],
             'GT_precision':['P_A','P_W'],'GT_recall':['R_A','R_W'],'GT_all':ROLES}[mode]
    for k in changed:z[k]=true[k]
    a=float(analytic_t(z['P_A'],z['R_A']));w=float(analytic_t(z['P_W'],z['R_W']))
    return dict(T_A=a,T_W=w,delta_T=w-a)

def ci(a):
    a=np.asarray(a,float);valid=np.isfinite(a)
    return dict(ci95=np.quantile(a[valid],[.025,.975]).tolist() if valid.any() else None,
                bootstrap_defined=int(valid.sum()),bootstrap_undefined=int((~valid).sum()))

def pack(value,samples):
    return dict(mean=float(value) if np.isfinite(value) else None,**ci(samples))

def hierarchy(rows,values):
    """Per-cell moment arrays -> equal condition/order/source totals."""
    cells=defaultdict(list);orders=defaultdict(list);sources=defaultdict(list)
    for r,v in zip(rows,values):cells[r['source_id'],r['order'],r['condition']].append(v)
    for (s,o,c),vv in cells.items():orders[s,o].append(np.mean(vv,axis=0))
    for (s,o),vv in orders.items():sources[s].append(np.mean(vv,axis=0))
    ids=sorted(sources)
    return ids,np.array([np.mean(sources[s],axis=0) for s in ids])

def regression_values(a):
    a=np.asarray(a,float);vy=np.maximum(0.,a[...,2]-a[...,0]**2);vp=np.maximum(0.,a[...,3]-a[...,1]**2)
    rho=np.divide(a[...,4]-a[...,0]*a[...,1],np.sqrt(vy*vp),
        out=np.full(a.shape[:-1],np.nan),where=(vy>EPS)&(vp>EPS))
    r2=np.divide(a[...,5],vy,out=np.full(a.shape[:-1],np.nan),where=vy>EPS)*(-1)+1
    return dict(r2=r2,mae=a[...,6],mse=a[...,5],rho=rho,truth_mean=a[...,0],prediction_mean=a[...,1],
                below_zero=a[...,7],above_one=a[...,8])

def regression_moments(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    assert y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all()
    return np.array([y.mean(),p.mean(),np.mean(y*y),np.mean(p*p),np.mean(y*p),
        np.mean((y-p)**2),np.mean(abs(y-p)),np.mean(p<0),np.mean(p>1)])

def regression(rows,ys,ps,draws=DRAWS):
    ids,m=hierarchy(rows,[regression_moments(y,p) for y,p in zip(ys,ps)]);n=len(ids)
    weights=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)/n
    val=regression_values(m.mean(0));boot=regression_values(weights@m)
    result={k:pack(val[k],boot[k]) for k in val}
    leave={}
    for i,s in enumerate(ids):
        a=regression_values(np.delete(m,i,axis=0).mean(0)) if n>1 else {}
        leave[str(s)]={k:float(v) if np.isfinite(v) else None for k,v in a.items()}
    return dict(cells=len(rows),sources=n,metrics=result,source_moments={str(s):m[i].tolist() for i,s in enumerate(ids)},
                delete_one_source=leave,draws=draws,seed=SEED),boot

def decision(rows,scores,draws=DRAWS):
    pairs=[(r,float(x)) for r,x in zip(rows,scores) if r['eligible'] and abs(r['delta_t'])>EPS]
    ids=sorted({r['source_id'] for r,x in pairs});n=len(ids)
    counts=dict(cells=len(rows),sources=len({r['source_id'] for r in rows}),eligible=sum(r['eligible'] for r in rows),
        noop=sum(not r['eligible'] for r in rows),neutral=sum(r['eligible'] and abs(r['delta_t'])<=EPS for r in rows),
        helpful=sum(r['delta_t']>EPS for r,x in pairs),harmful=sum(r['delta_t']< -EPS for r,x in pairs),
        informative_sources=n,accepted=sum(r['eligible'] and x>0 for r,x in zip(rows,scores)),
        accepted_helpful=sum(r['delta_t']>EPS and x>0 for r,x in pairs),
        accepted_harmful=sum(r['delta_t']< -EPS and x>0 for r,x in pairs),
        accepted_severe_v_harm=sum(r['eligible'] and x>0 and r['delta_v']< -.05 for r,x in zip(rows,scores)))
    bins={s:[(r,x) for r,x in pairs if r['source_id']==s] for s in ids}
    h=[np.array([x for r,x in bins[s] if r['delta_t']>EPS]) for s in ids]
    f=[np.array([x for r,x in bins[s] if r['delta_t']< -EPS]) for s in ids]
    if n:
        pos=np.array([bool(len(x)) for x in h],float);neg=np.array([bool(len(x)) for x in f],float)
        mat=np.zeros((n,n));tp=np.zeros(n);fp=np.zeros(n);acc=np.zeros(n);ap=np.zeros(n);at=np.zeros(n)
        for i,s in enumerate(ids):
            if len(h[i]):tp[i]=np.mean(h[i]>0)
            if len(f[i]):fp[i]=np.mean(f[i]>0)
            v=bins[s];acc[i]=np.mean([(x>0)==(r['delta_t']>EPS) for r,x in v])
            ap[i]=np.mean([x>0 and r['delta_t']>EPS for r,x in v]);at[i]=np.mean([x>0 for r,x in v])
            for j in range(n):
                if len(h[i]) and len(f[j]):
                    d=h[i][:,None]-f[j][None,:];mat[i,j]=np.mean((d>0)+.5*(d==0))
        bs=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws).astype(float)
        def vals(weights):
            weights=np.asarray(weights,float);ph=weights@pos;nf=weights@neg
            div=lambda a,b:np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=b>0)
            tpr=div(weights@tp,ph);fpr=div(weights@fp,nf)
            return dict(auc=div(np.einsum('...i,ij,...j->...',weights,mat,weights),ph*nf),
                within_source_auc=div(weights@np.diag(mat),weights@(pos*neg)),tpr=tpr,fpr=fpr,
                balanced_accuracy=.5*(tpr+1-fpr),accuracy=div(weights@acc,weights.sum(-1)),
                accepted_precision=div(weights@ap,weights@at))
        point=vals(np.ones(n));samples=vals(bs)
        metrics={k:pack(point[k],samples[k]) for k in point};leave={}
        for i,s in enumerate(ids):
            w=np.ones(n);w[i]=0;a=vals(w)
            leave[str(s)]={k:float(v) if np.isfinite(v) else None for k,v in a.items()}
        counts.update(helpful_sources=int(pos.sum()),harmful_sources=int(neg.sum()),mixed_sources=int((pos*neg).sum()))
    else:metrics={};samples={};leave={}
    utility=[]
    for r,x in zip(rows,scores):
        take=r['eligible'] and x>0
        utility.append(np.array([r['delta_t'] if take else 0.,r['delta_v'] if take else 0.,
            max(r['delta_v'],0.) if take else 0.,min(r['delta_v'],0.) if take else 0.,float(take)]))
    uid,u=hierarchy(rows,utility);un=len(uid)
    ub=np.random.default_rng(SEED).multinomial(un,np.full(un,1/un),size=draws)/un
    pointu=u.mean(0);bootu=ub@u
    utilities={k:pack(pointu[j],bootu[:,j]) for j,k in enumerate(['delta_t','delta_v','gross_v_gain','gross_v_loss','accept_rate'])}
    return dict(counts=counts,metrics=metrics,delete_one_source=leave,utility=utilities,
        utility_source_values={str(s):u[i].tolist() for i,s in enumerate(uid)},draws=draws,seed=SEED),samples
