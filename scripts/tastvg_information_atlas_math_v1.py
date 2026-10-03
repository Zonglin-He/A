"""CPU-only linear-accessibility math; no model or data-provider imports."""
import hashlib
import warnings
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, average_precision_score

ALPHAS = [.001, .01, .1, 1., 10., 100., 1000.]
FRAME_TASKS = ['position', 'event', 'start_distance', 'end_distance', 'phase']
CANDIDATE_TASKS = ['precision', 'recall', 'tiou', 'delta']
VIEWS = {'Endpoint': np.r_[0:512], 'Inside': np.r_[512:768],
         'Context': np.r_[768:1280], 'Contrast': np.r_[1280:1792],
         'Full': np.r_[0:1792], 'Geometry': np.r_[0:3]}
MOMENT_NAMES = ['y', 'y2', 'prediction', 'mse', 'mae', 'within_r2',
                'auc', 'ap', 'prevalence', 'logloss']


def frame_labels(ids, gt):
    ids = np.asarray(ids, float); s, e = gt
    assert e > s and np.all(np.diff(ids) > 0)
    width = ids[-1] + 1 - ids[0]
    pos = (ids - ids[0]) / width
    event = (ids >= s) & (ids < e)
    phase = np.where(event, (ids-s)/(e-s), np.nan)
    return dict(position=pos, event=event.astype(float),
                start_distance=(ids-s)/width, end_distance=(e-ids)/width,
                phase=phase)


def candidate_labels(intervals, gt, anchor):
    a = np.asarray(intervals, float); s, e = gt
    assert e > s and np.all(a[:, 1] > a[:, 0])
    inter = np.maximum(0., np.minimum(a[:, 1], e)-np.maximum(a[:, 0], s))
    p = inter/(a[:, 1]-a[:, 0]); r = inter/(e-s)
    t = inter/(a[:, 1]-a[:, 0]+e-s-inter)
    return dict(precision=p, recall=r, tiou=t, delta=t-t[anchor])


def equal_source_weights(groups):
    g = np.asarray(groups); unique, counts = np.unique(g, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist()))
    w = np.array([1/lookup[z] for z in g.tolist()], float)
    return w/w.sum()


def shuffle_labels(labels, source, family, mask=None, anchor=None):
    seed = int(hashlib.sha256(f'20261003|{source}|{family}'.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed); y = np.asarray(labels).copy()
    indices = np.arange(len(y)) if mask is None else np.flatnonzero(mask)
    if anchor is not None: indices = indices[indices != anchor]
    perm = rng.permutation(indices); y[indices] = y[perm]
    return y, indices, perm


def normalization(x, weights, intercept=True):
    x = np.asarray(x, float); weights = np.asarray(weights, float)
    center = weights@x
    std = np.sqrt(weights@((x-center)**2))
    std = np.where(std < 1e-8, 1., std)
    return (center if intercept else np.zeros(x.shape[1])), std


def ridge_path(x, y, groups, vx, vy, vgroups, intercept=True):
    """Return all ridge models. Selection uses source-validation MSE only."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if y.ndim == 1: y = y[:, None]
    if vy.ndim == 1: vy = vy[:, None]
    w = equal_source_weights(groups); vw = equal_source_weights(vgroups)
    mean, std = normalization(x, w, intercept)
    a = (x-mean)/std; va = (vx-mean)/std
    bias = w@y if intercept else np.zeros(y.shape[1])
    cov = a.T@(w[:, None]*a); rhs = a.T@(w[:, None]*(y-bias))
    ev, vec = np.linalg.eigh(cov); ev = np.maximum(ev, 0.); proj = vec.T@rhs
    models = []; rows = []
    for alpha in ALPHAS:
        coef = vec@(proj/(ev[:, None]+alpha)); pred = va@coef+bias
        tr = a@coef+bias
        models.append(dict(mean=mean, std=std, weight=coef, bias=bias, alpha=alpha,
                           intercept=intercept, model='ridge'))
        rows.append(dict(alpha=alpha, validation_MSE=(vw@((pred-vy)**2)).tolist(),
                         training_MSE=(w@((tr-y)**2)).tolist(),
                         equation_error=np.max(abs(cov@coef+alpha*coef-rhs),axis=0).tolist()))
    selected=[min(range(len(ALPHAS)), key=lambda i:(rows[i]['validation_MSE'][j], ALPHAS[i]))
              for j in range(y.shape[1])]
    return models, rows, selected


def logistic_path(x, y, groups, vx, vy, vgroups):
    w = equal_source_weights(groups); vw = equal_source_weights(vgroups)
    mean, std = normalization(x, w); a = (x-mean)/std; va = (vx-mean)/std
    assert set(np.unique(y)) == {0., 1.}
    models=[]; rows=[]
    for alpha in ALPHAS:
        # sklearn's loss is a weighted mean; the regularizer is 1/(C*sum_weight).
        fit=LogisticRegression(C=1/alpha, solver='lbfgs', max_iter=1000, tol=1e-8,
                               fit_intercept=True)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always',ConvergenceWarning)
            fit.fit(a, y, sample_weight=w)
        bad=[str(z.message) for z in captured if issubclass(z.category,ConvergenceWarning)]
        assert not bad, ('logistic_nonconvergence',alpha,bad)
        coef=fit.coef_[0];bias=float(fit.intercept_[0]);z=a@coef+bias;vz=va@coef+bias
        loss=np.logaddexp(0,vz)-vy*vz
        residual=a.T@(w*(expit(z)-y))+alpha*coef
        intercept_error=float(abs(w@(expit(z)-y)))
        assert max(np.max(abs(residual)),intercept_error)<1e-5,('logistic_KKT',alpha)
        models.append(dict(mean=mean,std=std,weight=coef,bias=bias,alpha=alpha,
                           model='logistic',intercept=True))
        rows.append(dict(alpha=alpha,validation_logloss=float(vw@loss),
            training_logloss=float(w@(np.logaddexp(0,z)-y*z)),
            KKT_error=float(np.max(abs(residual))),intercept_error=intercept_error,
            iterations=int(fit.n_iter_[0]),warnings=bad))
    selected=min(range(len(ALPHAS)),key=lambda i:(rows[i]['validation_logloss'],ALPHAS[i]))
    return models,rows,selected


def predict(model,x):
    z=(np.asarray(x)-model['mean'])/model['std']@model['weight']+model['bias']
    return expit(z) if model['model']=='logistic' else z


def moments(y,pred,event=False,logits=None):
    y=np.asarray(y,float);p=np.asarray(pred,float)
    if not len(y):return None
    assert y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all()
    err=p-y;var=float(np.var(y));v=dict(y=float(y.mean()),y2=float((y*y).mean()),
        prediction=float(p.mean()),mse=float((err*err).mean()),mae=float(abs(err).mean()),
        within_r2=float(1-(err*err).mean()/var) if var>1e-12 else None,
        auc=None,ap=None,prevalence=None,logloss=None,n=len(y))
    if event:
        assert (p>=0).all() and (p<=1).all()
        if logits is not None:
            z=np.asarray(logits,float);v['logloss']=float((np.logaddexp(0,z)-y*z).mean())
        else:
            clipped=np.clip(p,1e-15,1-1e-15)
            v['logloss']=float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean())
        v['prevalence']=float(y.mean())
        if len(np.unique(y))==2:
            v['auc']=float(roc_auc_score(y,p));v['ap']=float(average_precision_score(y,p))
    return v


def source_moments(rows,metric):
    """Condition -> order -> source; never weight longer clips as more sources."""
    import collections
    co=collections.defaultdict(list)
    for r in rows:
        if r['metrics'].get(metric) is not None:
            m=r['metrics'][metric]
            co[(r['source_index'],r['order'],r['condition'])].append(
                [np.nan if m[k] is None else m[k] for k in MOMENT_NAMES])
    def mean(a):
        a=np.asarray(a,float);good=np.isfinite(a);s=np.nansum(a,axis=0);n=good.sum(0)
        return np.divide(s,n,out=np.full(s.shape,np.nan),where=n>0)
    oc=collections.defaultdict(list);sc=collections.defaultdict(list)
    for (s,o,c),a in co.items():oc[s,o].append(mean(a))
    for (s,o),a in oc.items():sc[s].append(mean(a))
    return {s:mean(a) for s,a in sorted(sc.items())}


def metric_from_moments(a,field):
    a=np.asarray(a,float)
    if field=='r2':
        den=a[...,1]-a[...,0]**2
        return np.divide(a[...,3],den,out=np.full(den.shape,np.nan),where=den>1e-12)*(-1)+1
    return a[...,MOMENT_NAMES.index(field)]


def bootstrap_summary(rows,metric,draws=10000,seed=20261003):
    src=source_moments(rows,metric);ids=list(src)
    if not ids:return dict(sources=0,metrics={})
    a=np.array(list(src.values()));rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=draws)/len(ids)
    good=np.isfinite(a);den=weights@good
    b=np.divide(weights@np.nan_to_num(a),den,out=np.full(den.shape,np.nan),where=den>0)
    nn=good.sum(0);one=np.divide(np.nansum(a,axis=0),nn,out=np.full(nn.shape,np.nan),where=nn>0)
    ans={}
    for f in ['r2','mse','mae','within_r2','auc','ap','prevalence','logloss']:
        value=float(metric_from_moments(one,f));vals=metric_from_moments(b,f);valid=np.isfinite(vals)
        ans[f]=dict(mean=value if np.isfinite(value) else None,
            ci95=np.quantile(vals[valid],[.025,.975]).tolist() if valid.any() else None,
            bootstrap_defined=int(valid.sum()),sources_defined=int(np.isfinite(metric_from_moments(a,f)).sum()))
    return dict(sources=len(ids),source_moments={str(k):[None if not np.isfinite(z) else float(z) for z in v] for k,v in src.items()},
                metrics=ans,draws=draws,seed=seed)


def paired_difference(rows,left,right,field,draws=10000,seed=20261003):
    aa=source_moments(rows,left);bb=source_moments(rows,right);ids=sorted(set(aa)&set(bb))
    if not ids:return dict(sources=0,mean=None,ci95=None)
    a=np.array([aa[i] for i in ids]);b=np.array([bb[i] for i in ids]);rng=np.random.default_rng(seed)
    w=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=draws)/len(ids)
    def ave(v,weights):
        mask=np.isfinite(v);den=weights@mask
        return np.divide(weights@np.nan_to_num(v),den,out=np.full(den.shape,np.nan),where=den>0)
    bs=metric_from_moments(ave(a,w),field)-metric_from_moments(ave(b,w),field)
    point=float(metric_from_moments(ave(a,np.ones((1,len(ids)))/len(ids)),field)[0]-
                metric_from_moments(ave(b,np.ones((1,len(ids)))/len(ids)),field)[0])
    finite=np.isfinite(bs)
    return dict(sources=len(ids),mean=point if np.isfinite(point) else None,
                ci95=np.quantile(bs[finite],[.025,.975]).tolist() if finite.any() else None,
                bootstrap_defined=int(finite.sum()))
