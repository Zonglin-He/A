"""Frozen-score diagnostic statistics; no training or model imports."""
import collections
import numpy as np

EPS = 1e-12
SEED = 20261003
DRAWS = 10000
FIELDS = ['P_A', 'R_A', 'delta_P', 'delta_R']
SIGNS = {'P_A': -1., 'R_A': -1., 'delta_P': 1., 'delta_R': 1.}
RECIPES = {
    'Inside_Endpoint': ('Inside', 'Endpoint', 'real'),
    'Inside_Context': ('Inside', 'Context', 'real'),
    'Full': ('Full', 'Full', 'real'),
    'Geometry': ('Geometry', 'Geometry', 'real'),
    'Shuffle_Inside_Endpoint': ('Inside', 'Endpoint', 'shuffle'),
    'Shuffle_Inside_Context': ('Inside', 'Context', 'shuffle'),
}

def extract(scores, anchor, winner):
    result = {}
    for name, (pv, rv, control) in RECIPES.items():
        p = np.asarray(scores[f'candidate/{pv}/precision/{control}'], float)
        r = np.asarray(scores[f'candidate/{rv}/recall/{control}'], float)
        assert p.shape == r.shape == (32,) and np.isfinite(p).all() and np.isfinite(r).all()
        result[name] = dict(P_A=float(p[anchor]), R_A=float(r[anchor]),
            P_W=float(p[winner]), R_W=float(r[winner]),
            delta_P=float(p[winner]-p[anchor]), delta_R=float(r[winner]-r[anchor]))
    return result

def label(delta):
    return 'helpful' if delta > EPS else 'harmful' if delta < -EPS else 'neutral'

def ci(values):
    a = np.asarray(values, float); v = a[np.isfinite(a)]
    return dict(ci95=np.quantile(v, [.025, .975]).tolist() if len(v) else None,
                bootstrap_defined=int(len(v)), bootstrap_undefined=int(len(a)-len(v)))

def div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape,b.shape), np.nan), where=b > 0)

def weighted_quantiles(x, w):
    x, w = np.asarray(x,float), np.asarray(w,float)
    idx = np.argsort(x, kind='stable'); x, w = x[idx], w[idx]
    c = np.cumsum(w)/w.sum()
    return [float(x[min(np.searchsorted(c,q,side='left'),len(x)-1)]) for q in [.25,.5,.75]]

def binary(rows, recipe, field, outcome='t', draws=DRAWS, seed=SEED):
    rr = [r for r in rows if r['eligible'] and r[f'label_{outcome}'] != 'neutral']
    ids = sorted({r['source_id'] for r in rr})
    n = len(ids); bins = {s: [r for r in rr if r['source_id']==s] for s in ids}
    h = [[r for r in bins[s] if r[f'label_{outcome}']=='helpful'] for s in ids]
    f = [[r for r in bins[s] if r[f'label_{outcome}']=='harmful'] for s in ids]
    pos = np.array([bool(z) for z in h],float); neg = np.array([bool(z) for z in f],float)
    counts = dict(cells=len(rows),sources=len({r['source_id'] for r in rows}),
        eligible=sum(r['eligible'] for r in rows),noop=sum(not r['eligible'] for r in rows),
        neutral_replacements=sum(r['eligible'] and r[f'label_{outcome}']=='neutral' for r in rows),
        helpful=sum(len(z) for z in h),harmful=sum(len(z) for z in f),
        helpful_sources=int(pos.sum()),harmful_sources=int(neg.sum()),informative_sources=n,
        mixed_sources=int((pos*neg).sum()))
    if not n:
        return dict(counts=counts,orientation=SIGNS[field],auc=None,within_source_auc=None,
                    distributions=None,per_source=[],orders={},draws=draws,seed=seed)
    hp = [np.array([r['readouts'][recipe][field] for r in z],float) for z in h]
    fp = [np.array([r['readouts'][recipe][field] for r in z],float) for z in f]
    mat = np.zeros((n,n)); hm=np.zeros(n); fm=np.zeros(n)
    for i in range(n):
        if len(hp[i]): hm[i]=hp[i].mean()
        if len(fp[i]): fm[i]=fp[i].mean()
        for j in range(n):
            if not len(hp[i]) or not len(fp[j]):continue
            d=SIGNS[field]*(hp[i][:,None]-fp[j][None,:])
            mat[i,j]=np.mean((d>0).astype(float)+.5*(d==0))
    mixed=pos*neg; diag=np.diag(mat)
    rng=np.random.default_rng(seed); w=rng.multinomial(n,np.full(n,1/n),size=draws).astype(float)
    auc_boot=div(np.einsum('bi,ij,bj->b',w,mat,w), (w@pos)*(w@neg))
    local_boot=div(w@diag,w@mixed)
    bh=div(w@hm,w@pos); bf=div(w@fm,w@neg)
    def point_pack(value, samples):
        return dict(mean=float(value) if np.isfinite(value) else None,**ci(samples))
    den=pos.sum()*neg.sum(); value=float(mat.sum()/den) if den else np.nan
    local=float(diag.sum()/mixed.sum()) if mixed.sum() else np.nan
    def quant(arr, flag):
        xx=[];ww=[]
        for z in arr:
            if len(z):xx.extend(z);ww.extend([1/len(z)]*len(z))
        return weighted_quantiles(xx,ww) if flag.sum() else None
    dh=div(hm.sum(),pos.sum()); df=div(fm.sum(),neg.sum())
    auc=point_pack(value,auc_boot); auc['leave_one_source_out']={}
    within=point_pack(local,local_boot);within['leave_one_source_out']={}
    difference=point_pack(dh-df,bh-bf);difference['leave_one_source_out']={}
    for i,s in enumerate(ids):
        keep=np.ones(n);keep[i]=0
        d=(keep@pos)*(keep@neg); numerator=keep@mat@keep
        auc['leave_one_source_out'][str(s)]=float(numerator/d) if d else None
        den0=keep@mixed;within['leave_one_source_out'][str(s)]=float(keep@diag/den0) if den0 else None
        t=div(keep@hm,keep@pos)-div(keep@fm,keep@neg)
        difference['leave_one_source_out'][str(s)]=float(t) if np.isfinite(t) else None
    per=[dict(source_id=s,helpful=len(h[i]),harmful=len(f[i]),helpful_mean=float(hm[i]) if pos[i] else None,
              harmful_mean=float(fm[i]) if neg[i] else None,within_auc=float(diag[i]) if mixed[i] else None) for i,s in enumerate(ids)]
    return dict(counts=counts,orientation=SIGNS[field],auc=auc,within_source_auc=within,
        distributions=dict(helpful=point_pack(dh,bh),harmful=point_pack(df,bf),
            helpful_quantiles=quant(hp,pos),harmful_quantiles=quant(fp,neg),difference=difference),
        per_source=per,orders={},draws=draws,seed=seed)

def source_mean(rows, fields, draws=DRAWS):
    if not rows:return dict(cells=0,sources=0,metrics={})
    bins=collections.defaultdict(list)
    for r in rows:bins[(r['source_id'],r['order'],r['condition'])].append([r[f] for f in fields])
    orders=collections.defaultdict(list); sources=collections.defaultdict(list)
    for (s,o,c),v in bins.items():orders[s,o].append(np.mean(v,axis=0))
    for (s,o),v in orders.items():sources[s].append(np.mean(v,axis=0))
    ids=sorted(sources);mat=np.array([np.mean(sources[s],axis=0) for s in ids]);n=len(ids)
    w=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)/n
    boot=w@mat; vals=mat.mean(0)
    out={f:dict(mean=float(vals[j]),**ci(boot[:,j]),source_values={str(s):float(mat[i,j]) for i,s in enumerate(ids)},
        leave_one_source_out={str(s):float(np.delete(mat,i,axis=0)[:,j].mean()) if n>1 else None for i,s in enumerate(ids)}) for j,f in enumerate(fields)}
    return dict(cells=len(rows),sources=n,metrics=out)

def quadrant(p,r):
    sign=lambda x:'up' if x>EPS else 'down' if x< -EPS else 'tie'
    return sign(p)+'/'+sign(r)

def quadrants(rows,recipe):
    out={}
    for q in ['up/up','up/down','down/up','down/down','up/tie','tie/up','down/tie','tie/down','tie/tie']:
        rr=[r for r in rows if r['eligible'] and quadrant(r['readouts'][recipe]['delta_P'],r['readouts'][recipe]['delta_R'])==q]
        pack=[dict(source_id=r['source_id'],order=r['order'],condition=r['condition'],dt=r['delta_t'],
            dv=r['delta_v'] if r['delta_v'] is not None else 0.,helpful=float(r['label_t']=='helpful'),
            harmful=float(r['label_t']=='harmful'),neutral=float(r['label_t']=='neutral'),
            true_delta_P=r.get('true_delta_P',0.),true_delta_R=r.get('true_delta_R',0.)) for r in rr]
        out[q]=dict(counts=dict(cells=len(rr),helpful=sum(r['label_t']=='helpful' for r in rr),harmful=sum(r['label_t']=='harmful' for r in rr),
            neutral=sum(r['label_t']=='neutral' for r in rr),sources=len({r['source_id'] for r in rr})),
            summary=source_mean(pack,['dt','dv','helpful','harmful','neutral']+(['true_delta_P','true_delta_R'] if rows and rows[0]['domain']=='target' else [])))
    return out

def correlation(rows, x, y, draws=DRAWS):
    if not rows:return dict(mean=None,ci95=None,sources=0)
    ids=sorted({r['source_id'] for r in rows});m=[]
    for s in ids:
        rr=[r for r in rows if r['source_id']==s]
        a=np.array([r[x] for r in rr]);b=np.array([r[y] for r in rr])
        m.append([a.mean(),b.mean(),(a*a).mean(),(b*b).mean(),(a*b).mean()])
    m=np.array(m);n=len(ids)
    def from_moments(a):
        vx=np.maximum(0,a[...,2]-a[...,0]**2);vy=np.maximum(0,a[...,3]-a[...,1]**2)
        return div(a[...,4]-a[...,0]*a[...,1],np.sqrt(vx*vy))
    p=from_moments(m.mean(0));w=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=draws)/n
    boot=from_moments(w@m)
    return dict(mean=float(p) if np.isfinite(p) else None,**ci(boot),sources=n,
        leave_one_source_out={str(s):float(from_moments(np.delete(m,i,axis=0).mean(0))) if n>1 and np.isfinite(from_moments(np.delete(m,i,axis=0).mean(0))) else None for i,s in enumerate(ids)})
