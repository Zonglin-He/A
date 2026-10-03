"""Matched source-balanced linear delta predictors; no individual safety claim."""
import numpy as np
EPS=1e-12
IQR_EPS=1e-8
RCOND=1e-12
ARMS=['A','L32','Pair-Norm','M0','M1']

def feature(scores, anchor):
    z=np.asarray(scores,dtype=np.float64)
    assert z.shape==(32,) and np.isfinite(z).all()
    ids=np.flatnonzero(z.max()-z<=EPS)
    unique=len(ids)==1
    k=int(ids[0]) if unique else int(anchor)
    iqr=float(np.percentile(z,75)-np.percentile(z,25));den=iqr+IQR_EPS
    a=float((z[anchor]-np.median(z))/den)
    m=float((z[k]-z[anchor])/den)
    return dict(top_index=k,unique_top=unique,eligible=bool(unique and z[k]>z[anchor]+EPS),
                margin=m,anchor_score=a,IQR=iqr,top_context=float((z[k]-np.median(z))/den))

def pseudo_pairs(rows):
    pairs=[];metadata=[]
    for si,r in enumerate(rows):
        f=feature(r['scores'],0)
        meta=dict(source_id=r['source_id'],source_index=si,unique_top=f['unique_top'],top_index=f['top_index'],
                  IQR=f['IQR'],zero_IQR=f['IQR']==0,top_context=f['top_context'])
        if f['unique_top']:
            for j in range(32):
                v=feature(r['scores'],j)
                pairs.append(dict(source_id=r['source_id'],source_index=si,pseudo_anchor=j,top_index=v['top_index'],
                                  margin=v['margin'],anchor_score=v['anchor_score'],top_context=v['top_context'],
                                  eligible=bool(v['eligible']),y=float(r['candidate_t'][v['top_index']]-r['candidate_t'][j]),weight=1/32))
        meta.update(pseudo_anchors=32 if f['unique_top'] else 0,total_weight=1. if f['unique_top'] else 0.)
        metadata.append(meta)
    return pairs,metadata

def design(rows, arm):
    assert arm in ['M0','M1']
    return np.asarray([[1.,r['margin']]+([r['anchor_score']] if arm=='M1' else []) for r in rows],dtype=float)

def fit(rows, arm):
    x=design(rows,arm);y=np.asarray([r['y'] for r in rows]);w=np.asarray([r['weight'] for r in rows])
    assert len(x) and np.isfinite(x).all() and np.isfinite(y).all() and (w>0).all()
    coef,_,rank,singular=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=RCOND)
    return dict(arm=arm,coefficients=coef.tolist(),rank=int(rank),columns=x.shape[1],singular_values=singular.tolist(),
                condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
                source_count=len({r['source_id'] for r in rows}),pairs=len(rows),rcond=RCOND,
                feature_ranges={f:[float(min(r[f] for r in rows)),float(max(r[f] for r in rows))] for f in ['margin','anchor_score']})

def predict(model, f):
    x=[1.,f['margin']]+([f['anchor_score']] if model['arm']=='M1' else [])
    return float(np.dot(x,model['coefficients']))

def choose(scores, anchor, models):
    f=feature(scores,anchor);choices=dict(A=int(anchor),L32=f['top_index']);evidence={}
    for arm in ['M0','M1']:
        mu=predict(models[arm],f)
        outside=any(f[n]<models[arm]['feature_ranges'][n][0] or f[n]>models[arm]['feature_ranges'][n][1]
                    for n in (['margin'] if arm=='M0' else ['margin','anchor_score']))
        accept=f['eligible'] and mu>EPS
        choices[arm]=f['top_index'] if accept else int(anchor)
        evidence[arm]=dict(predicted_delta_t=mu,accepted=bool(accept),outside_marginal_source_ranges=bool(outside),
                           reason='accepted' if accept else ('no_unique_positive_margin' if not f['eligible'] else 'nonpositive_predicted_mean'))
    return dict(**f,choices=choices,evidence=evidence)

def quartiles(rows):
    ordered=sorted(rows,key=lambda r:(r['A8_t'],r['cell_key']))
    return [list(g) for g in np.array_split(np.asarray(ordered,dtype=object),4)]

def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<2 or np.std(x)<=EPS or np.std(y)<=EPS:return None
    return float(np.corrcoef(x,y)[0,1])
