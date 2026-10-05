"""Label-free physical-time acquisition and dual-offset readout.

The endpoint interpolation is a discrete-grid readout, not a calibrated
continuous-time density or independent-view likelihood. No GT enters it.
"""
import numpy as np

SEED = 20261005
EPS = 1e-12

def softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

def tts_positions(ids, extent, scores, k=4):
    ids = np.asarray(ids, float); scores = np.asarray(scores, float)
    a, b = map(int, extent)
    assert len(ids) == len(scores) and np.isfinite(scores).all()
    assert np.all(np.diff(ids) > 0) and 0 <= a <= b < len(ids)
    k = min(int(k), b-a+1)
    if k == 0: return []
    edges = np.linspace(ids[a], ids[b]+1, k+1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = [i for i in range(a, b+1) if lo <= ids[i] < hi]
        if pool: out.append(min(pool, key=lambda i: (-scores[i], i)))
    # Same physical coverage fallback as Uniform4, without exceeding budget.
    while len(out) < k:
        rest = [i for i in range(a, b+1) if i not in out]
        out.append(max(rest, key=lambda i: (min(abs(ids[i]-ids[j]) for j in out), -i)))
    return sorted(out)

def boundary(logits, records, ids):
    ids = np.asarray(ids, float)
    assert len(logits) == len(records) == 2 and len(ids) >= 2
    assert np.all(np.diff(ids) > 0)
    views = []
    for z, r in zip(logits, records):
        x = np.asarray(z, float).reshape(-1, 2); grid = np.asarray(r['frame_ids'], float)
        assert len(x) == len(grid) and np.all(np.diff(grid) > 0)
        p = softmax(x)
        # Interpolate probability values onto existing merged sampled frames;
        # zero outside this offset's support, then normalize discrete masses.
        a = np.stack([np.interp(ids, grid, p[:, j], left=0., right=0.) for j in range(2)], 1)
        a = np.maximum(a, EPS); a /= a.sum(axis=0, keepdims=True)
        views.append(a)
    q = np.sqrt(views[0]*views[1]); q /= q.sum(axis=0, keepdims=True)
    logspan = np.log(q[:, 0])[:, None]+np.log(q[:, 1])[None, :]
    logspan[np.tril_indices(len(ids))] = -np.inf
    start, end = np.unravel_index(int(logspan.argmax()), logspan.shape)
    mean = (views[0]+views[1])/2
    js = .5*sum((a*np.log(a/mean)).sum(axis=0) for a in views)
    return dict(interval=[int(ids[start]), int(ids[end])+1], indices=[int(start),int(end)],
                start=q[:, 0].tolist(), end=q[:, 1].tolist(), views=[a.tolist() for a in views],
                js_start=float(js[0]), js_end=float(js[1]), js_mean=float(js.mean()))

def tube_summary(boxes, ids, interval, tts):
    b = np.asarray(boxes, float); ids = np.asarray(ids, float); tts = np.asarray(tts, float)
    assert b.shape == (len(ids),4) and len(tts)==len(ids)
    speed = np.linalg.norm(np.diff(b[:, :2], axis=0), axis=1)/np.maximum(np.diff(ids), 1)
    span = max(ids[-1]+1-ids[0], 1)
    a,e = interval; inside=(ids>=a)&(ids<e)
    return np.r_[b.mean(0), b.var(0), speed.mean() if len(speed) else 0.,
                 speed.var() if len(speed) else 0., ((a+e)/2-ids[0])/span, (e-a)/span,
                 tts.mean(),tts.std(),tts.min(),tts.max(),tts[inside].mean() if inside.any() else 0.]

def cosine(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/den,-1,1)) if den else 0.

def key_vector(q):
    parts=[]
    for name in ['query','spatial']:
        a=np.asarray(q[name],float); parts.append(a/max(np.linalg.norm(a),EPS))
    parts.append(np.asarray(q['geometry'],float))
    return np.concatenate(parts)

def pair_features(d,r):
    a,b=key_vector(d),key_vector(r)
    return np.r_[a,b,np.abs(a-b),cosine(d['query'],r['query']),cosine(d['spatial'],r['spatial'])]

def ridge_predict(train, y, test, alpha=1.):
    train,test=np.asarray(train,float),np.asarray(test,float); y=np.asarray(y,float)
    assert alpha == 1. and len(train)>0
    mu=train.mean(0); sd=train.std(0); sd=np.where(sd>1e-12,sd,1.)
    x=(train-mu)/sd; t=(test-mu)/sd; ym=y.mean(); w=np.linalg.solve(x@x.T+alpha*np.eye(len(x)),y-ym)
    return ym+(t@x.T)@w
