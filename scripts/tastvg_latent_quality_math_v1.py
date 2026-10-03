"""Pure CPU math for a source-supervised, frozen temporal linear probe."""
import numpy as np

ALPHAS = [.001, .01, .1, 1., 10., 100., 1000.]
WINDOW_SECONDS = 1.
EPS = 1e-12


def interval_features(hidden, ids, pairs, fps):
    h = np.asarray(hidden, np.float64); f = np.asarray(ids, np.int64)
    assert h.shape == (len(f), 256) and np.isfinite(h).all()
    assert np.all(np.diff(f) > 0) and fps > 0
    out, contexts = [], []
    for i, j in pairs:
        assert 0 <= i < j < len(f)
        s, e = f[i], f[j] + 1
        left = (f < s) & (f >= s-WINDOW_SECONDS*fps)
        right = (f >= e) & (f < e+WINDOW_SECONDS*fps)
        ml = h[left].mean(0) if left.any() else np.zeros(256)
        mr = h[right].mean(0) if right.any() else np.zeros(256)
        out.append(np.concatenate([h[i], h[j], h[i:j+1].mean(0), ml, mr,
                                   h[i]-ml, h[j]-mr]))
        contexts.append(dict(left_frames=int(left.sum()), right_frames=int(right.sum()),
                             inside_frames=j-i+1))
    return np.stack(out), contexts


def geometry_features(ids, pairs):
    f = np.asarray(ids, np.float64); width = f[-1]+1-f[0]
    xy = np.array([[(f[i]-f[0])/width, (f[j]+1-f[0])/width] for i,j in pairs])
    return np.column_stack([xy, xy[:,1]-xy[:,0]])


def temporal_iou(intervals, gt):
    a = np.asarray(intervals, np.float64)
    inter = np.maximum(0., np.minimum(a[:,1],gt[1])-np.maximum(a[:,0],gt[0]))
    return inter/(a[:,1]-a[:,0]+gt[1]-gt[0]-inter)


def fit_path(x, y, vx, vy, val_sources, val_top1=True):
    """Regularized least squares; scaling and fitting use training rows only."""
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    mean=x.mean(0);std=x.std(0);std=np.where(std<1e-8,1.,std)
    xx=(x-mean)/std;ym=float(y.mean()); n=len(y)
    cov=xx.T@xx/n; rhs=xx.T@(y-ym)/n
    eigen,vec=np.linalg.eigh(cov); eigen=np.maximum(eigen,0.)
    proj=vec.T@rhs;vxx=(np.asarray(vx)-mean)/std
    groups=np.asarray(val_sources);scores=[];models=[]
    for alpha in ALPHAS:
        w=vec@(proj/(eigen+alpha));pred=vxx@w+ym
        selected=[]
        for src in dict.fromkeys(groups.tolist()):
            ids=np.flatnonzero(groups==src);selected.append(float(vy[ids[int(np.argmax(pred[ids]))]]))
        tr=xx@w+ym;res=cov@w+alpha*w-rhs
        model=dict(mean=mean,std=std,weight=w,bias=ym,alpha=alpha)
        scores.append(dict(alpha=alpha,validation_top1_tIoU=float(np.mean(selected)),
            validation_MSE=float(np.mean((pred-vy)**2)),training_MSE=float(np.mean((tr-y)**2)),
            normal_equation_max_error=float(np.max(np.abs(res))),
            prediction_min=float(pred.min()),prediction_max=float(pred.max())))
        models.append(model)
    best=min(range(len(ALPHAS)), key=lambda i:(-scores[i]['validation_top1_tIoU'],
                                              scores[i]['validation_MSE'], ALPHAS[i]))
    return models[best],scores,best


def predict(model, x):
    return (np.asarray(x,np.float64)-model['mean'])/model['std']@model['weight']+model['bias']


def pick(scores, anchor, n):
    z=np.asarray(scores[:n]);top=np.flatnonzero(z.max()-z<=EPS)
    return anchor if len(top)!=1 or z.max()<=z[anchor]+EPS else int(top[0])
