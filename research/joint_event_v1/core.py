"""Whole-interval semantic feedback. No annotations or visual-model execution."""
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import rankdata


def relative_event(premax, event_groups, noun_groups, selected):
    """Target-track event ownership relative to nominal identity ownership.

    Each word uses the existing trained object-token logits, normalized over
    the same15 tracked queries. No arbitrary cross-model feature cosine.
    """
    x=np.asarray(premax,dtype=np.float64)
    def owned(groups):
        values=[]
        for group in groups:
            z=x[:,:,group].mean(-1)
            values.append(z[:,selected]-logsumexp(z,axis=1))
        return np.stack(values,axis=1)
    y=owned(event_groups)-owned(noun_groups).mean(1,keepdims=True)
    return y


def hidden_features(offset, ids):
    h=offset['hidden']
    if hasattr(h,'numpy'):h=h.numpy()
    h=np.asarray(h,np.float64)[:,0]
    h/=np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-30)
    h=h.transpose(1,0,2).reshape(h.shape[1],-1)/np.sqrt(h.shape[0])
    x=np.asarray(offset['frame_ids'],float)+.5
    return np.stack([np.interp(np.asarray(ids)+.5,x,h[:,j])for j in range(h.shape[1])],axis=1)


def ridge_predict(x, y, train, positions):
    """One disposable per-query linear projection; fixed ridge1.

    Label centering/scaling uses training anchors ONLY in each leave-one fold.
    Unlabelled dense feature geometry may use the original full cache.
    """
    a=x[np.asarray(positions)][train];b=np.asarray(y)[train]
    xm=a.mean(0);ym=b.mean(0);ys=np.maximum(b.std(0),1e-6)
    scale=max(float(np.sqrt(np.square(a-xm).sum()/len(a))),1e-12)
    xc=(a-xm)/scale;yc=(b-ym)/ys
    coeff=np.linalg.solve(xc@xc.T+np.eye(len(a)),yc)
    prediction=((x-xm)/scale)@xc.T@coeff
    return prediction,ym,ys


def projections(offsets, ids, positions, labels):
    models=[];skills=[]
    for offset in offsets:
        x=hidden_features(offset,ids);pred,_,_=ridge_predict(x,labels,np.arange(8),positions)
        views=[pred];errors=np.zeros(labels.shape[1]);null=np.zeros_like(errors)
        for j in range(8):
            train=np.arange(8)!=j;v,mu,sd=ridge_predict(x,labels,train,positions)
            actual=(labels[j]-mu)/sd
            errors+=np.square(v[positions[j]]-actual);null+=np.square(actual)
            views.append(v)
        skills.append(dict(total_gain=float((null-errors).sum()),per_predicate_gain=(null-errors).tolist(),null_error=float(null.sum()),model_error=float(errors.sum())))
        models.append(views)
    return models,skills


def physical_membership(ids, pairs):
    centers=np.asarray(ids,float)+.5
    edges=np.r_[float(ids[0]),(centers[:-1]+centers[1:])/2,float(ids[-1])+1]
    width=np.diff(edges)
    a=np.asarray(ids)[pairs[:,0]];b=np.asarray(ids)[pairs[:,1]]+1
    overlap=np.maximum(0,np.minimum(b[:,None],edges[None,1:])-np.maximum(a[:,None],edges[None,:-1]))
    return overlap,width,(b-a).astype(float)


def calibrated_masks(models, width, occupancy):
    """Match native expected occupancy, not GT duration, for every predicate."""
    result=[]
    for view in models:
        variants=[]
        for z in view:
            lo=np.full(z.shape[1],-40.);hi=np.full_like(lo,40.)
            for _ in range(60):
                mid=(lo+hi)/2;p=expit(z+mid)
                below=(width[:,None]*p).sum(0)/width.sum()<occupancy
                lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
            variants.append(expit(z+(lo+hi)/2))
        result.append(variants)
    return result


def interval_scores(mask, overlap, width, lengths):
    """Soft event IoU: both boundaries determine coverage AND background spill.

    This ratio of expected intersection/union is not expected true GT IoU.
    Mean log over predicates requires joint support for the described parts;
    parts can occupy different moments, with no invented textual time order.
    """
    intersection=overlap@mask;total=(width[:,None]*mask).sum(0)
    soft_iou=intersection/np.maximum(lengths[:,None]+total[None]-intersection,1e-30)
    logs=np.log(np.maximum(soft_iou,1e-30))
    return logs.mean(1),logs


def independent_scores(mask,pairs):
    v=mask.mean(1);start=np.r_[0.,np.diff(v)];end=np.r_[-np.diff(v),0.]
    def rank(x):
        return (rankdata(x)-.5)/len(x)-.5 if np.ptp(x)>1e-14 else np.zeros_like(x)
    return rank(start)[pairs[:,0]]+rank(end)[pairs[:,1]]


def guard(proposal_index, native_index, scores_per_view, combined_logs, skills):
    gains=np.array([v[proposal_index]-v[native_index]for v in scores_per_view])
    predicate=combined_logs[proposal_index]-combined_logs[native_index]
    predictive=all(s['total_gain']>0 for s in skills)
    agree=bool(np.min(gains)>1e-12 and np.min(predicate)>=-1e-12)
    accept=bool(proposal_index!=native_index and predictive and agree)
    return accept,dict(accept=accept,predictive=bool(predictive),agreement=agree,
                      minimum_view_gain=float(np.min(gains)),minimum_predicate_gain=float(np.min(predicate)))
