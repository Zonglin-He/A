"""Identify pair-specific shared preference with two training captions; serve with one."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DUEL = HERE.parent/'3eed_v1_pretruncation_physical_duels_20261120'
sys.path.insert(0,str(DUEL))
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from common import PREV, anon, archive_npz, atomic, digest, read, setup
from fit_duels import change_count, loaded, match_scale
from fit import write_csv

WORK = HERE/'run_PRIVATE'
RESULTS = HERE/'results'
ARMS = tuple(read(HERE/'PROTOCOL.json')['comparators'])


def initialize():
    setup()
    WORK.mkdir(mode=0o700,exist_ok=True)
    RESULTS.mkdir(exist_ok=True)


def ridge(x,y):
    # With unit-RMS active coordinates, mean(X^T X) on its diagonal is n.
    return np.linalg.solve(x.T@x + len(x)*np.eye(x.shape[1]), x.T@y)


def contrasts(h,j,slots):
    idx = np.arange(len(j))[:,None]
    caption = np.arange(2)[None,:]
    a,b = slots.T
    return (h[idx,caption,a[:,None]]-h[idx,caption,b[:,None]],
            j[idx,caption,a[:,None]]-j[idx,caption,b[:,None]])


def fit_model(features, scores, slots, perm):
    dx,ds = contrasts(features.astype(float),scores.astype(float),slots)
    n,_,p = dx.shape
    scale = np.sqrt(np.mean(dx**2,axis=(0,1)))
    scale[scale<1e-10] = np.inf
    standardized = dx/scale
    x = standardized.reshape(-1,p)
    bias = ds.mean(1)
    intrinsic = (ds[:,0]-ds[:,1])/2
    target = np.broadcast_to(bias[:,None],ds.shape)
    shuffled = (ds[:,0]+ds[perm,1])/2
    mixed = np.empty_like(ds)
    mixed[:,0] = shuffled
    mixed[perm,1] = shuffled
    signs = np.array([1.,-1.])
    uniform = ds - intrinsic.mean()*signs

    def logistic(beta):
        logit = (ds-(standardized@beta))*signs
        loss = np.logaddexp(0.,-logit).mean() + .5*(beta@beta)
        grad = x.T@(expit(-logit)*signs).ravel()/len(x) + beta
        return loss,grad

    fit = minimize(logistic,np.zeros(p),jac=True,method='L-BFGS-B',
        options={'ftol':1e-14,'gtol':1e-9,'maxiter':200})
    assert fit.success and np.abs(fit.jac).max()<2e-6
    onehot = np.eye(4)[slots[:,0]]-np.eye(4)[slots[:,1]]
    scale_rank = np.sqrt(np.mean(onehot*onehot,axis=0))
    scale_rank[scale_rank<1e-10] = np.inf
    ranks = ridge(onehot/scale_rank,bias)/scale_rank
    heads = np.stack([np.zeros(p),ridge(x,target.ravel())/scale,fit.x/scale,
                      ridge(x,uniform.ravel())/scale,ridge(x,mixed.ravel())/scale,np.zeros(p)])
    return dict(weights=heads,rank=ranks), dict(fit_pairs=n,mean_intrinsic=float(intrinsic.mean()),
        negative_or_zero_intrinsic=int((intrinsic<=0).sum()),finite_features=int(np.isfinite(scale).sum()),
        max_logistic_gradient=float(np.abs(fit.jac).max()))


def predict(h,j,model,scales=None):
    base = np.asarray(j,dtype=float)
    out={}
    for i,a in enumerate(ARMS):
        correction=(model['rank'] if a=='rank_only' else np.asarray(h,dtype=float)@model['weights'][i])
        out[a]=base-correction
    if scales is not None:
        for a in ARMS[1:5]:
            out[a+'_matched']=base+scales[a]*(out[a]-base)
    return out


def fit_and_lock():
    initialize()
    seal = read(DUEL/'run_PRIVATE/SOURCE_SEAL.json')
    fitfile = DUEL/'run_PRIVATE/fit.npz'
    assert digest(fitfile)==seal['splits']['fit']['hash']
    d=loaded(fitfile)
    valid=d['available'].all(1)
    perm=loaded(DUEL/'run_PRIVATE/heads.npz')['permutation']
    assert digest(DUEL/'run_PRIVATE/heads.npz')==read(DUEL/'run_PRIVATE/HEADS_LOCK.json')['heads']
    model,diag=fit_model(d['h'][valid],d['j'][valid],d['slots'][valid],perm)
    archive_npz(WORK/'model.npz',**model)
    calfile = DUEL/'run_PRIVATE/native_cal.npz'
    assert digest(calfile)==seal['native_cal']
    native=loaded(calfile)
    pred=predict(native['h'],native['j'],model)
    counts={a:change_count(pred['v1'],s,1.) for a,s in pred.items()}
    budget=min(counts[a] for a in ARMS[1:5])
    scales={a:match_scale(pred['v1'],pred[a],budget) for a in ARMS[1:5]}
    actual={a:change_count(pred['v1'],pred[a],v) for a,v in scales.items()}
    atomic(WORK/'LOCK.json',dict(model=digest(WORK/'model.npz'),code=digest(HERE/'bias.py'),
        config=digest(HERE/'PROTOCOL.json'),scales=scales,source=digest(DUEL/'run_PRIVATE/SOURCE_SEAL.json'),
        target_labels_used=False,new_probe_used=False,final_access=False))
    atomic(RESULTS/'fit.json',dict(**diag,source_calibration_rows=len(native['j']),native_switches=counts,
                                    switch_budget=budget,matched_switches=actual,scales=scales))


if __name__=='__main__':
    fit_and_lock()
