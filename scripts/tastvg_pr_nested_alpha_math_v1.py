"""Same ridge; nested source holdout and prelocked A/W MAE selection."""
import numpy as np
from scripts.tastvg_pr_loso_math_v1 import (
    EPS, SEED, DRAWS, ROLES, POPS, source_weights, predict, interval_pr,
    analytic_readout, pack, regression, decision, subset_arrays, paired,
)
from scripts.tastvg_pr_accessibility_math_v1 import hierarchy
GRID=[.001,.01,.1,1.,10.,100.,1000.]
VARIANTS=[p+'/'+r for p in POPS for r in ['fixed','nested']]
PARTS={'P':slice(512,768),'R':slice(0,512)}

def nested_folds(dataset,sources):
    ids=sorted(set(map(int,sources)))
    return [dict(dataset=dataset,held_source=s,train_sources=[i for i in ids if i!=s],
        inner_sources=[i for i in ids if i!=s],fold_key=f'{dataset}/{s:03d}') for s in ids]

def select_alpha(scores):
    a=np.asarray(scores,float);assert a.shape==(len(GRID),) and np.isfinite(a).all()
    return GRID[int(np.argmin(a))]

def ridge_grid(x,y,groups,alphas=GRID):
    x=np.asarray(x,float);y=np.asarray(y,float);w=source_weights(groups)
    mean=w@x;std=np.sqrt(w@((x-mean)**2));std=np.where(std<1e-8,1.,std)
    a=(x-mean)/std;bias=float(w@y)
    cov=a.T@(w[:,None]*a);rhs=a.T@(w*(y-bias))
    ev,vec=np.linalg.eigh(cov);ev=np.maximum(ev,0.)
    coef=vec@((vec.T@rhs)[:,None]/(ev[:,None]+np.array(alphas)[None,:]))
    return [dict(mean=mean,std=std,weight=coef[:,j],bias=bias,alpha=float(t))
        for j,t in enumerate(alphas)]

def validation_errors(rows,truth,prediction):
    assert np.shape(truth)==np.shape(prediction)==(len(rows),2)
    errors=np.abs(np.asarray(truth)-prediction)
    ids,m=hierarchy(rows,errors);assert len(ids)==1
    result=dict(A=float(m[0,0]),W=float(m[0,1]),mean=float(m.mean()))
    for label in ['clean','corrupt']:
        ix=[i for i,r in enumerate(rows) if (r['condition']=='clean')==(label=='clean')]
        _,z=hierarchy([rows[i] for i in ix],errors[ix]);result[label]=float(z.mean())
    return result
