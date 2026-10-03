"""Pure CPU equations: predecessor ridge unchanged, deterministic source folds."""
import hashlib
import numpy as np
from scripts.tastvg_pr_role_math_v1 import (
    EPS, SEED, DRAWS, ROLES, source_weights, fit_ridge, predict, interval_pr,
    analytic_readout, pack, regression, decision,
)
POPS=['all','role']
LEVELS=['n4','n8','n12','nmax']
REGIMES=['in_sample']+LEVELS
VARIANTS=[p+'/'+r for p in POPS for r in REGIMES]

def source_pack_allowed(stage,path,mode,allowed_paths):
    if stage=='extract':return (isinstance(mode,str) and 'w' in mode and '+' not in mode) or path in allowed_paths
    return stage=='fit' and path in allowed_paths

def folds(dataset,sources):
    ids=sorted(set(map(int,sources)));assert len(ids)>12
    out=[]
    for held in ids:
        rest=sorted((s for s in ids if s!=held),key=lambda s:(
            hashlib.sha256(f'tastvg_pr_loso_v1|{dataset}|{held}|{s}'.encode()).hexdigest(),s))
        for level,n in zip(LEVELS,[4,8,12,len(rest)]):
            out.append(dict(dataset=dataset,held_source=held,level=level,
                train_sources=rest[:n],train_source_count=n,
                fold_key=f'{dataset}/{held:03d}/{level}'))
    return out

def population_indices(population,anchor,winner):
    assert population in POPS and 0<=anchor<32 and 0<=winner<32
    return np.arange(32) if population=='all' else np.array([anchor,winner])

def subset_arrays(packs,population):
    xx=[];ys={'P':[],'R':[]};groups=[]
    for source in sorted(packs):
        z=packs[source]
        for i in range(len(z['x'])):
            ix=population_indices(population,int(z['anchors'][i]),int(z['winners'][i]))
            xx.append(z['x'][i][ix])
            for n in ys:ys[n].append(z[n][i][ix])
            groups.extend([source]*len(ix))
    return np.vstack(xx),{n:np.concatenate(v) for n,v in ys.items()},np.array(groups)

def paired(a,b,ba,bb):
    return {k:pack(b['metrics'][k]['mean']-a['metrics'][k]['mean']
        if a['metrics'][k]['mean'] is not None and b['metrics'][k]['mean'] is not None else np.nan,
        bb[k]-ba[k]) for k in a['metrics']}
