"""Paired, recording-clustered statistics for the new readouts."""
from collections import defaultdict

import numpy as np


def cluster_weights(rows, draws=2000):
    rng=np.random.default_rng(20261128)
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[r['target']].append(i)
    weights=np.zeros((draws,len(rows)),dtype=float)
    for ids in groups.values():
        records=sorted({rows[i]['recording'] for i in ids})
        mapping={r:i for i,r in enumerate(records)}
        draw=rng.multinomial(len(records),np.full(len(records),1/len(records)),size=draws)
        weights[:,ids]=draw[:,[mapping[rows[i]['recording']] for i in ids]]
    return weights


def intervals(vals, weights):
    vals=np.array(vals)
    if not len(vals):
        return None
    draws=weights@vals/weights.sum(axis=1)
    return np.quantile(draws,[.025,.975]).tolist()


def contrast(x,y,weights,threshold=.5):
    arr=np.asarray(x)-np.asarray(y)
    s=np.asarray(x)>=threshold
    b=np.asarray(y)>=threshold
    delta=s.astype(float)-b.astype(float)
    return dict(miou_delta=100*float(arr.mean()),miou_ci95=intervals(100*arr,weights),
                acc50_delta=100*float(delta.mean()),acc50_ci95=intervals(100*delta,weights),
                rescue=int((s&~b).sum()),harm=int((~s&b).sum()))


def box_panel(rows, comparisons):
    if not rows:
        return dict(n=0)
    w=cluster_weights(rows)
    arms={arm:np.array([r['iou'][arm] for r in rows]) for arm in rows[0]['iou']}
    result={k:dict(miou=100*float(x.mean()),acc25=100*float(np.mean(x>=.25)),acc50=100*float(np.mean(x>=.5)))
            for k,x in arms.items()}
    return dict(n=len(rows),recordings=len({(r['target'],r['recording']) for r in rows}),
                metrics=result,comparisons={k:contrast(arms[a],arms[b],w) for k,(a,b) in comparisons.items()})
