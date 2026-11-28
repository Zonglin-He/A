"""Verify an exact deferred-ray-score rewrite for the two-candidate ranking."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import cv2
import numpy as np

from ablate import PAIR_EDITS, REGISTRATION, RUN, digest, edited, publish, sealed_inputs


def decide(scores, resolve):
    pca, qwen = (0. if v is None else v for v in scores)
    if pca==qwen:
        return resolve()
    return int(qwen>pca)


def predict(frame, replies):
    import axiswise_2d as trial

    bypass = edited(trial.official.prepare_pair,PAIR_EDITS['R_default_pca'])
    with patch.object(trial.official,'prepare_pair',bypass):
        ctx = trial.prepare(frame,replies)
    if ctx is None:
        return trial.fallback(frame,replies), None
    boxes = tuple(trial.fit(ctx,b,trial.witnesses(ctx,b),height_only=False) for b in ctx.boxes)
    scores = trial.rank(ctx,boxes)[1]
    calls = 0

    def resolve():
        nonlocal calls
        raw = np.asarray(replies['3d']['value'],float)
        upright,_ = trial.geometry.to_sensor(raw,ctx.transform)
        cap = max(.1,.5*min(upright[3:5]))
        costs = [trial.official.ray_score(b,ctx.selection,ctx.transform,cap) for b in ctx.boxes]
        calls = len(costs)
        return int(costs[1]<costs[0]-1.e-12*len(ctx.selection)*cap)

    winner = decide(scores,resolve)
    return trial.output(ctx,boxes,winner), calls


def guarded_rank_test():
    n = 0
    for pca,qwen,ray in product((None,0.,.2,.8),(None,0.,.2,.8),(0,1)):
        scores = pca,qwen
        old,alt = scores[ray],scores[1-ray]
        expected = 1-ray if alt is not None and alt>(old if old is not None else 0.) else ray
        visited=[]
        def resolve():
            visited.append(True)
            return ray
        assert decide(scores,resolve)==expected
        assert len(visited)==int((pca or 0.)==(qwen or 0.))
        n+=1
    return n


def verify(task):
    row,ref = task
    result,calls = predict(row['frame'],row['raw'])
    assert result==ref['outputs']['axis_2d'],row['frame']['uid']
    return calls


def main():
    cv2.setNumThreads(1)
    source,_,refs,pins,current = sealed_inputs()
    tested = guarded_rank_test()
    with ProcessPoolExecutor(10) as pool:
        calls=list(pool.map(verify,((r,refs[u]) for u,r in source.items()),chunksize=4))
    active=sum(c is not None for c in calls)
    assert active==372 and len(calls)==873
    arms = [('formal',current),('lazy',lambda f,r:predict(f,r)[0])]
    cases=[]
    for domain,platforms in (('3eed',{'waymo','drone','quad'}),('kitti',{'kitti'}),('mono',{'mono'})):
        for flag,count in ((True,12),(False,4)):
            pool=[u for u in source if source[u]['target'] in platforms and refs[u]['active']==flag]
            pool.sort(key=lambda u:sha256(('extent-timing-20261128|'+u).encode()).hexdigest())
            assert len(pool)>=count
            cases.extend((domain,flag,u) for u in pool[:count])
    data=[]
    for i,(domain,flag,uid) in enumerate(cases):
        row=source[uid]; args=row['frame'],row['raw']; truth=refs[uid]['outputs']['axis_2d']
        for _,fn in arms:
            assert fn(*args)==truth
        for t in range(3):
            order=arms if (i+t)%2==0 else arms[::-1]
            costs={}
            for name,fn in order:
                begin=perf_counter_ns()
                result=fn(*args)
                costs[name]=(perf_counter_ns()-begin)/1_000_000
                assert result==truth
            data.append(dict(domain=domain,C_ray=flag,ms=costs))
    timing={}
    for name,subset in (('all',data),('C_ray',[r for r in data if r['C_ray']])):
        x,y=([r['ms'][a] for r in subset] for a,_ in arms)
        d=np.asarray(y)-np.asarray(x)
        timing[name]=dict(n=len(d),formal_median_ms=float(np.median(x)),lazy_median_ms=float(np.median(y)),
                          paired_median_ms=float(np.median(d)),paired_p90_ms=float(np.quantile(d,.90)))
    publish(RUN/'lazy-ray-summary.json',dict(source=pins,source_code=digest(Path(__file__)),labels_read=False,
        toy_ranking_cases=tested,exact_replay=len(calls),active=active,ray_default_cases=sum(bool(c) for c in calls),
        formal_ray_cost_calls=2*active,lazy_ray_cost_calls=sum(c or 0 for c in calls),timing=timing))
    print('Exact',len(calls),'ray defaults',sum(bool(c) for c in calls),'timing',timing,flush=True)


if __name__=='__main__':
    main()
