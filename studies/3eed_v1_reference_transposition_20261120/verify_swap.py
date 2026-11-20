"""Independent checks for an involutive scene edit, zero interaction, and strict-IoU interval coverage."""
from collections import Counter
import argparse
import csv
import gzip
import numpy as np
from scipy.special import expit

from registry_swap import WORK, OBS, OUT, PRED, ROOT, key, digest, read, save
from posterior_swap import ARMS, agreement
from physical_swap import Exchange
from center_consensus import best_center, crossing_intervals
from batch_yaw_iou import many_iou


def check_math():
    a=np.array([0.,0.,0.,3.,2.,1.,0.])
    bs=np.tile(a,(3,1));bs[:,0]=[0,1,2]
    w=np.array([.02,.76,.22])
    ints=crossing_intervals(a,bs)
    idx,lo,hi,_,_=ints
    np.testing.assert_allclose(lo,[-1.,0.,1.],atol=3e-8)
    np.testing.assert_allclose(hi,[1.,2.,3.],atol=3e-8)
    np.testing.assert_array_equal(idx,np.arange(3))
    b,_=best_center(a,ints,w)
    assert 1<b[0]<2 and abs(w[many_iou(bs,b)>.5].sum()-.98)<1e-12
    # Disjoint supervision that only touches at its endpoints cannot earn both weights.
    bs[1,0]=2
    b,_=best_center(a,crossing_intervals(a,bs[:2]),np.ones(2))
    np.testing.assert_array_equal(b,a)
    rng=np.random.default_rng(542781)
    trials=24; probes=0; max_regret=0.
    for _ in range(trials):
        anchor=np.r_[rng.normal(size=3),rng.uniform(1.,4.,3),rng.uniform(-np.pi,np.pi)]
        boxes=np.tile(anchor,(21,1))
        boxes[:,:3]+=rng.normal(0,.36,(21,3))
        boxes[:,3:6]*=np.exp(rng.normal(0,.18,(21,3)))
        boxes[:,6]+=rng.normal(0,.35,21)
        w=rng.uniform(size=len(boxes));ints=crossing_intervals(anchor,boxes)
        ids,left,right,direction,_=ints
        final,_=best_center(anchor,ints,w)
        got=float(w[many_iou(boxes,final)>.5].sum())
        for t in np.r_[np.linspace(-3.,3.,85),0.]:
            direct=anchor.copy();direct[:2]+=t*direction
            hits=many_iou(boxes,direct)>.5
            bounded=np.zeros(len(boxes),bool);bounded[ids]=(left<t)&(t<right)
            np.testing.assert_array_equal(bounded,hits)
            max_regret=max(max_regret,w[hits].sum()-got);probes+=len(boxes)
    assert max_regret<1e-9
    h=rng.uniform(size=(48,2));g=rng.uniform(size=48);ok=rng.uniform(size=(48,2))>.3
    np.testing.assert_allclose(agreement(g,h,ok),agreement(1-g,1-h,ok),atol=2e-15)
    np.testing.assert_allclose(agreement(g,np.full_like(h,.5),ok),1.)
    np.testing.assert_allclose(agreement(np.full_like(g,.5),h,ok),1.)
    shape=rng.normal(size=(48,1)); scene=rng.normal(size=(48,1)); text=rng.normal(size=(48,1))
    log=(shape+scene+text)-(shape+scene)-(shape+text)+shape
    np.testing.assert_allclose(agreement(expit(log[:,0]),h,ok),1.,atol=1e-14)
    edit=Exchange(np.array([[0.,0.,0.],[3.,2.,1.]]),np.array([.5,.5,.5]))
    pts=np.vstack([rng.normal(0,.1,(25,3)),rng.normal(0,.1,(25,3))+edit.centers[1],[[7,1,2]]])
    np.testing.assert_allclose(edit.move(edit.move(pts)),pts,atol=1e-14)
    return dict(independent_iou_probes=probes,random_fixtures=trials,grid_objective_regret=max_regret,
                exact_interval_fixture=True,coincident_endpoint_control=True,exchange_involution=True,
                zero_interaction_and_simultaneous_sign_reversal=True)


def audit():
    recorded=read(OUT/'audit.json')
    assert recorded['predictions']==digest(PRED/'SEALED.json')
    pins=read(OBS/'SEALED.json')
    rows=read(WORK/'roster.json')
    groups=Counter()
    for rec in rows:
        name=key(rec)
        with np.load(OBS/name) as z,np.load(PRED/name) as pr:
            valid=pr['applied']
            np.testing.assert_array_equal(pr['boxes'][0],z['slate'][int(z['before'])])
            if not valid:
                np.testing.assert_array_equal(pr['boxes'],np.tile(pr['boxes'][0],(len(ARMS),1)))
                continue
            np.testing.assert_array_equal(pr['raw'][-4:],np.tile(pr['raw'][-4],(4,1)))
            np.testing.assert_array_equal(pr['refs'][-4:],np.tile(pr['refs'][-4],(4,1)))
            np.testing.assert_allclose(pr['boxes'][-2:,2:],np.tile(pr['raw'][-1,2:],(2,1)),rtol=0,atol=0)
            assert z['inverse_error']<5e-5 and np.min(z['moved_points'])>=4
            groups[rec['panel']]+=1
    with gzip.open(OUT/'all_rows.csv.gz','rt') as f:
        scores=list(csv.DictReader(f))
    assert len(scores)==len(rows)*len(ARMS)==recorded['rows'] and len(rows)==pins['n']
    # Independently recreate every full-denominator direction and paired threshold counter.
    index={}
    for row in scores:
        k=(row['group'],row['source'],row['target'],row['case'])
        index.setdefault(k,{})[row['arm']]=float(row['iou'])
    assert len(index)==len(rows) and all(set(d)==set(ARMS) for d in index.values())
    with (OUT/'paired.csv').open() as f:
        tables=list(csv.DictReader(f))
    count=0
    for r in tables:
        a,c=r['arm'],r['control']
        if a not in ('crossed_interval','crossed_raw'):
            continue
        parts={}
        for (group,s,t,_),v in index.items():
            if group==r['group'] and r['direction'] in ('macro',s+'_to_'+t):
                parts.setdefault((s,t),[]).append((v[a],v[c]))
        assert sum(map(len,parts.values()))==int(r['n'])
        for cut in (25,50):
            deltas=[];pos=neg=0
            for records in parts.values():
                vals=np.array(records)>cut/100
                x,y=vals.T
                pos+=int((x&~y).sum());neg+=int((y&~x).sum())
                deltas.append(100*(x.mean()-y.mean()))
            assert pos==int(r['rescues'+str(cut)]) and neg==int(r['harms'+str(cut)])
            assert abs(np.mean(deltas)-float(r['delta'+str(cut)]))<1e-10
            count+=1
    result=check_math()
    result.update(checked_direction_and_macro_counters=count,records=len(rows),active=dict(groups),
                  retained_every_fallback=True,source_manifest=digest(ROOT/'crossplatform/single_frame_sources.json'),
                  prediction_seal=recorded['predictions'],FINAL_access=False)
    save(OUT/'verification.json',result)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--math-only',action='store_true')
    if ap.parse_args().math_only:
        print(check_math())
    else:
        audit()
