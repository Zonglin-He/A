"""Exercise decision boundaries, then recount paired metrics using the public row table."""
from collections import defaultdict
import csv
import gzip
import math

import numpy as np

from predict_direct import ARMS, OUT, PRIVATE, digest, read, regions, save, selection, tree


def boundaries():
    rng = np.random.default_rng(24061)
    trials = 0
    for parents in ([],[-1],[2,2,-1],[1,2,3,-1],[-1]*12,[6,6,7,7,8,8,9,9,10,10,-1,-1]):
        parent = np.asarray(parents,int)
        n = len(parent)
        size = np.ones(n,int)
        for i,p in enumerate(parent):
            if p>=0:
                size[p]+=size[i]
        for _ in range(16):
            stop = rng.choice([0.,1.,.25,.5,1.-1e-12,1e-14],n)
            logits = rng.random((n,2))
            logits[rng.random((n,2))<.2] = 0.
            center = rng.integers(-4,5,n)
            view = np.stack([center-.5,center,center+.5,np.ones(n),size],axis=1)
            view[rng.random(n)<.2] = np.nan
            occupancy, overlap = selection.structure(parent,stop,size)
            for s,r in ((0,0),(0,1)):
                for side in (-1,1):
                    left,right = view[:,0],view[:,2]
                    valid = (right[:,None]<left) if side<0 else (left[:,None]>right)
                    pair = (occupancy*logits[:,s])[:,None]*(occupancy*logits[:,r])*(overlap==0)*valid
                    mass = tree.SideEvidence(parent,stop,view,shared_cut=False).marginal(logits[:,s],logits[:,r],side)
                    np.testing.assert_array_equal(mass>0,pair.sum(1)>0)
                    np.testing.assert_allclose(mass,pair.sum(1),rtol=1e-12,atol=1e-300)
                    trials+=1
    parent = np.full(4,-1)
    data = dict(parent=parent,stop=np.ones(4),size=np.ones(4,int),p=np.full((4,2),.5),
                view=np.array([[x,x,x,1,1] for x in (0,0,1,1)],float),assign=np.arange(4)[None,:])
    empty = dict(t_box=np.empty((0,7)),t_p=np.empty(0),t_project=np.empty((0,8)))
    points = regions.Returns(np.column_stack([np.arange(4.),np.zeros((4,2))]))
    assert selection.select(data,empty,(0,0),-1,0,False,points)==(0,2)
    assert selection.select(data,empty,(0,0),1,0,False,points)==(2,0)
    data['view'][:,:3]=0.
    assert selection.select(data,empty,(0,1),-1,0,False,points)==(-1,-1)
    return trials


def recount():
    rows = []
    with gzip.open(OUT/'ordinary_rows.csv.gz','rt') as stream:
        for r in csv.DictReader(stream):
            baseline = float(r[ARMS[0]])
            active = int(r['active'])
            for arm in ARMS:
                value = float(r[arm])
                assert 0<=value<=1 and (active or value==baseline)
            rows.append(r)
    assert len({(r['source'],r['case']) for r in rows})==len(rows)==18620
    groups = defaultdict(list)
    for r in rows:
        direction = r['source']+'_to_'+r['target']
        for scope in ('all','region' if int(r['active']) else 'fallback'):
            groups[scope,direction].append(r)
    def compare(r,expected):
        for key,value in expected.items():
            assert math.isclose(float(r[key]),value,abs_tol=1e-10), (r['direction'],r['arm'],key,value,r[key])
    with (OUT/'metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    count_fields=['rescue25','harm25','rescue50','harm50','changed_subject_box','improved_iou','worsened_iou']
    rate_fields=['acc25','acc50','miou','delta25_pp','delta50_pp','delta_miou_pp']
    by_scope = defaultdict(list)
    for m in metrics:
        if m['direction']=='macro':
            continue
        pool = groups[m['scope'],m['direction']]
        arm = m['arm']
        x = [float(r[arm]) for r in pool]
        y = [float(r[ARMS[0]]) for r in pool]
        n = len(pool)
        sums = dict(changed_subject_box=sum(int(r[arm+'_box_changed']) for r in pool),
                    improved_iou=sum(a>b+1e-8 for a,b in zip(x,y)),worsened_iou=sum(a<b-1e-8 for a,b in zip(x,y)))
        vals = dict(miou=100*math.fsum(x)/n,delta_miou_pp=100*math.fsum(a-b for a,b in zip(x,y))/n)
        for suffix,t in [('25',.25),('50',.50)]:
            gains = sum(a>t>=b for a,b in zip(x,y))
            losses = sum(b>t>=a for a,b in zip(x,y))
            sums['rescue'+suffix],sums['harm'+suffix]=gains,losses
            vals['acc'+suffix]=100*sum(a>t for a in x)/n
            vals['delta'+suffix+'_pp']=100*(gains-losses)/n
        expected=dict(n=n,**sums,**vals)
        compare(m,expected)
        by_scope[m['scope'],arm].append(expected)
    for m in metrics:
        if m['direction']=='macro':
            part=by_scope[m['scope'],m['arm']]
            assert len(part)==6
            combined={k:sum(r[k] for r in part) for k in ['n',*count_fields]}
            combined.update({k:math.fsum(r[k] for r in part)/6 for k in rate_fields})
            compare(m,combined)
    assert len(metrics)==3*7*4
    return len(metrics)


if __name__=='__main__':
    before=read(PRIVATE/'SEAL.json')
    assert digest(PRIVATE/'decisions.npz')==before['decisions']
    comparisons=boundaries()
    n=recount()
    save(OUT/'audit.json',dict(forest_comparisons=comparisons,same_noun_ties_both_directions=True,
         strict_touching_exclusion=True,empty_votes_neutral=True,rows_independently_recounted=18620,
         summary_rows_recounted=n,sealed_box_sha256=before['decisions'],
         all_nonactive_ious_invariant=True,FINAL_access=False))
    print('Verified',comparisons,'boundary comparisons and',n,'metric rows')
