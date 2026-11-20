"""Independent row recount and analytical checks for reflection and piecewise-constant face inference."""
from collections import Counter,defaultdict
import csv
import gzip
from itertools import product
import numpy as np

from observe import HERE, RESULTS, WORK, digest, read, save
from mirror import Mirror, alter
from predict_profiles import DECISIONS,SIGNATURE
from profiles import ARMS, contrast, project_axis


def interval_trials():
    rng=np.random.default_rng(20261120)
    tests=0
    for mode in ('joint_profile','paired_average','all_average','reverse_edge'):
        for trial in range(24):
            low=rng.uniform(-3,1,(4,4))
            high=low+rng.uniform(.1,3,(4,4))
            w=rng.random((4,4))
            center_length=1.7
            cut=np.unique(np.r_[low.ravel(),high.ravel(),0.,-center_length/2,center_length/2])
            mid=(cut[:-1]+cut[1:])*.5
            h=[]
            for m in mid:
                sums=[sum(w[v,i]/np.sqrt(high[v,i]-low[v,i]) for i in range(4) if low[v,i]<m<high[v,i]) for v in range(4)]
                a,b=(sums[0]+sums[1])/2,(sums[2]+sums[3])/2
                if mode=='paired_average': weight=a
                elif mode=='all_average': weight=(a+b)/2
                else:
                    numerator=b if mode=='reverse_edge' else a
                    weight=2*numerator*numerator/(a+b) if a+b else 0.
                h.append(weight)
            def evaluate(interval):
                l,r=interval
                overlap=np.maximum(0,np.minimum(r,cut[1:])-np.maximum(l,cut[:-1]))
                return overlap@np.array(h)/np.sqrt(r-l)
            candidates=[(a,b) for a in cut if a<=0 for b in cut if b>=0 and b>a]
            candidates.extend(zip(rng.uniform(cut[0],0,128),rng.uniform(0,cut[-1],128)))
            answer=project_axis(low,high,w,center_length,mode)
            assert abs(evaluate(answer)-max(map(evaluate,candidates)))<2.e-12
            tests+=1
    low=np.tile([-1.,-.5,-1.,-.5],(4,1)); high=np.tile([.5,1.,.5,1.],(4,1))
    np.testing.assert_allclose(project_axis(low,high,np.ones((4,4)),1.,'joint_profile'),[-1.,1.])
    return tests


def mirrors():
    rng=np.random.default_rng(137)
    points=rng.normal(size=(32,3))
    for trial in range(32):
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:
            q[:,0]*=-1
        m=np.eye(4)
        m[:3,:3]=q; m[:3,3]=rng.normal(size=3)
        mirror=Mirror.camera(m)
        new=mirror.points(points)
        np.testing.assert_allclose(np.linalg.norm(new-new[0],axis=1),np.linalg.norm(points-points[0],axis=1),atol=1e-13)
        np.testing.assert_allclose(mirror.points(new),points,atol=1.e-13)
        geom=np.column_stack((points,np.exp(rng.normal(size=(32,3))),rng.normal(size=32)))
        recovered=mirror.boxes(mirror.boxes(geom))
        np.testing.assert_allclose(recovered[:,:6],geom[:,:6],atol=1.e-13)
        np.testing.assert_allclose(np.cos(2*recovered[:,6]),np.cos(2*geom[:,6]),atol=1.e-13)
        np.testing.assert_allclose(np.sin(2*recovered[:,6]),np.sin(2*geom[:,6]),atol=1.e-13)
    sentence='the object on the left of the leftmost object'
    p=sentence.index('left')
    right=sentence.index('leftmost')
    spans=[(4,10),(right,len(sentence)),(p,p+4)]
    coherent,known=alter(sentence,spans,{p,right})
    assert coherent[known[-1][0]:known[-1][1]]=='right'
    assert coherent[known[1][0]:known[1][1]]=='rightmost object'
    changed,token=alter(sentence,spans,{p})
    assert changed[token[1][0]:token[1][1]]=='leftmost object'
    return 32


def balanced_rows():
    memory=defaultdict(dict)
    with gzip.open(RESULTS/'all_rows.csv.gz','rt') as fp:
        for row in csv.DictReader(fp):
            key=row['group'],row['source'],row['target'],row['case']
            assert row['arm'] not in memory[key]
            memory[key][row['arm']]=row
    assert len(memory)==501 and all(set(group)==set(ARMS) for group in memory.values())
    tab=list(csv.DictReader((RESULTS/'paired.csv').open()))
    achieved=0
    for row in tab:
        if row['direction']!='macro': continue
        group,arm,control=row['group'],row['arm'],row['control']
        pairs=Counter()
        dirs=defaultdict(lambda:np.zeros(3))
        for (g,s,t,case),arms in memory.items():
            if not (g==group or (group=='natural_all' and g.startswith('natural'))): continue
            a,b=(float(arms[n]['iou']) for n in (arm,control))
            diffs=[]
            for k in (25,50):
                up,down=(a>k/100),(b>k/100)
                pairs[f'rescue{k}']+=up and not down
                pairs[f'harm{k}']+=down and not up
                diffs.append(int(up)-int(down))
            dirs[s,t]+=np.r_[1,diffs]
        assert all(int(row[name])==pairs[name] for name in ('rescue25','harm25','rescue50','harm50'))
        expected=np.mean([v[1:]/v[0] for v in dirs.values()],axis=0)*100
        np.testing.assert_allclose(expected,[float(row['delta_acc25']),float(row['delta_acc50'])],atol=3.e-13)
        achieved+=1
    cov=list(csv.DictReader((RESULTS/'coverage.csv').open()))
    for c in cov:
        assert sum(int(c[k]) for k in ('both','only_subject','only_reference','neither'))==int(c['n'])
    return achieved


def run():
    input_seal=read(SIGNATURE)
    for name,value in input_seal['files'].items():
        assert digest(DECISIONS/name)==value
    stats=dict(decision_sha256=digest(SIGNATURE),decisions=len(input_seal['files']),interval_random_trials=interval_trials(),
               mirror_rigid_involution_trials=mirrors(),panel_control_recounts=balanced_rows(),
               analytic_interval_synthesis=True,outputs_unchanged=True,FINAL_access=False)
    stats['scores_sha256']=digest(RESULTS/'all_rows.csv.gz')
    stats['files']={p.name:digest(p) for p in sorted(RESULTS.iterdir()) if p.name!='audit.json' and p.is_file()}
    save(RESULTS/'audit.json',stats)
    print('Verified',stats['decisions'],'decisions and',stats['panel_control_recounts'],'pooled comparisons',flush=True)


if __name__=='__main__':
    run()

