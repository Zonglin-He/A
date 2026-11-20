"""Independent recount from the released per-case table and small-forest formula checks."""
from collections import defaultdict
import csv
import gzip
import numpy as np
from common import ARMS, CFG, HERE, OUT, digest, save
from crossplatform.physical_regions import selection


def check():
    names=ARMS+tuple(CFG['reference_controls'])
    groups=defaultdict(list);unique=set()
    with gzip.open(OUT/'ordinary_rows.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            key=r['case'],r['source']
            assert key not in unique
            unique.add(key)
            groups[r['source']+'_to_'+r['target']].append([float(r[name]) for name in names])
    with (OUT/'metrics.csv').open() as f:
        metrics={(r['direction'],r['arm']):r for r in csv.DictReader(f) if r['subset']=='all'}
    assert len(unique)==18620 and len(groups)==6
    all_metrics=[];all_counts=[]
    for direction,rows in groups.items():
        samples=np.array(rows)
        flags=samples[:,:,None]>[.25,.50]
        num=100*np.column_stack([flags.mean(0),samples.mean(0)])
        wins=(flags & ~flags[:,:1]).sum(0);losses=(~flags & flags[:,:1]).sum(0)
        counts=np.stack([wins,losses],axis=-1)
        def compare(name,values,flips):
            for j,arm in enumerate(names):
                r=metrics[name,arm]
                np.testing.assert_allclose(values[j],[float(r[k]) for k in ('acc25','acc50','miou')],atol=1.e-10,rtol=0)
                np.testing.assert_equal(flips[j],[[int(r['rescue25']),int(r['harm25'])],[int(r['rescue50']),int(r['harm50'])]])
        compare(direction,num,counts);all_metrics.append(num);all_counts.append(counts)
    compare('macro',np.mean(all_metrics,axis=0),np.sum(all_counts,axis=0))
    tests=0;maxerr=0.;rng=np.random.default_rng(8108)
    for n in (0,1,4,11):
        for rep in range(12):
            parent=np.array([rng.choice([-1,*range(i+1,n)]) for i in range(n)],int)
            stop=rng.choice([0.,.37,.5,.99,1.],n);sizes=np.ones(n)
            for i,p in enumerate(parent):
                if p>=0:sizes[p]+=sizes[i]
            x=rng.random((n,3));x/=x.sum(1,keepdims=True);u=np.arange(n,dtype=float)
            view=np.column_stack([u-.1,u,u+.1,np.ones(n),np.ones(n)])
            m,K=selection.structure(parent,stop,sizes)
            for side in (-1,1):
                for words in ((0,0),(0,1)):
                    j,_=selection.shared(parent,K==0,stop,x,words,view,side,0,False,'physical')
                    b=selection.closed_joint(parent,stop,m,x,words,view,K==0,side)
                    diff=float(np.max(abs(j-b),initial=0));assert diff<1e-12
                    maxerr=max(maxerr,diff);tests+=1
    save(OUT/'independent_recount.json',dict(ordinary_direction_records=len(unique),metrics_and_flips_recomputed=len(metrics),
         small_forest_checks=tests,edge_cases='empty, singleton, zero/one stop, same/different class, both horizontal orders',
         maximum_probability_error=maxerr,table_sha256=digest(OUT/'ordinary_rows.csv.gz'),FINAL_access=False))


if __name__=='__main__':check()
