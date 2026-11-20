"""Independent finite-graph reconstruction, inclusion check, and separate count of all task thresholds."""
import csv
import gzip
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from partition_paths import ARMS, GEOMS, OUT, SOURCES, WORK, catalog, digest, read, save, signed
from physical_partition import AXES, cuts, partition


def independent_cut(x,dims):
    n=len(x);ds=cdist(x,x)
    # Distinct distances on Gaussian inputs give independent neighborhoods; grid tie
    # fixture below uses n<=9, where all edges are always present.
    order=np.argsort(ds,axis=1,kind='stable')[:,1:min(9,n)]
    core=np.take_along_axis(ds,np.argsort(ds,axis=1)[:,min(n-1,4):min(n-1,4)+1],axis=1).ravel()
    u=np.repeat(np.arange(n),order.shape[1]);v=order.ravel()
    height=np.maximum.reduce([ds[u,v],core[u],core[v]])
    groups=set(frozenset([i]) for i in range(n))
    for h in np.unique(height):
        take=height<=h
        graph=coo_matrix((np.ones(take.sum()),(u[take],v[take])),shape=(n,n)).tocsr()
        k,ids=connected_components(graph,directed=False)
        groups.update(frozenset(np.flatnonzero(ids==g)) for g in range(k))
    result=[]
    for d in dims:
        candidates=set()
        if np.isfinite(d).all() and (d>0).all():
            for group in groups:
                xyz=x[list(group)];span=np.ptp(xyz[:,:2]@AXES.T,axis=0)
                if xyz[:,2].max()-xyz[:,2].min()<=d[2] and np.any((span<=d[0])&(np.roll(span,-8)<=d[1])):
                    candidates.add(group)
        maximal={g for g in candidates if not any(g<h for h in candidates)}
        if candidates:assert set.union(*(set(v) for v in maximal))==set(range(n)) and sum(map(len,maximal))==n
        result.append({g for g in maximal if len(g)>=8})
    return result


def forests():
    rng=np.random.default_rng(12345);tests=0
    base=np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[2,0,0],[2,1,0],[2,2,0],[1,2,0],[0,2,0]],float)
    for x in [base]+[rng.normal(size=(n,3)) for n in (16,27,40,54) for _ in range(4)]:
        dims=np.array([[.2,.1,.5],[1.7,.8,2],[3.,2.,2.4],[6,3.,3],[20,10,8],[np.nan]*3,[2.5,2,3]])
        expected=independent_cut(x,dims)
        for permutation in [np.arange(len(x)),rng.permutation(len(x))]:
            parent,size,marks,assign=cuts(x[permutation],dims)
            for c in range(7):
                actual={frozenset(permutation[np.flatnonzero(assign[c]==v)]) for v in np.flatnonzero(marks[:,c])}
                assert expected[c]==actual,(c,expected[c],actual)
            tests+=1
    return tests


def real_geometry():
    roster=catalog();s=read(WORK/'SEALED.json')
    frameids=sorted({r['frame'] for r in roster['rows'] if r['active']})
    # Uniform positions in a content-hashed, label-free inventory.
    selected=[frameids[i] for i in np.linspace(0,len(frameids)-1,24,dtype=int)]
    regions=0;instances=0;worst=0.;maxfrac=0.
    for fid in selected:
        f=roster['frames'][fid]
        with np.load(f['lex']) as z,np.load(f['word']) as w:data=partition(z,w,f['f'])
        receipt=WORK/'frame_seals'/(fid+'.json')
        assert digest(receipt)==s['frame_seals'][fid]
        plan=WORK/'plans'/(fid+'.npz');assert digest(plan)==read(receipt)['plan']
        with np.load(plan) as z:
            for k in z.files:np.testing.assert_equal(data[k],z[k])
        for i in range(len(data['parent'])):
            regions+=1
            pts=data['xyz'][(data['assign']==i).any(0)]
            assert len(pts)==data['size'][i]
            for c in np.flatnonzero(data['mark'][i,:6]):
                instances+=1;b=data['envelope'][i,c]
                v=pts-b[:3];co=np.cos(b[6]);si=np.sin(b[6])
                axes=np.column_stack((co*v[:,0]+si*v[:,1],-si*v[:,0]+co*v[:,1],v[:,2]))
                worst=max(worst,float(np.max(np.abs(axes)-b[3:6]/2)))
                spans=np.ptp(axes,axis=0);maxfrac=max(maxfrac,float(np.max(spans/data['template'][c])))
    assert worst<1e-10 and maxfrac<=1+1e-10
    return dict(frames=len(selected),nodes=regions,noun_regions=instances,max_outside_m=worst,max_extent_ratio=maxfrac)


def receipts():
    s=read(WORK/'SEALED.json');assert s['code']==signed() and not s['labels_used'] and not s['FINAL_access']
    for key,expected in s['files'].items():assert digest(WORK/'decisions'/(key+'.npz'))==expected
    for fid,pin in s['frame_seals'].items():
        path=WORK/'frame_seals'/(fid+'.json');assert digest(path)==pin
        p=read(path)['plan']
        if p:assert digest(WORK/'plans'/(fid+'.npz'))==p
    fs=read(WORK/'fresh'/'SEALED.json')
    for k,h in fs['files'].items():assert digest(WORK/'fresh'/(k+'.npz'))==h
    return len(s['files']),len(fs['files'])


def counts():
    counts=defaultdict(Counter);full={}
    for src in SOURCES:
        with gzip.open(OUT/(src+'_task_rows.csv.gz'),'rt') as h:
            for row in csv.DictReader(h):
                if row['panel']!='full':continue
                key=(src,row['target'],row['case'])
                arm=row['arm'];allarms=ARMS if arm=='all_fallback' else (arm,)
                for a in allarms:
                    table=full.setdefault(key,{})
                    assert a not in table;table[a]=[float(row[g]) for g in GEOMS]
    assert len(full)==18620 and all(set(t)==set(ARMS) for t in full.values())
    for (src,tgt,_),pred in full.items():
        for a,values in pred.items():
            for g,v in zip(GEOMS,values):
                c=counts[src,tgt,a,g];c['n']+=1;c['25']+=v>.25;c['50']+=v>.5;c['iou']+=v
    with (OUT/'direction.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['panel']=='full']
    assert len(rows)==6*len(ARMS)*len(GEOMS)
    for r in rows:
        c=counts[r['source'],r['target'],r['arm'],r['geometry']]
        for x,k in [('acc25','25'),('acc50','50'),('miou','iou')]:
            assert abs(float(r[x])-100*c[k]/c['n'])<1e-9
    return len(rows)


if __name__=='__main__':
    n,m=receipts()
    result=dict(forest_reconstructions=forests(),real=real_geometry(),direction_geometry_scores=counts(),
          row_seals=n,fresh_row_seals=m,production=signed(),FINAL_access=False)
    save(OUT/'verification.json',result)
    print(result,flush=True)
