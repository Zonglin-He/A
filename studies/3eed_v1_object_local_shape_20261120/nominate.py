"""Allocate local observations from each expression's sealed whole-scene role evidence."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np
from local_paths import *
from role_cut import structure,shared
from physical_nodes import assemble,overlap_prior
from physical_evidence import Returns
from scene_partition import build as partition

def cover(mass,K,visible,budget=8):
    residual=np.ones(len(mass))
    selected=[]
    for _ in range(budget):
        gain=np.maximum(K-(1.-residual),0.)@mass
        gain[~visible]=0
        if selected:gain[selected]=0.
        i=int(np.argmax(gain)) if len(gain) else -1
        if i<0 or gain[i]<=0:break
        selected.append(i)
        residual=np.minimum(residual,1.-K[i])
    return selected

def job(task):
    fid,info,rows,fp=task
    assert digest(info['lex'])==info['lh'] and digest(info['word'])==info['wh']
    if info['tree']:
        path,pin=info['tree'];assert digest(path)==pin
        with np.load(path) as z:data=dict(z)
    else:
        with np.load(info['lex']) as l,np.load(info['word']) as word:data=assemble(partition(l['sample'],l['logits'],l['xyz'],info['f']),word)
    points=Returns(data['xyz']);visible=np.isfinite(data['view'][:,0])
    m,K=structure(data['parent'],data['stop'],data['size'])
    oldfield=previous.WORK/'fields'/(fid+'.npz');assert digest(oldfield)==fp
    with np.load(oldfield) as f:np.testing.assert_equal(f['tree'],data['p'])
    ids=[];subjects=[];refs=[];noms=[];nominations=set()
    for r in rows:
        assert r['active'] and digest(r['role'])==r['rh']
        with np.load(r['role']) as z:
            us=overlap_prior(data,z,'t',points);ur=overlap_prior(data,z,'r',points)
            if len(K):
                joint,rf=shared(data['parent'],K==0,data['stop'],data['p'],r['words'],data['view'],
                            int(z['side']),int(z['depth']),bool(z['immediate']),'physical')
                paired=joint*(us[:,None]*ur)
                target=paired.sum(1);ref=paired.sum(0)
                if target.sum()==0:target=us*m*data['p'][:,r['words'][0]]*visible
                if ref.sum()==0:ref=ur*rf
                mask=np.zeros((2,len(K)),bool)
                mask[0,cover(target,K,visible)]=True;mask[1,cover(ref,K,visible)]=True
            else:mask=np.empty((2,0),bool)
            nominations.update(np.flatnonzero(mask.any(0)).tolist())
            noms.append(mask);subjects.append(us);refs.append(ur);ids.append(row_id(r))
    # Different sentences can share a capture but each accesses only its own two nomination masks.
    file=WORK/'plans'/(fid+'.npz')
    archive_npz(file,**{k:data[k] for k in ('parent','size','p','stop','view','assign','xyz','boxes','completed')},
                row=np.array(ids),nom=np.array(noms),us=np.array(subjects),ur=np.array(refs),crop_nodes=np.array(sorted(nominations),int))
    return fid,digest(file),len(nominations)

def make(workers):
    cat=catalog();seal=read(previous.WORK/'SEALED.json');pins=base()
    for path in (WORK,OUT,WORK/'plans') :path.mkdir(exist_ok=True,mode=0o700)
    by=defaultdict(list)
    for r in cat['rows']:
        if r['active']:by[r['frame']].append(r)
    assert set(by)==set(cat['frames'])
    tasks=[(k,cat['frames'][k],v,seal['observed'][k]) for k,v in by.items()]
    files={};counts={};t=perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(k,h,n) in enumerate(pool.map(job,tasks,chunksize=1),1):
            files[k]=h;counts[k]=n
            if i%300==0:print('planned',i,'of',len(tasks),round(perf_counter()-t),flush=True)
    save(WORK/'PLANS.json',dict(files=files,counts=counts,base=pins,code=digest(HERE/'nominate.py'),labels_used=False))
    print('planned',len(files),'source frames',sum(counts.values()),'local objects',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);args=parser.parse_args();make(args.workers)
