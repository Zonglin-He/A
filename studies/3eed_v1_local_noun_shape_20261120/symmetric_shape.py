"""Add the opposite side of the size discrepancy, retaining all original crops and controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from paths_shape import *
from shape_evidence import observed_extent, predicted_extent
from predict_shape import observations, run_case, FIELDS
from scene_faces import split_returns

CFG=OUT/'symmetric_protocol.json'
SW=WORK/'symmetric'
SF=SW/'fields'
SP=SW/'predictions'


def two_sided(seg,w,pre):
    x=seg['xyz'].astype(float); split=split_returns(x)
    seed,inv=np.unique(w['xyz'],axis=0,return_inverse=True)
    native=np.maximum(w['extent'].astype(float),1e-6)
    avg=np.zeros((len(seed),)+native.shape[1:])
    np.add.at(avg,inv,native)
    sizes=predicted_extent(avg/np.bincount(inv)[:,None,None])
    near=cKDTree(seed).query(x)[1];global_size=np.median(sizes,axis=0)
    result={};counts=[]
    for k,lab in enumerate(seg['labels']):
        prior=seg['p'+str(k)]
        ll=np.ones(prior.shape);gl=ll.copy()
        ids=np.flatnonzero(lab>=0)
        if len(ids):
            order=ids[np.argsort(lab[ids],kind='stable')]
            for g in np.split(order,np.flatnonzero(np.diff(lab[order]))+1):
                z=observed_extent(x[g],split[g]); valid=z>0
                if not valid.any():continue
                loc=np.median(sizes[np.unique(near[g])],axis=0)
                for arr,proto in [(ll,loc),(gl,global_size)]:
                    arr[lab[g[0]],:-1]=np.minimum(1.,z[valid]/proto[:,valid]).prod(1)
        wrong=ll.copy();wrong[:,:-1]=np.roll(ll[:,:-1],1,axis=1)
        for kind,atom in [('local',ll),('global',gl),('wrong',wrong)]:
            # pre is already the normalized posterior with the one-sided likelihood applied.
            q=pre[kind+str(k)]*atom
            result[kind+str(k)]=q/q.sum(1,keepdims=True)
        counts.append([len(prior),int((np.ptp(ll[:,:-1],axis=1)>1e-7).sum()),0,
            int((prior.argmax(1)!=result['local'+str(k)].argmax(1)).sum())])
    result['coverage']=np.asarray(counts)
    return result


def field_job(task):
    fid,sfile,sh,fhash,wsha=task
    assert digest(sfile)==sh and digest(FIELDS/(fid+'.npz'))==fhash
    with np.load(sfile) as f,np.load(FIELDS/(fid+'.npz')) as pre:
        if wsha:
            word=OBS/(fid+'.npz');assert digest(word)==wsha
            with np.load(word) as w:data=two_sided(dict(f),w,pre)
        else:data={}
    dst=SF/(fid+'.npz');archive_npz(dst,**data)
    return fid,digest(dst)


def precompute():
    sf=read(FIELDS/'SEALED.json');old=read(OBS/'SEALED.json')
    SW.mkdir(exist_ok=True,mode=0o700);SF.mkdir(exist_ok=True,mode=0o700)
    tasks=[]
    for fid,(fh,sh) in sf['files'].items():
        sfile=previous.SEG/(fid+'.npz')
        if not sfile.exists():sfile=WORK/'segments'/(fid+'.npz')
        tasks.append((fid,sfile,sh,fh,old['hashes'].get(fid)))
    with ProcessPoolExecutor(max_workers=7) as ex:files=dict(ex.map(field_job,tasks,chunksize=2))
    save(SF/'SEALED.json',dict(files=files,protocol=digest(CFG),old_fields=digest(FIELDS/'SEALED.json'),
        implementation=digest(HERE/'symmetric_shape.py')))


def decide(task):
    row,path,sha,fid,words,sfile,sh,fh=task
    assert digest(path)==sha and digest(sfile)==sh and digest(SF/(fid+'.npz'))==fh
    with np.load(path) as z,np.load(sfile) as src,np.load(SF/(fid+'.npz')) as new:
        data=run_case(z,dict(src),dict(new),words,int(fid[:8],16))
    dst=SP/name(row);archive_npz(dst,**data)
    return dst.name,digest(dst)


def predict():
    sf=read(SF/'SEALED.json');assert sf['protocol']==digest(CFG) and sf['implementation']==digest(HERE/'symmetric_shape.py')
    rows=observations();old=read(FIELDS/'SEALED.json')
    SP.mkdir(exist_ok=True,mode=0o700);tasks=[]
    for row,path,sha,_,_,fid,words,*_ in rows:
        sfile=previous.SEG/(fid+'.npz')
        if not sfile.exists():sfile=WORK/'segments'/(fid+'.npz')
        tasks.append((row,path,sha,fid,words,sfile,old['files'][fid][1],sf['files'][fid]))
    with ProcessPoolExecutor(max_workers=7) as ex:files=dict(ex.map(decide,tasks,chunksize=2))
    assert set(files)=={name(r[0]) for r in rows}
    save(SP/'SEALED.json',dict(files=files,n=len(rows),protocol=digest(CFG),fields=digest(SF/'SEALED.json'),
        original=digest(PRED/'SEALED.json'),labels_used=False,source_updates=0,target_updates=0,FINAL_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fields','predict'])
    precompute() if p.parse_args().stage=='fields' else predict()
