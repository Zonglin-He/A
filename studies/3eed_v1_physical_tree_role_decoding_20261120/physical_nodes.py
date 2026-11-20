"""Turn distinct nested point supports into a shared inventory, retaining their actual geometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from paths_tree import *
from scene_faces import split_returns
from physical_evidence import Returns
from ordered_choice import prepare

def measured_box(points,plane):
    mid=points[:,:2].mean(0)
    xy=points[:,:2]-mid
    _,directions=np.linalg.eigh(xy.T@xy)
    long=directions[:,-1]
    yaw=float(np.arctan2(long[1],long[0]) % np.pi)
    axes=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    local=xy@axes.T
    lo,hi=local.min(0),local.max(0)
    center=(lo+hi)/2@axes+mid
    low,top=np.min(points[:,2]),np.max(points[:,2])
    if np.isfinite(plane).all():
        low=min(low,center@plane[:2]+plane[2])
    b=np.array([*center,(top+low)/2,*(hi-lo),top-low,yaw])
    b[3:6]=np.maximum(b[3:6],1e-6)
    return b

def complete(box,dimensions):
    b=box.copy()
    yaw=b[6]
    axes=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    dims=dimensions.copy()
    dims[:2]=np.sort(dims[:2])[::-1] if b[3]>=b[4] else np.sort(dims[:2])
    slab=b[:2]@axes.T
    advance=np.where(np.abs(slab)>b[3:5]/2,np.sign(slab),0.)
    increase=np.maximum(dims-b[3:6],0.)
    b[:2]+=(advance*increase[:2]/2)@axes
    b[2]+=increase[2]/2
    b[3:6]+=increase
    return b

def assemble(seg,word):
    x=seg['xyz'].astype(float)
    loc,inv=np.unique(word['xyz'],return_inverse=True,axis=0)
    pred=np.maximum(word['extent'].astype(float),1e-6)
    dim=np.zeros((len(loc),)+pred.shape[1:])
    np.add.at(dim,inv,pred)
    dim/=np.bincount(inv)[:,None,None]
    near=cKDTree(loc).query(x)[1]
    allids=np.full_like(seg['labels'],-1)
    index={};para=[];levels=[];size=[];boxes=[];views=[];probs=[];stops=[];completed=[]
    for k,lab in enumerate(seg['labels']):
        ordered=np.flatnonzero(lab>=0)
        if not len(ordered):continue
        ordered=ordered[np.argsort(lab[ordered],kind='stable')]
        for ids in np.split(ordered,np.flatnonzero(np.diff(lab[ordered]))+1):
            key=int(ids.min()),len(ids)
            if key not in index:
                n=len(para);index[key]=n
                c=int(lab[ids[0]])
                box=measured_box(x[ids],seg['plane'])
                proto=np.median(dim[np.unique(near[ids])],axis=0)
                obs=box[3:6].copy();obs[:2]=np.sort(obs[:2])[::-1]
                canonical=proto.copy();canonical[:,:2]=np.sort(proto[:,:2],axis=1)[:,::-1]
                match=np.prod(np.minimum(obs,canonical)/np.maximum(obs,canonical),axis=1)
                s=seg['p'+str(k)][c,:-1]*match
                probs.append(s/max(s.sum(),np.finfo(float).tiny))
                stops.append(match.max());para.append(-1);levels.append(0)
                size.append(len(ids));boxes.append(box);views.append(seg['view'+str(k)][c])
                completed.append([complete(box,d) for d in proto])
            n=index[key];levels[n]|=1<<k
            allids[k,ids]=n
            if k:
                for child in np.unique(allids[k-1,ids]):
                    if child>=0 and child!=n:
                        assert child<n and para[child] in (-1,n)
                        para[child]=n
    count=len(para);nclass=len(CLASSES)-1
    return dict(parent=np.array(para,int),level=np.array(levels,int),size=np.array(size,int),
        boxes=np.array(boxes,float).reshape(count,7),completed=np.array(completed,float).reshape(count,nclass,7),
        view=np.array(views,float).reshape(count,5),p=np.array(probs,float).reshape(count,nclass),
        stop=np.array(stops),assign=allids,xyz=seg['xyz'],plane=seg['plane'])

def overlap_prior(data,z,prefix,points):
    boxes,_,mass,_=prepare(z,prefix)
    counts=data['size']; total=np.zeros(len(counts)); n=len(total)
    if not n:return total
    for b,w in zip(boxes,mass):
        loc=points.local(b)[0]
        if not len(loc):continue
        ids=data['assign'][:,loc]
        # One node can persist through several global cuts; count each return at most once per node.
        votes=np.zeros(n)
        for line in ids:
            v=line[line>=0]
            if len(v):
                votes=np.maximum(votes,np.bincount(v,minlength=n))
        j=votes/(counts+len(loc)-votes)
        if j.sum()>0:total+=w*j/j.sum()
    if total.sum()>0:total/=total.sum()
    return np.sqrt(1.+n*total)

def job(task):
    fid,sfile,sh,wh=task
    word=parent.OBS/(fid+'.npz')
    assert digest(word)==wh and digest(sfile)==sh
    with np.load(sfile) as seg,np.load(word) as w:info=assemble(dict(seg),w)
    path=OBJECTS/(fid+'.npz');archive_npz(path,**info)
    return fid,digest(path)
def run(workers):
    pins=base_pins();src=read(FIELDS/'SEALED.json')['files']; words=read(parent.OBS/'SEALED.json')['hashes']
    OBJECTS.mkdir(mode=0o700,exist_ok=True);tasks=[]
    for fid,wh in words.items():
        sf=parent.previous.SEG/(fid+'.npz')
        if not sf.exists():sf=parent.WORK/'segments'/(fid+'.npz')
        tasks.append((fid,sf,src[fid][1],wh))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out={}
        for k,(fid,h) in enumerate(ex.map(job,tasks,chunksize=1),1):
            out[fid]=h
            if k%100==0:print('physical scenes',k,len(tasks),flush=True)
    assert set(out)==set(words)
    save(OBJECTS/'SEALED.json',dict(files=out,input=pins,protocol=digest(SPEC),code=digest(HERE/'physical_nodes.py'),labels_used=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=7);run(p.parse_args().workers)
