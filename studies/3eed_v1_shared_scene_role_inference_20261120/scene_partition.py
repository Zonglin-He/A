"""A shared, nested segmentation of the measured scene and class evidence, before scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import softmax

from shared_paths import HERE, LEX, NEW, SEG, SPEC, read, save, digest, name, archive_npz, observed_roster
from collect import camera
from crossplatform.round7.camera_projection import project_model_points
from physical_evidence import lexicon
from scope183 import VIEWS


def terrain(xyz):
    cell=np.floor(xyz[:,:2]).astype(int)
    order=np.lexsort((xyz[:,2],cell[:,1],cell[:,0]))
    x=xyz[order[np.r_[True,(np.diff(cell[order],axis=0)!=0).any(1)]]]
    if len(x)<12:
        return np.full(3,np.nan)
    design=np.column_stack((x[:,:2],np.ones(len(x))))
    rng=np.random.default_rng(5927)
    slope=np.tan(np.deg2rad(25.))
    best=(-1,-np.inf)
    fit=None
    for _ in range(96):
        ids=rng.choice(len(x),3,replace=False)
        try:
            v=np.linalg.solve(design[ids],x[ids,2])
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(v[:2])>slope:continue
        error=np.abs(x[:,2]-design@v)
        hit=error<=.20
        if hit.sum()<12:continue
        rank=(hit.sum(),-np.median(error[hit]))
        if rank>best:best=rank;fit=hit
    if fit is None:return np.full(3,np.nan)
    v,_,rank,_=np.linalg.lstsq(design[fit],x[fit,2],rcond=None)
    return v if rank==3 and np.linalg.norm(v[:2])<=slope else np.full(3,np.nan)


def projection(x,f):
    view=VIEWS.index(f['waymo_view'])
    cam,size=camera(f['target'],view,str(f['calibration']),str(f['rgb']))
    p=project_model_points(x,cam,size)
    valid=p['in_image'];uv=np.where(valid[:,None],p['uv'],0.)
    nx=(size[0]+3)//4
    tile=np.floor(uv/4).astype(int);flat=tile[:,1]*nx+tile[:,0]
    zbuf=np.full(nx*((size[1]+3)//4),np.inf)
    np.minimum.at(zbuf,flat[valid],p['depth'][valid])
    seen=valid & (p['depth']<=zbuf[flat]+.30+.02*p['depth'])
    return uv[:,0]/size[0],p['depth'],seen


def levels(x,fg):
    cuts=np.full(8,np.nan)
    out=np.full((8,len(x)),-1,dtype=np.int32)
    ids=np.flatnonzero(fg)
    if len(ids)<9:return cuts,out
    dist,neigh=cKDTree(x[ids]).query(x[ids],k=9)
    u=np.repeat(np.arange(len(ids)),8)
    v=neigh[:,1:].reshape(-1)
    d=np.maximum.reduce((dist[:,1:].reshape(-1),dist[u,4],dist[v,4]))
    cuts=np.quantile(d,(np.arange(8)+.5)/8)
    for k,cut in enumerate(cuts):
        mask=d<=cut
        graph=coo_matrix((np.ones(mask.sum(),np.uint8),(u[mask],v[mask])),shape=(len(ids),len(ids))).tocsr()
        count,lab=connected_components(graph,directed=False)
        keep=np.bincount(lab,minlength=count)>=8
        lookup=np.where(keep,np.cumsum(keep)-1,-1)
        out[k,ids]=lookup[lab]
    return cuts,out


def describe(x,labels,pointscores,proj):
    u,depth,view=proj
    count=int(labels.max())+1
    sizes=np.bincount(labels[labels>=0],minlength=count)
    scores=np.zeros((count,pointscores.shape[1]))
    for k in range(scores.shape[1]):
        scores[:,k]=np.bincount(labels[labels>=0],weights=pointscores[labels>=0,k],minlength=count)/sizes
    desc=np.full((count,5),np.nan)
    ids=np.flatnonzero(view & (labels>=0))
    if len(ids):
        order=ids[np.argsort(labels[ids],kind='stable')]
        groups=np.split(order,np.flatnonzero(np.diff(labels[order]))+1)
        for ix in groups:
            g=int(labels[ix[0]])
            lo,mid,hi=np.quantile(u[ix],[.25,.5,.75])
            desc[g]=[lo,mid,hi,np.median(depth[ix]),len(ix)]
    return sizes.astype(np.int32),softmax(scores,axis=1),desc


def build(x,scores,seeds,frame):
    xyz=np.unique(x.astype(float),axis=0)
    ground=terrain(xyz)
    height=xyz[:,2]-xyz[:,:2]@ground[:2]-ground[2]
    cuts,lab=levels(xyz,height>.20)
    field,centered,_=lexicon(seeds,scores)
    centered-=centered.mean(0,keepdims=True)
    point_scores=centered[field.tree.query(xyz)[1]]
    proj=projection(xyz,frame)
    result=dict(xyz=xyz.astype(np.float32),plane=ground,labels=lab,cuts=cuts)
    for k in range(len(cuts)):
        size,prob,desc=describe(xyz,lab[k],point_scores,proj)
        result['size'+str(k)]=size;result['p'+str(k)]=prob;result['view'+str(k)]=desc
    return result


def worker(task):
    fid,path,sha,frame=task
    assert digest(path)==sha
    with np.load(path) as h:
        data=build(h['sample'],h['logits'],h['xyz'],frame)
    out=SEG/(fid+'.npz')
    archive_npz(out,**data)
    return fid,digest(out)


def run(extra,workers):
    rows=observed_roster()
    if extra:
        s=read(NEW/'SEALED.json')
        mapping={n:v['fid'] for n,v in s['files'].items()};folder=NEW/'lexical'
    else:
        folder=LEX/'private/observations';s=read(folder/'SEALED.json');mapping=s['files']
    frames={mapping[name(r)]:r['frame'] for r in rows if name(r) in mapping}
    tasks=[(fid,folder/(fid+'.npz'),s['hashes'][fid],f) for fid,f in frames.items()]
    SEG.mkdir(exist_ok=True,mode=0o700)
    files={}
    with ProcessPoolExecutor(max_workers=workers) as workers:
        for i,(fid,h) in enumerate(workers.map(worker,tasks,chunksize=2),1):
            files[fid]=h
            if i%100==0 or i==len(tasks):print('components',i,len(tasks),flush=True)
    save(SEG/('new.json' if extra else 'old.json'),dict(files=files,code=digest(HERE/'scene_partition.py'),
          protocol=digest(SPEC),labels_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cohort',choices=['old','new']);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    run(args.cohort=='new',args.workers)
