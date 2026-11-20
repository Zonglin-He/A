"""Shared-cut inference and physically feasible decisions for both roles."""
import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np


@lru_cache(maxsize=1)
def kernel():
    source = Path(__file__).with_name("role_cut.cpp")
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"3eed-physical-regions"
    cache.mkdir(mode=0o700,parents=True,exist_ok=True)
    target = cache/(key+".so")
    if not target.exists():
        with TemporaryDirectory(dir=cache) as tmp:
            out = Path(tmp)/"role_cut.so"
            subprocess.run(["c++","-std=c++17","-O3","-fPIC","-shared",str(source),"-o",str(out)],check=True)
            os.replace(out,target)
    library = ctypes.CDLL(str(target))
    fun = library.named_pairs
    ptr = lambda t:np.ctypeslib.ndpointer(dtype=t,flags="C_CONTIGUOUS")
    fun.restype = None
    fun.argtypes = [ctypes.c_int,ptr(np.int32),ptr(np.uint8),ptr(np.uint8)]+[ptr(np.float64)]*7+[ctypes.c_int]*4+[ptr(np.float64)]*2
    return fun


def prepare(z,prefix):
    boxes=z[prefix+'_box']
    p=np.asarray(z[prefix+'_p'],float)
    if not np.isfinite(p).all() or np.any(p<0) or p.max()<=0:
        raise ValueError('Invalid native semantic probabilities')
    p=p/p.max()
    # Canonical physical ties and duplicate geometries are independent of query array order.
    unique,idx,inv=np.unique(boxes,axis=0,return_index=True,return_inverse=True)
    strength=np.zeros(len(idx)); weight=np.zeros(len(idx))
    np.maximum.at(strength,inv,p)
    np.maximum.at(weight,inv,z[prefix+'_m'])
    weight/=weight.sum()
    proj=z[prefix+'_project'][idx]
    take=(proj[:,7]>0) & (z[prefix+'_count'][idx]>0)
    return unique[take],strength[take],weight[take],proj[take]


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


def structure(parents,stop,size):
    n=len(parents)
    ancestors=np.eye(n,dtype=bool);unoccluded=np.ones(n)
    for i in range(n-1,-1,-1):
        p=parents[i]
        if p>=0:
            assert p>i
            ancestors[i]|=ancestors[p]
            unoccluded[i]=unoccluded[p]*(1-stop[p])
    jaccard=ancestors*(size[:,None]/size)
    jaccard=np.maximum(jaccard,jaccard.T)
    return unoccluded*stop,jaccard

def layout(view,side,mode):
    visible=np.isfinite(view[:,0])
    v=np.nan_to_num(view)
    ref=v.copy()
    if mode=='wrong_reference_binding':
        order=np.flatnonzero(visible);order=order[np.argsort(v[order,1],kind='stable')]
        if len(order)>1:ref[order]=v[np.roll(order,max(1,len(order)//2))]
    if mode=='image_center':ref[:,:3]=.5
    g=side*(v[:,1,None]-ref[:,1])
    eligible=v[:,2,None]<ref[:,0] if side<0 else v[:,0,None]>ref[:,2]
    return v,ref,visible,np.where(eligible,g,0.)

def shared(parents,valid,stop,p,words,view,side,depth,nearest,mode):
    n=len(p);v,r,vis,gap=layout(view,side,mode)
    m=np.zeros((n,n));ref=np.zeros(n)
    f=lambda x:np.ascontiguousarray(x,dtype=np.float64)
    kernel()(n,np.ascontiguousarray(parents,dtype=np.int32),np.ascontiguousarray(valid,np.uint8),
        np.ascontiguousarray(vis,np.uint8),f(stop),f(p[:,words[0]]),f(p[:,words[1]]),f(v[:,1]),f(r[:,1]),
        f(r[:,3]),f(gap),side,depth,int(nearest),int(words[0]==words[1]),m,ref)
    return m,ref

def medoid(joint,K,us,ur,valid):
    mass=(us[:,None]*joint)*ur
    marginal=K@mass.sum(1)
    if marginal.max(initial=0.)<=0:return -1,-1,marginal
    # An emitted pair must still obey disjointness and the explicit directional relation.
    allowed=valid & (joint>0.)
    feasible=allowed.any(1)
    if not feasible.any():return -1,-1,marginal
    i=int(np.argmax(np.where(feasible,marginal,-1.)))
    score=K@(K[i]@mass)
    r=int(np.argmax(np.where(allowed[i],score,-1.)))
    return i,r,marginal


def closed_joint(parent,stop,occupancy,probs,words,view,valid,side):
    # Disjoint nodes share some ancestor survival events. Count each common event once.
    ancestors = np.eye(len(parent),dtype=bool)
    for i in reversed(range(len(parent))):
        if parent[i]>=0:ancestors[i] |= ancestors[parent[i]]
    gain = np.ones_like(valid,dtype=float)
    for i,s in enumerate(stop):
        if s == 1:continue  # Any strict descendant already has zero marginal.
        both = np.flatnonzero(ancestors[:,i] & (np.arange(len(parent))!=i))
        if len(both)>1:gain[np.ix_(both,both)] /= 1-s
    _,_,vis,gap = layout(view,side,'physical')
    return ((occupancy*probs[:,words[0]])[:,None]*(occupancy*probs[:,words[1]])*
            gain*valid*(gap>0)*vis[:,None]*vis[None,:])


def select(data, subject_proposals, words, side, depth, nearest, points):
    stop = data['stop']
    if not len(stop):
        return -1,-1
    occupancy, kernel_overlap = structure(data['parent'],stop,data['size'])
    valid = kernel_overlap == 0
    if depth or nearest:
        joint,_ = shared(data['parent'],valid,stop,data['p'],words,data['view'],side,depth,nearest,'physical')
    else:
        joint = closed_joint(data['parent'],stop,occupancy,data['p'],words,data['view'],valid,side)
    subject = overlap_prior(data,subject_proposals,'t',points)
    # The reference noun and its geometry already contribute to the joint mass.
    i,j,_ = medoid(joint,kernel_overlap,subject,1.0,valid)
    return i,j
