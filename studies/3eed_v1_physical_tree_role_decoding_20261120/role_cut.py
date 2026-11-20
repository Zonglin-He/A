"""Exact role-conditioned expected witness counts under shared and independent object inventories."""
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import numpy as np
from paths_tree import HERE, WORK, digest


def build_library():
    target=WORK/('cut_'+digest(HERE/'role_cut.cpp')[:14]+'.so')
    if not target.exists():
        subprocess.run(['c++','-std=c++17','-O3','-fPIC','-shared',str(HERE/'role_cut.cpp'),'-o',str(target)],check=True)
    return target


@lru_cache(maxsize=1)
def kernel():
    library=ctypes.CDLL(str(build_library()))
    f=library.named_pairs
    it=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    byte=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    fl=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    f.restype=None
    f.argtypes=[ctypes.c_int,it,byte,byte]+[fl]*7+[ctypes.c_int]*4+[fl,fl]
    return f


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


def separate(valid,occupancy,p,words,view,side,depth,nearest):
    v,r,vis,gap=layout(view,side,'physical')
    sub=occupancy*p[:,words[0]]*vis
    ref=occupancy*p[:,words[1]]*vis
    earlier=(depth*(r[:,3,None]-r[:,3])<0.) & vis[:,None]
    survives=1.-earlier*ref[:,None]
    landmark=ref*survives.prod(0)
    numerator=sub[:,None]*(~earlier if words[0]==words[1] else 1.)
    conditional=np.divide(numerator,survives,out=np.zeros_like(survives),where=survives>0)
    conditional*=valid & (gap>0) & vis[None,:]
    joint=conditional*landmark
    if nearest:
        # For a fixed side, the order in which subjects are met is common to all reference positions.
        order=np.argsort(side*v[:,1],kind='stable');by=conditional[order]
        survival=np.vstack([np.ones(len(p)),np.cumprod(1.-by[:-1],axis=0)])
        ties=np.r_[True,np.diff(v[order,1])!=0.]
        start=np.maximum.accumulate(np.where(ties,np.arange(len(p)),0))
        joint[order]*=survival[start]
    return joint,landmark


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


def decode(data,words,us,ur,side,depth,nearest):
    parent,stop,p,view=data['parent'],data['stop'],data['p'],data['view']
    n=len(stop);margin,K=structure(parent,stop,data['size'])
    valid=K==0;results={}
    exact,ref=shared(parent,valid,stop,p,words,view,side,depth,nearest,'physical')
    vis=np.isfinite(view[:,0])
    si=K@(us*margin*p[:,words[0]]*vis);ri=K@(ur*ref)
    results['independent']=(int(np.argmax(np.where(vis,si,-1))) if si.max(initial=0)>0 else -1,
        int(np.argmax(np.where(vis,ri,-1))) if ri.max(initial=0)>0 else -1,si)
    for arm in ['shared_cut','pair_relation','wrong_reference_binding','image_center','separate_occupancy','global_cut']:
        if arm=='shared_cut' or arm=='pair_relation' and not nearest:j=exact
        elif arm=='pair_relation':j=shared(parent,valid,stop,p,words,view,side,depth,False,'physical')[0]
        elif arm=='separate_occupancy':j=separate(valid,margin,p,words,view,side,depth,nearest)[0]
        elif arm=='global_cut':
            j=np.zeros((n,n))
            for k in range(8):
                members=np.flatnonzero((data['level']&(1<<k))!=0)
                if not len(members):continue
                ix=np.ix_(members,members)
                j[ix]+=separate(valid[ix],np.ones(len(members)),p[members],words,view[members],side,depth,nearest)[0]/8
        else:j=shared(parent,valid,stop,p,words,view,side,depth,nearest,arm)[0]
        results[arm]=medoid(j,K,us,ur,valid)
    return results
