"""Cut the physical merge forest separately at each measured noun envelope."""
import ctypes
from functools import lru_cache
import subprocess

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import softmax

from partition_paths import HERE, WORK, digest
from scene_partition import terrain, projection
from physical_evidence import lexicon
from physical_nodes import measured_box

ANGLES = np.arange(16)*np.pi/16
AXES = np.column_stack((np.cos(ANGLES),np.sin(ANGLES)))


@lru_cache(None)
def kernel():
    source=HERE/'merge_forest.cpp'
    library=WORK/('forest_'+digest(source)[:16]+'.so')
    if not library.exists():
        subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(library)],check=True)
    fun=ctypes.CDLL(str(library)).maximal_cuts
    arr=lambda dtype:np.ctypeslib.ndpointer(dtype=dtype,flags='C_CONTIGUOUS')
    fl=arr(np.float64);it=arr(np.int32);byte=arr(np.uint8)
    fun.argtypes=[ctypes.c_int,ctypes.c_int,fl,it,it,fl,fl,fl,it,it,byte,it,it]
    fun.restype=ctypes.c_int
    return fun


def cuts(foreground,templates):
    n=len(foreground)
    parent=np.empty(2*n,np.int32);size=np.empty_like(parent)
    marks=np.empty((2*n,7),np.uint8);assigned=np.full((7,n),-1,np.int32)
    coverage=np.zeros(7,np.int32)
    if n>1:
        k=min(n,9)
        ds,nb=cKDTree(foreground).query(foreground,k=k)
        u=np.repeat(np.arange(n,dtype=np.int32),k-1)
        v=nb[:,1:].ravel().astype(np.int32)
        core=ds[:,min(k-1,4)]
        height=np.maximum.reduce((ds[:,1:].ravel(),core[u],core[v]))
        u,v=np.minimum(u,v),np.maximum(u,v)
        order=np.lexsort((v,u,height))
        u,v,height=u[order],v[order],height[order]
    else:
        u=v=np.empty(0,np.int32);height=np.empty(0)
    m=kernel()(n,len(u),np.ascontiguousarray(foreground,dtype=float),u,v,height,AXES,
               np.ascontiguousarray(templates,dtype=float),parent,size,marks,assigned,coverage)
    valid=np.isfinite(templates).all(1)&(templates>0).all(1)
    np.testing.assert_array_equal(coverage,valid.astype(int)*n)
    return parent[:m],size[:m],marks[:m].astype(bool),assigned


def word_extent(word):
    loc,inv=np.unique(word['xyz'],axis=0,return_inverse=True)
    v=word['extent'].astype(float)
    dim=np.zeros((len(loc),6,3));count=np.zeros((len(loc),6),int)
    finite=np.isfinite(v).all(2)&(v>0).all(2)
    np.add.at(dim,inv,np.where(finite[:,:,None],v,0))
    np.add.at(count,inv,finite)
    avg=np.divide(dim,count[:,:,None],out=np.full_like(dim,np.nan),where=count[:,:,None]>0)
    template=np.full((7,3),np.nan)
    for c in range(6):
        values=avg[count[:,c]>0,c]
        if len(values):
            template[c]=np.median(values,axis=0)
            template[c,:2]=np.sort(template[c,:2])[::-1]
    ok=np.isfinite(template[:6]).all(1)
    if ok.any():template[-1]=np.median(template[:6][ok],axis=0)
    # Local emissions replay the incumbent's positive-clipping convention.
    historical=np.zeros_like(dim)
    np.add.at(historical,inv,np.maximum(v,1e-6))
    historical/=np.bincount(inv)[:,None,None]
    return loc,historical,template


def expand(boxes,dimensions):
    boxes=np.array(boxes,float,copy=True)
    co,si=np.cos(boxes[:,6]),np.sin(boxes[:,6])
    x,y=boxes[:,0],boxes[:,1]
    slab=np.column_stack((co*x+si*y,-si*x+co*y))
    delta=np.maximum(dimensions-boxes[:,3:6],0.)
    advance=np.where(abs(slab)>boxes[:,3:5]/2,np.sign(slab),0.)*delta[:,:2]/2
    boxes[:,0]+=advance[:,0]*co-advance[:,1]*si
    boxes[:,1]+=advance[:,0]*si+advance[:,1]*co
    boxes[:,2]+=delta[:,2]/2
    boxes[:,3:6]+=delta
    return boxes


def pca_complete(box,dimensions):
    dims=dimensions.copy()
    dims[:,:2]=np.sort(dims[:,:2],axis=1)
    if box[3]>=box[4]:dims[:,:2]=dims[:,1::-1]
    return expand(np.tile(box,(len(dims),1)),dims)


def oriented_complete(points,plane,box,dimensions):
    proj=points[:,:2]@AXES.T
    lo=proj.min(0);hi=proj.max(0);span=hi-lo
    perp=(np.arange(len(AXES))+8)%16
    scale=np.maximum(span/dimensions[:,0,None],span[perp]/dimensions[:,1,None])
    distance=abs((ANGLES-box[6]+np.pi/2)%np.pi-np.pi/2)
    axis_order=np.lexsort((np.arange(16),distance))
    k=axis_order[np.argmin(scale[:,axis_order],axis=1)]
    axes=AXES[k];perpendicular=np.column_stack((-axes[:,1],axes[:,0]))
    # The stored orthogonal direction can differ in sign from the positive local y axis.
    sign=np.where(k<8,1.,-1.)
    center=axes*((lo+hi)[k,None]/2)+perpendicular*(sign*(lo+hi)[perp[k]]/2)[:,None]
    top=points[:,2].max()
    low=np.minimum(points[:,2].min(),center@plane[:2]+plane[2]) if np.isfinite(plane).all() else np.full(len(k),points[:,2].min())
    bounds=np.column_stack((center,(top+low)/2,span[k],span[perp[k]],top-low,ANGLES[k]))
    bounds[:,3:6]=np.maximum(bounds[:,3:6],1e-6)
    return expand(bounds,dimensions)


def partition(lex,word,frame):
    x=np.unique(lex['sample'].astype(float),axis=0);plane=terrain(x)
    ids=np.flatnonzero(x[:,2]-x[:,:2]@plane[:2]-plane[2]>.20)
    locations,extent,template=word_extent(word)
    parent,size,mark,assigned=cuts(x[ids],template)
    n=len(parent);allids=np.full((7,len(x)),-1,np.int32);allids[:,ids]=assigned
    nodes=[None]*n
    for row in assigned:
        ix=np.flatnonzero(row>=0)
        if len(ix):
            ix=ix[np.argsort(row[ix],kind='stable')]
            for sub in np.split(ix,np.flatnonzero(np.diff(row[ix]))+1):
                v=row[sub[0]]
                if nodes[v] is None:nodes[v]=ids[sub]
    seeds,values,_=lexicon(lex['xyz'],lex['logits'])
    values-=values.mean(0)
    logits=values[seeds.tree.query(x)[1],:-1]
    nearest=cKDTree(locations).query(x)[1]
    u,depth,visible=projection(x,frame)
    boxes=np.empty((n,7));view=np.full((n,5),np.nan);probs=np.empty((n,6));stop=np.empty(n)
    local=np.empty((n,6,7));scene=np.empty_like(local);envelope=np.empty_like(local)
    for i,region in enumerate(nodes):
        assert region is not None and len(region)==size[i]
        xyz=x[region];box=measured_box(xyz,plane);boxes[i]=box
        dims=np.median(extent[np.unique(nearest[region])],axis=0)
        canonical=dims.copy();canonical[:,:2]=np.sort(dims[:,:2],axis=1)[:,::-1]
        observed=box[3:6].copy();observed[:2]=np.sort(observed[:2])[::-1]
        match=np.prod(np.minimum(observed,canonical)/np.maximum(observed,canonical),axis=1)
        probs[i]=softmax(logits[region].mean(0)+np.log(np.maximum(match,np.finfo(float).tiny)))
        stop[i]=match.max()
        selected=region[visible[region]]
        if len(selected):view[i]=[*np.quantile(u[selected],[.25,.5,.75]),np.median(depth[selected]),len(selected)]
        local[i]=pca_complete(box,dims)
        prior=np.where(np.isfinite(template[:6]),template[:6],canonical)
        scene[i]=pca_complete(box,prior)
        envelope[i]=oriented_complete(xyz,plane,box,prior)
    return dict(xyz=x,assign=allids,parent=parent,size=size,mark=mark,boxes=boxes,view=view,p=probs,
                stop=stop,local=local,scene_pca=scene,envelope=envelope,template=template,foreground=len(ids))
