"""A conditional-independent vote table preserves both observed margins, including unassigned mass."""
import ctypes
from functools import lru_cache
import subprocess
import numpy as np
from scipy.special import softmax
from vote_paths import HERE,WORK,digest
from role_cut import structure

def build():
    out=WORK/('return_votes_'+digest(HERE/'return_votes.cpp')[:16]+'.so')
    if not out.exists():
        subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared',str(HERE/'return_votes.cpp'),'-o',str(out)],check=True)
    return out

@lru_cache(maxsize=1)
def library():
    dll=ctypes.CDLL(str(build()))
    f=dll.spread
    doubles=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    ints=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    f.restype=None
    f.argtypes=[ctypes.c_int]*4+[doubles,ints,doubles,doubles,doubles,ints]
    return f

def distribute(data,boxes,occupancy):
    n=len(occupancy)
    xyz=np.ascontiguousarray(data['xyz'],np.float64)
    chain=np.ascontiguousarray(data['assign'],np.int32)
    b=np.ascontiguousarray(boxes,dtype=np.float64)
    out=np.zeros(b.shape[:2]+(n,),np.float64)
    supp=np.zeros(b.shape[:2],np.int32)
    assert chain.shape[1]==len(xyz) and np.all((chain>=-1)&(chain<n))
    assert np.isfinite(b).all() and np.all(b[:,:,3:]>=0)
    if n:
        library()(len(xyz),len(chain),n,supp.size,xyz,chain,b,np.ascontiguousarray(occupancy),out,supp)
    assert out.sum(-1).max(initial=0)<=1.+1e-11 
    return out,supp

def seed_words(logits,xyz):
    _,inverse,count=np.unique(xyz,axis=0,return_inverse=True,return_counts=True)
    s=np.mean(logits,axis=0,dtype=float)
    s-=s.mean(axis=1,keepdims=True)
    s-=np.sum(s/count[inverse,None],axis=0)/len(count)
    p=softmax(s,axis=1)
    q=p[:,:-1]/p[:,:-1].sum(axis=1,keepdims=True)
    w=p[:,:-1]/count[inverse,None]
    return q,w

def counts(g,q,w):
    direct=np.einsum('scv,sc->vc',g,w)
    mean=np.einsum('scv,sc->sv',g,q)
    null=mean.T@w
    # Verify margins with the unassigned sink as a physical destination.
    missing=np.maximum(1.-g.sum(axis=-1),0.)
    direct_sink=np.sum(w*missing,axis=0)
    null_sink=np.sum(w*np.sum(q*missing,axis=1,keepdims=True),axis=0)
    np.testing.assert_allclose(direct.sum(axis=1),null.sum(axis=1),atol=1e-9,rtol=1e-12)
    np.testing.assert_allclose(direct.sum(0)+direct_sink,w.sum(0),atol=1e-9,rtol=1e-12)
    np.testing.assert_allclose(null.sum(0)+null_sink,w.sum(0),atol=1e-9,rtol=1e-12)
    return direct,null

def unit(x):
    return x/x.sum(axis=1,keepdims=True)

def update(base,observed,independent):
    strength=1./base.shape[1]
    return unit(base*((observed+strength)/(independent+strength)))

def evidence(data,lex,centers,extents):
    base=data['p'];m,K=structure(data['parent'],data['stop'],data['size'])
    q,w=seed_words(lex['logits'],lex['xyz'])
    b=np.concatenate([centers,np.maximum(extents,1e-6)],axis=-1).astype(float)
    g,supp=distribute(data,b,m)
    d,e=counts(g,q,w)
    vals=dict(tree=base,association=update(base,d,e),inverse_association=update(base,e,d),
        direct_votes=unit(base+d),direct_null=unit(base+e))
    mean=np.einsum('sc,scd->sd',q,b)
    for arm,frozen in [('centers_only',slice(3,6)),('sizes_only',slice(0,3))]:
        mixture=b.copy()
        mixture[:,:,frozen]=mean[:,None,frozen]
        a,c=counts(distribute(data,mixture,m)[0],q,w)
        vals[arm]=update(base,a,c)
    return vals,(g,b,q,w),K,m,dict(mass=d.sum(),supported=int((supp>0).sum()),votes=supp.size,
        association_tv=float((abs(d-e).sum()+abs(d.sum(0)-e.sum(0)).sum())/(2*w.sum())))

def weighted_median(samples,weights):
    order=np.argsort(samples,axis=0,kind='stable')
    cum=np.cumsum(weights[order],axis=0)
    rank=(cum>=.5*weights.sum()).argmax(axis=0)
    return np.take_along_axis(samples,order[rank,np.arange(order.shape[1])][None],axis=0)[0]

def fit_faces(data,aux,node,word,neutral=False):
    g,b,q,w=aux
    if neutral:
        weights=(w[:,word,None]*q*g[:,:,node]).ravel();boxes=b.reshape(-1,6)
    else:
        weights=w[:,word]*g[:,word,node];boxes=b[:,word]
    active=weights>0
    if not active.any():return data['completed'][node,word]
    boxes=boxes[active];weights=weights[active]
    measured=data['boxes'][node]
    angle=measured[6]
    cs,sn=np.cos(angle),np.sin(angle)
    axes=np.array([[cs,sn,0],[-sn,cs,0],[0,0,1.]])
    center=boxes[:,:3]@axes.T
    dims=boxes[:,3:].copy()
    dims[:,:2]=np.sort(dims[:,:2],axis=1)
    if measured[3]>=measured[4]:dims[:,:2]=dims[:,[1,0]]
    proposed=np.column_stack((center-dims/2,center+dims/2))
    faces=weighted_median(proposed,weights)
    origin=measured[:3]@axes.T
    lo=np.minimum(faces[:3],origin-measured[3:6]/2)
    hi=np.maximum(faces[3:],origin+measured[3:6]/2)
    return np.r_[(hi+lo)/2@axes,hi-lo,angle]

