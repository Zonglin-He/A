"""Joint role probabilities in a single categorical, physically partitioned scene."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Role:
    index: np.ndarray
    assign: np.ndarray
    prior: np.ndarray
    view: np.ndarray
    observed: np.ndarray
    independent: np.ndarray


def anchor_returns(boxes,points):
    return [points.local(box)[0] for box in boxes]


def world(levels,level,seed):
    labels=levels['labels'][level]
    prob=levels['p'+str(level)]
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),level]))
    cdf=prob.cumsum(axis=1);cdf[:,-1]=1.
    def draw():
        u=rng.random((128,len(prob)))
        return np.sum(u[:,:,None]>=cdf[None,:,:],axis=-1).astype(np.int8)
    return labels,levels['size'+str(level)],levels['view'+str(level)],draw(),draw()


def role(cells,weights,word,scene):
    labels,sizes,desc,cats,independent=scene
    w=np.zeros((len(cells),len(sizes)))
    for i,ids in enumerate(cells):
        labs=labels[ids];labs=labs[labs>=0]
        if len(labs):
            intersection=np.bincount(labs,minlength=len(sizes))
            w[i]=weights[i]*intersection/(len(ids)+sizes-intersection)
    weight=w.sum(0)
    index=np.flatnonzero((weight>0)&np.isfinite(desc[:,0]))
    return Role(index,w[:,index]/weight[index],weight[index],desc[index],
                cats[:,index]==word,independent[:,index]==word)


def norm_rows(w):
    total=w.sum(1,keepdims=True)
    return np.divide(w,total,out=np.zeros_like(w,dtype=float),where=total>0)


def ref_weights(ok,desc,mass,depth):
    if not depth:return norm_rows(ok*mass)
    ranks=desc[:,3]*depth
    fav=np.where(ok,ranks[None,:],np.inf).min(1,keepdims=True)
    return norm_rows(mass*ok*(ranks[None,:]==fav))


def coexist(t,r,same,side,center):
    pos,low,high=r.view[:,1],r.view[:,0],r.view[:,2]
    if center:
        pos=low=high=np.full(len(pos),.5)
    dx=side*(t.view[:,1,None]-pos)
    okay=t.view[:,2,None]<low if side<0 else t.view[:,0,None]>high
    return np.where(okay & ~same & (dx>0),dx,np.inf)


def intersection(lt,t,lr,r,labels):
    if lt==lr:return t.index[:,None]==r.index
    # The cuts are nested. A smaller component has a unique parent in the larger cut.
    lower,higher=sorted((lt,lr))
    child,parent=labels[lower],labels[higher]
    valid=child>=0
    belong=np.full(int(child.max())+1,-1,int)
    belong[child[valid]]=parent[valid]
    if lt<lr:return belong[t.index][:,None]==r.index
    return t.index[:,None]==belong[r.index]


def probability(t,r,overlap,side,depth,immediate,arm,draws):
    mass=np.zeros((len(t.index),len(r.index)),dtype=float)
    if not mass.size:return mass
    tok=t.observed[draws]
    rok=(r.independent if arm=='separate_worlds' else r.observed)[draws]
    if arm=='wrong_reference' and len(r.index)>1:
        order=np.lexsort((r.index,r.view[:,1]));dr=np.roll(order,len(order)//2)
        misplaced=rok.copy();misplaced[:,order]=rok[:,dr];rok=misplaced
    ref=ref_weights(rok,r.view,r.prior,depth)
    gap=coexist(t,r,overlap,side,arm=='image_center')
    for j in np.flatnonzero(ref.any(0)):
        available=np.flatnonzero(np.isfinite(gap[:,j]))
        if not len(available):continue
        if immediate:
            # More than one component at exactly the same horizontal distance shares the vote.
            distance=gap[available,j]
            nearest=np.min(np.where(tok[:,available],distance,np.inf),axis=1)
            mask=tok[:,available] & (nearest[:,None]==distance)
        else:mask=tok[:,available]
        subject=norm_rows(mask*t.prior[available])
        mass[available,j]=ref[:,j]@subject/len(draws)
    return mass


def marginal_roles(levels,depth,nt,nr):
    s=np.zeros(nt);r=np.zeros(nr)
    for t,ref in levels:
        if len(t.index):s+=t.assign@norm_rows(t.observed*t.prior).mean(0)/8
        if len(ref.index):r+=ref.assign@ref_weights(ref.observed,ref.view,ref.prior,depth).mean(0)/8
    return s,r


def infer(tables,labels,side,depth,immediate,arm,nt,nr):
    """Return anchor beliefs and the same shared assignments used for shape measurements."""
    total=np.zeros((nt,nr));parts=[]
    for lt,(t,_) in enumerate(tables):
        refs=range(8) if arm=='separate_scales' else (lt,)
        for lr in refs:
            r=tables[lr][1]
            draws=np.arange(128)
            if arm=='separate_scales':draws=draws[lr::8]
            overlap=intersection(lt,t,lr,r,labels)
            mass=probability(t,r,overlap,side,depth,immediate,arm,draws)/(8*len(refs))
            total+=t.assign@mass@r.assign.T
            if arm=='joint_raw':parts.append((lt,t,r,mass))
    return total,parts
