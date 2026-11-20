"""Exact integration of first qualifying physical hypotheses over a common semantic cutoff."""
import numpy as np

def ordered_mass(prob, order, cut=None):
    mass=np.zeros_like(prob,dtype=float)
    if len(order)==0:return mass
    p=prob[order]
    if cut is not None:
        eligible=np.flatnonzero(p>=cut)
        if len(eligible):mass[order[eligible[0]]]=1.
    else:
        previous=np.maximum.accumulate(np.r_[0.,p[:-1]])
        mass[order]=np.maximum(p-previous,0.)
    return mass

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

def relations(st,rf,side):
    dx=side*(st[:,0,None]-rf[:,0])
    separated=st[:,3,None]<rf[:,2] if side<0 else st[:,2,None]>rf[:,3]
    return np.where(separated & (dx>0),dx,np.inf)

def reference_mass(pr,mr,rp,depth,wrong,cut):
    if wrong and len(pr)>1:
        order=np.argsort(rp[:,0],kind='stable')
        shifted=np.roll(order,1)
        p=pr.copy();m=mr.copy()
        p[order]=pr[shifted];m[order]=mr[shifted]
        pr,mr=p,m
    if depth==0:return mr
    order=np.argsort(depth*rp[:,1],kind='stable')
    return ordered_mass(pr,order,cut)

def joint(pt,ref,dist,immediate,cut=None):
    joint_mass=np.zeros_like(dist)
    for i in np.flatnonzero(ref>0):
        ids=np.flatnonzero(np.isfinite(dist[:,i]))
        if not len(ids):continue
        if immediate:
            order=ids[np.argsort(dist[ids,i],kind='stable')]
            joint_mass[:,i]=ref[i]*ordered_mass(pt,order,cut)
        else:
            p=pt[ids];total=p.sum()
            if total>0:joint_mass[ids,i]=ref[i]*p*(p.max()/total)
    return joint_mass

def choices(joint_mass,subject_affinity,reference_affinity):
    if joint_mass.sum()<=0:return None
    i=int(np.argmax(subject_affinity@joint_mass.sum(1)))
    conditioned=subject_affinity[i] @ joint_mass
    j=int(np.argmax(reference_affinity @ conditioned))
    return i,j
