"""A role-exchange comparison over physical boxes, and its conditional geometry estimate."""
import numpy as np
from scipy.special import softmax

from batch_yaw_iou import many_iou


ARMS=('v1','subject_overlap','positive_product','independent_roles','wrong_reference',
      'ordered_raw','ordered_frozen_subject','ordered_full')


def affinity(left,right):
    """L2 inner product of square roots of the two uniform box densities."""
    overlap=np.stack([many_iou(right,b) for b in left])
    a,b=np.prod(left[:,3:6],axis=1)[:,None],np.prod(right[:,3:6],axis=1)[None,:]
    common=overlap*(a+b)/(1+overlap)
    return np.clip(common/np.sqrt(a*b),0.,1.)


def direction_tensor(box):
    theta=box[:,6]
    c,s=np.cos(theta),np.sin(theta)
    ll,ww=box[:,3]**2,box[:,4]**2
    return np.stack((ll*c*c+ww*s*s,(ll-ww)*c*s,ll*s*s+ww*c*c),axis=1)


def fuse(base,other,mass,alternative):
    if not np.any(alternative>0):
        return base.copy()
    shapes=np.vstack((base,other))
    weights=np.r_[mass,alternative]
    weights=weights/weights.sum()
    center=weights@shapes[:,:3]
    h=np.exp(weights@np.log(shapes[:,5]))
    xx,xy,yy=weights@direction_tensor(shapes)
    vals,axes=np.linalg.eigh([[xx,xy],[xy,yy]])
    if vals[0]<=0:
        raise ValueError('Nonpositive squared edge length')
    yaw=np.arctan2(axes[1,1],axes[0,1])
    if vals[1]-vals[0]<=1e-12*max(vals[1],1.):
        yaw=base[6]
    return np.r_[center,np.sqrt(vals[::-1]),h,(yaw+np.pi/2)%np.pi-np.pi/2]


def marginal_product(matrix):
    return np.outer(matrix.sum(1),matrix.sum(0))


def decide(original,s,r,conditional,sd,rd,reverse,mode):
    """Axis order is forward subject, forward reference, reverse real subject, reverse real reference."""
    before=int(np.argmax(original))
    if mode=='v1':
        return before,int(np.argmax(conditional[before])),s[before].copy(),0.,original.copy()
    forward=original[:,None]*conditional
    if mode=='independent_roles':
        forward=marginal_product(forward)
        reverse=marginal_product(reverse)
    if mode=='wrong_reference':
        rd=np.roll(rd,1,axis=0)
    ss=affinity(s,sd)[:,None,:,None]
    direct=ss*affinity(r,rd)[None,:,None,:]
    swap=affinity(s,rd)[:,None,None,:]*affinity(r,sd)[None,:,:,None]
    signed=direct-swap
    if mode=='subject_overlap':
        signed=np.broadcast_to(ss,signed.shape)
    elif mode=='positive_product':
        signed=direct
    prior=forward[:,:,None,None]*reverse[None,None,:,:]
    stay=prior*(1-np.maximum(-signed,0))
    move=prior*np.maximum(signed,0)
    joint=(stay+move).sum(axis=(2,3))
    marginal=joint.sum(1)
    if marginal.sum()<=1e-15:
        return before,int(np.argmax(conditional[before])),s[before].copy(),0.,original.copy()
    marginal=marginal/marginal.sum()
    choice=before if mode=='ordered_frozen_subject' else int(np.argmax(marginal))
    ref=int(np.argmax(joint[choice]))
    raw_mass=stay[choice].sum()
    new_mass=move[choice].sum(axis=(0,2))
    denom=raw_mass+new_mass.sum()
    if mode=='ordered_raw' or denom==0:
        box=s[choice].copy()
        ratio=0.
    else:
        box=fuse(s[choice],sd,raw_mass,new_mass)
        ratio=float(new_mass.sum()/denom)
    return choice,ref,box,ratio,marginal


def posterior(v1,forward_logits,reverse_logits):
    # Row marginals of the factual distribution come from the full natural sentence.
    return softmax(v1),softmax(forward_logits,axis=1),softmax(reverse_logits.ravel()).reshape(reverse_logits.shape).T
