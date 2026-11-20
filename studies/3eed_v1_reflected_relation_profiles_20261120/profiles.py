"""Aligned subject-reference evidence and the exact interval projection of its spatial profiles."""
import numpy as np
from scipy.special import softmax

from roles_joint import affinity, direction_tensor

ARMS = ('v1','paired_average','all_average','weighted_boxes','subject_only','independent_roles',
        'wrong_reference','reverse_edge','joint_raw','fixed_pair_geometry','joint_profile')


def contrast(positive, negative):
    return np.divide(2*positive**2, positive+negative, out=np.zeros_like(positive),
                     where=(positive+negative)>0)


def pool_sign(values, mode):
    pos = (values[0]+values[1])/2
    neg = (values[2]+values[3])/2
    if mode == 'paired_average':
        return pos
    if mode == 'all_average':
        return (pos+neg)/2
    if mode == 'reverse_edge':
        return contrast(neg,pos)
    return contrast(pos,neg)


def intervals(anchor, boxes):
    """Orthogonal projections of each measured cuboid onto the chosen box's physical axes."""
    angle=anchor[6]
    axes=np.array([[np.cos(angle),np.sin(angle),0.],[-np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    center=(boxes[...,:3]-anchor[:3])@axes.T
    yaw=boxes[...,6]-angle
    c,s=np.abs(np.cos(yaw)),np.abs(np.sin(yaw))
    ll,ww,hh=boxes[...,3],boxes[...,4],boxes[...,5]
    radii=np.stack((ll*c+ww*s,ll*s+ww*c,hh),axis=-1)/2
    return axes,center-radii,center+radii


def project_axis(low,high,weight,original_length,mode):
    """Maximize the normalized integral of a piecewise constant, nonnegative profile."""
    raw=np.array([-original_length/2,original_length/2])
    edges=np.unique(np.r_[low.ravel(),high.ravel(),raw,0.])
    edges.sort()
    mid=(edges[:-1]+edges[1:])/2
    inside=(low[...,None]<mid)&(mid<high[...,None])
    cell=np.einsum('vk,vkm->vm',weight/np.sqrt(high-low),inside)
    profile=pool_sign(cell,mode)
    if not np.any(profile):
        return raw
    integral=np.r_[0.,np.cumsum(profile*np.diff(edges))]
    # Every interval contains (at least at the boundary) this hypothesis's original center.
    left,right=np.where((edges[:,None]<=0)&(edges[None,:]>=0)&(edges[None,:]>edges[:,None]))
    scores=(integral[right]-integral[left])/np.sqrt(edges[right]-edges[left])
    inds=np.searchsorted(edges,raw)
    before=(integral[inds[1]]-integral[inds[0]])/np.sqrt(original_length)
    top=float(scores.max())
    tol=64*np.finfo(float).eps*max(top,abs(before),np.finfo(float).tiny)
    if before>=top-tol:
        return raw
    tied=np.flatnonzero(scores>=top-tol)
    distance=np.square(edges[left[tied]]-raw[0])+np.square(edges[right[tied]]-raw[1])
    winner=tied[np.argmin(distance)]
    return edges[[left[winner],right[winner]]]


def project_box(anchor, boxes, weight, mode):
    axes,lo,hi=intervals(anchor,boxes)
    faces=np.stack([project_axis(lo[...,k],hi[...,k],weight,anchor[3+k],mode) for k in range(3)])
    new=np.r_[anchor[:3]+faces.mean(1)@axes,faces[:,1]-faces[:,0],anchor[6]]
    assert np.isfinite(new).all() and (new[3:6]>0).all()
    return new


def whole_box(anchor, boxes, weights):
    if weights.sum()<=0:
        return anchor.copy()
    w=weights.ravel()/weights.sum()
    flat=boxes.reshape(-1,7)
    center=w@flat[:,:3]
    height=np.exp(w@np.log(flat[:,5]))
    xx,xy,yy=w@direction_tensor(flat)
    values,vectors=np.linalg.eigh([[xx,xy],[xy,yy]])
    if min(values)<=0:
        return anchor.copy()
    yaw=np.arctan2(vectors[1,1],vectors[0,1]) if np.ptp(values)>1e-12*max(values) else anchor[6]
    return np.r_[center,np.sqrt(values[::-1]),height,(yaw+np.pi/2)%np.pi-np.pi/2]


def predict(subject,reference,joint,original_choice,active,original_ref_valid):
    before,before_ref=map(int,original_choice)
    original=subject[0,before]
    native_ref=reference[0,before_ref] if original_ref_valid else np.full(7,np.nan)
    raw=np.repeat(original[None],len(ARMS),axis=0)
    boxes=raw.copy()
    refs=np.repeat(native_ref[None],len(ARMS),axis=0)
    choice=np.tile(np.array([before,before_ref]),(len(ARMS),1))
    if not active:
        return dict(raw=raw,boxes=boxes,refs=refs,choice=choice)
    subjects=subject[:2].reshape(8,7)
    references=reference[:2].reshape(8,7)
    ks=np.stack([affinity(subjects,b) for b in subject])
    kr=np.stack([affinity(references,b) for b in reference])
    p=softmax(joint.reshape(4,16),axis=1).reshape(4,4,4)
    np.testing.assert_array_equal(np.argmax(p[0].sum(1)),before)
    for n,mode in enumerate(ARMS[1:],1):
        probability=p
        ref_kernel=kr
        if mode=='independent_roles':
            probability=p.sum(2,keepdims=True)*p.sum(1,keepdims=True)
        elif mode=='wrong_reference':
            probability=p.copy()
            probability[1:]=np.roll(probability[1:],1,axis=2)
        elif mode=='subject_only':
            ref_kernel=np.ones_like(kr)
        votes=np.einsum('vik,vkl,vjl->vijk',ks,probability,ref_kernel,optimize=True)
        fields=votes.sum(-1)
        scores=pool_sign(fields,mode)
        i,j=np.unravel_index(scores.argmax(),scores.shape)
        if mode=='fixed_pair_geometry':
            i,j=before,before_ref
        anchor=subjects[i]
        choice[n]=[i,j]
        raw[n]=anchor
        refs[n]=references[j] if mode!='subject_only' else np.nan
        if mode=='joint_raw':
            boxes[n]=anchor
        elif mode=='weighted_boxes':
            # Use all four cells to weight each positive pair before ordinary box averaging.
            den=fields.sum(0)
            purity=np.divide(2*fields[:2].sum(0),den,out=np.zeros_like(den),where=den>0)
            compatible=np.stack([purity[4*v:4*v+4,4*v:4*v+4] for v in range(2)])
            weights=np.einsum('vk,vkl,vl,vkl->vk',ks[:2,i],p[:2],kr[:2,j],compatible)
            boxes[n]=whole_box(anchor,subject[:2],weights)
        else:
            boxes[n]=project_box(anchor,subject,votes[:,i,j],mode)
    return dict(raw=raw,boxes=boxes,refs=refs,choice=choice)
