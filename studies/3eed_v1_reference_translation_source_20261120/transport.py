"""Translate a reverse-query box by the predicted drift of the corresponding reference."""
import numpy as np
from roles_joint import affinity, fuse


ARMS=('v1','no_translation','reference_translation','opposite_translation','unrelated_translation','raw_subject_only')


def translate_pool(base, shifts):
    boxes=np.broadcast_to(base[None,:,None,:],(*shifts.shape[:3],7)).copy()
    boxes[...,:3]+=shifts
    return boxes


def translate_readout(p,subjects,refs,condition,reverse_subjects,reverse_refs,reverse,arm):
    """Inputs and named tensor axes are the same as in the sealed reciprocal-readout experiment."""
    j,k,l=len(refs),len(reverse_subjects),len(reverse_refs)
    rr,rs=affinity(refs,reverse_refs),affinity(refs,reverse_subjects)
    anchors_r=reverse_refs
    anchors_s=reverse_subjects
    if arm=='unrelated_translation':
        anchors_r=np.roll(anchors_r,1,axis=0)
        anchors_s=np.roll(anchors_s,1,axis=0)
    sign={'no_translation':0.,'opposite_translation':-1.}.get(arm,1.)
    d=sign*(refs[:,None,:3]-anchors_r[None,:,:3])
    cross=sign*(refs[:,None,:3]-anchors_s[None,:,:3])
    direct_boxes=translate_pool(reverse_subjects,np.broadcast_to(d[:,None,:,:],(j,k,l,3)))
    flipped=translate_pool(reverse_refs,np.broadcast_to(cross[:,None,:,:],(j,l,k,3))).swapaxes(1,2)
    affinity_direct=affinity(subjects,direct_boxes.reshape(-1,7)).reshape(len(subjects),j,k,l)
    affinity_flipped=affinity(subjects,flipped.reshape(-1,7)).reshape(len(subjects),j,k,l)
    signed=affinity_direct*rr[None,:,None,:]-affinity_flipped*rs[None,:,:,None]
    prior=p[:,None,None,None]*condition[:,:,None,None]*reverse[None,None,:,:]
    keep=prior*(1-np.maximum(-signed,0))
    adjust=prior*np.maximum(signed,0)
    posterior=(keep+adjust).sum(axis=(1,2,3))
    if posterior.sum()<=1e-15:
        return int(p.argmax()),subjects[p.argmax()].copy(),p.copy()
    posterior/=posterior.sum()
    choice=int(posterior.argmax())
    fixed=subjects[choice]
    if arm=='raw_subject_only':
        return choice,fixed.copy(),posterior
    return choice,fuse(fixed,direct_boxes.reshape(-1,7),keep[choice].sum(),adjust[choice].ravel()),posterior
