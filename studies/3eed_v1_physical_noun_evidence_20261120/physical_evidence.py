"""Word comparisons and local return distributions in common physical coordinates."""
import numpy as np
from scipy.spatial import cKDTree
from batch_yaw_iou import many_iou
from ordered_choice import relations
from roles_joint import affinity


def axes(box):
    c,s=np.cos(box[6]),np.sin(box[6])
    return np.array([[c,s,0],[-s,c,0],[0,0,1.]])


class Returns:
    def __init__(self, points):
        self.xyz=np.unique(points.astype(float),axis=0)
        self.tree=cKDTree(self.xyz)

    def local(self, box):
        half=box[3:6]*.5
        ids=np.asarray(self.tree.query_ball_point(box[:3],np.linalg.norm(half)+1.e-7),int)
        off=(self.xyz[ids]-box[:3]) @ axes(box).T
        inside=(abs(off)<=half+1.e-7).all(1)
        return ids[inside],off[inside]


def lexicon(xyz, scores):
    seeds=Returns(xyz)
    _,inv=np.unique(xyz,axis=0,return_inverse=True)
    centered=scores.astype(float)-scores.mean(-1,keepdims=True,dtype=float)
    count=np.bincount(inv)
    averaged=[]
    for matrix in (centered.mean(0),centered[0]):
        out=np.zeros((len(count),matrix.shape[1]))
        np.add.at(out,inv,matrix)
        averaged.append(out/count[:,None])
    return seeds,averaged[0],averaged[1]


def box_classes(boxes, seeds, scores, ids=None):
    if ids is None:
        ids=[seeds.local(b)[0] for b in boxes]
    logits=np.zeros((len(boxes),scores.shape[1]))
    available=np.array([len(v)>0 for v in ids])
    for i in np.flatnonzero(available):
        logits[i]=scores[ids[i]].mean(0)
    labels=np.where(available,logits.argmax(1),-1)
    return labels,ids


def choose(T,R,tp,rp,mt,mr,dist,mt_ok,mr_ok,depth,immediate,anchor,independent=False):
    ti,ri=np.flatnonzero(mt_ok),np.flatnonzero(mr_ok)
    if not len(ti) or not len(ri):
        return None
    if depth:
        z=depth*rp[ri,1]
        ri=ri[z==z.min()]
    keep=dist[np.ix_(ti,ri)]
    if independent:
        feasible=np.ones(keep.shape,bool)
    else:
        feasible=np.isfinite(keep)
        if immediate:
            feasible &= (keep==keep.min(0,keepdims=True))
    ti_active=feasible.any(1);ri_active=feasible.any(0)
    if not ti_active.any():return None
    J=(mt[ti,None]*mr[None,ri])*feasible
    ti,ri=ti[ti_active],ri[ri_active]
    J=J[np.ix_(ti_active,ri_active)]
    overlap=many_iou(T[ti],anchor)
    if overlap.max()>.25:
        subject=anchor
        rows=overlap*(overlap>.25)
    else:
        scores=affinity(T[ti],T[ti]) @ J.sum(1)
        i=int(np.argmax(scores))
        subject=T[ti[i]]
        rows=np.zeros(len(ti));rows[i]=1.
    j=int(np.argmax(affinity(R[ri],R[ri]) @ (rows @ J)))
    return subject,R[ri[j]]


def complete(anchor, candidates, subject_class, seeds, normal, points, point_class, donor_mode='physical'):
    """Minimize sum of absolute donor span errors under the observed-face constraints."""
    out=anchor.copy()
    symmetric=anchor.copy()
    sides=np.zeros(2,int);delta=np.zeros(2,float);support=np.zeros(2,int)
    if subject_class<0:return out,symmetric,sides,delta,support
    idx,local=points.local(anchor)
    local=local[point_class[idx]==subject_class]
    if len(local)<2:return out,symmetric,sides,delta,support
    for axis in range(2):
        u=local[:,axis]
        if np.ptp(u)>0 and (u.min()>0 or u.max()<0):sides[axis]=1 if u.min()>0 else -1
    if not sides.any():return out,symmetric,sides,delta,support
    matches,_=box_classes(candidates,seeds,normal)
    donor_ids=np.flatnonzero((many_iou(candidates,anchor)==0) & ((matches==subject_class) if donor_mode!='blind' else True))
    # Candidates arrive in descending original full-sentence support with canonical geometry ties.
    chosen=[];spans=[[],[]]
    if donor_mode=='blind':donor_point_class=None
    elif donor_mode=='wrong':donor_point_class=normal.argmax(1)[points.nearest_seed]
    else:donor_point_class=point_class
    for k in donor_ids:
        donor=candidates[k]
        if chosen and many_iou(np.stack(chosen),donor).max()>0:continue
        ids,off=points.local(donor)
        if donor_point_class is not None:off=off[donor_point_class[ids]==subject_class]
        if len(off)<2:continue
        donor_order=np.argsort(-donor[3:5],kind='stable')
        anchor_order=np.argsort(-anchor[3:5],kind='stable')
        usable=False
        for ax,da in zip(anchor_order,donor_order):
            u=off[:,da]
            if sides[ax] and u.min()<0<u.max():
                spans[ax].append(float(np.ptp(u)));usable=True
        if usable:chosen.append(donor)
    direction=axes(anchor)
    for ax,s in enumerate(sides):
        support[ax]=len(spans[ax])
        if not support[ax]:continue
        delta[ax]=max(0.,np.median(spans[ax])-anchor[3+ax])
        out[3+ax]+=delta[ax];symmetric[3+ax]+=delta[ax]
        out[:3]-=s*direction[ax]*delta[ax]/2
    return out,symmetric,sides,delta,support
