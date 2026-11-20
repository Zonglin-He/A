"""Continuous center readout maximizing posterior mass above the localization IoU threshold."""
import numpy as np
from batch_yaw_iou import many_iou


def axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([np.stack((c,s),axis=-1), np.stack((-s,c),axis=-1)],axis=-2)


def feasible_translation(anchor, boxes, direction):
    """Separating-axis constraints for positive intersection, before imposing the IoU threshold."""
    org_axes, box_axes = axes(anchor[6]), axes(boxes[:,6])
    normals = np.concatenate([np.broadcast_to(org_axes,(len(boxes),2,2)), box_axes],axis=1)
    offset = np.einsum('nji,ni->nj',normals,boxes[:,:2]-anchor[:2])
    radii = abs(normals @ org_axes.T) @ (anchor[3:5]*.5)
    radii += np.einsum('njk,nk->nj',abs(normals @ box_axes.transpose(0,2,1)),boxes[:,3:5]*.5)
    slope = normals @ direction
    parallel = abs(slope)<1e-12
    valid = ~np.any(parallel & (abs(offset)>=radii),axis=1)
    den = np.where(parallel,1.,slope)
    first, second = (offset-radii)/den,(offset+radii)/den
    lo = np.where(parallel,-np.inf,np.minimum(first,second)).max(1)
    hi = np.where(parallel,np.inf,np.maximum(first,second)).min(1)
    valid &= (lo<hi) & (abs(boxes[:,2]-anchor[2])<(boxes[:,5]+anchor[5])*.5)
    return lo,hi,valid


def crossing_intervals(anchor, boxes, threshold=.5):
    direction = axes(anchor[6])[int(anchor[4]>anchor[3])]
    left,right,overlap = feasible_translation(anchor,boxes,direction)
    va, vb = np.prod(anchor[3:6]), np.prod(boxes[:,3:6],axis=1)
    overlap &= np.minimum(va,vb)>threshold*np.maximum(va,vb)
    idx = np.flatnonzero(overlap)
    if not len(idx):
        return idx, np.empty(0), np.empty(0), direction, np.empty(0,bool)
    group = boxes[idx].copy()
    def iou_at(t):
        moved = group.copy()
        moved[:,:2] -= t[:,None]*direction
        return many_iou(moved,anchor)
    lo,hi = left[idx].copy(),right[idx].copy()
    # Intersection volume is unimodal on a one-dimensional translation of convex boxes.
    for _ in range(45):
        m1, m2 = (2*lo+hi)/3, (lo+2*hi)/3
        f1,f2 = iou_at(m1),iou_at(m2)
        lo,hi = np.where(f1<=f2,m1,lo),np.where(f2<=f1,m2,hi)
    peak = (lo+hi)*.5
    good = iou_at(peak)>threshold
    idx,group,peak = idx[good],group[good],peak[good]
    if not len(idx):
        return idx, np.empty(0), np.empty(0), direction, np.empty(0,bool)
    lo,hi = left[idx].copy(),peak.copy()
    for _ in range(38):
        mid = (lo+hi)*.5
        inside = iou_at(mid)>threshold
        lo,hi = np.where(inside,lo,mid),np.where(inside,mid,hi)
    first = (lo+hi)*.5
    lo,hi = peak.copy(),right[idx].copy()
    for _ in range(38):
        mid = (lo+hi)*.5
        inside = iou_at(mid)>threshold
        lo,hi = np.where(inside,mid,lo),np.where(inside,hi,mid)
    last=(lo+hi)*.5
    # Two independently solved coincident roots must not create a fictitious optimal sliver.
    tol=2e-9*(1+np.maximum.reduce((abs(first),abs(last),np.maximum(group[:,3],group[:,4]))))
    first,last=first+tol,last-tol
    nonempty=first<last
    return idx[nonempty],first[nonempty],last[nonempty],direction,(many_iou(group[nonempty],anchor)>threshold)


def best_center(anchor, intervals, weights):
    ids,lo,hi,direction,at_zero = intervals
    w = weights[ids]
    valid = w>0
    if not np.any(valid):
        return anchor.copy(),0.
    lo,hi,w = lo[valid],hi[valid],w[valid]
    points,event = np.unique(np.r_[lo,hi],return_inverse=True)
    delta = np.bincount(event,weights=np.r_[w,-w],minlength=len(points))
    mass = np.cumsum(delta)[:-1]
    best = mass.max()
    zero = w[at_zero[valid]].sum()
    tol = 1.e-12*w.sum()
    if best-zero<=tol:
        return anchor.copy(),0.
    mids = ((points[:-1]+points[1:])*.5)[mass>=best-tol]
    t = mids[np.argmin(abs(mids))]
    result = anchor.copy()
    result[:2] += t*direction
    return result,float(t)
