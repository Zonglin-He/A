"""Three physical region generators with shared return admission and ranking."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

K = 4
NEIGHBORS = 8
TAU = 1.
DEPTH_GAP = .50
MIN_POINTS = 8
GROUND_HEIGHT = .20
ARMS = ('A_depth','B_adaptive','C_four')


def foreground(points,ground):
    return points[:,2]-points[:,:2]@ground[:2]-ground[2]>GROUND_HEIGHT


def accepted_components(labels):
    size = np.bincount(labels)
    good = size>=MIN_POINTS
    return np.where(good,np.cumsum(good)-1,-1)[labels]


def adaptive(sample,ground):
    """Keep edges from the symmetrized 8-NN graph with normalized length < 1."""
    ids = np.flatnonzero(foreground(sample,ground))
    labels = np.full(len(sample),-1,dtype=int)
    radii = np.zeros(len(sample),float)
    if len(ids)<=NEIGHBORS:
        return labels,radii
    points = sample[ids]
    dist,neighbors = cKDTree(points).query(points,k=NEIGHBORS+1)
    sigma = dist[:,K]
    a = np.repeat(np.arange(len(ids)),NEIGHBORS)
    b = neighbors[:,1:].ravel()
    length = dist[:,1:].ravel()
    retain = length < TAU*np.sqrt(sigma[a]*sigma[b])
    graph = coo_matrix((np.ones(retain.sum(),np.uint8),(a[retain],b[retain])),
                       shape=(len(ids),len(ids)))
    _,components = connected_components(graph,directed=False)
    labels[ids] = accepted_components(components)
    radii[ids] = TAU*sigma
    return labels,radii


def in_window_depth(returns,in_window,ground,optical):
    """One-dimensional depth runs composed of visible window Fit returns."""
    points = returns[:len(in_window)][in_window]
    points = points[foreground(points,ground)]
    if not len(points):
        return points,np.empty(0,int),np.empty(0,float)
    z = points@optical[:3]+optical[3]
    order = np.argsort(z,kind='stable')
    label = np.empty(len(points),int)
    label[order] = np.r_[0,(np.diff(z[order])>DEPTH_GAP).cumsum()]
    return points,accepted_components(label),np.full(len(points),DEPTH_GAP)


def single_map(seeds,labels,radius,returns):
    if not len(seeds):
        return np.full((1,len(returns)),-1,int)
    distance,nearest = cKDTree(seeds).query(returns)
    return np.where(distance<=radius[nearest],labels[nearest],-1)[None]


@dataclass
class Binding:
    regions: list
    members: list
    count: int
    admitted: int


def ranked(arm,sample,ground,scales,returns,in_window,selection,optical):
    from cray_simple import MIN_RETURNS, geometry, region_catalog
    from compact_pool import Region

    if not in_window.any():
        return Binding([],[],0,0)
    if arm=='C_four':
        cut,assign,birth = region_catalog(sample,ground,scales)
        seeds = sample
        def associate(points):
            return geometry.transfer(sample,points,cut,assign,birth)
    else:
        if arm=='B_adaptive':
            seeds = sample
            labels,radius = adaptive(sample,ground)
        else:
            assert arm=='A_depth'
            seeds,labels,radius = in_window_depth(returns,in_window,ground,optical)
        assign = labels[None]
        birth = np.zeros(int(labels.max(initial=-1))+1,int)
        def associate(points):
            return single_map(seeds,labels,radius,points)
    count = len(birth)
    if not count:
        return Binding([],[],0,0)
    votes = associate(returns)
    n = len(in_window)
    scene,central = votes[:,:n],votes[:,n:]
    inside = scene[:,in_window]
    exposure = np.bincount(scene[scene>=0],minlength=count)
    hits = np.bincount(inside[inside>=0],minlength=count)
    central_hits = np.bincount(central[central>=0],minlength=count)
    eligible = np.flatnonzero((central_hits>=MIN_RETURNS)&(hits>0))
    if not len(eligible):
        return Binding([],[],count,0)
    score = hits[eligible].astype(float)**2/exposure[eligible]
    winners = eligible[np.argsort(-score,kind='stable')[:3]]
    held = None if selection is None else associate(selection)
    regions = [Region(seeds[assign[birth[i]]==i],ground,None if held is None else held[birth[i]]==i)
               for i in winners]
    members = [np.flatnonzero(scene[birth[i]]==i).tolist() for i in winners]
    return Binding(regions,members,count,len(eligible))
