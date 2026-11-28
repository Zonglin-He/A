"""Bounded candidate descriptors made solely from predictions and observations."""
import cv2
import numpy as np
from scipy.stats import rankdata
import torch

from compact_pool import CAPACITY
from geometry_models import exact_iou
from physical_readout import axes, coordinates, silhouette_polygon

NODE_FIELDS = (
    'inside','halo','shell','core','span_x','span_y','span_z',
    'near_q10','near_q50','near_q90','vertical_q10','vertical_q50','vertical_q90','view_cosine',
    'hit','ray_median','ray_huber','ray_bias','pass','front',
    'image_iou','edge_left','edge_top','edge_right','edge_bottom','area_log','hull_fill',
    'prior_log_l','prior_log_w','prior_log_h','prior_cos2','prior_sin2',
    *(f'{kind}_{name}' for kind in ('z','rank','anchor_diff') for name in ('image_iou','ray_huber','shell','pass')),
    'anchor','primary',
)
EDGE_FIELDS = ('iou3d','bev','image_iou','image_3d_gap','shift_l','shift_w','shift_h','radial_delta',
               'log_l','log_w','log_h','yaw_cos','yaw_sin','yaw_cos2')


def point_stats(b, frame, pts, camera):
    local = ((pts-b[:3]) @ frame) / b[3:6]
    radius = np.max(abs(local), axis=1)
    local = local[radius <= .75]
    radius = radius[radius <= .75]
    eye = (camera-b[:3]) @ frame
    view = eye/max(np.linalg.norm(eye),1.e-9)
    face = int(np.argmax(abs(eye[:2])/b[3:5]))
    cosine = abs(view[face])
    if not len(local):
        return np.r_[np.zeros(13),cosine]
    inside = radius <= .5
    inner = local[inside]
    q = np.array([.1,.5,.9])
    near = .5 - np.sign(eye[face])*inner[:,face]
    vertical = .5 - np.where(eye[2] >= 0.,1.,-1.)*inner[:,2]
    quantiles = np.r_[np.quantile(near,q),np.quantile(vertical,q)] if len(inner) else np.zeros(6)
    return np.r_[inside.mean(), ((radius > .5) & (radius <= .6)).mean(),
                 ((radius >= .38) & inside).mean(), (radius <= .25).mean(),
                 np.ptp(local,axis=0), quantiles, cosine]


def ray_stats(b, frame, rays, eye, observed):
    if not len(rays):
        return np.zeros(6)
    directions = rays @ frame
    local = (eye-b[:3]) @ frame
    half = b[3:6]/2
    parallel = abs(directions) <= 1.e-12
    safe = np.where(parallel,1.,directions)
    left,right = (-half-local)/safe,(half-local)/safe
    near = np.where(parallel,-np.inf,np.minimum(left,right)).max(1)
    far = np.where(parallel,np.inf,np.maximum(left,right)).min(1)
    hit = (near > 0) & (far >= near) & ~(parallel & (abs(local) > half)).any(1)
    if not hit.any():
        return np.zeros(6)
    scale = max(.10,.5*min(b[3:5]))
    diff = (near[hit]-observed[hit])/scale
    magnitude = abs(diff)
    huber = np.where(magnitude<=1., .5*magnitude**2, magnitude-.5)
    return np.array([hit.mean(), np.median(magnitude), huber.mean(), diff.mean(),
                     np.count_nonzero(hit & (observed>far+.15*scale))/len(hit),
                     np.count_nonzero(hit & (observed<near-.15*scale))/len(hit)])


def iou2(a, b):
    both = np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0.).prod(-1)
    area = np.maximum(a[...,2:]-a[...,:2],0.).prod(-1)+np.maximum(b[...,2:]-b[...,:2],0.).prod(-1)
    return both/np.maximum(area-both,1.e-9)


def projection(b, case):
    hull = silhouette_polygon(b, case)
    image_stats = np.zeros(7)
    rect = np.zeros(4)
    if hull is not None and len(hull) >= 3:
        full = np.r_[hull.min(0),hull.max(0)]
        rect = np.clip(full,0,np.tile(case['size'],2))
        if np.min(rect[2:]-rect[:2]) < 2.:
            rect = np.zeros(4)
        else:
            area = np.prod(full[2:]-full[:2])
            image_stats[6] = cv2.contourArea(hull)/max(area,1.e-9)
            if case['first_window']:
                window = np.asarray(case['window'])
                size = window[2:]-window[:2]
                image_stats[0] = iou2(rect,window)
                image_stats[1:5] = (rect-window)/np.tile(size,2)
                image_stats[5] = np.log(np.prod(rect[2:]-rect[:2])/np.prod(size))
    yaw = 2*(b[6]-case['yaw'])
    return rect, np.r_[image_stats,np.log(b[3:6]/case['prior']),np.cos(yaw),np.sin(yaw)]


def overlaps(boxes):
    k = len(boxes)
    a = torch.as_tensor(boxes, dtype=torch.float64)
    bev = a.clone()
    bev[:,2] = 0.
    bev[:,5] = 1.
    paired = torch.cat((a.repeat_interleave(k,dim=0),bev.repeat_interleave(k,dim=0)))
    other = torch.cat((a.repeat(k,1),bev.repeat(k,1)))
    return exact_iou(paired,other).reshape(2,k,k).numpy()


def edge_features(boxes, frames, rects, eye):
    iou3, bev = overlaps(boxes)
    image = iou2(rects[:,None],rects[None,:])
    center = boxes[:,:3]
    span = boxes[:,3:6]
    offsets = ((center[None]-center[:,None])@frames)/span[:,None]
    depth = np.linalg.norm(center-eye,axis=-1)
    radial = (depth[None]-depth[:,None])/span[:,:2].max(1)[:,None]
    log_span = np.log(span[None]/span[:,None])
    yaw = boxes[None,:,6]-boxes[:,None,6]
    return np.concatenate((np.stack((iou3,bev,image,image-iou3),axis=-1), offsets, radial[...,None], log_span,
                           np.stack((np.cos(yaw),np.sin(yaw),np.cos(2*yaw)),axis=-1)),axis=-1)


def describe(case, pool):
    nodes = np.zeros((CAPACITY,len(NODE_FIELDS)),np.float32)
    edges = np.zeros((CAPACITY,CAPACITY,len(EDGE_FIELDS)),np.float32)
    valid = np.zeros(CAPACITY,bool)
    valid[:len(pool)] = True
    nodes[0,-2] = 1.
    if pool[0]['box'] is None:
        assert len(pool) == 1
        return nodes,edges,valid
    k = len(pool)
    boxes = np.asarray([r['box'] for r in pool])
    frames = np.asarray([axes(b[6]) for b in boxes])
    pts = coordinates(case['measured'])
    returns = coordinates(case['selection'])
    inverse = np.linalg.inv(case['transform'])
    eye = inverse[:3,3]
    rays = (returns/returns[:,2:]) @ inverse[:3,:3].T
    stats, rects = [], []
    for b,frame in zip(boxes,frames,strict=True):
        rect,image = projection(b,case)
        stats.append(np.r_[point_stats(b,frame,pts,eye),ray_stats(b,frame,rays,eye,returns[:,2]),image])
        rects.append(rect)
    stats = np.asarray(stats)
    # Equal values receive an average percentile, independent of candidate order.
    s = stats[:,[20,16,2,18]]
    relative = np.c_[(s-s.mean(0))/(s.std(0)+1.e-8),rankdata(s,method='average',axis=0)/k,s-s[0]]
    tags = [(i == 0, case['base']['readout'] in ('C_ray','H') and
             any(not s.startswith('regional/') for s in row['sources'])) for i,row in enumerate(pool)]
    nodes[:k] = np.c_[stats,relative,tags]
    edges[:k,:k] = edge_features(boxes,frames,np.asarray(rects),eye)
    assert nodes.shape[-1] == 46 and edges.shape[-1] == 14
    assert np.isfinite(nodes).all() and np.isfinite(edges).all()
    return nodes,edges,valid
