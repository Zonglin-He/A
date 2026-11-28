"""Generate and describe the reduced pool without materializing v2-only features."""
from math import pi
from unittest.mock import patch

import cv2
import numpy as np
import torch

import compact_pool
from compact_pool import box_key, pca_heading, projected
from descriptors import iou2
from geometry_models import exact_iou
from physical_readout import anchored, axes, coordinates, silhouette_polygon


def reduced_pool(case,frame,replies,cap):
    import minimal
    from cray_simple import anchor,horizontal_axes

    g = minimal.geometry
    stages = {}

    def capture(sample,terrain,scales,returns,window,selection=None):
        regions = compact_pool.ranked_regions(sample,terrain,scales,returns,window,selection)
        stages['H' if selection is None else 'C_ray'] = regions
        return (regions[0].points,regions[0].selection) if regions else (None,None)

    with patch.object(minimal,'choose_region',capture):
        base,context = minimal.predict(frame,replies,audit=True)
    assert base==case['base']
    pool = [base['box']]
    if base['box'] is None or not case['first_window']:
        return pool,case

    support = coordinates(case['measured'])
    regions = stages.get(base['readout'],[])
    primary = regions[0] if regions else None
    if primary is None and len(support)<3:
        return pool,case

    spec,_,transform = g.camera(frame)
    camera = g.model_to_camera_matrix(spec)
    eye = np.linalg.inv(camera)[:3,3]
    prior = np.asarray(case['prior'])
    upright,_ = g.to_sensor(np.asarray(replies['3d']['value']),transform)
    span = upright[3:6] if primary is not None else prior
    if primary is not None:
        pca = pca_heading(primary.points,primary.ground,span)
        qwen = upright[6]
        if frame['target']=='waymo':
            qwen += np.deg2rad(g.WAYMO_ANGLES[frame['suffix']])
    else:
        qwen = case['yaw']
        centered = support[:,:2]-support[:,:2].mean(0)
        _,vectors = np.linalg.eigh(centered.T@centered)
        pca = np.arctan2(vectors[1,-1],vectors[0,-1])

    def reference(region,yaw,size):
        if region is None:
            return anchored(support,np.asarray(case['eye']),np.asarray(case['initial']),size,yaw,0.)
        local = np.c_[region.points[:,:2]@horizontal_axes(yaw).T,region.points[:,2]]
        box = anchor(local.min(0),local.max(0),size,yaw,eye,region.ground)
        return g.sensor_frame_box(box,frame['target'],frame['suffix'])

    observed = case
    if context is not None:
        ev = context[0]
        observed = dict(case,measured=ev.support_sensor,selection=ev.selection,audit=[])
    else:
        observed = dict(case,audit=[])
    # Keep the first geometric representation even if a later retained
    # hypothesis coincides with an omitted, unprojected reference.
    first = {box_key(base['box']): [base['box'],True]}

    def add(box,active=True):
        chosen = first.setdefault(box_key(box),[box.tolist(),False])
        chosen[1] |= active

    stats = {'selection':[0,0],'audit':[0,0]}
    for yaw in (pca,pca+pi/2,qwen,qwen+pi/2):
        ref = reference(primary,yaw,span)
        add(ref,cap==17)
        add(projected(ref,observed,stats))
    for region in regions[1:]:
        for yaw in (pca_heading(region.points,region.ground,span),qwen):
            add(reference(region,yaw,span))
    for yaw in (pca,qwen):
        for scale in (.85,1.15):
            add(reference(primary,yaw,prior*scale))
    pool = [value for value,active in first.values() if active]
    assert len(pool)<=cap and stats['selection'][1]==0
    if regions:
        union = np.unique(np.concatenate([r.points for r in regions]),axis=0)
        case = dict(case,measured=g.rigid_xyz(union,np.linalg.inv(transform)@camera))
    return pool,case


def points4(b,frame,points):
    local = ((points-b[:3])@frame)/b[3:6]
    radius = np.max(abs(local),axis=1)
    close = radius<=.75
    if not close.any():
        return np.zeros(4)
    inside = radius[close]<=.50
    shell = inside&(radius[close]>=.38)
    return np.r_[inside.mean(),shell.mean(),np.ptp(local[close,:2],axis=0)]


def rays3(b,frame,rays,eye,depth):
    if not len(rays):
        return np.zeros(3)
    directions = rays@frame
    center = (eye-b[:3])@frame
    half = b[3:6]/2
    parallel = abs(directions)<=1.e-12
    denom = np.where(parallel,1.,directions)
    left,right = (-half-center)/denom,(half-center)/denom
    near = np.where(parallel,-np.inf,np.minimum(left,right)).max(1)
    far = np.where(parallel,np.inf,np.maximum(left,right)).min(1)
    hit = (near>0)&(far>=near)&~(parallel&(abs(center)>half)).any(1)
    if not hit.any():
        return np.zeros(3)
    scale = max(.10,.5*min(b[3:5]))
    residual = abs(near[hit]-depth[hit])/scale
    huber = np.where(residual<=1.,.5*residual**2,residual-.5)
    return np.r_[hit.mean(),huber.mean(),(hit&(depth>far+.15*scale)).mean()]


def image3(b,case):
    hull = silhouette_polygon(b,case)
    if hull is None or len(hull)<3:
        return np.zeros(3)
    full = np.r_[hull.min(0),hull.max(0)]
    rect = np.clip(full,0,np.tile(case['size'],2))
    if np.min(rect[2:]-rect[:2])<2:
        return np.zeros(3)
    fill = cv2.contourArea(hull)/max(np.prod(full[2:]-full[:2]),1.e-9)
    if not case['first_window']:
        return np.array([0.,0.,fill])
    window = np.asarray(case['window'])
    return np.r_[iou2(rect,window),np.log(np.prod(rect[2:]-rect[:2])/np.prod(window[2:]-window[:2])),fill]


def describe(case,pool):
    length = len(pool)
    if pool[0] is None:
        assert length==1
        return np.zeros((1,14),np.float32),np.zeros((1,1,8),np.float32)
    boxes = np.asarray(pool)
    transform = np.linalg.inv(case['transform'])
    eye = transform[:3,3]
    returns = coordinates(case['selection'])
    rays = (returns/returns[:,2:])@transform[:3,:3].T
    support = coordinates(case['measured'])
    frames = np.asarray([axes(b[6]) for b in boxes])
    nodes = np.asarray([np.r_[points4(b,frame,support),rays3(b,frame,rays,eye,returns[:,2]),
                             image3(b,case),np.log(b[3:6]/case['prior']),np.cos(2*(b[6]-case['yaw']))]
                        for b,frame in zip(boxes,frames,strict=True)])
    raw = torch.as_tensor(boxes,dtype=torch.float64)
    iou = exact_iou(raw.repeat_interleave(length,0),raw.repeat(length,1)).reshape(length,length).numpy()
    center,span,yaw = boxes[:,:3],boxes[:,3:6],boxes[:,6]
    local = ((center[None]-center[:,None])@frames)/span[:,None]
    edges = np.concatenate((iou[...,None],local,np.log(span[None]/span[:,None]),
                            np.cos(2*(yaw[None]-yaw[:,None]))[...,None]),axis=-1)
    return nodes.astype(np.float32),edges.astype(np.float32)


class MinimalVerifier:
    """Two-stage predictor consuming frozen replies on calibrated sensor inputs."""
    def __init__(self,checkpoint,cap,threshold,device='cuda'):
        from safetensors.torch import load_file
        from models import Arm,model_for

        if cap not in (13,17):
            raise ValueError('Expected the 13- or 17-candidate geometry')
        self.cap = cap
        self.threshold = threshold
        self.device = device
        self.model = model_for(Arm(cap,True,True,True))
        self.model.load_state_dict(load_file(checkpoint))
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self,frame,replies):
        import minimal
        from prepare_blind import prepare

        base = minimal.predict(frame,replies)
        row = dict(frame=frame,raw=replies,target=frame['target'])
        case = prepare((frame['uid'],row,dict(outputs=dict(axis_2d=base))))
        pool,case = reduced_pool(case,frame,replies,self.cap)
        if len(pool)==1 or self.threshold is None:
            return base
        x,e = describe(case,pool)
        k = len(pool)
        nodes = np.zeros((1,self.cap,14),np.float32)
        edges = np.zeros((1,self.cap,self.cap,8),np.float32)
        nodes[0,:k],edges[0,:k,:k] = x,e
        mask = (np.arange(self.cap)<k)[None]
        tensor = (torch.as_tensor(a,device=self.device) for a in (nodes,edges,mask))
        scores = self.model(*tensor)[0,1:k]
        value,index = scores.max(0)
        return dict(base,box=pool[int(index)+1],verifier_replaced=True) if value>self.threshold else base
