"""Candidate geometry, silhouette lifting, and posterior IoU decoding."""
from dataclasses import dataclass
from itertools import product
from math import pi
from unittest.mock import patch

import cv2
import numpy as np
import torch

from geometry_models import exact_iou, ray_frame, scene_for


SIGNS=np.asarray(list(product((-.5,.5),repeat=3)))
LATTICE_DIMS=np.array([(l,w,h) for l,w,h in product((.82,1.18),(.85,1.15),(.85,1.15))]+[(1.,1.,1.)]*8)
LATTICE_QUANTILES=np.r_[[.10]*8,[.0,.06,.12,.18,.0,.06,.12,.18]]
LATTICE_SHIFT=np.r_[[0.]*12,[.20]*4]


def angle(x):
    return (x+pi/2)%pi-pi/2


def axes(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array(((c,-s,0.),(s,c,0.),(0.,0.,1.)))


def coordinates(values):
    return np.asarray(values,float).reshape(-1,3)


def silhouette_polygon(box, case):
    box=np.asarray(box,float)
    corners=SIGNS*box[3:6]@axes(box[6]).T+box[:3]
    matrix=np.asarray(case['transform'],float)
    camera=corners@matrix[:3,:3].T+matrix[:3,3]
    if (camera[:,2]<.1).any():
        return None
    pixels=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),np.asarray(case['K']),np.asarray(case['dist']))[0][:,0]
    if not np.isfinite(pixels).all():
        return None
    return cv2.convexHull(pixels.astype(np.float32)).reshape(-1,2)


def rectangle_iou(box, case, window=None):
    window=case['window'] if window is None else window
    if window is None:
        return 0.
    poly=silhouette_polygon(box,case)
    if poly is None:
        return 0.
    rect=np.clip(np.r_[poly.min(0),poly.max(0)],0,np.tile(case['size'],2))
    if (rect[2:]-rect[:2]).min()<2:
        return 0.
    b=np.asarray(window)
    intr=np.maximum(0.,np.minimum(rect[2:],b[2:])-np.maximum(rect[:2],b[:2])).prod()
    union=np.prod(rect[2:]-rect[:2])+np.prod(b[2:]-b[:2])-intr
    return float(intr/max(union,1.e-8))


class Silhouette:
    def __init__(self, record, case):
        self.field=np.asarray(record['mask'],np.float32)
        self.case=case
        h,w=self.field.shape
        self.shape=(3*h,3*w)
        self.mask=np.repeat(np.repeat(self.field,3,0),3,1)
        self.mass=float(self.mask.sum())
        self.pixel_scale=np.array([self.shape[1],self.shape[0]])/case['size']

    def score(self, box):
        poly=silhouette_polygon(box,self.case)
        if poly is None:
            return 0.
        coords=np.clip(poly*self.pixel_scale,-1.e5,1.e5)
        if len(coords)<3:
            return 0.
        cell=np.zeros(self.shape,np.uint8)
        cv2.fillConvexPoly(cell,np.rint(coords*16).astype(np.int32),1,shift=4)
        overlap=float((cell*self.mask).sum())
        return overlap/max(1.e-8,self.mass+cell.sum()-overlap)

    def at(self,pixels):
        uv=np.asarray(pixels)*self.pixel_scale/3
        xy=uv.astype(int)
        h,w=self.field.shape
        xy=np.clip(xy,0,[w-1,h-1])
        return self.field[xy[:,1],xy[:,0]]


def make_scene(case):
    if case['initial'] is None:
        return None
    pt=coordinates(case['points'])
    if not len(pt):
        pt=np.asarray(case['initial'],float)[None]
        use=[False]
    else:
        use=[True]*len(pt)
    eye=np.asarray(case['eye'])
    initial=np.asarray(case['initial'])
    origin=pt.mean(0) if any(use) else initial
    convert=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32)
    basis=ray_frame(convert((origin-eye)[None]))[0].numpy()
    l,t,r,b=case['tangents']
    camera_rays=np.array([[x,y,1.] for x in (l,r) for y in (t,b)])
    transform=np.asarray(case['transform'])
    aimed=camera_rays@transform[:3,:3]@basis
    uv=aimed[:,:2]/aimed[:,2:].clip(min=.15)
    tangent=np.r_[uv.min(0),uv.max(0)].clip(-2,2) if case['window'] is not None else np.zeros(4)
    return scene_for(convert(pt[None]),convert((pt-eye)[None]),torch.tensor([use]),convert(eye[None]),
                     convert(np.array(case['prior'])[None]),convert(tangent[None]),convert(initial[None]))


def headings(case, cloud):
    initial=case['base']['box']
    values=[b[6] for b in case['reference']]
    if initial is not None:
        values.append(initial[6])
    if len(cloud)>=3:
        centered=cloud[:,:2]-cloud[:,:2].mean(0)
        _,vectors=np.linalg.eigh(centered.T@centered)
        main=vectors[:,-1]
        values.append(np.arctan2(main[1],main[0]))
    values += [case['yaw'], case['yaw']+pi/2, case['yaw']-pi/4,case['yaw']+pi/4,
               case['yaw']-pi/6,case['yaw']+pi/6,*np.linspace(-pi/2,pi/2,12,endpoint=False)]
    distinct=[]
    for yaw in values:
        if all(abs(angle(yaw-old))>pi/36 for old in distinct):
            distinct.append(float(angle(yaw)))
        if len(distinct)==6:
            break
    return distinct


def anchored(cloud, eye, initial, span, yaw, q, shift=0.):
    basis=axes(yaw)
    if not len(cloud):
        return np.r_[initial,span,yaw]
    p=cloud@basis
    p_eye=eye@basis
    low,high=np.quantile(p,[q,1-q],axis=0)
    mid=(low+high)/2
    mode=(p_eye<low).astype(float)-(p_eye>high)
    c=mid+mode*(span-(high-low))/2
    c[2]=low[2]+span[2]/2
    c[:2]=(1-shift)*c[:2]+shift*mid[:2]
    return np.r_[c@basis.T,span,yaw]


def lattice(case):
    cloud=coordinates(case['measured'])
    if not len(cloud):
        cloud=coordinates(case['points'])
    eye=np.asarray(case['eye'])
    prior=np.asarray(case['prior'])
    initial=np.asarray(case['initial'])
    yaws=headings(case,cloud)
    pool=[]
    for yaw in yaws:
        for dims,q,shift in zip(LATTICE_DIMS,LATTICE_QUANTILES,LATTICE_SHIFT,strict=True):
            pool.append(anchored(cloud,eye,initial,prior*dims,yaw,q,shift))
    pool.extend(case['fitted'])
    if case['base']['box'] is not None:
        pool.insert(0,case['base']['box'])
    pool=np.asarray(pool,float)
    _,first=np.unique(pool.round(7),axis=0,return_index=True)
    return pool[np.sort(first)]


def ray_selection_cost(box,case):
    from live_2d import live
    rays=coordinates(case['selection'])
    if not len(rays):
        return 0.
    depth=live.first_depth(np.asarray(box),rays,np.asarray(case['transform']))
    cap=max(.10,.5*min(case['prior'][:2]))
    return float(np.mean(np.where(depth>0,np.minimum(abs(depth-rays[:,2]),cap),cap)))


def iou_mbr(pool,weights=None):
    k=len(pool)
    x=torch.tensor(np.asarray(pool),dtype=torch.float64)
    matrix=exact_iou(x.repeat_interleave(k,dim=0),x.repeat(k,1)).reshape(k,k).numpy()
    value=matrix.mean(1) if weights is None else matrix@np.asarray(weights)/sum(weights)
    return int(value.argmax()), float(value.max())


@dataclass
class AnchorInterval:
    reference: np.ndarray
    mode: np.ndarray
    required: np.ndarray

    def project(self, candidate):
        old=self.reference[3:6]
        dimensions=np.minimum(old,np.maximum(self.required,candidate[3:6]))
        out=self.reference.copy()
        out[:3]+=axes(out[6])@(self.mode*(dimensions-old)/2)
        out[3:6]=dimensions
        return out


def interval(ref,case):
    from live_2d import live
    measured=coordinates(case['measured'])
    rays=coordinates(case['selection'])
    mat=np.asarray(case['transform'])
    depth=live.first_depth(ref,rays,mat)
    eye,fit,entries=live.protected_points(ref,measured,rays,depth,mat)
    half=ref[3:6]/2
    mode=np.r_[(eye[:2]<-half[:2]).astype(float)-(eye[:2]>half[:2]),1.]
    if (abs(eye)<=half+1.e-8).all():
        return AnchorInterval(ref,mode,ref[3:6])
    points=np.r_[fit,entries]
    if not len(points):
        return AnchorInterval(ref,mode,np.zeros(3))
    low,high=points.min(0),points.max(0)
    required=np.where(mode>0,half+high+1.e-8,np.where(mode<0,half-low+1.e-8,2*(np.maximum(-low,high)+1.e-8)))
    return AnchorInterval(ref,mode,required)


def flow_intervals(case):
    cloud=coordinates(case['measured'])
    if not len(cloud) and not len(case['selection']):
        return []
    prior=np.asarray(case['prior'])
    eye=np.asarray(case['eye'])
    center=np.asarray(case['initial'])
    references=[np.asarray(b) for b in case['reference']]
    for yaw in headings(case,cloud):
        if all(abs(angle(yaw-b[6]))>pi/36 for b in references):
            references.append(anchored(cloud,eye,center,1.22*prior,yaw,.07))
    return [interval(b,case) for b in references]


def first_hit_counts(reference,box,case,key):
    from live_2d import live
    from replay_factorial import ray_changes
    rays=coordinates(case[key])
    mat=np.asarray(case['transform'])
    return ray_changes(live.first_depth(np.asarray(reference),rays,mat),live.first_depth(np.asarray(box),rays,mat))


def matched_strict_pair(case,field):
    if not case['reference'] or case['base']['heading'] not in ('pca','qwen'):
        return None
    idx=int(case['base']['heading']=='qwen')
    a,b=case['reference'][idx],case['fitted'][idx]
    if np.max(abs(np.asarray(a)-b))<=1.e-8 or not len(case['audit']):
        return None
    if any(first_hit_counts(a,b,case,name)['changed'] for name in ('selection','audit')):
        return None
    return dict(pair=[b,a],rect=int(rectangle_iou(a,case)>rectangle_iou(b,case)),
                mask=int(field.score(a)>field.score(b)),i2d_difference=abs(rectangle_iou(a,case)-rectangle_iou(b,case)),
                mask_difference=abs(field.score(a)-field.score(b)))


def new_bind(frame,replies,field):
    """Weight the existing region vote with the frozen, LiDAR-prompted mask."""
    import minimal
    from cray_simple import region_catalog, MIN_RETURNS
    from geometry import sensor_pixels

    spec,_,_=minimal.geometry.camera(frame)
    camera=minimal.geometry.model_to_camera_matrix(spec)
    geom=minimal.geometry
    def vote(sample,terrain,scales,fit_returns,in_window,selection=None):
        radii,assigned,births=region_catalog(sample,terrain,scales)
        if not len(births) or not in_window.any():
            return None,None
        mapped=geom.transfer(sample,fit_returns,radii,assigned,births)
        n=len(in_window)
        scene,central=mapped[:,:n],mapped[:,n:]
        window=scene[:,in_window]
        density=np.bincount(scene[scene>=0],minlength=len(births))
        central_n=np.bincount(central[central>=0],minlength=len(births))
        source=sensor_pixels(geom.rigid_xyz(fit_returns[:n][in_window],camera),spec)
        weights=1. if source is None else .15+.85*field.at(source)
        expanded=np.broadcast_to(weights,window.shape)
        hits=np.bincount(window[window>=0],weights=expanded[window>=0],minlength=len(births))
        valid=np.flatnonzero((central_n>=MIN_RETURNS)&(hits>0))
        if not len(valid):
            return None,None
        chosen=valid[np.argmax(hits[valid]**2/density[valid])]
        level=births[chosen]
        region=sample[assigned[level]==chosen]
        return (region,None) if selection is None else (
            region,geom.transfer(sample,selection,radii,assigned,births)[level]==chosen)
    with patch.object(minimal,'choose_region',vote):
        prediction,context=minimal.predict(frame,replies,audit=True)
    if context is not None:
        _,candidates,_=context
        scores=[field.score(c[0]) for c in candidates]
        if scores[0]!=scores[1]:
            prediction=dict(prediction,box=candidates[int(scores[1]>scores[0])][0].tolist())
    return prediction['box']
