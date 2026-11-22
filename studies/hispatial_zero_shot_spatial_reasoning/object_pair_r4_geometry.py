"""Measure object-pair relations directly on frozen MoGe visible support."""
import itertools
import warnings

import numpy as np
from scipy.spatial import cKDTree

from height_geometry import UP, estimate_up, inside_rect, unit
from object_region_audit import ious
from object_region_plan import proposal_regions, valid_xyz


CATEGORIES = ('distance_data','horizontal_distance_data','vertical_distance_data','direction')
ARMS = ('window_robust','visible_centroid','robust_center','multi_hypothesis',
        'support_nn','camera_robust','mask_robust')
BOX_ARMS = ARMS[:5]
MIN_POINTS = 24


def geometric_median(points):
    here=points.mean(axis=0)
    for _ in range(128):
        distance=np.linalg.norm(points-here,axis=1)
        updated=np.average(points,axis=0,weights=1/np.maximum(distance,1e-8))
        if np.linalg.norm(updated-here)<=1e-7:
            return updated
        here=updated
    return here


def clock_error(first,second):
    return min(abs(first-second),12-abs(first-second))


def basis(up):
    optical=np.array([0.,0.,1.])
    forward=unit(optical-(optical@up)*up)
    right=unit(np.cross(forward,up))
    return np.stack((right,up,forward),axis=1)


def relationship(first,second,category,up):
    if category=='distance_data':
        return float(np.linalg.norm(second-first)),np.stack((first,second)).tolist()
    if up is None:
        return None,None
    coords=np.stack((first,second))@basis(up)
    dx,dy,dz=coords[1]-coords[0]
    if category=='horizontal_distance_data':
        return float(abs(dx)),coords[:,0].tolist()
    if category=='vertical_distance_data':
        return float(abs(dy)),coords[:,1].tolist()
    if np.hypot(dx,dz)<1e-8:
        return None,None
    angle=(np.degrees(np.arctan2(-dx,dz))+360)%360
    hour=12-int(angle//30)
    return hour,coords[:,[0,2]].tolist()


def output(value=None,evidence=None,alternatives=None):
    return {'estimate':value,'evidence':evidence,'alternatives':[] if alternatives is None else alternatives}


def measure(first,second,category,up):
    if first is None or second is None:
        return output()
    return output(*relationship(first,second,category,up))


def neighbor_relation(a,b):
    if a is None or b is None:
        return output()
    a2b=cKDTree(b).query(a)[0]
    b2a=cKDTree(a).query(b)[0]
    return output(float((a2b.sum()+b2a.sum())/(len(a)+len(b))),
                  [float(a2b.mean()),int(len(a)),float(b2a.mean()),int(len(b))])


def one_object(grid,usable,rect,candidates,x,y):
    crop=usable & inside_rect(x,y,rect)
    matches=ious(rect,[row['box'] for row in candidates])
    hypotheses=[]
    for j in np.argsort(-matches,kind='stable'):
        region=candidates[j]
        if matches[j]<.25:
            break
        if any(region['scale']==h['scale'] for h in hypotheses):
            continue
        support=grid[crop & region['grid_mask']]
        if len(support)<MIN_POINTS:
            continue
        hypotheses.append(dict(region=region['id'],scale=region['scale'],iou=float(matches[j]),
                               points=support,center=geometric_median(support)))
    return grid[crop],hypotheses


def multi_relation(a,b,category,up):
    choices=[relationship(x['center'],y['center'],category,up) for x,y in itertools.product(a,b)]
    valid=[p for p in choices if p[0] is not None]
    values=[p[0] for p in valid]
    if not values:
        return output()
    if category=='direction':
        idx=min(range(len(values)),key=lambda j:sum(clock_error(values[j],v) for v in values))
    else:
        idx=int(np.abs(np.asarray(values)-np.median(values)).argmin())
    return output(*valid[idx],alternatives=values)


def prepare_scene(rgb,xyz,config):
    grid=xyz[:3,2::4,2::4].transpose(1,2,0).astype(np.float64)
    usable=valid_xyz(xyz[:,2::4,2::4])
    x,y=np.meshgrid(np.arange(2,448,4),np.arange(2,448,4))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Got image with third dimension of 6',category=RuntimeWarning)
        candidates,_=proposal_regions(rgb,xyz,config)
    return grid,usable,x,y,candidates


def object_points(scene,rects,region_masks):
    grid,valid,x,y,candidates=scene
    up,plane=estimate_up(grid,valid,x,y,rects)
    window,hypotheses=zip(*(one_object(grid,valid,rect,candidates,x,y) for rect in rects),strict=True)
    points=[values[0]['points'] if values else None for values in hypotheses]
    masked=[grid[valid&mask] for mask in region_masks] if region_masks is not None else [np.empty((0,3))]*2
    return dict(up=up,plane=plane,window=window,hypotheses=hypotheses,points=points,masked=masked)


def readout(scene,category):
    up=scene['up']
    a,b=scene['points']
    ha,hb=scene['hypotheses']
    centers=[values[0]['center'] if values else None for values in (ha,hb)]
    mask_center=[geometric_median(v) if len(v)>=MIN_POINTS else None for v in scene['masked']]
    windows=[geometric_median(v) if len(v)>=MIN_POINTS else None for v in scene['window']]
    preds = {
        'window_robust':measure(*windows,category,up),
        'visible_centroid':measure(a.mean(axis=0) if a is not None else None,
                                    b.mean(axis=0) if b is not None else None,category,up),
        'robust_center':measure(*centers,category,up),
        'multi_hypothesis':multi_relation(ha,hb,category,up),
        'support_nn':neighbor_relation(a,b) if category=='distance_data' else output(),
        'camera_robust':measure(*centers,category,UP) if category!='distance_data' else output(),
        'mask_robust':measure(*mask_center,category,up)
    }
    calibration=dict(up=up.tolist() if up is not None else None,plane=scene['plane'],
                     window_counts=[len(v) for v in scene['window']],mask_counts=[len(v) for v in scene['masked']],
                     region_counts=[len(a) if a is not None else 0,len(b) if b is not None else 0],
                     region_ids=[ha[0]['region'] if ha else None,hb[0]['region'] if hb else None],
                     hypotheses_per_object=[len(ha),len(hb)])
    return preds,calibration
