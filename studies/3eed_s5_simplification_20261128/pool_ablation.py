"""Counterfactual candidate pools and linear-cost anchor descriptors."""
from contextlib import ExitStack
from math import pi
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'3eed_adaptive_bind_20261128'))
from run_adaptive_bind import Trace,cache_geometry,common_fit
from compact_pool import box_key,pca_heading,projected
from geometry_models import exact_iou
from lean_geometry import image3,points4,rays3
from physical_readout import anchored,axes,coordinates,interval
from prepare_blind import prepare

CAP = 13
FOUR = ('base','no90','extent','geometry')
SINGLE = ('adaptive','adaptive_geometry')
POOLS = FOUR+SINGLE
RANK = {'base':('pose','region','size'),'no90':('pose','region','size'),
        'extent':('pose','region','extent'),'geometry':('pose','region','extent')}


def boundary_candidates(ref,obs):
    """Endpoints with identical measured first intersections to the reference."""
    from live_2d import live
    from replay_factorial import ray_changes

    rays = np.r_[coordinates(obs['selection']),coordinates(obs['audit'])]
    transform = np.asarray(obs['transform'])
    before = live.first_depth(ref,rays,transform)
    if not np.any(before>0):
        return []
    extent = interval(ref,dict(obs,selection=rays))
    target = ref.copy()
    target[3:6] = 0.
    small = extent.project(target)
    if (small[3:6]>1.e-6).all() and not ray_changes(before,live.first_depth(small,rays,transform))['changed']:
        return [small,ref]
    return [ref]


def generate(frame,replies,binding):
    """Build all requested variants from a shared frozen anchor and evidence."""
    import minimal
    from cray_simple import anchor,horizontal_axes

    g = minimal.geometry
    spec,size,transform = g.camera(frame)
    optical = g.model_to_camera_matrix(spec)[2]
    _,window = common_fit(g,frame,replies,spec,size,transform)
    trace = Trace(binding,optical,window)
    with patch.object(minimal,'choose_region',trace.choose):
        base,context = minimal.predict(frame,replies,audit=True)
        case = prepare((frame['uid'],dict(frame=frame,raw=replies,target=frame['target']),
                        dict(outputs=dict(axis_2d=base))))
    regions = trace.stages[base['readout']][0].regions if base['readout'] in trace.stages else []
    primary = regions[0] if regions else None
    keys = FOUR if binding=='C_four' else ('base','geometry')
    pools = {name:[base['box']] for name in keys}
    if base['box'] is None or not case['first_window']:
        return pools,case,base
    support = coordinates(case['measured'])
    if primary is None and len(support)<3:
        return pools,case,base

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

    def ref(region,yaw,dimensions):
        if region is None:
            return anchored(support,np.asarray(case['eye']),np.asarray(case['initial']),dimensions,yaw,0.)
        local = np.c_[region.points[:,:2]@horizontal_axes(yaw).T,region.points[:,2]]
        b = anchor(local.min(0),local.max(0),dimensions,yaw,eye,region.ground)
        return g.sensor_frame_box(b,frame['target'],frame['suffix'])

    observed = case
    if context is not None:
        ev = context[0]
        observed = dict(case,measured=ev.support_sensor,selection=ev.selection,audit=ev.audit)
    first = {name:{box_key(base['box']):[base['box'],True]} for name in keys}

    def add(name,box,retained=True):
        chosen = first[name].setdefault(box_key(box),[box.tolist(),False])
        chosen[1] |= retained

    refs = {}
    stats = {'selection':[0,0],'audit':[0,0]}
    for yaw,quarter_turn in ((pca,False),(pca+pi/2,True),(qwen,False),(qwen+pi/2,True)):
        reference = ref(primary,yaw,span)
        if not quarter_turn:
            refs[yaw] = reference
        limited = projected(reference,dict(observed,audit=[]),stats)
        for name in keys:
            if quarter_turn and name in ('geometry','no90'):
                continue
            add(name,reference,False)
            add(name,limited)
    assert stats['selection'][1]==0
    for region in regions[1:]:
        for yaw in (pca_heading(region.points,region.ground,span),qwen):
            proposal = ref(region,yaw,span)
            for name in keys:
                add(name,proposal)
    for yaw in (pca,qwen):
        scaled = [ref(primary,yaw,prior*scale) for scale in (.85,1.15)]
        endpoints = boundary_candidates(refs[yaw],observed)
        for name in keys:
            for proposal in endpoints if 'extent' in RANK[name] else scaled:
                add(name,proposal)
    pools = {name:[box for box,keep in first[name].values() if keep] for name in keys}
    assert all(len(pool)<=CAP for pool in pools.values())
    if regions:
        union = np.unique(np.concatenate([r.points for r in regions]),axis=0)
        case = dict(case,measured=g.rigid_xyz(union,np.linalg.inv(transform)@camera))
    return pools,case,base


def generate_all(frame,replies,ground_shift=0.):
    import minimal

    with ExitStack() as stack:
        g = minimal.geometry
        if ground_shift:
            terrain = g.regions.terrain
            def shifted(*args):
                model = np.asarray(terrain(*args)).copy()
                model[2] += ground_shift
                return model
            stack.enter_context(patch.object(g.regions,'terrain',shifted))
        cache_geometry(stack,g)
        four,case,base = generate(frame,replies,'C_four')
        single,adaptive_case,adaptive_base = generate(frame,replies,'B_adaptive')
    return [(four,case,base),({'adaptive':single['base'],'adaptive_geometry':single['geometry']},
                             adaptive_case,adaptive_base)]


def node_features(case,boxes):
    inverse = np.linalg.inv(case['transform'])
    returns = coordinates(case['selection'])
    rays = (returns/returns[:,2:])@inverse[:3,:3].T
    support = coordinates(case['measured'])
    frames = np.asarray([axes(b[6]) for b in boxes])
    values = [np.r_[points4(b,frame,support),rays3(b,frame,rays,inverse[:3,3],returns[:,2]),
                    image3(b,case),np.log(b[3:6]/case['prior']),np.cos(2*(b[6]-case['yaw']))]
              for b,frame in zip(boxes,frames,strict=True)]
    return np.asarray(values,np.float32),frames


def edge_features(boxes,frames,anchor_only=False):
    n = len(boxes)
    center,span,yaw = boxes[:,:3],boxes[:,3:6],boxes[:,6]
    ref = slice(0,1) if anchor_only else slice(None)
    l = len(boxes[ref])
    raw = torch.as_tensor(boxes,dtype=torch.float64)
    pairs = exact_iou(raw[ref].repeat_interleave(n,0),raw.repeat(l,1)).reshape(l,n).numpy()
    local = ((center[None]-center[ref,None])@frames[ref])/span[ref,None]
    edges = np.concatenate((pairs[...,None],local,np.log(span[None]/span[ref,None]),
                            np.cos(2*(yaw[None]-yaw[ref,None]))[...,None]),axis=-1)
    return (edges[0] if anchor_only else edges).astype(np.float32)


def tensors(pools,case,truth=None):
    from score_paradigms import ious

    unique = {}
    lookups = {name:[unique.setdefault(tuple(b) if b is not None else None,len(unique)) for b in pool]
               for name,pool in pools.items()}
    if None in unique:
        assert len(unique)==1
        nodes = np.zeros((1,14),np.float32)
        edges = np.zeros((1,1,8),np.float32)
        labels = np.zeros(1,np.float32)
    else:
        boxes = np.asarray(list(unique))
        nodes,frames = node_features(case,boxes)
        edges = edge_features(boxes,frames)
        if truth is not None:
            labels = ious(boxes,truth).astype(np.float32)
    output = {}
    for name,idx in lookups.items():
        count = len(idx)
        x,e = np.zeros((CAP,14),np.float32),np.zeros((CAP,CAP,8),np.float32)
        x[:count],e[:count,:count] = nodes[idx],edges[np.ix_(idx,idx)]
        valid = np.arange(CAP)<count
        if truth is None:
            output[name] = x,e,valid
        else:
            y = np.zeros(CAP,np.float32)
            y[:count] = labels[idx]
            output[name] = x,e,valid,y
    return output
