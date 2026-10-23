"""Fixed TA-STVG paper panel derived from one content-bound complete capture.

No model, filesystem, annotations, optimizer or GT-dependent action selection.
"""
from collections import deque
import numpy as np
import torch
from stvg.spatial import SpatialGeometry, evidence_dp
from stvg.temporal import final_mbr
from stvg.prior import from_capture

ARMS = ('Source', 'SharedInit', 'Interpolation', 'OursS', 'Global6', 'Full', 'NoLocal', 'Independent')

def interpolation(E, ids, positions, native_boxes):
    result = np.asarray(E, dtype=np.float64).copy()
    times = np.asarray(ids, dtype=np.float64)
    boxes = np.asarray(native_boxes, dtype=np.float64)
    valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
    anchors = [(int(t), boxes[j]) for j,t in enumerate(positions) if valid[j]]
    for t, b in anchors:
        result[t] = b
    for (a, ba), (b, bb) in zip(anchors[:-1], anchors[1:]):
        w = ((times[a:b+1] - times[a]) / (times[b] - times[a]))[:, None]
        result[a:b+1, :2] = (1-w)*ba[:2] + w*bb[:2]
        result[a:b+1, 2:] = np.exp((1-w)*np.log(ba[2:]) + w*np.log(bb[2:]))
    return result

def component_box(p):
    """Archived >.5 own-peak,4-connected,grid-edge box; largest mask mass wins.

    Ties retain original row-major component order. No GT or direction selection.
    Invalid padded cells already have zero propagated probability.
    """
    a = np.asarray(p, dtype=np.float64)
    assert a.ndim == 2 and np.isfinite(a).all() and (a >= 0).all()
    if a.max() <= 1e-15:
        return None
    mask = a > .5*a.max(); seen = np.zeros_like(mask);h,w = a.shape; best = None
    for y in range(h):
        for x in range(w):
            if not mask[y,x] or seen[y,x]:continue
            stack=deque([(y,x)]);seen[y,x]=True;cells=[]
            while stack:
                yy,xx=stack.popleft();cells.append((yy,xx))
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny,nx=yy+dy,xx+dx
                    if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True;stack.append((ny,nx))
            mass=sum(float(a[yy,xx])for yy,xx in cells)
            if best is None or mass>best[0]:
                ys,xs=zip(*cells);x0,x1=min(xs)/w,(max(xs)+1)/w;y0,y1=min(ys)/h,(max(ys)+1)/h
                best=(mass,np.array([(x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0]))
    return best[1]

def derive(cap):
    """All methods share the same ERef,8 anchors,native boxes and fixed settings."""
    needed={'native_boxes','side_maps','side_reference_boxes','side_available','source_anchors','height','width'}
    assert needed.issubset(cap['geometry']), 'Incomplete capture: collect missing evidence once'
    assert cap['GT_read'] is False
    g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry'])
    ids=list(map(int,cap['frame_ids']));pos=list(map(int,cap['positions8']));T=len(ids)
    assert len(pos)==8 and len(set(pos))==8 and np.array_equal(ids,np.asarray(g.frame_ids))
    q0=from_capture(cap)
    from stvg.prototype import one_step
    q,_=one_step(cap['content_features'],q0)
    # Global6 is a sealed historical reference only, never a production prior.
    global_interval=list(map(int,cap['reference_Global6_interval']))
    full_interval=final_mbr(ids,q)['interval'];native_interval=list(map(int,cap['native_interval']))
    bank=g.for_interval(full_interval);full=evidence_dp(bank);native=evidence_dp(g.for_interval(native_interval))
    valid=bank.valid.clone();valid[~bank.active,1:]=False
    independent_actions=bank.score_unary.masked_fill(~valid,-torch.inf).argmax(1)
    independent=bank.boxes[torch.arange(T),independent_actions]
    no_local=np.asarray(cap['E'],dtype=np.float64).copy();boxes=np.asarray(g.native_boxes)
    for j,t in enumerate(pos):
        # Same predicted-event legality and ERef fallback as Full; direct native
        # boxes replace selection only where the method actually outputs an event.
        if bank.event[t] and np.isfinite(boxes[j]).all() and (boxes[j,2:]>0).all():
            no_local[t]=boxes[j]
    assert np.array_equal(no_local[~np.asarray(g.observed)],np.asarray(cap['E'])[~np.asarray(g.observed)])
    arrays=dict(Source=np.asarray(cap['Source']),SharedInit=np.asarray(cap['E']),
        Interpolation=interpolation(cap['E'],ids,pos,boxes),OursS=native['final'].numpy(),Global6=np.asarray(cap['reference_Global6']),Full=full['final'].numpy(),
        NoLocal=no_local,Independent=independent.numpy(),frame_ids=np.asarray(ids),positions8=np.asarray(pos),
        Full_actions=full['actions'].numpy(),Independent_actions=independent_actions.numpy())
    for name in ARMS:
        arrays[name+'_interval']=np.asarray(global_interval if name=='Global6' else full_interval if name in ('Full','NoLocal','Independent')else native_interval)
    mask_boxes=np.zeros((T,2,4),dtype=np.float64);mask_available=np.zeros((T,2),bool)
    for t in range(T):
        for side in range(2):
            if int(g.source_anchors[t,side])<0:continue
            box=component_box(np.asarray(g.side_maps[t,side]))
            if box is not None:mask_boxes[t,side]=box;mask_available[t,side]=True
    arrays.update(mask_boxes=mask_boxes,mask_available=mask_available,
        complete_boxes=np.asarray(g.side_reference_boxes),complete_available=np.asarray(g.side_available),
        source_anchors=np.asarray(g.source_anchors),native_boxes=boxes,
        nearest_observation_seconds_placeholder=np.min(abs(np.asarray(ids)[:,None]-np.asarray(ids)[pos][None]),axis=1))
    assert all(np.isfinite(arrays[a]).all()and(arrays[a][:,2:]>0).all()for a in ARMS)
    return arrays,dict(GT_read=False,spatial_updates=0,expert_observations=8,heavy_forwards=0,
        native_temporal_alias='OursS',shared_init_is_mechanism_only=True,
        independent_replacements=int((independent_actions!=full['actions']).sum()))
