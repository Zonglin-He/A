"""F22 label-free S2 functions. No labels, tuning, source IDs or outcome readers."""
import numpy as np
import torch
from torchvision.ops import box_convert, nms
from vg_tta.decota_spatial_extension_v1 import pool_gate
from methods.decota_refine_uniform_v1.api import reconstruct


def candidate_stages(detection):
    xy=box_convert(detection['all_boxes'].float().cpu(),'cxcywh','xyxy').clamp(0,1)
    old=detection['all_phrase_scores'].float().cpu()
    target=detection['target_all_scores'].float().cpu()
    keep=nms(xy,old,.5)
    # Match F21: truncate first, then remove invalid geometry.
    old3=keep[:3];old3=old3[(xy[old3,2:]>xy[old3,:2]).all(1)]
    order=torch.argsort(target[keep],descending=True,stable=True)
    new3=keep[order[:3]];new3=new3[(xy[new3,2:]>xy[new3,:2]).all(1)]
    return dict(boxes=box_convert(xy,'xyxy','cxcywh'),raw=list(range(len(xy))),
                nms=keep.tolist(),old_top3=old3.tolist(),target_top3=new3.tolist())


def probe_from_detection(detection,position,frame_id,ranking='old_top3'):
    if ranking not in ('old_top3','target_top3'):raise ValueError(ranking)
    stages=candidate_stages(detection);idx=stages[ranking]
    return pool_gate(dict(boxes=stages['boxes'][idx],target_scores=detection['target_all_scores'][idx],
        candidate_ids=idx,position=position,frame_id=frame_id))


def independent(native,ids,probes,support=None,reference0=False):
    accepted=[z for z in probes if z['accepted'] and (support is None or z['position'] in support)]
    anchors=[];path=[]
    for i,z in enumerate(accepted):
        j=0 if reference0 and i==0 else int(np.argmax(np.asarray(z['target_scores'])))
        anchors.append(dict(position=z['position'],frame_id=z['frame_id'],box=z['boxes'][j].tolist(),
            score=float(z['target_scores'][j]),margin=z['margin']))
        path.append(dict(position=z['position'],candidate_id=z['candidate_ids'][j],pool_index=j))
    boxes,audit=reconstruct(native,anchors,ids,'absolute')
    return dict(boxes=boxes,anchors=anchors,path=path,support=[z['position'] for z in anchors],reconstruction=audit)


def old_reconstruct(native,ids,probes,support=None):
    anchors=[{k:z[k] for k in ('position','frame_id','box','score','margin')}
        for z in probes if z['accepted'] and (support is None or z['position'] in support)]
    boxes,audit=reconstruct(native,anchors,ids,'absolute')
    return dict(boxes=boxes,anchors=anchors,support=[z['position'] for z in anchors],reconstruction=audit)


def time_control(z,ids,native,spec):
    if spec['kind']=='native':return list(native)
    if spec['kind']=='prior':
        from vg_tta.native_coverage_calibration_v1 import native_at_length
        return list(native_at_length(z,ids,fraction=spec['value'])['indices'])
    if spec['kind']=='pm':
        from vg_tta.posterior_mass_coverage_v1 import select
        return list(select(z,ids,tau=spec['value'])['indices'])
    raise ValueError(spec)


TEMPORAL_CONTROLS=[dict(kind='native',value=None)]+[
    dict(kind=k,value=v) for k,vs in [('prior',[.2,.4,.6,.8,1.]),('pm',[.5,.7,.8,.9,.95])] for v in vs]
LAMBDAS=[0.,.1,.3,1.,3.,10.,30.]
