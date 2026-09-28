"""Offline, label-explicit case diagnostics. Never imported by the predictor."""
import re
import numpy as np


def masks(ids, gt, interval, valid):
    ids=np.asarray(ids);valid=np.asarray(valid,bool)
    a,b=ids[interval[0]],ids[interval[1]]+1
    selected=(ids>=a)&(ids<b)
    union=(ids>=min(a,gt[0]))&(ids<max(b,gt[1]))
    return selected&valid,max(1,int(union.sum()))


def temporal_contributions(quality, valid, ids, gt, frozen, adapted):
    """Exact change = changed selected numerator + changed union penalty.

    The order is explicit: denominator is changed first, numerator second.
    This is accounting, not a causal explanation of the model's decision.
    """
    q=np.nan_to_num(np.asarray(quality,float));m0,d0=masks(ids,gt,frozen,valid);m1,d1=masks(ids,gt,adapted,valid)
    n0=float(q[m0].sum());n1=float(q[m1].sum())
    coverage=(n1-n0)/d1;penalty=n0*(1/d1-1/d0)
    return dict(selected_quality_change=coverage,union_change=penalty,total=coverage+penalty,
        added_GT_frames=int((m1&~m0).sum()),removed_GT_frames=int((m0&~m1).sum()),
        added_GT_quality=float(q[m1&~m0].mean()) if (m1&~m0).any() else None,
        frozen_union_count=d0,adapted_union_count=d1)


def spatial_contributions(before,after,valid,ids,gt,interval,accepted):
    delta=np.nan_to_num(np.asarray(after,float)-np.asarray(before,float))
    selected,den=masks(ids,gt,interval,valid);anchors=np.zeros(len(ids),bool)
    anchors[list(accepted)]=True
    a=float(delta[selected&anchors].sum()/den);other=float(delta[selected&~anchors].sum()/den)
    return dict(accepted_anchor_change=a,nonanchor_change=other,total=a+other)


def data_features(x,gt,q0,q1):
    meta=x['input'];ids=np.asarray(x['frame_ids']);times=ids/meta['fps'];valid=np.asarray(gt['valid'],bool)
    truth=np.asarray(gt['boxes'],float);base=np.asarray(x['predictions']['frozen']['boxes'],float)
    horizon=ids[-1]+1-ids[0];anchors=[p['position'] for p in x['pseudo']];called=x['keyframes'] if x['expert'] else []
    accepted=np.zeros(len(ids),bool);accepted[anchors]=True
    called_mask=np.zeros(len(ids),bool);called_mask[called]=True
    hull=np.zeros(len(ids),bool)
    if anchors:hull[min(anchors):max(anchors)+1]=True
    def avg(a,mask):return float(np.asarray(a)[mask].mean()) if mask.any() else None
    area=truth[:,2]*truth[:,3];adj=valid[1:]&valid[:-1];dt=np.diff(times)
    gt_step=np.linalg.norm(np.diff(truth[:,:2],axis=0),axis=1)
    native_step=np.linalg.norm(np.diff(base[:,:2],axis=0),axis=1)
    norm=np.maximum(np.sqrt((area[:-1]+area[1:])/2),1e-6)
    speed=gt_step/np.maximum(dt,1e-8)/norm
    e=np.linalg.norm(base[:,:2]-truth[:,:2],axis=1)/np.maximum(np.sqrt(area),1e-6)
    scale=np.abs(np.log(np.maximum(base[:,2:],1e-8)/np.maximum(truth[:,2:],1e-8))).mean(-1)
    gaps=np.diff(times[sorted(anchors)]) if len(anchors)>1 else np.array([])
    words=re.findall(r"[a-z]+",meta['caption'].lower())
    text=x['expert'][0]['text'] if x['expert'] else ''
    reasons={}
    for p in x['expert']:reasons[p['reason']]=reasons.get(p['reason'],0)+1
    top=[float(p['score']) for p in x['expert']];margin=[float(p['margin']) for p in x['expert']]
    return dict(duration_seconds=horizon/meta['fps'],grid_positions=len(ids),
        gt_event_fraction=(gt['interval'][1]-gt['interval'][0])/horizon,
        gt_event_seconds=(gt['interval'][1]-gt['interval'][0])/meta['fps'],
        gt_center_fraction=((gt['interval'][0]+gt['interval'][1])/2-ids[0])/horizon,
        gt_valid_frames=int(valid.sum()),gt_box_area_mean=avg(area,valid),
        gt_min_side_pixels=avg(np.minimum(truth[:,2]*meta['width'],truth[:,3]*meta['height']),valid),
        gt_edge_fraction=avg(((truth[:,:2]-truth[:,2:]/2<.02)|(truth[:,:2]+truth[:,2:]/2>.98)).any(-1),valid),
        gt_speed_target_units_per_second=avg(speed,adj),
        native_step_given_gt_adjacent=avg(native_step,adj),gt_step_given_gt_adjacent=avg(gt_step,adj),
        native_center_error_target_units=avg(e,valid),native_abs_log_scale_error=avg(scale,valid),
        caption_words=len(words),question=int(x['query_type']=='question'),
        relative_location_words=sum(w in {'left','right','front','behind','next','beneath','above','below'} for w in words),
        interaction_words=sum(w in {'hold','holding','holds','carry','carrying','wear','wearing','push','pushes','pull','pulls'} for w in words),
        expert_phrase=text,expert_phrase_words=len(re.findall(r'[a-z]+',text.lower())),
        expert_unavailable=int(not x['expert']),expert_called=len(called),expert_accepted=len(anchors),
        expert_rejection_reasons=reasons,expert_top_score_mean=float(np.mean(top)) if top else None,
        expert_margin_mean=float(np.mean(margin)) if margin else None,
        accepted_gt_anchors=int((accepted&valid).sum()),hull_gt_fraction=avg(hull,valid),
        max_anchor_gap_seconds=float(gaps.max()) if len(gaps) else None,
        max_anchor_gap_fraction=float(gaps.max()*meta['fps']/horizon) if len(gaps) else None,
        native_called_gt_sIoU=avg(q0,called_mask&valid),method_called_gt_sIoU=avg(q1,called_mask&valid),
        native_uncalled_gt_sIoU=avg(q0,~called_mask&valid),method_uncalled_gt_sIoU=avg(q1,~called_mask&valid),
        native_hull_interior_sIoU=avg(q0,hull&~accepted&valid),method_hull_interior_sIoU=avg(q1,hull&~accepted&valid))
