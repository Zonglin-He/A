"""Released Vidi metrics for a previously sealed, possibly endpoint-short grid.

This adapter does not change predictions or complete the sampling grid. It
uses the historical TubeDETR interval decoder and the released Vidi Tube and
compare_tubes implementations. Only prediction boxes inside the already
decoded interval are interpolated. Ground-truth boxes remain sparse.
"""
from numbers import Integral
import math
import numpy as np
import torch
from vg_tta.metrics import interval_from_logits, box_cxcywh_to_xyxy
from vg_tta.anyground_sparse_eval import official_api


def evaluate_cached_grid(prediction, sample_ids, metadata, media):
    ids=list(sample_ids)
    fps=float(media['fps']); count=int(media['frame_count'])
    if not math.isfinite(fps) or fps <= 0 or count < 2:
        raise ValueError('invalid physical media geometry')
    if len(ids)<2 or any(isinstance(v,bool) or not isinstance(v,Integral) for v in ids):
        raise ValueError('cached grid requires at least two integer positions')
    ids=list(map(int,ids))
    if ids != sorted(set(ids)) or ids[0]<0 or ids[-1]>=count:
        raise ValueError('cached grid must be unique, ordered and inside media')
    boxes=prediction['pred_boxes']
    logits=prediction['pred_sted']
    if not torch.is_tensor(boxes) or tuple(boxes.shape)!=(len(ids),4) or not torch.isfinite(boxes).all():
        raise ValueError('invalid cached boxes')
    if not torch.is_tensor(logits) or tuple(logits.shape) not in ((len(ids),2),(1,len(ids),2)):
        raise ValueError('invalid cached temporal logits')
    start,end=interval_from_logits(logits)
    pred_start,pred_end=ids[start],ids[end]+1
    xyxy=box_cxcywh_to_xyxy(boxes.detach().float().cpu()).numpy()
    frames=np.arange(pred_start,pred_end)
    interpolated=np.stack([np.interp(frames,ids,xyxy[:,j]) for j in range(4)],axis=1)
    Tube,compare=official_api()
    predicted_tube=Tube(step_ms=1000)
    for frame,box in zip(frames,interpolated):
        predicted_tube.add_bbox(int(round(frame/fps*1000)),tuple(map(float,box)))
    gt_tube=Tube(step_ms=1000)
    gt_fps=float(metadata['meta_info']['fps'])
    width=float(metadata['meta_info']['width']); height=float(metadata['meta_info']['height'])
    if any(not math.isfinite(v) or v<=0 for v in (gt_fps,width,height)):
        raise ValueError('invalid released metadata')
    annotated=set()
    for tube in metadata['spatio_temporal_label']['tubes']:
        for frame,box in sorted(tube['bbox'].items(),key=lambda x:int(x[0])):
            frame=int(frame)
            if len(box)!=4 or not np.isfinite(box).all() or not 0<=frame<count:
                raise ValueError('invalid released sparse GT box')
            annotated.add(frame)
            normalized=(box[0]/width,box[1]/height,box[2]/width,box[3]/height)
            gt_tube.add_bbox(int(round(frame/gt_fps*1000)),normalized)
    official=compare(gt_tube,predicted_tube,multi_boxes_policy='first')
    gt_start,gt_end=map(float,metadata['temporal_range'].split())
    if not all(math.isfinite(v) for v in (gt_start,gt_end)) or gt_end<=gt_start:
        raise ValueError('invalid released temporal range')
    predicted=(pred_start/fps,pred_end/fps)
    intersection=max(0.,min(predicted[1],gt_end)-max(predicted[0],gt_start))
    union=max(predicted[1],gt_end)-min(predicted[0],gt_start)
    return {
        'official_vidi_volume_iou':official['v_iou_3d'],
        'official_vidi_mean_2d_iou':official['legacy_v_iou'],
        'official_vidi_timestamp_iou':official['t_iou'],
        'continuous_interval_tiou':intersection/union if union>0 else 0.,
        'predicted_interval_seconds':list(predicted),
        'gt_annotated_frames':len(annotated),
        'gt_timestamp_bins':gt_tube.get_length(),
        'prediction_timestamp_bins':predicted_tube.get_length(),
        'official_all_metrics':official,
        'gt_interpolation':False,
        'metric_convention':'AnyGroundBench released Vidi evaluator; step_ms1000; first box per bin; sparse GT unchanged; sealed old sampling grid and temporal endpoints unchanged',
        'sampling_grid_extended':False,
        'prediction_interval_extended':False,
    }
