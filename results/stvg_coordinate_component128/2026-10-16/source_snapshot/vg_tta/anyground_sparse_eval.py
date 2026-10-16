"""Use the released AnyGroundBench Vidi tube evaluator without filling GT gaps.

These metrics are deliberately NOT named corrected STVG vIoU. The released
evaluator quantizes timestamps and offers volume IoU and mean 2D IoU; its
treatment of sparse annotations differs from the dense TubeDETR evaluator.
"""
from pathlib import Path
import sys
import numpy as np
import torch
from vg_tta.metrics import interval_from_logits, box_cxcywh_to_xyxy

ROOT = Path(__file__).resolve().parents[1]


def official_api():
    repo = str(ROOT / 'external/AnyGroundBench')
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.eval.tube import Tube
    from src.eval.vidi_evaluate import compare_tubes
    return Tube, compare_tubes


def sample_indices(frame_count, fps, max_frames=200):
    if frame_count < 2 or fps <= 0:
        raise ValueError('requires positive fps and at least two video frames')
    n = min(frame_count, max_frames, max(2, int(np.ceil(frame_count / fps * 5))))
    # Includes both endpoints; bounded, uniform, annotation-independent sampling.
    return np.unique(np.linspace(0, frame_count - 1, n).round().astype(int)).tolist()


def evaluate_sparse(prediction, sample_ids, metadata, media):
    Tube, compare = official_api()
    fps = float(media['fps'])
    count = int(media['frame_count'])
    if sample_ids[0] != 0 or sample_ids[-1] != count - 1:
        raise ValueError('sample grid must include video endpoints')
    start, end = interval_from_logits(prediction['pred_sted'])
    pred_start, pred_end = sample_ids[start], sample_ids[end] + 1
    boxes = box_cxcywh_to_xyxy(prediction['pred_boxes'].detach().float().cpu()).numpy()
    if len(boxes) != len(sample_ids):
        raise ValueError('box/sample mismatch')
    # Interpolate predictions only, never missing ground-truth boxes.
    frames = np.arange(pred_start, pred_end)
    interpolated = np.stack([np.interp(frames, sample_ids, boxes[:, j]) for j in range(4)], axis=1)
    pred_tube = Tube(step_ms=1000)
    for frame, box in zip(frames, interpolated):
        pred_tube.add_bbox(int(round(frame / fps * 1000)), tuple(map(float, box)))
    gt_tube = Tube(step_ms=1000)
    gt_fps = float(metadata['meta_info']['fps'])
    width = float(metadata['meta_info']['width'])
    height = float(metadata['meta_info']['height'])
    annotated_frames = set()
    for tube in metadata['spatio_temporal_label']['tubes']:
        for frame, box in sorted(tube['bbox'].items(), key=lambda x: int(x[0])):
            frame = int(frame)
            annotated_frames.add(frame)
            normalized = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
            gt_tube.add_bbox(int(round(frame / gt_fps * 1000)), normalized)
    official = compare(gt_tube, pred_tube, multi_boxes_policy='first')
    gt_start, gt_end = map(float, metadata['temporal_range'].split())
    predicted = (pred_start / fps, pred_end / fps)
    intersection = max(0., min(predicted[1], gt_end) - max(predicted[0], gt_start))
    union = max(predicted[1], gt_end) - min(predicted[0], gt_start)
    return {'official_vidi_volume_iou': official['v_iou_3d'],
            'official_vidi_mean_2d_iou': official['legacy_v_iou'],
            'official_vidi_timestamp_iou': official['t_iou'],
            'continuous_interval_tiou': intersection / union if union > 0 else 0.,
            'predicted_interval_seconds': list(predicted), 'gt_annotated_frames': len(annotated_frames),
            'gt_timestamp_bins': gt_tube.get_length(), 'prediction_timestamp_bins': pred_tube.get_length(),
            'official_all_metrics': official, 'gt_interpolation': False,
            'metric_convention': 'AnyGroundBench released Vidi evaluator; step_ms1000; first box per bin; sparse GT unchanged'}
