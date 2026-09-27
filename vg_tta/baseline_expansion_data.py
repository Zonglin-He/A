"""Data-only bridges for the preregistered baseline expansion.

Reuse the audited dataset decoders and capture their *actual* raw pixels before
normalization. Labels are returned separately and never passed to adaptation.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


class CaptureTransform:
    def __init__(self, transform):
        self.transform = transform
        self.raw = None

    def __call__(self, frames, targets):
        self.raw = np.asarray(frames)
        return self.transform(frames, targets)


def make_dataset(spec):
    from datasets.video_transforms import make_video_transforms
    transform = CaptureTransform(make_video_transforms('val', cautious=True, resolution=224))
    kind = spec['kind']
    if kind == 'anygroundbench':
        from scripts.anygroundbench_floor_loader import AnyGroundBenchVariableDuration
        dataset = AnyGroundBenchVariableDuration(spec['root'], spec['annotation'],
                                                fps=5, max_frames=200, resolution=224, stride=2)
        dataset.transform = transform
        rows = dataset.annotations
    else:
        if kind == 'hcstvg':
            from datasets.hcstvg import VideoModulatedSTGrounding
            extra = {'video_max_len_train': 100}
        elif kind == 'vidstg':
            from datasets.vidstg import VideoModulatedSTGrounding
            extra = {}
        else:
            raise ValueError(kind)
        dataset = VideoModulatedSTGrounding(spec['root'], spec['annotation'], transforms=transform,
            is_train=False, video_max_len=200, fps=5, tmp_crop=False, tmp_loc=True, stride=2, **extra)
        rows = dataset.annotations if kind == 'hcstvg' else dataset.annotations['videos']
    return dataset, rows, transform


def source_id(row, kind):
    stem = Path(row['video_path']).stem
    if kind == 'hcstvg':
        return stem.split('_', 1)[1] if '_' in stem else stem
    if kind == 'vidstg':
        return str(row['original_video_id'])
    # Mouse recordings: all views and clips from one date/animal stay together.
    if '__' in stem and 'mouse' in stem:
        return stem.split('__')[1]
    # Football metadata can supply an explicit original capture identity.
    return str(row.get('source_cluster', row.get('source_video_id', stem)))


def select_sources(rows, kind, count, seed, overrides=None, all_queries=False):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        source = overrides[str(i)] if overrides is not None else source_id(row, kind)
        groups[source].append(i)
    order = sorted(groups, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    picked = order[:count]
    indices = []
    for source in picked:
        members = sorted(groups[source], key=lambda i: hashlib.sha256(f'{seed}:{source}:{i}'.encode()).hexdigest())
        indices.extend(members if all_queries else members[:1])
    return sorted(indices), {str(i): s for s in picked for i in groups[s] if i in indices}


def shift_pixels(raw, condition):
    """Same corrupted raw video for every spatial view, never a clean crop."""
    if raw.ndim != 4 or raw.shape[-1] != 3 or raw.dtype != np.uint8:
        raise ValueError('expected uint8 THWC pixels')
    positions = np.arange(len(raw))
    if condition == 'subsample2':
        return raw[::2].copy(), positions[::2].tolist()
    if condition == 'clean':
        return raw, positions.tolist()
    if condition == 'low_light3':
        result = np.round(raw.astype(np.float32) * .4).astype(np.uint8)
    elif condition == 'blur3':
        # Gaussian severity 3 specified in equivalent res224 pixels. Not motion blur.
        sigma = 1.6 * min(raw.shape[1:3]) / 224
        kernel = max(3, int(np.ceil(sigma * 6)) | 1)
        result = np.stack([cv2.GaussianBlur(frame, (kernel, kernel), sigma,
                                          borderType=cv2.BORDER_REFLECT_101) for frame in raw])
    else:
        raise ValueError(condition)
    return result, positions.tolist()


def lift_prediction(prediction, positions, original_frame_ids):
    """Evaluate 2x sampling on the SAME original grid as the clean counterpart.

    Spatial coordinates interpolate by source timestamps; start/end choices
    remain restricted to actually observed positions. No GT is needed here.
    """
    boxes = prediction['pred_boxes'].detach().float().cpu()
    logits = prediction['pred_sted'].detach().float().cpu()
    n = len(original_frame_ids)
    if positions == list(range(n)):
        return {'pred_boxes': boxes, 'pred_sted': logits}
    if len(positions) != len(boxes) or len(positions) < 2:
        raise ValueError('at least two retained samples required')
    if positions != sorted(set(positions)) or positions[0] < 0 or positions[-1] >= n:
        raise ValueError('invalid sampling positions')
    timestamps = np.asarray(original_frame_ids, dtype=np.float64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError('strictly increasing timestamps required')
    lifted = np.stack([np.interp(timestamps, timestamps[positions], boxes[:, d].numpy())
                       for d in range(4)], axis=-1)
    # Finite sentinel keeps the shared native decoder and metric replay usable.
    full_logits = torch.full((1, n, 2), -1e20, dtype=torch.float32)
    full_logits[:, positions] = logits
    return {'pred_boxes': torch.from_numpy(lifted).float(), 'pred_sted': full_logits}
