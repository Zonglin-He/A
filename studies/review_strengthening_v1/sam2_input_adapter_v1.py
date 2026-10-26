"""Lossless RGB and selected-mask adapter for the pinned official SAM 2 API.

Importing this module creates no model. The loader replaces only file decoding:
the pinned official PIL resize, float32 buffer and normalization are unchanged.
The original vendor source is never edited. Real propagation needs a separate
frozen execution plan, GPU lease and paid-attempt/commit audit in its caller.
"""
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def selected_prompts(teacher, height, width):
    """No padded teacher canvas is accepted without an explicit crop mapping."""
    assert teacher['nested_mask_all_false'] is True
    logits = torch.as_tensor(teacher['native_mask_logits']).detach().cpu().float()
    assert logits.ndim == 3 and logits.shape[0] == 8
    assert torch.isfinite(logits).all()
    shape = teacher['input_shape']
    assert len(shape) == 4 and shape[0] == 8 and shape[1] == 3
    assert min(height, width, shape[-2], shape[-1]) > 0
    # ReferDINO uses aspect-preserving short-side resize without padding.
    # An integer rounding of at most one resized pixel is permitted here.
    assert abs(shape[-1] - width * shape[-2] / height) <= 1.01
    return (F.interpolate(logits[:, None], size=(height, width),
                          mode='bilinear', align_corners=False)[:, 0] > 0).numpy()


def lossless_rgb_loader(video_path, image_size, offload_video_to_cpu,
                        img_mean=(.485, .456, .406), img_std=(.229, .224, .225),
                        async_loading_frames=False, compute_device='cpu'):
    frames = video_path
    assert isinstance(frames, np.ndarray) and frames.dtype == np.uint8
    assert frames.ndim == 4 and frames.shape[-1] == 3 and len(frames) > 0
    assert async_loading_frames is False
    count, height, width, _ = frames.shape
    images = torch.zeros(count, 3, image_size, image_size, dtype=torch.float32)
    for i, frame in enumerate(frames):
        # Exactly the pinned official _load_img_as_tensor operation, with an
        # already-decoded RGB PIL image instead of a lossy JPEG round trip.
        resized = np.array(Image.fromarray(frame).convert('RGB').resize(
            (image_size, image_size)))
        images[i] = torch.from_numpy(resized / 255.0).permute(2, 0, 1)
    mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        mean = mean.to(compute_device)
        std = std.to(compute_device)
    images -= mean
    images /= std
    return images, height, width


@contextmanager
def scoped_lossless_loader(predictor_module):
    original = predictor_module.load_video_frames
    predictor_module.load_video_frames = lossless_rgb_loader
    try:
        yield
    finally:
        predictor_module.load_video_frames = original


def mask_box(mask):
    """Normalized cx,cy,w,h, half-open pixel bounds; empty is zero area."""
    mask = np.asarray(mask)
    assert mask.ndim == 2 and mask.dtype == np.bool_
    height, width = mask.shape
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return np.zeros(4, dtype=np.float64)
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    return np.array([(x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
                     (x1 - x0) / width, (y1 - y0) / height], dtype=np.float64)


def propagate_eight(predictor, predictor_module, frames, positions, prompts):
    """Official all-conditioning propagation; called only by a gated GPU worker.

    Forward covers earliest..end, reverse covers earliest-1..0. The duplicate
    conditioning output at earliest is checked exactly, never fused or selected.
    No source boxes, query text, GT, our candidates or our DP enter this function.
    """
    positions = [int(p) for p in positions]
    assert len(set(positions)) == len(positions) == 8
    assert min(positions) >= 0 and max(positions) < len(frames)
    assert prompts.shape == (8, frames.shape[1], frames.shape[2])
    assert prompts.dtype == np.bool_
    with scoped_lossless_loader(predictor_module):
        state = predictor.init_state(frames, offload_video_to_cpu=True,
                                     offload_state_to_cpu=False,
                                     async_loading_frames=False)
    for j in sorted(range(8), key=lambda j: positions[j]):
        predictor.add_new_mask(state, frame_idx=positions[j], obj_id=1,
                               mask=prompts[j])
    # No tracking has happened; all eight are initial conditioning frames.
    assert set(state['temp_output_dict_per_obj'][0]['cond_frame_outputs']) == set(positions)
    boxes = {}; earliest_mask = None; empty = []
    first = min(positions)
    for reverse in (False, True):
        for frame_idx, object_ids, scores in predictor.propagate_in_video(
                state, start_frame_idx=first, reverse=reverse):
            assert object_ids == [1]
            assert scores.shape == (1, 1, frames.shape[1], frames.shape[2])
            assert torch.isfinite(scores).all()
            mask = (scores[0, 0] > 0).detach().cpu().numpy()
            if frame_idx in boxes:
                assert frame_idx == first and reverse
                assert np.array_equal(mask, earliest_mask)
                continue
            if frame_idx == first:
                earliest_mask = mask.copy()
            boxes[frame_idx] = mask_box(mask)
            if not mask.any():
                empty.append(frame_idx)
    assert set(boxes) == set(range(len(frames)))
    return np.stack([boxes[i] for i in range(len(frames))]), dict(
        conditioning_frames=sorted(positions), complete_native_grid=True,
        empty_output_frames=sorted(empty), source_fallback=False, our_DP=False,
        one_object=True, fresh_state=True)
