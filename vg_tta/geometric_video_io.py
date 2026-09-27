"""Label-free original-pixel view construction matching native HC-STVG decoding."""
from pathlib import Path
import numpy as np
import torch


def decode_hc_video(path, *, width, height, sample_count):
    """Uses only video path/dimensions/sample count; no temporal or box labels."""
    import ffmpeg
    if min(width, height, sample_count) <= 0:
        raise ValueError('positive dimensions and sample_count required')
    out, _ = (ffmpeg.input(str(Path(path)), ss=0, t=20)
              .filter('fps', fps=sample_count / 20)
              .output('pipe:', format='rawvideo', pix_fmt='rgb24')
              .run(capture_stdout=True, quiet=True))
    frames = np.frombuffer(out, np.uint8).reshape(-1, height, width, 3)
    if len(frames) != sample_count:
        raise RuntimeError('raw decoded frame count differs from locked temporal sampling')
    return frames


def normalize_raw_view(frames, *, resolution, crop=None):
    """Apply upstream inference resize/normalization directly to decoded pixels."""
    from datasets.video_transforms import make_video_transforms
    if crop is not None:
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= frames.shape[2] and 0 <= y0 < y1 <= frames.shape[1]):
            raise ValueError('crop lies outside original image')
        frames = frames[:, y0:y1, x0:x1]
    video, _ = make_video_transforms('val', cautious=True, resolution=resolution)(frames, None)
    if video.ndim != 4 or video.shape[0] != 3 or not torch.isfinite(video).all():
        raise RuntimeError('invalid normalized video')
    return video


def valid_boxes(boxes):
    """Reject malformed predictions; does not assert semantic correctness."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError('expected T x 4 boxes')
    return (torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 1e-4).all(-1)
            & (boxes[:, :2] >= 0).all(-1) & (boxes[:, :2] <= 1).all(-1))


def valid_crop_boxes(boxes, crop, image_hw, margin=0.015):
    """Reject boxes truncated by artificial crop edges, not true image edges."""
    h, w = image_hw
    x0, y0, x1, y1 = crop
    left_top = boxes[:, :2] - boxes[:, 2:] / 2
    right_bottom = boxes[:, :2] + boxes[:, 2:] / 2
    valid = valid_boxes(boxes)
    if x0 > 0: valid &= left_top[:, 0] > margin
    if y0 > 0: valid &= left_top[:, 1] > margin
    if x1 < w: valid &= right_bottom[:, 0] < 1 - margin
    if y1 < h: valid &= right_bottom[:, 1] < 1 - margin
    return valid
