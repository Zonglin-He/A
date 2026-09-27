"""Expanded dense evaluation: metadata-only decoding, separately invoked labels.

Uses the exact native validation box transform on a dummy frame (geometry is
content-independent). No annotation is read by decode_raw or the predictor.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from vg_tta.unanchored_dense_shift_data_v1 import sampled_ids, FFMPEG, FFMPEG_SHA, sha


def decode_raw(query):
    assert sha(query['video_path']) == query['video_sha256']
    assert sha(FFMPEG) == FFMPEG_SHA
    if query['kind'] == 'vidstg':
        from scripts.evaluate_fullspan_scale_shift_corruptions_v1 import _decode_raw_metadata_only
        raw, ids = _decode_raw_metadata_only(query, Path(query['video_path']))
    else:
        import ffmpeg
        ids = sampled_ids(query, 'hcstvg')
        data, _ = (ffmpeg.input(query['video_path'], ss=0, t=20)
            .filter('fps', fps=len(ids)/20.)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True, cmd=str(FFMPEG)))
        raw = np.frombuffer(data, np.uint8).reshape(-1, query['height'], query['width'], 3).copy()
    assert ids == query['frame_ids'] and len(raw) == len(ids)
    if query.get('raw_pixel_sha256'):
        assert hashlib.sha256(np.ascontiguousarray(raw)).hexdigest() == query['raw_pixel_sha256']
    return raw, ids


@lru_cache(maxsize=8)
def read_annotation(path, expected_sha):
    assert sha(path) == expected_sha
    return json.loads(Path(path).read_text())


def build_labels(annotation, row, kind, frame_ids):
    """Explicit GT-only helper. Call after prediction barrier or for labeled oracle."""
    from vg_tta.tubedetr_runtime import add_repo_to_path
    add_repo_to_path(Path(__file__).resolve().parents[1] / 'external/TubeDETR')
    from datasets.video_transforms import make_video_transforms, prepare
    width, height = int(row['width']), int(row['height'])
    trajectory = (row['trajectory'] if kind == 'hcstvg' else
                  annotation['trajectories'][str(row['original_video_id'])][str(row['target_id'])])
    targets, positive = [], []
    for pos, frame in enumerate(frame_ids):
        inside = int(row['tube_start_frame']) <= frame < int(row['tube_end_frame'])
        if inside:
            ann = ({'bbox': trajectory[frame-int(row['tube_start_frame'])]} if kind == 'hcstvg'
                   else trajectory[str(frame)])
            positive.append(pos)
            targets.append(prepare(width, height, [ann]))
        else:
            targets.append(prepare(width, height, []))
    if not positive:
        raise ValueError('No sampled GT foreground; do not silently omit query')
    dummy = np.zeros((1, height, width, 3), dtype=np.uint8)
    _, targets = make_video_transforms('val', cautious=True, resolution=224)(dummy, targets)
    assert len(targets) == len(frame_ids)
    return targets, (positive[0], positive[-1]), (int(row['tube_start_frame']), int(row['tube_end_frame']))


class DenseBridge:
    def __init__(self, spec):
        self.spec = spec

    decode_raw = staticmethod(decode_raw)

    def labels(self, query):
        data = read_annotation(self.spec['annotation'], self.spec['annotation_sha256'])
        rows = data if query['kind'] == 'hcstvg' else data['videos']
        row = rows[int(query['index'])]
        assert row['caption'] == query['caption']
        assert str(row['original_video_id']) == str(query['original_video_id'])
        return build_labels(data, row, query['kind'], query['frame_ids'])

    def score(self, query, prediction):
        from vg_tta.metrics import compute_stvg_metrics, interval_from_logits
        targets, interval, physical = self.labels(query)
        metrics = compute_stvg_metrics(prediction['pred_boxes'], targets,
            interval_from_logits(prediction['pred_sted']), interval,
            frame_ids=query['frame_ids'], gt_frame_interval=physical)
        return {'index': query['index'], 'source': query['source'],
                'metric': metrics['vIoU_corrected'], 'metrics': metrics}
