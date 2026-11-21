"""Label-free camera projection and four signed image-edge measurements."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '3eed_qwen_four_scale_recording_confirmation_20261121'
SPAN = HERE.parent / '3eed_visible_amodal_span_20261121'
PUBLIC, PRIVATE = HERE / 'results', HERE / 'private'
PROTOCOL = HERE / 'PROTOCOL.json'
sys.path.insert(0, str(SPAN))
from common import inputs, read, write, digest
from geometry import camera, extent_corners, rigid_xyz, raw_to_model_points, project_model_points
from score import rotation, projected, rectangle, given_rect

TOL = 4.
SIDES = ('left', 'top', 'right', 'bottom')
EDGE_PAIRS = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]


def edges(window, rect):
    w = np.asarray(window)
    b = np.asarray(rect)
    excess = np.r_[b[:2] - w[:2], w[2:] - b[2:]]
    qwh = w[2:] - w[:2]
    overlap = np.maximum(np.minimum(w[2:], b[2:]) - np.maximum(w[:2], b[:2]), 0.)
    return dict(signed=excess.tolist(), worst_px=float(max(0., excess.max())),
                worst_fraction=float(max(0., (excess / np.tile(qwh, 2)).max())),
                coverage=float(overlap.prod() / qwh.prod()))


def projection(box, mode, spec, size, mat):
    if box is None:
        return dict(status='missing_box')
    local = extent_corners(box[:3], box[3:6], rotation(box[6]))
    depth = rigid_xyz(local, mat)[:, 2]
    if depth.min() <= .1:
        return dict(status='behind_camera' if depth.max() <= .1 else 'crosses_near_plane')
    primary = projected(box, spec, mat, size)
    # Alternate path applies the actual loader's sensor-to-model transform and
    # calibrates the corners and the interiors of all twelve edges together.
    t = np.linspace(0., 1., 33)[:, None]
    edge_pts = np.concatenate([local[i] + t * (local[j] - local[i]) for i, j in EDGE_PAIRS])
    samples = np.concatenate([local, edge_pts])
    model = raw_to_model_points(samples, mode['target'], mode['suffix'])
    uv = project_model_points(model, spec, size)['uv']
    assert np.isfinite(uv).all()
    alternate = rectangle(uv[:8], size)
    assert (alternate is None) == (primary is None)
    dense = rectangle(uv, size)
    if primary is None:
        return dict(status='off_image', dense=None if dense is None else dense.tolist())
    assert dense is not None
    gap = float(np.max(np.abs(alternate - primary)))
    assert gap < 1e-5
    return dict(status='ok', rect=primary.tolist(), dense=dense.tolist(), crosscheck_px=gap)


def raw_window(pixel, size):
    if pixel is None:
        return None
    return given_rect(np.asarray(pixel) * np.tile(size, 2) / 1000., size)


def run():
    panel, frozen = inputs()
    PUBLIC.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True)
    out = {}
    for f in panel:
        uid = f['uid']
        entry = frozen[uid]
        spec, size, mat = camera(f)
        rect = raw_window(entry['pixel'], size)
        pred = projection(entry['boxes']['H'], f, spec, size, mat)
        out[uid] = dict(window=None if rect is None else rect.tolist(), image_size=size.tolist(),
                        projection=pred, selected=bool(entry['selected']),
                        distortion_assumed_zero=spec.validated()[3],
                        edge_measure=edges(rect, pred['rect']) if rect is not None and pred['status']=='ok' else None)
    write(PRIVATE / 'observations.json', out)
    write(PUBLIC / 'observation_seal.json', dict(frames=len(out), protocol=digest(PROTOCOL),
          code=digest(__file__), parent_seal=digest(BASE/'results/prediction_seal.json'),
          observations=digest(PRIVATE/'observations.json'), target_labels_read=False, new_model_calls=0, FINAL_access=False))
    print('Sealed image-only and frozen predictions', len(out))


if __name__ == '__main__':
    run()
