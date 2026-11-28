"""Runtime, source isolation, and the registered frozen geometry interface."""
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO.parent
UPSTREAM = ROOT/'assets/upstream-3eed'
DATA = ROOT/'assets/data/3eed_dataset'
RUN = ROOT/'runs/waymo-posttraining-20261128'
MODEL = ROOT/'assets/models/Qwen3-VL-4B-Instruct/ebb281ec70b05090aa6165b016eac8ec08e71b17'
REGISTRY = REPO/'QWEN_CURRENT_METHOD.json'
sys.path[:0] = [str(REPO/'tools/3eed_recovery'), str(HERE.parent/'3eed_inference_time_vlm_20261128')]
from run_qwen_3eed import VIEWS, check_geometry, digest, publish, read, source_paths

CFG = read(HERE/'protocol.json')
JOINT = ('Locate the single object referred to in the image. Return one JSON array containing one object with '
         '"bbox_2d":[xmin,ymin,xmax,ymax] normalized from 0 to 1000, '
         '"bbox_3d":[x,y,z,length,width,height,roll,pitch,yaw] in the camera frame with angles in units of pi, '
         'and "label" as the predicted object class. Return only JSON. Description: ')
SHAPE = ('Locate the object in the image and use its calibrated physical measurements below. '
         'Predict the full length, width, height in metres and the axial yaw in units of pi in the LiDAR frame. '
         'Return only {"shape":[length,width,height,yaw]}. Description: ')
FENCE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.I)


def source_guard():
    """Permit only current source artifacts plus the pinned software and model assets."""
    blocked = tuple(str(path)+'/' for path in [DATA/'drone', DATA/'quad', ROOT/'assets/archives',
                                              ROOT/'assets/wildrefer-source'])
    run_root = str(ROOT/'runs')+'/'
    allowed = str(RUN)+'/'
    target_splits = {f'{p}_{s}.txt' for p in ('drone', 'quad') for s in ('train', 'val')}

    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, Path, bytes)):
            return
        path = os.path.abspath(os.fsdecode(args[0]))
        name = os.path.basename(path)
        if (path.startswith(blocked) or (path.startswith(run_root) and not path.startswith(allowed))
                or name in target_splits or (path.startswith(allowed) and name.startswith(('test-', 'scores-')))):
            raise PermissionError(f'Waymo source process denied: {path}')
    sys.addaudithook(audit)


def setup():
    source_paths(UPSTREAM)
    check_geometry()
    reg = read(REGISTRY)
    assert reg['method'] == 'ray_equivalent_height_only_2d_rank'
    for key in ('prediction_entry', 'method_lock'):
        lock = reg[key]
        assert digest(REPO/lock['file']) == lock['sha256']
    for file, sha in read(REPO/reg['method_lock']['file'])['config']['historical_geometry'].items():
        assert digest(REPO/file) == sha
    return reg


def readout(text, task):
    from infer import parse
    value, status = parse(text, task)
    return dict(raw=text, value=value, status=status)


def replies(text):
    return {key:readout(text, key) for key in ('2d','3d')}


def valid_shape(text):
    try:
        value = np.asarray(json.loads(FENCE.sub('', text.strip()))['shape'], float)
        if value.shape == (4,) and np.isfinite(value).all() and min(value[:3]) > 0 and abs(value[3]) <= 1:
            return value
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    return None


def truth_iou(boxes, truth):
    from score import canonical_box, score_boxes
    evaluated = [(k, canonical_box(v)) for k,v in boxes.items() if v is not None]
    output = dict.fromkeys(boxes, 0.)
    if evaluated:
        vals, error = score_boxes(np.stack([v for _,v in evaluated]), np.asarray(truth))
        assert error < 2e-10 and np.isfinite(vals).all()
        output.update({k:float(v) for (k,_),v in zip(evaluated, vals, strict=True)})
    return output


def backend(frame, parsed):
    from live_2d import predict
    return predict(geometry_frame(frame), parsed)


def geometry_frame(frame):
    """Materialize the same camera-view field as the established manifest loader."""
    return dict(frame,waymo_view=VIEWS[frame['suffix']])


def compact(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def source_bank(part):
    manifest = RUN/f'source-{part}.json'
    receipt = read(RUN/'source-seal.json')
    assert receipt['protocol'] == digest(HERE/'protocol.json')
    assert digest(manifest) == receipt[part]['file']
    rows = read(manifest)
    assert all(row['frame']['target']=='waymo' and Path(row['frame']['rgb']).is_relative_to(DATA/'waymo') for row in rows)
    for row in rows:
        row['frame']=geometry_frame(row['frame'])
    return rows
