"""Protocol anchors and the one frame of return coordinates used by every readout."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
LAST = HERE.parent / '3eed_v1_ordered_landmark_witnesses_20261120'
sys.path.insert(0, str(LAST))
from paths_witness import ROOT, name, read, save, digest, SOURCES
from collect import archive_npz

WORK = HERE / 'private'
OUT = HERE / 'results'
OBS = WORK / 'observations'
PRED = WORK / 'predictions'
OLD = LAST / 'private'
SPEC = HERE / 'PROTOCOL.json'
CFG = read(SPEC)
CLASSES = tuple(CFG['classes'])
ARMS = tuple(CFG['controls'])


def panel():
    assert CFG['base'] == digest(ROOT / 'crossplatform/CURRENT_METHOD.json')
    assert CFG['weights'] == digest(ROOT / 'crossplatform/single_frame_sources.json')
    for filename, key in [('roster.json', 'roster'), ('observed/SEALED.json', 'previous_observations'),
                           ('predictions/SEALED.json', 'previous_predictions')]:
        assert CFG[key] == digest(OLD / filename)
    return read(OLD / 'roster.json')


def sample(frame):
    from runner_helpers import sample_points
    from utils.transform_waymo import transform_to_front_view
    points, _, _ = sample_points(dict(pointpath=Path(frame['points']), platform=frame['target'],
                                     seed=int(frame['seed'])))
    if frame['target'] == 'waymo':
        points, _ = transform_to_front_view(points, np.empty((0, 7)), frame['waymo_view'])
    return np.ascontiguousarray(points, dtype=np.float32)
