"""Pinned public interfaces used to replay the impact of noun-conditioned geometry."""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / '3eed_v1_shared_scene_role_inference_20261120'
sys.path.insert(0, str(SCENE))
import shared_paths as previous
from shared_paths import ROOT, SOURCES, CLASSES, LEX, name, read, save, digest, archive_npz
from registry_swap import ordinary, frame
WORK = HERE / 'private'
OUT = HERE / 'results'
SPEC = HERE / 'PROTOCOL.json'
OBS = WORK / 'word'
PRED = WORK / 'predictions'
ARMS = ('v1', 'subject_only', 'frozen_words', 'scene_template', 'local_shape',
        'wrong_noun', 'wrong_reference', 'image_center', 'image_extreme')
def tag(value):
    return hashlib.sha256(repr(('local-noun-shape/1', value)).encode()).hexdigest()[:22]
def rec_key(f):
    return f['target'], Path(f['calibration']).resolve().parent.parent.name
