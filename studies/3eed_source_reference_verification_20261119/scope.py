"""Paths, input provenance and sealed groups for source reference verification."""
from pathlib import Path
import hashlib
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIVATE = HERE / 'private'
RESULTS = HERE / 'results'
SPEC = HERE / 'protocol.json'
PRIOR = ROOT / 'studies/codex_overnight_20261119'
sys.path.insert(0, str(ROOT / 'research/round171'))
from scope171 import PLATFORMS, atomic_json, read, sha

PINNED = ROOT / 'research/round168/SOURCE_INPUT_SEAL.json'
ARCHIVED = ROOT / 'crossplatform/round168/source_calibration_PRIVATE/ROSTER.json'


def stable(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()


def train_path(platform):
    path = ROOT / 'crossplatform/round161' / f'{platform}_train_PRIVATE.json'
    old = read(PINNED)
    assert not old['FINAL_access'] and sha(path) == old['monitored_source_manifests'][platform]['train']
    return path


def former_frames():
    from scope170 import R159, REC_INPUT

    cert = read(R159/'RECORDING_INPUT_SEAL.json')
    assert sha(REC_INPUT) == cert['input_sha256'] and not cert['FINAL_access']
    assert sha(ARCHIVED) == read(PINNED)['source_roster_sha256']
    a = PRIOR/'private/appearance/source'
    assert sha(a/'ROSTER.json') == read(a/'ROSTER_SEAL.json')['roster']
    points, images, frames, file_paths = set(), set(), set(), set()
    for x in read(REC_INPUT) + read(ARCHIVED) + read(a/'ROSTER.json'):
        row = x.get('row',x)
        frames.add(Path(row['metapath']).resolve())
        file_paths.update(Path(row[k]).resolve() for k in ('pointpath','imagepath'))
        if 'point_sha256' in row:
            points.add(row['point_sha256'])
        if 'image_sha256' in row:
            images.add(row['image_sha256'])
        if 'scan' in x:
            points.add(x['scan'])
        if 'cam' in x:
            images.add(x['cam'])
    return frames, file_paths, points, images
