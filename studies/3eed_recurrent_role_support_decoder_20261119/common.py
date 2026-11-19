"""Read-only benchmark and source contracts plus atomic study artifacts."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIVATE = HERE / 'private'
CACHE = Path('/dev/shm/3eed_recurrent_role_support_20261119')
RESULTS = HERE / 'results'
SPEC = HERE / 'protocol.json'
R165 = ROOT / 'research/round165'
R97 = ROOT / 'research/round97'
PLATFORMS = ('waymo', 'drone', 'quad')
ORDER = tuple((s, t) for s in PLATFORMS for t in PLATFORMS if s != t)
ARMS = ('joint', 'without_feedback', 'without_rgb', 'fixed_proposals')
METRICS = ('mIoU', 'Acc25', 'Acc50')
SENTINEL = ' . not mentioned'

sys.path[:0] = [str(ROOT / p) for p in (
    'crossplatform/round3/p0/upstream', 'crossplatform/round7/experiment',
    'crossplatform/round7', 'research/round5/audit',
    'research/round6/source_validation_scoring', 'research/round160', 'research/round97',
    'research/round76', 'crossplatform/round51', '.')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest()


def stable(*parts):
    h = hashlib.sha256()
    for value in parts:
        h.update(str(value).encode('utf-8'))
        h.update(b'\0')
    return h.hexdigest()


def json_atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')
    os.replace(temp, path)


def npz_atomic(path, **fields):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f, **fields)
    os.replace(temp, path)


def canonical(text):
    return ' '.join(text.replace(',', ' ,').split())


def recording(row):
    return Path(row['metapath']).resolve().parent.parent.as_posix()


def frame(row):
    return Path(row['metapath']).resolve().as_posix()


def camera_view(row):
    return int(row['suffix'] or 0) if row['platform'] == 'waymo' else 0


def points_for(row):
    from runner_helpers import sample_points
    from utils.transform_waymo import transform_to_front_view

    points, ids, _ = sample_points(row)
    if row['platform'] == 'waymo':
        views = ('F', 'FL', 'FR', 'SL', 'SR')
        points, _ = transform_to_front_view(points, np.empty((0, 7)), views[camera_view(row)])
    return np.ascontiguousarray(points, np.float32), ids


def target_path(target):
    return R97 / 'input' / f'TARGET_{target.upper()}_PRIVATE.json'


def source_path(source):
    return ROOT / 'crossplatform/round161' / f'{source}_train_PRIVATE.json'


def make_lock():
    r165 = read(R165 / 'INPUT_LOCK.json')
    forward = read(R165 / 'ORDINARY_FORWARD_SEAL.json')
    endpoints = read(R165 / 'ORDINARY_CUDA_ENDPOINT_SEAL.json')
    reg97 = read(R97 / 'experiment/DESIGN_LOCK.json')['inputs_sha256']
    reg168 = read(ROOT / 'research/round168/SOURCE_INPUT_SEAL.json')
    assert not (r165['protected_payload_access'] or forward['FINAL_access'] or
                endpoints['FINAL_access'] or reg168['FINAL_access'])
    paths = {}
    for target in PLATFORMS:
        source, allowed = source_path(target), target_path(target)
        paths[str(allowed.relative_to(ROOT))] = reg97[str(allowed.relative_to(ROOT))]
        paths[str(source.relative_to(ROOT))] = reg168['monitored_source_manifests'][target]['train']
    proofs = (R165 / 'INPUT_LOCK.json', R165 / 'ORDINARY_FORWARD_SEAL.json',
              R165 / 'ORDINARY_CUDA_ENDPOINT_SEAL.json',
              ROOT / 'crossplatform/single_frame_sources.json')
    paths.update((p.relative_to(ROOT).as_posix(), sha(p)) for p in proofs)
    sizes = read(SPEC)['target_protocol']['counts_by_target_per_source']
    assert forward['n'] == endpoints['n'] == 2 * sum(sizes.values()) == 18620
    out = {'protocol': sha(SPEC), 'input_pins': paths, 'target_counts': sizes,
           'ordinary_rows': forward['n'], 'protected_inputs_opened': False}
    path = PRIVATE / 'INPUT_LOCK.json'
    if path.exists():
        assert read(path) == out
    else:
        json_atomic(path, out)
    verify_inputs()
    return out


@lru_cache(maxsize=1)
def verify_inputs():
    lock = read(PRIVATE / 'INPUT_LOCK.json')
    assert sha(SPEC) == lock['protocol']
    for p, digest in lock['input_pins'].items():
        assert sha(ROOT / p) == digest, f'Pinned input {p} changed'
    return lock


@lru_cache(maxsize=3)
def ordinary_rows(target):
    lock = verify_inputs()
    rows = read(target_path(target))
    assert len(rows) == lock['target_counts'][target]
    assert all(r['platform'] == target and canonical(r['caption']) == r['caption'] for r in rows)
    return rows
