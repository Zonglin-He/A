"""Pins and paths for the continuation of the sealed recurrent-decoder study."""
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'3eed_recurrent_role_support_decoder_20261119'
PRIVATE = HERE/'private'
RESULTS = HERE/'results'
SPEC = HERE/'protocol.json'
sys.path.insert(0, str(PARENT))

from common import (ROOT, ORDER, PLATFORMS, CACHE, PRIVATE as OLD_PRIVATE, SPEC as OLD_SPEC,
                    array_sha, camera_view, canonical, frame, json_atomic, npz_atomic,
                    ordinary_rows, points_for, read, recording, sha, stable, verify_inputs)

CELLS = ('off_off', 'role_only', 'geometry_only', 'joint')
NEW = CELLS[1:3]
BASE = 'r158_pca'
ARMS = (BASE, *CELLS)
OLD_CELL = {'off_off': 'without_feedback', 'joint': 'joint'}
THRESHOLDS = (.25, .50)
OLD_PRED = OLD_PRIVATE/'predictions'
PREDICTIONS = PRIVATE/'predictions'


def pin_prior():
    original = verify_inputs()
    evaluation = read(OLD_PRIVATE/'EVALUATION_LOCK.json')
    prediction = read(OLD_PRED/'PREDICTION_SEAL.json')
    scoring = read(OLD_PRIVATE/'SCORING_SEAL.json')
    published = read(PARENT/'publication/PUBLICATION_RECEIPT.json')
    assert published['commit'] == read(SPEC)['parent_public_commit']
    assert sha(OLD_PRIVATE/'EVALUATION_LOCK.json') == prediction['contract']
    assert sha(OLD_PRED/'PREDICTION_SEAL.json') == scoring['predictions']
    assert evaluation['input_lock'] == sha(OLD_PRIVATE/'INPUT_LOCK.json')
    assert not prediction['target_truth_used'] and not prediction['FINAL_access'] and not scoring['FINAL_access']
    assert original['ordinary_rows'] == prediction['n'] == 18620
    assert all(sha(PARENT/file) == h for file,h in evaluation['code'].items())
    paths = [SPEC, HERE/'study.py', HERE/'factor_decoder.py', HERE/'train_factorial.py',
             PARENT/'train.py', OLD_PRIVATE/'EVALUATION_LOCK.json',
             OLD_PRIVATE/'INPUT_LOCK.json', OLD_PRED/'PREDICTION_SEAL.json',
             OLD_PRIVATE/'SCORING_SEAL.json', PARENT/'results/independent_audit.json']
    for source in PLATFORMS:
        old = OLD_PRIVATE/source
        seal = read(old/'MODEL_SEAL.json')
        feat = old/'FEATURE_SEAL.json'
        paths.extend((old/'MODEL_SEAL.json', feat, old/'ROSTER_SEAL.json'))
        for name in OLD_CELL.values():
            receipt = old/f'{name}_TRAINING.json'
            model = old/f'{name}.pth'
            training = read(receipt)
            assert sha(receipt) == seal['arms'][name]
            assert training['inputs']['trainer'] == sha(PARENT/'train.py')
            assert training['inputs']['decoder'] == sha(PARENT/'decoder.py')
            assert training['inputs']['features'] == sha(feat)
            assert sha(model) == evaluation['checkpoints'][f'{source}/{name}'] == training['checkpoint']
            paths.extend((model, receipt))
    current = {'files': {p.relative_to(ROOT).as_posix():sha(p) for p in paths},
               'recipe': read(OLD_SPEC)['source_protocol']['training'], 'n': prediction['n'], 'FINAL_access': False}
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    lock = PRIVATE/'INPUT_LOCK.json'
    if lock.exists():
        assert read(lock) == current
    else:
        json_atomic(lock, current)
    return current


def check_lock():
    lock = read(PRIVATE/'INPUT_LOCK.json')
    assert not lock['FINAL_access']
    for f,h in lock['files'].items():
        assert sha(ROOT/f) == h, f'Study dependency changed: {f}'
    return lock


def state(phase, note, receipt=None):
    key = 'role_geometry_factorial_20261119'
    stamp = datetime.now().astimezone().isoformat()
    d = dict(study=key, state=phase, updated_local=stamp, n_expected=18620,
             target_updates=0, source_backbone_updates=0, protected_access=False, note=note,
             protocol=(HERE/'protocol.json').relative_to(ROOT).as_posix())
    if receipt is not None:
        d['public_commit'] = receipt['commit']
        d['publication_receipt'] = (HERE/'publication/PUBLICATION_RECEIPT.json').relative_to(ROOT).as_posix()
    p = ROOT/'research/STATE.json'
    registry = read(p)
    direction = registry['directions']['crossplatform']
    if phase in ('source_training', 'frozen_prediction', 'post_seal_scoring', 'publication_pending'):
        assert direction['active_round'] in (None, key)
        direction.update(active_round=key, execution_open=True)
    elif direction['active_round'] == key:
        direction.update(active_round=None, execution_open=False)
    registry[key] = d
    json_atomic(HERE/'STATE.json', d)
    json_atomic(p, registry)
