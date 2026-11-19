"""Reuse the R175/R165 locks and register the isolated diagnostic run."""
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
R175 = HERE.parent / '3eed_r175_relational_support_feedback_20261119'
FACTOR = HERE.parent / '3eed_role_geometry_factorial_20261119'
sys.path.insert(0, str(R175))

from scope175 import (N, ORDER, PLATFORMS, PRIVATE as PRIOR, R165_DATA, ROOT,
                      array_sha, camera_view, frame, json_atomic, npz_atomic,
                      ordinary_rows, points_for, read, recording, sha)
from predict175 import OUTPUT as ORIGINAL, PRED_ARMS

PRIVATE = HERE / 'private'
RESULTS = HERE / 'results'
SPEC = HERE / 'protocol.json'


def state(phase, note, receipt=None):
    file = ROOT / 'research/STATE.json'
    ledger = read(file)
    route = ledger['directions']['crossplatform']
    assert route['active_round'] in (None, 'R176')
    done = phase.startswith('completed')
    route.update(active_round=None if done else 'R176', execution_open=not done)
    value = dict(round='R176', state=phase, updated_local=datetime.now().astimezone().isoformat(),
                 n_expected=N, protocol=SPEC.relative_to(ROOT).as_posix(), source_backbone_updates=0,
                 target_updates=0, FINAL_access=False, note=note)
    if receipt:
        value['commit'] = receipt['commit']
        value['publication_receipt'] = (HERE/'publication/PUBLICATION_RECEIPT.json').relative_to(ROOT).as_posix()
    ledger['R176'] = value
    json_atomic(HERE/'STATE.json', value)
    json_atomic(file, ledger)


def input_lock():
    from predict175 import verify
    from score175 import original

    verify()
    receipt = R175/'publication/PUBLICATION_RECEIPT.json'
    assert read(receipt)['commit'] == read(SPEC)['parent_commit']
    assert not read(receipt)['FINAL_access']
    sealed_scores = PRIOR/'SCORING_SEAL.json'
    scores = read(sealed_scores)
    paths = [SPEC, HERE/'scope176.py', HERE/'diagnose176.py', receipt, sealed_scores,
             ORIGINAL/'PREDICTION_SEAL.json']
    for s, t in ORDER:
        route = f'{s}_to_{t}'
        p = PRIOR/'score'/f'{route}.npz'
        assert sha(p) == scores['scored'][route]
        assert len(original(route)[0]) == len(ordinary_rows(t))
        paths.extend((p, R165_DATA/route/'SEAL.json', R165_DATA/route/'BBF_CUDA_RECEIPT.json'))
    return dict(n=N, FINAL_access=False,
                files={p.relative_to(ROOT).as_posix():sha(p) for p in paths})


def locked():
    lock = read(PRIVATE/'INPUT_LOCK.json')
    assert lock['n'] == N and not lock['FINAL_access']
    assert all(sha(ROOT/p) == digest for p, digest in lock['files'].items())
    return lock
