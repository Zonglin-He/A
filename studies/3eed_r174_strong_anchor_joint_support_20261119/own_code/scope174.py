"""Stable paths and frozen inputs for the strong-anchor study."""
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
R173 = HERE.parent/'3eed_recurrent_role_support_decoder_20261119'
FACTOR = HERE.parent/'3eed_role_geometry_factorial_20261119'
PRIVATE = HERE/'private'
RESULTS = HERE/'results'
SPEC = HERE/'protocol.json'
sys.path.insert(0,str(R173))

from common import (ROOT, PLATFORMS, ORDER, SENTINEL, PRIVATE as OLD_PRIVATE, SPEC as OLD_SPEC,
                    CACHE as OLD_CACHE, array_sha, camera_view, canonical, frame, json_atomic,
                    npz_atomic, ordinary_rows, points_for, read, recording, sha, stable, verify_inputs)

CELLS = ('unary','self_only','cross_only','full')
NAMES = ('r158_pca',*CELLS)
THRESHOLDS = (.25,.50)
R97_DATA = ROOT/'crossplatform/round97'
R165_DATA = ROOT/'crossplatform/round165/ordinary'
OLD_PRED = OLD_PRIVATE/'predictions'
SYNTH_CACHE = Path('/dev/shm/3eed_R174_source_synthetic_20261119')


def register():
    prior=verify_inputs()
    lock=read(OLD_PRIVATE/'EVALUATION_LOCK.json')
    seal=read(OLD_PRED/'PREDICTION_SEAL.json')
    out=read(FACTOR/'publication/PUBLICATION_RECEIPT.json')
    assert out['commit']==read(SPEC)['parent_commit']
    assert all(sha(R173/file)==digest for file,digest in lock['code'].items())
    assert prior['ordinary_rows']==seal['n']==read(SPEC)['target']['n']
    assert not (seal['FINAL_access'] or seal['target_truth_used'] or read(SPEC)['target']['FINAL_access'])
    files=[SPEC,HERE/'scope174.py',HERE/'capture_source174.py',HERE/'anchors174.py',
           OLD_PRIVATE/'INPUT_LOCK.json',OLD_PRIVATE/'EVALUATION_LOCK.json',OLD_PRED/'PREDICTION_SEAL.json',
           ROOT/'research/round165/ORDINARY_FORWARD_SEAL.json',R97_DATA/'MATRIX_SEALED.json',
           ROOT/'research/round158/grounding158.py', ROOT/'research/round76/prepare76.py',
           ROOT/'research/round89/core89.py',ROOT/'research/round90/core90.py']
    for src in PLATFORMS:
        p=OLD_PRIVATE/src
        info=read(p/'ROSTER_SEAL.json')
        files.extend((p/'ROSTER_SEAL.json',p/'FEATURE_SEAL.json'))
        for filename,digest in info['rosters'].items():
            f=p/filename
            assert sha(f)==digest
            files.append(f)
    box=dict(files={x.relative_to(ROOT).as_posix():sha(x) for x in files},n=seal['n'],
             train=read(OLD_SPEC)['source_protocol']['training'],FINAL_access=False)
    PRIVATE.mkdir(exist_ok=True,mode=0o700)
    p=PRIVATE/'INPUT_LOCK.json'
    if p.exists():
        assert read(p)==box
    else:
        json_atomic(p,box)
    return box


def check_locked():
    p=read(PRIVATE/'INPUT_LOCK.json')
    assert p['n']==18620 and not p['FINAL_access']
    for f,h in p['files'].items():
        assert sha(ROOT/f)==h, f'Changed dependency {f}'
    return p


def state(phase,note,receipt=None):
    tag='R174'
    stamp=datetime.now().astimezone().isoformat()
    row=dict(round=tag,state=phase,updated_local=stamp,target_updates=0,source_backbone_updates=0,
             FINAL_access=False,n_expected=18620,note=note,
             protocol=SPEC.relative_to(ROOT).as_posix())
    if receipt:
        row['commit']=receipt['commit']
        row['publication_receipt']=(HERE/'publication/PUBLICATION_RECEIPT.json').relative_to(ROOT).as_posix()
    p=ROOT/'research/STATE.json'
    doc=read(p)
    route=doc['directions']['crossplatform']
    if phase in ('source_features','source_training','frozen_prediction','post_seal_scoring','publication_pending'):
        assert route['active_round'] in (None,tag)
        route.update(active_round=tag,execution_open=True)
    elif route['active_round']==tag:
        route.update(active_round=None,execution_open=False)
    doc[tag]=row
    json_atomic(HERE/'STATE.json',row)
    json_atomic(p,doc)
