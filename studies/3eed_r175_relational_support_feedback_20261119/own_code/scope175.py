"""R175 source and ordinary-validation evidence contract."""
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
R173 = HERE.parent/'3eed_recurrent_role_support_decoder_20261119'
R174 = HERE.parent/'3eed_r174_strong_anchor_joint_support_20261119'
PRIVATE, RESULTS, SPEC = HERE/'private', HERE/'results', HERE/'protocol.json'
sys.path.insert(0, str(R173))

from common import (CACHE as OLD_CACHE, ORDER, PLATFORMS, PRIVATE as OLD_PRIVATE,
                    ROOT, SENTINEL, array_sha, camera_view, frame, json_atomic, npz_atomic,
                    ordinary_rows, points_for, read, recording, sha, stable, verify_inputs)

R165_DATA = ROOT/'crossplatform/round165/ordinary'
NEW_CACHE = Path('/dev/shm/3eed_R175_source_relations_20261119')
BASELINES = ('official_bbf_axis','official_bbf_pca','native_text_pca','r158_pca')
CELLS = ('full','self_geometry','pair_only','without_dual')
CONTROL = ('full_off','full_shuffled')
NAMES = (*BASELINES,*CELLS,*CONTROL)
RELATIONS = tuple(read(SPEC)['relation_kinds'])
N = 18620


def register():
    base = verify_inputs()
    previous = R174/'publication/PUBLICATION_RECEIPT.json'
    receipt = read(previous)
    assert receipt['commit'] == read(SPEC)['parent_commit'] and not receipt['FINAL_access']
    assert base['ordinary_rows'] == read(SPEC)['target_n'] == N
    contract = [SPEC, HERE/'scope175.py', HERE/'roles175.py', HERE/'source175.py', previous,
                R173/'common.py', R173/'prepare.py', R173/'evidence.py', R173/'decoder.py',
                OLD_PRIVATE/'INPUT_LOCK.json', ROOT/'research/round165/ORDINARY_FORWARD_SEAL.json',
                ROOT/'research/round165/ORDINARY_CUDA_ENDPOINT_SEAL.json',
                ROOT/'crossplatform/single_frame_sources.json']
    for p in PLATFORMS:
        folder = OLD_PRIVATE/p
        roster = read(folder/'ROSTER_SEAL.json')
        contract.extend((folder/'ROSTER_SEAL.json',folder/'FEATURE_SEAL.json'))
        for name, digest in roster['rosters'].items():
            assert sha(folder/name) == digest
            contract.append(folder/name)
    p = PRIVATE/'INPUT_LOCK.json'
    data = dict(files={x.relative_to(ROOT).as_posix():sha(x) for x in contract},n=N,
                protocol=sha(SPEC),FINAL_access=False)
    PRIVATE.mkdir(exist_ok=True,mode=0o700)
    if p.exists():
        assert read(p)==data
    else:
        json_atomic(p,data)
    return data


def check_locked():
    d = read(PRIVATE/'INPUT_LOCK.json')
    assert d['n']==N and not d['FINAL_access']
    for name,digest in d['files'].items():
        assert sha(ROOT/name)==digest, name
    return d


def state(phase,note,receipt=None):
    stamp = datetime.now().astimezone().isoformat()
    entry = dict(round='R175',state=phase,updated_local=stamp,n_expected=N,
                 protocol=SPEC.relative_to(ROOT).as_posix(),target_updates=0,
                 source_backbone_updates=0,FINAL_access=False,note=note)
    if receipt:
        entry.update(commit=receipt['commit'],
                     publication_receipt=(HERE/'publication/PUBLICATION_RECEIPT.json').relative_to(ROOT).as_posix())
    p = ROOT/'research/STATE.json'
    ledger = read(p)
    route = ledger['directions']['crossplatform']
    if phase in ('source_supervision','source_features','source_training','frozen_prediction',
                 'post_seal_scoring','publication_pending'):
        assert route['active_round'] in (None,'R175')
        route.update(active_round='R175',execution_open=True)
    elif route['active_round']=='R175':
        route.update(active_round=None,execution_open=False)
    ledger['R175']=entry
    json_atomic(HERE/'STATE.json',entry)
    json_atomic(p,ledger)
