"""R174 registration and a private, ordinary-only replay contract."""
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
R174 = HERE.parent/'3eed_r174_strong_anchor_joint_support_20261119'
PRIVATE = HERE/'private'
RESULTS = HERE/'results'
SPEC = HERE/'protocol.json'
sys.path.insert(0, str(R174))

from scope174 import (ORDER, PLATFORMS, PRIVATE as PRIOR, R165_DATA, ROOT, array_sha,
                      json_atomic, npz_atomic, ordinary_rows, points_for, read, sha)
from predict174 import OUTPUT as ORIGINAL, lock_evaluation

PROTOCOL = read(SPEC)
N = PROTOCOL['target']['n']
ARMS = tuple(PROTOCOL['arms'])


def update_state(phase, note, receipt=None):
    state_path = ROOT/'research/STATE.json'
    state = read(state_path)
    route = state['directions']['crossplatform']
    assert route['active_round'] in (None, 'R177')
    stamp = datetime.now().astimezone().isoformat()
    r174_public = read(R174/'publication/PUBLICATION_RECEIPT.json')
    assert r174_public['commit'] == PROTOCOL['main_method']['commit'] and not r174_public['FINAL_access']
    registered = PROTOCOL['main_method'] | {
        'protocol': (R174/'protocol.json').relative_to(ROOT).as_posix(),
        'evaluation_lock_sha256': sha(PRIOR/'EVALUATION_LOCK.json'),
        'role': 'paper_and_research_main'
    }
    route.update(main_method=registered, current_working_candidate='R174_full',
                 current_working_candidate_retained=True,
                 current_method_registration=registered['protocol'],
                 active_recipe='R174 full source-trained role-conditioned decoder; target-frozen',
                 active_report=(HERE/'REPORT_ZH.md').relative_to(ROOT).as_posix(),
                 working_update_scope='no test-time parameter update', active_FINAL_access=False)
    state['R174'].update(current_main_method=True, current_main_variant='full',
                         current_main_selection='user_selected')
    done = phase.startswith('completed')
    route.update(active_round=None if done else 'R177', execution_open=not done)
    record = dict(round='R177', state=phase, note=note, main_method='R174_full',
                  updated_local=stamp, ordinary_direction_rows=N,
                  protocol=SPEC.relative_to(ROOT).as_posix(),
                  target_updates=0, source_retraining=0, FINAL_access=False)
    if receipt:
        record.update(commit=receipt['commit'], publication_receipt=(
            HERE/'publication/PUBLICATION_RECEIPT.json').relative_to(ROOT).as_posix())
        route.update(latest_completed_report=(HERE/'REPORT_ZH.md').relative_to(ROOT).as_posix(),
                     latest_completed_publication=receipt['commit'],
                     latest_public_commit=receipt['commit'],
                     latest_public_report=(
                         'https://github.com/'+receipt['repository']+'/blob/'+receipt['commit']+
                         '/'+receipt['prefix']+'/REPORT_ZH.md'))
    state['R177'] = record
    json_atomic(HERE/'STATE.json', record)
    json_atomic(state_path, state)


def input_lock():
    p = ORIGINAL/'PREDICTION_SEAL.json'
    prior = read(p)
    prior_contract = lock_evaluation()
    assert prior['n'] == N and prior['contract'] == prior_contract
    assert not prior['FINAL_access'] and not prior['target_truth_read']
    files = [SPEC, HERE/'scope177.py', HERE/'model177.py', HERE/'predict177.py',
             p, PRIOR/'EVALUATION_LOCK.json', R174/'publication/PUBLICATION_RECEIPT.json']
    cells = read(PRIOR/'EVALUATION_LOCK.json')['models']
    for s,t in ORDER:
        d = f'{s}_to_{t}'
        f = ORIGINAL/d/'SEAL.json'
        g = R165_DATA/d/'SEAL.json'
        assert sha(f)==prior['directions'][d][0] and sha(g)==read(f)['source_seal']
        assert not read(g)['protected_payload_access'] and not read(f)['FINAL_access']
        files.extend((f,g))
    lock = dict(n=N, arms=list(ARMS), source_models={s:cells[f'{s}/full'] for s in PLATFORMS},
                inputs={f.relative_to(ROOT).as_posix():sha(f) for f in files}, FINAL_access=False)
    dest = PRIVATE/'EXECUTION_LOCK.json'
    if dest.exists():
        assert read(dest)==lock
    else:
        json_atomic(dest,lock)
    return sha(dest)
