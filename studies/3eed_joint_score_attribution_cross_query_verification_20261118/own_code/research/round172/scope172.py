"""Anchor the analysis to the sealed R171 candidates and prior visual identities."""
from datetime import datetime
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R171 = ROOT / 'research/round171'
WORK = ROOT / 'crossplatform/round172'
OUT = HERE / 'results'
PROTOCOL = HERE / 'experiment/PROTOCOL.json'
PARENT = 'fa61a76d10e31790e2be031a77126814b8010643'

sys.path.insert(0, str(R171))
from scope171 import (ARMS, INPUT as NEW_INPUT, PLATFORMS, PRIOR_INPUT, R170, SPLITS, WORK as R171_WORK,
                      atomic_json, read, sha)


def lock(phase, sources):
    p = read(PROTOCOL)
    assert p['parent_commit'] == PARENT
    assert read(R171 / 'publication/PUBLICATION_RECEIPT.json')['commit'] == PARENT
    statuses = [read(R171 / f) for f in ('PREDICTION_SEAL.json', 'EVALUATION_AUDIT.json', 'SCALAR_AUDIT.json')]
    assert not any(v['FINAL_access'] for v in statuses)
    previous = read(R171 / 'PREDICTION_LOCK.json')['files']
    for name in ('assignment171.py', 'predict171.py', 'scope171.py'):
        key = f'research/round171/{name}'
        assert previous[key] == sha(ROOT / key)
    roster_pins = ((PRIOR_INPUT, R170 / 'INPUT_SEAL.json'), (NEW_INPUT, R171 / 'INPUT_SEAL.json'))
    for path, seal in roster_pins:
        assert read(seal)['registry'] == sha(path)
    paths = [PROTOCOL, HERE / 'scope172.py', *(HERE / f for f in sources),
             R171 / 'PREDICTION_SEAL.json', R171 / 'EVALUATION_AUDIT.json', R171 / 'SCALAR_AUDIT.json',
             R171 / 'VISUAL_TRUTH_SEAL.json', R170 / 'VISUAL_TRUTH_SEAL.json',
             R171 / 'INPUT_SEAL.json', R170 / 'INPUT_SEAL.json', R171 / 'PREDICTION_LOCK.json',
             R171 / 'SCORING_LOCK.json', R171 / 'publication/PUBLICATION_RECEIPT.json',
             ROOT / 'research/round167/retrieval167.py', ROOT / 'research/round169/requery169.py']
    result = dict(phase=phase, files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                  FINAL_access=False)
    dest = HERE / f'{phase.upper()}_LOCK.json'
    if dest.exists():
        assert read(dest) == result
    else:
        atomic_json(dest, result)
    return sha(dest)


def status(phase, **details):
    path = ROOT / 'research/STATE.json'
    state = read(path)
    assert state['R161']['state'] == 'paused_by_user'
    assert state['R171']['public_commit'] == PARENT
    stamp = datetime.now().astimezone().isoformat()
    r = dict(round='R172', state=phase, last_updated=stamp, model_forwards=0, source_training_runs=0,
             target_parameter_updates=0, FINAL_access=False, **details)
    atomic_json(HERE / 'STATE.json', r)
    state['R172'] = r
    state.update(current_round='R172', current_stage='R172_PAIR_ATTRIBUTION_AND_CROSS_QUERY_VERIFICATION',
                 manual_active_state='round172/STATE.json', run_state='R172_'+phase.upper(), updated_local=stamp)
    active = phase in ('sealing_evidence', 'attribution', 'paired_readout')
    state['directions']['crossplatform'].update(active_round='R172' if active else None, execution_open=active)
    atomic_json(path, state)
