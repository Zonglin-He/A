"""Register the next reference-evidence experiment on the sealed ordinary population."""
from datetime import datetime
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R168 = ROOT / 'research/round168'
WORK = ROOT / 'crossplatform/round169'
OUT = HERE / 'results'
sys.path.insert(0, str(R168))
from scope168 import CAL, ORDER, PLATFORMS, R166, R167, atomic_json, read, roster, sha
from score168 import verify_seals

PROTOCOL = HERE / 'experiment/PROTOCOL.json'
CACHE_ARMS = ('independent', 'old_left', 'new_left', 'permuted_left')
QUERY_ARMS = ('A_original_reference_tokens', 'B_canonical_original_subject', 'C_reference_phrase',
              'D_reference_as_subject')
PANEL = WORK / 'panel_PRIVATE'
SOURCE_PANEL = WORK / 'source_review_PRIVATE'
N = 18620
PANEL_N = 54


def lock(phase, filenames):
    verify_seals()
    policy = read(PROTOCOL)
    assert tuple(policy['cache_arms']) == CACHE_ARMS and tuple(policy['query_arms']) == QUERY_ARMS
    paths = (PROTOCOL, HERE / 'scope169.py', R168 / 'PREDICTION_SEAL.json', R168 / 'EVALUATION_AUDIT.json',
             *(HERE / x for x in filenames))
    payload = dict(phase=phase, sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, FINAL_access=False)
    dest = HERE / (phase.upper() + '_LOCK.json')
    if dest.exists():
        assert read(dest) == payload
    else:
        atomic_json(dest, payload)
    return sha(dest)


def status(phase, **kwargs):
    path = ROOT / 'research/STATE.json'
    state = read(path)
    assert state['R161']['state'] == 'paused_by_user'
    assert state['R168']['state'] == 'completed_published_verified'
    stamp = datetime.now().astimezone().isoformat()
    entry = dict(round='R169', state=phase, last_updated=stamp, ordinary_direction_records=N,
                 FINAL_access=False, source_training_runs=0, target_parameter_updates=0, **kwargs)
    atomic_json(HERE / 'STATE.json', entry)
    state['R169'] = entry
    state.update(current_round='R169', current_stage='R169_REFERENCE_AS_SUBJECT',
                 manual_active_state='round169/STATE.json', run_state='R169_' + phase.upper(), updated_local=stamp)
    active = phase in ('cache_replay', 'cache_scoring', 'requery', 'panel_scoring', 'source_review')
    state['directions']['crossplatform'].update(active_round='R169' if active else None, execution_open=active)
    atomic_json(path, state)
