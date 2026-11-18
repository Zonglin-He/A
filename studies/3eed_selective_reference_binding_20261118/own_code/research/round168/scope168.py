"""Freeze the ordinary R167 replay and isolated source-only calibration."""
from datetime import datetime
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / 'crossplatform/round168'
R167 = ROOT / 'research/round167'
R166 = ROOT / 'research/round166'
R165 = ROOT / 'research/round165'
sys.path[:0] = [str(R167), str(ROOT / 'research/round88')]
from scope167 import ORDER, PLATFORMS, W165, W166, roster, sha, read, atomic_json, lock as parent_lock

PROTOCOL = HERE / 'experiment/PROTOCOL.json'
OUT = HERE / 'results'
ARMS = ('independent', 'r167_joint', 'left_only', 'overlap_only', 'old_ref_abstain', 'new_ref_abstain',
        'scrambled_ref_abstain', 'new_ref_independent')
CAL = WORK / 'source_calibration_PRIVATE'
TOK = ROOT / 'crossplatform/upstream/data/roberta_base'
SENTINEL = ' . not mentioned'


def lock():
    parent_lock()
    assert tuple(read(PROTOCOL)['arms']) == ARMS
    pinned = (PROTOCOL, HERE / 'scope168.py', HERE / 'select168.py', HERE / 'calibrate168.py', HERE / 'replay168.py',
              R167 / 'PREDICTION_SEAL.json', R167 / 'CONTROLLED_PREDICTION_SEAL.json',
              R167 / 'EVALUATION_AUDIT.json', ROOT / 'research/round88/routing88.py',
              TOK / 'tokenizer.json', TOK / 'tokenizer_config.json')
    receipt = dict(files={x.relative_to(ROOT).as_posix(): sha(x) for x in pinned}, n=18620,
                   target_label_read=False, FINAL_access=False, source_model_training_steps=0)
    path = HERE / 'INPUT_LOCK.json'
    if path.exists():
        assert read(path) == receipt
    else:
        roster()
        atomic_json(path, receipt)
    return receipt


def status(phase, **kwargs):
    ds = ROOT / 'research/STATE.json'
    s = read(ds)
    assert s['R161']['state'] == 'paused_by_user'
    t = datetime.now().astimezone().isoformat()
    r = dict(round='R168', state=phase, last_updated=t, ordinary_direction_records=18620,
             FINAL_access=False, source_training_runs=0, target_parameter_updates=0, **kwargs)
    atomic_json(HERE / 'STATE.json', r)
    s['R168'] = r
    s.update(current_round='R168', manual_active_state='round168/STATE.json',
             current_stage='R168_SELECTIVE_REFERENCE_BINDING', run_state='R168_' + phase.upper(), updated_local=t)
    active = phase in ('source_calibration', 'replay', 'scoring')
    s['directions']['crossplatform'].update(active_round='R168' if active else None, execution_open=active)
    atomic_json(ds, s)
