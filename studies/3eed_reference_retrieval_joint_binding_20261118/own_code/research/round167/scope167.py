"""Read only the R165 ordinary population and hash every experimental dependency."""
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / 'crossplatform/round167'
W166 = ROOT / 'crossplatform/round166'
W165 = ROOT / 'crossplatform/round165'
R166 = ROOT / 'research/round166'
R165 = ROOT / 'research/round165'
R97 = ROOT / 'research/round97'
PROTOCOL = HERE / 'experiment/PROTOCOL.json'
sys.path[:0] = [str(ROOT), *(str(ROOT / f'research/round{r}') for r in (166, 165, 162, 160, 158))]
from common160 import ORDER, PLATFORMS, read, sha
from manifest162 import atomic_json

ARMS = ('r158_pca', 'independent', 'fixed_reference', 'joint', 'r88_ref3_joint')
NROLE, BANK, NOVEL = 5, 32, 10
OUT = HERE / 'results'


@lru_cache(maxsize=1)
def roster():
    contract = read(R97 / 'experiment/DESIGN_LOCK.json')['inputs_sha256']
    sizes = read(R165 / 'INPUT_LOCK.json')['ordinary_counts']
    records = {}
    for target in PLATFORMS:
        path = R97 / 'input' / f'TARGET_{target.upper()}_PRIVATE.json'
        assert contract[path.relative_to(ROOT).as_posix()] == sha(path)
        rows = read(path)
        assert len(rows) == sizes[target] and all(x['platform'] == target for x in rows)
        records[target] = rows
    assert sum(map(len, records.values())) * 2 == 18620
    return records


def parents():
    p165 = read(R165 / 'ORDINARY_CUDA_ENDPOINT_SEAL.json')
    p166 = read(R166 / 'PREDICTION_SEAL.json')
    assert not p165['FINAL_access'] and not p166['FINAL_access']
    assert p165['n'] == p166['n'] == 18620
    assert read(R166 / 'EVALUATION_AUDIT.json')['prediction_seal'] == sha(R166 / 'PREDICTION_SEAL.json')
    for source, target in ORDER:
        tag = f'{source}_to_{target}'
        a = W165 / 'ordinary' / tag / 'SEAL.json'
        b = W166 / 'ordinary' / tag / 'SEAL.json'
        assert sha(a) == p165['per_direction'][tag]['direction_forward_seal']
        assert sha(b) == p166['directions'][tag]['seal']
        assert read(b)['parent_forward_sha256'] == sha(a)
    return p165, p166


def lock():
    assert tuple(read(PROTOCOL)['arms']) == ARMS
    dependencies = (PROTOCOL, HERE / 'scope167.py', HERE / 'retrieval167.py', HERE / 'predict167.py',
                    R165 / 'INPUT_LOCK.json', R165 / 'ORDINARY_CUDA_ENDPOINT_SEAL.json',
                    R166 / 'INPUT_LOCK.json', R166 / 'PREDICTION_SEAL.json', R166 / 'predict166.py',
                    R97 / 'experiment/DESIGN_LOCK.json', ROOT / 'crossplatform/single_frame_sources.json')
    value = dict(files={p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
                 n=18620, FINAL_access=False, target_label_read=False, source_training_runs=0)
    p = HERE / 'INPUT_LOCK.json'
    if p.exists():
        assert read(p) == value
    else:
        roster()
        parents()
        atomic_json(p, value)
    return value


def status(phase, **details):
    stamp = datetime.now().astimezone().isoformat()
    state_path = ROOT / 'research/STATE.json'
    overall = read(state_path)
    assert overall['R161']['state'] == 'paused_by_user'
    entry = dict(round='R167', state=phase, updated=stamp, ordinary_direction_records=18620,
                 FINAL_access=False, source_training_runs=0, target_parameter_updates=0, **details)
    atomic_json(HERE / 'STATE.json', entry)
    overall['R167'] = entry
    overall.update(current_round='R167', manual_active_state='round167/STATE.json',
                   current_stage='R167_LANGUAGE_INSTANCE_JOINT_ASSIGNMENT',
                   run_state='R167_' + phase.upper(), updated_local=stamp)
    active = phase in ('preflight', 'running', 'scoring', 'counterfactual_review')
    overall['directions']['crossplatform'].update(active_round='R167' if active else None, execution_open=active)
    atomic_json(state_path, overall)
