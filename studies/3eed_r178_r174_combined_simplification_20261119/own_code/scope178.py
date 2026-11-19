"""R178 registration, lineage and ordinary-only inputs."""
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
R177 = HERE.parent/'3eed_r177_r174_fine_ablation_20261119'
PRIVATE = HERE/'private'
RESULTS = HERE/'results'
SPEC = HERE/'protocol.json'
sys.path.insert(0, str(R177))

from scope177 import (N, ORDER, ORIGINAL, PLATFORMS, PRIOR, R165_DATA, ROOT,
                      PRIVATE as P177, input_lock as lock177, json_atomic,
                      npz_atomic, ordinary_rows, read, sha)

PROTOCOL = read(SPEC)
ARMS = tuple(PROTOCOL['arms'])


def state(phase, note, receipt=None, decision=None):
    workspace = ROOT/'research/STATE.json'
    registry = read(workspace)
    route = registry['directions']['crossplatform']
    assert route['active_round'] in (None,'R178')
    assert route['main_method']['round'] == 'R174'
    stamp = datetime.now().astimezone().isoformat()
    complete = phase.startswith('completed')
    route.update(active_round=None if complete else 'R178',execution_open=not complete,
                 active_report=(HERE/'REPORT_ZH.md').relative_to(ROOT).as_posix(),
                 active_FINAL_access=False)
    record = dict(round='R178',state=phase,n=N,main_method='R174',note=note,
                  updated_local=stamp,protocol=SPEC.relative_to(ROOT).as_posix(),
                  source_retraining=0,target_updates=0,FINAL_access=False)
    if decision:
        record['profiles'] = decision['profiles']
        route['main_method']['combined_inference_profiles'] = {
            'study': 'R178',
            'protocol': SPEC.relative_to(ROOT).as_posix(),
            'implementation': (HERE/'model178.py').relative_to(ROOT).as_posix(),
            'decision': (RESULTS/'DECISION.json').relative_to(ROOT).as_posix(),
            **decision['profiles']
        }
    if receipt:
        record.update(commit=receipt['commit'], publication_receipt=(
            HERE/'publication/PUBLICATION_RECEIPT.json').relative_to(ROOT).as_posix())
        route.update(latest_completed_report=(HERE/'REPORT_ZH.md').relative_to(ROOT).as_posix(),
                     latest_completed_publication=receipt['commit'],latest_public_commit=receipt['commit'],
                     latest_public_report=(
                         'https://github.com/'+receipt['repository']+'/blob/'+receipt['commit']+
                         '/'+receipt['prefix']+'/REPORT_ZH.md'))
    registry['R178'] = record
    json_atomic(HERE/'STATE.json',record)
    json_atomic(workspace,registry)


def lock():
    earlier = P177/'predictions/PREDICTION_SEAL.json'
    original = ORIGINAL/'PREDICTION_SEAL.json'
    receipt = R177/'publication/PUBLICATION_RECEIPT.json'
    stamp, released = read(earlier),read(receipt)
    assert stamp['n']==N and stamp['lock']==lock177() and not stamp['FINAL_access']
    assert released['commit']==PROTOCOL['parent_commit'] and not released['FINAL_access']
    inputs = [SPEC,HERE/'scope178.py',HERE/'model178.py',HERE/'predict178.py',
              earlier,original,P177/'EXECUTION_LOCK.json',receipt]
    for s,t in ORDER:
        direction = f'{s}_to_{t}'
        previous = earlier.parent/direction/'SEAL.json'
        saved = ORIGINAL/direction/'SEAL.json'
        archive = R165_DATA/direction/'SEAL.json'
        assert sha(previous)==stamp['directions'][direction]['seal']
        assert read(previous)['original_seal']==sha(saved)
        assert read(saved)['source_seal']==sha(archive) and not read(archive)['protected_payload_access']
        inputs.extend((previous,saved,archive))
    contract = dict(n=N,arms=list(ARMS),inputs={x.relative_to(ROOT).as_posix():sha(x) for x in inputs},
                    source_models=read(P177/'EXECUTION_LOCK.json')['source_models'],
                    FINAL_access=False)
    path = PRIVATE/'EXECUTION_LOCK.json'
    if path.exists():
        assert read(path)==contract
    else:
        json_atomic(path,contract)
    return sha(path)
