"""Frozen inputs and per-phase seals for the native-reference experiment."""
from datetime import datetime
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R170 = ROOT / 'research/round170'
R169 = ROOT / 'research/round169'
WORK = ROOT / 'crossplatform/round171'
OUT = HERE / 'results'
SPEC = HERE / 'experiment/PROTOCOL.json'
INPUT = WORK / 'INPUT_PRIVATE.json'
SPLITS = ('replay170', 'new_frames')
ARMS = ('r158_pca', 'C_mapped_independent', 'C_mapped_joint', 'D_mapped_independent', 'D_mapped_joint',
        'D_native_independent', 'D_native_fixed_reference', 'D_native_linked_joint', 'D_native_joint')
R170_COMMIT = '145fb621661b7095631444cec684e678f97c403f'
R169_COMMIT = '8c2c8b4a97254368bb194d5b1b98cb6e8c4690ef'

sys.path.insert(0, str(R170))
from scope170 import (INPUT as PRIOR_INPUT, ORDER, PLATFORMS, R159, REC_INPUT, ROOT as PARENT_ROOT,
                      SOURCE_SETTINGS, WORK as PRIOR_WORK, atomic_json, read, sha)

assert ROOT == PARENT_ROOT


def lock(phase, programs):
    config = read(SPEC)
    assert tuple(config['arms']) == ARMS and not config['FINAL_access']
    assert config['r169_commit'] == R169_COMMIT and config['r170_commit'] == R170_COMMIT
    assert read(R170 / 'publication/PUBLICATION_RECEIPT.json')['commit'] == R170_COMMIT
    assert read(R169 / 'publication/PUBLICATION_RECEIPT.json')['commit'] == R169_COMMIT
    old_prediction = read(R170 / 'PREDICTION_LOCK.json')['hashes']
    old_requery = read(R169 / 'PANEL_FORWARD_LOCK.json')['sha256']
    old_paths = ('research/round170/predict170.py', 'research/round170/select170.py',
                 'research/round168/calibrate168.py', 'research/round166/init166.py',
                 'research/round162/support162.py', 'research/round97/parse97.py',
                 'research/round162/experiment/SOURCE_PARAMETERS.json', 'crossplatform/single_frame_sources.json',
                 'research/round169/requery169.py')
    for key in old_paths:
        assert sha(ROOT / key) == (old_prediction | old_requery)[key]
    assert sha(PRIOR_INPUT) == read(R170 / 'INPUT_SEAL.json')['registry']
    parents = ('research/round170/INPUT_SEAL.json', 'research/round170/PREDICTION_SEAL.json',
               'research/round170/VISUAL_TRUTH_SEAL.json', 'research/round170/EVALUATION_AUDIT.json',
               'research/round170/SCALAR_AUDIT.json', 'research/round159/RECORDING_INPUT_SEAL.json',
               'research/round159/RECORDING_PREDICTION_SEAL.json', 'research/round167/retrieval167.py',
               'research/round168/select168.py', 'research/round168/replay168.py')
    paths = [SPEC, HERE / 'scope171.py', *(HERE / p for p in programs),
             *(ROOT / p for p in (*old_paths, *parents))]
    data = dict(phase=phase, files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, FINAL_access=False)
    dest = HERE / (phase.upper() + '_LOCK.json')
    if dest.exists():
        assert read(dest) == data
    else:
        atomic_json(dest, data)
    return sha(dest)


def status(phase, **details):
    path = ROOT / 'research/STATE.json'
    global_status = read(path)
    assert global_status['R161']['state'] == 'paused_by_user'
    assert global_status['R170']['public_commit'] == R170_COMMIT
    stamp = datetime.now().astimezone().isoformat()
    row = dict(round='R171', state=phase, last_updated=stamp, source_training_runs=0, target_parameter_updates=0,
               FINAL_access=False, **details)
    atomic_json(HERE / 'STATE.json', row)
    global_status['R171'] = row
    global_status.update(current_round='R171', current_stage='R171_NATIVE_REFERENCE_JOINT_ASSIGNMENT',
                         run_state='R171_' + phase.upper(), manual_active_state='round171/STATE.json', updated_local=stamp)
    active = phase in ('selecting', 'predicting', 'blind_review', 'scoring')
    global_status['directions']['crossplatform'].update(active_round='R171' if active else None, execution_open=active)
    atomic_json(path, global_status)
