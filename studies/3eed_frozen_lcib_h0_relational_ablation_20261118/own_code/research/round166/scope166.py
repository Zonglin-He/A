"""Lock the R165 paired inputs and the pre-existing LCIB initialization."""
from datetime import datetime
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORK=ROOT/'crossplatform/round166'
R165=ROOT/'research/round165'
R162=ROOT/'research/round162'
PROTOCOL=HERE/'experiment/PROTOCOL.json'
SETTINGS=R162/'experiment/SOURCE_PARAMETERS.json'
sys.path.insert(0,str(R165))
from scope165 import WORK as W165, ORDER, PLATFORMS, W97, atomic_json, check_lock as parent_lock, ordinary, read, sha
from common160 import npz_atomic

ARMS=('r158_pca','h0_full','h0_independent','h0_no_ref_role','h0_original_only')
MAX_ROLES, MAX_HYPOTHESES, BLOCK=5,6,128


def inputs():
    parent_lock()
    roster=ordinary()
    parent=read(R165/'ORDINARY_CUDA_ENDPOINT_SEAL.json')
    assert parent['n']==18620 and not parent['FINAL_access']
    assert read(R165/'SCALAR_AUDIT.json')['all_rows']==18620
    assert read(R165/'STATE.json')['public_commit']=='0c68f2c94c0d00cf91305f16df4e0884c97300e0'
    assert tuple(read(PROTOCOL)['arms'])==ARMS
    for source,target in ORDER:
        tag=f'{source}_to_{target}'
        p=W165/'ordinary'/tag
        receipt=parent['per_direction'][tag]
        assert sha(p/'SEAL.json')==receipt['direction_forward_seal']
        assert len(roster[target])==receipt['n']
    return roster,parent


def lock():
    files=[PROTOCOL,SETTINGS,HERE/'scope166.py',HERE/'init166.py',HERE/'predict166.py',
           R162/'evidence162.py',R162/'geometry162.py',R162/'support162.py',R162/'solver162.py',
           R165/'INPUT_LOCK.json',R165/'ORDINARY_CUDA_ENDPOINT_SEAL.json',R165/'ORDINARY_FORWARD_SEAL.json',
           R165/'SCALAR_AUDIT.json',W97/'MATRIX_SEALED.json']
    values={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    p=HERE/'INPUT_LOCK.json'
    if p.exists():
        assert values==read(p)['files']
    else:
        inputs()
        atomic_json(p,dict(files=values,records=18620,directions=6,FINAL_access=False,
                           source_training_runs=0, target_parameter_updates=0))


def status(step, **details):
    ts=datetime.now().astimezone().isoformat()
    overall=read(ROOT/'research/STATE.json')
    assert overall['R161']['state']=='paused_by_user'
    entry=dict(round='R166',state=step,ordinary_direction_records=18620,updated=ts,FINAL_access=False,**details)
    atomic_json(HERE/'STATE.json',entry)
    overall['R166']=entry
    running=step=='running'
    overall.update(current_round='R166',manual_active_state='round166/STATE.json',
                   current_stage='R166_FROZEN_H0_ABLATIONS',run_state='R166_'+step.upper(),updated_local=ts)
    overall['directions']['crossplatform'].update(active_round='R166' if running else None,execution_open=running)
    atomic_json(ROOT/'research/STATE.json',overall)
