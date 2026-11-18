"""Lock R169 and register a separate ordinary-recording confirmation panel."""
from datetime import datetime
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
R169=ROOT/'research/round169'
R159=ROOT/'research/round159'
WORK=ROOT/'crossplatform/round170'
OUT=HERE/'results'
SPEC=HERE/'experiment/PROTOCOL.json'
sys.path.insert(0,str(R169))
from scope169 import CAL, ORDER, PLATFORMS, R168, R167, atomic_json, read, roster, sha

ARMS=('r158_pca','C_independent','C_joint','D_independent','D_joint')
VARIANTS=('C','D')
FROZEN='8c2c8b4a97254368bb194d5b1b98cb6e8c4690ef'
INPUT=WORK/'INPUT_PRIVATE.json'
SOURCE_SETTINGS=ROOT/'research/round162/experiment/SOURCE_PARAMETERS.json'
REC_INPUT=ROOT/'crossplatform/round159/RECORDING_INPUTS_PRIVATE.json'
TRAIN_INDEX=ROOT/'research/round69/input/SCORING_INDEX_PRIVATE.json'
WAYMO_INDEX=ROOT/'research/round76/input/SOURCE_TRAIN512_PRIVATE.json'


def lock(phase,programs):
    rule=read(SPEC)
    assert tuple(rule['arms'])==ARMS and rule['method_commit']==FROZEN and rule['FINAL_access'] is False
    assert read(R169/'publication/PUBLICATION_RECEIPT.json')['commit']==FROZEN
    pins=read(R169/'PANEL_FORWARD_LOCK.json')['sha256']
    for name in ('cache169.py','panel169.py','requery169.py'):
        assert sha(R169/name)==pins['research/round169/'+name]
    from scope167 import lock as r167_lock
    r167_lock()
    old=read(ROOT/'research/round166/INPUT_LOCK.json')['files']
    assert sha(SOURCE_SETTINGS)==old[str(SOURCE_SETTINGS.relative_to(ROOT))]
    files=(SPEC,HERE/'scope170.py',R169/'PANEL_FORWARD_LOCK.json',R169/'QUERY_IDENTITY_AUDIT.json',
           R169/'QUERY_JOINT_AUDIT.json',R169/'publication/PUBLICATION_RECEIPT.json',R159/'RECORDING_INPUT_SEAL.json',
           R159/'RECORDING_PREDICTION_SEAL.json',SOURCE_SETTINGS,TRAIN_INDEX,WAYMO_INDEX,
           ROOT/'crossplatform/single_frame_sources.json',ROOT/'research/round97/parse97.py',
           ROOT/'research/round166/init166.py',ROOT/'research/round162/support162.py',
           ROOT/'research/round168/calibrate168.py',*(HERE/x for x in programs))
    entry=dict(phase=phase,hashes={str(p.relative_to(ROOT)):sha(p) for p in files},FINAL_access=False)
    target=HERE/(phase.upper()+'_LOCK.json')
    if target.exists():
        assert read(target)==entry
    else:
        atomic_json(target,entry)
    return sha(target)


def status(phase, **details):
    path=ROOT/'research/STATE.json'
    state=read(path)
    assert state['R161']['state']=='paused_by_user'
    assert state['R169']['public_commit']==FROZEN
    now=datetime.now().astimezone().isoformat()
    result=dict(round='R170',state=phase,last_updated=now,FINAL_access=False,source_training_runs=0,
                target_parameter_updates=0,**details)
    atomic_json(HERE/'STATE.json',result)
    state['R170']=result
    state.update(current_round='R170',current_stage='R170_LOCKED_REQUERY_RECORDING_CONFIRMATION',
                 run_state='R170_'+phase.upper(),manual_active_state='round170/STATE.json',updated_local=now)
    active=phase in ('selecting','predicting','blind_review','scoring')
    state['directions']['crossplatform'].update(active_round='R170' if active else None,execution_open=active)
    atomic_json(path,state)
