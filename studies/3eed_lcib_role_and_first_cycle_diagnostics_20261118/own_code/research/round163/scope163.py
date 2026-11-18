"""Pin the R162 registries, cached evidence and native replay dependencies."""
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / 'crossplatform/round163'
PARENT = ROOT / 'research/round162'
PARENT_WORK = ROOT / 'crossplatform/round162'
PROTOCOL = HERE / 'experiment/PROTOCOL.json'
SETTINGS = PARENT / 'experiment/SOURCE_PARAMETERS.json'
sys.path.insert(0,str(PARENT))
from manifest162 import PLATFORMS, atomic_json, read, sha, sensor_row, target_rows


def source_scope():
    plan = read(PARENT/'CALIBRATION_INPUTS.json')
    path = PARENT_WORK/'SOURCE_VAL_PRIVATE.json'
    assert sha(path)==plan['input_sha256']
    rows=read(path)
    assert len(rows)==192 and all(sum(r['source']==s and r['platform']==s for r in rows)==64 for s in PLATFORMS)
    return rows


def pinned_parent():
    parent=read(PARENT/'STATE.json')
    assert parent['public_commit']==read(PROTOCOL)['parent']
    for name,digest in read(PARENT/'experiment/TARGET_LOCK.json')['pinned'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError(f'Parent changed: {name}')
    original=read(PARENT/'CALIBRATION_SEAL.json')
    registry=PARENT_WORK/'CALIBRATION_FILES_PRIVATE.json'
    assert original['n']==192 and sha(registry)==original['file_index_sha256']
    assert sha(PARENT/'CALIBRATION_SEAL.json')==read(SETTINGS)['calibration_prediction_seal_sha256']
    return read(registry)


def begin():
    local=HERE/'STATE.json'
    if local.exists():
        raise FileExistsError(local)
    stamp=datetime.now().astimezone().isoformat()
    facts=dict(round='R163',state='running',role_edges=33,source_rows=192,
               protocol='round163/experiment/PROTOCOL.json',parent=read(PROTOCOL)['parent'],
               baseline_queue='paused',FINAL_access=False,updated=stamp)
    atomic_json(local,facts)
    overall=read(ROOT/'research/STATE.json')
    assert overall['R161']['state']=='paused_by_user'
    overall['R163']=facts
    overall.update(current_round='R163',current_stage='R163_ROLE_AND_GEOMETRY_DIAGNOSTICS',
                   manual_active_state='round163/STATE.json',run_state='R163_RUNNING',updated_local=stamp)
    overall['directions']['crossplatform'].update(active_round='R163',execution_open=True)
    atomic_json(ROOT/'research/STATE.json',overall)


if __name__=='__main__':
    pinned_parent()
    source_scope()
    begin()
