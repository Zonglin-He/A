"""Pin current code closure and immutable CRC inputs before any model run."""
from pathlib import Path
import sys,time,importlib.metadata
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_crc_common_v1 import *
from scripts.lock_stvg_ecld_runtime002 import closure

def main():
    assert not (BASE/'RUNTIME.json').exists()
    seeds=[p for p in (ROOT/'scripts').glob('*crc*.py') if not p.name.startswith(('report_', 'publish_', 'audit_'))]
    sources=closure(sorted(seeds))
    pins={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}
    for name in ['DESIGN_LOCK.json','ROSTER_LOCK.json','inputs_teacher/CPU_CONTRACTS.json','inputs_teacher/INPUT_BARRIER.json','inputs_teacher/TEACHER_JOB_LOCK.json','inputs_teacher/PARENT_TOP1_CUSTODY.json']:
        p=BASE/name; pins[str(p.relative_to(ROOT))]=sha(p)
    for name in ['TEACHER_JOB_LOCK.json','TEACHER_GLOBAL_BARRIER.json','CANDIDATE_INPUT_BARRIER.json','REFERENCE_LOCK.json','STUDENT_GLOBAL_BARRIER.json']:
        p=PARENT/name;pins[str(p.relative_to(ROOT))]=sha(p)
    for p in [OLD/'DEVELOPMENT_ROSTER_LOCK.json',ROOT/'protocols/stvg_contrastive_correction_v1.md',ROOT/'artifacts/stvg_final_table1_v1/PREFLIGHT_ROSTER_LOCK.json',PARENT/'GLOBAL_PREDICTION_BARRIER.json']:
        pins[str(p.relative_to(ROOT))]=sha(p)
    protected={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'methods/CURRENT_METHOD.json',OLD/'FROZEN_METHOD_LOCK.json']}
    from scripts.run_stvg_crc_teacher_v1 import EXPECTED_GENERATION
    receipt=ROOT/'artifacts/stvg_target_teacher_distillation_v1/ROOT_OFFICIAL_MODEL_FILE_READBACK.json'
    teacher=dict(schema='stvg-crc-teacher-runtime-v1',status='locked',GT_read=False,design_lock_sha256=sha(BASE/'DESIGN_LOCK.json'),roster_sha256=sha(BASE/'ROSTER_LOCK.json'),input_barrier_sha256=sha(BASE/'inputs_teacher/INPUT_BARRIER.json'),job_lock_sha256=sha(BASE/'inputs_teacher/TEACHER_JOB_LOCK.json'),parent_input_barrier_sha256=sha(PARENT/'CANDIDATE_INPUT_BARRIER.json'),parent_job_lock_sha256=sha(PARENT/'TEACHER_JOB_LOCK.json'),parent_teacher_barrier_sha256=sha(PARENT/'TEACHER_GLOBAL_BARRIER.json'),parent_top1_custody_sha256=sha(BASE/'inputs_teacher/PARENT_TOP1_CUSTODY.json'),model_name='XiaowenZhang/stvg-r1-model-7b',checkpoint_path='checkpoints/STVG-R1-7B',model_readback_path=str(receipt.relative_to(ROOT)),model_readback_sha256=sha(receipt),model_revision=read(receipt)['revision'],generation=EXPECTED_GENERATION,pins=dict(pins),protected=protected,time=time.time())
    write(BASE/'TEACHER_RUNTIME.json',teacher)
    pins[str((BASE/'TEACHER_RUNTIME.json').relative_to(ROOT))]=sha(BASE/'TEACHER_RUNTIME.json')
    versions={}
    for p in ['torch','numpy','transformers','accelerate','safetensors']:
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]=None
    write(BASE/'RUNTIME.json',dict(status='locked',GT_read=False,pins=pins,protected=protected,local_static_import_closure_count=len(sources),environment_versions=versions,bootstrap_executed=False,full_started=False,time=time.time()))
    print('CRC runtime pinned',len(sources),len(pins))
if __name__=='__main__':main()
