"""Finite root CPU closure, with a separate pre-GT scoring-code lock.

This entry point never waits for a GPU job, launches inference, or publishes.
The root invokes `run` only after the existing controller completes its global
prediction barrier. `prepare` hashes code and metadata, without opening GT.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,json,time,hashlib,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/tastvg_negative_evidence_v1'
PUB=ROOT/'results/tastvg_negative_evidence/2026-10-04'
LOCK=BASE/'SCORING_RUNTIME_LOCK.json'
CODE=(
    'scripts/finalize_tastvg_negative_evidence_cpu_v1.py',
    'scripts/score_tastvg_negative_evidence_v1.py',
    'scripts/audit_tastvg_negative_evidence_v1.py',
    'scripts/report_tastvg_negative_evidence_v1.py',
    'scripts/audit_tastvg_dta_oracle_r1_v1.py',
    'vg_tta/tastvg_oracle_event5_v1.py',
    'vg_tta/tastvg_paper48_metrics_v1.py',
    'scripts/tastvg_negative_evidence_common_v1.py',
    'scripts/decota_matrix_common_v1.py',
    'tests/test_tastvg_negative_evidence_v1.py',
    'vg_tta/tastvg_negative_evidence_v1.py',
)

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def prepare():
    assert not (BASE/'GT_EXPOSURE.json').exists()
    write(LOCK,dict(status='frozen_before_new_GT',time=time.time(),GT_read=False,
        prediction_runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        cohort_sha256=sha(BASE/'COHORT.json'),
        code={p:sha(ROOT/p) for p in CODE},
        scope='CPU scoring, independent arithmetic audit and reporting; no change to prediction runtime or scientific design.'))
    print('SCORING_CODE_FROZEN_NO_GT',flush=True)

def verify():
    lock=read(LOCK);assert lock['GT_read'] is False
    pins=dict(lock['code'])
    for p in sorted((BASE/'scoring_revisions').glob('*.json')):
        revision=read(p);assert revision['scientific_change'] is False
        pins.update(revision['pin_overrides'])
    for p,h in pins.items():assert sha(ROOT/p)==h,p
    assert sha(BASE/'RUNTIME_LOCK.json')==lock['prediction_runtime_lock_sha256']
    assert sha(BASE/'COHORT.json')==lock['cohort_sha256']
    return lock

def run():
    verify()
    assert read(BASE/'STATUS.json')['status']=='completed_pending_root_scoring_audit_publication'
    from scripts.tastvg_negative_evidence_common_v1 import verify_seal
    barrier=verify_seal()
    assert barrier['local_experts']==288 and barrier['reset_u_arrivals']==1152
    assert not (BASE/'WORKER_FAILURE.json').exists()
    assert not (BASE/'GT_EXPOSURE.json').exists()
    stage='score'
    try:
        from scripts.score_tastvg_negative_evidence_v1 import score
        score();verify()
        stage='independent_root_audit'
        from scripts.audit_tastvg_negative_evidence_v1 import root,public,write as audit_write
        root();verify()
        stage='public_arithmetic_audit'
        audit_write(PUB/'PUBLIC_AUDIT.json',public(PUB));verify()
        stage='report_figures'
        from scripts.report_tastvg_negative_evidence_v1 import report
        report();verify()
        stage='public_report_audit'
        audit_write(PUB/'PUBLIC_REPORT_AUDIT.json',public(PUB));verify()
        write(BASE/'CPU_COMPLETION.json',dict(status='scored_audited_reported_pending_root_visual_publication',
            time=time.time(),GT_read=True,scoring_runtime_lock_sha256=sha(LOCK),
            global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
            public_files={str(p.relative_to(PUB)):sha(p) for p in sorted(PUB.rglob('*')) if p.is_file()},
            production_promoted=False))
        print('FINITE_CPU_ROOT_PIPELINE_COMPLETE_PENDING_VISUAL_AND_GITHUB',flush=True)
    except BaseException as e:
        write(BASE/('CPU_FAILURE_'+str(time.time_ns())+'.json'),dict(stage=stage,time=time.time(),
            error=repr(e),traceback=traceback.format_exc(),
            GT_read=(BASE/'GT_EXPOSURE.json').exists(),
            scoring_runtime_lock_sha256=sha(LOCK),scientific_change=False))
        raise

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','run')
    globals()[sys.argv[1]]()
