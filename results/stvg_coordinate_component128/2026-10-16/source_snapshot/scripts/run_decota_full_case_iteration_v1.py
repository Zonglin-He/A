"""TA-STVG-only finite full-evaluation queue followed by CPU case analysis.

Never calls the legacy causal/baseline batch. No recurring automation or new
method selection. Existing valid per-query receipts are resumed in place.
"""
import argparse,fcntl,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.run_stvg_fullscale_v1 import plan as source_plan,OUT as SOURCE
from scripts.run_decota_local_iteration_v1 import protect

OUT=ROOT/'artifacts/decota_full_case_iteration_v1'
COHORTS=['hcstvg1_test','vidstg_test']
PY=ROOT/'.conda/tubedetr/bin/python'
FILES=['scripts/run_decota_full_case_iteration_v1.py','scripts/analyze_decota_full_cases_v1.py',
       'vg_tta/decota_full_case_diagnostics_v1.py','tests/test_decota_full_case_diagnostics_v1.py',
       'protocols/decota_full_case_iteration_v1.md']


def verify():
    p=read(OUT/'lock.json');source_plan()
    for f,h in {**p['code'],**p['protected']}.items():assert sha(ROOT/f)==h,f
    assert sha(SOURCE/'lock.json')==p['source_lock_sha256']
    return p


def prepare():
    p=source_plan()
    if (OUT/'lock.json').exists():return verify()
    write(OUT/'lock.json',dict(version='decota_full_case_iteration_v1',created=time.time(),backbones=['tastvg'],
        cohorts=COHORTS,source_lock_sha256=sha(SOURCE/'lock.json'),counts={c:p['counts'][c] for c in COHORTS},
        code={f:sha(ROOT/f) for f in FILES},protected=protect(),nominal_queries=sum(len(p['rows'][c]) for c in COHORTS),
        max_queue_seconds=24*3600,min_free_bytes=40*2**30,method_stage_only=True,
        resume_matching_receipts=True,GT_used_for_predictions=False,corruption=False,
        automatic_analysis=True,automatic_algorithm_change=False,old_causal_queue_resumed=False,
        user_scope_revision='TA-STVG only; TubeDETR explicitly excluded'))
    return verify()


def snapshot(cohort):
    d=SOURCE/'method/tastvg'/cohort
    p=read(d/'progress.json') if (d/'progress.json').exists() else {'done':0}
    if (d/'barrier.json').exists():p.update(done=read(d/'barrier.json')['queries'],barrier=True)
    return p


def child_run(cmd,cohort,stage,deadline):
    logpath=OUT/'logs'/f'{cohort}_{stage}.log';logpath.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2'}
    with open(logpath,'ab',buffering=0) as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        try:
            while child.poll() is None:
                state=dict(status='running',stage=stage,cohort=cohort,backbone='tastvg',child_pid=child.pid,
                    command=cmd,updated=time.time(),prediction_progress=snapshot(cohort),log=str(logpath))
                status(OUT/'progress.json',state)
                if time.time()>deadline:raise TimeoutError('Finite queue budget reached; receipts preserved')
                if shutil.disk_usage(ROOT).free<40*2**30:raise RuntimeError('Free disk below safety guard; receipts preserved')
                time.sleep(15)
            if child.returncode:raise RuntimeError(f'Child exit {child.returncode}: {logpath}')
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()


def batch():
    p=prepare();lease=open(OUT/'queue.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.time();deadline=began+p['max_queue_seconds']
    try:
        for cohort in COHORTS:
            verify();d=SOURCE/'method/tastvg'/cohort
            if not (d/'barrier.json').exists():
                child_run(['bash',str(ROOT/'scripts/with_local_cuda.sh'),str(PY),'-B','-u',str(ROOT/'scripts/run_stvg_fullscale_v1.py'),
                    'method','--backbone','tastvg','--cohort',cohort],cohort,'method',deadline)
            bar=read(d/'barrier.json');assert bar['all_available_complete'] and bar['model_state_unchanged']
            # Model jobs remain fixed even after a prior cell's labels are read.
            child_run([str(PY),'-B','-u',str(ROOT/'scripts/analyze_decota_full_cases_v1.py'),'--cohort',cohort],cohort,'analysis',deadline)
            verify();assert (OUT/'analysis'/cohort/'complete.json').exists()
        status(OUT/'progress.json',dict(status='completed',stage='full_reference_and_case_artifacts',
            started=began,updated=time.time(),backbone='tastvg',human_visual_review_pending=True,
            single_factor_experiments_started=False,production_changed=False))
        write(OUT/'queue_complete.json',dict(started=began,finished=time.time(),backbone='tastvg',
            full_reference_done=True,case_artifacts_generated=True,human_visual_review_pending=True,
            hypotheses_not_yet_causally_verified=True,single_factor_experiments_started=False))
    except BaseException as exc:
        status(OUT/'progress.json',dict(status='failed',updated=time.time(),error=str(exc),traceback=traceback.format_exc(),
            receipts_preserved=True,production_changed=False))
        raise


def launch():
    prepare();f=OUT/'process.json'
    if f.exists():
        old=read(f);proc=Path('/proc')/str(old['pid'])/'cmdline'
        if proc.exists() and str(Path(__file__).resolve()).encode() in proc.read_bytes() and b'batch' in proc.read_bytes():
            print('ALREADY_RUNNING',old['pid'],flush=True);return
    with open(OUT/'queue.log','ab',buffering=0) as log:
        command=[str(PY),'-B','-u',str(Path(__file__).resolve()),'batch']
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    status(f,dict(pid=child.pid,started=time.time(),command=command,log=str(OUT/'queue.log'),finite_not_recurring=True))
    print('STARTED_TASTVG_ONLY_QUEUE',child.pid,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','batch','launch','verify']);a=ap.parse_args();globals()[a.stage]()
