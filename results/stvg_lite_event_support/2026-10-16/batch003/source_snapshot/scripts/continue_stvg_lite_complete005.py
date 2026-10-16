"""Finite final recipe -> reclocked full loop -> fixed disjoint confirmation.

This dispatcher never selects a recipe from confirmation or final-loop scores.
All GPU subprocesses are serial in one explicitly owned bounded cgroup.
"""
from pathlib import Path
import copy
import os
import subprocess
import sys
import threading
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,SEARCH,COMP,read,write,sha,verify,load_case
from scripts import launch_stvg_instance_identity_v1 as resource
UNIT='stvg-lite-complete-005.service'
EXEC=BASE/'execution_complete005'
TEMPLATE=BASE/'COMPLETE_DISPATCH_RUNTIME_TEMPLATE005.json'


def verify_template():
    from scripts.stvg_lite_recovery005_common import verify_recovery005
    verify_recovery005()
    verify();r=read(TEMPLATE);assert r['status']=='locked' and not r['GT_read']
    for rel,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/rel)==h,rel
    recipe=read(BASE/'FINAL_RECIPE_LOCK.json');assert recipe['status']=='locked' and not recipe['suite_final']
    assert recipe['selected_spatial_recipe']in ('full','lite_backtrack','lite_fullstep') and recipe['selected_response_scope']in ('full_grid','event_support')
    assert recipe['selected_temporal_lock_sha256']==sha(BASE/'SELECTED_TEMPORAL_LOCK.json')
    return r,recipe


def command(stage,script,*args,gpu=False,teacher=False):
    if gpu:
        v=resource.sample();g=1024**3
        assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    resource.STAGE=stage
    write(BASE/'COMPLETE_STATUS005.json',dict(status='running',stage=stage,GT_read=False,time=time.time()),mutable=True)
    out=EXEC/stage;out.mkdir(parents=True,exist_ok=False)
    interpreter=ROOT/('.venv/stvg_native_distill_v1/bin/python'if teacher else'.conda/tubedetr/bin/python')
    with(out/'worker.log').open('xb')as log:
        p=subprocess.Popen([str(interpreter),'-B','scripts/'+script,*args],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        write(out/'LAUNCH.json',dict(status='running',stage=stage,worker_pid=p.pid,controller_pid=os.getpid(),GT_read=False,time=time.time()))
        code=p.wait()
    write(out/'WORKER_EXIT.json',dict(status='exited',exit_code=code,stage=stage,GT_read=False,time=time.time()))
    if code:raise RuntimeError(stage+' failed: '+str(code))


def source_wrapper():
    verify_template()
    from scripts.stvg_native_instance_common_v1 import guard
    from scripts import run_stvg_lite_confirmation_source_v1 as source
    source.verify()
    original=source._save_case
    def save_with_explicit_seal(case,adapter):
        item=original(case,adapter);receipt=ROOT/item['receipt_path']
        d=read(receipt);assert d['schema']=='stvg-lite-confirmation-source-receipt-v1' and not d['GT_read']
        if d.get('status')!='sealed':
            d['status']='sealed';write(receipt,d,mutable=True)
        return item
    source._save_case=save_with_explicit_seal
    try:
        sys.addaudithook(guard);source.run(all_confirmation=True)
    finally:source._save_case=original


def confirm_temporal_runtime():
    bar=read(BASE/'confirmation/SOURCE_CAPTURE_BARRIER.json');assert bar['status']=='complete' and bar['count']==64 and not bar['GT_read']
    rt=read(TEMPLATE);pins=dict(rt['pins'])
    for p in [BASE/'FINAL_RECIPE_LOCK.json',BASE/'confirmation/SOURCE_CAPTURE_BARRIER.json',BASE/'confirmation/ROSTER_LOCK.json',BASE/'confirmation/SOURCE_RUNTIME.json']:
        pins[str(p.relative_to(ROOT))]=sha(p)
    write(BASE/'confirmation/TEMPORAL_RUNTIME.json',dict(status='locked',count=64,recipe_sha256=sha(BASE/'FINAL_RECIPE_LOCK.json'),source_barrier_sha256=sha(BASE/'confirmation/SOURCE_CAPTURE_BARRIER.json'),pins=pins,protected=rt['protected'],GT_read=False,selection_using_confirmation_GT=False))


def cached_native_refs():
    """Re-use only the already exact same-source Native-clock teacher evidence."""
    import numpy as np
    from scripts import stvg_component_native007_common as old
    report=old.input_readback();assert report['status']=='pass' and report['count']==128
    old.verify();readback=read(old.BASE/'ROOT_REFERENCE_READBACK007.json');assert readback['status']=='pass' and readback['count']==128
    teachers=read(old.BASE/'TEACHER_GLOBAL_BARRIER.json');jobs=read(old.BASE/'TEACHER_JOB_LOCK.json')
    assert teachers['status']=='complete' and teachers['count']==jobs['count'] and not teachers['GT_read']
    for f in teachers['files']:assert sha(ROOT/f['path'])==f['sha256']
    refs=read(old.BASE/'REFERENCE_LOCK.json');times=read(old.BASE/'TEMPORAL_LOCK.json')
    cases={r['probe_alias']:load_case(r)[0]for r in read(SEARCH/'ROSTER_LOCK.json')['records']}
    assert refs['status']==times['status']=='locked' and refs['count']==times['count']==128 and not refs['GT_read'] and not times['GT_read']
    for row in refs['records']:assert row['source_case_sha256']==cases[row['case_id']]['source_case_sha256']
    for row in times['records']:assert row['selected_interval']==cases[row['case_id']]['Native_interval']
    out=BASE/'native_inputs';out.mkdir(exist_ok=True)
    write(out/'REFERENCE_LOCK.json',refs);write(out/'TEMPORAL_LOCK.json',times)
    write(out/'ROOT_REFERENCE_READBACK.json',dict(status='pass',count=128,GT_read=False,CPU_only=True,scope='same Source Native clock and unchanged candidate/teacher protocol, full original input and response SHA exact reuse',reused_teacher_jobs=teachers['count'],new_teacher_jobs=0,original_job_lock_sha256=sha(old.BASE/'TEACHER_JOB_LOCK.json'),original_teacher_barrier_sha256=sha(old.BASE/'TEACHER_GLOBAL_BARRIER.json'),original_reference_sha256=sha(old.BASE/'REFERENCE_LOCK.json')))


def prepare_trials(cohort):
    rt,recipe=verify_template();scope=BASE if cohort=='development'else BASE/'confirmation';count=128 if cohort=='development'else 64
    roster=SEARCH/'ROSTER_LOCK.json'if cohort=='development'else scope/'ROSTER_LOCK.json'
    source=SEARCH/'SOURCE_CAPTURE_BARRIER.json'if cohort=='development'else scope/'SOURCE_CAPTURE_BARRIER.json'
    inputs={'full':BASE/'reclocked_inputs'if cohort=='development'else scope/'temporal_inputs','native':scope/'native_inputs'}
    trial_ids={'full':'C_FULL','native':'C_NATIVE'}
    for kind,b in inputs.items():
        ref=read(b/'REFERENCE_LOCK.json');times=read(b/'TEMPORAL_LOCK.json')
        assert ref['status']==times['status']=='locked' and ref['count']==times['count']==count and not ref['GT_read'] and not times['GT_read']
        tid=trial_ids[kind];trial=scope/'trials'/tid;trial.mkdir(parents=True,exist_ok=False)
        for name in ('REFERENCE_LOCK.json','TEMPORAL_LOCK.json'):(trial/name).write_bytes((b/name).read_bytes())
        configs={ds:{m:copy.deepcopy(recipe['configs'][ds])for m in ('M0','M2')}for ds in ('hc2','vidstg')}
        settings={ds:{m:copy.deepcopy(recipe['response_settings'][ds])for m in ('M1','M2')}for ds in configs}
        design=dict(status='locked',trial_id=tid,scope='fixed final recipe, same-information and same-parameters original OPD control; no selection from final-loop outputs',cohort=cohort,reference_clock=kind,recipe=recipe['selected_spatial_recipe'],response_scope=recipe['selected_response_scope'],configs=configs,response_settings=settings,family_reuse_from={ds:{m:None for m in ('M0','M2')}for ds in configs},GT_read=False,recipe_sha256=sha(BASE/'FINAL_RECIPE_LOCK.json'))
        write(trial/'DESIGN_LOCK.json',design);pins=dict(rt['pins'])
        for p in [TEMPLATE,BASE/'FINAL_RECIPE_LOCK.json',roster,source,b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json',trial/'REFERENCE_LOCK.json',trial/'TEMPORAL_LOCK.json',trial/'DESIGN_LOCK.json']:
            pins[str(p.relative_to(ROOT))]=sha(p)
        for p in b.glob('ROOT*READBACK*.json'):pins[str(p.relative_to(ROOT))]=sha(p)
        write(trial/'RUNTIME.json',dict(status='locked',trial_id=tid,pins=pins,protected=rt['protected'],GT_read=False))
    write(scope/'COMPLETE_PIPELINE_LOCK.json',dict(status='locked',count=count,cohort=cohort,trials=trial_ids,recipe_sha256=sha(BASE/'FINAL_RECIPE_LOCK.json'),GT_read=False,no_further_method_selection=True))


def inputs(cohort,clock):
    prefix=cohort+'_'+clock
    # Same frozen teacher protocol; its subprocess must exit before Student.
    script='prepare_stvg_lite_reclocked_inputs002.py'
    command(prefix+'_inputs',script,'--cohort',cohort,'--clock',clock,'--prepare')
    folder=BASE/('reclocked_inputs'if clock=='temporal'else'native_inputs')if cohort=='development'else BASE/'confirmation'/('temporal_inputs'if clock=='temporal'else'native_inputs')
    runtime_path=folder/'TEACHER_RUNTIME.json';runtime=read(runtime_path)
    assert runtime['status']=='locked' and runtime['GT_read']is False and runtime['execution_authorized']is False
    assert runtime['study_id']=='stvg_lite_event_support_v1' and runtime['cohort']==cohort and runtime['clock']==clock
    for rel,h in {**runtime['pins'],**runtime['protected']}.items():assert sha(ROOT/rel)==h,rel
    before=sha(runtime_path);(folder/'TEACHER_RUNTIME_PREAUTHORIZATION.json').write_bytes(runtime_path.read_bytes())
    runtime['execution_authorized']=True;write(runtime_path,runtime,mutable=True)
    write(folder/'ROOT_TEACHER_AUTHORIZATION.json',dict(status='pass',runtime_sha256_before=before,runtime_sha256_after=sha(runtime_path),scope='root authorization under prelocked finite task; full input readback still required before model load',GT_read=False,CPU_only=True))
    command(prefix+'_inputs_root',script,'--cohort',cohort,'--clock',clock,'--readback')
    command(prefix+'_teacher',script,'--cohort',cohort,'--clock',clock,'--teacher',gpu=True,teacher=True)
    command(prefix+'_reference',script,'--cohort',cohort,'--clock',clock,'--references')


def final_student(cohort):
    prepare_trials(cohort)
    for tid in ('C_FULL','C_NATIVE'):
        command(cohort+'_'+tid+'_student','run_stvg_lite_complete_student_v1.py',cohort,tid,gpu=True)
        command(cohort+'_'+tid+'_root','review_stvg_lite_complete002.py',cohort,tid)
    command(cohort+'_five_arm_score','score_stvg_lite_complete_v1.py',cohort)


def main():
    rt,recipe=verify_template();EXEC.mkdir(exist_ok=False)
    resource.BASE=EXEC;resource.UNIT=UNIT
    g=1024**3
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines()if'='in x)
    assert [int(props[k])for k in ('MemoryHigh','MemoryMax','MemorySwapMax')]==[16*g,20*g,g]
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        write(BASE/'STATUS.json',dict(status='running',stage='final_reclocked_development',done=4,total=5,GT_read=False,time=time.time()),mutable=True)
        command('development_temporal_reference_recovery002','prepare_stvg_lite_reclocked_inputs002.py','--cohort','development','--clock','temporal','--references')
        cached_native_refs();final_student('development')
        # Frozen final recipe remains unchanged after development complete-loop scores.
        frozen=sha(BASE/'FINAL_RECIPE_LOCK.json')
        command('confirmation_source','continue_stvg_lite_complete005.py','--source-only',gpu=True)
        confirm_temporal_runtime()
        command('confirmation_temporal','run_stvg_lite_confirmation_temporal_v1.py','run',gpu=True,teacher=True)
        command('confirmation_temporal_root','run_stvg_lite_confirmation_temporal_v1.py','readback')
        inputs('confirmation','temporal');inputs('confirmation','native');final_student('confirmation')
        assert sha(BASE/'FINAL_RECIPE_LOCK.json')==frozen
        write(BASE/'MEASUREMENT_COMPLETION.json',dict(status='complete_pending_independent_study_view_public_archive',count=192,development_count=128,confirmation_count=64,GT_read=True,bootstrap_executed=False,CURRENT_promoted=False,recipe_unchanged_after_confirmation=True))
        write(BASE/'COMPLETE_STATUS005.json',dict(status='complete_pending_independent_study_view_public_archive',stage='root_handoff',count=192,GT_read=True,time=time.time()),mutable=True)
        write(BASE/'STATUS.json',dict(status='complete_pending_independent_study_view_public_archive',stage='root_handoff',done=5,total=5,GT_read=True,time=time.time()),mutable=True)
    finally:stop.set()


if __name__=='__main__':
    try:
        if '--source-only'in sys.argv:source_wrapper()
        else:main()
    except BaseException as e:
        write(BASE/'COMPLETE_STATUS005.json',dict(status='failed',stage='required_root_engineering_review',GT_read=False,time=time.time()),mutable=True)
        write(BASE/'COMPLETE_FAILURE005.json',dict(status='failed',stage='required_root_engineering_review',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True)
        raise
