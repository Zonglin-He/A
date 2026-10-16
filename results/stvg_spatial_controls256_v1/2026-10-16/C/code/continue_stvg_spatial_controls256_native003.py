"""Human-authorized six-group schedule; original scientific functions unchanged."""
from pathlib import Path
import ast, inspect, os, select, subprocess, sys, threading, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import continue_stvg_spatial_controls256_native002 as previous
from scripts import run_stvg_spatial_controls256_native_v1 as old
from scripts.prepare_stvg_spatial_controls256_v1 import BASE, read, put, sha
UNIT='stvg-spatial-controls256-native-003.service'
CORE=('Full_E1','A','B','C','A_BT','B_BT')
def verify():
    for p,h in read(BASE/'NATIVE_DISPATCH_RUNTIME003.json')['pins'].items():assert sha(ROOT/p)==h,p
    previous.verify()
    d=read(BASE/'EXECUTION_SCOPE_LOCK003.json');assert tuple(d['active_groups'])==CORE
def tasks():
    result=[('qualification',None)]
    for arm in CORE:
        if arm not in ('A_BT','B_BT'):result += [('teacher',arm),('teacher_root',arm)]
        result += [(s,arm) for s in ('student_prepare','student_worker','student_root','score')]
    return result
def contracts():
    t=tasks();assert len(t)==33 and [a for s,a in t if s=='score']==list(CORE)
    assert [a for s,a in t if s=='teacher']==list(CORE[:4])
    assert all(a not in ('FULL_ORDER','A_ORDER','B_G10','B_GR11') for _,a in t)
    for a in CORE:assert t.index(('student_root',a))<t.index(('score',a))
    from scripts import run_stvg_spatial_controls256_student_v1 as student
    for a in CORE:assert student.recipe(a)==('lite_backtrack' if a.endswith('_BT') else 'lite_fullstep')
    print('CORE6_SCHEDULE_CONTRACTS_PASS_NO_MODEL_NO_GT')
def wait_input(pid):
    if not (BASE/'SPATIAL_INPUT_HANDOFF.json').exists():
        fd=os.pidfd_open(pid);p=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)
    verify()
def supervise(pid):
    wait_input(pid)
    from scripts import launch_stvg_instance_identity_v1 as runner
    from scripts.stvg_native_instance_common_v1 import write
    runner.BASE=BASE;runner.UNIT=UNIT;runner.STAGE='native256_core6'
    v=runner.sample();g=1024**3;assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=runner.monitor,args=(stop,),daemon=True).start()
    try:
        for i,(stage,arm) in enumerate(tasks()):
            write(BASE/'CORE6_STATUS003.json',dict(status='running',stage=stage,arm=arm,done=i,total=len(tasks()),count=256,GT_read=stage=='score'),mutable=True)
            out=BASE/'execution/native256_core6'/(arm or 'qualification')/stage;out.mkdir(parents=True,exist_ok=False)
            if stage.startswith('student_'):script=ROOT/'scripts/run_stvg_spatial_controls256_student003.py';mode=stage.split('_',1)[1]
            elif stage=='score':script=ROOT/'scripts/score_stvg_spatial_controls256002.py';mode=arm
            else:script=Path(__file__);mode=stage
            inter='.venv/stvg_native_distill_v1/bin/python' if stage in ('qualification','teacher') else '.conda/tubedetr/bin/python'
            args=[str(ROOT/inter),'-B',str(script),mode]+([arm] if arm and stage!='score' else [])
            with (out/'worker.log').open('xb') as log:
                tick=time.perf_counter();p=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
                put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
            put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,(stage,arm)
            if stage=='score':
                put(BASE/'trials'/arm/'GROUP_PUBLICATION_HANDOFF003.json',dict(status='ready_for_independent_export_view_public',count=256,arm=arm,GT_read=True,root_sha256=sha(BASE/'trials'/arm/'ROOT_READBACK.json'),cpu_sha256=sha(BASE/'trials'/arm/'CPU_COMPLETION002.json')))
        put(BASE/'CORE6_MEASUREMENT_HANDOFF003.json',dict(status='complete_pending_independent_semantic_public_archive',count=256,group_count=6,GT_read=True))
        write(BASE/'CORE6_STATUS003.json',dict(status='root_handoff',stage='core6_measurements_complete',done=len(tasks()),total=len(tasks()),count=256,GT_read=True),mutable=True)
    except Exception as e:
        put(BASE/'CORE6_FAILURE003.json',dict(status='failed',stage=stage,arm=arm,error_type=type(e).__name__,error=str(e),GT_read=stage=='score'));raise
    finally:stop.set()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='contracts':contracts()
    elif mode=='after_inputs':supervise(int(sys.argv[2]))
    else:
        verify()
        if mode=='qualification':old.qualification()
        elif mode=='teacher':assert sys.argv[2] in CORE[:4];old.teacher(sys.argv[2])
        else:
            previous.old.teacher_root(sys.argv[2])
            if sys.argv[2]=='Full_E1':
                from scripts.audit_stvg_spatial_controls256_policies_v1 import main
                main()
