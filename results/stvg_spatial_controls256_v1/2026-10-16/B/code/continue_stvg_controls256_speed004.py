"""Adopt one live Teacher without signalling it, qualify speed and finish six arms."""
from pathlib import Path
import argparse,hashlib,json,os,select,signal,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE,read,put,sha
from scripts.stvg_native_instance_common_v1 import write
from scripts import stvg_pipeline_speed_v1 as speed
UNIT='stvg-spatial-controls256-native-004.service'
OLD_UNIT='stvg-spatial-controls256-native-003.service'
CORE=('Full_E1','A','B','C','A_BT','B_BT')
def verify():
    from scripts.continue_stvg_spatial_controls256_native003 import verify as original
    original()
    for p,h in read(BASE/'NATIVE_DISPATCH_RUNTIME004.json')['pins'].items():assert sha(ROOT/p)==h,p
def identity(pid):
    p=Path('/proc')/str(pid);raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,state=fields[0],ppid=int(fields[1]),start_ticks=int(fields[19]),cmdline_sha256=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest(),cgroup=(p/'cgroup').read_text(),exit_status=int(fields[49]))
def tasks():
    out=[('teacher_root','Full_E1'),('speed_qualification',None)]
    for arm in CORE:
        if arm!='Full_E1' and arm not in ('A_BT','B_BT'):out += [('teacher',arm),('teacher_root',arm)]
        out += [(s,arm) for s in ('student_prepare','student_worker','student_root','score')]
    return out
def adopt(controller,teacher):
    verify();lock=read(speed.BASE/'LIVE_HANDOFF_LOCK.json');assert controller==lock['controller']['pid'] and teacher==lock['teacher']['pid']
    for kind,pid in (('controller',controller),('teacher',teacher)):
        now=identity(pid)
        for k in ('pid','ppid','start_ticks','cmdline_sha256','cgroup'):assert now[k]==lock[kind][k],(kind,k)
    assert identity(teacher)['ppid']==controller
    # Pause only the CPU coordinator. Its live R1 child remains in the original
    # protected unit and runs to a normal exit without interruption.
    os.kill(controller,signal.SIGSTOP)
    time.sleep(.05);assert identity(controller)['state'] in ('T','t')
    put(speed.BASE/'CONTROLLER_HANDOFF_STATUS.json',dict(status='waiting_original_teacher_normal_exit',pid=teacher,controller_pid=controller,count=256,GT_read=False,live_GPU_worker_signalled=False))
    fd=os.pidfd_open(teacher);poll=select.poll();poll.register(fd,select.POLLIN)
    from scripts import launch_stvg_instance_identity_v1 as r
    def old_sample():
        # The original supervisor is paused. Its hardware limits remain live;
        # retain its exact resource guard by sampling its original cgroup.
        selfcg=Path('/proc/self/cgroup').read_text();oldcg=lock['teacher']['cgroup']
        assert selfcg.startswith('0::') and oldcg.startswith('0::')
        cg=Path('/sys/fs/cgroup')/oldcg.splitlines()[0][3:].lstrip('/')
        stats={k:int(v) for k,v in (x.split() for x in (cg/'memory.stat').read_text().splitlines())}
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0])*2**20
        import shutil
        return dict(current=int((cg/'memory.current').read_text()),anon=stats.get('anon',0)+stats.get('shmem',0),available=available,gpu_free=free,own_gpu_active=True,disk_free=shutil.disk_usage(ROOT).free)
    try:
        while not poll.poll(1000):
            value=old_sample();why=r.trigger(value)
            if why:
                put(speed.BASE/'HANDOFF_RESOURCE_STOP_RECEIPT.json',dict(status='original_resource_guard_stop',reason=why,GT_read=False))
                subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=SIGTERM',OLD_UNIT],check=True);raise RuntimeError(why)
        end=identity(teacher);assert end['state']=='Z' and end['start_ticks']==lock['teacher']['start_ticks'];code=os.waitstatus_to_exitcode(end['exit_status']);assert code==0,code
        b=BASE/'teacher/Full_E1/TEACHER_GLOBAL_BARRIER.json';bar=read(b);assert bar['status']=='complete' and bar['count']==256 and not bar['GT_read']
        put(BASE/'execution/native256_core6/Full_E1/teacher/ADOPTED_WORKER_EXIT004.json',dict(status='exited',exit_code=code,pid=teacher,barrier_sha256=sha(b),normal_exit_observed_from_original_child_zombie=True,not_reaped_by_new_coordinator=True,GT_read=False))
        # Only the coordinator and an already-exited child remain. Stopping the
        # old unit now cannot interrupt model inference or lose Teacher work.
        subprocess.run(['systemctl','--user','stop',OLD_UNIT],check=True,timeout=120)
        put(speed.BASE/'CONTROLLER_HANDOFF_COMPLETION.json',dict(status='complete',count=256,original_teacher_exit_code=code,original_GPU_signalled=False,original_controller_retired_after_GPU_exit=True,GT_read=False))
    finally:os.close(fd)
def supervise(controller,teacher):
    adopt(controller,teacher);verify()
    from scripts import launch_stvg_instance_identity_v1 as r
    r.BASE=BASE;r.UNIT=UNIT;r.STAGE='native256_speed004'
    v=r.sample();assert v['available']>=8*2**30 and v['gpu_free']>=26*2**30 and v['disk_free']>=30*2**30
    stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start()
    stage=arm=None
    try:
        for i,(stage,arm) in enumerate(tasks()):
            write(BASE/'CORE6_STATUS004.json',dict(status='running',stage=stage,arm=arm,done=i,total=len(tasks()),count=256,GT_read=stage=='score'),mutable=True)
            if stage=='speed_qualification':
                out=speed.BASE/'execution/qualification';script=ROOT/'scripts/qualify_stvg_pipeline_speed_v1.py';mode=None
            else:
                out=BASE/'execution/native256_core6'/(arm or 'qualification')/stage
                if stage.startswith('student_'):script=ROOT/'scripts/run_stvg_controls256_speed_student_v1.py';mode=stage.split('_',1)[1]
                elif stage=='score':script=ROOT/'scripts/score_stvg_spatial_controls256002.py';mode=arm
                elif stage=='teacher':script=ROOT/'scripts/run_stvg_controls256_speed_teacher_v1.py';mode='teacher'
                else:script=ROOT/'scripts/continue_stvg_spatial_controls256_native003.py';mode=stage
            out.mkdir(parents=True,exist_ok=False)
            inter='.venv/stvg_native_distill_v1/bin/python' if stage=='teacher' else '.conda/tubedetr/bin/python'
            args=[str(ROOT/inter),'-B',str(script)]+([mode] if mode else [])+([arm] if arm and stage!='score' else [])
            with (out/'worker.log').open('xb') as log:
                tick=time.perf_counter();p=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
                put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
            put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,(stage,arm)
            if stage=='score':
                put(BASE/'trials'/arm/'GROUP_PUBLICATION_HANDOFF003.json',dict(status='ready_for_independent_export_view_public',count=256,arm=arm,GT_read=True,engineering_runtime_sha256=sha(BASE/'NATIVE_DISPATCH_RUNTIME004.json'),root_sha256=sha(BASE/'trials'/arm/'ROOT_READBACK.json'),cpu_sha256=sha(BASE/'trials'/arm/'CPU_COMPLETION002.json')))
        d=dict(status='complete_pending_independent_semantic_public_archive',count=256,group_count=6,engineering_revision='004',runtime_sha256=sha(BASE/'NATIVE_DISPATCH_RUNTIME004.json'),GT_read=True)
        put(BASE/'CORE6_MEASUREMENT_HANDOFF004.json',d);put(BASE/'CORE6_MEASUREMENT_HANDOFF003.json',dict(d,compatibility_receipt_for_original_global_waiter=True))
        write(BASE/'CORE6_STATUS004.json',dict(status='root_handoff',stage='core6_measurements_complete',done=len(tasks()),total=len(tasks()),count=256,GT_read=True),mutable=True)
    except Exception as e:
        put(BASE/'CORE6_FAILURE004.json',dict(status='failed',stage=stage,arm=arm,error_type=type(e).__name__,error=str(e),GT_read=stage=='score'));raise
    finally:stop.set()
def contracts():
    assert len(tasks())==32 and [a for s,a in tasks() if s=='score']==list(CORE)
    assert tasks()[0]==('teacher_root','Full_E1') and tasks()[1]==('speed_qualification',None)
    assert [a for s,a in tasks() if s=='teacher']==['A','B','C']
    for a in CORE:assert tasks().index(('student_root',a))<tasks().index(('score',a))
    assert os.waitstatus_to_exitcode(0)==0 and os.waitstatus_to_exitcode(256)==1
    print('SPEED004_HANDOFF_SCHEDULE_CONTRACTS_PASS_NO_MODEL_NO_GT')
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else supervise(int(sys.argv[2]),int(sys.argv[3]))
