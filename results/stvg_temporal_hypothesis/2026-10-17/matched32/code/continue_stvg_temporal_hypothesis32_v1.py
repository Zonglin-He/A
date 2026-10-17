"""Finite B Teacher-signal ->512 Student -> unchanged Full; no score-based retry."""
from pathlib import Path
import os,sys,subprocess,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_hypothesis32_v1 import BASE,read,write,sha,verify
UNIT='stvg-temporal-hypothesis32-001.service'
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    from scripts.continue_stvg_semantic_preference_4anchor32_v1 import props
    verify();assert read(BASE/'DEVELOPMENT_GATE.json')['status']=='triggered';resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='native_hypothesis32';g=2**30;p=props(UNIT,'MemoryHigh','MemoryMax','MemorySwapMax');assert [int(p[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*g,24*g,g]
    def trigger(v):
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',properties=p,admission=v,GT_read=False));now=time.time();write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=now,deadline_epoch=now+14400,includes_Full_not_reset=True,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    def stage(label,script,command=None,teacher=False,gt=False):
        verify();write(BASE/'STATUS.json',dict(status='running',stage=label,count=32,GT_read=gt),mutable=True);folder=BASE/'execution'/label;folder.mkdir(parents=True,exist_ok=False);py=ROOT/('.venv/stvg_native_distill_v1/bin/python' if teacher else '.conda/tubedetr/bin/python');env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        with (folder/'worker.log').open('xb') as log:
            tick=time.perf_counter();child=subprocess.Popen([str(py),'-B',str(ROOT/'scripts'/script)]+([command] if command else []),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=gt))
            try:rc=child.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
        write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=gt));assert rc==0,label
    try:
        script='stvg_temporal_hypothesis32_v1.py'
        for command in ['teacher','teacher_root','fit','readout']:stage(command,script,command,teacher=command=='teacher')
        informative=read(BASE/'SIGNAL_COMPLETION.json')['informative_count']
        if informative:
            stage('full_prepare','stvg_temporal_hypothesis_full32_v1.py','prepare');full=BASE/'full';paths=[ROOT/'scripts'/n for n in ['stvg_temporal_hypothesis_full32_v1.py','stvg_temporal_semantic_full32_v1.py','stvg_r1_geometry32_measure_v1.py','stvg_overnight_student_v1.py','run_stvg_instance_student32_v1.py','run_stvg_soft_ep_coordinate_trial003.py','stvg_soft_ep_coordinate_fit002.py','stvg_spatial_lite_fit_v1.py','stvg_soft_ep_response_v1.py','audit_stvg_spatial_lite_fit002.py','review_stvg_lite_complete002.py']]+[full/n for n in ['INPUT_BARRIER.json','INPUT_JOB_LOCK.json','TEMPORAL_LOCK.json','EXACT_REFERENCE_REUSE_LOCK.json','SOURCE_CAPTURE_BARRIER.json','ROSTER_LOCK.json']]+[BASE/n for n in ['PRIMARY_GLOBAL_PREDICTION_BARRIER.json','PRIMARY_ROOT_READBACK.json','PRIMARY_COMPLETION.json','FIT_BARRIER.json','EXECUTION_RUNTIME.json']];pins=dict(read(BASE/'EXECUTION_RUNTIME.json')['pins']);pins.update({str(p.relative_to(ROOT)):sha(p) for p in paths});write(full/'EXECUTION_RUNTIME.json',dict(status='locked',count=32,pins=pins,GT_read=False));write(full/'ACTIVE_EXECUTION.json',dict(read(BASE/'ACTIVE_EXECUTION.json'),same_parent_deadline_no_reset=True))
            for command in ['teacher','teacher_root','setup','worker','root','integrated']:stage('full_'+command,'stvg_temporal_hypothesis_full32_v1.py',command,teacher=command=='teacher')
        else:write(BASE/'FULL_NOT_RUN.json',dict(status='not_run_no_Teacher_signal',count=32,GT_read=False))
        stage('score','score_stvg_temporal_hypothesis32_v1.py',gt=True);write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False));write(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',count=32,GT_read=False),mutable=True);write(BASE/'COMPLETION.json',dict(status='complete',count=32,GT_read=True))
    except BaseException as e:write(BASE/'FAILURE.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=32,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':main()
