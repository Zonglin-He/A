"""Bounded serial Native temporal study; no result-dependent continuation."""
from pathlib import Path
import os,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_semantic_view32_v1 import BASE,verify,read,write,sha
from scripts import continue_stvg_semantic_preference_4anchor32_v1 as handoff
UNIT='stvg-temporal-semantic-view32-001.service'
def release():
    handoff.BASE=BASE;return handoff.release()
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();handoff.BASE=BASE;handoff.UNIT=UNIT;resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='temporal_semantic_view32'
    g=2**30;p=handoff.props(UNIT,'MemoryHigh','MemoryMax','MemorySwapMax');assert [int(p[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*g,24*g,g]
    def trigger(v):
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;stop=threading.Event()
    try:
        qp=handoff.props(handoff.QUNIT,'MainPID','ActiveState','ControlGroup')
        if int(qp['MainPID'])==0 and qp['ActiveState']=='failed':
            assert read(handoff.OLD/'FAILURE.json')['status']=='technical_failure'
            write(BASE/'QWEN_INACTIVE_READBACK.json',dict(status='inactive_failed_interface_no_signal',properties=qp,original_failure_sha256=sha(handoff.OLD/'FAILURE.json'),Qwen_retried=False,GT_read=False))
        else:handoff.hold()
        v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
        write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',properties=p,admission=v,GT_read=False));tick=time.time();write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=tick,deadline_epoch=tick+14400,GT_read=False));threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
        stages=[('teacher','stvg_temporal_semantic_view32_v1.py'),('teacher_root','stvg_temporal_semantic_view32_v1.py'),('targets','stvg_temporal_semantic_grounder32_v1.py'),('fit','stvg_temporal_semantic_view32_v1.py'),('readout','stvg_temporal_semantic_grounder32_v1.py')]
        for stage,script in stages:
            verify();write(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,GT_read=False),mutable=True);folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False)
            py=ROOT/('.venv/stvg_native_distill_v1/bin/python' if stage=='teacher' else '.conda/tubedetr/bin/python');env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'worker.log').open('xb') as log:
                tick=time.perf_counter();child=subprocess.Popen([str(py),'-B',str(ROOT/'scripts'/script),stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=False))
                try:rc=child.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=False));assert rc==0,stage+' engineering failure; preserve all existing outputs'
        write(BASE/'STATUS.json',dict(status='primary_complete_pending_full',stage='full_continuation',count=32,GT_read=False),mutable=True);write(BASE/'PRIMARY_COMPLETION.json',dict(status='complete',count=32,GT_read=False))
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=32,GT_read=False),mutable=True);raise
    finally:stop.set();release()
if __name__=='__main__':release() if len(sys.argv)>1 and sys.argv[1]=='release' else main()
