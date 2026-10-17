"""Single finite serial supervisor; original guards, no unrelated unit signals."""
from pathlib import Path
import os,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_pairwise_caption256_v1 import BASE,read,write,verify
UNIT='stvg-pairwise-caption256-001.service'
def main():
    from scripts import launch_stvg_instance_identity_v1 as r
    from scripts.continue_stvg_semantic_preference_4anchor32_v1 import props
    verify();r.BASE=BASE;r.UNIT=UNIT;r.STAGE='pairwise_caption256';g=2**30
    limits=props(UNIT,'MemoryHigh','MemoryMax','MemorySwapMax');assert [int(limits[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*g,24*g,g]
    def trigger(v):
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active']and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    r.trigger=trigger;v=r.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',unit=UNIT,properties=limits,admission=v,GT_read=False));write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=time.time(),deadline_epoch=time.time()+6*3600,GT_read=False));stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start()
    stages=[('teacher_R1','teacher',['R1']),('teacher_QWEN','teacher',['QWEN']),('teacher_root','teacher_root',[]),('fit_R1','fit',['R1']),('fit_QWEN','fit',['QWEN']),('readout','readout',[]),('score','score',[]),('report','report',[])]
    try:
        for stage,fn,args in stages:
            verify();folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False);write(BASE/'STATUS.json',dict(status='running',stage=stage,count=256,GT_read=stage=='score'),mutable=True)
            envname='.venv/stvg_native_distill_v1/bin/python'if stage=='teacher_R1'else'.venv/stvg_qwen35_geometry_v1/bin/python'if stage=='teacher_QWEN'else'.conda/tubedetr/bin/python';script='report_stvg_pairwise_caption256_v1.py'if stage=='report'else'stvg_pairwise_caption256_v1.py';cmd=[str(ROOT/envname),'-B',str(ROOT/'scripts'/script)]+([]if stage=='report'else[fn,*args])
            with (folder/'worker.log').open('xb')as log:
                tick=time.perf_counter();p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'))
                try:rc=p.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage
        write(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',count=256,GT_read=True),mutable=True);write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=256,GT_read=False))
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(status='technical_failure',error=repr(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=256,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':main()
