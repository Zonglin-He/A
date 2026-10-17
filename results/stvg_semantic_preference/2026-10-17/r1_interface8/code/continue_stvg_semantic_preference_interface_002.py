"""One bounded serial interface follow-up. It dispatches zero Student fits or GT scores."""
from pathlib import Path
import os,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_semantic_preference_interface_v2 import BASE,QWEN,read,write,sha,verify as verify_shared
UNIT='stvg-semantic-preference-interface-002.service'
def verify():
    d=verify_shared()
    for p,h in read(BASE/'FOLLOWUP_RUNTIME003.json')['pins'].items():assert sha(ROOT/p)==h,p
    return d
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    runtime=verify();deadline=runtime['dependency_deadline_epoch']
    props=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines())
    assert [int(props[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*2**30,24*2**30,2**30]
    write(BASE/'QWEN_ACTUAL_LIVE_RESOURCE_LIMITS002.json',dict(status='pass',unit=UNIT,properties=props,GT_read=False))
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='preference_interface'
    def trigger(v):
        g=2**30
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        while True:
            verify();assert time.time()<deadline,'bounded dependency wait expired'
            p=QWEN/'publication/measured32/PUBLICATION_COMPLETION.json'
            state=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show','stvg-qwen35-geometry32-002.service','-p','ActiveState,Result,MainPID'],text=True).splitlines())
            if state['ActiveState']=='failed':raise RuntimeError('Qwen Direct32 dependency failed; no automatic recovery/model retry')
            if p.exists() and read(p)['status']=='complete' and state['MainPID']=='0' and state['Result']=='success':
                assert read(QWEN/'CPU_COMPLETION.json')['count']==32
                assert read(QWEN/'arms/QWEN_OPD/trials/G32/ROOT_READBACK.json')['status']=='pass'
                assert read(QWEN/'D2_ROOT_READBACK.json')['status']=='pass'
                write(BASE/'QWEN_DEPENDENCY_COMPLETION002.json',dict(status='pass',count=32,receipts={str(x.relative_to(ROOT)):sha(x) for x in [p,QWEN/'CPU_COMPLETION.json',QWEN/'QUALIFICATION_COMPLETION.json',QWEN/'MODEL_SNAPSHOT_LOCK.json',QWEN/'publication/measured32/FINAL_GITHUB_RECEIPT.json']},GT_read=False));break
            time.sleep(20)
        while True:
            assert time.time()<deadline;v=resource.sample()
            if v['gpu_free']>=26*2**30 and v['available']>=8*2**30 and v['disk_free']>=30*2**30:break
            time.sleep(20)
        assert read(BASE/'R1_COMPLETION002.json')['status']=='complete' and read(BASE/'R1_ROOT_READBACK.json')['status']=='pass'
        remaining=7200-read(BASE/'R1_COMPLETION002.json')['compute_seconds'];assert remaining>0
        assert not (BASE/'Qwen_ACTIVE_EXECUTION002.json').exists(),'do not reset started qualification budget'
        write(BASE/'Qwen_ACTIVE_EXECUTION002.json',dict(status='running',start_epoch=time.time(),deadline_epoch=time.time()+remaining,scope='maximum32 prefill forwards/no Student/no GT',GT_read=False))
        for stage in ['Qwen','root']:
            verify();write(BASE/'STATUS.json',dict(status='running',stage=stage,count=8,GT_read=False),mutable=True)
            folder=BASE/'execution002'/stage;folder.mkdir(parents=True,exist_ok=False)
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            py=ROOT/('.venv/stvg_qwen35_geometry_v1/bin/python' if stage=='Qwen' else '.venv/stvg_native_distill_v1/bin/python' if stage=='R1' else '.conda/tubedetr/bin/python')
            with (folder/'worker.log').open('xb') as log:
                tick=time.perf_counter();child=subprocess.Popen([str(py),'-B',str(ROOT/'scripts/stvg_semantic_preference_interface_v2.py'),stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,stage=stage,GT_read=False))
                try:code=child.wait(timeout=max(1,read(BASE/'Qwen_ACTIVE_EXECUTION002.json')['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:
                    subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=SIGTERM',UNIT],check=True);raise
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=code,wall_seconds=time.perf_counter()-tick,GT_read=False));assert code==0,'interface stage failed; outputs preserved, no retry'
    except BaseException as e:
        write(BASE/'QWEN_FAILURE002.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False))
        write(BASE/'STATUS.json',dict(status='failed',stage='interface_dependency_or_execution',count=8,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':main()
