"""One serial, guarded four-anchor R1 study; downloader stays active throughout."""
from pathlib import Path
import inspect,os,signal,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_semantic_preference_4anchor32_v1 import BASE,OLD,verify,read,write,sha
UNIT='stvg-semantic-preference-4anchor32-001.service'
QUNIT='stvg-qwen35-geometry32-002.service'
def props(unit,*names):return dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',unit,'-p',','.join(names)],text=True).splitlines())
def hold():
    q=props(QUNIT,'MainPID','ControlGroup','ActiveState');pid=int(q['MainPID']);assert pid>0 and q['ActiveState']=='active'
    original=read(OLD/'STATUS.json');assert original['status']=='waiting_download' and not (OLD/'ACTIVE_EXECUTION.json').exists()
    cg=Path('/sys/fs/cgroup')/q['ControlGroup'].lstrip('/');owned={int(x) for f in cg.rglob('cgroup.procs') for x in f.read_text().split()};assert owned=={pid}
    ticks=Path(f'/proc/{pid}/stat').read_text().split()[21]
    write(BASE/'QWEN_PRIORITY_HANDOFF002.json',dict(status='verified_CPU_priority_handoff',Qwen_unit=QUNIT,Qwen_pid=pid,Qwen_proc_start_ticks=ticks,Qwen_original_status=original,R1_unit=UNIT,proof_only_CPU_controller_in_cgroup=True,download_not_paused=True,GT_read=False))
    os.kill(pid,signal.SIGSTOP)
    for _ in range(20):
        if Path(f'/proc/{pid}/stat').read_text().split()[2]=='T':break
        time.sleep(.01)
    assert Path(f'/proc/{pid}/stat').read_text().split()[2]=='T'
    owned={int(x) for f in cg.rglob('cgroup.procs') for x in f.read_text().split()};assert owned=={pid} and not (OLD/'ACTIVE_EXECUTION.json').exists(),'Qwen model race; release without signalling model'
    write(OLD/'STATUS.json',dict(status='waiting_priority_R1_four_anchor32',stage='verified_CPU_handoff',count=32,actual_model_calls=0,GT_read=False),mutable=True)

def release():
    if not (BASE/'QWEN_PRIORITY_HANDOFF002.json').exists():return
    if (BASE/'QWEN_RESUME_RECEIPT002.json').exists() and read(BASE/'QWEN_RESUME_RECEIPT002.json')['status']=='resumed_verified_CPU_controller':return
    r=read(BASE/'QWEN_PRIORITY_HANDOFF002.json');pid=r['Qwen_pid'];p=Path(f'/proc/{pid}/stat')
    if not p.exists():
        write(BASE/'QWEN_RESUME_RECEIPT002.json',dict(status='identity_process_missing_no_signal',GT_read=False),mutable=True);return
    assert p.read_text().split()[21]==r['Qwen_proc_start_ticks'];q=props(QUNIT,'MainPID','ControlGroup');assert int(q['MainPID'])==pid and Path(f'/proc/{pid}/cgroup').read_text().strip().endswith(q['ControlGroup'])
    os.kill(pid,signal.SIGCONT)
    status=read(OLD/'STATUS.json')
    if status.get('status')=='waiting_priority_R1_four_anchor32':write(OLD/'STATUS.json',r['Qwen_original_status'],mutable=True)
    write(BASE/'QWEN_RESUME_RECEIPT002.json',dict(status='resumed_verified_CPU_controller',pid=pid,original_process_identity_verified=True,downloader_not_signalled=True,GT_read=False),mutable=True)

def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='four_anchor32'
    p=props(UNIT,'MemoryHigh','MemoryMax','MemorySwapMax');assert [int(p[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*2**30,24*2**30,2**30]
    def trigger(v):
        g=2**30
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        hold();v=resource.sample();assert v['available']>=8*2**30 and v['gpu_free']>=26*2**30 and v['disk_free']>=30*2**30
        write(BASE/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',properties=p,admission=v,GT_read=False));tick=time.time();write(BASE/'ACTIVE_EXECUTION.json',dict(status='running',start_epoch=tick,deadline_epoch=tick+7200,GT_read=False));write(BASE/'R1_ACTIVE_EXECUTION002.json',dict(status='running',start_epoch=tick,deadline_epoch=tick+7200,GT_read=False))
        stages=[('teacher','collect'),('setup','setup'),('lock','lock'),('GEOM','worker'),('GEOM_root','root'),('PREF','worker'),('PREF_root','root'),('propagate','propagate'),('score','score')]
        for stage,command in stages:
            verify();write(BASE/'STATUS.json',dict(status='running',stage=stage,count=32,GT_read=stage=='score'),mutable=True)
            folder=BASE/'execution'/stage;folder.mkdir(parents=True,exist_ok=False);py=ROOT/('.venv/stvg_native_distill_v1/bin/python' if stage=='teacher' else '.conda/tubedetr/bin/python');args=[command]+([stage.split('_')[0]] if command in ['worker','root'] else [])
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'worker.log').open('xb') as f:
                t=time.perf_counter();child=subprocess.Popen([str(py),'-B',str(ROOT/'scripts/stvg_semantic_preference_4anchor32_v1.py'),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=stage=='score'))
                try:rc=child.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=stage=='score'));assert rc==0,stage+' failed; preserve without scientific retry'
        write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=32,GT_read=False));write(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',stage='root_publication',count=32,GT_read=False),mutable=True)
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',count=32,GT_read=False),mutable=True);raise
    finally:stop.set();release()
if __name__=='__main__':release() if len(sys.argv)>1 and sys.argv[1]=='release' else main()
