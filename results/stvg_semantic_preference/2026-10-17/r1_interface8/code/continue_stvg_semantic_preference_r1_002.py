"""Run the already locked R1 half now; reliably release only the verified Qwen CPU controller."""
from pathlib import Path
import os,signal,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_semantic_preference_interface_v2 import BASE,QWEN,read,write,sha,verify,categorical_from_logprobs,restore_candidate_order
UNIT='stvg-semantic-preference-r1-002.service'
def release():
    r=read(BASE/'QWEN_PRIORITY_HANDOFF002.json');pid=r['Qwen_pid'];p=Path(f'/proc/{pid}/stat')
    if not p.exists():
        write(BASE/'QWEN_RESUME_RECEIPT002.json',dict(status='identity_process_missing_no_signal',GT_read=False),mutable=True);return
    actual=p.read_text().split()[21]
    assert actual==r['Qwen_proc_start_ticks'],'refuse signal reused PID'
    state=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',r['Qwen_unit'],'-p','MainPID,ControlGroup'],text=True).splitlines())
    assert int(state['MainPID'])==pid and Path(f'/proc/{pid}/cgroup').read_text().strip().endswith(state['ControlGroup'])
    if (BASE/'QWEN_RESUME_RECEIPT002.json').exists() and read(BASE/'QWEN_RESUME_RECEIPT002.json')['status']=='resumed_verified_CPU_controller':return
    os.kill(pid,signal.SIGCONT)
    status=read(QWEN/'STATUS.json')
    if status.get('status')=='waiting_priority_R1':write(QWEN/'STATUS.json',r['Qwen_original_status'],mutable=True)
    write(BASE/'QWEN_RESUME_RECEIPT002.json',dict(status='resumed_verified_CPU_controller',pid=pid,original_process_identity_verified=True,downloader_not_signalled=True,GT_read=False),mutable=True)
def root():
    d=read(BASE/'teachers/R1/QUALIFICATION_COMPLETION.json');assert d['status']=='pass' and d['count']==8 and d['actual_prefill_calls']<=16
    jobs={j['job_id']:j for j in read(BASE/'INPUT_JOB_LOCK.json')['records']};seen=set()
    for f in d['files']:
        assert sha(ROOT/f['path'])==f['sha256'];o=read(ROOT/f['path']);j=jobs[o['job_id']];seen.add(o['job_id'])
        for e in o['evidence']:
            p=categorical_from_logprobs(e['raw_label_logprobs']);assert p==e['conditional_label_posterior']
            assert restore_candidate_order(p,e['display_order'])==e['candidate_order_posterior']
        if len(j['candidates'])>=2:
            assert len(o['evidence'])==2
            avg=[sum(e['candidate_order_posterior'][k] for e in o['evidence'])/2 for k in range(len(j['candidates'])+1)]
            assert avg==o['permutation_averaged_preference'] and abs(sum(avg)-1)<1e-12
    assert seen==set(read(BASE/'QUALIFICATION_PLAN.json')['probe_jobs'])
    write(BASE/'R1_ROOT_READBACK.json',dict(status='pass',count=8,actual_prefill_calls=d['actual_prefill_calls'],teacher_receipt_sha256=sha(BASE/'teachers/R1/QUALIFICATION_COMPLETION.json'),free_generations=0,Student_fits=0,new_GT_readers=0,scope='R1 interface only, semantic accuracy and adaptation value untested',GT_read=False))
    write(BASE/'R1_PUBLICATION_HANDOFF.json',dict(status='ready',count=8,scope='R1 preference interface engineering only',GT_read=False))
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    verify();resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='R1_preference_interface'
    props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines())
    assert [int(props[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*2**30,24*2**30,2**30]
    def trigger(v):
        g=2**30
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        v=resource.sample();assert v['available']>=8*2**30 and v['gpu_free']>=26*2**30 and v['disk_free']>=30*2**30
        write(BASE/'R1_ACTUAL_LIVE_RESOURCE_LIMITS002.json',dict(status='pass',properties=props,admission=v,GT_read=False))
        write(BASE/'R1_ACTIVE_EXECUTION002.json',dict(status='running',start_epoch=time.time(),deadline_epoch=time.time()+7200,cap_prefill=16,GT_read=False))
        write(BASE/'STATUS.json',dict(status='running',stage='R1_interface_early',count=8,GT_read=False),mutable=True)
        folder=BASE/'execution002/R1';folder.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        with (folder/'worker.log').open('xb') as log:
            tick=time.perf_counter();child=subprocess.Popen([str(ROOT/'.venv/stvg_native_distill_v1/bin/python'),'-B',str(ROOT/'scripts/stvg_semantic_preference_interface_v2.py'),'R1'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=False))
            try:code=child.wait(timeout=max(1,read(BASE/'R1_ACTIVE_EXECUTION002.json')['deadline_epoch']-time.time()))
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30);raise
        elapsed=time.perf_counter()-tick;write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=code,wall_seconds=elapsed,GT_read=False));assert code==0,'R1 interface failed; preserved, no model retry'
        root();write(BASE/'R1_COMPLETION002.json',dict(status='complete',count=8,compute_seconds=elapsed,Student_fits=0,new_GT_readers=0,GT_read=False))
        write(BASE/'STATUS.json',dict(status='R1_interface_complete_Qwen_pending',stage='root_R1_publication',count=8,GT_read=False),mutable=True)
    except BaseException as e:
        write(BASE/'R1_FAILURE002.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False))
        write(BASE/'STATUS.json',dict(status='failed',stage='R1_interface_early',count=8,GT_read=False),mutable=True);raise
    finally:stop.set();release()
if __name__=='__main__':release() if len(sys.argv)>1 and sys.argv[1]=='release' else main()
