"""Finite human-priority Q0 controller; preserves original stopped processes."""
import os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_common_v1 import BASE,PYTHON,read,write,status,sha,verify
from scripts.pause_stvg_opd_for_geometry_v1 import process
O=ROOT/'artifacts/stvg_opd_optimization_geometry_v1'
S=ROOT/'artifacts/stvg_opd_supplement_v1/baselines'


def pause_priority():
    assert not (BASE/'PRIORITY_PAUSE_RECEIPT.json').exists()
    if not (O/'GLOBAL_PREDICTION_BARRIER.json').exists():
        c=read(O/'CONTROLLER_STATUS.json');s=read(O/'GPU_STATUS.json')
        assert c['status']=='running' and c['stage']=='formal' and s['status']=='running_common_state_formal'
        cp=int(c['controller_pid']);wp=int(c['worker_pid']);assert wp==s['pid']
        assert 'common_state_geometry_lease006.py controller' in process(cp)['command']
        assert 'common_state_geometry_lease006.py formal' in process(wp)['command']
        folder=O/'common_state';status_path=O/'GPU_STATUS.json';triplet=True
        original_pause=read(O/'BASELINE_PAUSE_RECEIPT.json')
        additional=[process(original_pause[r]['pid']) for r in ['controller','worker']]
        assert all(p['state']=='T' for p in additional)
        task='O_common_state'
    else:
        assert (O/'BASELINE_RESUMPTION_RECEIPT.json').exists()
        s=read(S/'STATUS.json');assert s['status']=='running' and s['job']=='formal'
        cp=int(s['controller_pid']);wp=int(s['worker_pid'])
        assert 'continue_stvg_opd_baseline_supplement_v1.py' in process(cp)['command']
        assert 'run_stvg_opd_baseline_supplement_v1.py formal' in process(wp)['command']
        folder=S/'formal'/s['dataset']/s['candidate'];status_path=folder/'STATUS.json';triplet=False;additional=[]
        task='S_baseline'
    old=read(status_path);identities=[process(wp),process(cp)]
    os.kill(cp,signal.SIGSTOP)
    deadline=time.monotonic()+90
    try:
        while time.monotonic()<deadline:
            now=read(status_path)
            if now['done']>old['done'] and (not triplet or now['done']%3==0):break
            time.sleep(.01)
        else:raise RuntimeError('No next saved complete boundary; no worker signal sent')
        os.kill(wp,signal.SIGSTOP);time.sleep(.5)
        assert process(cp)['state']=='T' and process(wp)['state']=='T'
    except BaseException:
        os.kill(cp,signal.SIGCONT);raise
    prefix={}
    for p in sorted(folder.glob('*/*.pt')):
        rc=p.with_suffix('.json')
        if not rc.exists():continue
        h=sha(p);receipt=read(rc);assert h==receipt['sha256'] and not receipt['GT_read']
        prefix[str(p.relative_to(ROOT))]=dict(sha256=h,receipt_sha256=sha(rc),bytes=p.stat().st_size)
    assert len(prefix)>=now['done']
    verify()
    write(BASE/'PRIORITY_PAUSE_RECEIPT.json',dict(status='paused_original_processes_at_saved_boundary',
        task=task,prior_status=old,accepted_stage_snapshot=now,prefix=prefix,
        resume_roles=identities,must_remain_stopped=[process(wp),process(cp)]+additional,
        original_process_and_in_memory_state_preserved=True,no_kill_or_restart=True,GT_read=False,time=time.time()))
    print('Q0_PRIORITY_PAUSED',task,now['done'],wp,cp,flush=True)


def resume_priority():
    if (BASE/'PRIORITY_RESUMPTION_RECEIPT.json').exists():return
    pause=read(BASE/'PRIORITY_PAUSE_RECEIPT.json')
    for p,h in pause['prefix'].items():
        assert sha(ROOT/p)==h['sha256'] and sha((ROOT/p).with_suffix('.json'))==h['receipt_sha256']
    verify()
    for identity in pause['resume_roles']:
        p=process(identity['pid']);assert p['state']=='T' and p['start_ticks']==identity['start_ticks']
        os.kill(identity['pid'],signal.SIGCONT)
    write(BASE/'PRIORITY_RESUMPTION_RECEIPT.json',dict(status='same_original_processes_resumed_after_Q0_GPU_exit',
          resumed_task=pause['task'],prefix_files_reverified=len(pause['prefix']),original_memory_preserved=True,
          worker_pid=pause['resume_roles'][0]['pid'],controller_pid=pause['resume_roles'][1]['pid'],GT_read=False,time=time.time()))


def child(script,job,CPU=False):
    env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    if CPU:env['CUDA_VISIBLE_DEVICES']=''
    with (BASE/(job+'.log')).open('a') as log:
        p=subprocess.Popen([str(PYTHON),'-B',script,job],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        status(BASE/'STATUS.json',dict(status='running_Q0_'+job,stage=job,controller_pid=os.getpid(),worker_pid=p.pid,GT_read=job=='score',CPU_only=CPU,time=time.time()))
        assert p.wait()==0, 'Q0 '+job+' failed; actual evidence preserved'


def run():
    verify();pause_priority()
    try:
        child('scripts/run_stvg_opd_query_q0_v1.py','qualify')
        child('scripts/run_stvg_opd_query_q0_v1.py','qualification_root',True)
        child('scripts/run_stvg_opd_query_q0_v1.py','formal')
    finally:
        # child() returns only after its actual process exit, including failure.
        resume_priority()
    child('scripts/run_stvg_opd_query_q0_v1.py','deployment_root',True)
    child('scripts/score_stvg_opd_query_q0_v1.py','score',True)
    status(BASE/'STATUS.json',dict(status='pending_actual_independent_statistics_PDF_view_publication_archive',
          task_complete=False,GT_read=True,CPU_only=True,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'controller_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),time=time.time()));raise
