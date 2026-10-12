"""Scheduling-only revision: wait for actual O context exit; retain HC qualification."""
import inspect,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_common_v1 import BASE,PYTHON,read,write,status,sha,verify
import scripts.continue_stvg_opd_query_q0_v1 as original
REC=BASE/'serial_GPU_context002';O=original.O


def revision_verify():
    r=verify()
    for p,h in read(REC/'REVISION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    return r


def dispatch():
    original.verify=revision_verify
    for name in ['pause_priority','resume_priority']:
        fn=getattr(original,name);s=inspect.getsource(fn)
        s=s.replace("BASE/'PRIORITY_PAUSE_RECEIPT.json'","REC/'PRIORITY_PAUSE_RECEIPT.json'")
        s=s.replace("BASE/'PRIORITY_RESUMPTION_RECEIPT.json'","REC/'PRIORITY_RESUMPTION_RECEIPT.json'")
        ns=dict(original.__dict__,REC=REC)
        exec(compile(s,str(Path(__file__)), 'exec'),ns);setattr(original,name,ns[name])
    def child(script,job,CPU=False):
        if script=='scripts/run_stvg_opd_query_q0_v1.py':script='scripts/run_stvg_opd_query_q0_serial002.py'
        env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
        if CPU:env['CUDA_VISIBLE_DEVICES']=''
        else:env.pop('CUDA_VISIBLE_DEVICES',None)
        with (REC/(job+'.log')).open('a') as log:
            p=subprocess.Popen([str(PYTHON),'-B',script,job],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            status(BASE/'STATUS.json',dict(status='running_Q0_serial002_'+job,stage=job,controller_pid=os.getpid(),worker_pid=p.pid,CPU_only=CPU,GT_read=job=='score',time=time.time()))
            assert p.wait()==0,'Q0 actual serial002 '+job+' failed; evidence preserved'
    original.child=child


def run():
    revision_verify();dispatch()
    while not (O/'BASELINE_RESUMPTION_RECEIPT.json').exists():
        s=read(O/'CONTROLLER_STATUS.json');assert 'failed' not in s['status'],'Actual O dependency failed'
        status(BASE/'STATUS.json',dict(status='CPU_waiting_for_actual_O_GPU_exit_then_priority_Q0',
              CPU_only=True,pid=os.getpid(),GT_read=False,qualification_complete_queries=2,formal_done=0,total=64,time=time.time()))
        time.sleep(15)
    assert read(O/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed'
    old_worker=read(O/'GPU_STATUS.json')['pid'];assert not (Path('/proc')/str(old_worker)).exists()
    original.run()


if __name__=='__main__':
    try:run()
    except BaseException:
        p=REC/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',revision='serial002',evidence=str(p.relative_to(ROOT)),time=time.time()));raise
