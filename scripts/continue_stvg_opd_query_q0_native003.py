"""Finite Native-support Q0 priority after actual O GPU context exit."""
import os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_native_scope003 import BASE,context
context()
import scripts.continue_stvg_opd_query_q0_v1 as original
from scripts.decota_matrix_common_v1 import read,status


def run():
    original.verify()
    while not (original.O/'BASELINE_RESUMPTION_RECEIPT.json').exists():
        s=read(original.O/'CONTROLLER_STATUS.json');assert 'failed' not in s['status']
        status(BASE/'STATUS.json',dict(status='CPU_waiting_for_actual_O_context_exit_then_Native_Q0_priority',pid=os.getpid(),total=64,done=0,
              new_scope_qualification_queries=0,GT_read=False,CPU_only=True,time=time.time()))
        time.sleep(15)
    assert read(original.O/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed'
    assert not (Path('/proc')/str(read(original.O/'GPU_STATUS.json')['pid'])).exists()
    def child(script,job,CPU=False):
        if script=='scripts/run_stvg_opd_query_q0_v1.py':script='scripts/run_stvg_opd_query_q0_native003.py'
        env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        if CPU:env['CUDA_VISIBLE_DEVICES']=''
        else:env.pop('CUDA_VISIBLE_DEVICES',None)
        # The old scorer binds BASE/PUB at import time; set the isolated context first.
        command=[str(original.PYTHON),'-B',script,job]
        if script=='scripts/score_stvg_opd_query_q0_v1.py':
            command=[str(original.PYTHON),'-B','-c','from scripts.stvg_opd_query_q0_native_scope003 import context; context(); from scripts.score_stvg_opd_query_q0_v1 import run; run()']
        with (BASE/(job+'.log')).open('a') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            status(BASE/'STATUS.json',dict(status='running_Native_Q0_'+job,stage=job,controller_pid=os.getpid(),worker_pid=p.pid,GT_read=job=='score',CPU_only=CPU,time=time.time()))
            assert p.wait()==0,'Native Q0 actual '+job+' failed; preserve'
    original.child=child;original.run()


if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'controller_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p),time=time.time()));raise
