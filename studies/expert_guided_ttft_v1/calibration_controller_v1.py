"""Single source-calibration owner. Stop on any fault; no retries or target run."""
from pathlib import Path
import os,sys,time,json,subprocess,traceback
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/expert_guided_ttft_v1/source_calibration_v1'
sys.path[:0]=[str(S),str(R/'tools'),str(R)]
from full import atomic,identity,read,sha
import calibration_runtime_v1 as run
def main():
 p=run.check();assert not(D/'controller.json').exists();atomic(D/'controller.json',dict(**identity(),plan_fingerprint=p['fingerprint']));exits={}
 try:
  for stage in ['acquire','gpu_audit','score','audit']:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'if stage in ['acquire','gpu_audit']else'',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');log=D/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
   with log.open('x')as f:
    child=subprocess.Popen([sys.executable,'-B',str(S/'calibration_runtime_v1.py'),stage],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    st=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split();atomic(D/'children'/(stage+'.json'),dict(pid=child.pid,startticks=int(st[19]),stage=stage,argv=[sys.executable,'-B',str(S/'calibration_runtime_v1.py'),stage]));atomic(D/'state.json',dict(status='running',stage=stage,child=child.pid,controller=identity()));code=child.wait()
   e=dict(stage=stage,exit_code=code,actual_subprocess_wait=True,elapsed_wall_seconds=time.perf_counter()-tick,log_sha256=sha(log));atomic(D/'exits'/(stage+'.json'),e);exits[stage]=e;assert code==0,(stage,code)
  atomic(D/'controller_complete.json',dict(status='all_source_calibration_predict_GPUreplay_officialscore_independentCPUaudit_complete_PENDING_ROOT_source_acceptance',exits=exits,worker=identity(),target_released=False));atomic(D/'state.json',dict(status='complete_pending_ROOT',all_actual_child_wait0=True,target_released=False));print('Source calibration complete pending ROOT, no target launch',flush=True)
 except BaseException:atomic(D/'controller_failure.json',dict(error=traceback.format_exc(),exits=exits,worker=identity(),all_paid_state_preserved=True));raise
if __name__=='__main__':main()
