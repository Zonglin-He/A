"""One owned CPU job after the actual source GPU exit; no target release."""
from source_calibration import *
import subprocess,traceback
def prepare_execution():
 assert not(K/'execution_plan.json').exists();plan=check_calibration()
 p=dict(schema='source_calibration_execution_v1',calibration_fingerprint=plan['fingerprint'],code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'source_calibration.py']},max_CPU_heavy=1,GPU=False,source_exit_gate='require real acquire_actual_exit.json actualwait0, all72seal and original worker PID/start not live; no inferred wait code',stages=['controls','score','audit'],target_release=False)
 p['fingerprint']=digest(p);atomic(K/'execution_plan.json',p);atomic(P/'source_calibration_execution_protocol.json',p);print(p['fingerprint'],flush=True)
def live(pid,start):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[21]==str(start)
 except FileNotFoundError:return False
def queue():
 plan=read(K/'execution_plan.json');assert plan['fingerprint']==digest({k:v for k,v in plan.items()if k!='fingerprint'})
 for f,h in plan['code_sha256'].items():assert sha(R/f)==h,f
 assert not(K/'controller.json').exists();atomic(K/'controller.json',identity());start=time.perf_counter()
 try:
  while not(C/'acquire_actual_exit.json').exists():parent.resources(start);atomic(K/'controller_state.json',dict(status='waiting_actual_source_GPU_tool_wait0_and_all72seals',controller=identity(),target_release=False));time.sleep(15)
  ex=read(C/'acquire_actual_exit.json');assert ex['actual_tool_wait_exit_code']==0 and ex['actual_wait_returned'];w=read(C/'worker.json');assert not live(w['pid'],w['startticks']);seals()
  for stage in plan['stages']:
   parent.resources(start);out=K/(stage+'.log');cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/'source_calibration.py'),stage];tick=time.perf_counter()
   with out.open('wb')as log:
    child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});cid=dict(pid=child.pid,startticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]));atomic(K/'controller_state.json',dict(status='running_'+stage,controller=identity(),child=cid,argv=cmd,target_release=False));code=child.wait()
   atomic(K/(stage+'_exit.json'),dict(stage=stage,worker=cid,exit_code=code,actual_subprocess_wait=True,elapsed_wall_seconds=time.perf_counter()-tick));assert code==0,(stage,code)
  atomic(K/'controller_state.json',dict(status='source_validation_scored_audited_PENDING_ROOT_candidate_acceptance',controller=identity(),target_release=False,whole_six_complete=False))
 except BaseException:atomic(K/'controller_failure.json',dict(error=traceback.format_exc(),controller=identity(),target_release=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare_execution','queue']);x=ap.parse_args();globals()[x.stage]()
