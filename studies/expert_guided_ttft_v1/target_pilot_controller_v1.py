"""One target-pilot owner; no target tuning or automatic full expansion."""
from pathlib import Path
import os,sys,subprocess,time,traceback
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;sys.path[:0]=[str(S),str(R),str(R/'tools')]
import target_pilot_v1 as t
def main():
 p=t.check();d=t.D;assert not(d/'controller.json').exists();t.atomic(d/'controller.json',dict(**t.identity(),plan_fingerprint=p['fingerprint']));done={}
 try:
  for stage,ds in [('GPU',None),('GPU_audit',None)]+[(s,ds)for ds in t.c.TARGETS for s in ['score','audit']]:
   key=stage+('_'+ds if ds else'');argv=[sys.executable,'-B',str(S/'target_pilot_v1.py'),stage]+(['--dataset',ds]if ds else[]);env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'if stage in ['GPU','GPU_audit']else'',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');log=d/'logs'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
   with log.open('xb')as f:
    child=subprocess.Popen(argv,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);stat=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split();who=dict(pid=child.pid,startticks=int(stat[19]));t.atomic(d/'children'/(key+'.json'),dict(**who,argv=argv));t.atomic(d/'state.json',dict(status='running',stage=stage,dataset=ds,child=who,controller=t.identity()));code=child.wait()
   e=dict(actual_subprocess_wait=True,exit_code=code,worker=who,elapsed_wall_seconds=time.perf_counter()-start,log_sha256=t.sha(log));t.atomic(d/'exits'/(key+'.json'),e);done[key]=e;assert code==0,(key,code)
  t.atomic(d/'controller_complete.json',dict(status='all256_TTFT_pilot_savedstate_officialscore_independentaudit_complete_PENDING_ROOT',exits=done,controller=t.identity(),target_config_changed=False,full_target_started=False));t.atomic(d/'state.json',dict(status='complete_pending_ROOT',all_actual_child_wait0=True,full_target_started=False));print('Target TTFT pilot256 complete pending ROOT',flush=True)
 except BaseException:t.atomic(d/'controller_failure.json',dict(error=traceback.format_exc(),exits=done,controller=t.identity(),all_paidstate_preserved=True));raise
if __name__=='__main__':main()
