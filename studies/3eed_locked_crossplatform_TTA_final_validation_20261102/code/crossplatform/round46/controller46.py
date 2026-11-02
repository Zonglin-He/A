"""Sequential prediction-only controller. No task scorer or automatic GT access."""
from pathlib import Path
import sys,subprocess,time
from common46 import context,args,dump,seal,sha,ROOT,read

STAGES=[('source_heads46.py','grounder_python'),('oldg46.py:parse','grounder_python'),('oldg46.py:prepare','grounder_python'),('oldg46.py:detect','vlm_python'),('oldg46.py:evidence','vlm_python'),('oldg46.py:bank','grounder_python'),('assemble46.py','grounder_python'),('external46.py','grounder_python')]
def main():
 a=args(__doc__);plan,rows,out=context(a);assert not out.exists(),'Preserve prior attempts; authorize a fresh output path'
 out.mkdir();dump(out/'RUN_BINDING.json',{'plan_sha256':sha(a.plan),'authorization_sha256':sha(a.authorization),'manifest_sha256':plan['manifest_sha256'],'n':len(rows),'root_primary':'R30_beta1','secondary':'R24_query_mlp_lr1_s16','no_scoring_in_controller':True})
 start=time.perf_counter();trace=[]
 stages=STAGES.copy()
 if plan.get('fresh_frames'):stages.insert(1,('frames46.py','grounder_python'))
 for spec,interpreter in stages:
  script,_,stage=spec.partition(':');command=[plan[interpreter],str(ROOT/'crossplatform/round46'/script),'--plan',str(Path(a.plan).resolve()),'--authorization',str(Path(a.authorization).resolve()),'--output',str(out)]
  if stage:command+=['--stage',stage]
  t=time.perf_counter();result=subprocess.run(command,cwd=ROOT,timeout=plan['stage_timeouts_seconds'][stage or script]);trace.append({'stage':spec,'returncode':result.returncode,'seconds':time.perf_counter()-t});dump(out/'STAGES.json',trace)
  if result.returncode:raise SystemExit('Stage failed; preserve attempt, no scoring or automatic retry: '+spec)
 from common46 import verify
 for name in ('source','oldG/textM64_attempt001','oldG/preparedM64_attempt001','oldG/dinoM64_attempt001','oldG/evidenceM64_attempt001','oldG/bankM64_attempt001','main','external'):verify(out/name)
 if plan.get('fresh_frames'):verify(out/'frames')
 seal(out,'R46_ALL_LOCKED_PREDICTIONS_AND_CONTROLS_SEALED_BEFORE_TASKSCORE',n=len(rows),main_arms=12,primary='R30_beta1',secondary='R24_query_mlp_lr1_s16',main_protocol='A208adapt_to_frozen_final',external_protocol='episodic_current_input_nativepre_and_cleanpost',no_task_GT=True,plan_sha256=sha(a.plan),authorization_sha256=sha(a.authorization),seconds=time.perf_counter()-start)
 print('R46_ALL_PREDICTIONS_SEALED',len(rows),flush=True)
if __name__=='__main__':main()
