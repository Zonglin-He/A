"""Administrative continuation only; do not invent/retry experiments on errors."""
from pathlib import Path
import json,time,os,subprocess,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round128'
def state(stage,**kwargs):
 p=H/'STATE.json';s=json.loads(p.read_text());s.update(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs);p.write_text(json.dumps(s,indent=2)+'\n');print(stage,flush=True)
def main():
 info=json.load(open(H/'RUNNING_GPU.json')) if (H/'RUNNING_GPU.json').exists() else {};pid=info.get('pid',-1);state('waiting_running_CloudFixer_producer',producer_pid=pid)
 while (H/'RUNNING_GPU.json').exists():
  try:os.kill(pid,0)
  except ProcessLookupError:
   if not (H/'CLOUDFIXER_MATRIX_SEALED.json').exists():raise RuntimeError('CloudFixer producer ended without seal')
  time.sleep(10)
 assert (H/'CLOUDFIXER_MATRIX_SEALED.json').exists()
 for resource,name in [('gpu','run128_zero_controls'),('cpu','audit_score128')]:
  if name=='audit_score128':
   deadline=time.monotonic()+1800
   while not (H/'MMTTA_INDEPENDENT_MATRIX_SEALED.json').exists() or (H/'RUNNING_CPU.json').exists():
    assert time.monotonic()<deadline,'Independent modality repair did not seal within the finite wait'
    time.sleep(10)
  state(name);cmd=[str(R/'crossplatform/env/bin/python'),str(R/'tools/experiment_guard_round128.py'),'--resource',resource,'--timeout-seconds','7200','--',str(R/'crossplatform/env/bin/python'),str(H/(name+'.py'))]
  with (H/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(name+' failed; retain artifacts and inspect')
 state('all_predictions_sealed_and_initial_task_scoring_complete',measurement='awaiting_independent_scalar_audit')
if __name__=='__main__':
 try:main()
 except Exception as e:state('STOPPED_ON_FAILURE',status='failed',failure=str(e));raise
