"""Wait for sealed GPU completion, then execute each CPU audit/scoring stage guarded."""
from pathlib import Path
import time,subprocess,json
R=Path(__file__).resolve().parents[2];H=R/'research/round134';B=R/'crossplatform/round134'
def main():
 end=time.monotonic()+10800
 while not (B/'MATRIX_SEALED.json').exists() or (H/'RUNNING_GPU.json').exists():
  if time.monotonic()>end:raise RuntimeError('bounded matrix wait expired')
  if not (H/'RUNNING_GPU.json').exists() and not (B/'MATRIX_SEALED.json').exists():
   receipts=sorted((H/'job_receipts').glob('*.json'))
   if receipts and any(json.loads(p.read_text()).get('returncode') not in [0,None] for p in receipts):raise RuntimeError('model attempt failed; retain evidence, no scoring')
  time.sleep(5)
 for script in ['policy134','audit134','score134','audit_scalars134','analyze134','report134','export134','audit_public134']:
  print('BEGIN',script,flush=True)
  with (H/(script+'_attempt001.log')).open('w') as log:
   result=subprocess.run(['python','tools/experiment_guard_round134.py','--resource','cpu','--timeout-seconds','3600','--','crossplatform/env/bin/python',str(H/(script+'.py'))],cwd=R,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(script+' failed; inspect retained log')
  print('COMPLETE',script,flush=True)
 print('R134_EVALUATION_ALL_COMPLETE',flush=True)
if __name__=='__main__':main()
