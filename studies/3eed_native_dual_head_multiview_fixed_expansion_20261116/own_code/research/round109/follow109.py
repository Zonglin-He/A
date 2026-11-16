"""Finite follow-through of this exact authorized run, never a recurring automation."""
from pathlib import Path
import json,os,sys,time,subprocess
R=Path(__file__).resolve().parents[2];H=R/'research/round109';regpid=2384979
while not (R/'crossplatform/round109/registration_PRIVATE/SEALED.json').exists():
 try:os.kill(regpid,0)
 except ProcessLookupError:raise RuntimeError('Registration process terminal withoutseal; inspect attempt receipt, do not restart blindly')
 time.sleep(5)
# Wait for the registrar's guard to release its unique CPU resource after writing seal.
while (R/'research/EXPLORATION_20261115/RUNNING_CPU.json').exists():
 v=json.loads((R/'research/EXPLORATION_20261115/RUNNING_CPU.json').read_text())
 if v.get('pid')!=regpid:raise RuntimeError('Unexpected CPU resource owner; do not interfere')
 time.sleep(1)
for script,resource,timeout in [('lock_code109','cpu',600),('run109','gpu',10800),('audit109','cpu',3600),('score109','cpu',3600),('report109','cpu',1800)]:
 with (H/(script+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/'tools/exploration_guard_round109.py'),'--resource',resource,'--timeout-seconds',str(timeout),'--',sys.executable,str(H/(script+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('FOLLOW109',script,'complete',flush=True)
