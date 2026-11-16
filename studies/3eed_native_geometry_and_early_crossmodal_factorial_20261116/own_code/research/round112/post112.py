"""Finite staged completion of this sealed experiment only, deadline guarded."""
from pathlib import Path
import time,sys,subprocess
R=Path(__file__).resolve().parents[2];H=R/'research/round112';end=time.monotonic()+7200
while not (R/'crossplatform/round112/MATRIX_SEALED.json').exists():
 assert time.monotonic()<end;time.sleep(5)
for n in ['audit112','score112','report112']:
 with (H/(n+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/'tools/exploration_guard_round112.py'),'--resource','cpu','--timeout-seconds','3600','--',sys.executable,str(H/(n+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('POST112',n,'complete',flush=True)
