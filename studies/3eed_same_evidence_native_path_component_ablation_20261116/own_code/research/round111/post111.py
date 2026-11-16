"""Finite follow-through of exactly the present fixed experiment, not a standing job."""
from pathlib import Path
import subprocess,time,sys
R=Path(__file__).resolve().parents[2];H=R/'research/round111';end=time.monotonic()+7200
while not (R/'crossplatform/round111/MATRIX_SEALED.json').exists():
 assert time.monotonic()<end;time.sleep(5)
for n in ['audit111','score111','report111']:
 with (H/(n+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/'tools/exploration_guard_round111.py'),'--resource','cpu','--timeout-seconds','3600','--',sys.executable,str(H/(n+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('POST111',n,'complete',flush=True)
