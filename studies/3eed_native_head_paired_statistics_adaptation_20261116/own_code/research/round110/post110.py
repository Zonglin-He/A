"""Finite follow-through of the current sealed experiment only."""
from pathlib import Path
import time,subprocess,sys
R=Path(__file__).resolve().parents[2];H=R/'research/round110';end=time.monotonic()+3600
while not (H/'NATIVE_AUDIT.json').exists():
 assert time.monotonic()<end;time.sleep(5)
for n in ['score110','report110']:
 with (H/(n+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/'tools/exploration_guard_round110.py'),'--resource','cpu','--timeout-seconds','3600','--',sys.executable,str(H/(n+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('POST110',n,'complete',flush=True)
