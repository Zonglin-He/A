"""Finite continuation after separately resumed native audit, no duplicate models/audit."""
from pathlib import Path
import subprocess,time,sys
R=Path(__file__).resolve().parents[2];H=R/'research/round112';end=time.monotonic()+3600
while not (H/'NATIVE_AUDIT.json').exists():
 assert time.monotonic()<end;time.sleep(5)
for n in ['score112','report112']:
 with (H/(n+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/'tools/exploration_guard_round112.py'),'--resource','cpu','--timeout-seconds','3600','--',sys.executable,str(H/(n+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('POSTSCORE112',n,'complete',flush=True)
