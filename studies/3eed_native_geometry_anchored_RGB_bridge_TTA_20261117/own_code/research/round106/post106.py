"""Finite follow-through of this exact run, no automation or recurrence."""
from pathlib import Path
import time,sys,subprocess,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round106';end=time.monotonic()+10800
while not (R/'crossplatform/round106/MATRIX_SEALED.json').exists():
 assert time.monotonic()<end,'finite prediction-seal wait timed out'
 time.sleep(5)
for name in ['audit106','score106','report106']:
 with (H/(name+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/'tools/exploration_guard_round106.py'),'--resource','cpu','--timeout-seconds','3600','--',sys.executable,str(H/(name+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('POST106',name,'complete',flush=True)
