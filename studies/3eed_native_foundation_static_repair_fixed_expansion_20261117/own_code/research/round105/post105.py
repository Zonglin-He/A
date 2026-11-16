"""Finite follow-through for this specific experiment; no scheduler or recurrence."""
from pathlib import Path
import sys,time,subprocess,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round105';seal=R/'crossplatform/round105/MATRIX_SEALED.json';end=time.monotonic()+7200
while not seal.exists():
 if time.monotonic()>end:raise TimeoutError('R105 prediction seal unavailable; no scoring')
 time.sleep(2)
for name in ['audit105','score105','report105']:
 cmd=[str(R/'crossplatform/env/bin/python'),str(R/'tools/exploration_guard_round105.py'),'--resource','cpu','--timeout-seconds','3600','--',str(R/'crossplatform/env/bin/python'),str(H/(name+'.py'))]
 with (H/(name+'.log')).open('w') as out:subprocess.run(cmd,cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
 print('POST105',name,'complete',flush=True)
