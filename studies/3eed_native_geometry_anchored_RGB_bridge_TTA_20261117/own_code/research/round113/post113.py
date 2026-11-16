"""Finite controller for this already-running experiment only; all stages guarded."""
from pathlib import Path
import time,sys,subprocess,json,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round113';end=time.monotonic()+14400
while not (R/'crossplatform/round113/MATRIX_SEALED.json').exists():
 state=json.loads((R/'research/EXPLORATION_20261115/STATE.json').read_text());assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(state['deadline_utc']);assert time.monotonic()<end;time.sleep(5)
for name in ['audit113','score113','audit_scores113','report113']:
 with (H/(name+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/'tools/exploration_guard_round113.py'),'--resource','cpu','--timeout-seconds','3600','--',sys.executable,str(H/(name+'.py'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('POST113',name,'complete',flush=True)
