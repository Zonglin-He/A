"""Finite dependent audit/scoring of this one confirmed active prediction job."""
from pathlib import Path
import json,time,datetime,subprocess,os
R=Path(__file__).resolve().parents[2];H=R/'research/round120';N=R/'research/EXPLORATION_20261115';deadline=datetime.datetime.fromisoformat(json.loads((N/'STATE.json').read_text())['deadline_utc'])
while not (R/'crossplatform/round120/MATRIX_SEALED.json').exists():
 assert datetime.datetime.now(datetime.timezone.utc)<deadline
 assert (N/'RUNNING_GPU.json').exists(),'GPU producer stopped before matrix seal; inspect original handle, never restart automatically'
 record=json.loads((N/'RUNNING_GPU.json').read_text());os.kill(record['pid'],0);time.sleep(10)
for s in ['audit120.py','score120.py','audit_scores120.py']:
 with (H/s.replace('.py','.log')).open('w') as log:subprocess.run([str(R/'crossplatform/env/bin/python'),str(R/'tools/exploration_guard_round120.py'),'--resource','cpu','--timeout-seconds','7200','--',str(R/'crossplatform/env/bin/python'),str(H/s)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
print('POST120_COMPLETE',flush=True)
