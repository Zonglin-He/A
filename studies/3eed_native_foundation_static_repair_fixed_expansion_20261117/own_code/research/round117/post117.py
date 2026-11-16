"""Finite dependent audit/score queue for this already running prediction job."""
from pathlib import Path
import time,subprocess,datetime,json,os
R=Path(__file__).resolve().parents[2];H=R/'research/round117'
def main():
 deadline=datetime.datetime.fromisoformat(json.loads((R/'research/EXPLORATION_20261115/STATE.json').read_text())['deadline_utc'])
 while not (R/'crossplatform/round117/MATRIX_SEALED.json').exists():
  assert datetime.datetime.now(datetime.timezone.utc)<deadline;time.sleep(10)
 for s in ['audit117.py','score117.py','audit_scores117.py']:
  with (H/(s.replace('.py','.log'))).open('w') as log:subprocess.run(['python',str(R/'tools/exploration_guard_round117.py'),'--resource','cpu','--timeout-seconds','3600','--',str(R/'crossplatform/env/bin/python'),str(H/s)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('POST117_COMPLETE',flush=True)
if __name__=='__main__':main()
