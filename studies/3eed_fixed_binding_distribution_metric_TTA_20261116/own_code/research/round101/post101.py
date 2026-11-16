"""Finite serial audit/score pipeline, every scientific child deadline guarded."""
from pathlib import Path
import subprocess,json,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round101';N=R/'research/EXPLORATION_20261115';PY=R/'crossplatform/env/bin/python'
def main():
 for resource,script in [('gpu','audit101.py'),('cpu','score101.py'),('cpu','report101.py')]:
  s=json.loads((N/'STATE.json').read_text());s.update(current_round='R101',current_stage=script,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(N/'STATE.json').write_text(json.dumps(s,indent=2)+'\n')
  with (H/(script+'.log')).open('a') as f:subprocess.run([str(PY),str(R/'tools/exploration_guard_round101.py'),'--resource',resource,'--timeout-seconds','3600','--',str(PY),str(H/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('POST101_COMPLETE',flush=True)
if __name__=='__main__':main()
