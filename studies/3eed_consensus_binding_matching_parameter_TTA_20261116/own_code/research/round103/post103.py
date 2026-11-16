"""Wait only for the observed finite prediction PID, then guarded audit and scoring."""
from pathlib import Path
import json,os,time,subprocess,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round103';N=R/'research/EXPLORATION_20261115';PY=R/'crossplatform/env/bin/python';rd=lambda p:json.loads(Path(p).read_text())
def main():
 active=N/'RUNNING_GPU.json';job=rd(active);assert any(Path(x).resolve()==H/'run103.py' for x in job['command'][1:]) and '--engineering' not in job['command'];pid=job['pid'];os.kill(pid,0);(H/'OBSERVED_PREDICTION_PID.json').write_text(json.dumps(job,indent=2)+'\n');end=datetime.datetime.fromisoformat(rd(N/'STATE.json')['deadline_utc'])
 while active.exists() and rd(active).get('pid')==pid:
  assert datetime.datetime.now(datetime.timezone.utc)<end;time.sleep(2)
 receipts=[rd(f) for f in (N/'job_receipts').glob('*_'+str(pid)+'.json')];assert len(receipts)==1 and receipts[0]['returncode']==0;assert rd(R/'crossplatform/round103/MATRIX_SEALED.json')['n']==768
 for resource,script in [('gpu','audit103.py'),('cpu','score103.py'),('cpu','report103.py')]:
  state=rd(N/'STATE.json');state.update(current_round='R103',current_stage=script,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(N/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
  with (H/(script+'.log')).open('a') as f:subprocess.run([str(PY),str(R/'tools/exploration_guard_round103.py'),'--resource',resource,'--timeout-seconds','3600','--',str(PY),str(H/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
 state=rd(N/'STATE.json');state.update(current_stage='R103_DEV_COMPLETE_AWAIT_REVIEW_AND_PUBLICATION',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(N/'STATE.json').write_text(json.dumps(state,indent=2)+'\n');print('POST103_COMPLETE',flush=True)
if __name__=='__main__':main()
