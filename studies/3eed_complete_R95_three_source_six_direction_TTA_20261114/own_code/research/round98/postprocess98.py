"""Finite audited postprocessing; never starts unless all six prediction seals exist."""
from pathlib import Path
import json,subprocess,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round98';B=R/'crossplatform/round98';PY=R/'crossplatform/env/bin/python'
def stage(name):
 p=H/'STATE.json';s=json.loads(p.read_text());s.update(current_stage=name,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(s,indent=2)+'\n');print('POST98',name,flush=True)
def run(name,resource,timeout):
 stage(name)
 with (H/(name+'.log')).open('a') as f:subprocess.run([str(PY),str(R/'tools/manual_r98_guard.py'),'--resource',resource,'--timeout-seconds',str(timeout),'--',str(PY),str(H/(name+'.py'))],cwd=R,check=True,stdout=f,stderr=subprocess.STDOUT)
def main():
 assert all((B/(s+'_to_'+t)/'SEALED.json').exists() for s in ['waymo','drone','quad'] for t in ['waymo','drone','quad'] if s!=t)
 if not (H/'NATIVE_AUDIT.json').exists():run('audit98','gpu',14400)
 if not (H/'scores_attempt001/SEALED.json').exists():run('score98','cpu',14400)
 if not (H/'SCALAR_AUDIT.json').exists():run('audit_scalar98','cpu',1800)
 run('report98','cpu',1800);stage('all_six_complete_R95_scored_audited_await_publication')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  p=H/'STATE.json';s=json.loads(p.read_text());s.update(state='R98_POSTPROCESS_FAILED_ARTIFACTS_RETAINED',failure_type=type(e).__name__,failure_message=str(e));p.write_text(json.dumps(s,indent=2)+'\n');raise
