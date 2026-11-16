"""Finite complete-user-method benchmark, serial GPU, resume sealed directions only."""
from pathlib import Path
import json,subprocess,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round98';B=R/'crossplatform/round98';PY=R/'crossplatform/env/bin/python'
def stage(v):
 p=H/'STATE.json';s=json.loads(p.read_text());s.update(state='R98_RUNNING_USER_CONFIRMED_COMPLETE_R95',current_stage=v,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(s,indent=2)+'\n');print('PIPELINE98',v,flush=True)
def main():
 if not (H/'experiment/CODE_LOCK.json').exists():subprocess.run([str(PY),str(H/'lock98.py')],cwd=R,check=True)
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   if (B/(src+'_to_'+tgt)/'SEALED.json').exists():continue
   stage(src+'_to_'+tgt)
   with (H/(src+'_to_'+tgt+'.log')).open('a') as f:ret=subprocess.run([str(PY),str(R/'tools/manual_r98_guard.py'),'--resource','gpu','--timeout-seconds','14400','--',str(PY),str(H/'run98.py'),'--source',src,'--target',tgt],cwd=R,stdout=f,stderr=subprocess.STDOUT)
   assert ret.returncode==0,(src,tgt,ret.returncode)
 stage('all_six_R95_predictions_sealed_await_native_audit');print('PIPELINE98_PREDICTIONS_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  p=H/'STATE.json';s=json.loads(p.read_text());s.update(state='R98_EXECUTION_FAILED_ARTIFACTS_RETAINED',failure_type=type(e).__name__,failure_message=str(e));p.write_text(json.dumps(s,indent=2)+'\n');raise
