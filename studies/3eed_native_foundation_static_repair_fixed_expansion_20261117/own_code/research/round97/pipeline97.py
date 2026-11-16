"""Finite authorized six-direction experiment; no timer or future experiment generation."""
from pathlib import Path
import json,subprocess,sys,hashlib,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round97';B=R/'crossplatform/round97';PY=R/'crossplatform/env/bin/python'
def read(p):return json.loads(p.read_text())
def run(script,args,log,resource='gpu',timeout=7200):
 with (H/log).open('a') as f:
  c=subprocess.run([str(PY),str(R/'tools/manual_r97_guard.py'),'--resource',resource,'--timeout-seconds',str(timeout),'--',str(PY),str(H/script),*args],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert c.returncode==0,(script,args,c.returncode,log)
def stage(s):
 p=H/'STATE.json';v=read(p);v.update(state='R97_RUNNING_FIXED_R91_SIX_DIRECTION_TTA',current_stage=s,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(v,indent=2)+'\n');print('PIPELINE97',s,flush=True)
def main():
 assert read(H/'ENGINEERING_R91_PARITY_AUDIT.json')['state'].startswith('PASS_')
 for f,s in read(H/'experiment/SOURCE_CODE_LOCK.json')['files_sha256'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==s,f
 for pl in ['drone','quad']:
  if not (B/pl/'SOURCE_COMPLETE.json').exists():stage('source_'+pl+'_preparation');run('prepare_source97.py',['--source',pl],'source_'+pl+'.log')
 assert (B/'ROLE_PARSE_COMPLETE.json').exists(),'all caption roles must finish before target pipeline'
 if not (H/'experiment/TARGET_CODE_LOCK.json').exists():subprocess.run([str(PY),str(H/'lock97.py')],cwd=R,check=True)
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   if not (B/(src+'_to_'+tgt)/'SEALED.json').exists():stage('target_'+src+'_to_'+tgt);run('run97.py',['--source',src,'--target',tgt],src+'_to_'+tgt+'.log',timeout=14400)
 stage('all_six_prediction_sealed_await_independent_audit');print('PIPELINE97_PREDICTIONS_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  v=read(H/'STATE.json');v.update(state='R97_EXECUTION_FAILED_ARTIFACTS_RETAINED',failure_type=type(e).__name__,failure_message=str(e));(H/'STATE.json').write_text(json.dumps(v,indent=2)+'\n');raise
