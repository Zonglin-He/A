"""Expand unchanged R99 once; reuse byte-identical DEV predictions, exclusive GPU."""
from pathlib import Path
import json,hashlib,subprocess,os,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round100';B=R/'crossplatform/round100';OLD=R/'crossplatform/round99';N=R/'research/EXPLORATION_20261115';PY=R/'crossplatform/env/bin/python';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stage(v):
 s=rd(N/'STATE.json');s.update(current_round='R100',current_stage=v,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(N/'STATE.json').write_text(json.dumps(s,indent=2)+'\n');print('PIPELINE100',v,flush=True)
def main():
 p=H/'experiment/CODE_LOCK.json'
 if not p.exists():
  files=set(H.glob('*.py'));files.add(R/'tools/exploration_guard_round100.py');files.update(R/f for f in rd(R/'research/round99/experiment/CODE_LOCK.json')['code_sha256']);files.add(R/'research/round99/core99.py');p.write_text(json.dumps({'code_sha256':{str(f.relative_to(R)):sha(f) for f in sorted(files)},'design_sha256':sha(H/'experiment/DESIGN_LOCK.json')},indent=2)+'\n')
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;O=B/name
   if (O/'SEALED.json').exists():continue
   O.mkdir(exist_ok=True);oldseal=rd(OLD/name/'SEALED.json');linked=0
   for path,h in oldseal['files'].items():
    f=OLD/name/path;assert sha(f)==h;new=O/path;new.parent.mkdir(parents=True,exist_ok=True)
    if not new.exists():os.link(f,new)
    assert sha(new)==h;linked+=1
   (H/(name+'_REUSE_PROOF.json')).write_text(json.dumps({'old_seal_sha256':sha(OLD/name/'SEALED.json'),'reused_inputs':128,'verified_hardlinked_files':linked},indent=2)+'\n');stage(name)
   with (H/(name+'.log')).open('a') as f:subprocess.run([str(PY),str(R/'tools/exploration_guard_round100.py'),'--resource','gpu','--timeout-seconds','10800','--',str(PY),str(H/'run100.py'),'--source',src,'--target',tgt],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 stage('R100_ALL_SIX_SEALED_AWAIT_AUDIT_SCORING')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  s=rd(N/'STATE.json');s.update(status='execution_failure_retained',failure_type=type(e).__name__,failure_message=str(e));(N/'STATE.json').write_text(json.dumps(s,indent=2)+'\n');raise
