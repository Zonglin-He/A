"""One finite registered experiment, all model calls serial under deadline guard."""
from pathlib import Path
import subprocess,json,hashlib,datetime
R=Path(__file__).resolve().parents[2];H=R/'research/round99';B=R/'crossplatform/round99';N=R/'research/EXPLORATION_20261115';PY=R/'crossplatform/env/bin/python'
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stage(x):
 s=rd(N/'STATE.json');s.update(current_round='R99',current_stage=x,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(N/'STATE.json').write_text(json.dumps(s,indent=2)+'\n');print(x,flush=True)
def call(src,tgt,engineering=False):
 stage(('engineering_' if engineering else '')+src+'_to_'+tgt)
 with (H/(('engineering_' if engineering else '')+src+'_to_'+tgt+'.log')).open('a') as f:subprocess.run([str(PY),str(R/'tools/exploration_guard_20261115.py'),'--resource','gpu','--timeout-seconds','3600','--',str(PY),str(H/'run99.py'),'--source',src,'--target',tgt]+(['--engineering'] if engineering else []),cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 for src,tgt in [('drone','quad'),('quad','waymo')]:
  if not (B/('engineering_'+src+'_to_'+tgt)/'SEALED.json').exists():call(src,tgt,True)
 p=H/'experiment/CODE_LOCK.json'
 if not p.exists():
  files=set(H.glob('*.py'));files.add(R/'tools/exploration_guard_20261115.py');files.update(R/f for f in rd(R/'research/round98/experiment/CODE_LOCK.json')['code_sha256']);files.add(R/'research/round98/core98.py');p.write_text(json.dumps({'code_sha256':{str(f.relative_to(R)):sha(f) for f in sorted(files)},'design_sha256':sha(H/'experiment/DESIGN_LOCK.json'),'cohort_sha256':sha(H/'experiment/COHORT_LOCK.json')},indent=2)+'\n')
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src!=tgt and not (B/(src+'_to_'+tgt)/'SEALED.json').exists():call(src,tgt)
 stage('R99_ALL_SIX_PREDICTIONS_SEALED_WAIT_AUDIT')
if __name__=='__main__':main()
