from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round108';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'ENGINEERING.json')['n']==12;p=H/'experiment/CODE_LOCK.json';assert not p.exists();files={k:sha(R/k) for k in rd(R/'research/round107/experiment/CODE_LOCK.json')['files']};files.update({str(f.relative_to(R)):sha(f) for f in H.glob('*.py')});files['tools/exploration_guard_round108.py']=sha(R/'tools/exploration_guard_round108.py');p.write_text(json.dumps({'files':files,'dependency_sha256':{'crossplatform/round3/p0/upstream/models/modules.py':sha(R/'crossplatform/round3/p0/upstream/models/modules.py')},'engineering_sha256':sha(H/'ENGINEERING.json'),'design_sha256':sha(H/'experiment/DESIGN_LOCK.json')},indent=2)+'\n');print('LOCK108',len(files))
if __name__=='__main__':main()
