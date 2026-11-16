from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round107';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert rd(H/'ENGINEERING.json')['n']==12;p=H/'experiment/CODE_LOCK.json';assert not p.exists();old=rd(R/'research/round106/experiment/CODE_LOCK.json')['files'];files={k:sha(R/k) for k in old};files.update({str(p.relative_to(R)):sha(p) for p in (R/'research/round106').glob('*.py')});files.update({str(p.relative_to(R)):sha(p) for p in H.glob('*.py')});files['tools/exploration_guard_round107.py']=sha(R/'tools/exploration_guard_round107.py');p.write_text(json.dumps({'files':files,'engineering_sha256':sha(H/'ENGINEERING.json'),'design_sha256':sha(H/'experiment/DESIGN_LOCK.json'),'old_matrix_sha256':sha(R/'crossplatform/round106/MATRIX_SEALED.json')},indent=2)+'\n');print('CODE107_LOCKED',len(files))
if __name__=='__main__':main()
