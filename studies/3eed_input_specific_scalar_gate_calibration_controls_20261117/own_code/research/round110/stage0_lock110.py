from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round110';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=H/'experiment/STAGE0_CODE_LOCK.json';assert not p.exists();old=rd(R/'research/round108/experiment/CODE_LOCK.json')['files'];files={k:sha(R/k) for k in old};files.update({str(f.relative_to(R)):sha(f) for f in H.glob('*.py')});files['tools/exploration_guard_round110.py']=sha(R/'tools/exploration_guard_round110.py');p.write_text(json.dumps({'files':files,'design_sha256':sha(H/'experiment/DESIGN_LOCK.json'),'upstream_dependency_sha256':rd(R/'research/round108/experiment/CODE_LOCK.json')['dependency_sha256']},indent=2)+'\n');print('STAGE0_LOCK110',len(files))
