from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round109';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert rd(H/'BRIDGE.json')['n']==6;assert (R/'crossplatform/round109/registration_PRIVATE/SEALED.json').is_file();p=H/'experiment/CODE_LOCK.json';assert not p.exists();files={k:sha(R/k) for k in rd(R/'research/round108/experiment/CODE_LOCK.json')['files']};files.update({str(f.relative_to(R)):sha(f) for f in H.glob('*.py')});files['tools/exploration_guard_round109.py']=sha(R/'tools/exploration_guard_round109.py');inputs=[H/'ANCHORS_PRIVATE.json',H/'PAIRS_PRIVATE.json',R/'crossplatform/round109/registration_PRIVATE/SEALED.json']+list((H/'input').glob('*.json'));p.write_text(json.dumps({'files':files,'dependency_sha256':rd(R/'research/round108/experiment/CODE_LOCK.json')['dependency_sha256'],'input_sha256':{str(f.relative_to(R)):sha(f) for f in inputs},'bridge_sha256':sha(H/'BRIDGE.json'),'input_lock_sha256':sha(H/'experiment/INPUT_LOCK.json')},indent=2)+'\n');print('LOCK109',len(files))
if __name__=='__main__':main()
