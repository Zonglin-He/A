from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round110';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'ENGINEERING.json')['n']==12;p=H/'experiment/CODE_LOCK.json';assert not p.exists();stage=rd(H/'experiment/STAGE0_CODE_LOCK.json');assert all(sha(R/k)==v for k,v in stage['files'].items());files={k:sha(R/k) for k in stage['files']};files.update({str(f.relative_to(R)):sha(f) for f in H.glob('*.py')});p.write_text(json.dumps({'files':files,'upstream_dependency_sha256':stage['upstream_dependency_sha256'],'original_extraction_sha256':sha(R/'crossplatform/round110/original_PRIVATE/SEALED.json'),'wrong_anchor_sha256':sha(H/'WRONG_ANCHOR_PRIVATE.json'),'engineering_sha256':sha(H/'ENGINEERING.json'),'design_sha256':sha(H/'experiment/DESIGN_LOCK.json')},indent=2)+'\n');print('LOCK110',len(files))
if __name__=='__main__':main()
