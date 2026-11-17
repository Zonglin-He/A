"""Prepare verified prior-blob provenance for public-only files; no network mutation."""
from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parents[2];H=R/'research/round127';P=H/'publication';O=P/'public_export_attempt001'
rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'independent_public_audit_attempt001/AUDIT_RESULT.json')['findings']==0
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304
 prior={}
 for old in sorted((R/'research').glob('round*/publication/UPLOAD_INDEX_PRIVATE.json')):
  if old.parent==P:continue
  receipt=old.parent/'PUBLICATION_RECEIPT.json'
  if not receipt.exists():continue
  for i,f in enumerate(rd(old).get('files',[])):
   if 'git_blob_sha1' not in f:continue
   proof=old.parent/f'REMOTE_VERIFY_{i}.txt'
   if proof.exists():prior.setdefault(f['git_blob_sha1'],[]).append((proof,rd(receipt)['commit']))
 files=[];provenance=[]
 for i,f in enumerate(sorted(x for x in O.rglob('*') if x.is_file())):
  b=f.read_bytes();sha256=hashlib.sha256(b).hexdigest();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
  files.append({'path':str(f.relative_to(O)),'local_path':str(f.relative_to(R)),'bytes':len(b),'sha256':sha256,'git_blob_sha1':blob})
  saved=None
  for proof,commit in prior.get(blob,[]):
   if proof.read_bytes()==b:
    shutil.copyfile(proof,P/f'REMOTE_VERIFY_{i}.txt');saved={'path':str(proof.relative_to(R)),'commit':commit,'sha256':sha256};break
  provenance.append({'index':i,'renewed':saved is not None,'prior':saved})
 (P/'UPLOAD_INDEX_PRIVATE.json').write_text(json.dumps({'prefix':'studies/3eed_deployment_consistent_complete_readout_TTA_20261117','files':files},indent=2)+'\n')
 (P/'REMOTE_PROVENANCE_PRIVATE.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print({'files':len(files),'bytes':sum(f['bytes'] for f in files),'renewed_proofs':sum(p['renewed'] for p in provenance)},flush=True)
if __name__=='__main__':main()
