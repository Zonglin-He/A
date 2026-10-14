"""Join two previously audited public stages with finite-scope closing receipts."""
import json,hashlib,shutil,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];M=R/'artifacts/stvg_matched_support_reference_v1';B=R/'artifacts/stvg_local_motion_regions_v1';S=B/'joint_public_stage'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not S.exists();S.mkdir();paths={}
 def add(p,dest):
  raw=p.read_bytes()
  if dest in paths:assert (S/dest).read_bytes()==raw;return
  q=S/dest;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);paths[dest]=dict(path=dest,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),blob_sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),mode='100644',encoding='base64'if q.suffix.lower()in('.png','.pdf')else'utf-8')
 for base in (M,B):
  m=json.loads((base/'PUBLIC_MANIFEST.json').read_text());rows=m if isinstance(m,list)else m['files']
  for row in rows:
   q=base/'public_stage'/row['path'];assert sha(q)==row['sha256'];add(q,row['path'])
  for name in ['ROOT_CLOSING','FINAL_COMPLETION','ARCHIVE_CLOSING_SNAPSHOT_RECEIPT','TECHNICAL_DECISION']:
   p=base/(name+'.json');assert p.exists();d=json.loads(p.read_text());assert d.get('status')in('pass','complete','stop_fixed_local_motion_region_recipe_before_R1_OPD','execute_one_preplanned_local_motion_region_coverage_test');add(p,f'results/{base.name.removesuffix("_v1")}/2026-10-14/closing/{name}.json')
 add(M/'INITIAL_FINAL_GITHUB_RECEIPT.json','results/stvg_matched_support_reference/2026-10-14/closing/INITIAL_FINAL_GITHUB_RECEIPT.json')
 for base,name in [(M,'PUBLIC_SCALAR_AUDIT002'),(B,'PUBLIC_SCALAR_AUDIT003')]:
  p=base/(name+'.json');d=json.loads(p.read_text());assert d['status']=='pass';add(p,f'results/{base.name.removesuffix("_v1")}/2026-10-14/closing/{name}.json')
 add(B/'INITIAL_FINAL_GITHUB_RECEIPT.json','results/stvg_local_motion_regions/2026-10-14/closing/INITIAL_FINAL_GITHUB_RECEIPT.json')
 for name in ['ROOT_ANONYMOUS_SCALAR_READBACK003','ROOT_READBACK']:
  add(B/(name+'.json'),f'results/stvg_local_motion_regions/2026-10-14/closing/{name}.json')
 for name in ['scripts/stvg_verified_public_transfer_v1.py','scripts/prepare_stvg_spatial_reference_closing_v1.py']:add(R/name,name)
 # Scientific source enums and relative cache paths are public; personal absolute paths are not.
 for p in S.rglob('*'):
  if p.is_file()and p.suffix in('.py','.md','.json'):
   b=p.read_bytes();assert not re.search(rb'/(?:home|tmp)/[A-Za-z0-9_.-]+/',b),str(p)
 m=sorted(paths.values(),key=lambda x:x['path']);q=B/'JOINT_PUBLIC_MANIFEST.json';assert not q.exists();q.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(files=len(m),bytes=sum(x['bytes']for x in m),manifest_sha256=sha(q))))
if __name__=='__main__':main()
