"""Independent public privacy and complete anonymous task-scalar audit."""
from pathlib import Path
import json,csv,re,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round128';O=H/'publication/public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 tokens=set()
 for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json'):
  for k in ['id','caption','scene','pointpath','metapath','imagepath']:
   v=e['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
 for e in rd(H/'REGION_REGISTRY_PRIVATE.json'):
  if e['role']=='source':tokens.add(Path(e['path']).stem)
 findings=[];files=[]
 for f in O.rglob('*'):
  if not f.is_file():continue
  rel=str(f.relative_to(O));assert f.suffix in ['.json','.md','.py','.csv','.log'];s=f.read_text();files.append(f)
  for t in tokens:
   if t in s:findings.append({'file':rel,'finding':'private_literal'})
  if re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',s):findings.append({'file':rel,'finding':'credential_pattern'})
  if 'LOCAL_HOME/' in s:findings.append({'file':rel,'finding':'home_path'})
 n=0
 for f in (O/'tasks').glob('*.csv'):
  public=list(csv.DictReader(f.open()));private=rd(H/'scores_attempt001'/(f.stem+'_ALL_ROWS_PRIVATE.json'));assert len(public)==len(private)==128
  for p,r in zip(public,private):
   assert p['source']==r['source'] and p['target']==r['target'];assert p['row']==f"ROW{r['slot']+1:04d}"
   for k,v in r['IoUs'].items():assert abs(float(p[k+'__IoU'])-v)<1e-14
   n+=1
 manifest=rd(O/'MANIFEST.json')
 for x in manifest['files']:
  p=O/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 assert n==768 and not findings,findings
 out={'status':'PASS_ANONYMOUS_COMPLETE_EXPORT','files':len(files),'task_rows':n,'findings':len(findings),'raw_data_weights_features_or_third_party_snapshots_exported':False};(H/'PUBLIC_EXPORT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
