"""Independent full scalar-table correspondence and private-literal audit."""
from pathlib import Path
import json,csv,hashlib,re
R=Path(__file__).resolve().parents[2];H=R/'research/round129';O=H/'publication/public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 private=set()
 for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json'):
  for k in ['id','caption','scene','pointpath','metapath','imagepath']:
   v=e['row'].get(k)
   if isinstance(v,str) and len(v)>=12:private.add(v)
 for p in rd(H/'SOURCE_CALIBRATION.json')['source_files']:private.add(Path(p).stem)
 findings=[];n=0;files=[]
 for f in O.rglob('*'):
  if not f.is_file():continue
  assert f.suffix in ['.json','.py','.csv','.md','.log'];s=f.read_text();files.append(f)
  for t in private:
   if t in s:findings.append((str(f.relative_to(O)),'private_literal'))
  if 'LOCAL_HOME/' in s:findings.append((str(f.relative_to(O)),'local_home'))
  if re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',s):findings.append((str(f.relative_to(O)),'credential_pattern'))
 for f in (O/'tasks').glob('*.csv'):
  pub=list(csv.DictReader(f.open()));rows=rd(H/'scores_attempt001'/(f.stem+'_ALL_ROWS_PRIVATE.json'));assert len(pub)==len(rows)
  for i,(p,r) in enumerate(zip(pub,rows)):
   for k in ['cohort','source','target']:assert p[k]==r[k]
   assert p['row']==f'ROW{i+1:04d}' and len([k for k in p if k.endswith('__IoU')])==25
   for k,x in r['arms'].items():assert abs(float(p[k+'__IoU'])-x['IoU'])<1e-14
   n+=1
 for x in rd(O/'MANIFEST.json')['files']:
  p=O/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 assert n==2304 and not findings,findings
 out={'status':'PASS_COMPLETE_ANONYMOUS_SCALARS_PRIVATE_LITERAL_SCAN','files':len(files),'task_rows':n,'arms':25,'findings':len(findings),'raw_media_coordinates_ids_features_weights_or_third_party_exported':False};(H/'PUBLIC_EXPORT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
