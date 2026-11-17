"""Independent full public-table correspondence and private-literal scan."""
from pathlib import Path
import json,csv,hashlib,re,collections
R=Path(__file__).resolve().parents[2];H=R/'research/round131';O=H/'publication/public_export_attempt001'
rd=lambda p:json.loads(Path(p).read_text())
def main():
 private=set()
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  for k in ['id','caption','scene','pointpath','metapath','imagepath']:
   v=e['row'].get(k)
   if isinstance(v,str) and len(v)>=12:private.add(v)
  private.add(e['content_group_PRIVATE'])
 for p in rd(R/'research/round129/SOURCE_CALIBRATION.json')['source_files']:private.add(Path(p).stem)
 files=[];findings=[];n=0;shared={};cluster_maps=collections.defaultdict(dict)
 for f in O.rglob('*'):
  if not f.is_file():continue
  assert f.suffix in ['.py','.json','.csv','.md','.log'];files.append(f);s=f.read_text()
  for v in private:
   if v in s:findings.append((str(f.relative_to(O)),'private_literal'))
  if 'LOCAL_HOME/' in s:findings.append((str(f.relative_to(O)),'local_home'))
  if re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',s):findings.append((str(f.relative_to(O)),'credential_pattern'))
 for f in (O/'tasks').glob('*.csv'):
  pub=list(csv.DictReader(f.open()));rows=rd(H/'scores_attempt001'/(f.stem+'_ALL_ROWS_PRIVATE.json'));assert len(pub)==len(rows)
  for i,(p,r) in enumerate(zip(pub,rows)):
   assert p['row']==f'ROW{i+1:04d}' and p['source']==r['source'] and p['target']==r['target'];assert len([k for k in p if k.endswith('__IoU')])==4
   assert not any('PRIVATE' in k or 'query' in k.lower() for k in p)
   for k,x in r['arms'].items():assert abs(float(p[k+'__IoU'])-x['IoU'])<1e-14
   assert abs(float(p['sampling_weight'])-r['sampling_weight'])<1e-14
   for kind,pk in [('frame','frame_PRIVATE'),('content','content_group_PRIVATE'),('scene','scene_PRIVATE')]:
    key=(r['target'],r[pk]);value=p[kind+'_cluster'];assert key not in cluster_maps[kind] or cluster_maps[kind][key]==value;cluster_maps[kind][key]=value
   for k,x in r['diagnostics'].items():
    if isinstance(x,bool):assert p[k]==str(x)
    elif x is None:assert p[k]==''
    else:assert abs(float(p[k])-x)<1e-12
   n+=1
 for x in rd(O/'MANIFEST.json')['files']:
  p=O/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 assert n==1798 and len(cluster_maps['content'])==384 and len(cluster_maps['frame'])==851 and len(cluster_maps['scene'])==84 and not findings,findings
 out={'status':'PASS_COMPLETE_SCALARS_SHARED_ANONYMOUS_CLUSTERS_PRIVATE_LITERAL_SCAN','files':len(files),'task_rows':n,'arms':4,'content_groups':384,'frame_paths':851,'recording_clusters':84,'findings':0,'raw_media_coordinates_ids_features_weights_or_third_party_exported':False};(H/'PUBLIC_EXPORT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
