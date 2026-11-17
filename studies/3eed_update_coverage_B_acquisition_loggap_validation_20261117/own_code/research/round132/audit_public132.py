from common132 import *
import csv,re,collections
O132=H132/'publication/public_export_attempt001'
def main():
 tokens=set()
 entries=rd(R132/'research/round131/INPUT_REGISTRY_PRIVATE.json')+[e['entry'] for e in rd(H132/'SOURCE_CALIBRATION_INPUTS_PRIVATE.json')]
 for e in entries:
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
 for a in rd(H132/'TARGET_B3_CANDIDATES_PRIVATE.json'):
  for c in a['candidates']:
   for role in ['A','B']:
    f=c['pair'][role]
    for k in ['frame_PRIVATE','content_group_PRIVATE','scene_PRIVATE','metapath','pointpath','imagepath']:
     v=f.get(k)
     if isinstance(v,str) and len(v)>=12:tokens.add(v)
 findings=[];files=[];n=0;maps=collections.defaultdict(dict)
 for p in O132.rglob('*'):
  if not p.is_file():continue
  assert p.suffix in ['.csv','.md','.py','.json','.log'];files.append(p);s=p.read_text()
  for token in tokens:
   if token in s:findings.append((str(p.relative_to(O132)),'private_literal'))
  if 'LOCAL_HOME/' in s:findings.append((str(p.relative_to(O132)),'local_home'))
  if re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',s):findings.append((str(p.relative_to(O132)),'credential_pattern'))
 for p in (O132/'tasks').glob('*.csv'):
  pub=list(csv.DictReader(p.open()));rows=rd(H132/'scores_attempt001'/(p.stem+'_ALL_ROWS_PRIVATE.json'));assert len(pub)==len(rows)
  for i,(x,r) in enumerate(zip(pub,rows)):
   assert x['row']==f'ROW{i+1:04d}' and x['source']==r['source'] and x['target']==r['target'];assert len([k for k in x if k.endswith('__IoU')])==12;assert not any(k.endswith('_PRIVATE') for k in x)
   for k,v in r['IoUs'].items():assert abs(float(x[k+'__IoU'])-v)<1e-14
   for kind,pk in [('frame','frame_PRIVATE'),('content','content_group_PRIVATE'),('scene','scene_PRIVATE')]:
    key=(r['target'],r[pk]);value=x[kind+'_cluster'];assert key not in maps[kind] or maps[kind][key]==value;maps[kind][key]=value
   for kind,diag in r['diagnostics'].items():
    for k,v in diag.items():
     actual=x[kind+'__'+k]
     if v is None:assert actual==''
     elif isinstance(v,bool):assert actual==str(v)
     else:assert abs(float(actual)-v)<1e-12
   n+=1
 for x in rd(O132/'MANIFEST.json')['files']:p=O132/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
 assert n==1798 and len(maps['content'])==384 and len(maps['frame'])==851 and len(maps['scene'])==84 and not findings,findings
 out={'status':'PASS12ARMS_FULL_ANONYMOUS_SCALARS_CLUSTERS_AND_PRIVATE_LITERAL_SCAN','n':n,'values':n*12,'files':len(files),'private_tokens':len(tokens),'findings':0,'raw_data_coordinates_features_weights_or_thirdparty_exported':False};dump(H132/'PUBLIC_EXPORT_AUDIT.json',out);print(out)
if __name__=='__main__':main()
