from common133 import *
import csv,re,collections
O133=H133/'publication/public_export_attempt001'
def main():
 tokens=set()
 for e in rd(H133/'INPUT_REGISTRY_PRIVATE.json'):
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 findings=[];files=[];n=0;maps=collections.defaultdict(dict)
 for p in O133.rglob('*'):
  if not p.is_file():continue
  assert p.suffix in ['.csv','.md','.py','.json','.log'];files.append(p);s=p.read_text()
  for token in tokens:
   if token in s:findings.append((str(p.relative_to(O133)),'private_literal'))
  if 'LOCAL_HOME/' in s:findings.append((str(p.relative_to(O133)),'local_home'))
  if re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',s):findings.append((str(p.relative_to(O133)),'credential_pattern'))
 for p in (O133/'tasks').glob('*.csv'):
  pub=list(csv.DictReader(p.open()));rows=rd(H133/'scores_attempt001'/(p.stem+'_ALL_ROWS_PRIVATE.json'));assert len(pub)==len(rows)
  for i,(x,r) in enumerate(zip(pub,rows)):
   assert x['row']==f'ROW{i+1:04d}' and x['source']==r['source'] and x['target']==r['target'];assert len([k for k in x if k.endswith('__IoU')])==9;assert not any(k.endswith('_PRIVATE') for k in x)
   for k,v in r['IoUs'].items():assert abs(float(x[k+'__IoU'])-v)<1e-14
   for kind,pk in [('frame','frame_PRIVATE'),('content','content_group_PRIVATE'),('scene','scene_PRIVATE')]:
    key=(r['target'],r[pk]);value=x[kind+'_cluster'];assert key not in maps[kind] or maps[kind][key]==value;maps[kind][key]=value
   for step,diag in r['diagnostics'].items():
    for k,v in diag.items():
     actual=x['step'+step+'__'+k]
     if isinstance(v,bool):assert actual==str(v)
     else:assert abs(float(actual)-v)<1e-12
   n+=1
 prior=list(csv.DictReader((O133/'diagnostics/R132_ALL76_CASES.csv').open()));assert len(prior)==76 and sum(r['old_change']=='better' for r in prior)==49 and sum(r['old_change']=='worse' for r in prior)==27
 for x in rd(O133/'MANIFEST.json')['files']:p=O133/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
 assert n==1798 and len(maps['content'])==384 and len(maps['frame'])==851 and len(maps['scene'])==84 and not findings,findings
 out={'status':'PASS9ARMS_ALLSCALARS_CLUSTER_DEPENDENCE_ALL76PRIORCASES_AND_PRIVATE_SCAN','n':n,'task_values':n*9,'files':len(files),'prior_good':49,'prior_bad':27,'private_tokens':len(tokens),'findings':0,'raw_media_coordinates_ids_features_weights_or_third_party_exported':False};dump(H133/'PUBLIC_EXPORT_AUDIT.json',out);print(out)
if __name__=='__main__':main()
