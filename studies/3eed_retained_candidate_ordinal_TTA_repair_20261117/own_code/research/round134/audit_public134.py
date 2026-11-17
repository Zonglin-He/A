from common134 import *
import re,csv,collections
O134=H134/'publication/public_export_attempt001'
def main():
 tokens=set()
 for e in rd(H134/'INPUT_REGISTRY_PRIVATE.json'):
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 findings=[];files=[];n=0;maps=collections.defaultdict(dict)
 for p in O134.rglob('*'):
  if not p.is_file():continue
  assert p.suffix in ['.csv','.json','.py','.md','.log'];files.append(p);txt=p.read_text()
  for token in tokens:
   if token in txt:findings.append((str(p.relative_to(O134)),'private_literal'))
  if 'LOCAL_HOME/' in txt:findings.append((str(p.relative_to(O134)),'local_home'))
  if re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',txt):findings.append((str(p.relative_to(O134)),'credential_pattern'))
 for p in (O134/'tasks').glob('*.csv'):
  pub=list(csv.DictReader(p.open()));rows=rd(H134/'scores_attempt001'/(p.stem+'_ALL_ROWS_PRIVATE.json'));assert len(pub)==len(rows)
  for i,(x,r) in enumerate(zip(pub,rows)):
   assert x['row']==f'ROW{i+1:04d}' and x['source']==r['source'] and x['target']==r['target'];assert len([k for k in x if k.endswith('__IoU')])==13 and not any(k.endswith('_PRIVATE') for k in x)
   for k,v in r['IoUs'].items():assert abs(float(x[k+'__IoU'])-v)<1e-14
   for kind,pk in [('frame','frame_PRIVATE'),('content','content_group_PRIVATE'),('scene','scene_PRIVATE')]:
    key=(r['target'],r[pk]);v=x[kind+'_cluster'];assert key not in maps[kind] or maps[kind][key]==v;maps[kind][key]=v
   for arm,diag in r['diagnostics'].items():
    for k,v in diag.items():
     actual=x[arm+'__'+k]
     if isinstance(v,bool):assert actual==str(v)
     else:assert abs(float(actual)-v)<1e-12
   n+=1
 old=list(csv.DictReader((O134/'diagnostics/R132_ALL76_CASES.csv').open()));nc=list(csv.DictReader((O134/'diagnostics/OLD_NO_QUERY_CHANGE508.csv').open()));assert len(old)==76 and sum(r['old_change']=='better' for r in old)==49 and sum(r['old_change']=='worse' for r in old)==27 and len(nc)==508
 for f in rd(O134/'MANIFEST.json')['files']:p=O134/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
 assert n==1798 and len(maps['content'])==384 and len(maps['frame'])==851 and len(maps['scene'])==84 and not findings,findings
 out={'status':'PASS_ALL13ARMS_SCALARS_CLUSTERS_76CASES_508NOCHANGE_AND_PRIVATE_SCAN','n':n,'task_values':n*13,'files':len(files),'private_tokens':len(tokens),'findings':0,'raw_media_coordinates_ids_features_weights_or_third_party_exported':False};dump(H134/'PUBLIC_EXPORT_AUDIT.json',out);print(out)
if __name__=='__main__':main()
