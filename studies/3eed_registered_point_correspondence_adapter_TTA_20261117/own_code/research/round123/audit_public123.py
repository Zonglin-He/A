from pathlib import Path
import json,csv,hashlib,math,re,ahocorasick
R=Path(__file__).resolve().parents[2];H=R/'research/round123';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def eq(a,b):return a=='' if b is None else a==str(b) if isinstance(b,(str,bool)) else float(a)==b

def main():
 ff=sorted(f for f in O.rglob('*') if f.is_file());mf=rd(O/'MANIFEST.json');assert len(ff)==len(mf['files'])+1
 for f in mf['files']:assert sha(O/f['path'])==f['sha256'] and (O/f['path']).stat().st_size==f['bytes']
 for x in rd(P/'EXPORT_INDEX_PRIVATE.json')['sources']:assert (O/x['public']).read_text()==(R/x['source']).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 n=values=aliases=0;scmap={};rowmap={};physical=[]
 for f in sorted((H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json')):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rows=rd(f);got=cr(O/'tasks'/(name+'.csv'));assert len(got)==len(rows);scmap[name]=sorted({r['scene_PRIVATE'] for r in rows});rowmap[name]={}
  for i,(a,b) in enumerate(zip(got,rows)):
   tag=f'ROW{i+1:05d}';rowmap[name][b['id_PRIVATE']]=tag;assert a['row']==tag and a['scene_cluster']==f"CLUSTER{scmap[name].index(b['scene_PRIVATE'])+1:03d}"
   for k in ['cohort','source','target','protected_alias_fallback','B_active','C_available']:assert eq(a[k],b[k])
   for arm,v in b['arms'].items():
    for k,vv in v.items():
     if k!='query_PRIVATE':assert eq(a[arm+'__'+k],vv)
    values+=1
   n+=1;aliases+=b['protected_alias_fallback']
 for r in rd(H/'PROCESS_ROWS_PRIVATE.json'):
  name=r['cohort']+'_'+r['source']+'_to_'+r['target'];physical.append({k:v for k,v in r.items() if k!='id_PRIVATE'}|{'row':rowmap[name][r['id_PRIVATE']]})
 for co in ['DEV','EXP']:
  for src in ['waymo','drone','quad']:
   want=[r for r in physical if r['cohort']==co and r['source']==src];got=cr(O/'correspondence'/co/(src+'.csv'));assert len(got)==len(want)
   for a,b in zip(got,want):assert all(eq(a[k],v) for k,v in b.items())
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS','POSTSTRATIFIED_METHOD_ROWS','POSTSTRATIFIED_AGGREGATE_ROWS']:
  want=rd(H/'scores_attempt001'/(name+'.json'));got=cr(O/'results'/(name+'.csv'));assert len(got)==len(want)
  for a,b in zip(got,want):assert all(eq(a[k],v) for k,v in b.items())
 for name,scs in scmap.items():
  co,src,_,tgt=name.split('_');want=[r for r in rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json') if r['cohort']==co and r['source']==src and r['target']==tgt];got=cr(O/'scenes'/(name+'.csv'));assert len(want)==len(got)
  for a,b in zip(got,want):assert a['scene_cluster']==f"CLUSTER{scs.index(b['scene_PRIVATE'])+1:03d}" and all(eq(a[k],v) for k,v in b.items() if k!='scene_PRIVATE')
 want=rd(H/'GOOD_FAILURE_CASES_PRIVATE.json');got=cr(O/'diagnostics/GOOD_FAILURE_CASES.csv') if want else [];assert len(want)==len(got)
 for a,b in zip(got,want):name=b['cohort']+'_'+b['source']+'_to_'+b['target'];assert a['row']==rowmap[name][b['id_PRIVATE']] and all(eq(a[k],v) for k,v in b.items() if k!='id_PRIVATE')
 tokens={(x['kind'],x['value']) for x in rd(R/'research/round96/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json')['tokens']}
 for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json'):
  for kind,k in [('id','id'),('scene','scene'),('caption','caption')]:tokens.add((kind,e['row'][k]))
 tt={}
 for k,t in tokens:tt.setdefault(t,set()).add(k)
 auto=ahocorasick.Automaton()
 for t,k in tt.items():auto.add_word(t,(t,k))
 auto.make_automaton();denied={'id','id_PRIVATE','scene_PRIVATE','caption','metapath','imagepath','pointpath','source_boxes','query_PRIVATE','queries','coordinates','features','weights','api_key','access_token','private_key'};findings=[]
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&denied
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
  elif isinstance(x,float):assert math.isfinite(x)
 for f in ff:
  assert not f.is_symlink() and f.suffix in ['.py','.md','.json','.csv','.log'];s=f.read_text()
  if f.suffix=='.json':check(json.loads(s))
  if f.suffix=='.csv':assert not set(next(csv.reader(s.splitlines())))&denied
  for end,(t,kk) in auto.iter(s):
   start=end-len(t)+1
   for k in kk:
    if k in {'id','scene','alias'} and ((start>0 and re.match('[A-Za-z0-9_]',s[start-1])) or (end+1<len(s) and re.match('[A-Za-z0-9_]',s[end+1]))):continue
    findings.append((str(f.relative_to(O)),k,hashlib.sha256(t.encode()).hexdigest()))
 assert not findings,findings;assert n==2304 and values==57600 and aliases==154;out={'state':'PASS_ALL_ANONYMOUS_TASK_NEGATIVES_CORRESPONDENCE_AND_PRIVACY','n':n,'values':values,'aliases':aliases,'files':len(ff),'bytes':sum(f.stat().st_size for f in ff),'private_token_count':len(tokens),'findings':0};d=H/'independent_public_audit_attempt001';d.mkdir(exist_ok=False);(d/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
