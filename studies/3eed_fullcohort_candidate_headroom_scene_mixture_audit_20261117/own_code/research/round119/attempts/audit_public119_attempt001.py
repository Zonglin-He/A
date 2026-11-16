"""Independent scalar fidelity for all18,620rows, every positive/negative and privacy."""
from pathlib import Path
import json,csv,hashlib,re,math,ahocorasick
R=Path(__file__).resolve().parents[2];H=R/'research/round119';O=H/'analysis_attempt001';P=H/'publication';D=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):
 with p.open() as f:return list(csv.DictReader(f))
def parts(p):return [r for f in sorted(p.glob('part*.csv')) for r in csvread(f)]
def eq(a,v):return a==str(v) if isinstance(v,(str,bool)) or v is None else float(a)==v
def main():
 mf=rd(D/'MANIFEST.json');ff=sorted(p for p in D.rglob('*') if p.is_file());assert len(ff)==len(mf['files'])+1
 for f in mf['files']:assert sha(D/f['path'])==f['sha256'] and (D/f['path']).stat().st_size==f['bytes']
 for f in rd(P/'EXPORT_INDEX_PRIVATE.json')['sources']:assert (D/f['public']).read_text()==(R/f['source']).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 mapping={};rowmap={};n=0;values=0
 for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rr=rd(f);got=parts(D/'tasks'/name);sc=sorted({r['scene_PRIVATE'] for r in rr});mapping[name]=sc;rowmap[name]={};assert len(got)==len(rr)
  for i,(a,r) in enumerate(zip(got,rr)):
   tag=f'ROW{i+1:06d}';assert a['row']==tag and a['scene_cluster']==f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}";rowmap[name][r['id_PRIVATE']]=tag
   for k in ['source','target','cohort','oracle_IoU','unknown_roles','visible_seeds']:assert eq(a[k],r[k])
   for k,v in r['values'].items():assert float(a[k+'__IoU'])==v;values+=1
   n+=1
 for fn in ['METHOD_ROWS','PAIRED_ROWS','COMPOSITION_ROWS','AGGREGATE_ROWS','R118_POSTSTRATIFIED_METHOD_ROWS','R118_POSTSTRATIFIED_AGGREGATE_ROWS']:
  a=csvread(D/'results'/(fn+'.csv'));b=rd(O/(fn+'.json'));assert len(a)==len(b)
  for x,y in zip(a,b):assert all(eq(x[k],v) for k,v in y.items())
 for fn in ['SCENE_ROWS_PRIVATE','R118_SCENE_SCALARS_PRIVATE','POSTSTRATIFICATION_WEIGHTS_PRIVATE']:
  allrows=rd(O/(fn+'.json'))
  for name,sc in mapping.items():
   src,_,tgt=name.split('_');want=[{k:v for k,v in x.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{sc.index(x['scene_PRIVATE'])+1:03d}"} for x in allrows if x['source']==src and x['target']==tgt];got=parts(D/'scenes'/fn.replace('_PRIVATE','')/name);assert len(want)==len(got)
   for a,b in zip(got,want):assert all(eq(a[k],v) for k,v in b.items())
 got=csvread(D/'results/GOOD_FAILURE_CASES.csv');want=rd(O/'GOOD_FAILURE_CASES_PRIVATE.json');assert len(got)==len(want)
 for a,r in zip(got,want):assert a['row']==rowmap[r['source']+'_to_'+r['target']][r['id_PRIVATE']] and all(eq(a[k],v) for k,v in r.items() if k!='id_PRIVATE')
 tokens={(x['kind'],x['value']) for x in rd(R/'research/round96/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json')['tokens']}
 for f in (R/'research/round97/input').glob('*.json'):
  for r in rd(f):
   for kind,k in [('id','id'),('scene','scene'),('caption','caption')]:tokens.add((kind,r[k]))
 b={}
 for kind,t in tokens:b.setdefault(t,set()).add(kind)
 auto=ahocorasick.Automaton()
 for t,k in b.items():auto.add_word(t,(t,k))
 auto.make_automaton();denied={'id','id_PRIVATE','row_PRIVATE','scene_PRIVATE','caption','metapath','imagepath','pointpath','source_boxes','query_PRIVATE','queries','coordinates','features','weights','api_key','access_token','private_key'}
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&denied
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
  elif isinstance(x,float):assert math.isfinite(x)
 findings=[]
 for f in ff:
  assert not f.is_symlink() and f.suffix in ['.py','.md','.json','.csv'];s=f.read_text()
  if f.suffix=='.json':check(json.loads(s))
  if f.suffix=='.csv':assert not set(next(csv.reader(s.splitlines())))&denied
  for end,(t,kk) in auto.iter(s):
   start=end-len(t)+1
   for k in kk:
    if k in {'id','scene','alias'} and ((start>0 and re.match('[A-Za-z0-9_]',s[start-1])) or (end+1<len(s) and re.match('[A-Za-z0-9_]',s[end+1]))):continue
    findings.append((str(f.relative_to(D)),k,hashlib.sha256(t.encode()).hexdigest()))
 assert not findings,findings;assert n==18620 and values==148960
 out={'state':'PASS_ALL_FULLCOHORT_VALUES_NEGATIVES_POSTSTRATIFICATION_SCENES_AND_PRIVACY','files':len(ff),'bytes':sum(f.stat().st_size for f in ff),'n':n,'reused_task_values':values,'new_GT_scoring':0,'source_files_checked':len(rd(P/'EXPORT_INDEX_PRIVATE.json')['sources']),'privacy_findings':0,'deny_tokens':len(tokens)};d=H/'independent_public_audit_attempt001';d.mkdir();(d/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
