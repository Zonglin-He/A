"""Independent all-value comparison, complete case tails and privacy scan."""
from pathlib import Path
import json,csv,hashlib,math,ahocorasick
R=Path(__file__).resolve().parents[2];H=R/'research/round102';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cr(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def eq(v,x):return str(v)==x if isinstance(v,(str,bool)) else float(x)==v
def main():
 files=sorted(f for f in O.rglob('*') if f.is_file());mf=rd(O/'MANIFEST.json');assert {str(f.relative_to(O)) for f in files}=={f['path'] for f in mf['files']}|{'MANIFEST.json'}
 for f in mf['files']:assert sha(O/f['path'])==f['sha256'] and (O/f['path']).stat().st_size==f['bytes']
 values=n=0
 for scope,D,expected in [('DEV768',H,768),('FULL18620',H/'full',18620)]:
  count=0;S=D/'scores_attempt001'
  for f in sorted(S.glob('*_ALL_ROWS_PRIVATE.json')):
   name=f.name.replace('_ALL_ROWS_PRIVATE.json','');rr=rd(f);pub=[r for p in sorted((O/scope/'tasks').glob(name+'_part*.csv')) for r in cr(p)];assert len(pub)==len(rr);sc=sorted({r['scene_PRIVATE'] for r in rr})
   for i,(r,x) in enumerate(zip(rr,pub)):
    assert x['row']=='ROW'+str(i+1).zfill(6) and x['scene_cluster']=='CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3)
    for k in ['source','target']:assert x[k]==r[k]
    for arm,z in r['arms'].items():
     for k,v in z.items():
      if k!='query_PRIVATE':assert float(x[arm+'__'+k])==v
     values+=1
    count+=1;n+=1
  assert count==expected
  for key in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS','STRATUM_METHOD_ROWS','STRATUM_PAIRED_ROWS']:
   if (S/(key+'.json')).exists():
    a=rd(S/(key+'.json'));b=cr(O/scope/'results'/(key+'.csv'));assert len(a)==len(b)
    for r,x in zip(a,b):assert all(eq(v,x[k]) for k,v in r.items())
  cases=rd(D/'GOOD_FAILURE_CASES_PRIVATE.json');published=cr(O/scope/'diagnostics/GOOD_FAILURE_SCALARS.csv')
  assert len(cases)==len(published)
  for r,x in zip(cases,published):assert all(eq(r[k],v) for k,v in x.items())
 assert n==19388 and values==447460
 tokens={(x['kind'],x['value']) for x in rd(R/'research/round96/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json')['tokens']}
 for f in (R/'research/round97/input').glob('*.json'):
  for r in rd(f):
   for kind,k in [('id','id'),('scene','scene'),('caption','caption')]:tokens.add((kind,r[k]))
 auto=ahocorasick.Automaton();bucket={}
 for kind,t in tokens:bucket.setdefault(t,set()).add(kind)
 for t,k in bucket.items():auto.add_word(t,(t,k))
 auto.make_automaton();denied={'id','id_PRIVATE','row_PRIVATE','scene_PRIVATE','caption','metapath','imagepath','pointpath','source_boxes','query_PRIVATE','queries','coordinates','features','weights','api_key','access_token','private_key'}
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&denied
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
  elif isinstance(x,float):assert math.isfinite(x)
 for f in files:
  assert not f.is_symlink() and f.suffix in {'.py','.md','.csv','.json'};s=f.read_text();hits=list(auto.iter(s));assert not hits,(f,len(hits))
  if f.suffix=='.json':check(json.loads(s))
 for f in rd(P/'EXPORT_INDEX_PRIVATE.json')['own_code_files']:assert (O/'own_code'/f).read_text()==(R/f).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 out={'state':'PASS_R102_COMPLETE_DEV_FULL_PUBLIC_VALUES_AND_PRIVACY','files':len(files),'bytes':sum(f.stat().st_size for f in files),'n':n,'readouts':values,'privacy_findings':0,'deny_tokens':len(tokens)};A=H/'independent_public_audit_attempt001';A.mkdir(exist_ok=True);(A/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
