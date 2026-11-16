"""Separate public scalar fidelity, timing/update metadata and privacy verification."""
from pathlib import Path
import json,csv,hashlib,math,re,ahocorasick
R=Path(__file__).resolve().parents[2];H=R/'research/round117';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def eq(a,v):return a==str(v) if isinstance(v,(str,bool)) else float(a)==v
def main():
 mf=rd(O/'MANIFEST.json');ff=sorted(f for f in O.rglob('*') if f.is_file());assert len(ff)==len(mf['files'])+1
 for f in mf['files']:assert sha(O/f['path'])==f['sha256'] and (O/f['path']).stat().st_size==f['bytes']
 for f in rd(P/'EXPORT_INDEX_PRIVATE.json')['sources']:assert (O/f['public']).read_text()==(R/f['source']).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 values=aliases=n=0;wanted=[]
 for f in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json'):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rr=rd(f);got=cr(O/'tasks'/(name+'.csv'));assert len(rr)==len(got);sc=sorted({r['scene_PRIVATE'] for r in rr})
  for i,(a,r) in enumerate(zip(got,rr)):
   alias=f'ROW{i+1:06d}';assert a['row']==alias and a['scene_cluster']==f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}"
   for k in ['source','target','protected_alias_fallback','C_available','seconds']:assert eq(a[k],r[k])
   for k,x in r['arms'].items():
    for m,v in x.items():
     if m!='query_PRIVATE':assert float(a[k+'__'+m])==v
    values+=1
   n+=1;aliases+=r['protected_alias_fallback']
 scenes=[]
 for x in rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json'):
  name=x['source']+'_to_'+x['target'];sc=sorted({r['scene_PRIVATE'] for r in rd(H/'scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))});scenes.append({k:v for k,v in x.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{sc.index(x['scene_PRIVATE'])+1:03d}"})
 got=cr(O/'results/ALL_SCENE_ROWS.csv');assert len(got)==len(scenes)
 for a,b in zip(got,scenes):assert all(eq(a[k],v) for k,v in b.items())
 wanted=[]
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:
  a=cr(O/'results'/(name+'.csv'));b=rd(H/'scores_attempt001'/(name+'.json'));assert len(a)==len(b)
  for x,y in zip(a,b):assert all(eq(x[k],v) for k,v in y.items())
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
    findings.append((str(f.relative_to(O)),k,hashlib.sha256(t.encode()).hexdigest()))
 assert not findings,findings;assert n==1536 and aliases==110 and values==35328
 out={'state':'PASS_ALL_ANONYMOUS_TASK_TIMING_PROPOSAL_NEGATIVE_TAIL_AND_PRIVACY','files':len(ff),'bytes':sum(f.stat().st_size for f in ff),'n':n,'aliases':aliases,'task_values':values,'update_proposals':len(wanted),'source_files_checked':len(rd(P/'EXPORT_INDEX_PRIVATE.json')['sources']),'privacy_findings':0,'deny_tokens':len(tokens)};d=H/'independent_public_audit_attempt001';d.mkdir();(d/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
