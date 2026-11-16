"""Independent anonymous source task/negative/diagnostic evidence and privacy audit."""
from pathlib import Path
import csv,json,hashlib,math,re,ahocorasick
R=Path(__file__).resolve().parents[2];H=R/'research/round124';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def eq(a,b):return a=='' if b is None else a==str(b) if isinstance(b,(str,bool)) else float(a)==b

def main():
 ff=sorted(p for p in O.rglob('*') if p.is_file());mf=rd(O/'MANIFEST.json');assert len(ff)==len(mf['files'])+1
 for f in mf['files']:assert sha(O/f['path'])==f['sha256'] and (O/f['path']).stat().st_size==f['bytes']
 for x in rd(P/'EXPORT_INDEX_PRIVATE.json')['sources']:assert (O/x['public']).read_text()==(R/x['source']).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 rr=rd(H/'scores_attempt001/ALL_ROWS_PRIVATE.json');rowmap={};scmap={};n=values=0
 for src in ['waymo','drone','quad']:
  rows=[r for r in rr if r['source']==src];got=cr(O/'tasks'/f'{src}.csv');assert len(got)==len(rows)==128;scmap[src]=sorted({r['scene_PRIVATE'] for r in rows})
  for i,(a,b) in enumerate(zip(got,rows)):
   tag=f'ROW{i+1:05d}';rowmap[(src,b['id_PRIVATE'])]=tag;assert a['row']==tag and a['scene_cluster']==f"CLUSTER{scmap[src].index(b['scene_PRIVATE'])+1:03d}"
   for k in ['source','frame_group','eligible','oracle_IoU']:assert eq(a[k],b[k])
   for arm,v in b['arms'].items():
    for k,val in v.items():
     if k!='query_PRIVATE':assert eq(a[arm+'__'+k],val)
    values+=1
   n+=1
 for name in ['PROCESS_ROWS_PRIVATE','GOOD_FAILURE_CASES_PRIVATE']:
  want=rd(H/(name+'.json'))
  for src in ['waymo','drone','quad']:
   rows=[r for r in want if r['source']==src];file=O/'diagnostics'/name.replace('_PRIVATE','')/(src+'.csv');got=cr(file) if file.exists() else [];assert len(got)==len(rows)
   for a,b in zip(got,rows):assert a['row']==rowmap[(src,b['id_PRIVATE'])] and all(eq(a[k],v) for k,v in b.items() if k!='id_PRIVATE')
 for name in ['AGGREGATE_ROWS','ALL_PAIRED_ROWS']:
  for a,b in zip(cr(O/'results'/(name+'.csv')),rd(H/'scores_attempt001'/(name+'.json'))):assert all(eq(a[k],v) for k,v in b.items())
 got=cr(O/'results/SCENE_ROWS.csv');want=rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json');assert len(got)==len(want)
 for a,b in zip(got,want):assert a['scene_cluster']==f"CLUSTER{scmap[b['source']].index(b['scene_PRIVATE'])+1:03d}" and all(eq(a[k],v) for k,v in b.items() if k!='scene_PRIVATE')
 tokens={(x['kind'],x['value']) for x in rd(R/'research/round96/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json')['tokens']}
 for e in rd(R/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json'):
  for kind,k in [('id','id'),('scene','scene'),('caption','caption')]:tokens.add((kind,e['row'][k]))
 auto=ahocorasick.Automaton();tt={}
 for k,t in tokens:tt.setdefault(t,set()).add(k)
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
  assert not f.is_symlink() and f.suffix in ['.json','.md','.py','.csv','.log'];s=f.read_text()
  if f.suffix=='.json':check(json.loads(s))
  if f.suffix=='.csv':assert not set(next(csv.reader(s.splitlines())))&denied
  for end,(t,kk) in auto.iter(s):
   start=end-len(t)+1
   for k in kk:
    if k in {'id','scene','alias'} and ((start>0 and re.match('[A-Za-z0-9_]',s[start-1])) or (end+1<len(s) and re.match('[A-Za-z0-9_]',s[end+1]))):continue
    findings.append((str(f.relative_to(O)),k,hashlib.sha256(t.encode()).hexdigest()))
 assert not findings,findings;assert n==384 and values==2688;out={'state':'PASS_ALL_ANONYMOUS_SOURCE_TASK_NEGATIVES_DIAGNOSTICS_SCENES_AND_PRIVACY','n':n,'task_values':values,'files':len(ff),'bytes':sum(f.stat().st_size for f in ff),'private_token_count':len(tokens),'findings':0};d=H/'independent_public_audit_attempt001';d.mkdir(exist_ok=False);(d/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
