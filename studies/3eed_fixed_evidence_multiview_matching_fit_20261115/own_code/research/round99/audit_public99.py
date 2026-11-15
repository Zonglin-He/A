"""Independent per-task/fit/trial preservation and complete private-token scan."""
from pathlib import Path
import json,csv,hashlib,math,re,ahocorasick
R=Path(__file__).resolve().parents[2];H=R/'research/round99';B=R/'crossplatform/round99';P=H/'publication';O=P/'public_export_attempt001';S=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def eq(x,y):return str(x)==y if isinstance(x,(bool,str)) else float(y)==x
def main():
 manifest=rd(O/'MANIFEST.json');files=sorted(f for f in O.rglob('*') if f.is_file());assert {str(f.relative_to(O)) for f in files}=={x['path'] for x in manifest['files']}|{'MANIFEST.json'}
 for f in manifest['files']:assert sha(O/f['path'])==f['sha256']
 for f in rd(P/'EXPORT_INDEX_PRIVATE.json')['own_code_files']:assert (O/'own_code'/f).read_text()==(R/f).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 n=values=trialn=0
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rows=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));pub=csvread(O/'tasks'/(name+'.csv'));assert len(rows)==len(pub)==128;want=[];fitwant=[]
  for i,(r,p) in enumerate(zip(rows,pub)):
   alias='ROW'+str(i+1).zfill(4);assert p['row']==alias and p['source']==r['source'] and p['target']==r['target']
   for arm,x in r['arms'].items():
    for k,v in x.items():
     if k!='query_PRIVATE':assert float(p[arm+'__'+k])==v
    values+=1
   for fit,x in rd(B/name/'records_PRIVATE'/(r['id_PRIVATE']+'.json')).items():
    fitwant.append({'row':alias,'source':r['source'],'target':r['target'],'fit':fit,**{k:v for k,v in x.items() if k!='trials'}})
    for j,t in enumerate(x['trials']):want.append({'row':alias,'source':r['source'],'target':r['target'],'fit':fit,'trial_index':j,**t})
   n+=1
  got=[x for f in sorted((O/'trials').glob(name+'_part*.csv')) for x in csvread(f)];assert len(got)==len(want)
  for a,b in zip(got,want):assert all(eq(v,a[k]) for k,v in b.items())
  fitgot=csvread(O/'fits'/(name+'.csv'));assert len(fitgot)==len(fitwant)
  for a,b in zip(fitgot,fitwant):assert all(eq(v,a[k]) for k,v in b.items())
  trialn+=len(got)
 assert n==768 and values==15360
 registry=rd(R/'research/round96/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json');tokens={(x['kind'],x['value']) for x in registry['tokens']}
 for p in (R/'research/round97/input').glob('*.json'):
  for r in rd(p):
   for kind,k in [('id','id'),('scene','scene'),('caption','caption')]:tokens.add((kind,r[k]))
 buckets={}
 for kind,t in tokens:buckets.setdefault(t,set()).add(kind)
 auto=ahocorasick.Automaton()
 for t,kinds in buckets.items():auto.add_word(t,(t,kinds))
 auto.make_automaton();findings=[];denied={'id','id_PRIVATE','row_PRIVATE','scene_PRIVATE','caption','metapath','imagepath','pointpath','source_boxes','query_PRIVATE','queries','coordinates','features','weights','api_key','access_token','private_key'}
 def check(v):
  if isinstance(v,dict):
   assert not set(v)&denied
   for x in v.values():check(x)
  elif isinstance(v,list):
   for x in v:check(x)
  elif isinstance(v,float):assert math.isfinite(v)
 for f in files:
  assert not f.is_symlink() and f.suffix in {'.py','.md','.json','.csv'};t=f.read_text()
  if f.suffix=='.json':check(json.loads(t))
  if f.suffix=='.csv':assert not set(next(csv.reader(t.splitlines())))&denied
  for end,(token,kinds) in auto.iter(t):
   start=end-len(token)+1
   for kind in kinds:
    if kind in {'id','scene','alias'} and ((start>0 and re.match('[A-Za-z0-9_]',t[start-1])) or (end+1<len(t) and re.match('[A-Za-z0-9_]',t[end+1]))):continue
    findings.append({'file':str(f.relative_to(O)),'kind':kind,'token_sha256':hashlib.sha256(token.encode()).hexdigest()})
 assert not findings,findings
 result={'state':'PASS_R99_ALL_TASK_VALUES_FIT_TRIALS_AND_PRIVACY','files':len(files),'bytes':sum(f.stat().st_size for f in files),'n':n,'readouts':values,'trial_rows':trialn,'privacy_findings':0,'own_code_files_verified':len(rd(P/'EXPORT_INDEX_PRIVATE.json')['own_code_files']),'deny_tokens':len(tokens)};d=H/'independent_public_audit_attempt001';d.mkdir();(d/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
