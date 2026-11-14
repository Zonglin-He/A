"""Independent full anonymous-outcome, fit-endpoint and privacy audit."""
from pathlib import Path
import json,csv,hashlib,re,math,ahocorasick
R=Path(__file__).resolve().parents[2];H=R/'research/round98';P=H/'publication';O=P/'public_export_attempt001';S=H/'scores_attempt001';B=R/'crossplatform/round98';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def equal(v,s):return str(v)==s if isinstance(v,(str,bool)) else float(s)==v
def main():
 for f,s in rd(H/'experiment/PUBLIC_EXPORT_LOCK.json')['inputs_sha256'].items():assert sha(R/f)==s,f
 manifest=rd(O/'MANIFEST.json');idx=rd(P/'EXPORT_INDEX_PRIVATE.json');files=sorted(f for f in O.rglob('*') if f.is_file());assert {str(f.relative_to(O)) for f in files}=={f['path'] for f in manifest['files']}|{'MANIFEST.json'}
 for f in manifest['files']:assert sha(O/f['path'])==f['sha256'] and (O/f['path']).stat().st_size==f['bytes']
 for f in idx['own_code_files']:assert (O/'own_code'/f).read_text()==(R/f).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 n=values=fitn=0
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rows=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));pub=[r for f in sorted((O/'tasks').glob(name+'_part*.csv')) for r in csvread(f)];fit=[r for f in sorted((O/'fit_endpoints').glob(name+'_part*.csv')) for r in csvread(f)];expect=[];sc=sorted({r['scene_PRIVATE'] for r in rows});assert len(pub)==len(rows)
  for a,b in zip(pub,rows):
   assert a['row']==b['row'] and a['source']==b['source'] and a['target']==b['target'] and a['scene_cluster']=='CLUSTER'+str(sc.index(b['scene_PRIVATE'])+1).zfill(3)
   for arm,z in b['arms'].items():
    for k,v in z.items():
     if k!='query_PRIVATE':assert float(a[arm+'__'+k])==v
    values+=1
   for kind,u in b['fits'].items():expect.append({'row':b['row'],'source':b['source'],'target':b['target'],'fit':kind,**u})
   n+=1
  assert len(fit)==len(expect)
  for a,b in zip(fit,expect):
   for k,v in b.items():assert equal(v,a[k])
  fitn+=len(fit)
 assert n==18620 and values==539980 and fitn==74480
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:
  a=csvread(O/'results'/(name+'.csv'));b=rd(S/(name+'.json'));assert len(a)==len(b)
  for x,y in zip(a,b):
   for k,v in y.items():assert equal(v,x[k])
 registry=rd(R/'research/round96/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json');tokens={(x['kind'],x['value']) for x in registry['tokens']}
 for f in (R/'research/round97/input').glob('*.json'):
  for r in rd(f):
   for kind,k in [('id','id'),('scene','scene'),('caption','caption')]:tokens.add((kind,r[k]))
 buckets={}
 for kind,t in tokens:buckets.setdefault(t,set()).add(kind)
 auto=ahocorasick.Automaton()
 for t,k in buckets.items():auto.add_word(t,(t,k))
 auto.make_automaton();findings=[];denied={'id','id_PRIVATE','scene_PRIVATE','caption','metapath','imagepath','pointpath','source_boxes','query_PRIVATE','query_index','queries','coordinates','features','weights','api_key','access_token','private_key'}
 def check(v):
  if isinstance(v,dict):
   assert not set(v)&denied
   for x in v.values():check(x)
  elif isinstance(v,list):
   for x in v:check(x)
  elif isinstance(v,float):assert math.isfinite(v)
 for f in files:
  assert not f.is_symlink() and f.suffix in {'.py','.md','.json','.csv'};text=f.read_text();assert '\x00' not in text
  if f.suffix=='.json':check(json.loads(text))
  if f.suffix=='.csv':assert not set(next(csv.reader(text.splitlines())))&denied
  for end,(t,kinds) in auto.iter(text):
   start=end-len(t)+1
   for kind in kinds:
    if kind in {'id','scene','alias'} and ((start>0 and re.match('[A-Za-z0-9_]',text[start-1])) or (end+1<len(text) and re.match('[A-Za-z0-9_]',text[end+1]))):continue
    findings.append({'file':str(f.relative_to(O)),'kind':kind,'sha256':hashlib.sha256(t.encode()).hexdigest()})
 assert not findings,findings
 result={'state':'PASS_R98_INDEPENDENT_COMPLETE_VALUES_FIT_ENDPOINTS_NEGATIVES_AND_PRIVACY','files':len(files),'bytes':sum(f.stat().st_size for f in files),'direction_task_rows':n,'readout_task_values':values,'fit_endpoint_rows':fitn,'privacy_findings':0,'deny_tokens':len(tokens),'own_code_files_verified':len(idx['own_code_files'])};out=H/'independent_public_audit_attempt001';out.mkdir();(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
