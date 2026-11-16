"""Independent anonymous scalar fidelity, full negative tails and privacy audit."""
from pathlib import Path
import json,csv,hashlib,re,math,ahocorasick
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,default=1);arg=ap.parse_args();tag=str(arg.attempt).zfill(3)
R=Path(__file__).resolve().parents[2];H=R/'research/round107';B=R/'crossplatform/round107';P=H/'publication';O=P/('public_export_attempt'+tag);S=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def eq(v,s):return str(v)==s if isinstance(v,(str,bool)) else float(s)==v
def main():
 mf=rd(O/'MANIFEST.json');files=sorted(p for p in O.rglob('*') if p.is_file());assert {str(f.relative_to(O)) for f in files}=={x['path'] for x in mf['files']}|{'MANIFEST.json'}
 for f in mf['files']:assert sha(O/f['path'])==f['sha256'] and (O/f['path']).stat().st_size==f['bytes']
 for f in rd(P/'EXPORT_INDEX_PRIVATE.json')['own_code_files']:assert (O/'own_code'/f).read_text()==(R/f).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 n=values=fitn=0
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));pub=csvread(O/'tasks'/(name+'.csv'));fp=O/'fit_endpoints'/(name+'.csv');fits=csvread(fp) if fp.exists() else [];assert len(rr)==len(pub);want=[];sc=sorted({r['scene_PRIVATE'] for r in rr})
  for i,(a,r) in enumerate(zip(pub,rr)):
   alias='ROW'+str(i+1).zfill(6);assert a['row']==alias and a['source']==r['source'] and a['target']==r['target'];assert a['scene_cluster']=='CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3)
   for arm,x in r['arms'].items():
    for k,v in x.items():
     if k!='query_PRIVATE':assert float(a[arm+'__'+k])==v
    values+=1
   for fit,x in r['fits'].items():want.append({'row':alias,'source':r['source'],'target':r['target'],'fit':fit,**{k:v for k,v in x.items() if k!='fit_view_labels'}})
   n+=1
  assert len(fits)==len(want)
  for a,r in zip(fits,want):assert all(eq(v,a[k]) for k,v in r.items())
  fitn+=len(fits)
 assert n==768 and values==42240
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:
  a=csvread(O/'results'/(name+'.csv'));rr=rd(S/(name+'.json'));assert len(a)==len(rr)
  for x,r in zip(a,rr):assert all(eq(v,x[k]) for k,v in r.items())
 registry=rd(R/'research/round96/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json');tokens={(x['kind'],x['value']) for x in registry['tokens']}
 for f in (R/'research/round97/input').glob('*.json'):
  for r in rd(f):
   for kind,k in [('id','id'),('scene','scene'),('caption','caption')]:tokens.add((kind,r[k]))
 buckets={}
 for kind,t in tokens:buckets.setdefault(t,set()).add(kind)
 auto=ahocorasick.Automaton()
 for t,kinds in buckets.items():auto.add_word(t,(t,kinds))
 auto.make_automaton();findings=[];denied={'id','id_PRIVATE','row_PRIVATE','scene_PRIVATE','caption','metapath','imagepath','pointpath','source_boxes','query_PRIVATE','queries','coordinates','features','weights','api_key','access_token','private_key'}
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&denied
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
  elif isinstance(x,float):assert math.isfinite(x)
 for f in files:
  assert not f.is_symlink() and f.suffix in {'.py','.md','.json','.csv'};s=f.read_text()
  if f.suffix=='.json':check(json.loads(s))
  if f.suffix=='.csv':assert not set(next(csv.reader(s.splitlines())))&denied
  for end,(t,kinds) in auto.iter(s):
   start=end-len(t)+1
   for kind in kinds:
    if kind in {'id','scene','alias'} and ((start>0 and re.match('[A-Za-z0-9_]',s[start-1])) or (end+1<len(s) and re.match('[A-Za-z0-9_]',s[end+1]))):continue
    findings.append({'file':str(f.relative_to(O)),'kind':kind,'token_sha256':hashlib.sha256(t.encode()).hexdigest()})
 assert not findings,findings
 out={'state':'PASS_R107_ALL_PUBLIC_SCALARS_FITS_NEGATIVES_AND_PRIVACY','files':len(files),'bytes':sum(f.stat().st_size for f in files),'n':n,'readouts':values,'fit_endpoint_rows':fitn,'privacy_findings':0,'deny_tokens':len(tokens),'own_code_files_verified':len(rd(P/'EXPORT_INDEX_PRIVATE.json')['own_code_files'])};p=H/('independent_public_audit_attempt'+tag);p.mkdir();(p/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
