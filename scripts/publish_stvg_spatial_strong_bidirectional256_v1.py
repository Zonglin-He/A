"""Explicit allowlist publication, dual-ref leases, remote byte/tree receipts."""
from pathlib import Path
import sys,os,json,hashlib,ast,re,urllib.request,concurrent.futures
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_spatial_strong_bidirectional256_v1 import D,P,read,put,sha
from scripts.publish_stvg_referdino_finalize_v1 import git,remote,unrelated,REFS,REPO

def stage():
 assert read(P/'AUDIT_PUBLIC.json')['status']=='pass'and read(P/'INDEPENDENT_SPATIAL_AUDIT_PUBLIC.json')['status']=='pass'and read(P/'ROOT_AUDIT_PUBLIC.json')['status']=='pass'
 p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items())
 pub=D/'publication';pub.mkdir(exist_ok=True)
 files=['protocols/stvg_spatial_strong_bidirectional256_v1.md','protocols/stvg_spatial_strong_bidirectional256_seed41_revision001.md','scripts/seed41_stvg_spatial_strong_bidirectional256_revision001.py']+[f'scripts/{name}_stvg_spatial_strong_bidirectional256_v1.py'for name in ['run','score','audit','complete','profile','review','publish']]+[str(f.relative_to(R))for f in sorted(P.glob('*'))if f.is_file()]
 private=[]
 for ds in ['hc2','vidstg']:
  raw=read(R/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']
  for r in p['records']:
   if r['dataset']==ds:private.extend([r['parent'],raw[r['query_ordinal']]['input']['caption']])
 sensitive=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)');manifest=[]
 for f in files:
  b=(R/f).read_bytes();s=b.decode();assert not sensitive.search(s),f;assert all(v not in s for v in private if len(v)>=8),(f,'raw caption or parent')
  if f.endswith('.py'):ast.parse(s)
  if f.endswith('.json'):json.loads(s)
  manifest.append(dict(path=f,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
 assert git('remote','get-url','origin').decode().strip()in ['https://github.com/Zonglin-He/A.git','https://github.com/Zonglin-He/A','git@github.com:Zonglin-He/A.git']
 f=pub/'COMMIT_PLAN.json'
 if f.exists():assert read(f)['files']==manifest;return read(f)
 old=remote();assert len(old)==2 and len(set(old.values()))==1;parent=old[REFS[0]];assert git('rev-parse','HEAD').decode().strip()==parent
 assert not git('diff','--name-only','HEAD').strip()and not git('diff','--cached','--name-only').strip()
 put(pub/'UNRELATED_UNTRACKED_BEFORE.json',unrelated());env=dict(os.environ,GIT_INDEX_FILE=str(pub/'private-export.index'));git('read-tree',parent,env=env)
 for row in manifest:
  blob=git('hash-object','-w','--stdin',data=(R/row['path']).read_bytes()).decode().strip();assert blob==row['git_blob'];git('update-index','--add','--cacheinfo','100644',blob,row['path'],env=env)
 tree=git('write-tree',env=env).decode().strip();ident=git('show','-s','--format=%an%n%ae%n%cn%n%ce',parent).decode().splitlines();assert len(ident)==4
 env=dict(os.environ,GIT_AUTHOR_NAME=ident[0],GIT_AUTHOR_EMAIL=ident[1],GIT_COMMITTER_NAME=ident[2],GIT_COMMITTER_EMAIL=ident[3]);message='Evaluate frozen spatial OPD plus Strong on both 128-parent migration panels\n';commit=git('commit-tree',tree,'-p',parent,data=message.encode(),env=env).decode().strip();changed=set(git('diff-tree','--no-commit-id','--name-only','-r',commit).decode().splitlines());assert changed<=set(files)
 plan=dict(commit=commit,tree=tree,parent=parent,old_refs=old,files=manifest,message=message.strip());put(f,plan);return plan

def verify():
 p=read(D/'publication/COMMIT_PLAN.json');assert remote()==dict.fromkeys(REFS,p['commit']);git('fetch','origin',*REFS);assert git('rev-parse',p['commit']+'^{tree}').decode().strip()==p['tree']
 def one(r):
  u='https://raw.githubusercontent.com/Zonglin-He/A/'+p['commit']+'/'+r['path']
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'STVG-publication-audit'}),timeout=60)as f:b=f.read()
  assert len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']and git('show',p['commit']+':'+r['path'])==b
  assert git('ls-tree',p['commit'],'--',r['path']).decode().strip()=='100644 blob '+r['git_blob']+'\t'+r['path'];return dict(**r,remote_bytes_hash_blob_tree='pass')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:checked=list(pool.map(one,p['files']))
 assert not git('diff','--name-only','HEAD').strip();git('merge','--ff-only',p['commit']);assert unrelated()==read(D/'publication/UNRELATED_UNTRACKED_BEFORE.json');assert not git('diff','--name-only','HEAD').strip()and not git('diff','--cached','--name-only').strip()
 receipt=dict(status='pass',commit=p['commit'],tree=p['tree'],parent=p['parent'],files=checked,count=len(checked),bytes=sum(r['bytes']for r in checked),both_refs_fresh_CAS=True,remote_actual_bytes_hash_blob_tree=True,unrelated_untracked_preserved=True,registry_promotion=False,scope='frozen bidirectional128-parent panel five-row evaluation, no new selection')
 put(D/'publication/FINAL_GITHUB_RECEIPT.json',receipt);print(json.dumps({k:v for k,v in receipt.items()if k!='files'}),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='stage':print(json.dumps({k:v for k,v in stage().items()if k!='files'}))
 elif sys.argv[1]=='verify':verify()
