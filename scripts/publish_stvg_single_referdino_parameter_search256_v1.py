"""Publish the one-expert replacement and retained completed/paused development work."""
import os,sys,json,hashlib,ast,re,urllib.request,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.publish_stvg_referdino_finalize_v1 import git,remote,unrelated,REFS,REPO
D=ROOT/'artifacts/stvg_single_referdino_parameter_search256_v1'
P=ROOT/'results/stvg_error_state/2026-10-20/single_referdino_parameter_search256'
def read(f):return json.loads(Path(f).read_text())
def put(f,x):Path(f).parent.mkdir(parents=True,exist_ok=True);Path(f).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def immutable():
 p=read(D/'PLAN_LOCK.json')
 for f,h in {**p['pins'],**p['protected'],**p['source_files']}.items():assert sha(ROOT/f)==h,f
 for name in ['SCORING_IMPLEMENTATION_LOCK.json','ADDITIONAL_EXECUTION_LOCK.json','OFFLINE_ANALYSIS_IMPLEMENTATION_LOCK.json','SELECTED_RUNTIME_IMPLEMENTATION_LOCK.json']:
  for f,h in read(D/name)['pins'].items():assert sha(ROOT/f)==h,f
 for ds in ['hc2','vidstg']:
  for ph in ['coarse','fine']:
   for e in read(D/ds/f'{ph.upper()}_FIT_BARRIER.json')['files'].values():assert sha(ROOT/e['path'])==e['sha256']
   for f,h in read(D/ds/ph/'PREDICTION_BARRIER.json')['files'].items():assert sha(ROOT/f)==h,f
 for f,h in read(D/'SELECTED_PREDICTION_BARRIER.json')['files'].items():assert sha(ROOT/f)==h,f

def stage():
 immutable();assert read(P/'SELECTED_ROOT_AUDIT_PUBLIC.json')['status']=='pass'
 files=['README.md','REVIEW_START_HERE.md','protocols/stvg_single_referdino_parameter_search256_v1.md']
 files += ['methods/stvg_single_referdino_opd_v1/'+f for f in ['tuned.py','tuned_cli.py','tuned_config_20261020.json','TUNED_README_20261020.md']]
 files += ['scripts/'+f for f in ['stvg_single_referdino_parameter_search256_common_v1.py','run_stvg_single_referdino_parameter_search256_v1.py','score_stvg_single_referdino_parameter_search256_v1.py','audit_stvg_single_referdino_parameter_search256_v1.py','audit_stvg_single_referdino_parameter_readouts256_v1.py','control_stvg_single_referdino_parameter_search256_v1.py','analyze_stvg_single_referdino_parameter_effects256_v1.py','finalize_stvg_single_referdino_parameter_search256_v1.py','audit_stvg_single_referdino_selected256_v1.py','report_stvg_single_referdino_parameter_search256_v1.py','profile_stvg_single_referdino_tuned256_v1.py','publish_stvg_single_referdino_parameter_search256_v1.py']]
 files += [str(f.relative_to(ROOT))for f in sorted(P.glob('*'))if f.is_file()]
 files += [str(f.relative_to(ROOT))for f in sorted((P/'all_parameter_outcomes').glob('*.json'))]
 for ds in ['hc2','vidstg']:
  files += [str(f.relative_to(ROOT))for f in sorted((P/ds).glob('*.json'))]
  for ph in ['coarse','fine']:files += [str(f.relative_to(ROOT))for f in sorted((P/ds/ph).glob('*AUDIT_PUBLIC.json'))]
 assert len(files)==len(set(files))
 sensitive=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
 plan=read(ROOT/'artifacts/stvg_spatial_strong_bidirectional256_v1/PLAN_LOCK.json');raw=[]
 for ds in ['hc2','vidstg']:
  rows=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']
  for r in plan['records']:
   if r['dataset']==ds:raw.extend([r['parent'],rows[r['query_ordinal']]['input']['caption']])
 manifest=[]
 for f in files:
  b=(ROOT/f).read_bytes();s=b.decode();assert not sensitive.search(s),f;assert all(v not in s for v in raw if len(v)>=8),f
  if f.endswith('.py'):ast.parse(s)
  if f.endswith('.json'):json.loads(s)
  manifest.append(dict(path=f,bytes=len(b),sha256=sha(ROOT/f),git_blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
 assert git('remote','get-url','origin').decode().strip()in ['https://github.com/Zonglin-He/A.git','https://github.com/Zonglin-He/A','git@github.com:Zonglin-He/A.git']
 pub=D/'publication';pub.mkdir(exist_ok=True);old=remote();assert len(old)==2 and len(set(old.values()))==1;parent=old[REFS[0]]
 assert git('rev-parse','HEAD').decode().strip()==parent
 assert not git('diff','--name-only','HEAD').strip()and not git('diff','--cached','--name-only').strip()
 put(pub/'UNRELATED_UNTRACKED_BEFORE.json',unrelated());env=dict(os.environ,GIT_INDEX_FILE=str(pub/'private-export.index'));git('read-tree',parent,env=env)
 for row in manifest:
  blob=git('hash-object','-w','--stdin',data=(ROOT/row['path']).read_bytes()).decode().strip();assert blob==row['git_blob'];git('update-index','--add','--cacheinfo','100644',blob,row['path'],env=env)
 tree=git('write-tree',env=env).decode().strip();ident=git('show','-s','--format=%an%n%ae%n%cn%n%ce',parent).decode().splitlines();assert len(ident)==4
 env=dict(os.environ,GIT_AUTHOR_NAME=ident[0],GIT_AUTHOR_EMAIL=ident[1],GIT_COMMITTER_NAME=ident[2],GIT_COMMITTER_EMAIL=ident[3])
 msg='Tune single ReferDINO OPD jointly on HC128 and Vid128 with complete signed parameter outcomes\n';commit=git('commit-tree',tree,'-p',parent,data=msg.encode(),env=env).decode().strip()
 assert set(git('diff-tree','--no-commit-id','--name-only','-r',commit).decode().splitlines()) <= set(files)
 out=dict(commit=commit,tree=tree,parent=parent,old_refs=old,files=manifest,message=msg.strip());put(pub/'COMMIT_PLAN.json',out);return out
def verify():
 p=read(D/'publication/COMMIT_PLAN.json');assert remote()==dict.fromkeys(REFS,p['commit']);git('fetch','origin',*REFS);assert git('rev-parse',p['commit']+'^{tree}').decode().strip()==p['tree']
 def one(r):
  u='https://raw.githubusercontent.com/Zonglin-He/A/'+p['commit']+'/'+r['path']
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'STVG-cached-analysis-audit'}),timeout=60)as f:b=f.read()
  assert len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']and git('show',p['commit']+':'+r['path'])==b
  assert git('ls-tree',p['commit'],'--',r['path']).decode().strip()=='100644 blob '+r['git_blob']+'\t'+r['path']
  return dict(**r,remote_actual_bytes_hash_blob_tree='pass')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(one,p['files']))
 assert not git('diff','--name-only','HEAD').strip();git('merge','--ff-only',p['commit']);assert unrelated()==read(D/'publication/UNRELATED_UNTRACKED_BEFORE.json');immutable()
 receipt=dict(status='pass',commit=p['commit'],tree=p['tree'],files=rows,count=len(rows),bytes=sum(x['bytes']for x in rows),both_refs_fresh_CAS=True,remote_actual_bytes_hash_blob_tree=True,immutable_original_predictions_results_and_registries=True,new_offline_GT_scoring=True,single_expert_joint_parameter_search_and_actual_OPD_updates=True)
 put(D/'publication/FINAL_GITHUB_RECEIPT.json',receipt);print(json.dumps({k:v for k,v in receipt.items()if k!='files'}))
if __name__=='__main__':
 if sys.argv[1]=='stage':print(json.dumps({k:v for k,v in stage().items()if k!='files'}))
 elif sys.argv[1]=='verify':verify()
