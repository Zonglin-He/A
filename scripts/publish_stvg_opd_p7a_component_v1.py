"""Exact source/anonymous outcome export for the finite internal OPD factorial."""
import ast,csv,gzip,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p7a_component_common_v1 import *
from scripts.publish_stvg_opd_actionability_audit_v1 import CHECKOUT,BRANCH,git,put,safe,private_attachment_path
from scripts.publish_stvg_opd_LN_state_intervention_v1 import metadata

def run(closing=False):
 verify();standby()
 for n in ['ROOT_QUALIFICATION.json','ROOT_DEPLOYMENT_READBACK.json','ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_VISUAL_REVIEW.json']:
  assert read(BASE/n)['status']=='pass'
 assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
 subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
 base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
 assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
 unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
 def copy(p,rel=None):
  rel=str(p.relative_to(ROOT)) if rel is None else rel;data=p.read_bytes()
  assert p.suffix in ['.py','.md','.json','.gz','.csv','.pdf']
  if p.suffix=='.json':safe(read(p))
  elif p.suffix=='.gz':
   with gzip.open(p,'rt') as f:
    for line in f:safe(json.loads(line))
  elif p.suffix=='.csv':
   with p.open(newline='') as f:
    for row in csv.DictReader(f):safe(row)
  elif p.suffix in ['.py','.md']:
   assert not private_attachment_path(data.decode())
   if p.suffix=='.py':ast.parse(data.decode())
  q=CHECKOUT/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p);owned.add(rel)
 sources=SCIENCE+['scripts/report_stvg_opd_p7a_component_v1.py','scripts/audit_stvg_opd_p7a_component_public_v1.py',
  'scripts/report_stvg_opd_p7a_component_capture_scope001.py','scripts/audit_stvg_opd_p7a_component_capture_scope001.py',
  'scripts/publish_stvg_opd_p7a_component_v1.py','scripts/close_stvg_opd_p7a_component_v1.py']
 for n in sources:copy(ROOT/n)
 from scripts.publish_stvg_opd_projection_qualification_v1 import source_proof
 source_proof()  # inherited original helpers/prepare-only authorization alias
 proof=[]
 for n,h in read(BASE/'RUNTIME.json')['pins'].items():
  assert sha(ROOT/n)==h and sha(CHECKOUT/n)==h,n
  proof.append(dict(path=n,sha256=h,status='exact_bytes'))
 if not (BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json').exists():
  write(BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',dict(status='pass',scope='all original runtime science pins and all new reporting/publication source exact bytes; inherited used science checked separately',
   count=len(proof),hashes={n:sha(ROOT/n) for n in sources},records=proof,CPU_only=True,GT_read=False,time=time.time()))
 for n in ['ROWS.jsonl.gz','PARENT_ROWS.jsonl.gz','SUMMARY.json','REPORT.md','METRICS.csv','CONTRASTS.csv','COSTS.csv','FIT_ROWS.csv','PARENT_MEANS.csv']:copy(OUT/n)
 if not closing:
  names=['RUNTIME.json','CPU_CONTRACTS.json','SOURCE_STATE_CUSTODY_READBACK.json','BASELINE_STANDBY_IDENTITY.json','LAUNCH.json','AUXILIARY_SOURCE_FREEZE.json',
   'ARCHIVE_PRELOCK.json','ARCHIVE_PRELOCK_FINAL_CHECK.json','QUALIFICATION_BARRIER.json','ROOT_QUALIFICATION.json','GLOBAL_PREDICTION_BARRIER.json',
   'ROOT_DEPLOYMENT_READBACK.json','GT_EXPOSURE.json','CPU_COMPLETION.json','ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json',
   'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json','MECHANISM_DECISION.json','FIGURE_EXPORT.json','ROOT_VISUAL_REVIEW.json','CAPTURE_CALL_SCOPE_READBACK.json',
   'ROOT_REPORT_CSV_READBACK.json','PDF_AUTHORING_RUNTIME_RECEIPT.json']
  names+=['worker_exit/'+m+'.json' for m in ['qualification','qualification_root','formal','deployment_root','score']]
  names+=['presentation_capture_count001/'+m+'.json' for m in ['CAPTURE_RECEIPT','REVISION_RUNTIME','RESOLUTION']]
  names+=['public_receipt_projection002/'+m+'.json' for m in ['CAPTURE_RECEIPT','REVISION_RUNTIME']]
  names+=['public_receipt_projection002/attempt2/CAPTURE_RECEIPT.json','public_receipt_projection002/REVISION_RUNTIME003.json']
  for n in names:
   p=BASE/n;raw=read(p);projection=metadata(raw)
   if n=='RUNTIME.json':projection['pins']=raw['pins']
   if n=='ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json':projection['hashes']=raw['hashes']
   if n=='SOURCE_STATE_CUSTODY_READBACK.json':
    projection['unique_inputs']=raw['unique_inputs'];projection['parent_sources']=raw['parent_sources']
   if n=='CAPTURE_CALL_SCOPE_READBACK.json':
    for k in ['qualification_full_capture_invocations','formal_full_capture_invocations']:projection[k]=raw[k]
   if n in ['ROOT_QUALIFICATION.json','ROOT_DEPLOYMENT_READBACK.json']:
    assert all(isinstance(v,int) for v in raw['counts'].values());projection['counts']=raw['counts']
   if n in ['QUALIFICATION_BARRIER.json','GLOBAL_PREDICTION_BARRIER.json']:
    projection['original_joint_carry_full_controls']=raw['original_joint_carry_full_controls']
   if n in ['ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','CPU_COMPLETION.json']:
    assert all(x.startswith(str(OUT.relative_to(ROOT))) for x in raw['files']);projection['files']=raw['files']
    projection['rows']=raw['rows'];projection['parent_arm_rows']=raw['parent_arm_rows']
    if 'summary_reuses_declared_pure_function' in raw:
     assert type(raw['summary_reuses_declared_pure_function']) is bool
     projection['summary_reuses_declared_pure_function']=raw['summary_reuses_declared_pure_function']
   put(OUT/'receipts'/n,dict(original_private_receipt_sha256=sha(p),projection=projection))
  d=read(BASE/'DESIGN_LOCK.json');records=[]
  for r in d['records']:
   s=d['stages'][r['dataset']]
   records.append(dict(parent_id=r['parent_id'],dataset=r['dataset'],source=s['source'],source_model_sha256=r['source_model_sha256'],pixel_sha256=r['pixel_sha256'],
    frame_ids_sha256=r['frame_ids_sha256'],evidence_sha256=r['evidence_sha256'],
    old_complete_fit_sha256={o:x['sha256'] for o,x in r['originals'].items()},input_sha256=r['input']['input_sha256'],
    arrival={o:x['arrival'] for o,x in r['originals'].items()},
    configs={a:{('actions_per_round' if k=='samples' else k):v for k,v in c.items()} for a,c in s['configs'].items()}))
  put(OUT/'ANONYMOUS_PROTOCOL.json',dict(status='complete_finite_internal_component_experiment',records=records,arms=ARMS,parents=64,parents_each=32,
   orders=['order1','order2'],qualification_complete_GPU_fits=64,formal_complete_GPU_fits=512,qualification_rounds=1600,formal_rounds=12800,
   qualification_captures=16,formal_captures=128,old_joint_carry_controls=128,joint_active_parameters=1792,LN_active_parameters=1536,
   bootstrap=BOOT,seed=SEED,historically_exposed_development=True,independent_confirmation=False,new_DINO_calls=0,
   primary='I1 parent-macro mvIoU',secondary='I0 parent-macro mvIoU',Query_reset_each_arrival=True,Adam_reset_each_arrival=True,
   complete_own_history_each_arm=True,reset_absolute_source_LN=True,no_alternate_objective=True,parameter_search_executed=False,
   production_promoted=False,monitor_recreated=False,paper_main_experiments_executed=False,
   excluded_categories=['private media','annotations','weights','raw cached input','complete fits','parameter or gradient vectors','personal conversation']))
  for n,b in read(BASE/'ROOT_VISUAL_REVIEW.json')['PDF_files'].items():
   p=ROOT/n;assert sha(p)==b['sha256'] and p.stat().st_size==b['bytes'];copy(p)
 else:
  assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
  for n in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']:
   p=BASE/n;put(OUT/'closing'/n,dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
  for e in read(BASE/'PUBLIC_STAGE.json')['files']:assert sha(CHECKOUT/e['path'])==e['sha256'];owned.add(e['path'])
 for p in OUT.rglob('*.json'):
  if p.name not in ['SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json','PUBLIC_SCALAR_AUDIT.json']:copy(p)
 z=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_p7a_component_capture_scope001.py'),str(CHECKOUT)],capture_output=True,text=True,
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
 assert z.returncode==0,z.stdout+z.stderr;a=json.loads(z.stdout);assert a['status']=='pass'
 if not closing:write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(a,actual_public_checkout=True,time=time.time()));write(OUT/'PUBLIC_SCALAR_AUDIT.json',a)
 else:
  old=read(OUT/'PUBLIC_SCALAR_AUDIT.json');assert a['checks']==old['checks'] and a['files']==old['files']
 copy(OUT/'PUBLIC_SCALAR_AUDIT.json')
 rel=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
 write(manifest,dict(status='pass',scope='all512 actual complete formal fits/six contrasts/each order/256 parent-arm rows/all signs and unknowns/original OPD/source/config/actual viewed PDF; excludes self hash',
  files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n!=rel}));copy(manifest,rel)
 entries=[]
 for n in sorted(owned):
  p=CHECKOUT/n;data=p.read_bytes();entries.append(dict(path=n,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
 with tempfile.TemporaryDirectory(prefix='OPD-component-public-index-') as folder:
  env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
  for e in entries:
   h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
  tree=git('write-tree',env=env)
 write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
  base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=[n for n in unrelated if n not in owned],
  scope='finite fixed historically exposed development internal OPD scope x state lifetime experiment; no alternate objective, tuning, main launch or promotion',time=time.time()))
 print(json.dumps(dict(status='reviewed_staged',file_count=len(entries),bytes=sum(e['bytes'] for e in entries),expected_tree=tree,base_commit=base)))

if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='closing')
