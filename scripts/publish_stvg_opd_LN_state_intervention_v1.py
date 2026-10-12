"""Finite exact anonymous export; private tensors/media/text are never staged."""
import ast,gzip,hashlib,json,os,re,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_LN_state_intervention_common_v1 import BASE,OUT,read,write,sha,verify
from scripts.publish_stvg_opd_actionability_audit_v1 import git,CHECKOUT,BRANCH,put,safe,private_attachment_path
def metadata(value):
 if not isinstance(value,dict):return {}
 out={}
 for k,x in value.items():
  # Reject whole containers before considering their contents, including variants.
  if re.search('records|files|payload|settings|rows|summary|private|originals|process|command|prefix|bindings|provenance',k,re.I):continue
  if isinstance(x,bool):out[k]=x
  elif isinstance(x,str) and (k in ['status','scope','stage','dataset'] or re.fullmatch('[0-9a-f]{40,64}',x)):out[k]=x
  elif isinstance(x,(int,float)) and (k=='time' or re.search('count|total|done|fits|parents|rounds|coordinates|pairs|bytes|pages|PDFs|calls|checks|episodes|captures|scalar|tensors',k,re.I)):out[k]=x
  elif k in ['counts','checks','full_comparisons'] and isinstance(x,dict):out[k]=metadata(x)
 safe(out);return out
def run(closing=False):
 verify();assert read(BASE/'ROOT_READBACK.json')['status']==read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
 assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
 subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
 base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
 assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
 unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
 def copy(p,relative=None):
  relative=str(p.relative_to(ROOT)) if relative is None else relative;data=p.read_bytes()
  assert p.suffix in ['.py','.md','.json','.gz','.csv','.pdf']
  if p.suffix=='.json':safe(read(p))
  elif p.suffix=='.gz':
   with gzip.open(p,'rt') as f:
    for line in f:safe(json.loads(line))
  elif p.suffix in ['.py','.md']:
   assert not private_attachment_path(data.decode())
   if p.suffix=='.py':ast.parse(data.decode())
  q=CHECKOUT/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p);owned.add(relative)
 sources=['protocols/stvg_opd_LN_state_intervention_v1.md','scripts/stvg_opd_LN_state_intervention_common_v1.py','scripts/run_stvg_opd_LN_state_intervention_v1.py',
  'scripts/score_stvg_opd_LN_state_intervention_v1.py','scripts/audit_stvg_opd_LN_state_intervention_v1.py','scripts/report_stvg_opd_LN_state_intervention_v1.py',
  'scripts/audit_stvg_opd_LN_state_intervention_public_v1.py','scripts/publish_stvg_opd_LN_state_intervention_v1.py','scripts/close_stvg_opd_LN_state_intervention_v1.py']
 for n in sources:copy(ROOT/n)
 for n in ['ROWS.jsonl.gz','SUMMARY.json','REPORT.md','ALL_PARENT_THREE_ARM_READOUTS.csv','BEFORE_AFTER_PARENT_SUMMARY.json']:copy(OUT/n)
 if not closing:
  for n in ['RUNTIME.json','CPU_CONTRACTS.json','CPU_SCORING_CONTRACTS.json','SOURCE_STATE_CUSTODY_READBACK.json','SOURCE_CHECKPOINT_READBACK.json','BASELINE_STANDBY_IDENTITY.json',
   'QUALIFICATION_BARRIER.json','ROOT_QUALIFICATION.json','GLOBAL_PREDICTION_BARRIER.json','GT_EXPOSURE.json','CPU_COMPLETION.json',
   'ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','MECHANISM_DECISION.json','FIGURE_EXPORT.json','ROOT_VISUAL_REVIEW.json']:
   p=BASE/n;put(OUT/'receipts'/n,dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
  put(OUT/'SOURCE_CHECKPOINTS_PUBLIC.json',read(BASE/'SOURCE_CHECKPOINT_READBACK.json'))
  put(OUT/'DESIGN_PUBLIC.json',dict(status='complete_fixed_LN_state_intervention',targets=['hc2','vidstg'],parents_each=32,parents=64,arms=['Source','Inherited_order1','Inherited_order2'],
   new_formal_GPU_fits=192,actual_qualification_GPU_fits=24,actual_formal_rounds=4800,new_DINO_calls=0,history_exposed=True,
   baseline_dev_query_overlap={'hc2':0,'vidstg':0},Query_Residual_reset=True,Adam_reset=True,only_entering_LN_state_intervened=True,
   no_cross_query_state_writeback=True,GT_only_after_global_seal=True,primary='Omega0 real unvisited U',secondary='Omega0 and Omega1 union common U',
   bootstrap=10000,seed=20261013,gate_implemented=False,production_promoted=False,manuscript_written=False,remaining_formal_baseline_cancelled=True,
   raw_states_samples_private_GT_geometry_media_text_exported=False))
  for rel,b in read(BASE/'ROOT_VISUAL_REVIEW.json')['PDF_files'].items():
   p=ROOT/rel;assert sha(p)==b['sha256'] and p.stat().st_size==b['bytes'];copy(p)
 else:
  assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
  for n in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']:
   p=BASE/n;put(OUT/'closing'/n,dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
  for e in read(BASE/'PUBLIC_STAGE.json')['files']:assert sha(CHECKOUT/e['path'])==e['sha256'];owned.add(e['path'])
 for p in OUT.rglob('*.json'):
  if p.name not in ['SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json','PUBLIC_SCALAR_AUDIT.json']:copy(p)
 z=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_LN_state_intervention_public_v1.py'),str(CHECKOUT)],capture_output=True,text=True,
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
 assert z.returncode==0,z.stdout+z.stderr;a=json.loads(z.stdout);assert a['status']=='pass'
 if not closing:write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(a,actual_public_checkout=True,time=time.time()));write(OUT/'PUBLIC_SCALAR_AUDIT.json',a)
 else:assert a==read(OUT/'PUBLIC_SCALAR_AUDIT.json')
 copy(OUT/'PUBLIC_SCALAR_AUDIT.json')
 relative=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
 write(manifest,dict(status='pass',scope='exact complete controlled LN initialization/all64 parent three-arm scalars/positive-negative-neutral-unknown/two PDFs three actual viewed pages/source/closure; excludes self hash and all private witnesses',
  files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n!=relative}));copy(manifest,relative)
 entries=[]
 for n in sorted(owned):
  p=CHECKOUT/n;data=p.read_bytes();entries.append(dict(path=n,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
 with tempfile.TemporaryDirectory(prefix='LN-state-public-index-') as folder:
  env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
  for e in entries:
   h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
  tree=git('write-tree',env=env)
 write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,
  files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=unrelated,scope='one complete fixed historically exposed controlled LN state intervention; no new deployable gate/main promotion',time=time.time()))
 print(json.dumps(dict(status='reviewed_staged',closing=closing,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),base_commit=base,expected_tree=tree)))
if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='closing')
