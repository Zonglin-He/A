"""Finite privacy-reviewed anonymous full audit source/scalars/figures export."""
import ast,gzip,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_benefit_retention_common_v1 import *
from scripts.publish_stvg_opd_actionability_audit_v1 import CHECKOUT,BRANCH,git,put,safe,private_attachment_path
from scripts.publish_stvg_opd_LN_state_intervention_v1 import metadata
from scripts.audit_stvg_opd_state_benefit_retention_public_v1 import run as public_audit

def source_proof():
 proof=[];prior=read(old.BASE/'RUNTIME.json')
 for name,h in {**prior['pins'],**read(BASE/'RUNTIME.json')['pins']}.items():
  source=ROOT/name;dest=CHECKOUT/name;assert dest.exists() and sha(source)==h
  if sha(dest)==h:proof.append(dict(path=name,status='exact_bytes',sha256=h));continue
  assert name=='scripts/stvg_opd_design_ablation_common_v1.py'
  a=ast.parse(source.read_text());b=ast.parse(dest.read_text())
  def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  aa,bb=functions(a),functions(b);assert set(aa)==set(bb)
  names=[n for n in aa if n!='prepare'];assert all(aa[n]==bb[n] for n in names)
  imports=lambda tree:[ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
  assert imports(a)==imports(b)
  proof.append(dict(path=name,status='all_used_scientific_functions_and_imports_AST_exact_public_prepare_attachment_alias_only',functions=names,private_sha256=h,public_sha256=sha(dest)))
 if not (BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json').exists():
  write(BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',dict(status='pass',scope='all new audit source exact bytes and all prior actual science functions/imports; one historical public prepare privacy alias preserved',count=len(proof),records=proof,CPU_only=True,GT_read=False,time=time.time()))
 return proof
def run(closing=False):
 verify();assert all(read(BASE/n)['status']=='pass' for n in ['ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_VISUAL_REVIEW.json'])
 assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
 subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
 base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
 assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
 unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
 def copy(path,rel=None):
  rel=str(path.relative_to(ROOT)) if rel is None else rel;data=path.read_bytes();assert path.suffix in ['.py','.md','.json','.gz','.csv','.pdf']
  if path.suffix=='.json':safe(read(path))
  elif path.suffix=='.gz':
   with gzip.open(path,'rt') as f:
    for line in f:safe(json.loads(line))
  elif path.suffix in ['.py','.md']:
   assert not private_attachment_path(data.decode())
   if path.suffix=='.py':ast.parse(data.decode())
  target=CHECKOUT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);assert sha(target)==sha(path);owned.add(rel)
 sources=SCIENCE+['scripts/report_stvg_opd_state_benefit_retention_audit_v1.py',
  'scripts/audit_stvg_opd_state_benefit_retention_public_v1.py','scripts/publish_stvg_opd_state_benefit_retention_audit_v1.py',
  'scripts/close_stvg_opd_state_benefit_retention_audit_v1.py']
 for name in sources:copy(ROOT/name)
 source_proof()
 for name in ['PARENT_ROWS.jsonl.gz','ROUND_ROWS.jsonl.gz','FRAME_ROUND_ROWS.jsonl.gz','DENSE_ENDPOINT_ROWS.jsonl.gz','ALL_PARENT_CONTRASTS.csv','SUMMARY.json','REPORT.md']:copy(OUT/name)
 if not closing:
  for name in ['CPU_CONTRACTS.json','RUNTIME.json','ARCHIVE_PRELOCK.json','GLOBAL_READOUT_BARRIER.json','GT_EXPOSURE.json',
   'CPU_COMPLETION.json','ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',
   'MECHANISM_DECISION.json','FIGURE_EXPORT.json','ROOT_VISUAL_REVIEW.json']:
   path=BASE/name;put(OUT/'receipts'/name,dict(original_private_receipt_sha256=sha(path),projection=metadata(read(path))))
  put(OUT/'DESIGN_PUBLIC.json',dict(status='complete_cache_only_state_benefit_retention_audit',datasets=['hc2','vidstg'],parents_each=32,parents=64,
   arms=ARMS,original_sealed_fits=192,saved_updates=4800,arm_boundaries=4992,history_exposed=True,bootstrap=BOOT,seed=SEED,
   primary='genuine GT-event decoder frames excluding all Omega0 visits',secondary='common U and observed frames; all-frame output distance',
   initial_correct_IoU=.5,initial_advantage_pp=1,early_quarter_steps={'hc2':10,'vidstg':3},persistent_final_quarter_start={'hc2':30,'vidstg':8},
   new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,new_forward_calls=0,new_backward_calls=0,new_optimizer_steps=0,
   GT_only_offline_after_all_cached_readouts_sealed=True,no_GT_selected_step_or_parameter=True,previous_query_retention_tested=False,
   catastrophic_forgetting_established=False,production_promoted=False,remaining_formal_baseline_cancelled=True,manuscript_written=False,
   raw_GT_media_text_state_gradient_vectors_exported=False))
  records=[]
  for r in read(BASE/'DESIGN_LOCK.json')['records']:
   records.append(dict(parent_id=r['index'],dataset=r['dataset'],config=r['config'],source_model_sha256=r['source_model_sha256'],
    input_pixel_sha256=r['pixel_sha256'],matched_state_pair_sha256=r['pair_sha256'],
    sealed_fits_sha256={a:sha(old.BASE/'formal'/f"{r['index']:05}"/(a+'_repeat0.pt')) for a in ARMS}))
  write_rows(OUT/'ANONYMOUS_PROTOCOL_AND_INTEGRITY.jsonl.gz',records);copy(OUT/'ANONYMOUS_PROTOCOL_AND_INTEGRITY.jsonl.gz')
  for rel,b in read(BASE/'ROOT_VISUAL_REVIEW.json')['PDF_files'].items():
   path=ROOT/rel;assert sha(path)==b['sha256'] and path.stat().st_size==b['bytes'];copy(path)
 else:
  assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
  for name in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']:
   path=BASE/name;put(OUT/'closing'/name,dict(original_private_receipt_sha256=sha(path),projection=metadata(read(path))))
  for e in read(BASE/'PUBLIC_STAGE.json')['files']:assert sha(CHECKOUT/e['path'])==e['sha256'];owned.add(e['path'])
 for path in OUT.rglob('*.json'):
  if path.name not in ['SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json','PUBLIC_SCALAR_AUDIT.json']:copy(path)
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
 z=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_state_benefit_retention_public_v1.py'),str(CHECKOUT)],capture_output=True,text=True,env=env)
 assert z.returncode==0,z.stdout+z.stderr;a=json.loads(z.stdout);assert a['status']=='pass'
 if not closing:write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(a,actual_public_checkout=True,time=time.time()));write(OUT/'PUBLIC_SCALAR_AUDIT.json',a)
 else:assert a==read(OUT/'PUBLIC_SCALAR_AUDIT.json')
 copy(OUT/'PUBLIC_SCALAR_AUDIT.json')
 relative=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
 write(manifest,dict(status='pass',scope='complete cache-only retention audit/all anonymous parent-frame-round positive-negative-neutral-unknown scalars/all source/config/two PDFs four actual pages/closing; excludes self hash and all private vector/media witnesses',
  files={name:dict(sha256=sha(CHECKOUT/name),bytes=(CHECKOUT/name).stat().st_size) for name in sorted(owned) if name!=relative}));copy(manifest,relative)
 entries=[]
 for name in sorted(owned):
  path=CHECKOUT/name;data=path.read_bytes();entries.append(dict(path=name,bytes=len(data),sha256=sha(path),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
 with tempfile.TemporaryDirectory(prefix='state-retention-public-index-') as folder:
  env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
  for e in entries:
   h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
  tree=git('write-tree',env=env)
 write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,
  files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=unrelated,
  scope='all192 existing fits cache-only current-query state benefit retention audit; no preservation algorithm or main promotion',time=time.time()))
 print(json.dumps(dict(status='reviewed_staged',closing=closing,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),base_commit=base,expected_tree=tree)))
if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='closing')
