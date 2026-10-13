"""Finite publication of negative qualification; never a formal-results claim."""
import ast,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_observation_response_projection_common_v1 import *
from scripts.publish_stvg_opd_actionability_audit_v1 import CHECKOUT,BRANCH,git,put,safe,private_attachment_path
from scripts.publish_stvg_opd_LN_state_intervention_v1 import metadata
from scripts.audit_stvg_opd_projection_qualification_public_v1 import run as public_audit

def source_proof():
 pins=dict(read(BASE/'RUNTIME.json')['pins'])
 for p in sorted(BASE.glob('*/REVISION_RUNTIME.json'),key=lambda p:read(p)['time']):pins.update(read(p)['pin_overrides'])
 pins.update(read(BASE/'decoder_derivative_root006/RUNTIME.json')['pins'])
 proof=[]
 for name,h in pins.items():
  assert sha(ROOT/name)==h;dest=CHECKOUT/name;assert dest.exists(),name
  if sha(dest)==h:proof.append(dict(path=name,status='exact_bytes',sha256=h));continue
  assert name=='scripts/stvg_opd_design_ablation_common_v1.py'
  a,b=[ast.parse(p.read_text()) for p in [ROOT/name,dest]]
  fun=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
  aa,bb=fun(a),fun(b);assert set(aa)==set(bb) and all(aa[n]==bb[n] for n in aa if n!='prepare')
  imp=lambda t:[ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.Import,ast.ImportFrom))]
  assert imp(a)==imp(b)
  proof.append(dict(path=name,status='all_used_science_functions_and_imports_AST_exact; historical_prepare_attachment_alias',private_sha256=h,public_sha256=sha(dest)))
 if not (BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json').exists():write(BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',dict(status='pass',scope='all active qualification/revision/root source exact bytes and original science AST; fixed decoder source exact',count=len(proof),records=proof,CPU_only=True,GT_read=False,time=time.time()))

def run(closing=False):
 verify();assert all(read(BASE/n)['status']=='pass' for n in ['ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_VISUAL_REVIEW.json','ROOT_INDEPENDENT_DECODER_DERIVATIVE_READBACK.json'])
 assert not read(BASE/'TECHNICAL_DECISION.json')['formal_eligible']
 assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
 subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
 base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
 assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
 unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
 def copy(path,rel=None):
  rel=str(path.relative_to(ROOT)) if rel is None else rel;data=path.read_bytes()
  if path.suffix=='.json':safe(read(path))
  elif path.suffix in ['.py','.md']:
   assert not private_attachment_path(data.decode())
   if path.suffix=='.py':ast.parse(data.decode())
  else:assert path.suffix=='.pdf'
  target=CHECKOUT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);assert sha(target)==sha(path);owned.add(rel)
 sources=SCIENCE+['scripts/audit_stvg_opd_observation_response_projection_qualification_final_v1.py',
  'scripts/finalize_stvg_opd_projection_qualification_scalars_v1.py','scripts/root_stvg_opd_projection_decoder_derivatives_v1.py',
  'scripts/report_stvg_opd_observation_response_projection_v1.py','scripts/audit_stvg_opd_projection_qualification_public_v1.py',
  'scripts/publish_stvg_opd_projection_qualification_v1.py','external/TA-STVG/models/grounding_model/query_decoder.py']
 for name in sources:copy(ROOT/name)
 source_proof()
 for name in ['QUALIFICATION_ROWS.json','QUALIFICATION_SUMMARY.json','TECHNICAL_STATISTICS.json','FINITE_DIFFERENCE_ROWS.json','INDEPENDENT_FD_READBACK.json','REPORT.md']:copy(OUT/name)
 if not closing:
  for name in ['CPU_CONTRACTS.json','RUNTIME.json','SOURCE_STATE_CUSTODY_READBACK.json','ARCHIVE_PRELOCK.json','QUALIFICATION_BARRIER.json','ROOT_QUALIFICATION.json',
   'ROOT_READBACK.json','ROOT_INDEPENDENT_DECODER_DERIVATIVE_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',
   'TECHNICAL_DECISION.json','DECODER_DERIVATIVE_SCOPE_SOURCE_READBACK.json','FIGURE_EXPORT.json','ROOT_VISUAL_REVIEW.json','CPU_COMPLETION.json']:
   path=BASE/name;put(OUT/'receipts'/name,dict(original_private_receipt_sha256=sha(path),projection=metadata(read(path))))
  for folder in ['proposal_state_reinsertion001','Adam_snapshot_storage002','revision_dispatch003','repeat_scope_readback004','CPU_qualification_readback005','CPU_completion_scalar_dispatch007']:
   for name in ['CAPTURE_RECEIPT.json','REVISION_RUNTIME.json','RESOLUTION.json']:
    path=BASE/folder/name
    if path.exists():put(OUT/'engineering'/folder/name,dict(original_private_receipt_sha256=sha(path),projection=metadata(read(path))))
  records=[]
  for r in read(BASE/'DESIGN_LOCK.json')['records']:
   records.append(dict(parent_id=r['index'],dataset=r['dataset'],config=r['config'],state='Inherited_order1',source_model_sha256=r['source_model_sha256'],pixel_sha256=r['pixel_sha256'],A_complete_fit_sha256=r['A_sha256'],qualified=r['index'] in read(BASE/'DESIGN_LOCK.json')['qualification_indices']))
  put(OUT/'ANONYMOUS_PROTOCOL_AND_INTEGRITY.json',dict(records=records,entering_state='Inherited_order1',datasets=['hc2','vidstg'],historically_exposed=True,
   qualification_queries_each=4,accepted_A_complete_replays=16,actual_A_replays_including_engineering=24,actual_fresh_captures_including_independent_root=32,
   candidate_local_proposals=144,FD_contexts=21,FD_checks=84,formal_fits_executed=0,GT_read=False,new_DINO_calls=0,production_promoted=False,
   exclusions='media/annotations/weights/raw cached inputs/full fits/parameter states/Jacobians/gradient and proposal vectors/conversation'))
  p=OUT/'figures/TECHNICAL_QUALIFICATION.pdf';copy(p)
 else:
  assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
  for name in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']:
   path=BASE/name;put(OUT/'closing'/name,dict(original_private_receipt_sha256=sha(path),projection=metadata(read(path))))
  for e in read(BASE/'PUBLIC_STAGE.json')['files']:assert sha(CHECKOUT/e['path'])==e['sha256'];owned.add(e['path'])
 for p in OUT.rglob('*.json'):
  if p.name not in ['SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json','PUBLIC_SCALAR_AUDIT.json']:copy(p)
 a=public_audit(CHECKOUT)
 if not closing:write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(a,actual_public_checkout=True,time=time.time()));write(OUT/'PUBLIC_SCALAR_AUDIT.json',a)
 else:assert a==read(OUT/'PUBLIC_SCALAR_AUDIT.json')
 copy(OUT/'PUBLIC_SCALAR_AUDIT.json')
 relative=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
 write(manifest,dict(status='pass',scope='bounded actual numerical qualification complete, failed FD/repeat and all anonymous local probes/costs/source/one PDF two viewed pages; no candidate formal or GT results',files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n!=relative}));copy(manifest,relative)
 entries=[]
 import hashlib
 for name in sorted(owned):
  p=CHECKOUT/name;data=p.read_bytes();entries.append(dict(path=name,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
 with tempfile.TemporaryDirectory(prefix='projection-qualification-index-') as folder:
  env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
  for e in entries:
   h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
  tree=git('write-tree',env=env)
 write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
  base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(x['bytes'] for x in entries),unrelated_untracked_preserved=unrelated,
  scope='current fixed technical qualification only; formal gate failed; no192 candidate fits or GT efficacy',time=time.time()))
 print(json.dumps(dict(status='reviewed_staged',file_count=len(entries),bytes=sum(x['bytes'] for x in entries),expected_tree=tree,base_commit=base)))

if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='closing')
