"""Finite R15 exporter. Preparation is not build permission; no network/upload."""
from pathlib import Path
import argparse,collections,csv,gzip,hashlib,io,json,os,re,sys,tokenize
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from export_projection import STUDY,ARMS,ROLES,ROW_FIELDS,project_result,numeric,label,ensure_public,normalized_private_key

CODE_FILES=(
 *('crossplatform/round15/'+n for n in ('scene_observation.py','joint_geometry.py','semantic_graph.py','semantic_runner.py','semantic_model_runner.py','assemble_direct.py','check_geometry_cpu.py','check_assembly_cpu.py','semantic_check_cpu.py','semantic_integration_cpu.py','semantic_model_check_cpu.py')),
 *('crossplatform/round15/'+n for n in ('semantic_runner_v2.py','semantic_model_runner_v2.py','semantic_prompt_v2_check_cpu.py','semantic_graph_v3.py','semantic_runner_v3.py','semantic_model_runner_v3.py','semantic_binding_v3_check_cpu.py')),
 'crossplatform/round15/semantic_graph_v4.py','crossplatform/round15/semantic_model_runner_v4.py',
 'research/round15/experiment/compose_score_release.py','research/round15/evaluation/audit_direct_unlabelled.py',
 'research/round15/association/audit_semantic_coverage_v4.py','research/round15/geometry/audit_scored204.py',
 'research/round15/evaluation/summarize_pipeline.py','research/round15/evaluation/render_postscore_private.py',
 'research/round15/evaluation/review_release_composer.py','research/round15/evaluation/check_unlabelled_audit_cpu.py',
 *('crossplatform/round13/'+n for n in ('association.py','association_runner.py','input_io.py','geometry_observation.py')),
 'crossplatform/round7/camera_projection.py','crossplatform/round7/regression_adapter.py','crossplatform/round7/experiment/runner_helpers.py',
 'crossplatform/round12/capture_source.py','crossplatform/round12/capture_masks.py','crossplatform/round9/run_diagnostic.py','crossplatform/round6/factory.py',
 'research/round12/evaluation/metrics.py','research/round5/audit/independent_core_v2.py','tools/obb_iou.py',
 *('research/round15/evaluation/'+n for n in ('score_direct.py','check_score_cpu.py','audit_captures.py','review_geometry.py','review_semantic.py','review_assembly.py','review_binding_v3.py','review_budget_v4.py')),
 *('research/round15/input/'+n for n in ('prepare_inputs.py','decode_content_inventory.py','prepare_engineering4_addendum.py')),
 *('research/round15/publication/'+n for n in ('export_projection.py','build_export.py','check_export_cpu.py','prepare_readiness.py')))
STATIC_FILES=('research/round6/UPSTREAM_COMPATIBILITY.patch','research/round15/evaluation/PROTOCOL_PROPOSAL.md','research/round15/evaluation/SCORER_INTERFACE.md','research/round15/evaluation/UNLABELLED_AUDITOR_INTERFACE.md')
REQUIRED_HISTORY={'source_bad_hash_CLI_zero_forward','semantic_engineering_v1_four_cases_twelve_calls_schema_failure','assembly_engineering_metadata_runtime_failure','semantic_engineering_v2_fixed_schema_twelve_calls','quote_binding_v3_pre_GT_addendum','metadata_runtime_correction_no_method_change','semantic_engineering_v3_zero_complete_chain_echo_warning','decode_budget_v4_addendum'}
REQUIRED_HISTORY|={'semantic_engineering_v4_actual_scope','historical_and_current_engineering_exposure'}
REQUIRED_AUDITS={'source_capture','semantic_actual','direct_actual','scoring_actual','final_scientific'}
REQUIRED_PROTOCOLS={'input','experiment_card','source_capture','mask_capture','semantic','assembly'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def resolve(p):
 p=Path(p);return p if p.is_absolute() else ROOT/p
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def artifact(x):
 if set(x)!={'path','sha256'}:raise ValueError('Explicit source file/hash required')
 p=resolve(x['path'])
 if sha(p)!=x['sha256']:raise ValueError('Source binding changed '+str(x['path']))
 return p
def pointer(v,parts):
 if not isinstance(parts,list):raise ValueError('Explicit pointer components only, no executable expressions')
 for k in parts:v=v[int(k)] if isinstance(v,list) else v[k]
 return v
def scalar_projection(entry):
 path=artifact(entry['source']);v=read(path);result={'original_sha256':sha(path)}
 for public,parts in entry['fields'].items():
  if normalized_private_key(public):raise ValueError('Private field requested')
  value=pointer(v,parts)
  result[public]=label(value) if isinstance(value,str) and public in ('state','status') else numeric(value)
 for parts,want in entry.get('assertions',[]):
  if pointer(v,parts)!=want:raise ValueError('Audit/history assertion failed')
 ensure_public(result);return result
def verify_seal(base,digest,state,n=None):
 base=resolve(base)
 if sha(base/'SEALED.json')!=digest:raise ValueError('Actual seal hash mismatch')
 s=read(base/'SEALED.json')
 if s['state']!=state or n is not None and s.get('n_samples',s.get('n_cases'))!=n:raise ValueError('Actual formal seal scope')
 for rel,h in s['files'].items():
  p=Path(rel)
  if p.is_absolute() or '..' in p.parts or sha(base/p)!=h:raise ValueError('Seal member mismatch')
 return base,s
def guard_ready(a):
 if sha(a.readiness)!=a.readiness_sha256:raise ValueError('Root readiness byte hash')
 q=read(a.readiness)
 if q.get('schema')!='R15_finite_export_readiness_v1' or q.get('build_authorized') is not True or q.get('public_study')!=STUDY:raise ValueError('No final R15 root build authorization')
 if q.get('builder_sha256')!=sha(__file__) or q.get('projection_helper_sha256')!=sha(HERE/'export_projection.py'):raise ValueError('Builder/helper not the reviewed versions')
 if q.get('all_scientific_outputs_final') is not True or q.get('no_pending_conditional_branch') is not True:raise ValueError('Pending scientific stage: reject build')
 spec=read(artifact(q['finite_spec']))
 if spec.get('state')!='FROZEN_R15_FINITE_PUBLIC_SOURCE_ALLOWLIST':raise ValueError('Preparation spec is not final publication source freeze')
 if spec['code_sha256']!={p:sha(ROOT/p) for p in CODE_FILES} or spec['static_sha256']!={p:sha(ROOT/p) for p in STATIC_FILES}:raise ValueError('Finite code allowlist changed; new explicit registration required')
 db,ds=verify_seal(q['direct_predictions']['directory'],q['direct_predictions']['seal_sha256'],state='sealed_R15_eight_direct_arms_before_GT',n=204)
 sb,ss=verify_seal(q['scored_results']['directory'],q['scored_results']['seal_sha256'],state='sealed_R15_postrelease_CPU_scoring',n=204)
 release=artifact(q['score_release']);rr=read(release)
 if rr.get('CPU_SCORE_RELEASE') is not True or rr.get('all_direct_predictions_and_helpers_sealed_before_GT') is not True:raise ValueError('No actual prelabel release')
 if ss['release_sha256']!=sha(release) or ss['prediction_seal_sha256']!=sha(db/'SEALED.json') or rr['prediction_seal_sha256']!=sha(db/'SEALED.json'):raise ValueError('Score/release/prediction mismatch')
 if ds['n_arms']!=8 or tuple(ds['arm_names'])!=ARMS:raise ValueError('Actual eight arm schema')
 if not set(ds['program_sha256'])<=set(CODE_FILES):raise ValueError('Actual producer code missing from finite public source closure')
 report=artifact(q['report']);decision=artifact(q['decision']);review=read(artifact(q['final_review']))
 if review.get('report_sha256')!=sha(report) or review.get('decision_sha256')!=sha(decision) or review.get('claims_supported') is not True or review.get('publication_scientific_ready') is not True:raise ValueError('Final independent report/decision review missing/stale')
 if set(q['audit_projections'])!=REQUIRED_AUDITS or not REQUIRED_HISTORY<=set(q['history_events']):raise ValueError('Required final audit or failed-attempt lineage absent')
 if set(q['protocol_projections'])!=REQUIRED_PROTOCOLS:raise ValueError('Finite actual protocol projections required')
 for name,e in q['audit_projections'].items():
  v=read(artifact(e['source']))
  if not isinstance(v.get('state'),str) or not v['state'].startswith('PASS') or e['fields'].get('state')!=['state']:raise ValueError('Actual PASS audit and state projection required '+name)
 if q['decision_projection']['source']!=q['decision']:raise ValueError('Decision scalar projection must use actual reviewed decision')
 # Every external scalar is bound before any task result or input manifest is parsed.
 for e in [*q['audit_projections'].values(),*q['history_events'].values(),*q['protocol_projections'].values(),q['coverage_projection'],q['progression_projection'],q['decision_projection']]:scalar_projection(e)
 casespath=artifact(q['case_registry']);manifestpath=artifact(q['input_registry']);artifact(q['source_input_registry'])
 if sha(casespath)!=ds['case_index_sha256'] or sha(manifestpath)!=ds['manifest_sha256']:raise ValueError('Anonymous join registries not the actual scored sources')
 return q,spec,sb,report,decision,casespath

def privacy_markers(q):
 markers=set()
 for key in ('input_registry','source_input_registry'):
  rows=read(artifact(q[key]))
  for r in rows:
   for k in ('id','scene','caption','imagepath','pointpath','metapath'):
    if isinstance(r.get(k),str) and len(r[k])>=4:markers.add(r[k])
 return sorted(markers,key=len,reverse=True)
def privacy_scan(path,markers):
 data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
 text=data.decode('utf-8')
 for m in markers:
  if m in text:raise ValueError('Actual private registry marker leaked in '+str(path))
 if path.suffix=='.py':
  # Only exact universal placeholder literals, never actual host directory values.
  for t in tokenize.generate_tokens(io.StringIO(text).readline):
   if t.type==tokenize.STRING and '/home/' in t.string:
    import ast
    value=ast.literal_eval(t.string)
    if value not in ('/home/','/home/private/model'):raise ValueError('Actual host path in code')
 else:
  if '/home/' in text or re.search(r'(?:[A-Za-z]:\\Users\\|file://)',text):raise ValueError('Host path outside portable code')
 if path.suffix=='.json':ensure_public(json.loads(text))
 if path.suffix not in ('.py','.md','.json','.patch','.gz'):raise ValueError('Nonallowlisted media/data/weight extension')
 return len(data)

def main(a):
 q,spec,sb,report,decision,casespath=guard_ready(a)
 if a.output.exists():raise FileExistsError(a.output)
 result=read(sb/'RESULTS_PRIVATE.json');cases=read(casespath)
 if result['n']!=204 or result['case_role_counts']!={ROLES[0]:64,ROLES[1]:140}:raise ValueError('Actual scored case denominators')
 scene_keys=sorted({(c['platform'],c['scene_PRIVATE']) for c in cases},key=lambda k:hashlib.sha256(('R15_public_scene_v1|'+k[0]+'|'+k[1]).encode()).hexdigest())
 scene_alias={k:k[0]+'_scene_'+str(1+sum(x[0]==k[0] for x in scene_keys[:i])).zfill(2) for i,k in enumerate(scene_keys)}
 rows,summary,tails=project_result(result,cases,scene_alias)
 if len(rows)!=1632 or collections.Counter((r['case_role'],r['platform']) for r in rows)!=collections.Counter({(ROLES[0],'drone'):256,(ROLES[0],'quad'):256,(ROLES[1],'drone'):512,(ROLES[1],'quad'):608}):raise ValueError('All204x8 explicit scalar row denominators')
 markers=privacy_markers(q);a.output.mkdir(parents=True);provenance=[]
 def put(rel,data,source=None,operation='new_explicit_projection'):
  p=a.output/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  provenance.append({'public_path':rel,'original_source_sha256':sha(source) if source else None,'exported_sha256':sha(p),'operation':operation})
 for rel in (*CODE_FILES,*STATIC_FILES):
  # Current finite sources have workspace-relative Path(__file__) roots. Any
  # actual host path aborts rather than an unreviewed semantic source rewrite.
  source=ROOT/rel;data=source.read_bytes();operation='byteexact_relative_workspace_source'
  if source.suffix=='.md' and str(ROOT.parent) in data.decode():
   data=data.decode().replace(str(ROOT),'/absolute/workspace').replace(str(ROOT.parent),'/absolute/local_home').encode();operation='documentation_workspace_and_runtime_placeholders_only'
  put(rel,data,source,operation)
 put('REPORT.md',report.read_bytes(),report,'byteexact_privacy_checked_report')
 decisionobj=read(decision)
 # Decision is never copied wholesale. Explicit scalar mapping is root-bound.
 decisionprojection=scalar_projection(q['decision_projection']);write(a.output/'results/DECISION_SCALARS.json',decisionprojection)
 for rel,obj in [('results/SCALAR_SUMMARY.json',summary),('results/SIGNED_TAILS.json',tails)]:write(a.output/rel,obj)
 raw=io.StringIO(newline='');w=csv.DictWriter(raw,fieldnames=ROW_FIELDS,lineterminator='\n');w.writeheader();w.writerows(rows)
 (a.output/'results').mkdir(exist_ok=True)
 with (a.output/'results/ALL204_EIGHT_ARMS_SCALARS.csv.gz').open('wb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(raw.getvalue().encode())
 for group,outdir in [('audit_projections','audits'),('history_events','history'),('protocol_projections','protocol')]:
  for name,e in q[group].items():write(a.output/outdir/(name+'.json'),scalar_projection(e))
 for name,key in [('NO_GT_COVERAGE','coverage_projection'),('PROGRESSION_GATE','progression_projection')]:write(a.output/'results'/(name+'.json'),scalar_projection(q[key]))
 write(a.output/'PROVENANCE.json',{'study':STUDY,'builder_sha256':sha(__file__),'readiness_sha256':sha(a.readiness),'source_scored_seal_sha256':sha(sb/'SEALED.json'),'original_report_sha256':sha(report),'original_decision_sha256':sha(decision),'source_code_and_report':provenance,'decision_projection_operation':'explicit_scalar_fields_only','row_projection_operation':'all204x8_whitelisted_error_IoU_status_scalars_and_five_digit_aliases; no real IDs/query/mask indices or original coordinates','old_scientific_sources_modified':False})
 (a.output/'README.md').write_text('# '+STUDY+'\n\nFinite code and direct-G scalar evidence. [Report](REPORT.md), [all204 eight-arm scalar rows](results/ALL204_EIGHT_ARMS_SCALARS.csv.gz), [summary including source-pool oracle](results/SCALAR_SUMMARY.json), [failure lineage](history/), [provenance](PROVENANCE.json).\n\nFresh64 is recent-content exclusion only; every target VAL content has historical model/GT exposure. Two fresh and two regression inputs were also fixed unlabelled engineering probes; the formal denominator counts each once. Regression140 is separate. Source is Waymo-only epoch100; helper instance masks and VLM hypotheses are fallible observations. Both fresh gates failed, so no conditional TTA or parameter update was executed. No weights, raw data, captions, masks, features, raw model answers, annotation coordinates, query/mask indices or PRIVATE originals are included. No public document establishes blind validation or certified identity.\n\nWorkspace-relative code preserves the original directory layout; use a new local workspace and re-register private inputs/asset paths, hashes and leases. Reconstruct the official external 3EED implementation at commit `843a901f5367a6be5cc484b97c6ecad462961dfa` and apply `research/round6/UPSTREAM_COMPATIBILITY.patch`; the full upstream vendor is not redistributed. Obtain the registered source checkpoint and generic helper assets separately. Frozen historical locks cannot be replayed against substituted paths without a new registration. Documents saying Prepared describe historical registration boundaries; the completed scientific state is the final REPORT and decision scalars. See `research/round15/evaluation/SCORER_INTERFACE.md` for the post-seal score boundary.\n')
 allfiles=sorted(p for p in a.output.rglob('*') if p.is_file())
 for p in allfiles:privacy_scan(p,markers)
 manifest={'study':STUDY,'builder_sha256':sha(__file__),'readiness_sha256':sha(a.readiness),'listed_count':len(allfiles),'listed_bytes':sum(p.stat().st_size for p in allfiles),'files':{str(p.relative_to(a.output)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in allfiles},'n_case_arm_scalar_rows':1632,'no_upload_or_remote_mutation':True}
 write(a.output/'MANIFEST.json',manifest);privacy_scan(a.output/'MANIFEST.json',markers)
 print(json.dumps({'state':'BUILT_LOCAL_ONLY_AWAIT_INDEPENDENT_ACTUAL_EXPORT_AUDIT','files_including_manifest':len(allfiles)+1,'manifest_sha256':sha(a.output/'MANIFEST.json')}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--readiness',type=Path,required=True);p.add_argument('--readiness-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
