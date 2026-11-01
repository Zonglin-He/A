"""Explicit scalar/status-only local R39/R40 export; never copy raw bbox records."""
import json,hashlib,io,csv,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'research/round40';EXP=HERE/'experiment';OLD=ROOT/'research/round39';PREFIX='studies/3eed_native_bbox_mask_association_diagnostic_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def safe(s):return s.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_MODEL_DIR','LOCAL_MODEL_DIR').replace('LOCAL_HOME','LOCAL_HOME')
def protocol(x):
 if isinstance(x,dict):return {k:protocol(v) for k,v in x.items() if k not in ['model_dir','files']}
 if isinstance(x,list):return [protocol(v) for v in x]
 if isinstance(x,str):return safe(x)
 return x
def main():
 assert read(EXP/'COMBINED_DECISION.json')['run']=='completed';assert read(OLD/'independent_audit/AUDIT_RESULT.json')['state']=='PASS_R39_ORIGINAL_CENTER_PIPELINE';assert read(HERE/'independent_audit/AUDIT_RESULT.json')['state']=='PASS_R40_CACHED_RECTANGLE_ASSIGNMENT'
 pub=HERE/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(p,dst):q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(safe(Path(p).read_text()))
 def cpjson(p,dst):dump(out/dst,protocol(read(p)))
 for folder in [ROOT/'crossplatform/round39',ROOT/'crossplatform/round40',OLD/'independent_audit',HERE/'independent_audit']:
  for p in sorted(folder.glob('*.py')):cp(p,p.relative_to(ROOT))
 for p in [HERE/'finalize40_combined.py',HERE/'prepare_identity40_public_summary.py',Path(__file__),OLD/'experiment/register39.py',OLD/'independent_identity/join39.py']:cp(p,p.relative_to(ROOT))
 cp(EXP/'COMBINED_REPORT.md','REPORT.md');cp(OLD/'experiment/REPORT.md','history/R39_ORIGINAL_CENTER_REPORT.md');cp(EXP/'REPORT.md','history/R40_ORIGINAL_RECTANGLE_REPORT.md')
 for n in ['PLAN_LOCK','ATTEMPT002_IMPLEMENTATION_LOCK','TECHNICAL_FIX_BUDGET_AUTHORIZATION']:cpjson(OLD/'experiment'/(n+'.json'),Path('protocol/R39')/(n+'.json'))
 for n in ['PLAN_LOCK','IMPLEMENTATION_LOCK']:cpjson(EXP/(n+'.json'),Path('protocol/R40')/(n+'.json'))
 for tag,d in [('R39',OLD),('R40',HERE)]:
  for n in ['TASK_SUMMARY','ALL_NEGATIVES','COVERAGE_UNKNOWN','SCENE_ROWS','FACTOR_CONTRASTS','COSTS','DECISION']:cpjson(d/'experiment'/(n+'.json'),Path('results')/tag/(n+'.json'))
  for n in ['CPU_TECH_CHECKS','CPU_AUDIT']:cpjson(d/'experiment'/(n+'.json'),Path('audit')/tag/(n+'.json'))
  cpjson(d/'independent_audit/AUDIT_RESULT.json',Path('audit')/tag/'INDEPENDENT_AUDIT_RESULT.json')
 cpjson(EXP/'COMBINED_DECISION.json','results/COMBINED_DECISION.json');cpjson(EXP/'MECHANISM_SUBGROUPS.json','results/R40/MECHANISM_SUBGROUPS.json');cpjson(EXP/'IDENTITY_PUBLIC_SUMMARY.json','identity/IDENTITY_PUBLIC_SUMMARY.json');cpjson(OLD/'independent_identity/AUDIT_RESULT.json','identity/R39_IDENTITY_AUDIT_RESULT.json')
 cpjson(OLD/'independent_audit/ALL208_CALL_SCALARS.json','results/R39/ALL208_CALL_SCALARS.json');cpjson(HERE/'independent_audit/ALL208_ASSIGNMENT_SCALAR_FLAGS.json','results/R40/ALL208_ASSIGNMENT_SCALAR_FLAGS.json')
 # Only own source prose metadata and official URLs, never original notebook cells.
 official=read(OLD/'experiment/OFFICIAL_EXCERPTS_SNAPSHOT.json');allowed=['url','commit','raw_sha256','schema_shape','coordinate_contract_summary','official_model_reference'];dump(out/'SOURCES.json',{k:official[k] for k in allowed}|{'raw_notebook_cells_prompts_or_thirdparty_source_not_redistributed':True,'read_only_evidence_not_executed':True,'original_excerpt_container_sha256':sha(OLD/'experiment/OFFICIAL_EXCERPTS_SNAPSHOT.json')})
 failure=list((ROOT/'crossplatform/round39/inference208_attempt001').glob('FAILURE*.json'));assert len(failure)==1;f=read(failure[0]);fields=['state','phase','type','message','newcalls','automatic_answer_retry'];dump(out/'history/ORIGINAL001_POSTGENERATION_FAILURE.json',{k:f[k] for k in fields}|{'actual_generation_count':1,'raw_answer_lost':True,'tokens_and_time_unknown':True,'newcalls_zero_is_function_return_counter_not_actual_generation_zero':True})
 cp(OLD/'experiment/RUN_TECHNICAL_HISTORY.md','history/R39_TECHNICAL_HISTORY.md');stage=read(OLD/'experiment/CPU_ATOMIC_STAGE_CONTROL.json');assert stage['not_executed'];dump(out/'audit/R39/ATOMIC_STAGE_CONTROL_SUMMARY.json',{'synthetic_literal_roundtrip_saved':True,'no_actual_native_answer_or_coordinates_exported':True,'literal_never_executed':True,'stage_before_parse_imagehash_actual208_verified':True})
 sets={'R39':read(OLD/'experiment/ALL_ROWS_PRIVATE.json'),'R40':read(EXP/'ALL_ROWS_PRIVATE.json')};allrows=sum(sets.values(),[]);scenes={p:sorted({r['scene_PRIVATE'] for r in allrows if r['platform']==p}) for p in ['drone','quad']};counts={};statuses={}
 for tag,rr in sets.items():
  scalar=[];flags=[]
  for r in rr:
   for a,v in r['IoU'].items():scalar.append({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'arm':a,'IoU':v,'query':r['queries'][a],'query_changed_vs_source':r['queries'][a]!=r['queries']['source']})
   for a,z in r['readouts'].items():
    keys=['mapped','status','group_size','query_changed','old_subset_contains_selected_mask','full_group_nonempty'];rec={'alias':r['alias'],'platform':r['platform'],'arm':a,**{k:z[k] for k in keys}}
    for k in ['center_is_identity_probe_not_3D_center','assignment_status','no_second_best_if_3D_group_empty']:
     if k in z:rec[k]=z[k]
    flags.append(rec)
  b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(scalar[0]),lineterminator='\n');w.writeheader();w.writerows(scalar);(out/'results'/tag/'ALL_TASK_SCALARS.csv').write_text(b.getvalue());counts[tag]=len(scalar);statuses[tag]=len(flags);dump(out/'results'/tag/'ALL_READOUT_STATUS_FLAGS.json',flags)
 assert counts=={'R39':2704,'R40':3120} and statuses=={'R39':2080,'R40':2496}
 budget=[]
 for label,filename in [('failed001','GENERATION_BUDGET.jsonl'),('successful002','GENERATION_BUDGET_ATTEMPT002.jsonl')]:
  rows=[json.loads(t) for t in (OLD/'experiment'/filename).read_text().splitlines()];allowed=['attempt_index','alias','interface','phase','plan_sha256','started_unix'];budget.extend({'attempt':label,**{k:r[k] for k in allowed if k in r},'raw_lost':label=='failed001'} for r in rows)
 assert len(budget)==209;dump(out/'results/ALL209_ACTUAL_GENERATION_LEDGER.json',budget)
 runtime=[]
 for rel in ['crossplatform/round39/inference208_attempt002','crossplatform/round40/predictions208_attempt001']:
  d=read(ROOT/rel/'SEALED.json');runtime.append({'attempt':rel,**{k:v for k,v in d.items() if k!='files'}})
 dump(out/'results/SEALED_RUNTIME_METADATA.json',runtime)
 prior=ROOT/'research/round38/publication/RECEIPT_FINAL.json';receipt=read(prior) if prior.exists() else {};dump(out/'DEPENDENCIES.json',{'prior_R37_R38_commit':receipt.get('commit'),'prior_R37_R38_prefix':'studies/3eed_native_grounding_complete_mapping_schema_compatibility_20261102','prior_R37_R38_export_payload_sha256':sha(ROOT/'research/round38/publication/TREE_PAYLOAD_LOCAL.json'),'prior_failure_histories_and_immutable24_annotations_retained_in_priorstudy':True,'source':'same Waymo-only checkpoint and frozen localQwen2.5VL3B','licensed_private_inputs_and_raw_answers_required':True,'R39_actual_generations':209,'R40_new_generations':0,'no_thirdparty_source_or_notebook_snapshots':True,'no_U_T_FINAL_or_7B_download_touched':True})
 (out/'README.md').write_text('''# R39/R40 — Native bbox and standard rectangle-mask association diagnostic

Read REPORT.md, both original decisions and COMBINED_DECISION. R39 center bbox
20.733712, R40 rectangleIoU21.038120 mIoU vs point22.740349, legacyG24.727373,
source19.663563. RectIoU improves center+.304408pp (4good3bad), yet both bbox
configurations remain below point/legacy and belowsource onDrone. No TTA/default
adoption. SameA208104eachplatform exposeddevelopment,2Drone/3Quad historyscenes,
not independentconfirmation. Do not rescue globalfailure with tiny identitysubset.

R39 is new single-image nativebbox output; center is only a2Didentityprobe. R40
is standard cached rectangleIoU against ALL detectorrectangles, not a stronger
bboxmodel/newalgorithm. Unique positive max; ties/allzero/invalidinventoryUNKNOWN.
Select before checking3Dgroup;26emptygroup winners retainedUNKNOWN, no secondbest.
Rawbboxvalues/grid/units/schema/masks/source3Dboxes/scores/representatives fixed.
Old11/R39 and13/R40 controls query/boxes exact. All2704+3120taskscalars and1733+
2469negative comparisons retained; duplicatecachedarms are not new observations.
Unknowns and allreadoutstatusflags explicit, noGTfeedbackfilter or hiddenpositive.

R39actual209=one001postgeneration NameError rawlost+208successful002. Initial
localnewcalls0 is a returncounter, not zeroactualgeneration. Originalfailure,
ledger and preGTtechnicalimport/rawfirstatomicrepair+budget209 authorization kept.
Lost answer tokens/time unknown; no fabricated costs or availableanswer selection.
208atomicrawstage/call answers exact;12preflight included208,fullphase196new.
R40newVLM/GPU/TTA0, original209costs remain. Stage durability proven for this run,
not arbitrary powerfailure guarantee. OriginalR37pointunit1/2and1/8diagnostics,
loader0call anddocreference fixes remain in precedingstudy, not erased bybbox.

Immutable24provisionalidentityhypotheses:18clear/6ambiguous, historicalartifact
exposure/nonexpert/notgold. R39center10matches7contradicts1unknown;11centerhits
include1overlapunknown, not11correct. Full10/10vsoldsubset5/10 also admits6wrong
identities. Smallsubsethigherthanpoint does not overturn wholeA208comparison.
Identity results never alter annotations/parser/method or filter supervision.

IndependentCPU audits13arms2704/15arms3120values, allnegatives/costs/coverage/
scenes/contrasts/subgroups. AllmaskIoU vectors maxerror0; rawstage invariance and
seal-before-score audited. Exacttie/allzero not observed inallA, fixtures tested.
No new3Dbox/regression/sourcelearning/studentupdate. Officialcookbookuses7B;
local3B/custom3EEDprompt is not official7BREC orRefCOCOreproduction. Modelhash and
R37geometry evidence reused byhash, not newGPUweight inspection or 7Bloading.

Private recovery/rawcase objects NEVER serialized generically: explicit fields
limit publication to anonymousaliases,scalars,flags,statuses,IoU,counts andowncode.
No actualbbox/pointvalues/nativeanswers/labels/captions/maskor3Dcoordinates/rawIDs/
real scenes or paths/media/features/weights/thirdparty notebook snapshots exported.
Private licensed assets/prior data bindings needed to reproduce originalseals.
''')
 # Structured private data above use explicit allowlists. Final exact-token scan
 # uses only already-exposed528manifest, never U/T/FINAL or7B assets.
 history=read(ROOT/'research/round27/input/ALL528_PRIVATE.json');forbidden={str(r[k]) for r in history for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)};findings=[];files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 for p in files:
  text=p.read_text()
  for token in forbidden:
   if token in text:findings.append({'file':p.relative_to(out).as_posix(),'type':'private_token','token_sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',text):findings.append({'file':p.relative_to(out).as_posix(),'type':'home_path'})
 assert not findings,findings[:10]
 blocked={'raw_answer_PRIVATE','raw_unique_position','new_value','old_accepted_value','original_point','bbox_center_resized','original_bbox_xyxy','bbox_original_xyxy','bbox_resized_unchanged','mask_IoU','mask_index','mask_indices','source_boxes','boxes','masks_bool','image_size_wh','caption','label','ignored_label_PRIVATE','raw_file','imagepath','pointpath','metapath','id_PRIVATE','scene_PRIVATE','scene'};keyfind=[]
 def walk(x,path):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in blocked:keyfind.append({'file':path,'key':k})
    walk(v,path)
  elif isinstance(x,list):
   for v in x:walk(v,path)
 for p in out.rglob('*.json'):walk(read(p),p.relative_to(out).as_posix())
 assert not keyfind,keyfind[:10]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','data_policy':'explicit scalar/status/flag allowlists; no rawcase/recovery/assignment/identity generic serialization','history_rows_scanned':528,'exact_private_token_findings':findings,'structured_private_field_findings':keyfind,'task_rows':counts,'all_readout_status_rows':statuses,'negative_comparisons':{'R39':1733,'R40':2469},'actual_generations_ledger':209,'no_actual_positions_answers_labels_captions_coords_rawIDs_scenes_paths_media_features_weights':True,'no_U_T_FINAL_or_7B_assets_read':True})
 code=sorted(out.rglob('*.py'));dump(out/'protocol/FINAL_CODE_SNAPSHOT.json',{'own_code_only':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in code]})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True,'private_payload_excluded':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];payload=[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files];target=pub/'TREE_PAYLOAD_LOCAL.json';dump(target,payload)
 for x,p in zip(payload,files):assert x['content']==p.read_text() and x['path']==PREFIX+'/'+p.relative_to(out).as_posix()
 for x in read(out/'MANIFEST.json')['files']:
  p=out/x['path'];assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
 result={'state':'PASS','n':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':target.stat().st_size,'payload_sha256':sha(target),'prefix':PREFIX,'manifest_payload_and_all_filehashes_exact':True,'private_token_and_structured_field_findings':[],'all5824taskscalars':True,'all4202negative_comparisons':True,'all209actual_generations_counted':True,'identity_R39_sealed_annotations_unchanged':True,'no_new_GPU_VLM_optimizer_FINAL_or_7B':True,'not_published':True};dump(pub/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
