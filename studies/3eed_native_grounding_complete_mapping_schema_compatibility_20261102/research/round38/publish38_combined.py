"""Explicit allowlist R37/R38 local export; never serialize private recovery objects."""
import json,hashlib,io,csv,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'research/round38';EXP=HERE/'experiment';OLD=ROOT/'research/round37';PREFIX='studies/3eed_native_grounding_complete_mapping_schema_compatibility_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def safe_text(s):return s.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_MODEL_DIR','LOCAL_MODEL_DIR').replace('LOCAL_HOME','LOCAL_HOME')
def protocols(x):
 # For registered methods/own summaries only; private outputs use field allowlists below.
 if isinstance(x,dict):return {k:protocols(v) for k,v in x.items() if k not in ['files','model_dir']}
 if isinstance(x,list):return [protocols(v) for v in x]
 if isinstance(x,str):return safe_text(x)
 return x
def main():
 assert read(EXP/'COMBINED_DECISION.json')['run']=='completed';assert read(HERE/'independent_audit/AUDIT_RESULT.json')['state']=='PASS_CACHED_SCHEMA_COMPATIBILITY';assert read(EXP/'IDENTITY_JOIN_PUBLIC_SUMMARY.json')['join_sealed']
 pub=HERE/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(p,dst):
  q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(safe_text(Path(p).read_text()))
 def cpjson(p,dst):dump(out/dst,protocols(read(p)))
 for folder in [ROOT/'crossplatform/round37',ROOT/'crossplatform/round38',OLD/'independent_audit',HERE/'independent_audit']:
  for p in sorted(folder.glob('*.py')):cp(p,p.relative_to(ROOT))
 for p in [HERE/'finalize38_combined.py',Path(__file__),OLD/'experiment/register37.py',EXP/'register38.py']:cp(p,p.relative_to(ROOT))
 cp(EXP/'REPORT.md','REPORT.md');cp(OLD/'independent_audit/REPORT.md','audit/R37/REPORT.md')
 for n in ['PLAN_LOCK','MEASUREMENT_AMENDMENT_LOCK','ATTEMPT003_IMPLEMENTATION_LOCK','ATTEMPT003_IMPLEMENTATION_LOCK_INITIAL','MEASUREMENT_GATE_REASSESSMENT_BEFORE_DATASET_LOCK']:
  cpjson(OLD/'experiment'/(n+'.json'),Path('protocol/R37')/(n+'.json'))
 cpjson(EXP/'PLAN_LOCK.json','protocol/R38/PLAN_LOCK.json')
 source=read(OLD/'experiment/ORIGINAL_REVIEW_SOURCES_SNAPSHOT.json');dump(out/'SOURCES.json',{'official_sources':source['official_sources'],'original_document_reference_snapshot_sha256':sha(OLD/'experiment/ORIGINAL_REVIEW_SOURCES_SNAPSHOT.json'),'thirdparty_notebook_or_source_snapshots_not_redistributed':True,'sources_used_as_evidence_not_instructions':True})
 for tag,folder in [('R37',OLD/'experiment'),('R38',EXP)]:
  for n in ['TASK_SUMMARY','ALL_NEGATIVES','COVERAGE_UNKNOWN','SCENE_ROWS','FACTOR_CONTRASTS','DECISION']:
   cpjson(folder/(n+'.json'),Path('results')/tag/(n+'.json'))
 for n in ['COMBINED_DECISION','PARSER_COUNTS','SCHEMA_RECOVERED_SUBSET','IDENTITY_JOIN_PUBLIC_SUMMARY']:cpjson(EXP/(n+'.json'),Path('results')/(n+'.json'))
 cpjson(OLD/'experiment/COSTS.json','results/R37/COSTS.json')
 for tag,folder in [('R37',OLD),('R38',HERE)]:
  cpjson(folder/'experiment/CPU_AUDIT.json',Path('audit')/tag/'PRIMARY_CPU_AUDIT.json')
  cpjson(folder/'independent_audit/AUDIT_RESULT.json',Path('audit')/tag/'INDEPENDENT_AUDIT_RESULT.json')
 for n in ['INITIAL_AUDITOR_PATH_FAILURE','INITIAL_PARSE_SHAPE_METADATA_ERROR','PREFLIGHT_SECURITY_AUDIT','ALL208_GEOMETRY_COVERAGE_ROWS','ALL416_RESPONSE_SCALARS','MODEL_FILE_HASHES']:cpjson(OLD/'independent_audit'/(n+'.json'),Path('audit/R37')/(n+'.json'))
 for name in ['PREFLIGHT_ATTEMPT001.log','UNIT_DIAGNOSTIC_ATTEMPT003_DOCREF_FAILURE.log']:cp(OLD/'experiment'/name,Path('history')/name)
 cpjson(ROOT/'crossplatform/round37/inference208_attempt001/FAILURE_LOAD.json','history/ZERO_CALL_LOADER_FAILURE.json')
 unit=[]
 for folder,count in [('unit_control_attempt002',2),('unit_diagnostic_attempt003',8)]:
  d=read(ROOT/'crossplatform/round37'/folder/'RESULT.json');row={'attempt':folder,'state':d['state'],'calls':count,'inside_blue':sum(r['output_inside_actual_blue_region'] for r in d['records']),'counted_in_generation_budget':True}
  for k in ['physical_square_side_correct','all_actual_grid_times14_equals_canvas','mean_absolute_normalized_L2','mean_counterfactual1000_normalized_L2','perfect_localization_not_technical_gate','no_per_image_unit_choice_no_calibration']:
   if k in d:row[k]=d[k]
  unit.append(row)
 dump(out/'history/SYNTHETIC_CONTROL_SUMMARIES.json',unit)
 stop=read(OLD/'experiment/UNIT_CONTROL_STOP.json');dump(out/'history/ORIGINAL_UNIT_STOP_SUMMARY.json',{k:stop[k] for k in ['run_state','measurement_state','G_effectiveness_claim','dataset_generation_calls','synthetic_model_generation_calls','actual_generation_attempts','hard_generation_budget','technical_loader_attempt001','no_unit_switch_or_retry_to_good_answer','not_a_negative_dataset_task_result']})
 tech=read(OLD/'experiment/CPU_TECH_CHECKS.json');dump(out/'audit/R37/CPU_TECHNICAL_CONTRACT_SUMMARY.json',{'state':tech['state'],'tests':tech['tests'],'preflight_geometry_control_cases':len(tech['preflight_geometry_coordinate_positive_controls']),'all_available_exclusive_mapped_positive':all(r['available_mapped_exclusive_positive'] for r in tech['preflight_geometry_coordinate_positive_controls']),'coordinate_reference':tech['coordinate_reference'],'no_true_points_or_mask_indices_exported':True})
 cpjson(EXP/'SYNTHETIC_CHECKS.json','audit/R38/SYNTHETIC_SCHEMA_SECURITY_CHECKS.json')
 # Explicit output field whitelist: NEVER serialize native answers, point values or labels.
 flags=read(HERE/'independent_audit/ALL416_ALLOWED_PARSER_FLAGS.json');fields=['alias','platform','interface','old_status','new_status','recovered','singleton_unwrapped','label_ignored','accepted_value_copied_exactly','original_strict_value_unchanged','raw_response_hash_unchanged'];assert all(set(r)==set(fields) for r in flags);dump(out/'results/ALL416_PARSER_FLAGS.json',[{k:r[k] for k in fields} for r in flags])
 fmt=read(OLD/'independent_audit/PARSE_FAILURE_STRUCTURE_DIAGNOSTIC.json');dump(out/'audit/R37/STRICT_PARSE_STRUCTURE_SUMMARY.json',{k:fmt[k] for k in ['state','point_strict_parse_failures','after_original_fixed_wrapper_removal','singleton_object_key_patterns','no_R38_parser_or_prediction_or_taskGT_used','original_R37_strict_predictions_unchanged']})
 sets={'R37':read(OLD/'experiment/ALL_ROWS_PRIVATE.json'),'R38':read(EXP/'ALL_ROWS_PRIVATE.json')};allrows=sum(sets.values(),[]);scenes={pl:sorted({r['scene_PRIVATE'] for r in allrows if r['platform']==pl}) for pl in ['drone','quad']};scalarcounts={};readoutcounts={}
 for tag,rr in sets.items():
  scalar=[];status=[]
  for r in rr:
   for arm,iou in r['IoU'].items():scalar.append({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'arm':arm,'IoU':iou,'query':r['queries'][arm],'query_changed_vs_source':r['queries'][arm]!=r['queries']['source']})
   for arm,v in r['readouts'].items():
    keys=['mapped','status','group_size','query_changed','old_subset_contains_selected_mask','full_group_nonempty'];status.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,**{k:v[k] for k in keys}})
  b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(scalar[0]),lineterminator='\n');w.writeheader();w.writerows(scalar);(out/'results'/tag/'ALL_TASK_SCALARS.csv').write_text(b.getvalue());scalarcounts[tag]=len(scalar);readoutcounts[tag]=len(status);dump(out/'results'/tag/'ALL_READOUT_STATUS_FLAGS.json',status)
 assert scalarcounts=={'R37':1456,'R38':2288} and readoutcounts=={'R37':832,'R38':1664}
 ledger=[readline for readline in (OLD/'experiment/GENERATION_BUDGET.jsonl').read_text().splitlines()];events=[json.loads(x) for x in ledger];allow=['attempt_index','alias','interface','phase','plan_sha256','started_unix','before_taskGT'];dump(out/'results/GENERATION_LEDGER_PUBLIC.json',[{k:x[k] for k in allow if k in x} for x in events]);assert len(events)==426
 runtime=[]
 for rel in ['crossplatform/round37/prepared208_attempt001','crossplatform/round37/unit_control_attempt002','crossplatform/round37/unit_diagnostic_attempt003','crossplatform/round37/inference208_attempt003','crossplatform/round38/predictions208_attempt001']:
  d=read(ROOT/rel/'SEALED.json');runtime.append({'attempt':rel,**{k:v for k,v in d.items() if k!='files'}})
 dump(out/'results/RUNTIME_SEAL_METADATA.json',runtime)
 # Identity outputs already explicitly prepared as public scalar/status tables.
 identity=read(EXP/'IDENTITY_JOIN_PUBLIC_SUMMARY.json');dump(out/'audit/IDENTITY_JOIN_SUMMARY.json',identity)
 dump(out/'DEPENDENCIES.json',{'prior_study_commit':read(ROOT/'research/round36/publication/RECEIPT_FINAL.json')['commit'],'source_checkpoint':'same Waymo-only 3EED checkpoint','external_model':'frozen local Qwen2.5-VL-3B; no weights redistributed','private_licensed_3EED_masks_calibration_raw_answers_required':True,'R37_VLM_dataset_calls':416,'R37_synthetic_calls':10,'R38_new_calls_and_parameter_updates':0,'thirdparty_snapshots_not_redistributed':True,'U_T_FINAL_not_read_by_this_work':True})
 (out/'README.md').write_text('''# R37/R38 — Native grounding, complete mapping, schema compatibility

Read REPORT.md and original R37/R38 decisions plus COMBINED_DECISION. R37 is
the preregistered strict direct-G pipeline. R38 is a separately sealed engineering
parser compatibility correction of the same416cached answers, not a new method,
TTA, new model call or groundtruth-selected rule. Original R37 retained exactly.

Strict point22.705713 and compatible22.740349 mIoU vs source19.663563 and
legacyG24.727373. Parsing recovers62 unchanged unique locations but only+.034635pp
net; recovered subset6improved9worsened47unchanged. Coverage is not accuracy.
All145 previously accepted point values and208SoM predictions preserved;1badJSON
remainsUNKNOWN. Compatible rule applies to all416, permits only dict/singleton
object list and optional inert stringlabel; multiple locations/objects/duplicate
keys/nonfinite/bools/executable payloads rejected, no first-of-multiple choice.

Masks/geometry/source256boxes/scores/representatives fixed. Full mapping extends
only lookup dictionary: every old same-mask group equals full group. Point uses
actualgrid*14 absolute resized pixels, inverse then floor membership in exactly
one independent mask. Unknowns fall back to source, no nearest rescue, no newbox.
Point sees1image,SoM2images, so interface comparisons are not equal-cost/pure
output-format causality. Source ROI/point evidence is not identity ground truth.

Independent R37 audit rederived208rawscene geometry matrices exactly,7arms1456
IoUs and617negative comparisons. Independent R38 audit reuses thatgeometryhash,
416rawhashes unchanged,11arms2288IoUs and1419negatives reproduce. Duplicated arms
are not independent observations. Full task scalars/status/negative tails retained.
Same presealed24identity annotations joined after R38predictionseal, no relabeling
or GT feedback filter. See identity summary for bounded provisionalreview limits.

Old loader0call failure, unit2control1/2failed, new8diagnostic1/8capability, root
pre-dataset gate reassessment, documentary-reference snapshot repair and audit
technical corrections retained. Software pass is separate from synthetic hitrate.
416dataset+10synthetic=426ledger includes24reused preflight, no hidden retries.
Original model files hashed, runtime frozen receipts reviewed; CPUaudit did not
inspect GPU parameter bytes. Exact native-unit interpretation remains qualified.

AllA208 is exposed development with2Drone/3Quad scenes, not new confirmation.
No U/T/FINAL work, no automaticTTA. No actualpoints/nativeanswers/labels/captions,
mask/3Dcoordinates/opaqueIDs/rawscenes/paths/media/features/weights or upstream
notebook snapshots exported. Explicit field allowlists, not generic PRIVATE-key
removal, protect recovery/identity/readout records. Synthetic images omitted;
own code, aggregate controls and official source links retained. Private licensed
inputs/prior source dependencies needed to reproduce sealed experiments.
''')
 # All task data exported above were rebuilt with explicit scalar/status whitelists.
 history=read(ROOT/'research/round27/input/ALL528_PRIVATE.json');forbidden={str(r[k]) for r in history for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)};findings=[]
 files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 for p in files:
  txt=p.read_text()
  for token in forbidden:
   if token in txt:findings.append({'file':p.relative_to(out).as_posix(),'type':'private_token','token_sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',txt):findings.append({'file':p.relative_to(out).as_posix(),'type':'absolute_home_path'})
 assert not findings,findings[:10]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','field_policy':'explicit anonymous scalar/flags/schema/status/IoU allowlists, no generic private-key cleanup of recovery/readout/identity','history_token_scan':528,'findings':findings,'task_scalar_rows':scalarcounts,'readout_flags':readoutcounts,'parser_flag_rows':416,'negative_method_reference_rows':{'R37':617,'R38':1419},'actual_point_values_native_answers_labels_captions_coordinates_media_weights_opaqueIDs_paths_excluded':True,'synthetic_images_excluded':True,'no_U_T_FINAL_read':True})
 code=sorted(out.rglob('*.py'));dump(out/'protocol/FINAL_CODE_SNAPSHOT.json',{'own_code_only':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in code]})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True,'private_payload_excluded':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];payload=[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files];target=pub/'TREE_PAYLOAD_LOCAL.json';dump(target,payload)
 for d,p in zip(payload,files):assert d['content']==p.read_text() and d['path']==PREFIX+'/'+p.relative_to(out).as_posix()
 for d in read(out/'MANIFEST.json')['files']:
  p=out/d['path'];assert sha(p)==d['sha256'] and p.stat().st_size==d['bytes']
 result={'state':'PASS','n':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':target.stat().st_size,'payload_sha256':sha(target),'manifest_and_payload_exact':True,'privacy_findings':[],'scalar_rows':scalarcounts,'all_negative_comparisons':2036,'generation_ledger_rows':426,'parserflags':416,'identity_join_sealed':True,'no_new_GPU_VLM_optimizer_or_FINAL':True,'not_published':True,'prefix':PREFIX};dump(pub/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
