"""Finite R10 public export; preparation is not a completed experiment.
Actual build requires root final source bindings, all sealed results and audits.
No network, publication, GPU, model or new scoring. Never copies private data.
"""
from pathlib import Path,PurePosixPath
from functools import lru_cache
import argparse,csv,gzip,hashlib,io,json,math
B=Path(__file__).resolve().parents[1];E=B/'research/round10/experiment';I=B/'research/round10/input_prepare';V=B/'research/round10/evaluation'
ARMS=('source','full_rgb','no_rgb_coordinate','rgb_gated_pointmask_ablation')
REQUIRED=('source_seal','prediction_seal','scalar_summary','score_release','source_unlabelled_audit','geometry_unlabelled_audit','geometry_observability_supplement','scored_independent_audit','mechanism_gate_independent_review','root_mechanism_review','root_report','root_decision','root_final_review')
READINESS=('source_and_all4_geometry_arms_sealed_before_GT','score_and_independent_audits_complete','fixedq0_and_same_support_RGB_coordinate_control_verified','whole_observation_not_heldout_and_not_neural_TTA_explicit','root_final_report_decision_review_passed','privacy_no_media_captions_GTcoordinates_masks_features_weights_or_originalIDs')
CODE_FILES=tuple('crossplatform/round10/'+x for x in ('capture_source.py','run_geometry.py','visibility_geometry.py','rgb_support.py','check_visibility_geometry_cpu.py'))+tuple('research/round10/input_prepare/'+x for x in ('prepare_inputs.py','lock_gallery.py','check_rgb_support_cpu.py','render_gallery.py','prepare_active_gallery.py','render_active_gallery.py'))+tuple('research/round10/evaluation/'+x for x in ('metrics.py','score_geometry.py','audit_source_unlabelled.py','audit_geometry_unlabelled.py','audit_geometry_scored.py','audit_mechanism_gate.py','check_metrics_cpu.py'))+('crossplatform/round9/run_diagnostic.py','crossplatform/round9/fixed_q0_geometry.py','crossplatform/round9/head_exact_fit.py','crossplatform/round8/association_geometry.py','crossplatform/round8/semantic_encoder.py','crossplatform/round7/camera_projection.py','crossplatform/round7/regression_adapter.py','crossplatform/round7/experiment/runner_helpers.py','crossplatform/round6/factory.py','research/round5/audit/independent_core_v2.py','tools/obb_iou.py','tools/build_round10_export.py','research/round10/mechanism/reduce_mechanism.py','research/round10/mechanism/check_reducer_cpu.py')
# These are finite known preparations. Missing files are errors, not fake results.
STATIC_JSON=tuple('research/round10/input_prepare/'+x for x in ('INPUT_PLAN_LOCK.json','COHORT_AVAILABILITY.json','INPUT_RECEIPT.json','POINT_OBSERVATION_SPLIT_PLAN.json','RGB_SUPPORT_CPU_FIXTURE.json','RGB_SUPPORT_READY.json','GALLERY_PLAN_LOCK_v2.json','GALLERY_RENDERER_READY.json','history/gallery_alias_pre_review/GALLERY_PLAN_LOCK.json'))+('crossplatform/round10/CPU_VISIBILITY_GEOMETRY_FIXTURE.json','research/round10/experiment/active_gallery12/PLAN_LOCK.json','research/round10/experiment/SOURCE_CAPTURE_PLAN_LOCK.json','research/round10/experiment/GEOMETRY_PLAN_LOCK.json','research/round10/experiment/POSTSEAL_RELEASE_RECEIPT.json',)+tuple('research/round10/evaluation/'+x for x in ('CPU_EVALUATION_PREPARATION_RECEIPT.json','CPU_METRIC_FIXTURE.json','PREDICTION_CONTRACT_DRAFT.json','PROTOCOL_DRAFT.json','SOURCE_CAPTURE_IMPLEMENTATION_REVIEW.json','SOURCE_CAPTURE_IMPLEMENTATION_REVIEW_v2.json','EVAL_CODE_LOCK.json','G_IMPLEMENTATION_INDEPENDENT_REVIEW.json','AUDITOR_ENTRY_PREPARATION.json'))+tuple('research/round10/mechanism/'+x for x in ('CPU_FIXTURE.json','READY.json','EXECUTION_RECEIPT.json','ACTUAL160_AUTHOR_INTERPRETATION.json','actual160_attempt001/SUMMARY.json','actual160_attempt001/SEALED.json'))+('research/round10/experiment/PROGRESSION_GATE.json','research/round10/experiment/ROOT_GALLERY10_REVIEW.json')
DOC_FILES=('research/round10/input_prepare/RGB_SUPPORT_CONTRACT.md','research/round10/experiment/REPRODUCTION.md','research/round10/experiment/BRIEF.md','research/round10/experiment/EXPERIMENT_CARD.md','research/round10/evaluation/EXECUTION_DRAFT.md','research/round10/mechanism/README.md','research/round10/mechanism/ACTUAL160_AUTHOR_INTERPRETATION.md')
FORBIDDEN={'rows_PRIVATE','id','scene','frame','caption','captions','utterance','ground_info','bbox_3d','gt_bbox','positive_map','GT7D_model','target7','target6','source6','box6','box7','center3','size3','source_boxes_raw6','source_boxes_eval6','arm_boxes_raw6','arm_boxes_eval6','pre_constraint_geometry6','RGB_mask2d_private','points','sampled_points_model','raw_indices','center_features','size_features','image_intrinsic','image_extrinsic','camera_distortion','extrinsic','intrinsic','embeddings','proj_queries','proj_tokens','roi_xyxy','source_projected_tight_xyxy','mask2d','uv','depth','coordinate_residual6','coordinate_residual6_by_arm','abs_error6_m'}
DROP_MAPS={'files','source_cache_files','normal_source_cache_files','source_input_paths','weights_relative_private_location','programroot','workspace_root'}
sha=lambda x:hashlib.sha256(x).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(rel):
 q=PurePosixPath(rel);assert q.parts and not q.is_absolute() and '..' not in q.parts
 p=B/q.as_posix();assert p.resolve().is_relative_to(B.resolve()) and p.is_file(),rel;return p

def main(a):
 assert not a.output.exists(),'New output only; never overwrite an export'
 assert a.output.resolve().is_relative_to(B.resolve())
 ready=read(E/'PUBLICATION_READINESS.json')
 for key in READINESS:assert ready.get(key) is True,('Missing actual final gate',key)
 specs=read(E/'EXPORT_FINAL_SOURCES.json');bound={}
 for key in REQUIRED:
  spec=specs[key];assert set(spec)=={'path','sha256'};p=local(spec['path']);assert file_sha(p)==spec['sha256'];bound[key]=p
 source=read(bound['source_seal']);pred=read(bound['prediction_seal']);scores=read(bound['scalar_summary']);sspath=bound['scalar_summary'].parent/'SEALED.json';ss=read(sspath)
 assert source['state']=='sealed_R10_source_forward_and_input_observations_no_G_no_GT' and source['n_samples']==160 and source['source_unchanged'] and source['source_repeat_bitexact']
 assert pred['state']=='sealed_R10_fixedq0_direct_geometry_before_GT' and pred['n_samples']==160 and pred['n_arms']==4 and pred['all_queries_fixed_original_q0'] and pred['no_GT_or_parameter_update'] and pred['same_support_and_acceptance_full_noRGB']
 assert pred['source_seal_sha256']==file_sha(bound['source_seal']) and pred['source_checkpoint_sha256']==source['source_checkpoint_sha256'] and pred['source_model_state_fingerprint']==source['source_model_state_fingerprint']
 for seal,path in ((source,bound['source_seal']),(pred,bound['prediction_seal'])):
  for name,digest in seal['files'].items():
   q=PurePosixPath(name);assert not q.is_absolute() and '..' not in q.parts;assert file_sha(path.parent/name)==digest
 assert pred['program_sha256_base']=='workspace_root'
 for rel,digest in pred['program_sha256'].items():assert file_sha(local(rel))==digest
 assert scores['state']=='R10_direct_geometry_scored_after_complete_seal' and scores['n']==ss['n_samples']==160 and ss['prediction_seal_sha256']==file_sha(bound['prediction_seal'])
 for name,digest in ss['files'].items():assert file_sha(bound['scalar_summary'].parent/name)==digest
 grant=read(bound['score_release']);assert grant['CPU_SCORE_RELEASE'] and grant['all_source_and_geometry_arms_sealed_before_GT'] and grant['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and grant['source_capture_seal_sha256']==file_sha(bound['source_seal'])
 assert grant['scorer_program_sha256']==file_sha(V/'score_geometry.py') and grant['metrics_program_sha256']==file_sha(V/'metrics.py')
 lock=read(bound['scalar_summary'].parent/'SCORER_LOCK.json');assert lock['release_sha256']==file_sha(bound['score_release'])
 ua=read(bound['source_unlabelled_audit']);assert ua['state']=='independent_actual_R10_source160_observation_binding_audit_pass' and ua['source_seal_sha256']==file_sha(bound['source_seal']) and ua['n']==160 and ua['targetGT_annotations_scoringindex_camera_metadata_content_or_G_read'] is False
 # Actual audit schemas are final-source contracts, not readiness substitutes.
 ga=read(bound['geometry_unlabelled_audit']);ra=read(bound['scored_independent_audit'])
 assert ga['state']=='independent_actual_R10_geometry4_unlabelled_audit_pass' and ga['source_capture_seal_sha256']==file_sha(bound['source_seal'])
 assert ra['state']=='independent_R10_geometry4_scored_reduction_audit_pass'
 for audit in (ga,ra):assert audit['n']==160 and audit['prediction_seal_sha256']==file_sha(bound['prediction_seal'])
 for audit,script in ((ua,'audit_source_unlabelled.py'),(ga,'audit_geometry_unlabelled.py'),(ra,'audit_geometry_scored.py')):assert audit['auditor_sha256']==file_sha(V/script)
 obs=read(bound['geometry_observability_supplement']);assert obs['prediction_seal_sha256']==file_sha(bound['prediction_seal'])
 resultpath=bound['scalar_summary'].parent/'RESULTS_PRIVATE.json';assert ra['results_sha256']==file_sha(resultpath) and ra['scored_seal_sha256']==file_sha(sspath)
 ma=read(bound['mechanism_gate_independent_review']);assert ma['state']=='independent_R10_mechanism_aggregate_and_progression_gate_review_complete' and ma['n']==160 and ma['auditor_sha256']==file_sha(V/'audit_mechanism_gate.py')
 assert ma['main_scoring_audit_sha256']==file_sha(bound['scored_independent_audit']) and ma['results_sha256']==file_sha(resultpath) and ma['scored_seal_sha256']==file_sha(sspath) and ma['geometry_seal_sha256']==file_sha(bound['prediction_seal'])
 assert ma['all_mechanism_and_remaining_summary_signedtail_columns_reduced'] and ma['no_fresh_GT_rawmetadata_newIoU_GPU_model_or_G_reexecution']
 mechanism=B/'research/round10/mechanism/actual160_attempt001';ms=read(mechanism/'SEALED.json');mu=read(mechanism/'SUMMARY.json')
 assert ms['state']=='sealed_author_no_GT_mechanism_reduction' and ms['reducer_sha256']==file_sha(B/'research/round10/mechanism/reduce_mechanism.py')
 assert ms['files']['SUMMARY.json']==file_sha(mechanism/'SUMMARY.json') and ma['mechanism_seal_sha256']==file_sha(mechanism/'SEALED.json') and ma['mechanism_summary_sha256']==file_sha(mechanism/'SUMMARY.json')
 assert mu['binding']['geometry_seal_sha256']==file_sha(bound['prediction_seal']) and mu['binding']['source_seal_sha256']==file_sha(bound['source_seal'])
 mechanism_review=read(bound['root_mechanism_review']);assert mechanism_review['mechanism_seal_sha256']==file_sha(mechanism/'SEALED.json')
 rr=read(bound['root_final_review']);assert rr['publication_ready'] and rr['report_sha256']==file_sha(bound['root_report']) and rr['decision_sha256']==file_sha(bound['root_decision']) and rr['mechanism_seal_sha256']==file_sha(mechanism/'SEALED.json')
 # Input-only corpus for output privacy. Target annotations/GT sidecars are not read.
 inputs=read(I/'diagnostic160_gpu_inputs.json');assert len(inputs)==160 and file_sha(I/'diagnostic160_gpu_inputs.json')==pred['manifest_sha256']==source['manifest_sha256']
 allinputs=read(B/'research/round8/experiment/full5912_gpu_inputs.json');ids={r['id'] for r in allinputs};scenes={r['scene'] for r in allinputs};caption_hashes={sha(r['caption'].encode()) for r in allinputs}
 @lru_cache(maxsize=32768)
 def textcheck(s):
  assert ('/'+'home'+'/') not in s and ('/'+'dataset_full'+'/') not in s and (chr(92)+'Users'+chr(92)) not in s and ('https:'+'//'+'1drv'+'.ms/') not in s
  assert not any(x in s for x in ids|scenes),('Private ID/scene',s[:80]);assert sha(s.encode()) not in caption_hashes,'Original caption payload';return s
 def clean(v):
  if isinstance(v,dict):
   assert not set(v)&FORBIDDEN and not any(k.endswith('_PRIVATE') for k in v)
   return {textcheck(str(k)):clean(x) for k,x in v.items()}
  if isinstance(v,list):return [clean(x) for x in v]
  if isinstance(v,str):return textcheck(v)
  if isinstance(v,float):assert math.isfinite(v)
  assert v is None or isinstance(v,(str,int,float,bool,list,dict));return v
 def project(v):
  if isinstance(v,dict):
   out={}
   for k,x in v.items():
    if k in FORBIDDEN|DROP_MAPS or k.endswith('_PRIVATE') or k.endswith(('_path','_dir')) or any(t in k.lower() for t in ('pointpath','imagepath','metapath')) or '.npz' in k:continue
    if k in ('accepted_axes','accepted_axis_counts6') and isinstance(x,list) and len(x)==6 and all(isinstance(z,int) and not isinstance(z,bool) and z>=0 for z in x):out[k+'_counts_by_coordinate']={a:int(z) for a,z in zip(('cx','cy','cz','sx','sy','sz'),x)}
    elif k in ('two_qualified_opposing_plane_axis_counts','strict_size_pair_axis_counts') and isinstance(x,list) and len(x)==3 and all(isinstance(z,int) and not isinstance(z,bool) and z>=0 for z in x):out[k+'_scalar_components']={a:int(z) for a,z in zip(('x','y','z'),x)}
    elif k=='mean_abs_error_by_coordinate_m' and isinstance(x,list) and len(x)==6:out[k+'_scalar_components']={a:z for a,z in zip(('cx','cy','cz','sx','sy','sz'),x)}
    else:out[k]=project(x)
   return out
  if isinstance(v,list):
   # Raw numeric3/6/7-vectors excluded even from CPU fixtures/history. Boolean
   # coordinate role masks are retained; six mean errors are separately scalars.
   if len(v) in (3,6,7) and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v):return {'numeric_vector_omitted_for_coordinate_privacy':True,'length':len(v)}
   return [project(x) for x in v]
  if isinstance(v,str):
   if v.startswith('/') or ('/'+'home'+'/') in v:return '[local_path_omitted]'
   return v.replace(str(B),'__WORKSPACE_ROOT__')
  return v
 pending={};provenance=[]
 def emit(name,data):
  assert name not in pending;pending[name]=data
 def js(name,value,origin=None,mode='scalar_aggregate_or_explicit_field_projection'):
  data=(json.dumps(clean(value),indent=2,allow_nan=False)+'\n').encode();emit(name,data)
  if origin:provenance.append({'source_relative':str(origin.relative_to(B)),'source_sha256':file_sha(origin),'export_relative':name,'export_sha256':sha(data),'mode':mode})
 for rel in CODE_FILES:
  p=local(rel);text=p.read_text().replace(str(B),'__WORKSPACE_ROOT__');textcheck(text);data=text.encode();emit('code/'+rel,data);provenance.append({'source_relative':rel,'source_sha256':file_sha(p),'export_relative':'code/'+rel,'export_sha256':sha(data),'mode':'code_only_workspace_path_projection_no_algorithm_change'})
 for rel in DOC_FILES:
  p=local(rel);data=textcheck(p.read_text().replace(str(B),'__WORKSPACE_ROOT__')).encode();emit('docs/'+rel,data);provenance.append({'source_relative':rel,'source_sha256':file_sha(p),'export_relative':'docs/'+rel,'export_sha256':sha(data),'mode':'documentation_workspace_path_projection'})
 for rel in STATIC_JSON:js('protocol/'+rel,project(read(local(rel))),local(rel),'explicit_private_paths_files_coordinates_and_PRIVATE_fields_removed_boolean_masks_preserved')
 for rel in ('research/round10/experiment/ROOT_ACTIVE_GALLERY12_REVIEW.json',):
  optional=B/rel
  if optional.is_file():js('protocol/'+rel,project(read(optional)),optional)
 for key in REQUIRED:
  if key in ('root_report','root_decision'):
   p=bound[key];data=textcheck(p.read_text().replace(str(B),'__WORKSPACE_ROOT__')).encode();emit('final/'+key+p.suffix,data);provenance.append({'source_relative':str(p.relative_to(B)),'source_sha256':file_sha(p),'export_relative':'final/'+key+p.suffix,'export_sha256':sha(data),'mode':'root_public_narrative_path_projection'})
  else:js('final/'+key+'.json',project(read(bound[key])),bound[key])
 # Sealed result rows only: explicit scalar allowlist, alias IDs/scenes, no GT boxes.
 actual=read(resultpath);assert {k:v for k,v in actual.items() if k!='rows_PRIVATE'}==scores
 rows=actual['rows_PRIVATE'];assert len(rows)==160 and len({r['id'] for r in rows})==160
 alias={r['id']:f'case_{i+1:06d}' for i,r in enumerate(inputs)};scene_alias={k:f'scene_{i+1:02d}' for i,k in enumerate(sorted({(r['platform'],r['scene']) for r in inputs}))};inputmap={r['id']:r for r in inputs}
 fields=('case_alias','platform','scene_alias','arm','role','iou','Acc25','Acc50','delta_iou_vs_source','center_L2_error_m','size_L2_error_m','center_shift_L2_m','size_shift_L2_m','accepted_axis_bitmask','actual_changed_axis_bitmask','accepted_any','actual_changed','raw_nonpositive_sizes','status');buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader();count=0
 for row in rows:
  item=inputmap[row['id']];assert row['platform']==item['platform'] and row['scene']==item['scene'] and tuple(row['arms'])==ARMS
  source_iou=row['arms']['source']['iou']
  for arm,x in row['arms'].items():
   ce=x['coordinate_errors'];record={'case_alias':alias[row['id']],'platform':row['platform'],'scene_alias':scene_alias[(row['platform'],row['scene'])],'arm':arm,'role':'frozen_source' if arm=='source' else 'direct_geometry_no_parameter_TTA','iou':x['iou'],'Acc25':x['iou']>.25,'Acc50':x['iou']>.5,'delta_iou_vs_source':x['iou']-source_iou,'center_L2_error_m':ce['center_L2_m'],'size_L2_error_m':ce['size_L2_m'],'center_shift_L2_m':x['center_shift_L2_m'],'size_shift_L2_m':x['size_shift_L2_m'],'accepted_axis_bitmask':sum(int(v)<<j for j,v in enumerate(x['accepted_axes'])),'actual_changed_axis_bitmask':sum(int(v)<<j for j,v in enumerate(x['actual_changed_axes'])),'accepted_any':x['accepted_any'],'actual_changed':x['actual_changed'],'raw_nonpositive_sizes':x['raw_nonpositive_sizes'],'status':x['status']};clean(record);writer.writerow(record);count+=1
 assert count==640
 # Full-population scalar reduction cross-check, retaining zero/harm/tie cases.
 for summary in scores['summary']:
  if summary['population']!='all':continue
  values=[r['arms'][summary['arm']]['iou'] for r in rows if r['platform']==summary['platform']];assert summary['n']==len(values)
  assert abs(math.fsum(values)/len(values)-summary['mean_iou'])<1e-14
 csvdata=buf.getvalue().encode();emit('results/ARM_SCALARS.csv.gz',gzip.compress(csvdata,mtime=0));js('results/ARM_SCALARS_SCHEMA.json',{'rows':640,'unique_cases':160,'arms_per_case':4,'gzip_mtime':0,'fields':list(fields),'scope':'Correlated arm rows, not640 independent samples. Nativeyaw0; error magnitudes are scalars, no actual or GT coordinates. Boolean coordinate roles represented by integer bitmasks. Negative/zero/fallback cases retained.'})
 js('EXPORT_PROVENANCE.json',{'sources':provenance,'final_sources':{k:{'source_relative':str(p.relative_to(B)),'source_sha256':file_sha(p)} for k,p in bound.items()},'mode':'finite_allowlist_explicit_projection_no_private_payloads','original_source_checkpoint_sha256':source['source_checkpoint_sha256'],'not_neural_TTA_or_heldout_observation_validation':True})
 emit('README.md',b'R10 fixed-q0 direct geometry diagnostic. Read final root report/decision for actual results and adoption. Four arms share original source instance selection; full/no-RGB-coordinate share the same RGB-filtered support and eligibility. GrabCut mask can contain neighboring objects/background. RGB-gated pointmask ablation is not pure-PC deployment. Whole observations and previously studied target VAL are not fresh heldout evidence. No neural parameter adaptation occurs in this stage. Private captions/media/points/GT coordinates/masks/features/weights are excluded. Alias640scalar rows preserve zeros, harms and fallback cases.\n')
 for name,data in pending.items():
  assert not any(x.encode() in data for x in ids|scenes),(name,'Private identifier leaked')
  assert not name.endswith(('.npz','.npy','.pt','.pth','.safetensors','.jpg','.png','.mp4','.wav'))
 a.output.mkdir(parents=True)
 for name,data in pending.items():p=a.output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 manifest={'study':'3eed_fixed_q0_visible_rgb_geometry_diagnostic_20261030','n_cases':160,'n_arm_rows':640,'files':[{'path':name,'sha256':sha(data),'bytes':len(data)} for name,data in sorted(pending.items())],'root_final_review_sha256':file_sha(bound['root_final_review']),'source_checkpoint_sha256':source['source_checkpoint_sha256'],'scope':'Finite public code/protocol/scalars only; no private input data or weights'};(a.output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'n_files':len(pending)+1,'manifest_sha256':file_sha(a.output/'MANIFEST.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=E/'public_export');main(p.parse_args())
