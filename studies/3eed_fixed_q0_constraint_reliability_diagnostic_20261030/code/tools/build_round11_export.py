"""Finite R11 public export, prepared only until actual final sources/gates exist.
No network, Git, GPU, model or new scoring; no private media or coordinate export.
"""
from pathlib import Path,PurePosixPath
from functools import lru_cache
import argparse,csv,gzip,hashlib,io,json,math
B=Path(__file__).resolve().parents[1];E=B/'research/round11/experiment';I=B/'research/round11/input_prepare';V=B/'research/round11/evaluation'
ARMS=('source','unconditional_full','no_rgb_coordinate','weighted_constraints','uniform_weighted_constraints','reliable_50','hash_random_50','source_agreement_50','low_plane_residual_50','point_count_50','stability_only_50')
CONTROLS=('reliability','hash_random','source_agreement','low_plane_residual','point_count','stability_only');GATES=ARMS[5:]
REQUIRED=('source_seal','prediction_seal','scalar_summary','score_release','source_unlabelled_audit','prediction_unlabelled_audit','scored_independent_audit','mechanism_summary','mechanism_seal','mechanism_gate_review','root_mechanism_review','root_report','root_decision','root_final_review')
READINESS=('all320_source_reliability_11arms_atomic6_and_policies_sealed_before_GT','actual_scored_and_independent_reduction_audits_passed','NA_defined0_exactcoverage_uniformcontrols_and_sharedsource_limits_explicit','no_neuralTTA_or_identity_confirmation_claim','root_report_decision_mechanism_and_finalreview_complete','privacy_no_originalIDs_scenes_captions_coordinates_masks_features_weights_or_media')
CODE_FILES=tuple('crossplatform/round11/'+x for x in ('capture_source.py','reliability.py','weighted_geometry.py','run_reliability.py','check_reliability_cpu.py'))+tuple('research/round11/input_prepare/'+x for x in ('prepare_inputs.py','inventory_assets.py','audit_source320.py','check_reliability_independent_cpu.py','check_weighted_bridge_independent_cpu.py'))+tuple('research/round11/evaluation/'+x for x in ('metrics.py','discrimination.py','score_reliability.py','audit_unlabelled.py','audit_scored.py','audit_mechanism_gate.py','check_metrics_cpu.py','check_reader_cpu.py','check_reduction_cpu.py','check_discrimination_cpu.py','check_discrimination_cpu_v2.py','history/pre_final_author_tie_alignment/discrimination.py','history/pre_axis_family_reporting/metrics.py'))+tuple('research/round11/mechanism/'+x for x in ('reduce_mechanism.py','reduce_mechanism_v2.py','check_reducer_cpu.py','check_reducer_v2_cpu.py','history/attempt001_hash_score_schema/reduce_mechanism.py'))+('research/round11/qualitative/prepare_gallery_plan.py','research/round11/qualitative/render_reliability_gallery.py','crossplatform/round10/visibility_geometry.py','crossplatform/round10/rgb_support.py','crossplatform/round9/run_diagnostic.py','crossplatform/round9/fixed_q0_geometry.py','crossplatform/round9/head_exact_fit.py','crossplatform/round8/association_geometry.py','crossplatform/round8/semantic_encoder.py','crossplatform/round7/camera_projection.py','crossplatform/round7/regression_adapter.py','crossplatform/round7/experiment/runner_helpers.py','crossplatform/round6/factory.py','research/round5/audit/independent_core_v2.py','tools/obb_iou.py','tools/build_round11_export.py')
STATIC_JSON=tuple('research/round11/input_prepare/'+x for x in ('INPUT_PLAN_LOCK.json','COHORT_AVAILABILITY.json','INPUT_RECEIPT.json','LOCAL_ASSOCIATION_ASSET_INVENTORY.json','PREPARATION_READY.json','SOURCE_CAPTURE_IMPLEMENTATION_REVIEW.json','SOURCE_AUDITOR_PREPARATION.json','RELIABILITY_INDEPENDENT_CPU_FIXTURE.json','WEIGHTED_BRIDGE_INDEPENDENT_CPU_FIXTURE.json','RELIABILITY_IMPLEMENTATION_INDEPENDENT_REVIEW.json'))+tuple('research/round11/evaluation/'+x for x in ('CPU_DISCRIMINATION_FIXTURE.json','CPU_DISCRIMINATION_FIXTURE_v2.json','CPU_METRICS_FIXTURE.json','CPU_PREPARATION_RECEIPT.json','CPU_READER_FIXTURE.json','CPU_REDUCTION_FIXTURE.json','CPU_REPEAT_WRITE_FAILURE.json','EVALUATION_SEMANTIC_REVISION.json','EVAL_CODE_LOCK.json','REDUCTION_AUDIT_CODE_LOCK.json','IMPLEMENTATION_INDEPENDENT_REVIEW.json','IMPLEMENTATION_INDEPENDENT_REVIEW_DRAFT.json','WRAPPER_IMPLEMENTATION_REVIEW.json','PROTOCOL_LOCK.json','PROTOCOL_v2.json'))+tuple('research/round11/mechanism/'+x for x in ('CPU_FIXTURE.json','CPU_FIXTURE_V2.json','EXECUTION_RECEIPT.json','ATTEMPT001_FAILURE.json','ACTUAL320_AUTHOR_INTERPRETATION.json'))+tuple('research/round11/experiment/'+x for x in ('SOURCE_CAPTURE_PLAN_LOCK.json','RELIABILITY_PLAN_LOCK.json','PROGRESSION_PREPLAN_LOCK.json','PROGRESSION_GATE.json','GALLERY_TASK_DIAGNOSTIC.json','AUTHORIZATION.json'))+('crossplatform/round11/IMPLEMENTATION_LOCK.json','crossplatform/round11/CPU_FIXTURE.json','crossplatform/round11/history/preflight_v1_before_correspondence_gate/IMPLEMENTATION_LOCK.json','research/round11/qualitative/GALLERY_PLAN_LOCK.json','research/round11/qualitative/VISUAL_REVIEW.json')
DOC_FILES=('crossplatform/round11/RELIABILITY_CONTRACT.md','crossplatform/round11/RELIABILITY_PROPOSAL.md','research/round11/input_prepare/AB_SEED_DIAGNOSTIC_PROPOSAL.md','research/round11/mechanism/README.md','research/round11/mechanism/ACTUAL320_AUTHOR_INTERPRETATION.md','research/round11/experiment/EXPERIMENT_CARD.md','research/round11/experiment/REPRODUCTION.md')
FORBIDDEN={'id','ids','case_id','scene','frame','caption','captions','utterance','ground_info','bbox_3d','gt_bbox','positive_map','GT7D_model','target7','target6','source6','box6','box7','center3','size3','source_boxes_raw6','source_boxes_eval6','arm_boxes_raw6','arm_boxes_eval6','atomic_boxes_raw6','atomic_boxes_eval6','pre_constraint_geometry6','RGB_mask2d_private','points','sampled_points_model','raw_indices','center_features','size_features','image_intrinsic','image_extrinsic','camera_distortion','embeddings','proj_queries','proj_tokens','roi_xyxy','mask2d','uv','depth','abs_error6_m','source_normalized_shift6','selected_ids'}
DROP_MAPS={'files','source_cache_files','source_input_paths','weights_relative_private_location','workspace_root','programroot'}
sha=lambda x:hashlib.sha256(x).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(rel):
 q=PurePosixPath(rel);assert q.parts and not q.is_absolute() and '..' not in q.parts;p=B/q.as_posix();assert p.resolve().is_relative_to(B.resolve()) and p.is_file(),rel;return p

def main(a):
 assert not a.output.exists() and a.output.resolve().is_relative_to(B.resolve()),'New bounded output required'
 ready=read(E/'PUBLICATION_READINESS.json')
 for key in READINESS:assert ready.get(key) is True,('Missing final gate',key)
 specs=read(E/'EXPORT_FINAL_SOURCES.json');bound={}
 for key in REQUIRED:
  spec=specs[key];assert set(spec)=={'path','sha256'};p=local(spec['path']);assert file_sha(p)==spec['sha256'];bound[key]=p
 source=read(bound['source_seal']);pred=read(bound['prediction_seal']);summary=read(bound['scalar_summary']);scoresealpath=bound['scalar_summary'].parent/'SEALED.json';ss=read(scoresealpath)
 assert source['state']=='sealed_R11_source_forward_and_input_observations_no_G_no_GT' and source['n_samples']==320 and source['source_unchanged'] and source['source_repeat_bitexact']
 assert pred['state']=='sealed_R11_reliability11_and_atomic6_before_GT' and pred['n_samples']==320 and pred['n_arms']==11 and pred['atomic_arms_per_sample']==6 and pred['all_queries_fixed_original_q0'] and pred['all_reliability_and_rank_weights_sealed_before_GT'] and pred['neural_parameter_updates_executed'] is False
 assert pred['source_seal_sha256']==file_sha(bound['source_seal']) and pred['source_checkpoint_sha256']==source['source_checkpoint_sha256'] and pred['source_model_state_fingerprint']==source['source_model_state_fingerprint']
 assert source['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 for seal,p in ((source,bound['source_seal']),(pred,bound['prediction_seal']),(ss,scoresealpath)):
  for rel,digest in seal['files'].items():q=PurePosixPath(rel);assert not q.is_absolute() and '..' not in q.parts;assert file_sha(p.parent/rel)==digest
 assert pred['program_sha256_base']=='workspace_root'
 for rel,digest in pred['program_sha256'].items():assert file_sha(local(rel))==digest
 assert summary['state']=='R11_reliability_scored_after_all_arm_seal' and summary['n']==ss['n_samples']==320 and ss['prediction_seal_sha256']==file_sha(bound['prediction_seal'])
 grant=read(bound['score_release']);assert grant['CPU_SCORE_RELEASE'] and grant['all_source_reliability_atomic_and_policy_arms_sealed_before_GT'] and grant['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and grant['source_capture_seal_sha256']==file_sha(bound['source_seal']) and ss['release_sha256']==file_sha(bound['score_release'])
 for key,name in (('scorer_program_sha256','score_reliability.py'),('metrics_program_sha256','metrics.py'),('discrimination_program_sha256','discrimination.py')):assert grant[key]==file_sha(V/name)
 ua=read(bound['source_unlabelled_audit']);assert ua['state']=='independent_actual_R11_source320_input_observation_binding_audit_pass' and ua['source_seal_sha256']==file_sha(bound['source_seal']) and ua['n']==320 and ua['auditor_sha256']==file_sha(I/'audit_source320.py')
 pa=read(bound['prediction_unlabelled_audit']);assert pa['state']=='PASS_R11_ACTUAL_UNLABELLED_PREDICTION_CONTRACT' and pa['n']==320 and pa['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and pa['source_capture_seal_sha256']==file_sha(bound['source_seal']) and pa['auditor_program_sha256']==file_sha(V/'audit_unlabelled.py')
 resultpath=bound['scalar_summary'].parent/'RESULTS_PRIVATE.json';ra=read(bound['scored_independent_audit']);assert ra['state']=='PASS_R11_INDEPENDENT_SCALAR_REDUCTION_AUDIT' and ra['n']==320 and ra['results_sha256']==file_sha(resultpath) and ra['scored_seal_sha256']==file_sha(scoresealpath) and ra['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and ra['release_sha256']==file_sha(bound['score_release']) and ra['auditor_sha256']==file_sha(V/'audit_scored.py') and not ra['failures']
 ms=read(bound['mechanism_seal']);mu=read(bound['mechanism_summary']);assert ms['state']=='sealed_R11_author_no_GT_mechanism_reduction' and ms['files']['SUMMARY.json']==file_sha(bound['mechanism_summary']) and ms['reducer_sha256']==file_sha(B/'research/round11/mechanism/reduce_mechanism_v2.py')
 assert mu['binding']['geometry_seal_sha256']==file_sha(bound['prediction_seal']) and mu['binding']['source_seal_sha256']==file_sha(bound['source_seal'])
 mg=read(bound['mechanism_gate_review']);assert mg['state']=='PASS_INDEPENDENT_MECHANISM_AND_PRELOCKED_GATE_REVIEW' and mg['prediction_seal_sha256']==file_sha(bound['prediction_seal']) and mg['scored_seal_sha256']==file_sha(scoresealpath) and mg['score_audit_sha256']==file_sha(bound['scored_independent_audit']) and mg['mechanism_seal_sha256']==file_sha(bound['mechanism_seal']) and mg['mechanism_summary_sha256']==file_sha(bound['mechanism_summary']) and mg['auditor_sha256']==file_sha(V/'audit_mechanism_gate.py')
 rootm=read(bound['root_mechanism_review']);rr=read(bound['root_final_review']);assert rootm['mechanism_seal_sha256']==file_sha(bound['mechanism_seal']) and rr['publication_ready'] and rr['report_sha256']==file_sha(bound['root_report']) and rr['decision_sha256']==file_sha(bound['root_decision']) and rr['mechanism_seal_sha256']==file_sha(bound['mechanism_seal'])
 inputs=read(I/'diagnostic320_gpu_inputs.json');assert len(inputs)==320 and file_sha(I/'diagnostic320_gpu_inputs.json')==pred['manifest_sha256']==source['manifest_sha256']
 corpus=read(B/'research/round8/experiment/full5912_gpu_inputs.json');ids={r['id'] for r in corpus};scenes={r['scene'] for r in corpus};caption_hashes={sha(r['caption'].encode()) for r in corpus}
 @lru_cache(maxsize=32768)
 def textcheck(s):
  assert ('/'+'home'+'/') not in s and ('/'+'dataset_full'+'/') not in s and (chr(92)+'Users'+chr(92)) not in s and ('https:'+'//'+'1drv'+'.ms/') not in s
  assert not any(x in s for x in ids|scenes) and sha(s.encode()) not in caption_hashes,'Private input string';return s
 def clean(v):
  if isinstance(v,dict):assert not set(v)&FORBIDDEN and not any(k.endswith('_PRIVATE') for k in v);return {textcheck(str(k)):clean(x) for k,x in v.items()}
  if isinstance(v,list):return [clean(x) for x in v]
  if isinstance(v,str):return textcheck(v)
  if isinstance(v,float):assert math.isfinite(v)
  assert v is None or isinstance(v,(int,float,bool));return v
 def project(v):
  if isinstance(v,dict):
   out={}
   for k,x in v.items():
    if k in FORBIDDEN|DROP_MAPS or k.endswith('_PRIVATE') or k.endswith(('_path','_dir')) or any(q in k.lower() for q in ('pointpath','imagepath','metapath')) or '.npz' in k:continue
    if k in ('actual_changed_axis_counts','original_coordinate_application_n_by_axis','coordinate_score_defined_n_by_axis') and isinstance(x,list) and len(x)==6:out[k+'_scalar_components']={a:z for a,z in zip(('cx','cy','cz','sx','sy','sz'),x)}
    else:out[k]=project(x)
   return out
  if isinstance(v,list):
   if len(v) in (3,6,7) and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v):return {'numeric_vector_omitted_for_coordinate_privacy':True,'length':len(v)}
   return [project(x) for x in v]
  if isinstance(v,str):
   if v.startswith('/') or ('/'+'home'+'/') in v:return '[local_path_omitted]'
   return v.replace(str(B),'__WORKSPACE_ROOT__')
  return v
 pending={};provenance=[]
 def emit(name,data):assert name not in pending;pending[name]=data
 def js(name,obj,origin=None,mode='explicit_scalar_and_privacy_field_projection'):
  data=(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n').encode();emit(name,data)
  if origin:provenance.append({'source_relative':str(origin.relative_to(B)),'source_sha256':file_sha(origin),'export_relative':name,'export_sha256':sha(data),'mode':mode})
 for rel in CODE_FILES:
  p=local(rel);data=textcheck(p.read_text().replace(str(B),'__WORKSPACE_ROOT__')).encode();emit('code/'+rel,data);provenance.append({'source_relative':rel,'source_sha256':file_sha(p),'export_relative':'code/'+rel,'export_sha256':sha(data),'mode':'code_workspace_path_projection_only'})
 for rel in DOC_FILES:
  p=local(rel);data=textcheck(p.read_text().replace(str(B),'__WORKSPACE_ROOT__')).encode();emit('docs/'+rel,data);provenance.append({'source_relative':rel,'source_sha256':file_sha(p),'export_relative':'docs/'+rel,'export_sha256':sha(data),'mode':'doc_workspace_path_projection_only'})
 for rel in STATIC_JSON:js('protocol/'+rel,project(read(local(rel))),local(rel))
 for rel in ('research/round11/experiment/ROOT_GALLERY_REVIEW.json','research/round11/evaluation/FINAL_REPORT_INDEPENDENT_REVIEW.json'):
  if (B/rel).is_file():js('protocol/'+rel,project(read(B/rel)),B/rel)
 for key in REQUIRED:
  p=bound[key]
  if key in ('root_report','root_decision'):
   data=textcheck(p.read_text().replace(str(B),'__WORKSPACE_ROOT__')).encode();emit('final/'+key+p.suffix,data);provenance.append({'source_relative':str(p.relative_to(B)),'source_sha256':file_sha(p),'export_relative':'final/'+key+p.suffix,'export_sha256':sha(data),'mode':'root_narrative_path_projection'})
  else:js('final/'+key+'.json',project(read(p)),p)
 actual=read(resultpath);assert {k:v for k,v in actual.items() if k!='rows_PRIVATE'}==summary;rows=actual['rows_PRIVATE'];assert len(rows)==320 and len({r['id'] for r in rows})==320
 aliases={r['id']:f'case_{i+1:06d}' for i,r in enumerate(inputs)};scene_alias={k:f'scene_{i+1:02d}' for i,k in enumerate(sorted({(r['platform'],r['scene']) for r in inputs}))};inputmap={r['id']:r for r in inputs}
 def make_writer(fields):
  buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();return buf,w
 armfields=('case_alias','platform','scene_alias','arm','role','iou','Acc25','Acc50','delta_iou_vs_source','center_L2_error_m','size_L2_error_m','center_shift_L2_m','size_shift_L2_m','changed_axis_bitmask','actual_changed','raw_nonpositive_sizes','status','original_C','case_r_stored','case_r_defined','case_r_nullable','gate_selected','control_value_stored','control_defined')
 atomfields=('case_alias','platform','scene_alias','coordinate_index','coordinate_family','candidate','score_stored','score_defined','score_nullable','IoU_effect_vs_source','coordinate_error_reduction_m','role')
 casefields=('case_alias','platform','scene_alias','original_C','case_r_stored','case_r_defined','case_r_nullable','original_G_IoU_effect','RGB_candidate','RGB_r_stored','RGB_r_defined','RGB_r_nullable','RGB_matched_IoU_effect')+tuple(f'{c}__{k}' for c in CONTROLS for k in ('stored','defined','nullable','selected'))
 ab,aw=make_writer(armfields);tb,tw=make_writer(atomfields);cb,cw=make_writer(casefields);na=nt=nc=0
 for row in rows:
  rawid=row['id'];item=inputmap[rawid];assert row['platform']==item['platform'] and row['scene']==item['scene'] and tuple(row['arms'])==ARMS
  base={'case_alias':aliases[rawid],'platform':row['platform'],'scene_alias':scene_alias[(row['platform'],row['scene'])]};cu=row['case_unit'];ru=row['unit_RGB_term'];src=row['arms']['source']['iou']
  for arm,x in row['arms'].items():
   er=x['coordinate_errors'];control=CONTROLS[GATES.index(arm)] if arm in GATES else None;cc=row['case_controls'][control] if control else None
   role='frozen_source' if arm=='source' else 'exactcoverage_originalG_rank_control' if arm in GATES else 'uniform_platform_mean_direct_constraints' if arm=='uniform_weighted_constraints' else 'reliability_weighted_direct_constraints' if arm=='weighted_constraints' else 'unconditional_direct_geometry'
   record={**base,'arm':arm,'role':role,'iou':x['iou'],'Acc25':x['iou']>.25,'Acc50':x['iou']>.5,'delta_iou_vs_source':x['iou']-src,'center_L2_error_m':er['center_L2_m'],'size_L2_error_m':er['size_L2_m'],'center_shift_L2_m':x['center_shift_L2_m'],'size_shift_L2_m':x['size_shift_L2_m'],'changed_axis_bitmask':sum(int(z)<<j for j,z in enumerate(x['actual_changed_axes'])),'actual_changed':x['actual_changed'],'raw_nonpositive_sizes':x['raw_nonpositive_sizes'],'status':x['status'],'original_C':cu['candidate'],'case_r_stored':cu['score'],'case_r_defined':cu['score_defined'],'case_r_nullable':cu['score'] if cu['score_defined'] else None,'gate_selected':row['gate_selected'][arm] if arm in GATES else None,'control_value_stored':cc['value'] if cc else None,'control_defined':cc['defined'] if cc else None};clean(record);aw.writerow(record);na+=1
  assert len(row['units_coordinate_application'])==6
  for u in row['units_coordinate_application']:
   j=u['coordinate'];assert j in range(6);record={**base,'coordinate_index':j,'coordinate_family':'center' if j<3 else 'size','candidate':u['candidate'],'score_stored':u['score'],'score_defined':u['score_defined'],'score_nullable':u['score'] if u['score_defined'] else None,'IoU_effect_vs_source':u['effect'],'coordinate_error_reduction_m':u['coordinate_error_reduction_m'],'role':'coordinate_substitution_application_not_face_causal_effect'};clean(record);tw.writerow(record);nt+=1
  record={**base,'original_C':cu['candidate'],'case_r_stored':cu['score'],'case_r_defined':cu['score_defined'],'case_r_nullable':cu['score'] if cu['score_defined'] else None,'original_G_IoU_effect':cu['effect'],'RGB_candidate':ru['candidate'],'RGB_r_stored':ru['score'],'RGB_r_defined':ru['score_defined'],'RGB_r_nullable':ru['score'] if ru['score_defined'] else None,'RGB_matched_IoU_effect':ru['effect']}
  for c in CONTROLS:
   v=row['case_controls'][c];gate=GATES[CONTROLS.index(c)];record.update({c+'__stored':v['value'],c+'__defined':v['defined'],c+'__nullable':v['value'] if v['defined'] else None,c+'__selected':row['gate_selected'][gate]})
  clean(record);cw.writerow(record);nc+=1
 assert (na,nt,nc)==(3520,1920,320)
 for tab in summary['summary']:
  if tab['population']=='all':values=[r['arms'][tab['arm']]['iou'] for r in rows if r['platform']==tab['platform']];assert len(values)==tab['n'] and abs(math.fsum(values)/len(values)-tab['mean_iou'])<1e-14
 for name,buf,n,fields in (('ARM_SCALARS',ab,na,armfields),('ATOMIC_SCALARS',tb,nt,atomfields),('CASE_DIAGNOSTIC_SCALARS',cb,nc,casefields)):
  emit('results/'+name+'.csv.gz',gzip.compress(buf.getvalue().encode(),mtime=0));js('results/'+name+'_SCHEMA.json',{'rows':n,'fields':list(fields),'gzip_mtime':0,'unique_cases':320,'not_independent_rows':True,'missing_semantics':'stored0 with definedFalse preserved; nullable field empty means NA, not defined0. Noncandidates retained.','coordinate_boundary':'Indices/families/bitmasks and error magnitudes/effects only; no actual or GT boxes/coordinates.'})
 js('EXPORT_PROVENANCE.json',{'sources':provenance,'final_sources':{k:{'source_relative':str(p.relative_to(B)),'source_sha256':file_sha(p)} for k,p in bound.items()},'source_checkpoint_sha256':source['source_checkpoint_sha256'],'scope':'Finite code/protocol/alias scalars; no rawdata/weights/media/GTcoordinates. No neuralTTA or semantic identity claim.'})
 emit('README.md',b'R11 fixed-q0 constraint reliability diagnostic. See final report and decision for measured results. A/B observed points are disjoint, but RGB/q0/ROI/sourcebackbone are shared. association_supported is compatibility, not identity confirmation; highagreement masks can contain background. Effective conflicts are defined0; unavailable evidence is NA. Exactcoverage ranking may select NA byhash to retain K; weighted-constraint failclosed semantics are separate. Uniform weights are unlabelled cohort/platform means. 3520arm+1920atomic+320case rows are related outputs of320cases/fivescenes, not independent samples. Atomic coordinate replacement is not face-causal deletion. No modelweights/privateIDs/scenes/captions/media/coordinates/masks/features are included.\n')
 for name,data in pending.items():assert not any(x.encode() in data for x in ids|scenes),(name,'Private identifier');assert not name.endswith(('.pt','.pth','.bin','.safetensors','.npz','.npy','.jpg','.png','.mp4','.wav'))
 a.output.mkdir(parents=True)
 for name,data in pending.items():p=a.output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 manifest={'study':'3eed_fixed_q0_constraint_reliability_diagnostic_20261030','unique_cases':320,'arm_rows':3520,'atomic_rows':1920,'case_diagnostic_rows':320,'files':[{'path':name,'bytes':len(data),'sha256':sha(data)} for name,data in sorted(pending.items())],'source_checkpoint_sha256':source['source_checkpoint_sha256'],'root_final_review_sha256':file_sha(bound['root_final_review'])};(a.output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'n_files':len(pending)+1,'manifest_sha256':file_sha(a.output/'MANIFEST.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=E/'public_export');main(p.parse_args())
