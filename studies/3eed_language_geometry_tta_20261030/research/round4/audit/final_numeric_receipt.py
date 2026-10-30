"""Cross-implementation comparison and final locked-parser scope, no model/GT read."""
from pathlib import Path
import json,hashlib,sys,re
import numpy as np
R=Path(__file__).resolve().parent;P=R.parents[2]/'crossplatform/round4';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=R/'INDEPENDENT_NUMERIC_AUDIT.json';assert not out.exists();receipts=[];parser=[];sys.path.insert(0,str(P/'adaptation'));from caption_regions import parse_target_horizontal_region
 for cohort,prefix in [('dev','DEV'),('confirm','CONFIRM')]:
  mine=read(R/(prefix+'_STAGE2_INDEPENDENT_RESULTS_PRIVATE.json'));diff=[]
  for stem in ['adaptation','readout']:
   parent=P/(cohort+'_'+stem+'_scored')/'RESULTS_PRIVATE.json';d=read(parent);m={(x['id'],x['arm']):x for x in d['rows']};errors=[abs(x['iou'][a]-m[x['id'],a]['iou']) for x in mine['rows'] for a in d['arms']];assert max(errors)<1e-11
   summarydiff=[]
   for t in d['summaries']:
    v=next(z for z in mine['summaries'] if z['platform']==t['platform'] and z['arm']==t['arm']);assert v['n']==t['n'] and v['scene_clusters']==t['scene_clusters'] and v['acc25_count']==t['acc25_count'] and v['acc50_count']==t['acc50_count'] and v['invalid_count']==t['invalid_count'];summarydiff.append(abs(v['mean_iou']-t['mean_iou']))
   diff.append({'parent':str(parent),'parent_sha256':sha(parent),'n_iou_cells':len(errors),'max_abs_iou_difference':max(errors),'all_aggregate_denominators_clusters_threshold_counts_invalid_counts_exact':True,'max_summary_mean_difference':max(summarydiff)})
  ownpath=R/(cohort+'_decomposition_v1')/'RESULTS_PRIVATE.json';own=read(ownpath);otherpath=P/(cohort+'_candidate_diagnostic')/'RESULTS_PRIVATE.json';other=read(otherpath);otherby={x['id']:x for x in other['rows']};de=[];booleans=0
  for x in own['rows']:
   y=otherby[x['id']]
   for a,b in [('rawyaw0','raw_yaw0'),('fixedPCA','fixed_PCA')]:
    g=x['geometries'][a];h=y['metrics'][b];de.extend([abs(g['top1_iou']-h['selected_iou']),abs(g['best256_iou_label_assisted']-h['best_candidate_iou'])]);assert g['selected_query']==y['selected_query']
    for t in ['0.25','0.5']:
     for k in ['1','5','10','32','256']:assert g['recall'][t][k]==h['topk'][k][t];booleans+=1
     assert g['decomposition'][t]['rankrecoverable_top1fails_any256passes']==h['rank_recoverable'][t];assert g['decomposition'][t]['candidate_coverage_failure_all256fail']==h['coverage_failure'][t];booleans+=2
   for k in ['center','size','yaw']:de.append(abs(x['interventions'][k]['iou']-y['fixed_selected_PCA_single_GT_replacements'][k]))
  assert max(de)<1e-11
  diff.append({'parent':str(otherpath),'parent_sha256':sha(otherpath),'n_numeric_cells':len(de),'max_abs_difference':max(de),'exact_boolean_rank_topk_cells':booleans})
  trace={x['id']:x for x in read(P/'adaptation'/(cohort+'_predictions')/'trace.json')};inputs=read(P/'source/inputs'/(cohort+'_manifest.json'));bad=[];triggers=[]
  for s in inputs:
   region,diagnostic=parse_target_horizontal_region(s['text']);d=None if region is None else region.__dict__
   assert d==trace[s['id']]['caption_region']
   # This finite lexical check counts only the independently demonstrated edge patterns;
   # it is not a complete semantic verification of all captions.
   if d:
    fragment=d['evidence_text'];risk=bool(re.search(r'\b(?:not|no|neither|nor)\b',fragment,re.I) or re.search(r'\b(?:of|beside|behind|near|between)\s+(?:(?:the|a|an|another|other)\s+)?(?:van|bicycle)\b',fragment,re.I))
    if risk:triggers.append(s['id'])
  parser.append({'cohort':cohort,'n':len(inputs),'final_parser_reproduces_all_sealed_trace_regions_exact':True,'defined_edgecase_lexical_trigger_n':len(triggers),'trigger_ids':triggers,'not_a_general_semanticcorrectness_certificate':True})
  receipts.append({'cohort':cohort,'taskscore_comparisons':diff})
 fixtures=[]
 for text in ['A car located to the left of a van in the image.','A car located to the left of a bicycle in the image.','A car located not on the left side of the image.']:
  region,d=parse_target_horizontal_region(text);fixtures.append({'synthetic_text':text,'actual_region':None if region is None else region.relation,'desired_failclosed':region is None})
 master=P/'CONFIRMATION_ALL_ARMS_SEALED.json';lock=R/'STAGE2_CONFIRM_SCORER_LOCK.json';assert sha(master)=='080f4a00d1d4a77323a1012e2a8ee0f0cc4d858f67aa8dfcc06ef8dd422c786a';assert master.stat().st_mtime_ns<lock.stat().st_mtime_ns
 result={'state':'passed_numeric_audit_with_declared_parser_coverage_limitation','independent_3Dconvex_vs_parent_2Dpolygon':receipts,'parser_actual_inputs':parser,'parser_scope_counterexamples':fixtures,'master_gate':{'sha256':sha(master),'981_hashes_verified_before_this_auditor_target_GT':True,'audit_scorer_lock_after_master_by_mtime':True,'cannot_prove_absence_of_any_unlogged_external_process_GT_read':True},'measurement_adoption_decision':{'query3':'Notadopted; exactlysame chosenboxes asTTAug anddirect inbothcohorts. Any sourcegain is readoutaggregation, notparameteradaptation.','geometry3':'Notadopted; lowersWaymoDEVandCONFIRMIoU, alsoinferiortosameinformationdirect onconfirm. Keep locked3step primary; do notpromote1step.','scope':'WaymoVALprimary andTRAINunionexposedDroneQuadauxiliary separate; nofulltraining/sourceonlyclaim'},'all_output_evidence':{f:sha(R/f) for f in ['STAGE2_DEV_SCORER_LOCK.json','STAGE2_CONFIRM_SCORER_LOCK.json','STAGE2_IMPLEMENTATION_REVIEW.json','CONFIRM_STAGE2_IMPLEMENTATION_REVIEW.json','DEV_STAGE2_INDEPENDENT_RESULTS_PRIVATE.json','CONFIRM_STAGE2_INDEPENDENT_RESULTS_PRIVATE.json','DEV_TEACHER_QUALITY_PRIVATE.json','CONFIRM_TEACHER_QUALITY_PRIVATE.json','DEV_TEACHER_QUALITY_SUPPLEMENT_PRIVATE.json','CONFIRM_TEACHER_QUALITY_SUPPLEMENT_PRIVATE.json','CONFIRM_ANNOTATION_ASSOCIATION_PRIVATE.json']},'failed_engineering_lineage':['DEV teacher supplement first4x4intrinsic matmul rejected before anyresult file; retainedpre_intrinsic4x4_failure script; finalexplicit3x3projection','CONFIRM firstaudit used nonexistentBBSdirname and failed beforetargetmetadata; retainedpre_BBS_path_failure script+log; corrected actualconfirmedsealedpath only'],'model_GPU_training_calls':0,'parent_or_old_files_modified':False,'no_push':True}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'receipt':str(out),'comparisons':receipts,'parser':parser},indent=2))
if __name__=='__main__':main()
