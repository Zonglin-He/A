"""R12 postseal finite mechanism/gate review. No new IoU/GT metadata/model."""
from pathlib import Path
import json,hashlib,collections,argparse,math
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 ps=read(a.predictions/'SEALED.json');tr=read(a.predictions/'trace.json');ms=read(a.mechanism/'SEALED.json');m=read(a.mechanism/'SUMMARY.json');r=read(a.scored/'RESULTS_PRIVATE.json');audit=read(a.score_audit)
 assert audit['state']=='PASS_R12_INDEPENDENT_SCALAR_REDUCTION_AUDIT' and audit['results_sha256']==sha(a.scored/'RESULTS_PRIVATE.json') and audit['prediction_seal_sha256']==sha(a.predictions/'SEALED.json')
 for name,h in ms['files'].items():assert sha(a.mechanism/name)==h
 assert m['binding']['geometry_seal_sha256']==sha(a.predictions/'SEALED.json')
 checks=0;maxerr=0.
 def cmp(x,y):
  nonlocal checks,maxerr
  checks+=1
  if isinstance(x,dict):
   for k,v in x.items():cmp(v,y[k])
  elif isinstance(x,list):
   assert len(x)==len(y)
   for u,v in zip(x,y):cmp(u,v)
  elif isinstance(x,(float,int)) and isinstance(y,(float,int)):
   err=abs(x-y);maxerr=max(maxerr,err);assert err<1e-11,(x,y)
  else:assert x==y,(x,y)
 for platform in ('all','drone','quad'):
  tt=[x for x in tr if platform=='all' or x['platform']==platform];d=[x['diagnostics'] for x in tt];real=[v for x in d for v in x.get('neighbor_attempts',[])];wrong=[x['wrongview']['attempt'] for x in d if 'wrongview' in x];assoc=collections.Counter(x.get('anchor_association',{}).get('status','not_attempted') for x in d);admitted=[x for x in d if 'multiview_geometry' in x];perpoint=[v for x in d for v in x.get('multiview',{}).get('neighbors',[])]
  def view(vv,retrieved):
   return {'retrieved_preselected':retrieved,'actually_attempted':len(vv),'not_attempted_anchor_or_noun':retrieved-len(vv),'global_PnP_pass':sum(v.get('registration',{}).get('accepted',False) for v in vv),'unique_mask_LK_track':sum('target_depthbound_unique_pairs' in v for v in vv),'depth_pairs_at_least6':sum(v.get('target_depthbound_unique_pairs',0)>=6 for v in vv),'static_pass':sum(v.get('static_support',{}).get('accepted',False) for v in vv),'static_contradiction':sum(v.get('contradiction',False) for v in vv),'unique_track_insufficient_depth_not_proven_motion':sum(v['status']=='NA_insufficient_depthbound_track_pairs_not_proven_motion' for v in vv),'registration_statuses':dict(collections.Counter(v['registration']['status'] for v in vv)),'terminal_statuses':dict(collections.Counter(v['status'] for v in vv))}
  expected={'n':len(tt),'noun_supported':sum(bool(x.get('target_noun',{}).get('COCO_class_family',[])) for x in d),'noun_statuses':dict(collections.Counter(x.get('target_noun',{}).get('scope','source_uncomputed') for x in d)),'anchor_association_statuses':dict(assoc),'anchor_multicandidate':assoc['NA_multiple_independent_samefamily_masks'],'anchor_unique_candidate':sum('selected_anchor_mask_index' in x for x in d),'anchor_zero_candidate':sum('selected_anchor_mask_index' not in x for x in d)-assoc['NA_multiple_independent_samefamily_masks'],'missing_real_roles':sum(v.get('status')=='missing_prelocked_no_replacement' for x in tt for v in x['roles']),'real_views':view(real,sum(v.get('status')!='missing_prelocked_no_replacement' for x in tt for v in x['roles'])),'wrong_views':view(wrong,sum(x['wrong_frame_alias'] is not None for x in tt)),'verified_neighbor_count_per_anchor':dict(collections.Counter(str(sum(v['verified'] for v in x.get('neighbor_attempts',[]))) for x in d)),'static_contradiction_veto_anchors':sum(x.get('multiview',{}).get('contradiction_veto',False) for x in d),'MV_admitted_anchors':len(admitted),'perpoint_verified_nonveto_view_n':len(perpoint)}
  actual=m['all'] if platform=='all' else m['by_platform'][platform];cmp(expected,actual)
  for key in ('visible_instance_points_before_transform','anchor_FOV','anchor_mask','anchor_mask_and_source_ROI','pixel_within3px','pixel_and_depth_within_threshold','fully_corroborated_neighbor_points'):
   vals=[float(x[key]) for x in perpoint];cmp({'n':len(vals),'sum':math.fsum(vals),'zero_count':sum(x==0 for x in vals)},actual['perpoint'][key])
  for arm,values in actual['arms'].items():
   pp=[row for row in r['rows_PRIVATE'] if platform=='all' or row['platform']==platform];cmp({'changed_samples':sum(row['arms'][arm]['actual_changed'] for row in pp),'accepted_samples':sum(row['arms'][arm]['accepted_any'] for row in pp),'changed_axis6':[sum(row['arms'][arm]['actual_changed_axes'][j] for row in pp) for j in range(6)],'statuses':dict(collections.Counter(row['arms'][arm]['status'] for row in pp))},values)
 pre=read(a.preplan);assert pre['both_micro_and_equal_scene_required'] and pre['preferred_candidate']=='multiview_instance_direct' and pre['separate_single_frame_candidate']=='single_instance_direct'
 tables={(x['platform'],x['arm']):x for x in r['summary'] if x['population']=='all'};pairs={(x['platform'],x['arm'],x['reference']):x for x in r['paired_contrasts'] if x['population']=='all'}
 candidates={};numbers={}
 for candidate,half in [('single_instance_direct','single_instance_half'),('multiview_instance_direct','multiview_instance_half')]:
  by={}
  for platform in ('drone','quad'):
   p=pairs[platform,candidate,'source'];h=pairs[platform,candidate,half];t=tables[platform,candidate]
   coord=all(p[field]['micro_mean']>0 and p[field]['equal_scene_mean']>0 for field in ('center_scene',)) or all(p[field]['micro_mean']>0 and p[field]['equal_scene_mean']>0 for field in ('size_scene',))
   b={'nonzero_changes':t['actual_changed_n']>0,'task_over_source_micro_positive':p['IoU_scene']['micro_mean']>0,'task_over_source_equal_scene_positive':p['IoU_scene']['equal_scene_mean']>0,'same_relevant_coordinate_improves_both_weights':coord,'positive_gain_sum_exceeds_harms':p['IoU_gain']['positive_sum']>abs(p['IoU_gain']['negative_sum']),'full_beats_fixed_half_micro':h['IoU_scene']['micro_mean']>0,'full_beats_fixed_half_equal_scene':h['IoU_scene']['equal_scene_mean']>0}
   b['common_candidate_prerequisites_pass']=all(b.values())
   if candidate=='multiview_instance_direct':
    s=pairs[platform,candidate,'single_instance_direct'];u=pairs[platform,candidate,'multiview_maskmatch_only_direct'];b['additional_MV_increment_over_single_both_weights']=s['IoU_scene']['micro_mean']>0 and s['IoU_scene']['equal_scene_mean']>0;b['additional_MV_increment_over_unfiltered_both_weights']=u['IoU_scene']['micro_mean']>0 and u['IoU_scene']['equal_scene_mean']>0
   b['all_candidate_prerequisites_pass']=b['common_candidate_prerequisites_pass'] and all(v for k,v in b.items() if k.startswith('additional_'));by[platform]=b
   numbers[platform+'|'+candidate]={'actual_changed':t['actual_changed_n'],'mIoU_percent':100*t['mean_iou'],'equal_scene_mIoU_percent':100*t['equal_scene_mean_iou'],'source_gain_micro_pp':100*p['IoU_scene']['micro_mean'],'source_gain_equal_scene_pp':100*p['IoU_scene']['equal_scene_mean'],'full_minus_half_micro_pp':100*h['IoU_scene']['micro_mean'],'source_gain_help_harm_zero':[p['IoU_gain'][k] for k in ('positive_count','negative_count','ties_within_tol')]}
  candidates[candidate]={'by_platform':by,'both_platforms_pass':all(x['all_candidate_prerequisites_pass'] for x in by.values())}
 assert not any(v['both_platforms_pass'] for v in candidates.values())
 split={}
 for platform in ('drone','quad'):
  rr=[x for x in r['rows_PRIVATE'] if x['platform']==platform];split[platform]={}
  for changed in (False,True):
   ss=[x for x in rr if x['arms']['multiview_instance_direct']['actual_changed']==changed];dd=[x['arms']['multiview_instance_direct']['iou']-x['arms']['single_instance_direct']['iou'] for x in ss];split[platform][str(changed)]={'n':len(ss),'MV_minus_single_IoU_sum':math.fsum(dd),'sourcefallback_when_false_not_newgeometry':not changed}
 out={'state':'PASS_R12_INDEPENDENT_MECHANISM_AND_PRELOCKED_PROGRESSION_REVIEW','prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'scored_seal_sha256':sha(a.scored/'SEALED.json'),'score_audit_sha256':sha(a.score_audit),'mechanism_seal_sha256':sha(a.mechanism/'SEALED.json'),'mechanism_summary_sha256':sha(a.mechanism/'SUMMARY.json'),'preplan_sha256':sha(a.preplan),'auditor_sha256':sha(Path(__file__)),'finite_mechanism_numeric_or_exact_checks':checks,'mechanism_max_difference':maxerr,'candidates':candidates,'neural_TTA_progression_pass':False,'neural_updates_executed':False,'key_numbers':numbers,'postseal_descriptive_MV_gain_decomposition_by_actual_change':split,'decision':'Neither predeclared candidate passes. No regression/TTA updates on currentfeedback; no score-based changes ofprimarybackend/thresholds/half/cohort.','limits':['RANSAC/ICP/PnP inputdevelopment criteria differ; 2/5PnP acceptance isnot superiority or taskconfirmation.','Helpermaskgeneration independentofq0, but targetassociation remains firstnoun/sourceROI condition; noidentityverified claim.','Track/ray/static/perpoint validity doesnotguarantee taskbenefit. MainMV lacksactualchanges inDrone andharmsQuadIoU.','MVrecovering over harmfulsingle/unfiltered isnotpositivefeedback versus source; main also loses tohalf inQuad.','Size0changes meanssizefeedback capacitynot tested. Only2/3scenes; unequalDrone32/18 bothmicro/equalweights retained.','Exactcontent disjoint anchors do not certifyuniquephysicaltimes/objects; oldpath-independence erratum preserved.'],'no_new_GT_metadata_IoU_model_GPU_or_method_changes':True};assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':out['state'],'checks':checks,'maxdiff':maxerr,'gate':False,'sha256':sha(a.output)}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('predictions','scored','mechanism','score-audit','preplan','output'):p.add_argument('--'+key,type=Path,required=True)
 main(p.parse_args())
