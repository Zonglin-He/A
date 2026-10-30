"""Pure synthetic CPU gate/scalar test. No real model/prediction/annotation input."""
from pathlib import Path
import json,tempfile,hashlib,importlib.util
import numpy as np
import score_crossdomain as S

def main():
 out=Path(__file__).resolve().parent;checks={}
 with tempfile.TemporaryDirectory() as tmp:
  t=Path(tmp);pred=t/'predictions';pred.mkdir();program=t/'program';program.mkdir();(program/'version.txt').write_text('Synthetic producer only');checkpoint=t/'synthetic_checkpoint.bin';checkpoint.write_bytes(b'not_a_real_model_synthetic_only')
  names=['source'];modes=['source']
  for mode in S.MODES:
   for arm in ('direct_G','param1','param3','outputspace1','outputspace3'):names.append(mode+'__'+arm);modes.append(mode)
  samples=[];index=[];synthetic_traces={}
  for platform in ('drone','quad'):
   sid='synthetic_'+platform;z=1.8 if platform=='drone' else 0.;source=np.array([1.,0.,z,2.,2.,2.],np.float32);boxes=np.tile([100.,100.,100.,2.,2.,2.],(256,1)).astype(np.float32);boxes[0]=source;boxes[6]=[0,0,z,2,2,2];scores=(1-np.arange(256)/1000).astype(np.float32);arms=np.tile(source,(len(names),1));w=np.zeros((3,6));w[0,0]=1
   if platform=='quad':w[1,0]=1 # accepted PC pseudo has exactly zero delta
   targets=np.tile(source,(3,1));targets[0,0]=.8
   for j,name in enumerate(names):
    if name.startswith('rgb_text'):
     arms[j,0]=.8 if 'direct_G' in name else (1.2 if name.endswith('3') else .9)
   synthetic_traces[sid]={'modes':{}}
   for mi,mode in enumerate(S.MODES):
    active=bool(w[mi].any());delta=bool(np.any(targets[mi]!=source));st={}
    for k in (1,3):st[str(k)]={'trace':[{'parameter_delta_max_by_name':{'synthetic_parameter':.001 if delta else 0.},'parameter_gradient_norm':1. if delta else 0.,'nonzero_gradient_parameter_tensors':1 if delta else 0} for _ in range(k if active else 0)]}
    synthetic_traces[sid]['modes'][mode]={'steps':st}
   np.savez_compressed(pred/(sid+'.npz'),source_boxes_raw6=boxes,source_boxes_eval6=S.evaluation_box(boxes),source_text_scores256=scores,source_native_top10=np.arange(10),fixed_query_index=np.array(0),arm_names=np.asarray(names),arm_modes=np.asarray(modes),arm_boxes_raw6=arms,arm_boxes_eval6=S.evaluation_box(arms),arm_status=np.asarray(['ok' if m in ('source','rgb_text') else 'skip_no_support' for m in modes]),mode_names=np.asarray(S.MODES),pseudo_target6=targets,pseudo_reliability6=w)
   meta=t/(sid+'_metadata_SYNTHETIC.json');S.dump(meta,{'ground_info':[{'bbox_3d':[0.,0.,0.,2.,2.,2.,0.,777.,888.],'class':'DO_NOT_CONSUME_CLASS','positive_map':'DO_NOT_CONSUME_MAP'}]});index.append({'id':sid,'platform':platform,'scene':'synthetic_scene_'+platform,'object_index':0,'meta_path':str(meta),'metadata_container_sha256':S.sha(meta),'waymo_view_suffix':None});samples.append({'id':sid,'platform':platform})
  manifest=t/'manifest.json';idx=t/'index.json';S.dump(manifest,samples);S.dump(idx,index);seal=pred/'SEALED.json';S.dump(seal,{'n_samples':2,'manifest_sha256':S.sha(manifest),'source_checkpoint_sha256':S.sha(checkpoint),'program_sha256':{'version.txt':S.sha(program/'version.txt')},'files':{p.name:S.sha(p) for p in pred.glob('*.npz')}})
  release=t/'release.json';grant={'CPU_SCORE_RELEASE':True,'all_stage_predictions_sealed_before_stage_GT_score':True,'stage':'pure_synthetic_fixture_not_real_experiment','prediction_seal_sha256':S.sha(seal),'manifest_sha256':S.sha(manifest),'scoring_index_sha256':S.sha(idx),'source_checkpoint_sha256':S.sha(checkpoint),'scorer_program_sha256':S.sha(Path(S.__file__))};S.dump(release,grant)
  gateargs=dict(predictions=pred,seal_path=seal,manifest=manifest,scoring_index=idx,checkpoint=checkpoint,release=release,release_sha=S.sha(release),program_root=program);S.gate_before_gt(**gateargs)
  grant['CPU_SCORE_RELEASE']=False;S.dump(release,grant);gateargs['release_sha']=S.sha(release)
  try:S.gate_before_gt(**gateargs);raise AssertionError('Unreleased score accepted')
  except PermissionError:checks['CPU_SCORE_RELEASE_false_rejected_before_annotation_read']=True
  grant['CPU_SCORE_RELEASE']=True;S.dump(release,grant);gateargs['release_sha']=S.sha(release);S.gate_before_gt(**gateargs)
  rows,n,counts,err,disagree=S.score_rows(samples,{x['id']:x for x in index},pred,synthetic_traces);summaries=S.summarize(rows,n);assert len(rows)==2 and err<1e-12 and disagree==0;assert all(r['arms']['rgb_text__param3']['negative_case'] for r in rows);assert all(not r['common_eligible_all_modes'] for r in rows);assert all(r['source_coverage']['5']['best_iou']<1 and r['source_coverage']['10']['best_iou']==1 for r in rows)
  pc_zero=[s for s in summaries if s['platform']=='quad' and s['mode']=='pc_only' and s['arm']=='pc_only__param3' and s['population']=='all_requested'][0];assert pc_zero['n_accepted_any_axis']==1 and pc_zero['n_nonzero_pseudo_delta']==0 and pc_zero['n_parameter_steps_executed']==3 and pc_zero['n_nonzero_parameter_update']==0
  identity=[s for s in summaries if s['platform']=='drone' and s['mode']=='pc_only' and s['arm']=='pc_only__param3' and s['population']=='all_requested'][0];assert identity['n_identity_skip_w0_and_output_source']==1
  checks['accepted_zero_delta_not_counted_as_effective_parameter_update']=True;checks['w0_identity_skip_independent_of_status_string']=True
  assert all(r['source_query']==0 for r in rows);assert all(r['pseudo_diagnostic']['rgb_text']['accepted_axes_move_toward_GT']==1 for r in rows)
  assert any(s['population']=='common_eligible_all_modes' and s['n_scored']==0 and s['mean_iou'] is None for s in summaries)
  # Strict thresholds and 3D volume dtype from exact synthetic cases.
  target=np.array([0.,0.,0.,2.,2.,2.,0.],np.float32);v,_=S.candidate_ious(np.array([[0.,0.,0.,1.,1.,2.],[0.,0.,0.,1.,2.,2.]]),target);assert v.tolist()==[.25,.5] and not(v[0]>.25) and not(v[1]>.5)
  ties=S.source_coverage(boxes,np.ones(256),np.array([6,0,1,2,3,4,5,7,8,9]),0,S.transform_gt([0.,0.,0.,2.,2.,2.,0.,0.,0.],'quad'))[0];assert ties['1']['best_iou']<1 and ties['5']['best_iou']==1
  checks['native_GPU_ties_kept_for_Top5_10_while_Top1_fixed_argmax']=True
  checks.update(selected_3D_and_batched_source_agree_max_error=err,strict_threshold_disagreements=disagree,all_negative_param3_cases_retained=True,sourceq_fixed_not_GT_best=True,source_top7_candidate_makes_Top10_coverage_not_Top5=True,zero_common_eligible_reports_zero_denominator_and_null=True,postseal_pseudo_direction_diagnostic_only=True,strict_greater_than_25_50_not_greater_equal=True,no_targetclass_positive_map_readout=True,source_transform_Drone_plus1point8_Quad_identity=True)
 receipt={'state':'pure_synthetic_CPU_scoring_contract_pass_not_real_result','scorer_sha256':S.sha(Path(S.__file__)),'fixture_sha256':S.sha(Path(__file__)),'checks':checks,'geometry_dependency_sha256':S.GEOMETRY_DEPENDENCY_SHA256,'no_real_model_GPU_targetGT_prediction_or_dataset_access':True};S.dump(out/'SYNTHETIC_SCORER_FIXTURE.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
