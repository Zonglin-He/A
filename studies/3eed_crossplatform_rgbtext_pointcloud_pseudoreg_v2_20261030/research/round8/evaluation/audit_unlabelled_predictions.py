"""Independent R8 sealed-output engineering audit; no targetGT/scoring-index API.
No model/GPU. Semantic/association validity remains a task-quality question.
"""
from pathlib import Path
import argparse,collections,json,math,time,traceback
import numpy as np
import score_crossdomain as S
ATOL=5e-5
PN=tuple('prediction_heads.5.'+head+'.net.8.'+suffix for head in ('center_residual_head','size_pred_head') for suffix in ('weight','bias'))

def close(a,b):return bool(np.allclose(a,b,rtol=0,atol=ATOL))
def audit_sample(a,t,plan):
 assert int(t['q_source0'])==a['q0'] and t['literal_fullcaption_for_CLIP_no_sentinel'] is True
 assert a['point_sample_sha256'].tobytes().hex()==t['point_sample_sha256'] and a['raw_indices_sha256'].tobytes().hex()==t['raw_indices_sha256'] and a['image_file_sha256'].tobytes().hex()==t['image_file_sha256']
 groups={int(g['group_id']):g for g in t['groups']};pool=t['candidate_pool'];assert len(pool)<=48 and len(set(pool))==len(pool)
 seen=[]
 for gid,g in groups.items():
  members=g['members'];seen+=members;assert g['anchor_query']==min(members,key=lambda q:(-a['source_text_scores256'][q],q));assert all(a['dedup_membership'][q]==gid for q in members);assert all(n>=8 for n in g['member_visible_support_counts'])
  if len(members)>1:
   mm=g['complete_link_pair_minima'];assert mm['shared_points']>=8 and mm['jaccard']>=.45 and mm['support_overlap_coefficient']>=.65 and mm['box_AABB_IoU']>=.1
 assert set(seen)==set(pool) and len(seen)==len(set(seen));assert set(np.flatnonzero(a['dedup_membership']>=0))==set(pool)
 roi=a['roi_group_ids'].tolist();assert roi==t['roi_group_ids'] and len(roi)<=16 and len(roi)==len(set(roi));cos=a['semantic_cosines_tight_context'];assert cos.shape==(len(roi),2) and np.isfinite(cos).all() and (np.abs(cos)<=1.00001).all();assert a['roi_xyxy'].shape==(len(roi),2,4)
 if roi:
  wh=t['camera']['image_size_wh'];W,H=wh;boxes=a['roi_xyxy'];assert (boxes[:,:,0]>=0).all() and (boxes[:,:,1]>=0).all() and (boxes[:,:,2]<=W).all() and (boxes[:,:,3]<=H).all() and (boxes[:,:,2]>boxes[:,:,0]).all() and (boxes[:,:,3]>boxes[:,:,1]).all();assert t['CLIP']['crop_receipt']['crop_count']==2*len(roi);assert t['CLIP']['crop_receipt']['roles']==[str(g)+'__'+role for g in roi for role in ('tight','context')]
 text=t['CLIP']['text_receipt'];assert text['literal_full_description_no_prompt_template_or_noun_replacement'] and text['context_length']==77 and text['text_count']==1
 stats={};maxerr=0.;maxinterval=0.
 for mi,mode in enumerate(S.MODES):
  mt=t['G_modes'][mode];diag=mt['G_diagnostics'];q1=int(a['q1'][mi]);assert mt['original_q0']==a['q0'] and mt['associated_q1']==q1;anchor=a['source_boxes_raw6'][q1];w=a['pseudo_reliability6'][mi];target=a['pseudo_target6'][mi];target64=a['pseudo_target6_float64'][mi];raw=a['raw_consensus_center'][mi];r=float(a['r_global'][mi]);assert mt['lr_has_no_global_trust_multiplier'] is True;assert r==mt['r_global_heuristic_strength_not_probability'];assert close(np.asarray(mt['raw_consensus_minus_anchor_m']),raw-anchor[:3]);assert close(np.asarray(mt['actual_target32_minus_anchor_m']),target[:3]-anchor[:3])
  cn=mt['feature_norm2_plus1']['center'];sn=mt['feature_norm2_plus1']['size'];lr=mt['actual_lr'];assert min(cn,sn)>=1 and math.isfinite(lr) and math.isclose(lr,.25*.1/max(cn,sn),rel_tol=1e-12)
  accepted=bool(w.any());actual_nonzero=bool(np.any((target-anchor)[w>0]!=0));raw_nonzero=bool(np.any(raw!=anchor[:3]));p64_nonzero=bool(np.any(target64!=anchor));result={'q_changed':q1!=a['q0'],'accepted_any_axis':accepted,'raw_consensus_nonzero':raw_nonzero,'final_target64_nonzero':p64_nonzero,'actual_target32_nonzero':actual_nonzero,'r_global':r,'reason':diag['reason'],'parameter_steps':0,'nonzero_gradient_steps':0,'nonzero_parameter_update_runs':0,'selected_changed_runs':0,'maximum_selected_center_delta_m':0.}
  if r==0:assert not accepted
  if roi:
   gid=int(a['selected_group_id'][mi]);assert gid==int(diag['selected_group']) and gid in roi;selected_position=roi.index(gid);supports=np.array([groups[g]['visible_support_count'] for g in roi],float);pc=np.log1p(supports)/max(np.log1p(supports).max(),1.);used=np.roll(cos.astype(float),1,axis=0) if mode=='semantic_shuffle' else cos.astype(float);fused=.5*used[:,0]+.5*used[:,1];rank=pc if mode=='pc_only' else fused+.02*pc;best=min(range(len(roi)),key=lambda i:(-rank[i],groups[roi[i]]['anchor_query']));assert best==selected_position and groups[gid]['anchor_query']==q1;assert close(np.asarray(diag['semantic_cosines_used']),used)
  for k in (1,3):
   name=mode+'__param'+str(k);pn=a['arm_boxes_raw6'][a['names'].index(name)];kn=a['arm_boxes_raw6'][a['names'].index(mode+'__outputspace'+str(k))];error=float(np.max(np.abs(pn-kn)));assert error<=ATOL;maxerr=max(maxerr,error);assert np.array_equal(pn[3:],anchor[3:]) and np.array_equal(kn[3:],anchor[3:]);excess=np.maximum(np.minimum(anchor,target)-pn,pn-np.maximum(anchor,target));assert np.all(excess<=ATOL);maxinterval=max(maxinterval,float(excess.max()));ts=mt['steps'].get(str(k),{}).get('trace',[]);assert len(ts)==(k if accepted else 0);result['parameter_steps']+=len(ts)
   if ts:
    assert close(ts[0]['selected_box_before'],anchor) and close(ts[-1]['selected_box_after'],pn);features=mt['steps'][str(k)]['feature_norms'];factor=lr*features['center_self_kernel_with_bias']/.1*float(w[:3].max())/float(w.sum());assert factor<=.25*(1+1e-5)
    for j,z in enumerate(ts):
     assert z['step']==j+1 and math.isfinite(z['loss']) and z['loss']>=0 and math.isfinite(z['parameter_gradient_norm']) and z['parameter_gradient_norm']>=0;assert z['clip_enabled'] is False and z['clip_factor']==1.;assert set(z['parameter_delta_max_by_name'])==set(PN);assert all(v>=0 and math.isfinite(v) for v in z['parameter_delta_max_by_name'].values());assert all(v==0 for n,v in z['parameter_delta_max_by_name'].items() if '.size_pred_head.' in n);assert z['output_space_replay_max_abs_error']<=ATOL;maxerr=max(maxerr,z['output_space_replay_max_abs_error']);g=np.asarray(z['weighted_SmoothL1_box_gradient6']);assert g.shape==(6,) and np.isfinite(g).all() and not g[3:].any()
     if z['parameter_gradient_norm']>0:assert z['nonzero_gradient_parameter_tensors']>0;result['nonzero_gradient_steps']+=1
    result['nonzero_parameter_update_runs']+=int(any(v>0 for v in ts[-1]['parameter_delta_max_by_name'].values()))
    if k==3:assert close(ts[0]['selected_box_after'],mt['steps']['1']['trace'][0]['selected_box_after'])
   changed=not np.array_equal(pn,anchor);result['selected_changed_runs']+=changed;result['maximum_selected_center_delta_m']=max(result['maximum_selected_center_delta_m'],float(np.linalg.norm(pn[:3]-anchor[:3])))
   if not accepted:assert np.array_equal(pn,anchor) and np.array_equal(kn,anchor)
  stats[mode]=result
 return {'id':t['id'],'platform':t['platform'],'q_source':a['q0'],'modes':stats,'roi_count':len(roi),'group_count':len(groups),'caption_truncated':text['truncated_count'],'max_kernel_error':maxerr,'max_interval_excess':maxinterval}

def main(a):
 start=time.perf_counter();sp=a.predictions/'SEALED.json';assert S.sha(sp)==a.expected_seal_sha256;se=S.read(sp);plan=S.read(a.predictions/'registered_plan.json');assert se['state']=='sealed_R8_strictFP32_crossdomain_predictions_no_target_GT_scoring' and se['all_19arms_sealed_before_any_target_GT_scoring'] and se['grounder_restored_and_CLIP_frozen'] and se['sourceQ0_not_retagged_as_associationQ1'];S.validate_strict_flags(se['strict_FP32_flags']);assert plan['strict_FP32_flags']==se['strict_FP32_flags'] and plan['alpha']==.25 and plan['beta_metres']==.1 and plan['steps']==[1,3] and plan['r_global_policy']=='shrink_shared_pseudo_target_no_extra_LR_multiplier';assert S.sha(a.predictions/'registered_plan.json')==se['plan_sha256'];assert S.sha(a.manifest)==se['manifest_sha256'];assert se['program_sha256_base']=='workspace_root'
 for name,digest in se['files'].items():assert S.sha(S.bounded(a.predictions,name))==digest
 for name,digest in se['program_sha256'].items():assert S.sha(S.bounded(a.workspace_root,name))==digest
 assert S.sha(Path(plan['source_checkpoint_path']))==se['source_checkpoint_sha256']==plan['source_checkpoint_sha256'];assets=Path(plan['CLIP_asset_dir']);conversion=assets/'SAFE_CONVERSION_RECEIPT.json';assert S.sha(conversion)==plan['CLIP_conversion_receipt_sha256'];cr=S.read(conversion);assert cr['state']=='safe_weights_only_official_to_local_safetensors_strict_model_verified';assert se['CLIP_resource']['derived_safetensors_sha256']==cr['derived_safetensors_sha256'];assert all(S.sha(assets/name)==digest for name,digest in cr['runtime_files_sha256'].items())
 rows=S.read(a.manifest);trace=S.read(a.predictions/'trace.json');lookup={t['id']:t for t in trace};assert len(rows)==len(trace)==se['n_samples'] and set(lookup)=={r['id'] for r in rows};assert {p.name for p in a.predictions.glob('*.npz')}=={r['id']+'.npz' for r in rows};repeat=S.read(a.predictions/'repeat_restore_control.json');assert all(repeat[k] is True for k in ['source_repeat_bitexact','source_final_Conv_bridge_bitexact','post_all_modes_grounder_fullstate_restored','frozen_CLIP_fullstate_unchanged']);assert repeat['strict_FP32_flags']==se['strict_FP32_flags'];observations=[]
 for row in rows:
  z=S.load_prediction(a.predictions/(row['id']+'.npz'));observations.append(audit_sample(z,lookup[row['id']],plan))
 summary={}
 for platform in ('drone','quad'):
  pp=[r for r in observations if r['platform']==platform];summary[platform]={'n':len(pp),'roi_count_minmax':[min(r['roi_count'] for r in pp),max(r['roi_count'] for r in pp)] if pp else None,'caption_truncated_count':sum(r['caption_truncated'] for r in pp),'modes':{}}
  for mode in S.MODES:
   rr=[r['modes'][mode] for r in pp];summary[platform]['modes'][mode]={key:sum(r[key] for r in rr) for key in ['q_changed','accepted_any_axis','raw_consensus_nonzero','final_target64_nonzero','actual_target32_nonzero','parameter_steps','nonzero_gradient_steps','nonzero_parameter_update_runs','selected_changed_runs']};summary[platform]['modes'][mode].update(samples_with_nonzero_parameter_update=sum(r['nonzero_parameter_update_runs']>0 for r in rr),samples_with_selected_change=sum(r['selected_changed_runs']>0 for r in rr),reason_histogram=dict(collections.Counter(r['reason'] for r in rr)))
 result={'state':'R8_unlabelled_engineering_audit_pass_not_pseudo_quality_score','source_auditor_sha256':S.sha(Path(__file__)),'prediction_seal_sha256':a.expected_seal_sha256,'phase':se['phase'],'n':len(observations),'summary':summary,'max_allquery_kernel_equivalence_error_m':max(r['max_kernel_error'] for r in observations),'registered_kernel_tolerance_m':ATOL,'max_selected_interval_excess_m':max(r['max_interval_excess'] for r in observations),'hashes_all_verified':True,'matched_fresh_strictFP32_source':True,'CLIP_frozen_resources_and_restore_receipts_verified':True,'raw_target64_actualtarget32_qchange_and_paramchange_separate':True,'rows_PRIVATE':observations,'no_target_metadata_GT_scoringindex_or_task_IoU_opened':True,'no_GPU_model_or_taskscore_by_auditor':True,'quality_boundary':'Nonzero G/gradient/update are engineering evidence only. Semantic association, correlated-source consensus and pseudo-box correctness require separately released task/qualitative analysis; counts do not establish improvement.','seconds':time.perf_counter()-start};assert not a.output.exists();S.dump(a.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ['summary','rows_PRIVATE']}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ('predictions','manifest','workspace-root','output'):ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--expected-seal-sha256',required=True);args=ap.parse_args()
 try:main(args)
 except BaseException as e:
  S.dump(args.output.with_name(args.output.stem+'_FAILED_'+str(time.time_ns())+'.json'),{'state':'audit_failed_preserved_no_tolerance_relaxation','type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'auditor_sha256':S.sha(Path(__file__))});raise
