"""Post-normal-score exploratory bridge CPU scorer. Not unique TTA/holdout evidence.
Only after bridge prediction seal/root release: derived sealed GT7D for3 selected
IoU, all comparisons to sealed normal DirectG/SGD/kernel; no raw metadata/model.
"""
from pathlib import Path
import argparse,json,math,collections
import numpy as np
from score_normal import sha,read,dump,bounded
from geometry_metrics import eval_box,candidate_ious,coordinate_errors
MODES=('rgb_fulltext','pc_only');ARMS=('source',)+tuple(m+'__pseudo_exactfit' for m in MODES)

def main(a):
 assert sha(a.release)==a.release_sha256;g=read(a.release);assert g['CPU_SCORE_RELEASE'] is True and g['stage']=='pseudo_exactfit_bridge_postscore_exploratory'
 for k,p in [('bridge_prediction_seal_sha256',a.predictions/'SEALED.json'),('normal_prediction_seal_sha256',a.normal_predictions/'SEALED.json'),('normal_scored_seal_sha256',a.normal_scored/'SEALED.json'),('source_checkpoint_sha256',a.source_checkpoint),('scorer_program_sha256',Path(__file__)),('normal_scorer_program_sha256',Path(__file__).with_name('score_normal.py')),('geometry_metrics_sha256',Path(__file__).with_name('geometry_metrics.py'))]:assert g[k]==sha(p),k
 for rel,h in g['geometry_dependencies_sha256'].items():assert sha(bounded(Path(__file__).resolve().parents[3],rel))==h
 s=read(a.predictions/'SEALED.json');normal=read(a.normal_predictions/'SEALED.json');scoreseal=read(a.normal_scored/'SEALED.json')
 assert s['state']=='sealed_post_normal_score_unlabelled_pseudo_exactfit_optimizer_bridge' and s['n_samples']==160 and s['n_arms']==3
 assert s['normal_prediction_seal_sha256']==sha(a.normal_predictions/'SEALED.json')==scoreseal['normal_prediction_seal_sha256']
 assert s['same_normal_directG_targets_no_new_G'] and s['all_query_indices_same_normal_q0'] and s['no_GT_scored_files_RGB_PC_CLIP_G_or_backbone_forward_consumed'] and s['not_unique_TTA_benefit_or_heldout_claim'] and s['source_state_restored']
 assert s['source_checkpoint_sha256']==normal['source_checkpoint_sha256']==g['source_checkpoint_sha256'] and s['normal_source_cache_files']==normal['source_cache_files']
 for k in ('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark'):assert s['strict_FP32_flags'][k] is False
 for root,seal in [(a.predictions,s),(a.normal_predictions,normal),(a.normal_scored,scoreseal)]:
  for rel,h in seal['files'].items():assert sha(bounded(root,rel))==h
 for rel,h in s['program_sha256'].items():assert sha(bounded(a.program_root,rel))==h
 # No derived GT values read until complete bridge/source/code/score release above.
 GT={r['id']:r for r in read(a.normal_scored/'PRIVATE_GT7D_SIDECAR.json')['rows']};old={r['id']:r for r in read(a.normal_scored/'RESULTS_PRIVATE.json')['rows_PRIVATE']};trace={r['id']:r for r in read(a.predictions/'trace.json')};assert set(GT)==set(old)==set(trace) and len(GT)==160
 a.output.mkdir(parents=True,exist_ok=False);dump(a.output/'SCORER_LOCK.json',{'state':'all_bridge_pred_source_and_normal_score_hashes_verified_before_derived_GT','release_sha256':sha(a.release),'bridge_seal_sha256':sha(a.predictions/'SEALED.json'),'normal_scored_seal_sha256':sha(a.normal_scored/'SEALED.json'),'post_normal_score_exploratory':True,'not_unique_TTA':True})
 rows=[];maxcoord=0.
 for sid,gt in GT.items():
  with np.load(a.predictions/(sid+'.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  with np.load(a.normal_predictions/(sid+'.npz'),allow_pickle=False) as z:n={k:z[k].copy() for k in z.files}
  assert tuple(d['arm_names'].tolist())==ARMS and d['arm_boxes_raw6'].shape==(3,6) and np.isfinite(d['arm_boxes_raw6']).all()
  q=int(d['q_source']);assert q==int(n['q_source'])==old[sid]['q_source'] and np.array_equal(d['arm_query_index'],np.full(3,q))
  assert np.array_equal(d['source_boxes_raw6'],n['source_boxes_raw6']) and np.array_equal(d['source_text_scores256'],n['source_text_scores256']) and np.array_equal(d['arm_boxes_raw6'][0],n['arm_boxes_raw6'][0])
  assert np.array_equal(d['arm_boxes_eval6'],eval_box(d['arm_boxes_raw6'])) and np.array_equal(d['mask_C'],[1,1,1,0,0,0])
  expected=np.stack([n['arm_boxes_raw6'][n['arm_names'].tolist().index(m+'__direct_G')] for m in MODES]);assert np.array_equal(expected,d['normal_directG_targets_raw6'])
  residual=d['arm_boxes_raw6'][1:]-expected;assert np.array_equal(residual,d['coordinate_residual_to_normal_directG6']) and np.array_equal(d['arm_boxes_raw6'][:,3:],np.tile(d['arm_boxes_raw6'][0,3:],(3,1)))
  vals,cost=candidate_ious(eval_box(d['arm_boxes_raw6']),np.asarray(gt['GT7D_model'],float));assert abs(vals[0]-old[sid]['arms']['source']['iou'])<1e-12 and np.isfinite(vals).all()
  row={'id':sid,'platform':gt['platform'],'scene':gt['scene'],'q_source':q,'identity_diagnostic':old[sid]['identity_diagnostic'],'post_normal_score_exploratory_NOT_confirmation':True,'arms':{}}
  row['arms']['source']={'iou':float(vals[0]),'coordinate_errors':coordinate_errors(d['arm_boxes_raw6'][0],gt['GT7D_model'])}
  for mi,m in enumerate(MODES):
   chosen=d['arm_boxes_raw6'][mi+1];base=d['arm_boxes_raw6'][0];target=expected[mi];e=chosen.astype(float)-target.astype(float);delta=target.astype(float)-base.astype(float);delta_norm=float(np.linalg.norm(delta[:3]));err_norm=float(np.linalg.norm(e[:3]));linf=float(abs(e[:3]).max());maxcoord=max(maxcoord,linf)
   applicable=bool(d['coordinate_check_applicable']);passed=linf<=5e-5 if applicable else True;assert passed==bool(d['exactfit_coordinate_pass_by_mode'][mi]);tr=trace[sid]['modes'][m];assert tr['coordinate_check_applicable']==applicable and tr['coordinate_atol5e_minus5_pass']==passed
   details=tr['exactfit'];updated=bool(details.get('observed_parameter_change',False));gradient=details.get('parameter_gradient_norm_by_name',{});nonzero_grad=any(x is not None and x>0 for x in gradient.values())
   row['arms'][ARMS[mi+1]]={'iou':float(vals[mi+1]),'coordinate_errors':coordinate_errors(chosen,gt['GT7D_model']),'normal_target_delta_L2_m':delta_norm,'normal_target_nonzero':bool(np.any(delta[:3]!=0)),'target_delta_le_registered_atol':delta_norm<=5e-5,'center_residual_L2_vs_direct_m':err_norm,'center_residual_Linf_vs_direct_m':linf,'relative_center_residual_vs_target_shift':err_norm/delta_norm if delta_norm>0 else None,'coordinate_check_applicable':applicable,'coordinate_pass':passed,'real_backward_and_optimizer_step':bool(details.get('real_backward_and_optimizer_step_executed',False)),'nonzero_parameter_update':updated,'nonzero_gradient':nonzero_grad,'selected_changed_vs_source':not np.array_equal(chosen,base),'delta_vs_source':float(vals[mi+1]-vals[0]),'delta_vs_normal_directG':float(vals[mi+1]-old[sid]['arms'][m+'__direct_G']['iou']),'delta_vs_normal_param3':float(vals[mi+1]-old[sid]['arms'][m+'__param3']['iou']),'delta_vs_normal_kernel3':float(vals[mi+1]-old[sid]['arms'][m+'__outputspace3']['iou']),'status':str(d['arm_status'][mi+1])}
  rows.append(row)
 summary=[]
 for platform in ('drone','quad'):
  for pop in ('all','annotated_spatially_confirmed'):
   pp=[r for r in rows if r['platform']==platform and (pop=='all' or r['identity_diagnostic']['annotated_spatially_confirmed'])];count=len(pp)
   for arm in ARMS:
    v=[r['arms'][arm] for r in pp];summary.append({'platform':platform,'population':pop,'arm':arm,'n':count,'scene_clusters':len({r['scene'] for r in pp}),'mean_iou':math.fsum(x['iou'] for x in v)/count if count else None,'Acc25':sum(x['iou']>.25 for x in v)/count if count else None,'Acc50':sum(x['iou']>.5 for x in v)/count if count else None,'center_error_L2_m_mean':math.fsum(x['coordinate_errors']['center_L2_m'] for x in v)/count if count else None,'post_normal_score_exploratory':True})
    if arm!='source':
     defined=[x['relative_center_residual_vs_target_shift'] for x in v if x['relative_center_residual_vs_target_shift'] is not None]
     summary[-1].update({k:sum(bool(x[k]) for x in v) for k in ['normal_target_nonzero','coordinate_check_applicable','coordinate_pass','real_backward_and_optimizer_step','nonzero_parameter_update','nonzero_gradient','selected_changed_vs_source','target_delta_le_registered_atol']});summary[-1].update({'mean_delta_vs_directG':math.fsum(x['delta_vs_normal_directG'] for x in v)/count if count else None,'mean_delta_vs_normal_param3':math.fsum(x['delta_vs_normal_param3'] for x in v)/count if count else None,'max_abs_delta_vs_directG':max((abs(x['delta_vs_normal_directG']) for x in v),default=None),'relative_residual_defined_n':len(defined),'relative_residual_median_defined':float(np.median(defined)) if defined else None,'relative_residual_max_defined':max(defined,default=None),'zero_target_n':sum(not x['normal_target_nonzero'] for x in v)})
 result={'state':'post_normal_score_exploratory_pseudo_exactfit_bridge_scored','n':len(rows),'summary':summary,'max_center_coordinate_Linf_residual_vs_directG_m':maxcoord,'rows_PRIVATE':rows,'scope':'Same normal q0/direct targets; source/cache frozen. Registered after normal score exposure, no new confirmation. Fits existing pseudo target rather than new signal; DirectG no-update controls same information; not unique adaptation benefit. Nativeyaw0,noPCA,noGTyaw,noGTonline input. Drone2/Quad3scenes priorresearch-exposed.'};dump(a.output/'RESULTS_PRIVATE.json',result);dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in result.items() if k!='rows_PRIVATE'});dump(a.output/'SEALED.json',{'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n_samples':len(rows),'post_normal_score_exploratory':True})
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('predictions','normal-predictions','normal-scored','release','source-checkpoint','program-root','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);main(p.parse_args())
