"""Independent sealed oracle13 arrays/trace/coordinate/reduction audit, no new IoU."""
from pathlib import Path
import argparse,json,hashlib,math,collections
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def near(a,b,atol=1e-12,rtol=0):
 if a is None or b is None:assert a is None and b is None
 else:assert np.allclose(a,b,atol=atol,rtol=rtol),(a,b)
def main(a):
 ss=read(a.scored/'SEALED.json');ps=read(a.predictions/'SEALED.json');ns=read(a.normal_predictions/'SEALED.json');normal=read(a.normal_scored/'SEALED.json')
 assert ps['normal_prediction_seal_sha256']==sha(a.normal_predictions/'SEALED.json')==normal['normal_prediction_seal_sha256']
 assert ps['normal_source_cache_files']==ns['source_cache_files'] and ps['all_query_indices_same_normal_q0'] and ps['source_model_parameters_buffers_grads_modes_restored']
 assert ps['source_model_state_fingerprint']==ns['source_model_state_fingerprint'] and ps['no_GT_yaw_RGB_PC_G_or_backbone_forward_in_GTphase']
 for root,s in [(a.scored,ss),(a.predictions,ps),(a.normal_predictions,ns),(a.normal_scored,normal)]:
  for rel,h in s['files'].items():assert sha(root/rel)==h
 d=read(a.scored/'RESULTS_PRIVATE.json');rows=d['rows_PRIVATE'];gt={r['id']:np.array(r['GT7D_model'],float) for r in read(a.normal_scored/'PRIVATE_GT7D_SIDECAR.json')['rows']};old={r['id']:r for r in read(a.normal_scored/'RESULTS_PRIVATE.json')['rows_PRIVATE']};tr={r['id']:r for r in read(a.predictions/'trace.json')};assert len(rows)==160 and set(gt)==set(old)==set(tr)=={r['id'] for r in rows}
 counts=collections.defaultdict(collections.Counter);maxfit=collections.defaultdict(float);maxcoord=0.;maxkernel=0.;gradient_relative_error_max=0.
 for r in rows:
  sid=r['id'];t=tr[sid];assert t['normal_Conv_replay_bitexact'] and t['same_normal_source_cache_no_fresh_backbone_or_slot'] and t['full_source_model_restored_after_allGTcontrols'] and t['GTyaw_not_consumed']
  with np.load(a.predictions/(sid+'.npz'),allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
  with np.load(a.normal_predictions/(sid+'.npz'),allow_pickle=False) as z:n={k:z[k].copy() for k in z.files}
  cachefile='source_cache/'+sid+'.npz'
  with np.load(a.normal_predictions/cachefile,allow_pickle=False) as z:cache={k:z[k].copy() for k in z.files}
  q=int(p['q_source']);source=p['source_boxes_raw6'][q].astype(float);target=p['target6_modelcoords'].astype(float);names=p['arm_names'].tolist();boxes=p['arm_boxes_raw6'].astype(float)
  assert q==int(n['q_source'])==int(cache['query_index'])==old[sid]['q_source'] and (p['arm_query_index']==q).all() and np.array_equal(p['source_boxes_raw6'],n['source_boxes_raw6']) and np.array_equal(p['source_boxes_raw6'],cache['boxes_raw6'][0])
  assert p['normal_source_cache_sha256'].tobytes().hex()==ns['source_cache_files'][cachefile]
  assert np.array_equal(target,gt[sid][:6]);near(r['arms']['source']['iou'],old[sid]['arms']['source']['iou']);assert r['identity_diagnostic']==old[sid]['identity_diagnostic']
  for j,name in enumerate(names):
   e=boxes[j]-target;ce=r['arms'][name]['coordinate_errors'];expected={'center_L2_m':np.linalg.norm(e[:3]),'size_L2_m':np.linalg.norm(e[3:]),'coordinate_Linf_m':abs(e).max(),'center_abs_error_m':abs(e[:3]),'size_abs_error_m':abs(e[3:]),'center_L2_gt_dimension_normalized':np.linalg.norm(e[:3]/target[3:]),'size_relative_abs_mean':np.mean(abs(e[3:])/target[3:])}
   for key,value in expected.items():near(ce[key],value);maxcoord=max(maxcoord,float(np.max(abs(np.asarray(ce[key])-value))))
  for group,mask in [('C',np.array([1,1,1,0,0,0],bool)),('S',np.array([0,0,0,1,1,1],bool)),('CS',np.ones(6,bool))]:
   key=r['platform']+'|'+group;g=t['groups'][group];counts[key]['n']+=1;direct=np.where(mask,target,source);active_error=(source-target)*mask
   for kind in ('direct_target','param1','param3','exactfit'):
    arm='GT_'+group+'__'+kind;j=names.index(arm);assert np.array_equal(boxes[j,~mask],source[~mask]) and np.array_equal(p['constructed_target6_by_arm'][j],direct) and np.array_equal(p['arm_coordinate_masks6'][j],mask.astype(np.float32));assert np.array_equal(p['coordinate_residual6_by_arm'][j],(boxes[j].astype(np.float32)-direct.astype(np.float32)))
    cap=r['capacity'][group]['controls']['direct' if kind=='direct_target' else kind];res=boxes[j]-direct;near(cap['supervised_coordinate_Linf_m'],abs(res[mask]).max());near(cap['supervised_coordinate_L2_m'],np.linalg.norm(res[mask]))
    if kind=='direct_target':assert np.array_equal(boxes[j],direct)
   exact=g['exactfit'];assert exact['real_backward_and_optimizer_step_executed'] and exact['selected_source_parameters_restored'] and exact['no_output_override_used'] and exact['inactive_coordinates_bitexact_source']
   err=float(abs(boxes[names.index('GT_'+group+'__exactfit')]-direct).max());maxfit[key]=max(maxfit[key],err);assert err<=5e-5 and g['exactfit_atol5e_minus5_pass'];near(exact['supervised_max_abs_coordinate_residual_m'],err)
   counts[key]['exactfit_nonzero_parameter_update']+=int(exact['observed_parameter_change']);counts[key]['exactfit_nonzero_gradient']+=int(any(x is not None and x>0 for x in exact['parameter_gradient_norm_by_name'].values()));counts[key]['exactfit_true_steps']+=1
   for head,active,h in [('center',mask[0],cache['center_features'][0,:,q].astype(float)),('size',mask[3],cache['size_features'][0,:,q].astype(float))]:
    prefix='prediction_heads.5.'+('center_residual_head' if head=='center' else 'size_pred_head')+'.net.8.'
    if not active:
     assert all(exact['parameter_delta_max_by_name'][prefix+s]==0 for s in ('weight','bias'))
    else:
     kernel=exact['separate_head_feature_kernel'][head];near(exact['actual_head_learning_rates'][head],1/kernel);near(kernel,float(h@h+1),atol=1e-6,rtol=2e-6)
     block=active_error[:3] if head=='center' else active_error[3:];expected_bias=float(np.linalg.norm(block));observed=exact['parameter_gradient_norm_by_name'][prefix+'bias'];near(observed,expected_bias,atol=1e-5,rtol=5e-6);gradient_relative_error_max=max(gradient_relative_error_max,abs(observed-expected_bias)/max(expected_bias,1e-30))
     near(exact['parameter_gradient_norm_by_name'][prefix+'weight'],expected_bias*np.linalg.norm(h),atol=1e-5,rtol=5e-6)
   for k in (1,3):
    steps=g['SGD'+str(k)]['trace'];assert len(steps)==k;counts[key]['SGD'+str(k)+'_steps']+=len(steps)
    for z in steps:
     assert z['parameter_gradient_norm']>=0 and math.isfinite(z['loss']);counts[key]['SGD'+str(k)+'_nonzero_gradsteps']+=int(z['parameter_gradient_norm']>0);maxkernel=max(maxkernel,z['output_space_replay_max_abs_error']);assert z['output_space_replay_max_abs_error']<=5e-5
     if group!='CS':
      inactive_head='size_pred_head' if group=='C' else 'center_residual_head';assert all(v==0 for n,v in z['parameter_delta_max_by_name'].items() if inactive_head in n)
    near(steps[-1]['selected_box_after'],boxes[names.index('GT_'+group+'__param'+str(k))])
 assert ps['all_exactfit_coordinate_checks_pass'] and d['all_exactfit_within_registered5e5']
 checked=0
 for s in d['summary']:
  pp=[r for r in rows if r['platform']==s['platform'] and (s['population']=='all' or r['identity_diagnostic']['annotated_spatially_confirmed'])];n=len(pp);assert s['n']==n and s['scene_clusters']==len({r['scene'] for r in pp});arm=s['arm'];vals=[r['arms'][arm]['iou'] for r in pp]
  near(s['mean_iou'],math.fsum(vals)/n if n else None);near(s['Acc25'],sum(v>.25 for v in vals)/n if n else None);near(s['Acc50'],sum(v>.5 for v in vals)/n if n else None)
  for field,key in [('center_L2_m_mean','center_L2_m'),('size_L2_m_mean','size_L2_m')]:near(s[field],math.fsum(r['arms'][arm]['coordinate_errors'][key] for r in pp)/n if n else None)
  if arm!='source':
   group,kind=arm[3:].split('__');key='direct' if kind=='direct_target' else kind;values=[r['capacity'][group]['controls'][key]['supervised_coordinate_Linf_m'] for r in pp];near(s['mean_supervised_Linf_m_vs_direct_target'],math.fsum(values)/n if n else None);near(s['max_supervised_Linf_m_vs_direct_target'],max(values,default=None))
  checked+=7
 out={'state':'independent_oracle13_samecache_coordinate_and_reduction_audit_pass','auditor_sha256':sha(Path(__file__)),'GT_prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'GT_results_sha256':sha(a.scored/'RESULTS_PRIVATE.json'),'n':160,'summary_fields_checked':checked,'max_coordinate_recalculation_error_m':maxcoord,'max_exactfit_error_by_platform_group_m':dict(maxfit),'true_gradient_parameter_step_counts':{k:dict(v) for k,v in counts.items()},'max_SmoothL1_allquery_kernel_error_m':maxkernel,'max_exactfit_bias_gradient_relative_discrepancy':gradient_relative_error_max,'same_normal_forward_query_cache_and_inactive_coordinate_preservation_verified':True,'source_restoration_scope':'Actual sealed trace+sourcefullstate fingerprint and assertions; reviewer did not reinstantiate GPU model.','no_new_IoU_GPU_model_original_metadata_or_GT_annotation_file_read':True,'derived_GT7D_sidecar_used_for_coordinate_audit':True,'claim_scope':'Supervised oracle control, not TTA or taskIoU upperbound. Confirmed subset13Drone/47Quad is post-GT spatial conditional with sourceIoU>.25, not semantic identity proof or population estimate.'};assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('scored','predictions','normal-predictions','normal-scored','output'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
