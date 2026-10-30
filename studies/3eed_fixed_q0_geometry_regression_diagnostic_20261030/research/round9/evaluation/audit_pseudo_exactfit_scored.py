"""Independent actual3-arm bridge audit, sealed arrays/rows only; no new IoU."""
from pathlib import Path
import argparse,json,hashlib,math,collections
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def near(a,b,atol=1e-12,rtol=0):
 if a is None or b is None:assert a is None and b is None,(a,b)
 else:assert np.allclose(a,b,atol=atol,rtol=rtol),(a,b)
def mean(v):return math.fsum(v)/len(v) if v else None

def main(a):
 ss=read(a.scored/'SEALED.json');ps=read(a.predictions/'SEALED.json');ns=read(a.normal_predictions/'SEALED.json');normal=read(a.normal_scored/'SEALED.json')
 assert ps['normal_prediction_seal_sha256']==sha(a.normal_predictions/'SEALED.json')==normal['normal_prediction_seal_sha256'] and ps['normal_source_cache_files']==ns['source_cache_files']
 assert ps['same_normal_directG_targets_no_new_G'] and ps['all_query_indices_same_normal_q0'] and ps['source_state_restored'] and ps['post_normal_score_exploratory_optimizer_bridge'] and ps['no_GT_scored_files_RGB_PC_CLIP_G_or_backbone_forward_consumed']
 assert ps['source_model_state_fingerprint']==ns['source_model_state_fingerprint'] and ps['strict_FP32_flags']==ns['strict_FP32_flags']
 for root,s in [(a.scored,ss),(a.predictions,ps),(a.normal_predictions,ns),(a.normal_scored,normal)]:
  for rel,h in s['files'].items():assert sha(root/rel)==h
 d=read(a.scored/'RESULTS_PRIVATE.json');rows=d['rows_PRIVATE'];gt={r['id']:np.array(r['GT7D_model'],float) for r in read(a.normal_scored/'PRIVATE_GT7D_SIDECAR.json')['rows']};old={r['id']:r for r in read(a.normal_scored/'RESULTS_PRIVATE.json')['rows_PRIVATE']};tr={r['id']:r for r in read(a.predictions/'trace.json')};assert len(rows)==160 and set(gt)==set(old)==set(tr)=={r['id'] for r in rows}
 counts=collections.defaultdict(collections.Counter);maxresidual=0.;maxcoord=0.;max_IoU_difference=0.;ratios=collections.defaultdict(list);meandiff=collections.defaultdict(list)
 for r in rows:
  sid=r['id'];t=tr[sid];assert t['normal_Conv_replay_bitexact'] and t['full_source_model_restored'] and t['post_normal_score_exploratory']
  with np.load(a.predictions/(sid+'.npz'),allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
  with np.load(a.normal_predictions/(sid+'.npz'),allow_pickle=False) as z:n={k:z[k].copy() for k in z.files}
  q=int(p['q_source']);assert q==int(n['q_source'])==old[sid]['q_source'] and (p['arm_query_index']==q).all()
  assert np.array_equal(p['source_boxes_raw6'],n['source_boxes_raw6']) and np.array_equal(p['source_text_scores256'],n['source_text_scores256']) and np.array_equal(p['arm_boxes_raw6'][0],n['arm_boxes_raw6'][0])
  assert p['normal_source_cache_sha256'].tobytes().hex()==ns['source_cache_files']['source_cache/'+sid+'.npz'] and p['normal_prediction_file_sha256'].tobytes().hex()==ns['files'][sid+'.npz']
  assert r['identity_diagnostic']==old[sid]['identity_diagnostic'];near(r['arms']['source']['iou'],old[sid]['arms']['source']['iou'])
  source=p['arm_boxes_raw6'][0].astype(float);names=p['arm_names'].tolist();assert tuple(names)==('source','rgb_fulltext__pseudo_exactfit','pc_only__pseudo_exactfit')
  for j,arm in enumerate(names):
   box=p['arm_boxes_raw6'][j].astype(float);error=box-gt[sid][:6];ce=r['arms'][arm]['coordinate_errors'];expected={'center_L2_m':np.linalg.norm(error[:3]),'size_L2_m':np.linalg.norm(error[3:]),'coordinate_Linf_m':abs(error).max(),'center_abs_error_m':abs(error[:3]),'size_abs_error_m':abs(error[3:]),'center_L2_gt_dimension_normalized':np.linalg.norm(error[:3]/gt[sid][3:6]),'size_relative_abs_mean':np.mean(abs(error[3:])/gt[sid][3:6])}
   for k,v in expected.items():near(ce[k],v);maxcoord=max(maxcoord,float(np.max(abs(np.asarray(ce[k])-v))))
   assert np.array_equal(box[3:],source[3:])
  for mi,mode in enumerate(('rgb_fulltext','pc_only')):
   key=r['platform']+'|'+mode;counts[key]['n']+=1;ra=r['arms'][mode+'__pseudo_exactfit'];target=n['arm_boxes_raw6'][n['arm_names'].tolist().index(mode+'__direct_G')].astype(float);chosen=p['arm_boxes_raw6'][mi+1].astype(float);delta=target-source;residual=chosen-target;dn=float(np.linalg.norm(delta[:3]));rn=float(np.linalg.norm(residual[:3]));linf=float(abs(residual[:3]).max());maxresidual=max(maxresidual,linf)
   assert np.array_equal(p['normal_directG_targets_raw6'][mi],target) and np.array_equal(p['coordinate_residual_to_normal_directG6'][mi],(chosen.astype(np.float32)-target.astype(np.float32)))
   nonzero=bool(np.any(delta[:3]!=0));relative=rn/dn if dn>0 else None;near(ra['normal_target_delta_L2_m'],dn);near(ra['center_residual_L2_vs_direct_m'],rn);near(ra['center_residual_Linf_vs_direct_m'],linf);near(ra['relative_center_residual_vs_target_shift'],relative);assert ra['normal_target_nonzero']==nonzero and ra['target_delta_le_registered_atol']==(dn<=5e-5)
   applicable=bool(p['coordinate_check_applicable']);assert ra['coordinate_check_applicable']==applicable and ra['coordinate_pass']==bool(p['exactfit_coordinate_pass_by_mode'][mi]) and (linf<=5e-5 or not applicable)
   trace=t['modes'][mode];exact=trace['exactfit'];assert trace['pseudo_equals_source']==(not nonzero) and trace['coordinate_check_applicable']==applicable
   steps=bool(exact.get('real_backward_and_optimizer_step_executed',False));update=bool(exact.get('observed_parameter_change',False));grad=any(x is not None and x>0 for x in exact.get('parameter_gradient_norm_by_name',{}).values());selected=not np.array_equal(chosen,source)
   assert ra['real_backward_and_optimizer_step']==steps and ra['nonzero_parameter_update']==update and ra['nonzero_gradient']==grad and ra['selected_changed_vs_source']==selected
   if applicable:
    assert exact['selected_source_parameters_restored'] and exact['inactive_coordinates_bitexact_source'] and exact['no_output_override_used'];assert all(v==0 for name,v in exact['parameter_delta_max_by_name'].items() if 'size_pred_head' in name)
    near(exact['supervised_max_abs_coordinate_residual_m'],linf)
   for name,value in [('target_nonzero',nonzero),('zero_target',not nonzero),('actual_backward_step',steps),('nonzero_parameter_update',update),('nonzero_gradient',grad),('selected_changed',selected),('coordinate_check_applicable',applicable),('coordinate_pass',ra['coordinate_pass'])]:counts[key][name]+=int(value)
   if nonzero:ratios[key].append(relative)
   else:assert rn==0 and not update and not grad and not selected
   for field,reference in [('delta_vs_source','source'),('delta_vs_normal_directG',mode+'__direct_G'),('delta_vs_normal_param3',mode+'__param3'),('delta_vs_normal_kernel3',mode+'__outputspace3')]:near(ra[field],ra['iou']-old[sid]['arms'][reference]['iou'])
   meandiff[key].append(ra['delta_vs_normal_directG']);max_IoU_difference=max(max_IoU_difference,abs(ra['delta_vs_normal_directG']))
 checked=0
 for s in d['summary']:
  pp=[r for r in rows if r['platform']==s['platform'] and (s['population']=='all' or r['identity_diagnostic']['annotated_spatially_confirmed'])];n=len(pp);v=[r['arms'][s['arm']] for r in pp];assert n==s['n'] and s['scene_clusters']==len({r['scene'] for r in pp});near(s['mean_iou'],mean([x['iou'] for x in v]));near(s['Acc25'],sum(x['iou']>.25 for x in v)/n if n else None);near(s['Acc50'],sum(x['iou']>.5 for x in v)/n if n else None);near(s['center_error_L2_m_mean'],mean([x['coordinate_errors']['center_L2_m'] for x in v]));checked+=5
  if s['arm']!='source':
   for field in ('normal_target_nonzero','coordinate_check_applicable','coordinate_pass','real_backward_and_optimizer_step','nonzero_parameter_update','nonzero_gradient','selected_changed_vs_source','target_delta_le_registered_atol'):assert s[field]==sum(bool(x[field]) for x in v);checked+=1
   defined=[x['relative_center_residual_vs_target_shift'] for x in v if x['relative_center_residual_vs_target_shift'] is not None];assert s['relative_residual_defined_n']==len(defined) and s['zero_target_n']==sum(not x['normal_target_nonzero'] for x in v);near(s['relative_residual_median_defined'],float(np.median(defined)) if defined else None);near(s['relative_residual_max_defined'],max(defined,default=None));near(s['mean_delta_vs_directG'],mean([x['delta_vs_normal_directG'] for x in v]));near(s['mean_delta_vs_normal_param3'],mean([x['delta_vs_normal_param3'] for x in v]));near(s['max_abs_delta_vs_directG'],max((abs(x['delta_vs_normal_directG']) for x in v),default=None));checked+=7
 out={'state':'independent_actual_bridge3_samecache_target_coordinate_and_reduction_audit_pass','auditor_sha256':sha(Path(__file__)),'bridge_prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'bridge_results_sha256':sha(a.scored/'RESULTS_PRIVATE.json'),'n':160,'summary_fields_checked':checked,'counts':{k:dict(v) for k,v in counts.items()},'max_coordinate_recalculation_error_m':maxcoord,'max_center_Linf_residual_to_normal_directG_m':maxresidual,'relative_residual_nonzero_target_only':{k:{'defined_n':len(v),'median':float(np.median(v)),'max':max(v)} for k,v in ratios.items()},'mean_IoU_bridge_minus_directG':{k:mean(v) for k,v in meandiff.items()},'max_abs_per_sample_IoU_bridge_minus_directG':max_IoU_difference,'same_q0_normalcache_sourcepool_shapeprior_directtargets_verified':True,'source_restore_scope':'Source fingerprint/sealed actual trace/assertions; no new GPU model instantiated.','no_new_IoU_GPU_model_original_metadata_or_GT_annotation_file_read':True,'derived_sealed_GT7D_sidecar_used_for_coordinate_recalculation':True,'mechanism_scope':'Registered after normal scores, exploratory optimizer bridge. Demonstrates fitting existing pseudo target; DirectG reproduces task gain, not unique adaptation or fresh confirmation. Exactfit changes L2+headnorm step together, not isolated LR causality.'};assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('scored','predictions','normal-predictions','normal-scored','output'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
