"""Cached-only mutually exclusive skips and full decision changes, before new experiments."""
from common132 import *
import numpy as np,collections
def numeric_failure(rec):
 status=rec.get('status','missing_status');p=rec.get('RGB_validation',{});c=rec.get('holdout_residuals',{});cfg=rec.get('config',{});bad=[]
 if rec.get('T_raw_B_to_A') is None:
  if status=='PnP_insufficient_correspondences':
   if 'n_pairs' not in rec:return 'unrecorded_visible_depth_or_correspondence_shortage',['legacy_no_visible_depth_and_RGB_match_counts']
   if rec.get('n_fit',0)<cfg.get('minimum_fit',10):bad.append('insufficient_fit_correspondences')
   if rec.get('n_holdout',0)<cfg.get('minimum_holdout',6):bad.append('insufficient_spatial_holdout_correspondences')
  else:bad.append(status)
 else:
  if p.get('n',0)<cfg.get('minimum_holdout',6):bad.append('insufficient_RGB_validation')
  if p.get('median_px',float('inf'))>cfg.get('maximum_pixel_median',3.):bad.append('RGB_median_reprojection')
  if p.get('inlier_fraction',0)<cfg.get('minimum_pixel_inlier_fraction',.6):bad.append('RGB_inlier_fraction')
  if c.get('n',0)<cfg.get('minimum_cloud_holdout',20):bad.append('insufficient_cloud_holdout')
  if c.get('median_m',float('inf'))>cfg.get('maximum_cloud_median_m',.15):bad.append('cloud_median_residual')
  if c.get('inlier_fraction',0)<cfg.get('minimum_cloud_inlier_fraction',.6):bad.append('cloud_inlier_fraction')
 return (bad[0] if bad else status),bad
def main():
 basecheck();reg=rd(R132/'research/round131/INPUT_REGISTRY_PRIVATE.json');anchors={a['A']['frame_PRIVATE']:a for a in rd(R132/'research/round131/ANCHORS_PRIVATE.json')};tau=rd(R132/'research/round129/SOURCE_CALIBRATION.json')['tau'];rows=[];pairrows=[]
 for a in anchors.values():
  if a['B_pair'] is None:pairrows.append({'frame_PRIVATE':a['A']['frame_PRIVATE'],'target':a['A']['platform'],'stage':'no_candidate_B','reason':'no_candidate_B','all_failed_gates':[]});continue
  rec=rd(R132/'crossplatform/round131/registration_PRIVATE'/(a['B_pair']+'.json'));status=rec['status']
  if not rec['geometry_cycle_supported']:
   stage='forward' if status=='forward_sensor_geometry_unsupported' else ('reverse' if status=='reverse_sensor_geometry_unsupported' else 'cycle_or_duplicate');reason,bad=numeric_failure(rec.get(stage,{})) if stage in ['forward','reverse'] else (status,[status])
  else:stage='registered';reason='geometry_cycle_supported';bad=[]
  pairrows.append({'frame_PRIVATE':a['A']['frame_PRIVATE'],'target':a['A']['platform'],'stage':stage,'reason':reason,'all_failed_gates':bad})
 pairmap={p['frame_PRIVATE']:p for p in pairrows}
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];D=R132/'crossplatform/round131/formal'/name;rec=rd(D/'records_PRIVATE'/(sid+'.json'))
  with np.load(D/'samples_PRIVATE'/(sid+'.npz')) as p:probs=dict(zip(p['arm_names'],p['scores']));p0=probs['zero_F'];pa=probs['partial_F'];q0=int(p0.argmax());qa=int(pa.argmax());boxes=p['source_boxes'].copy()
  with np.load(D/'features_PRIVATE'/(sid+'.npz')) as f:S=f['S'];C=f['C'];ass=f['assignment'];ng=int(ass.max()+1);g=[np.flatnonzero(ass==i) for i in range(ng)];pv=[];cv=[]
  for tag in ['pA','pB']:
   with np.load(D/'features_PRIVATE'/(sid+'.npz')) as f:p=f[tag]
   gp=np.array([p[x].mean() for x in g])[S];gp=gp/gp.sum() if len(gp) else gp;pv.append(gp);cv.append(gp.max()-gp<=tau if len(gp) else np.array([],bool))
  if not rec['B_available']:reason='no_usable_B';detail=pairmap[e['frame_PRIVATE']]['reason'] if pairmap[e['frame_PRIVATE']]['stage']!='registered' else 'no_valid_seed_after_registration'
  elif S.sum()<2:reason='insufficient_S';detail='fewer_than_two_supported_groups'
  elif np.array_equal(C,S):
   reason='C_equals_S';allA=bool(cv[0].all());allB=bool(cv[1].all());detail='both_views_all' if allA and allB else ('A_only_all' if allA else ('B_only_all' if allB else 'union_fills_complementary_exclusions'))
  else:reason='actual_update';detail='partial_task_nonzero'
  assert (reason=='actual_update')==rec['partial_eligible'];assert reason!='C_equals_S' or np.all(cv[0]|cv[1])
  # Fixed competitor: strongest OTHER geometry group before updating, never chosen with GT.
  alternatives=np.flatnonzero(ass!=ass[q0]);j=int(alternatives[np.argmax(p0[alternatives])]) if len(alternatives) else int(np.argsort(p0)[-2]);margin0=float(np.log(p0[q0])-np.log(p0[j]));margin1=float(np.log(pa[q0])-np.log(pa[j]));dm=margin1-margin0;rawdiff=np.log(pa)-np.log(p0);center=rawdiff-rawdiff.mean();q_changed=q0!=qa;group_changed=bool(ass[q0]!=ass[qa]);tr=rec['traces']['partial'];stepfr=[t['fraction'] for t in tr if t['accepted']]
  with np.load(D/'states_PRIVATE'/(sid+'.npz')) as f:delta=f['partial']
  rows.append({'source':e['source'],'target':e['target'],'id_PRIVATE':sid,'frame_PRIVATE':e['frame_PRIVATE'],'content_group_PRIVATE':e['content_group_PRIVATE'],'scene_PRIVATE':e['row']['scene'],'reason':reason,'detail':detail,'B_available':rec['B_available'],'S_groups':int(S.sum()),'C_groups':int(C.sum()),'A_probability_max':float(pv[0].max()) if len(pv[0]) else None,'B_probability_max':float(pv[1].max()) if len(pv[1]) else None,'A_probability_range':float(np.ptp(pv[0])) if len(pv[0]) else None,'B_probability_range':float(np.ptp(pv[1])) if len(pv[1]) else None,'A_max_below_tau':bool(len(pv[0]) and pv[0].max()<=tau),'B_max_below_tau':bool(len(pv[1]) and pv[1].max()<=tau),'query_changed':q_changed,'geometry_group_changed':group_changed,'same_group_query_switch':q_changed and not group_changed,'exact_box_changed':not np.array_equal(boxes[q0],boxes[qa]),'maximum_probability_change':float(abs(pa-p0).max()),'centered_log_score_change_L2':float(np.linalg.norm(center)),'parameter_L2':float(np.linalg.norm(delta)),'accepted_steps':len(stepfr),'shrunk_steps':sum(v<1 for v in stepfr),'min_accepted_fraction':min(stepfr) if stepfr else None,'gradient_L2_first':tr[0]['gradient_L2'] if tr else 0.,'margin_original_top_vs_best_other_group_before':margin0,'margin_original_top_vs_best_other_group_after':margin1,'margin_change':dm,'margin_direction':'unchanged_within_1e-8' if abs(dm)<=1e-8 else ('shrinking' if dm<0 else 'widening'),'q0_in_S':bool(S[ass[q0]]),'q0_in_C':bool(C[ass[q0]])})
 dump(H132/'DIAGNOSIS_WITHOUT_GT_PRIVATE.json',rows);dump(H132/'PAIR_DIAGNOSIS_PRIVATE.json',pairrows);dump(H132/'DIAGNOSIS_NO_GT_SEAL.json',{'n':len(rows),'files':{p.name:sha(p) for p in [H132/'DIAGNOSIS_WITHOUT_GT_PRIVATE.json',H132/'PAIR_DIAGNOSIS_PRIVATE.json']},'new_models_or_GT':False})
 # Existing sealed R131 scalar labels only append task descriptions after the reason/decision table is fixed.
 old={(r['source'],r['target'],r['id_PRIVATE']):r for f in (R132/'research/round131/scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)}
 for r in rows:
  a=old[(r['source'],r['target'],r['id_PRIVATE'])]['arms'];d=a['partial_F']['IoU']-a['zero_F']['IoU'];r.update(zero_IoU=a['zero_F']['IoU'],partial_IoU=a['partial_F']['IoU'],iou_changed=abs(d)>1e-12,IoU_change=d)
 summary={'n':len(rows),'reasons':dict(collections.Counter(r['reason'] for r in rows)),'details':dict(collections.Counter(r['detail'] for r in rows)),'unique_pair_stages':dict(collections.Counter(r['stage'] for r in pairrows)),'unique_pair_first_failure_reasons':dict(collections.Counter(r['stage']+'__'+r['reason'] for r in pairrows)),'parameter_changed':sum(r['parameter_L2']>0 for r in rows),'accepted_steps':sum(r['accepted_steps'] for r in rows),'shrunk_steps':sum(r['shrunk_steps'] for r in rows),'query_changed':sum(r['query_changed'] for r in rows),'geometry_group_changed':sum(r['geometry_group_changed'] for r in rows),'same_group_query_switch':sum(r['same_group_query_switch'] for r in rows),'exact_box_changed':sum(r['exact_box_changed'] for r in rows),'iou_changed':sum(r['iou_changed'] for r in rows),'active_margin_directions':dict(collections.Counter(r['margin_direction'] for r in rows if r['reason']=='actual_update')),'C_equals_S_with_either_max_below_tau':sum(r['reason']=='C_equals_S' and (r['A_max_below_tau'] or r['B_max_below_tau']) for r in rows),'prior_predictions_and_scalars_only':True,'new_backbone_or_GT_scoring':False,'FINAL_access':False}
 assert summary['reasons']=={'no_usable_B':1034,'C_equals_S':600,'actual_update':164};dump(H132/'DIAGNOSIS_ROWS_PRIVATE.json',rows);dump(H132/'DIAGNOSIS_SUMMARY.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
