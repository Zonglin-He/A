"""Independent all-observation frame and scientific decision audit."""
from common71 import *
methods=read(H/'ALL_METHOD_ROWS.json');scenes=read(H/'ALL_SCENE_ROWS.json');frames=read(H/'ALL_FRAME_CLUSTER_ROWS.json');fb=read(H/'FRAME_BALANCED_SUMMARY.json');diag=read(H/'COHORT_DIAGNOSTICS.json');decision=read(H/'DECISION.json');media=read(ROOT/'research/round69/input/SELECTED_MEDIA_HASHES_PRIVATE.json');checks=0
for name,co,n in [('val1280','VAL1280',1280),('train1024','TRAIN1024',1024),('dev336','DEV336_DIAGNOSTIC',336)]:
 tt=read(H/('score_'+name+'_attempt001/ALL_TASK_ROWS_PRIVATE.json'));groups={}
 for t in tt:
  m=media[t['id_PRIVATE']] if co!='DEV336_DIAGNOSTIC' else read(ROOT/'crossplatform/round70/prepared336_attempt001/cases_PRIVATE'/(t['id_PRIVATE']+'.json'));groups.setdefault((t['platform'],m['image_sha256'],m['point_sha256']),[]).append(t)
 for k in ARMS:
  row=next(x for x in methods if x['cohort']==co and x['method']==k);a=np.array([x['IoU'][k] for x in tt]);assert abs(row['balanced_mIoU']-a.mean()*100)<1e-12;assert row['n']==n
  for ref in REFS:
   d=a-np.array([x['IoU'][ref] for x in tt]);assert abs(row['gain_vs_'+ref+'_pp']-d.mean()*100)<1e-12;assert row['improved_vs_'+ref]==int((d>1e-10).sum()) and row['worsened_vs_'+ref]==int((d< -1e-10).sum());assert abs(row['largest_positive_vs_'+ref+'_pp']-max(0,d.max())*100/n)<1e-12;checks+=4
  vals={pl:[] for pl in ['drone','quad']}
  for i,((pl,_,__),rr) in enumerate(sorted(groups.items())):
   alias=co+'_'+pl+'_F'+str(i+1).zfill(5);v=np.mean([x['IoU'][k] for x in rr]);z=next(z for z in frames if z['cohort']==co and z['method']==k and z['frame_alias']==alias);assert abs(v-z['mean_frame_IoU'])<1e-12 and z['descriptions']==len(rr);vals[pl].append(v);checks+=1
  z=next(x for x in fb if x['cohort']==co and x['method']==k);assert abs(z['equal_frame_then_platform_mIoU']-np.mean([np.mean(v) for v in vals.values()])*100)<1e-12;checks+=1
 dd=np.array([x['IoU']['CLIP_direct']-x['IoU']['source'] for x in tt]);z=next(x for x in diag if x['cohort']==co);assert z['unique_media_frames']==len(groups);assert abs(z['CLIP_gain_vs_source_pp']-dd.mean()*100)<1e-12;assert z['improved_vs_source']==int((dd>1e-10).sum()) and z['worsened_vs_source']==int((dd< -1e-10).sum());checks+=4
for field,ref in [('CLIP_direct_over_source_both_expanded_cohorts','source'),('CLIP_direct_over_R59_both_expanded_cohorts','R59'),('CLIP_direct_over_matched_mismatch_both_expanded_cohorts','CLIP_mismatch')]:
 expected=all(next(x['balanced_mIoU'] for x in methods if x['cohort']==co and x['method']=='CLIP_direct')>next(x['balanced_mIoU'] for x in methods if x['cohort']==co and x['method']==ref) for co in ['VAL1280','TRAIN1024']);assert decision[field]==expected
assert decision['parameter_TTA_tested'] is False and decision['FINAL_access'] is False and decision['no_method_parameter_or_data_selection_after_results'] is True
out=H/'independent_derived_audit_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R71_INDEPENDENT_FRAME_METHOD_DECISION_DIAGNOSTIC_TABLES','n':2640,'checked_rows_and_metrics':checks,'no_new_GT_or_models':True};dump(out/'AUDIT_RESULT.json',rec);print(rec)
