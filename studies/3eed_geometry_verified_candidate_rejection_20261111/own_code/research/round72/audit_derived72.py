from common72 import *
from collections import defaultdict
M=read(H/'ALL_METHOD_ROWS.json');D=read(H/'COHORT_DIAGNOSTICS.json');C=read(H/'ALL_COVERAGE_ROWS.json');F=read(H/'ALL_FRAME_CLUSTER_ROWS.json');FB=read(H/'FRAME_BALANCED_SUMMARY.json');S=read(H/'ALL_SCENE_ROWS.json');checks=0
for name,co,n in [('val1280','VAL1280',1280),('train1024','TRAIN1024',1024),('dev336','DEV336_DIAGNOSTIC',336)]:
 tt=read(H/('score_'+name+'_attempt001/ALL_TASK_ROWS_PRIVATE.json'));cc=[x for x in C if x['cohort']==co];dd=next(x for x in D if x['cohort']==co);groups=defaultdict(list)
 assert dd['n']==n==len(tt)==len(cc)
 for field,expression in [('supported_inputs',lambda x:x['supported_groups']>0),('active_inputs',lambda x:x['main_rejected_groups']>0),('box_active_inputs',lambda x:x['box_rejected_groups']>0),('source_top1_rejected',lambda x:x['source_top1_rejected']),('rejected_groups',lambda x:x['main_rejected_groups']),('localization_good25_rejected_queries',lambda x:x['localization_good25_rejected_queries'])]:assert dd[field]==sum(int(expression(x)) for x in cc);checks+=1
 for t in tt:
  c=read(ROOT/'crossplatform/round72/evidence2640_camera_contract_attempt001/cases_PRIVATE'/(t['id_PRIVATE']+'.json'));groups[(t['platform'],c['image_sha256'],c['point_sha256'])].append(t)
 assert len(groups)==dd['unique_media_frames']
 for arm in ARMS:
  row=next(x for x in M if x['cohort']==co and x['method']==arm);values=np.array([x['IoU'][arm] for x in tt]);assert abs(row['balanced_mIoU']-values.mean()*100)<1e-12;assert row['query_changes_vs_source']==sum(x['queries'][arm]!=x['queries']['source'] for x in tt);assert row['improved_vs_source']==sum(x['IoU'][arm]>x['IoU']['source']+1e-10 for x in tt);assert row['worsened_vs_source']==sum(x['IoU'][arm]<x['IoU']['source']-1e-10 for x in tt);checks+=4
  vals=defaultdict(list)
  for i,((pl,_,__),rr) in enumerate(sorted(groups.items())):
   v=float(np.mean([x['IoU'][arm] for x in rr]));f=next(x for x in F if x['cohort']==co and x['method']==arm and x['frame_alias']==co+'_F'+str(i+1).zfill(5));assert f['descriptions']==len(rr) and abs(f['mean_IoU']-v)<1e-12;vals[pl].append(v);checks+=1
  f=next(x for x in FB if x['cohort']==co and x['method']==arm);assert abs(f['equal_frame_then_platform_mIoU']-np.mean([np.mean(x) for x in vals.values()])*100)<1e-12;checks+=1
  for pl in ['drone','quad']:
   for j,scene in enumerate(sorted({x['scene_PRIVATE'] for x in tt if x['platform']==pl})):
    sr=[x for x in tt if x['platform']==pl and x['scene_PRIVATE']==scene];row=next(x for x in S if x['cohort']==co and x['method']==arm and x['platform']==pl and x['scene_alias']==pl[0].upper()+'_S'+str(j+1).zfill(2));assert row['n']==len(sr) and abs(row['mIoU']-np.mean([x['IoU'][arm] for x in sr])*100)<1e-12;checks+=1
assert sum(x['active_inputs'] for x in D)==5 and sum(x['source_top1_rejected'] for x in D)==0
decision=read(H/'DECISION.json');assert not decision['matching_TTA_improves_task_over_direct'];assert not decision['direct_geometry_improves_source'];assert decision['FINAL_access'] is False
out=H/'independent_derived_audit_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R72_INDEPENDENT_COVERAGE_FRAME_SCENE_AND_DECISION_AUDIT','n':2640,'checks':checks,'active_inputs':5,'source_top1_rejected':0,'no_new_GT_or_models':True};dump(out/'AUDIT_RESULT.json',rec);print(rec)
