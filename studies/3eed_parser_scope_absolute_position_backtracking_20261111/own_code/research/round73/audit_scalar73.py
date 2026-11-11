from common73 import *
import itertools
tasks=read(H/'score_mech64_attempt001/ALL_TASK_ROWS_PRIVATE.json');iou={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(ROOT/'research/round69/score_all2304_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};neg=[];checks=0
for x in tasks:
 for k in ARMS:assert x['IoU'][k]==iou[x['id_PRIVATE']][x['queries'][k]];checks+=1
 for k,ref in itertools.permutations(ARMS,2):
  d=x['IoU'][k]-x['IoU'][ref]
  if d< -1e-10:neg.append({'row':x['row'],'platform':x['platform'],'part':x['part'],'split':x['split'],'arm':k,'reference':ref,'delta_pp':float(d*100)})
assert neg==read(H/'score_mech64_attempt001/ALL_ORDERED_NEGATIVE_COMPARISONS.json')
for row in read(H/'ALL_METHOD_ROWS.json')+read(H/'CELL_METHOD_ROWS.json'):
 tt=tasks if row['scope']=='ALL64' else [x for x in tasks if x['part']+'_'+x['platform']==row['scope']];v=np.array([x['IoU'][row['method']] for x in tt]);assert row['n']==len(tt) and abs(row['mIoU_percent']-v.mean()*100)<1e-12 and abs(row['Acc25_percent']-(v>.25).mean()*100)<1e-12;checks+=3
coverage=read(H/'ALL_COVERAGE_ROWS.json');diag=read(H/'COHORT_DIAGNOSTICS.json')['counts'];assert diag['known_source_projection']==sum(x['source_projection_known'] for x in coverage)==39;assert diag['accepted']==sum(x['accepted'] for x in coverage)==56;assert sum(x['source_top1_rejected'] for x in coverage)==0
assert all(not x['direct_query_changed'] and not x['TTA_query_changed'] for x in coverage)
out=H/'independent_scalar_audit_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R73_ALL_SAVED_TASK_NEGATIVE_METHOD_AND_COVERAGE_SCALARS','n':64,'task_values':512,'checks':checks,'ordered_negatives':len(neg),'no_new_GT_geometry_or_model_operations':True};dump(out/'AUDIT_RESULT.json',rec);print(rec)
