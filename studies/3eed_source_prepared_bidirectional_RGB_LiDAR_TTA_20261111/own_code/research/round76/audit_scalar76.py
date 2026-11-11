"""Independent scalar arithmetic for target results and full ordered negatives."""
from common76 import *
import itertools
T=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');ys={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};ng=[];checks=0
for x in T:
 for k in ARMS:assert x['IoU'][k]==ys[x['id_PRIVATE']][x['queries'][k]];checks+=1
 for k,r in itertools.permutations(ARMS,2):
  d=x['IoU'][k]-x['IoU'][r]
  if d< -1e-10:ng.append({'row':x['row'],'platform':x['platform'],'part':x['part'],'split':x['split'],'arm':k,'reference':r,'delta_pp':d*100})
assert ng==read(H/'score_dev336_attempt001/ALL_ORDERED_NEGATIVE_COMPARISONS.json')
for row in read(H/'ALL_METHOD_ROWS.json'):
 tt=T if row['scope']=='ALL336' else [x for x in T if x['split']==row['scope']];tt=tt if row['platform']=='balanced' else [x for x in tt if x['platform']==row['platform']];a=np.array([x['IoU'][row['method']] for x in tt]);assert len(tt)==row['n'];assert abs(row['mIoU_percent']-a.mean()*100)<1e-12 and abs(row['Acc25_percent']-(a>.25).mean()*100)<1e-12 and abs(row['Acc50_percent']-(a>.5).mean()*100)<1e-12;checks+=3
su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336']
for pl in ['drone','quad']:
 tt=[x for x in T if x['platform']==pl]
 for k in ARMS:
  a=np.array([x['IoU'][k] for x in tt]);assert abs(su[pl][k]['mIoU_percent']-a.mean()*100)<1e-12
  for s in ['unknown','fallback','no_parameter_update']:assert su[pl][k][s+'_n']==sum(x['statuses'][k][s] for x in tt)
steps=read(H/'ALL_OPTIMIZER_STEP_ROWS.json');assert len(steps)==2016 and not any(x['accepted'] and x['loss_after']>x['loss_before'] for x in steps)
for r in read(H/'OPTIMIZER_SUMMARY_ROWS.json'):
 a=[x for x in steps if x['branch']==r['branch'] and x['step']==r['step'] and (r['platform']=='balanced' or x['platform']==r['platform'])];assert len(a)==r['n'] and sum(x['accepted'] for x in a)==r['accepted']
 for k in ['loss_before','loss_after','aux_before','aux_after','keep_after']:assert abs(r[k+'_mean']-np.mean([x[k] for x in a]))<1e-12;checks+=1
out=H/'independent_scalar_audit_attempt001';assert not out.exists();out.mkdir();v={'state':'PASS_R76_ALL336_SAVED_TASK_NEGATIVE_STATUS_METHOD_AND_UPDATE_SCALARS','n':336,'task_values':336*len(ARMS),'ordered_negatives':len(ng),'optimizer_proposals':2016,'accepted':sum(x['accepted'] for x in steps),'checks':checks,'no_new_GT_geometry_or_model_operations':True};dump(out/'AUDIT_RESULT.json',v);print(v)
