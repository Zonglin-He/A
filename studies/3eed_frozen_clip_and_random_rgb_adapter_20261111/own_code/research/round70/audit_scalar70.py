"""Independent saved IoU table arithmetic, all paired negative comparisons."""
from common70 import *
import itertools
out=H/'independent_scalar_audit_attempt001';assert not out.exists();out.mkdir();sc=H/'score_dev336_attempt001';tt=read(sc/'ALL_TASK_ROWS_PRIVATE.json');summary=read(sc/'TASK_SUMMARY.json');old=read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json');neg=read(sc/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');actual=[]
for t,y in zip(tt,old):
 assert t['id_PRIVATE']==y['id_PRIVATE']
 for k in ARMS:assert t['IoU'][k]==float(y['candidate_IoUs_PRIVATE'][t['queries'][k]])
 for k,j in itertools.product(ARMS,repeat=2):
  d=t['IoU'][k]-t['IoU'][j]
  if k!=j and d< -1e-10:actual.append({'row':t['row'],'part':t['part'],'split':t['split'],'platform':t['platform'],'arm':k,'reference':j,'delta_pp':float(d*100)})
assert actual==neg
scopes={'ALL336':tt,**{s:[t for t in tt if t['split']==s] for s in set(t['split'] for t in tt) if s!='ALL336'}};checks=0
for scope,rr in scopes.items():
 for arm in ARMS:
  means=[]
  for pl in ['drone','quad']:
   v=np.array([t['IoU'][arm] for t in rr if t['platform']==pl]);got=summary[scope][pl][arm]
   for key,value in [('mIoU_percent',v.mean()*100),('Acc25_percent',(v>.25).mean()*100),('Acc50_percent',(v>.5).mean()*100)]:assert abs(got[key]-value)<1e-12;checks+=1
   means.append(v.mean()*100)
  assert abs(summary[scope]['balanced_mIoU_percent'][arm]-np.mean(means))<1e-12
rec={'state':'PASS_R70_INDEPENDENT_ALL_SAVED_TASK_SCALAR_ARITHMETIC','n':336,'arms':len(ARMS),'task_values':len(tt)*len(ARMS),'metric_checks':checks,'ordered_negatives':len(neg),'no_new_raw_GT_or_geometry_scoring':True};dump(out/'AUDIT_RESULT.json',rec);print(rec)
