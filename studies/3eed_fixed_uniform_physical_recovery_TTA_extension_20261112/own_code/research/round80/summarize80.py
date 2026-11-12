"""Independent saved-only all-value, scene and all-negative verification."""
from core80 import *
pin(read(H80/'experiment/SCORE_CODE_LOCK.json'))
import itertools
methods=[];scenes=[];counts={};checks=0;alltasks=[]
for part,n in [('DEV336',336),('VAL1280',1280),('TRAIN1024',1024)]:
 F=H80/('score_'+part.lower()+'_attempt001');s=read(F/'SEALED.json')
 for f,h in s['files'].items():assert sha(F/f)==h
 tasks=read(F/'ALL_TASK_ROWS_PRIVATE.json');su=read(F/'TASK_SUMMARY.json')['ALL'+str(n)];neg=read(F/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');alltasks+=tasks;expected=[(r['row'],a,b,100*(r['IoU'][a]-r['IoU'][b])) for r in tasks for a,b in itertools.permutations(ARMS80,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];assert expected==[(r['row'],r['arm'],r['reference'],r['delta_pp']) for r in neg];counts[part]={'n':n,'task_values':n*len(ARMS80),'ordered_negatives':len(neg)}
 for pl in ['drone','quad']:
  rr=[r for r in tasks if r['platform']==pl]
  for arm in ARMS80:
   v=np.array([r['IoU'][arm] for r in rr]);d=su[pl][arm];assert abs(v.mean()*100-d['mIoU_percent'])<1e-12 and np.mean(v>.25)*100==d['Acc25_percent'] and np.mean(v>.5)*100==d['Acc50_percent'];methods.append({'cohort':part,'platform':pl,'arm':arm,**d});checks+=3
  for j,sc in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   subset=[r for r in rr if r['scene_PRIVATE']==sc]
   for arm in ARMS80:scenes.append({'cohort':part,'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(j+1).zfill(2),'arm':arm,'n':len(subset),'mIoU_percent':float(np.mean([r['IoU'][arm] for r in subset])*100)})
 # Verify known completed R79 DEV scalar values rather than rescoring references silently.
 if part=='DEV336':
  old=read(ROOT/'research/round79/score_dev336_attempt001/TASK_SUMMARY.json')['ALL336']
  for arm in [a for a in ARMS80 if a!='uniform_wrong_sgd1']:
   for pl in ['drone','quad']:assert su[pl][arm]['mIoU_percent']==old[pl][arm]['mIoU_percent']
dump(H80/'ALL_METHOD_ROWS.json',methods);dump(H80/'ALL_SCENE_ROWS.json',scenes)
rows=[r for r in alltasks if r['part']!='DEV336'];pooled={a:float(np.mean([np.mean([r['IoU'][a] for r in rows if r['platform']==pl])*100 for pl in ['drone','quad']])) for a in ARMS80};dump(H80/'POOLED2304_SUMMARY.json',pooled)
contributions=[]
for r in alltasks:
 for ref in ['frozen_bi','frozen_direct3','uniform_wrong_sgd1']:
  n=sum(x['part']==r['part'] and x['platform']==r['platform'] for x in alltasks);contributions.append({'row':r['row'],'cohort':r['part'],'platform':r['platform'],'reference':ref,'delta_IoU_pp':100*(r['IoU']['uniform_sgd1']-r['IoU'][ref]),'contribution_to_balanced_mIoU_pp':50*(r['IoU']['uniform_sgd1']-r['IoU'][ref])/n})
dump(H80/'ALL_CONTRIBUTION_ROWS.json',contributions);O=H80/'independent_scalar_audit_attempt001';O.mkdir();receipt={'state':'PASS_R80_ALL2640_SAVED_TASK_VALUES_NEGATIVES_AND_SCENES','n':2640,'task_values':2640*len(ARMS80),'ordered_negatives':sum(v['ordered_negatives'] for v in counts.values()),'checks':checks,'cohort_counts':counts,'no_new_GT_models_geometry':True};dump(O/'AUDIT_RESULT.json',receipt);print(receipt)
