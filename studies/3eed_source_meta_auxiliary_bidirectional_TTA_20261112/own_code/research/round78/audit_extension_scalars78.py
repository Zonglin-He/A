"""Saved-only all-value and all-negative audit, no raw GT/model/media operations."""
from extension_common78 import *
import itertools
method=[];scenerows=[];checks=0;counts={}
for name,n in [('VAL1280',1280),('TRAIN1024',1024)]:
 F=H78/('score_'+name.lower()+'_attempt001');ss=read(F/'SEALED.json')
 for f,h in ss['files'].items():assert sha(F/f)==h
 tt=read(F/'ALL_TASK_ROWS_PRIVATE.json');ng=read(F/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');su=read(F/'TASK_SUMMARY.json')['ALL'+str(n)];arms=read(F/'SCORING_RECEIPT.json')['arm_names'];assert len(tt)==n;expect=[(r['row'],a,b,100*(r['IoU'][a]-r['IoU'][b])) for r in tt for a,b in itertools.permutations(arms,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];assert expect==[(r['row'],r['arm'],r['reference'],r['delta_pp']) for r in ng]
 for pl in ['drone','quad']:
  rr=[r for r in tt if r['platform']==pl]
  for a in arms:
   z=su[pl][a];v=np.array([r['IoU'][a] for r in rr]);assert abs(v.mean()*100-z['mIoU_percent'])<1e-12 and np.mean(v>.25)*100==z['Acc25_percent'] and np.mean(v>.5)*100==z['Acc50_percent'];checks+=3;method.append({'cohort':name,'platform':pl,'arm':a,**z})
  scenes=sorted({r['scene_PRIVATE'] for r in rr})
  for i,sc in enumerate(scenes):
   sr=[r for r in rr if r['scene_PRIVATE']==sc]
   for a in arms:scenerows.append({'cohort':name,'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(i+1).zfill(2),'n':len(sr),'arm':a,'mIoU_percent':float(np.mean([r['IoU'][a] for r in sr])*100)})
 counts[name]={'n':n,'task_values':n*len(arms),'ordered_negatives':len(ng)}
out=H78/'independent_extension_scalar_audit_attempt001';out.mkdir();r={'state':'PASS_R78_ALL2304_SAVED_METHOD_AND_NEGATIVE_SCALARS','n':2304,'task_values':27648,'ordered_negatives':sum(v['ordered_negatives'] for v in counts.values()),'checks':checks,'no_new_GT_geometry_or_model_operations':True};dump(out/'AUDIT_RESULT.json',r);dump(H78/'ALL_EXTENSION_METHOD_ROWS.json',method);dump(H78/'ALL_EXTENSION_SCENE_ROWS.json',scenerows);print(r)
