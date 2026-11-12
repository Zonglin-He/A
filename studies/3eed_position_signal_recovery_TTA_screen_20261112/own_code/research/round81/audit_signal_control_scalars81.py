"""Saved-only independent verification of new-control metrics and negative tails."""
from core81 import *
import itertools
rows=read(H81/'SIGNAL_CONTROL_TASK_ROWS.json');s=read(H81/'SIGNAL_CONTROL_SUMMARY.json');ng=read(H81/'SIGNAL_CONTROL_NEGATIVE_ROWS.json');arms=['frozen_bi','frozen_direct3','all__ALL','all__ALL__shuffle'];checks=0;expected=[]
assert len(rows)==2304 and len({r['row'] for r in rows})==2304
for part,n in [('VAL1280',1280),('TRAIN1024',1024)]:
 rr=[r for r in rows if r['cohort']==part];assert len(rr)==n
 for pl in ['drone','quad']:
  pr=[r for r in rr if r['platform']==pl]
  for a in arms:
   v=np.array([r[a+'_IoU'] for r in pr]);z=s[part][pl][a];assert z['n']==len(pr) and abs(v.mean()*100-z['mIoU_percent'])<1e-12 and np.mean(v>.25)*100==z['Acc25_percent'] and np.mean(v>.5)*100==z['Acc50_percent'];checks+=3
 for r in rr:
  for a,b in itertools.permutations(arms,2):
   if 'all__ALL__shuffle' in [a,b] and r[a+'_IoU']-r[b+'_IoU']<-1e-12:expected.append((r['row'],a,b,100*(r[a+'_IoU']-r[b+'_IoU'])))
assert expected==[(r['row'],r['arm'],r['reference'],r['delta_pp']) for r in ng];O=H81/'independent_signal_control_scalar_audit_attempt001';O.mkdir();result={'state':'PASS_R81_FULL_SHUFFLE_SAVED_SCALARS_AND_ALL_NEW_NEGATIVES','n':2304,'new_task_values':2304,'copied_reference_values':6912,'checks':checks,'new_control_ordered_negatives':len(ng),'no_new_GT_or_model_operations':True};dump(O/'AUDIT_RESULT.json',result);print(result)
