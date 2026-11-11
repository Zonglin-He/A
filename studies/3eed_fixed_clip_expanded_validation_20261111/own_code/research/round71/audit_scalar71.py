"""Independent all-observation saved-scalar and paired comparison check."""
from common71 import *
import itertools
out=H/'independent_scalar_audit_attempt001';assert not out.exists();out.mkdir();labels={x['id_PRIVATE']:x for root in ['research/round69/score_all2304_attempt001','research/round55/score_dev336_attempt001'] for x in read(ROOT/root/'ALL_CANDIDATE_IOUS_PRIVATE.json')};checked=0;negatives=0;count=0
for name,n in [('val1280',1280),('train1024',1024),('dev336',336)]:
 root=H/('score_'+name+'_attempt001');tt=read(root/'ALL_TASK_ROWS_PRIVATE.json');ss=read(root/'TASK_SUMMARY.json')['ALL'+str(n)];sc=read(root/'ALL_SCENE_SUMMARIES_PRIVATE.json');negative=[];assert len(tt)==n
 for x in tt:
  v=labels[x['id_PRIVATE']]['candidate_IoUs_PRIVATE']
  for k in ARMS:assert x['IoU'][k]==v[x['queries'][k]];checked+=1
  for k,ref in itertools.product(ARMS,repeat=2):
   d=x['IoU'][k]-x['IoU'][ref]
   if k!=ref and d< -1e-10:negative.append({'row':x['row'],'part':x['part'],'split':x['split'],'platform':x['platform'],'arm':k,'reference':ref,'delta_pp':float(d*100)})
 assert negative==read(root/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');negatives+=len(negative)
 for k in ARMS:
  means=[]
  for pl in ['drone','quad']:
   rr=[x for x in tt if x['platform']==pl];a=np.array([x['IoU'][k] for x in rr]);v=ss[pl][k]
   for field,value in [('mIoU_percent',a.mean()*100),('Acc25_percent',(a>.25).mean()*100),('Acc50_percent',(a>.5).mean()*100)]:assert abs(v[field]-value)<1e-12;checked+=1
   for ref in REFS:
    d=a-np.array([x['IoU'][ref] for x in rr]);assert abs(v['gain_vs_'+ref+'_pp']-d.mean()*100)<1e-12;assert v['improved_vs_'+ref]==int((d>1e-10).sum()) and v['worsened_vs_'+ref]==int((d< -1e-10).sum());checked+=3
   means.append(a.mean()*100)
  assert abs(ss['balanced_mIoU_percent'][k]-np.mean(means))<1e-12
 for s in sc:
  rr=[x for x in tt if x['platform']==s['platform'] and x['scene_PRIVATE']==s['scene_PRIVATE']];assert len(rr)==s['n']
  for k in ARMS:assert abs(np.mean([x['IoU'][k] for x in rr])*100-s['mIoU_percent'][k])<1e-12;checked+=1
 count+=len(tt)
rec={'state':'PASS_R71_INDEPENDENT_ALL_SAVED_TASK_SCALARS_AND_SCENE_METRICS','n':count,'task_values':count*5,'checks':checked,'ordered_negatives':negatives,'no_new_raw_GT_or_model_operations':True};dump(out/'AUDIT_RESULT.json',rec);print(rec)
