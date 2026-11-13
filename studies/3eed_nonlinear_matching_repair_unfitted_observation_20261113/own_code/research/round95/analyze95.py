"""Postseal whole-list task totals, all negatives, execution and unfitted-view transfer."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round95';P=R/'crossplatform/round95/prediction336_attempt001';rd=lambda p:json.loads(p.read_text())
def dump(n,x):(H/(n+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
T=rd(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=rd(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];arms=rd(P/'SEALED.json')['arm_names'];methods=[];effects=[];updates=[];trials=[];scenes=[]
for a in arms:
 for pl in ['balanced','drone','quad']:
  tt=T if pl=='balanced' else [x for x in T if x['platform']==pl];v=np.array([t['IoU'][a] for t in tt]);methods.append({'platform':pl,'arm':a,'n':len(tt),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100)})
pairs=[('lm_FC',b) for b in ['R91_frozen','ABC','R59','R94_direct','R94_Adam','R94_GN','direct_FC','direct_four','mixture_FC','lm_F','lm_C','lm_mixture_FC','lm_shuffle_FC','lm_capped_FC']]+[('lm_F',b) for b in ['R91_frozen','R94_direct','R94_mixture','R94_Adam','R94_GN','lm_shuffle_F','lm_capped_F']]+[('lm_C',b) for b in ['frozen_C','direct_C','lm_shuffle_C']]+[('direct_FC',b) for b in ['R91_frozen','R94_direct','direct_four','shuffle_FC']]
for a,b in pairs:
 for pl in ['balanced','drone','quad']:
  tt=T if pl=='balanced' else [x for x in T if x['platform']==pl];v=np.array([x['IoU'][a]-x['IoU'][b] for x in tt]);effects.append({'arm':a,'reference':b,'platform':pl,'n':len(tt),'net_pp':float(v.mean()*100),'better_n':int((v>1e-10).sum()),'worse_n':int((v<-1e-10).sum()),'unchanged_n':int((abs(v)<=1e-10).sum()),'query_changes':sum(t['queries'][a]!=t['queries'][b] for t in tt),'rescues_Acc25':sum(t['IoU'][a]>.25 and t['IoU'][b]<=.25 for t in tt),'breaks_Acc25':sum(t['IoU'][a]<=.25 and t['IoU'][b]>.25 for t in tt),'largest_positive_contribution_pp':float(max(0,v.max())*100/len(tt)),'largest_negative_contribution_pp':float(min(0,v.min())*100/len(tt))})
for t in T:
 for name,r in rd(P/'records_PRIVATE'/(t['id_PRIVATE']+'.json')).items():
  updates.append({'row':t['row'],'platform':t['platform'],'method':name,**{k:v for k,v in r.items() if k!='trials'}})
  for i,tr in enumerate(r['trials']):trials.append({'row':t['row'],'platform':t['platform'],'method':name,'attempt':i+1,**{k:v for k,v in tr.items() if k!='state_key'}})
sc=rd(H/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json')
for pl in ['drone','quad']:
 names=sorted(x['scene_PRIVATE'] for x in sc if x['platform']==pl)
 for x in [x for x in sc if x['platform']==pl]:
  for a in arms:scenes.append({'platform':pl,'scene_alias':pl[0].upper()+'_S'+str(names.index(x['scene_PRIVATE'])+1),'n':x['n'],'arm':a,'mIoU_percent':x['mIoU_percent'][a]})
for n,x in [('ALL_METHOD_ROWS',methods),('COMPONENT_EFFECT_ROWS',effects),('ALL_UPDATE_ROWS',updates),('ALL_TRIAL_ROWS',trials),('ALL_EXECUTION_ROWS',rd(P/'EXECUTION_ROWS.json')),('ALL_SCENE_ROWS',scenes)]:dump(n,x)
s=su['balanced_mIoU_percent'];d={'state':'R95_COMPLETED_EXECUTION_REPAIR_AND_UNFITTED_OBSERVATION_DEVELOPMENT_EVALUATED','primary':'lm_FC','primary_mIoU_percent':s['lm_FC'],'gains_pp':{b:s['lm_FC']-s[b] for b in ['R91_frozen','ABC','R59','R94_direct','direct_FC','direct_four','lm_shuffle_FC']},'passes_fixed_development_increment_checks':all(s['lm_FC']>s[b] for b in ['R91_frozen','direct_FC','direct_four','lm_shuffle_FC']),'robustly_adopted':False,'independent_confirmation':False,'source_training':False,'FINAL_access':False,'new_architecture':False,'no_follow_on_queued':True};dump('DECISION',d);print(d)
