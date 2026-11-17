"""Postseal task evaluation on existing exposed cached A256 IoUs only."""
from common137 import *
import numpy as np
def main():
 check();assert rd(H/'AUDIT.json')['status']=='PASS';assert rd(B/'direct/SEALED.json')['n']==1798;rows=[];metrics=[];scenes=[];pairs=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'direct'/name/'samples_PRIVATE'/(sid+'.npz'));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];q=dict(zip(a['arm_names'],a['queries']));rec=rd(B/'direct'/name/'records_PRIVATE'/(sid+'.json'));rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'has_B':rec['has_B'],'edited_queries':rec['edited_queries'],'IoUs':{k:float(y[v]) for k,v in q.items()},'queries_PRIVATE':{k:int(v) for k,v in q.items()}})
 arms=list(rows[0]['IoUs'])
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1. for r in rr])
   for arm in arms:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),**{k:float(np.average(v,weights=w)*100) for k,v in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}})
    for sc in sorted({r['scene_PRIVATE'] for r in rr}):
     ix=np.array([r['scene_PRIVATE']==sc for r in rr]);scenes.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'scene_PRIVATE':sc,'n':int(ix.sum()),'mIoU':float(np.average(y[ix],weights=w[ix])*100)})
   for ctrl in ['Frozen_A','Frozen_F','AB_mean','R135_A','R135_F','F_wrong_rebind','A_rebind']:
    dy=np.array([r['IoUs']['F_rebind']-r['IoUs'][ctrl] for r in rr]);pairs.append({'source':src,'target':tgt,'aggregation':ag,'method':'F_rebind','control':ctrl,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'query_changed':sum(r['queries_PRIVATE']['F_rebind']!=r['queries_PRIVATE'][ctrl] for r in rr)})
 aggregate=[{'aggregation':ag,'arm':arm,**{k:float(np.mean([m[k] for m in metrics if m['aggregation']==ag and m['arm']==arm])) for k in ['mIoU','Acc25','Acc50']}} for ag in ['raw','design_weighted'] for arm in arms];decision={'primary':'F_rebind','comparisons':{},'GT_only_postseal_scoring':True,'FINAL_access':False,'TTA_run':False}
 for ag in ['raw','design_weighted']:
  mm={m['arm']:m['mIoU'] for m in aggregate if m['aggregation']==ag};decision['comparisons'][ag]={c:mm['F_rebind']-mm[c] for c in ['Frozen_A','Frozen_F','F_wrong_rebind']}
 decision['direct_prerequisite_passed']=all(v>0 for d in decision['comparisons'].values() for v in d.values());decision['disposition']='lock_one_new_teacher_TTA_test' if decision['direct_prerequisite_passed'] else 'stop_visible_box_late_rebinding_no_TTA'
 for p,x in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('SCENE_ROWS_PRIVATE',scenes),('PAIRED_ROWS',pairs),('AGGREGATE_ROWS',aggregate),('DECISION',decision)]:dump(H/(p+'.json'),x)
 print(decision,flush=True)
if __name__=='__main__':main()
